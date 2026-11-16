from typing import Any

import pytest
from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCallError
from inspect_scout import Result, Transcript

from inspect_sentinel import MonitorFailedError
from inspect_sentinel.scout import Stage, as_reader

SAMPLE = "sample"


def checked(
    span: str,
    stage: Stage,
    step_id: str,
    parent: str | None,
    execution_id: str | None = None,
) -> list[Event]:
    # a step's sentinel span with the decision recorded in it
    return [
        SpanBeginEvent(id=span, parent_id=parent, type="sentinel", name="sentinel"),
        SentinelEvent(
            span_id=span,
            factory="inspect_sentinel/check",
            path="",
            function="check",
            step_id=step_id,
            execution_id=execution_id,
            conversation=parent or SAMPLE,
            stage=stage,
            kind="decision",
            status="reported",
            action="proceed",
        ),
        SpanEndEvent(id=span),
    ]


def generate(
    uuid: str,
    parent: str | None,
    *,
    message_id: str | None = None,
    error: str | None = None,
) -> ModelEvent:
    output = (
        ModelOutput(model="mockllm/model")
        if error is not None
        else ModelOutput.from_content("mockllm/model", "out")
    )
    if message_id is not None:
        output.message.id = message_id
    return ModelEvent(
        uuid=uuid,
        span_id=parent,
        model="mockllm/model",
        input=[ChatMessageUser(id="u", content="hi")],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=output,
        error=error,
    )


def agent(span: str) -> SpanBeginEvent:
    return SpanBeginEvent(id=span, parent_id=None, type="agent", name=span)


async def read(stage: Stage, events: list[Event]) -> list[Result]:
    # the transcript's messages are its generates' outputs, as a run's are
    messages: list[ChatMessage] = [
        e.output.message
        for e in events
        if isinstance(e, ModelEvent) and e.output.choices
    ]
    scan: Any = as_reader(stage=stage)
    return await scan(
        Transcript(transcript_id=SAMPLE, events=events, messages=messages)
    )


def cited(results: list[Result]) -> dict[str, list[str]]:
    return {
        str((r.metadata or {})["step_id"]): [ref.id for ref in r.references]
        for r in results
    }


@pytest.mark.anyio
async def test_an_old_input_step_cites_the_call_it_preceded() -> None:
    events = [
        *checked("s1", "model_input", "u1", None),
        generate("e1", None),
        *checked("s2", "model_input", "u2", None),
        generate("e2", None),
    ]

    assert cited(await read("model_input", events)) == {"u1": ["e1"], "u2": ["e2"]}


@pytest.mark.anyio
async def test_overlapping_generates_cite_nothing() -> None:
    # they can complete out of order
    # two generates of one conversation in flight at once; the second
    # completes first, so position cannot say which event is whose
    events = [
        *checked("s1", "model_input", "u1", None),
        *checked("s2", "model_input", "u2", None),
        generate("e2", None),
        generate("e1", None),
        *checked("s3", "model_input", "u3", None),
        generate("e3", None),
    ]

    assert cited(await read("model_input", events)) == {
        "u1": [],
        "u2": [],
        "u3": ["e3"],
    }


@pytest.mark.anyio
async def test_a_generate_in_another_conversation_does_not_overlap() -> None:
    events = [
        agent("a"),
        agent("b"),
        *checked("s1", "model_input", "u1", "a"),
        *checked("s2", "model_input", "u2", "b"),
        generate("eb", "b"),
        generate("ea", "a"),
    ]

    assert cited(await read("model_input", events)) == {"u1": ["ea"], "u2": ["eb"]}


def tool(
    uuid: str,
    parent: str | None,
    id: str = "shared",
    *,
    execution_id: str | None = None,
    bridged: bool = False,
    reported: bool = True,
    error: ToolCallError | None = None,
) -> ToolEvent:
    # a call with the id "shared", whose tool message is r<uuid>; `bridged`, a
    # host call a sandbox agent bridge ran, which links its run to the result
    # the scaffold reported (`message_id`) only where `reported`
    return ToolEvent(
        uuid=uuid,
        span_id=parent,
        id=id,
        execution_id=execution_id,
        function="addition",
        arguments={},
        message_id=f"r{uuid}" if reported or not bridged else None,
        metadata={"bridge": {"proposal_id": id}} if bridged else None,
        error=error,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["tool_call", "tool_result"])
async def test_two_agents_calls_sharing_an_id_cite_their_own_tool_events(
    stage: Stage,
) -> None:
    # both agents' calls are checked before either runs, and b's runs first
    events = [
        agent("a"),
        agent("b"),
        *checked("s1", "tool_call", "shared", "a"),
        *checked("s2", "tool_call", "shared", "b"),
        tool("eb", "b"),
        tool("ea", "a"),
        *checked("s3", "tool_result", "shared", "a"),
        *checked("s4", "tool_result", "shared", "b"),
    ]

    results = await read(stage, events)

    assert [[ref.id for ref in r.references] for r in results] == [["ea"], ["eb"]]


def keyed_citations(results: list[Result]) -> list[tuple[str, list[str]]]:
    # each row's step id and what it cites, in order, as rows can share a key
    return [
        (str((r.metadata or {})["step_id"]), [ref.id for ref in r.references])
        for r in results
    ]


# The bridged shapes below are those inspect_ai #6446 (df812940f) logs, whose
# replay names a host run's steps as it names a native execution's: the first
# rebuilt run of a call id, native or bridged, in log order, keeps the id, and
# a later one has its execution id.


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["tool_call", "tool_result"])
async def test_each_tool_of_a_shared_handout_is_read_under_the_id_its_run_has(
    stage: Stage,
) -> None:
    # one handout of "host_1" denotes two bridged tools: each tool's checks
    # name that tool's execution of the handout, and the scaffold runs
    # read_copy under the call's id, then read_file under an id of its own
    # (`shared_handout_run_in_reverse`). Replay builds read_copy's steps under
    # host_1 and none for read_file's run, so read_file's checks must not join
    # host_1's row
    events = [
        *checked("s1", "tool_call", "host_1", None, execution_id="h/host/read_file"),
        *checked("s2", "tool_call", "host_1", None, execution_id="h/copy/read_copy"),
        tool(
            "copy",
            None,
            "host_1",
            execution_id="h/copy/read_copy",
            bridged=True,
            reported=False,
        ),
        tool(
            "file",
            None,
            "own",
            execution_id="h/host/read_file",
            bridged=True,
            reported=False,
        ),
        *checked("s3", "tool_result", "host_1", None, execution_id="h/host/read_file"),
        *checked("s4", "tool_result", "host_1", None, execution_id="h/copy/read_copy"),
    ]

    # each citing its own run
    assert keyed_citations(await read(stage, events)) == [
        ("h/host/read_file", ["file"]),
        ("host_1", ["copy"]),
    ]


@pytest.mark.anyio
async def test_a_check_of_a_bridged_tool_that_never_ran_is_read_under_its_execution() -> (
    None
):
    # the handout's read_copy never ran (`repeated_host_call(..., shared=True)`):
    # replay has no step for it, so its check must neither join read_file's
    # host_1 row nor cite read_file's run
    events = [
        *checked("s1", "tool_call", "host_1", None, execution_id="h/host/read_file"),
        *checked("s2", "tool_call", "host_1", None, execution_id="h/copy/read_copy"),
        tool(
            "file",
            None,
            "host_1",
            execution_id="h/host/read_file",
            bridged=True,
            reported=False,
        ),
    ]

    assert keyed_citations(await read("tool_call", events)) == [
        ("host_1", ["file"]),
        ("h/copy/read_copy", []),
    ]


@pytest.mark.anyio
async def test_a_second_handout_of_a_host_call_id_is_read_under_its_execution() -> None:
    # the bridge hands host_1 over twice, a retry after the first run raised a
    # ToolError (#6370's B1 host retry): replay names the second handout's
    # steps by its execution. Once the id was handed out twice, the bridge
    # checks a result reported under it under no execution, and which run it
    # answers is unknown, so that check is read under its own event's id, which
    # no replayed step has, not under host_1, the first handout's
    events = [
        *checked("s1", "tool_call", "host_1", None, execution_id="h1/host/addition"),
        tool(
            "first",
            None,
            "host_1",
            execution_id="h1/host/addition",
            bridged=True,
            error=ToolCallError("unknown", "failed"),
        ),
        *checked("s2", "tool_result", "host_1", None, execution_id="h1/host/addition"),
        *checked("s3", "tool_call", "host_1", None, execution_id="h2/host/addition"),
        tool("second", None, "host_1", execution_id="h2/host/addition", bridged=True),
        *checked("s4", "tool_result", "host_1", None),
    ]

    results = await read("tool_call", events) + await read("tool_result", events)

    [unknown] = [e for e in events if isinstance(e, SentinelEvent)][-1:]
    assert [str((r.metadata or {})["step_id"]) for r in results] == [
        "host_1",
        "h2/host/addition",
        "host_1",
        str(unknown.uuid),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("host_checked", [True, False])
async def test_a_native_call_after_a_host_call_under_its_id_is_read_under_its_execution(
    host_checked: bool,
) -> None:
    # a host call ran under "shared", then a native call under the same id
    # (`test_a_native_call_reusing_a_bridged_call_id_keeps_its_tool_steps`):
    # replay builds the host run's steps under the id, also where no check of
    # it was recorded, since the bridge linked its reported result, and names
    # the native call's steps by its execution
    host = [
        tool("host", None, execution_id="h/host/addition", bridged=True),
    ]
    if host_checked:
        host = [
            *checked("s1", "tool_call", "shared", None, execution_id="h/host/addition"),
            *host,
            *checked(
                "s2", "tool_result", "shared", None, execution_id="h/host/addition"
            ),
        ]
    events = [
        *host,
        *checked("s3", "tool_call", "shared", None, execution_id="x1"),
        tool("native", None, execution_id="x1"),
        *checked("s4", "tool_result", "shared", None, execution_id="x1"),
    ]

    assert keyed_citations(await read("tool_result", events)) == [
        *([("shared", ["host"])] if host_checked else []),
        ("x1", ["native"]),
    ]


@pytest.mark.anyio
async def test_a_monitors_own_call_does_not_take_a_repeated_call_id() -> None:
    # a monitor's model call reuses the agent's call id "shared" inside the
    # sentinel span; replay skips that run, so the agent's run keeps the id
    events: list[Event] = [
        SpanBeginEvent(id="m", parent_id=None, type="sentinel", name="sentinel"),
        tool("own", "m", execution_id="x0"),
        SpanEndEvent(id="m"),
        tool("e1", None, execution_id="x1"),
        *checked("s1", "tool_result", "shared", None, execution_id="x1"),
    ]

    # the transcript holds no tool message, so the check cites its tool event
    assert keyed_citations(await read("tool_result", events)) == [("shared", ["e1"])]


@pytest.mark.anyio
async def test_a_checked_call_that_raised_a_parsing_error_keeps_its_id() -> None:
    # a tool that raised ToolParsingError itself after its call was checked
    # ran, so replay builds its steps under the id and names the model's
    # same-id retry by its execution id
    events = [
        tool("e1", None, execution_id="x1", error=ToolCallError("parsing", "bad")),
        *checked("s1", "tool_result", "shared", None, execution_id="x1"),
        tool("e2", None, execution_id="x2"),
        *checked("s2", "tool_result", "shared", None, execution_id="x2"),
    ]

    assert keyed_citations(await read("tool_result", events)) == [
        ("shared", ["e1"]),
        ("x2", ["e2"]),
    ]


@pytest.mark.anyio
async def test_an_old_output_step_cites_the_output_message_it_names() -> None:
    # before R7-D8 the step id was the output message's id
    events = [
        generate("e1", None, message_id="m1"),
        *checked("s1", "model_output", "m1", None),
        *checked("s2", "model_output", "missing", None),
    ]

    assert cited(await read("model_output", events)) == {"m1": ["m1"], "missing": []}


@pytest.mark.anyio
async def test_an_output_step_with_no_model_event_cites_nothing() -> None:
    events = checked("s1", "model_output", "missing", None)

    assert cited(await read("model_output", events)) == {"missing": []}


@pytest.mark.anyio
async def test_a_retried_generate_cites_its_first_attempt_and_its_final_output() -> (
    None
):
    # a failed attempt, then a retry with the same input in the same span
    events = [
        *checked("s1", "model_input", "first", None),
        generate("first", None, error="provider error"),
        generate("second", None, message_id="answer"),
        *checked("s2", "model_output", "first", None),
    ]

    assert cited(await read("model_input", events)) == {"first": ["first"]}
    assert cited(await read("model_output", events)) == {"first": ["answer"]}


@pytest.mark.anyio
async def test_a_failure_at_another_stage_fails_the_read() -> None:
    events: list[Event] = [
        SentinelEvent(
            factory="probe",
            path="probe",
            function="before",
            step_id="c1",
            conversation=SAMPLE,
            stage="tool_call",
            kind="observation",
            status="reported",
            suspicion=0.3,
        ),
        SentinelEvent(
            factory="probe",
            path="probe",
            function="after",
            step_id="c1",
            conversation=SAMPLE,
            stage="tool_result",
            kind="observation",
            status="error",
            error="RuntimeError: broke",
        ),
    ]
    scan: Any = as_reader("probe", stage="tool_call")

    with pytest.raises(MonitorFailedError, match="broke"):
        await scan(Transcript(transcript_id=SAMPLE, events=events))


@pytest.mark.anyio
async def test_a_superseded_decision_is_read_as_one() -> None:
    def decided(status: Any, action: Any) -> SentinelEvent:
        return SentinelEvent(
            factory="inspect_sentinel/check",
            path="",
            function="check",
            step_id="c1",
            conversation=SAMPLE,
            stage="tool_call",
            kind="decision",
            status=status,
            action=action,
        )

    results = await read(
        "tool_call", [decided("reported", "terminate"), decided("superseded", "reject")]
    )

    assert [(r.answer, (r.metadata or {}).get("status")) for r in results] == [
        ("terminate", None),
        ("reject", "superseded"),
    ]


@pytest.mark.anyio
async def test_a_decision_with_no_action_is_a_malformed_log() -> None:
    from inspect_ai._sentinel._integration import MalformedLogError

    # a log that skipped validation: the model refuses to build one
    events: list[Event] = [
        SentinelEvent.model_construct(
            factory="inspect_sentinel/check",
            path="",
            function="check",
            step_id="c1",
            conversation=SAMPLE,
            stage="tool_call",
            kind="decision",
            status="reported",
            action=None,
            suspicion=None,
            span_id=None,
            references=[],
        )
    ]

    scan: Any = as_reader(stage="tool_call")
    transcript = Transcript.model_construct(
        transcript_id=SAMPLE, events=events, messages=[], timelines=[]
    )

    with pytest.raises(MalformedLogError, match="has no action"):
        await scan(transcript)
