from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai._sentinel._integration import EvalInfo, steps_from_events
from inspect_ai.dataset import Sample
from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_scout import Result, Transcript

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Monitor,
    MonitorFailedError,
    Observation,
    monitor,
    observe_only,
)
from inspect_sentinel.scout import Stage, as_reader

SAMPLE = "sample"


def checked(
    span: str,
    stage: Stage,
    step_id: str,
    parent: str | None,
    execution_id: str | None = None,
    *,
    bridge: bool = False,
) -> list[Event]:
    # a step's sentinel span with the decision recorded in it; `bridge`, a
    # check an agent bridge recorded
    name = "sentinel_bridge" if bridge else "sentinel"
    return [
        SpanBeginEvent(id=span, parent_id=parent, type="sentinel", name=name),
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


def proposal(message_id: str, *call_ids: str) -> ModelEvent:
    # an agent turn whose output message `message_id` proposes `call_ids`
    event = generate(f"g{message_id}", None, message_id=message_id)
    event.output.message.tool_calls = [
        ToolCall(id=call_id, function="addition", arguments={}) for call_id in call_ids
    ]
    return event


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
    granted: str = "turn",
    error: ToolCallError | None = None,
) -> ToolEvent:
    # a call with the id "shared", whose tool message is r<uuid>; `bridged`, a
    # host call a sandbox agent bridge ran of the model's call proposed in the
    # output message `granted`
    return ToolEvent(
        uuid=uuid,
        span_id=parent,
        id=id,
        execution_id=execution_id,
        function="addition",
        arguments={},
        message_id=f"r{uuid}",
        call_message_id=granted if bridged else None,
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


# The bridged shapes below are those a sandbox agent bridge logs: it records
# its checks in `sentinel_bridge` spans, under the id it handed the call over
# under, naming the handout's execution (`execution_id`). Replay builds no
# steps for bridged calls, so read mode keys each such check by its execution,
# else by its sentinel span, and never by the call id, which a native call's
# steps can hold.


async def read_both(events: list[Event]) -> list[tuple[str, list[str]]]:
    return keyed_citations(
        await read("tool_call", events) + await read("tool_result", events)
    )


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["tool_call", "tool_result"])
async def test_each_tool_of_a_shared_handout_is_read_under_its_execution(
    stage: Stage,
) -> None:
    # one handout of "host_1" denotes two bridged tools: each tool's checks
    # name that tool's execution of the handout, and the scaffold runs
    # read_copy under the call's id, then read_file under an id of its own
    # (`shared_handout_run_in_reverse`)
    events = [
        proposal("turn", "host_1"),
        *checked("s1", "tool_call", "host_1", None, "h/host/read_file", bridge=True),
        *checked("s2", "tool_call", "host_1", None, "h/copy/read_copy", bridge=True),
        tool("copy", None, "host_1", execution_id="h/copy/read_copy", bridged=True),
        tool("file", None, "own", execution_id="h/host/read_file", bridged=True),
        *checked("s3", "tool_result", "host_1", None, "h/host/read_file", bridge=True),
        *checked("s4", "tool_result", "host_1", None, "h/copy/read_copy", bridge=True),
    ]

    # each citing its own run
    assert keyed_citations(await read(stage, events)) == [
        ("h/host/read_file", ["file"]),
        ("h/copy/read_copy", ["copy"]),
    ]


@pytest.mark.anyio
async def test_a_check_of_a_bridged_tool_that_never_ran_cites_nothing() -> None:
    # the handout's read_copy never ran (`repeated_host_call(..., shared=True)`),
    # so its check must not cite read_file's run under the same id
    events = [
        proposal("turn", "host_1"),
        *checked("s1", "tool_call", "host_1", None, "h/host/read_file", bridge=True),
        *checked("s2", "tool_call", "host_1", None, "h/copy/read_copy", bridge=True),
        tool("file", None, "host_1", execution_id="h/host/read_file", bridged=True),
    ]

    assert keyed_citations(await read("tool_call", events)) == [
        ("h/host/read_file", ["file"]),
        ("h/copy/read_copy", []),
    ]


@pytest.mark.anyio
async def test_a_bridge_check_under_an_id_a_native_call_holds_is_read_under_its_execution() -> (
    None
):
    # a native call under "shared" (a provider that repeats ids), then a host
    # call the bridge checked under the same id: the native call's steps hold
    # the id, so the host call's checks must not join their rows
    events = [
        proposal("turn", "shared"),
        *checked("s1", "tool_call", "shared", None, "x1"),
        tool("native", None, execution_id="x1"),
        *checked("s2", "tool_result", "shared", None, "x1"),
        *checked("s3", "tool_call", "shared", None, "h/host/addition", bridge=True),
        tool("host", None, execution_id="h/host/addition", bridged=True),
        *checked("s4", "tool_result", "shared", None, "h/host/addition", bridge=True),
    ]

    assert await read_both(events) == [
        ("shared", ["native"]),
        ("h/host/addition", ["host"]),
        ("shared", ["native"]),
        ("h/host/addition", ["host"]),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("host_checked", [True, False])
async def test_a_native_call_after_a_host_call_under_its_id_keeps_the_id(
    host_checked: bool,
) -> None:
    # a host call ran under "shared", then a native call under the same id
    # (`test_a_native_call_reusing_a_bridged_call_id_keeps_its_tool_steps`):
    # replay builds no steps for the host call, so the native call's steps
    # take the id, whether or not the bridge checked the host call
    host: list[Event] = [
        proposal("turn", "shared"),
        tool("host", None, execution_id="h/host/addition", bridged=True),
    ]
    if host_checked:
        host = [
            host[0],
            *checked("s1", "tool_call", "shared", None, "h/host/addition", bridge=True),
            host[1],
            *checked(
                "s2", "tool_result", "shared", None, "h/host/addition", bridge=True
            ),
        ]
    events = [
        *host,
        *checked("s3", "tool_call", "shared", None, "x1"),
        tool("native", None, execution_id="x1"),
        *checked("s4", "tool_result", "shared", None, "x1"),
    ]

    assert await read_both(events) == [
        *([("h/host/addition", ["host"])] if host_checked else []),
        ("shared", ["native"]),
        *([("h/host/addition", ["host"])] if host_checked else []),
        ("shared", ["native"]),
    ]


@pytest.mark.anyio
async def test_a_bridge_result_check_naming_no_execution_is_read_under_its_span() -> (
    None
):
    # the bridge hands host_1 over twice, a retry after the first run raised a
    # ToolError (#6370's B1 host retry), and a native call runs under the same
    # id. Once the id was handed out twice, the bridge checks a result reported
    # under it under no execution, and which run it answers is unknown, so that
    # check is read under the id of its sentinel span, not under host_1, which
    # the native call's steps hold
    events = [
        proposal("m1", "host_1"),
        *checked("s1", "tool_call", "host_1", None, "h1/host/addition", bridge=True),
        tool(
            "first",
            None,
            "host_1",
            execution_id="h1/host/addition",
            bridged=True,
            granted="m1",
            error=ToolCallError("unknown", "failed"),
        ),
        *checked("s2", "tool_result", "host_1", None, "h1/host/addition", bridge=True),
        proposal("m2", "host_1"),
        *checked("s3", "tool_call", "host_1", None, "h2/host/addition", bridge=True),
        tool(
            "second",
            None,
            "host_1",
            execution_id="h2/host/addition",
            bridged=True,
            granted="m2",
        ),
        *checked("s4", "tool_result", "host_1", None, bridge=True),
        tool("native", None, "host_1", execution_id="x1"),
        *checked("s5", "tool_result", "host_1", None, "x1"),
    ]

    assert [key for key, _ in await read_both(events)] == [
        "h1/host/addition",
        "h2/host/addition",
        "h1/host/addition",
        "s4",
        "host_1",
    ]


@pytest.mark.anyio
async def test_a_decision_at_an_untied_result_check_keeps_its_childrens_reports() -> (
    None
):
    # a monitor's observation and the protocol's decision at a result check the
    # bridge couldn't tie to one run: both are keyed by their sentinel span, so
    # the decision's row carries the observation, as at any other step
    events = [
        proposal("m1", "host_1"),
        tool(
            "first",
            None,
            "host_1",
            execution_id="h1/host/addition",
            bridged=True,
            granted="m1",
        ),
        proposal("m2", "host_1"),
        tool(
            "second",
            None,
            "host_1",
            execution_id="h2/host/addition",
            bridged=True,
            granted="m2",
        ),
        SpanBeginEvent(
            id="s1", parent_id=None, type="sentinel", name="sentinel_bridge"
        ),
        SentinelEvent(
            span_id="s1",
            factory="inspect_sentinel/suspicion",
            path="suspicion",
            function="suspicion",
            step_id="host_1",
            conversation=SAMPLE,
            stage="tool_result",
            kind="observation",
            status="reported",
            suspicion=0.7,
        ),
        *checked("s1", "tool_result", "host_1", None, bridge=True)[1:],
    ]

    [result] = await read("tool_result", events)

    assert (result.metadata or {})["step_id"] == "s1"
    assert [c["suspicion"] for c in (result.metadata or {})["reports"]] == [0.7]


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


@monitor
def quiet() -> Monitor:
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.0)

    return before


@pytest.fixture(scope="module")
def header(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, EvalInfo]:
    # a log written by this inspect_ai with a sentinel, whose header read mode
    # reads to replay a transcript as `as_scanner()` would
    [log] = eval(
        Task(dataset=[Sample(input="hi")], sentinel=observe_only(quiet())),
        model=get_model("mockllm/model"),
        log_dir=str(tmp_path_factory.mktemp("header")),
        display="none",
    )
    assert log.status == "success", log.error
    return log.location, EvalInfo.from_eval(read_eval_log(log.location).eval)


def bridged_turn(case: str) -> list[Event]:
    # a bridged model call proposing "a", the bridge's checks of its call and
    # result, and the next bridged request carrying the result, with the
    # proposal changed as `case` says (the review probes of 2026-11-23)
    first = proposal("out", "a")
    first.origin = "agent"
    first.requested_model = "inspect"
    later = generate("later", None)
    later.origin = "agent"
    later.requested_model = "inspect"
    later.input = [
        *first.input,
        first.output.message,
        ChatMessageTool(
            id="answer", content="2", tool_call_id="a", function="addition"
        ),
    ]
    before: list[Event] = []
    middle: list[Event] = []
    after: list[Event] = [later]
    if case == "monitor role":
        first.role = "monitor"
    elif case == "pending":
        first.pending = True
    elif case == "scorer origin":
        first.origin = "scorer"
    elif case == "no output message id":
        first.output.message.id = None
    elif case == "agent origin inside a sentinel span":
        first.span_id = "outer"
        before = [SpanBeginEvent(id="outer", name="sentinel", type="sentinel")]
        middle = [SpanEndEvent(id="outer")]
    elif case == "handed over under another id":
        first.agent_tool_call_ids = ["fresh"]
    elif case == "agent ids swapped":
        first.output.message.tool_calls = [
            ToolCall(id="a", function="addition", arguments={}),
            ToolCall(id="b", function="addition", arguments={}),
        ]
        first.agent_tool_call_ids = ["b", "a"]
    elif case == "agent ids repeated":
        first.output.message.tool_calls = [
            ToolCall(id="a", function="addition", arguments={}),
            ToolCall(id="b", function="addition", arguments={}),
        ]
        first.agent_tool_call_ids = ["a", "a"]
    elif case == "run by a native tool":
        run = tool("native", None, "a")
        run.call_message_id = "out"
        middle = [run]
    elif case == "native run from no proposing message":
        run = tool("native", None, "a", execution_id="native-run")
        run.call_message_id = "not-an-output"
        middle = [
            *checked("native-call", "tool_call", "a", None, "native-run"),
            run,
            *checked("native-result", "tool_result", "a", None, "native-run"),
        ]
    elif case == "no next request":
        after = []
    elif case == "result not in the next request":
        later.input = first.input
    return [
        *before,
        first,
        *middle,
        *checked("call", "tool_call", "a", None, "execution", bridge=True),
        *checked("result", "tool_result", "a", None, "execution", bridge=True),
        *after,
    ]


# each case's steps read mode names: a bridge check's only where replay
# built the step of the one model call that proposed its id
BOTH = {("tool_call", "gout/a"), ("tool_result", "gout/a")}
NAMED: dict[str, set[tuple[str, str]]] = {
    "plain": BOTH,
    "monitor role": set(),
    "pending": set(),
    "scorer origin": set(),
    "no output message id": set(),
    "agent origin inside a sentinel span": set(),
    "handed over under another id": set(),
    # the agent received the model's "b" as "a", and replay names that call's
    # steps "gout/a", the step the bridge's check under "a" was of
    "agent ids swapped": {("tool_call", "gout/a")},
    "agent ids repeated": set(),
    "run by a native tool": set(),
    # the native checks name their own native steps; replay builds no bridged
    # step for the call the native tool ran
    "native run from no proposing message": {("tool_call", "a"), ("tool_result", "a")},
    "no next request": {("tool_call", "gout/a")},
    "result not in the next request": {("tool_call", "gout/a")},
}


@pytest.mark.anyio
@pytest.mark.parametrize("case", list(NAMED))
async def test_a_bridge_check_names_only_a_step_replay_builds(
    case: str, header: tuple[str, EvalInfo]
) -> None:
    # read mode names a bridge check's step only where inspect_ai's replay of
    # the same events builds it
    location, eval_info = header
    events = bridged_turn(case)
    messages = [m for e in events if isinstance(e, ModelEvent) for m in e.input]
    steps = steps_from_events(
        messages,
        events,
        [],
        sample_uuid=SAMPLE,
        eval_info=eval_info,
    )
    built = {(s.stage, s.id) for s in steps if s.stage.startswith("tool")}
    transcript = Transcript(
        transcript_id=SAMPLE,
        source_type="eval_log",
        source_uri=location,
        events=events,
        messages=messages,
    )
    scan: Any = as_reader()
    rows: list[dict[str, Any]] = [r.metadata or {} for r in await scan(transcript)]
    named = {
        (str(row["stage"]), str(row["replay_step_id"]))
        for row in rows
        if str(row["stage"]).startswith("tool") and row["replay_step_id"] is not None
    }
    assert named <= built
    assert named == NAMED[case]
