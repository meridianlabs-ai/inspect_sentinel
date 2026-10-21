from typing import Any

import pytest
from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    SpanEndEvent,
)
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput
from inspect_scout import Result, Transcript

from inspect_sentinel import MonitorFailedError
from inspect_sentinel.scout import Stage, as_reader

SAMPLE = "sample"


def checked(span: str, stage: Stage, step_id: str, parent: str | None) -> list[Event]:
    # a step's sentinel span with the decision recorded in it
    return [
        SpanBeginEvent(id=span, parent_id=parent, type="sentinel", name="sentinel"),
        SentinelEvent(
            span_id=span,
            factory="inspect_sentinel/check",
            path="",
            function="check",
            step_id=step_id,
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
    scan: Any = as_reader("", stage=stage)
    return await scan(Transcript(transcript_id=SAMPLE, events=events))


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

    scan: Any = as_reader("", stage="tool_call")
    transcript = Transcript.model_construct(
        transcript_id=SAMPLE, events=events, messages=[], timelines=[]
    )

    with pytest.raises(MalformedLogError, match="has no action"):
        await scan(transcript)
