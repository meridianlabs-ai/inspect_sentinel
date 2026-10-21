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


def generate(uuid: str, parent: str | None, output: str = "out") -> ModelEvent:
    return ModelEvent(
        uuid=uuid,
        span_id=parent,
        model="mockllm/model",
        input=[ChatMessageUser(content="hi")],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content("mockllm/model", output),
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
async def test_an_old_output_step_cites_its_output_message() -> None:
    # before R7-D8 the step id was the output message's id
    events = [generate("e1", None), *checked("s1", "model_output", "m1", None)]

    assert cited(await read("model_output", events)) == {"m1": ["m1"]}


@pytest.mark.anyio
async def test_an_output_step_cites_its_generates_output() -> None:
    model = generate("e1", None)
    events = [model, *checked("s1", "model_output", "e1", None)]

    assert cited(await read("model_output", events)) == {
        "e1": [model.output.message.id or ""]
    }
