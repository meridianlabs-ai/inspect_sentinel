"""What a sentinel in the sidecar is given at each step, found in the traffic."""

import logging
from typing import Any, cast

import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Record,
    Refuse,
    UnreadableError,
    _inspect_ai,
)
from tests.sidecar._traffic import message, ran, reply, request, weather

_TOOLS = [
    {
        "name": "get_weather",
        "description": "Look up the weather.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    },
    # a tool the provider runs
    {"type": "web_search_20250305", "name": "web_search"},
]


@protocol
def at(stage: str, action: str = "terminate") -> Protocol:
    """Decide `action` at every step of one stage."""

    async def decide(context: Context, step: Step) -> Decision | None:
        if type(step).__name__ != stage:
            return Decision.proceed()
        return Decision(action=cast(Any, action), explanation=stage)

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["terminate", "escalate"])
async def test_request_ended_before_the_model_is_called_is_not_sent(
    action: str,
) -> None:
    handler = Handler(at("BeforeGenerate", action))

    answer = await handler.request(request())

    # an escalate nobody takes ends the run, as at any stage
    assert answer == Refuse(
        400, "A sentinel ended this run before the model was called."
    )
    with pytest.raises(UnreadableError, match="never shown"):
        await handler.reply(reply(message(weather("Oslo"))))


@pytest.mark.anyio
async def test_reply_ended_after_the_model_answered_is_not_passed_on() -> None:
    handler = Handler(at("AfterGenerate"))
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert answer == Refuse(400, "A sentinel ended this run at the model's reply.")
    records = await handler.records("x")
    assert records is not None
    # the run ended before any call in the reply was judged
    assert [record.stage for record in records] == [
        "model_input",
        "model_output",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["BeforeGenerate", "AfterGenerate"])
async def test_reject_at_a_generate_stage_is_a_failing_protocol(stage: str) -> None:
    handler = Handler(at(stage, "reject"))

    with pytest.raises(ValueError, match="cannot reject yet"):
        await handler.request(request())
        await handler.reply(reply(message(weather("Oslo"))))


@protocol
def reports_generates() -> Protocol:
    async def before(context: Context, step: BeforeGenerate) -> Decision | None:
        return Decision.proceed(
            "|".join(
                [
                    step.model,
                    _roles(step.input),
                    str(step.history == step.input),
                    str([tool.name for tool in step.tools]),
                    str(step.tool_choice),
                    str(step.config.max_tokens),
                    str(step.config.temperature),
                ]
            )
        )

    return before


@protocol
def reports_outputs() -> Protocol:
    async def after(context: Context, step: AfterGenerate) -> Decision | None:
        return Decision.proceed(
            f"{_roles(step.input)}|{[tool.name for tool in step.tools]}|{step.output.message.text}"
        )

    return after


def _roles(messages: list[Any] | None) -> str:
    return str(None if messages is None else [m.role for m in messages])


def _explanations(records: list[Record] | None) -> list[str]:
    assert records is not None
    return [cast(Any, record.detail).report.explanation for record in records]


@pytest.mark.anyio
async def test_generate_stages_are_given_what_the_request_offers() -> None:
    handler = Handler([reports_generates(), reports_outputs()])
    await handler.request(request(tools=_TOOLS, max_tokens=512, temperature=0.5))

    await handler.reply(
        reply(message({"type": "text", "text": "Checking."}, weather("Oslo")))
    )

    explanations = _explanations(await handler.records("x"))
    assert (
        "anthropic/claude-test|['system', 'user']|True"
        "|['get_weather', 'web_search']|auto|512|0.5"
    ) in explanations
    assert (
        "['system', 'user']|['get_weather', 'web_search']|Checking."
    ) in explanations


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, "[]|none"),
        ({"tools": _TOOLS[:1]}, "['get_weather']|auto"),
        (
            {
                "tools": _TOOLS[:1],
                "tool_choice": {"type": "tool", "name": "get_weather"},
            },
            "['get_weather']|ToolFunction(name='get_weather')",
        ),
    ],
)
async def test_tool_choice_is_what_the_model_is_sent(
    fields: dict[str, Any], expected: str
) -> None:
    handler = Handler(reports_generates())

    await handler.request(request(**fields))

    (explanation,) = _explanations(await handler.records("x"))
    assert explanation.split("|")[3:5] == expected.split("|")


@pytest.mark.anyio
async def test_a_tool_that_cannot_be_read_is_left_out_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(_inspect_ai, "_warned", set[str]())
    tools = [*_TOOLS[:1], {"type": "future_tool_20990101", "name": "future"}]
    handler = Handler(reports_generates())

    with caplog.at_level(logging.WARNING):
        for id in ("a", "b"):
            assert await handler.request(request(id, tools=tools)) == Pass()

    (explanation,) = _explanations(await handler.records("a"))
    assert explanation.split("|")[3] == "['get_weather']"
    # once for each kind of tool, however often it is offered
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "anthropic future_tool_20990101" in warnings[0]


@pytest.mark.anyio
async def test_a_tool_choice_that_cannot_be_read_is_unreadable() -> None:
    with pytest.raises(UnreadableError, match="tool choice or settings"):
        await Handler(reports_generates()).request(
            request(tool_choice={"type": "future_choice"})
        )


# the text beside each call in a reply, as `ran` writes it when the client
# sends the reply back
_LOOKING = {"type": "text", "text": "Looking."}


@protocol
def reports_histories() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        call = step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else "-"
        texts = [m.text for m in step.history or []]
        return Decision.proceed(f"{type(step).__name__}|{call}|{texts}")

    return decide


@pytest.mark.anyio
async def test_a_tool_step_is_given_the_conversation_before_a_compaction() -> None:
    handler = Handler(reports_histories())
    earlier = [
        *ran("Oslo", "rain", "toolu_1"),
        *ran("Lima", "sun", "toolu_2"),
    ]
    await handler.request(request("a", then=earlier))
    await handler.reply(reply(message(_LOOKING, weather("Cairo", "toolu_3")), id="a"))

    # the agent compacts its conversation, keeping only the latest turn
    await handler.request(
        request(
            "b",
            then=[
                {"role": "assistant", "content": "Oslo had rain and Lima sun."},
                {"role": "user", "content": "Go on."},
                *ran("Cairo", "fog", "toolu_3"),
            ],
        )
    )

    explanations = _explanations(await handler.records("b"))
    question = ["You are helpful.", "What is the weather?"]
    lookups = ["Looking.", "rain", "Looking.", "sun"]
    # the result is judged with every turn the sidecar has seen, those the
    # request no longer carries first, up to the call that made it
    assert explanations[0] == (
        f"AfterToolCall|toolu_3|"
        f"{[*question, *lookups, 'Oslo had rain and Lima sun.', 'Go on.', 'Looking.']}"
    )
    # the request is judged with the conversation it carries, as Inspect gives
    # a generate what its caller passed
    assert explanations[1] == (
        f"BeforeGenerate|-|"
        f"{[*question, 'Oslo had rain and Lima sun.', 'Go on.', 'Looking.', 'fog']}"
    )


@pytest.mark.anyio
async def test_a_call_is_judged_with_the_conversation_and_the_turn_that_made_it() -> (
    None
):
    handler = Handler(reports_histories())
    await handler.request(request(then=ran("Oslo", "rain", "toolu_1")))

    await handler.reply(
        reply(
            message({"type": "text", "text": "Now Lima."}, weather("Lima", "toolu_2"))
        )
    )

    explanations = _explanations(await handler.records("x"))
    assert explanations[-1] == (
        "BeforeToolCall|toolu_2|"
        f"{['You are helpful.', 'What is the weather?', 'Looking.', 'rain', 'Now Lima.']}"
    )


@protocol
def reports_views() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        if not isinstance(step, BeforeToolCall | AfterToolCall):
            return None
        view = step.view.call
        return Decision.proceed(view.content if view is not None else "no view")

    return decide


@pytest.mark.anyio
async def test_a_tool_step_is_given_the_view_inspect_would_make() -> None:
    handler = Handler(reports_views())
    await handler.request(request(then=ran("Oslo", "rain", "toolu_1")))

    assert await handler.reply(reply(message(weather("Lima", "toolu_2")))) == Pass()

    explanations = _explanations(await handler.records("x"))
    assert len(explanations) == 2
    assert "get_weather" in explanations[0] and "Oslo" in explanations[0]
    assert "get_weather" in explanations[1] and "Lima" in explanations[1]


@pytest.mark.anyio
async def test_a_tool_the_provider_ran_is_not_judged_as_a_call() -> None:
    # the provider has run it by the time the reply arrives, so there is no
    # choice left to offer the sentinel
    handler = Handler(reports_histories())
    await handler.request(request(tools=_TOOLS))

    await handler.reply(
        reply(
            message(
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "weather in Oslo"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [],
                },
                {"type": "text", "text": "It is raining."},
                stop_reason="end_turn",
            )
        )
    )

    records = await handler.records("x")
    assert records is not None
    assert [record.stage for record in records] == ["model_input", "model_output"]


@pytest.mark.anyio
async def test_a_reply_passed_on_stays_in_the_history_after_a_compaction() -> None:
    handler = Handler(reports_histories())
    await handler.request(request("a"))
    await handler.reply(
        reply(
            message({"type": "text", "text": "Which city?"}, stop_reason="end_turn"),
            id="a",
        )
    )

    # the agent compacts its conversation before the reply is ever sent back
    await handler.request(
        request("b", then=[{"role": "user", "content": "Summary so far. Oslo."}])
    )
    await handler.reply(reply(message(_LOOKING, weather("Oslo", "toolu_1")), id="b"))

    explanations = _explanations(await handler.records("b"))
    assert explanations[-1] == (
        "BeforeToolCall|toolu_1|"
        f"{['You are helpful.', 'What is the weather?', 'Which city?', 'Summary so far. Oslo.', 'Looking.']}"
    )


@pytest.mark.anyio
async def test_a_reply_passed_on_and_sent_back_is_kept_once() -> None:
    handler = Handler(reports_histories())
    await handler.request(request("a"))
    await handler.reply(reply(message(_LOOKING, weather("Oslo", "toolu_1")), id="a"))

    await handler.request(request("b", then=ran("Oslo", "rain", "toolu_1")))

    explanations = _explanations(await handler.records("b"))
    assert explanations[0] == (
        "AfterToolCall|toolu_1|"
        f"{['You are helpful.', 'What is the weather?', 'Looking.']}"
    )
