import logging

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolCall, ToolCallView

from inspect_sentinel import human, sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse, SidecarHost
from tests.sidecar._traffic import message, reply, request, weather


@protocol
def escalates() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.escalate("every call")

    return decide


_CALL = BeforeToolCall(
    conversation="c",
    message="",
    message_id="m1",
    call=ToolCall(id="toolu_1", function="get_weather", arguments={}),
    view=ToolCallView(),
    input=None,
    agent_messages=None,
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        (["approve", "reject", "terminate"], "reject"),
        (["approve", "terminate"], "terminate"),
        (["approve"], "terminate"),
    ],
)
async def test_a_person_who_cannot_be_asked_is_answered_for_with_caution(
    choices: list[str], expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        answer = await SidecarHost().ask_human(_CALL, (), choices)

    assert answer.decision == expected
    assert answer.reason == "No person can be asked from the sidecar yet."
    assert "get_weather" in caplog.text


@pytest.mark.anyio
async def test_an_escalation_to_a_person_is_not_let_through() -> None:
    # a threshold escalates every call, and human() is asked to decide
    handler = Handler(sequential([escalates(), human(stages=["tool_call"])]))
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)


@pytest.mark.anyio
async def test_a_person_cannot_be_asked_at_a_generate_stage() -> None:
    step = BeforeGenerate(
        model="anthropic/claude-test",
        conversation="c",
        input=[ChatMessageUser(content="Hello")],
        agent_messages=None,
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
    )

    with pytest.raises(NotImplementedError, match="model_input stage"):
        await SidecarHost().ask_human(step, (), ["approve"])


@pytest.mark.anyio
async def test_generate_needs_a_model_for_the_role_or_by_name() -> None:
    with pytest.raises(ValueError, match="no model for role 'monitor'"):
        await SidecarHost().generate("Is this suspicious?")


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_generate_uses_the_role_model() -> None:
    host = SidecarHost({"monitor": "mockllm/model"})

    output = await host.generate("Is this suspicious?")

    # mockllm names itself as the model that answered
    assert output.model == "mockllm"
