import logging

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolCall, ToolCallView

from inspect_sentinel._step import BeforeGenerate, BeforeToolCall
from inspect_sentinel.sidecar import SidecarHost

_CALL = BeforeToolCall(
    conversation="c",
    message="",
    call=ToolCall(id="toolu_1", function="get_weather", arguments={}),
    view=ToolCallView(),
    input=None,
    history=None,
)


@pytest.mark.anyio
async def test_a_person_asked_at_a_tool_stage_is_queued_and_the_step_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        answer = await SidecarHost().ask_human(_CALL, ["approve", "reject"])

    assert answer.decision == "approve"
    assert "Queued for review" in caplog.text
    assert "get_weather" in caplog.text


@pytest.mark.anyio
async def test_a_person_asked_without_approve_on_offer_fails() -> None:
    with pytest.raises(ValueError, match="approve was not among"):
        await SidecarHost().ask_human(_CALL, ["reject", "terminate"])


@pytest.mark.anyio
async def test_a_person_cannot_be_asked_at_a_generate_stage() -> None:
    step = BeforeGenerate(
        model="anthropic/claude-test",
        conversation="c",
        input=[ChatMessageUser(content="Hello")],
        history=None,
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
    )

    with pytest.raises(NotImplementedError, match="model_input stage"):
        await SidecarHost().ask_human(step, ["approve"])


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
