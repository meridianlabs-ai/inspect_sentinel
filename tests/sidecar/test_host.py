import logging
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolCallView
from inspect_ai.util import Store

from inspect_sentinel import HumanAnswer, auto_mode, human, sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._host import HostContext, enter_layer, running_step
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._runner import run_monitors
from inspect_sentinel._step import AfterGenerate, BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse, SidecarHost
from inspect_sentinel.sidecar._host import ModelUsed, RunHost, noting_models
from tests._fakes import FakeHost, ListRecorder
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


_GENERATE: dict[str, Any] = {
    "model": "anthropic/claude-test",
    "conversation": "c",
    "input": [ChatMessageUser(content="Hello")],
    "agent_messages": None,
    "tools": [],
    "tool_choice": "none",
    "config": GenerateConfig(),
}

_TURN = AfterGenerate(
    **_GENERATE, output=ModelOutput.from_content("anthropic/claude-test", "Hi.")
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("step", "about"),
    [
        pytest.param(BeforeGenerate(**_GENERATE), "the model's request", id="request"),
        pytest.param(_TURN, "the model's turn", id="turn"),
    ],
)
async def test_a_generate_nobody_can_be_asked_about_ends_the_run(
    step: BeforeGenerate | AfterGenerate, about: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        answer = await SidecarHost().ask_human(step, (), ["approve", "terminate"])

    # neither a request nor a turn can be rejected, so the cautious answer
    # ends the run
    assert answer.decision == "terminate"
    assert about in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("asks", ["nobody", "the eval"])
@pytest.mark.parametrize(
    "step", [BeforeGenerate(**_GENERATE), _TURN], ids=["request", "turn"]
)
async def test_a_generate_cannot_be_offered_for_rejection(
    step: BeforeGenerate | AfterGenerate, asks: str
) -> None:
    asked: list[object] = []

    async def ask(
        step: object, escalations: Escalations, choices: object
    ) -> HumanAnswer:
        asked.append(step)
        return HumanAnswer(decision="approve")

    host = RunHost(SidecarHost(), None, ask if asks == "the eval" else None)

    # Neither a request nor a turn can be rejected, so the answer could not
    # become a decision. The protocol is told when it asks, as Inspect's own
    # host tells it.
    with pytest.raises(ValueError, match="cannot be offered 'reject'"):
        await host.ask_human(step, (), ["approve", "reject", "terminate"])

    assert asked == []


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


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_a_reply_from_inspects_cache_bills_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path))
    host = SidecarHost({"monitor": "mockllm/model"})
    cached = GenerateConfig(cache=True)
    judge = ModelUsed("monitor", "mockllm/model")

    with noting_models() as noted:
        first = await host.generate("Is this suspicious?", config=cached)
        billed = noted.billed[judge]
        # the same prompt again, answered from inspect_ai's cache
        await host.generate("Is this suspicious?", config=cached)

    assert first.usage is not None
    assert billed.total_tokens == first.usage.total_tokens > 0
    assert noted.billed == {judge: billed}

    # a model whose only call was answered from the cache was called, and
    # billed nothing
    with noting_models() as noted:
        await host.generate("Is this suspicious?", config=cached)
    assert (noted.called, noted.billed) == ([judge], {})


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("roles", "model", "role"),
    [
        # the configured role, the role missing (so model= is used), and no
        # role but a model name: the same resolution generate() uses
        ({"monitor": "mockllm/model"}, None, None),
        ({}, "mockllm/model", None),
        ({"judge": "mockllm/model"}, None, "judge"),
    ],
)
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_input_budget_and_count_tokens_resolve_as_generate_does(
    roles: dict[str, str], model: str | None, role: str | None
) -> None:
    host = SidecarHost(roles)

    budget = await host.input_budget(model=model, role=role)
    count = await host.count_tokens("Is this suspicious?", model=model, role=role)

    # mockllm reports a 128k window less the output reserve, and counts tokens
    assert 0 < budget < 128_000
    assert count > 0


@pytest.mark.anyio
async def test_input_budget_needs_a_model_for_the_role_or_by_name() -> None:
    with pytest.raises(ValueError, match="no model for role 'monitor'"):
        await SidecarHost().input_budget()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_run_host_passes_the_budget_and_the_count_through() -> None:
    host = RunHost(SidecarHost({"monitor": "mockllm/model"}), None)

    assert 0 < await host.input_budget() < 128_000
    assert await host.count_tokens("Is this suspicious?") > 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("roles", "judged_by"),
    [
        ({"auto_mode": "mockllm/model", "monitor": "mockllm/other"}, "auto_mode"),
        ({"monitor": "mockllm/model"}, "monitor"),
        ({}, None),
    ],
)
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_auto_mode_judges_with_the_role_the_sidecar_has(
    roles: dict[str, str], judged_by: str | None
) -> None:
    built = HostContext(
        host=RunHost(SidecarHost(roles), None), recorder=ListRecorder(), store=Store()
    )
    with noting_models() as noted, running_step(built.recorder, built.store):
        enter_layer("", "")
        observations = await run_monitors(
            {"auto_mode": auto_mode()}, built.context, _CALL
        )

    # mockllm never calls the answer tool, so the judge fails either way
    [failed] = observations.failed
    assert [used.role for used in noted.called] == ([judged_by] if judged_by else [])
    if judged_by is None:
        assert "no model to judge with" in str(failed.error)


@pytest.mark.anyio
async def test_auto_mode_in_a_run_with_no_eval_takes_the_users_messages_as_the_task() -> (
    None
):
    judged = FakeHost(
        outputs=[
            ModelOutput.for_tool_call(
                model="fake",
                tool_name="answer",
                tool_arguments={"explanation": "Rule: none\nRoutine.", "risk": 0.1},
            )
        ],
        roles=["auto_mode"],
    )
    built = HostContext(
        host=RunHost(judged, None), recorder=ListRecorder(), store=Store()
    )
    with running_step(built.recorder, built.store):
        enter_layer("", "")
        observations = await run_monitors(
            {"auto_mode": auto_mode()}, built.context, _CALL
        )

    [reported] = observations
    assert reported.report.suspicion == {"risk": 0.1}
    prompt = judged.inputs[0][0].text
    assert "There is no TASK block" in prompt
    assert "TASK block is the only" not in prompt
    assert "[BEGIN TASK" not in prompt
