from typing import Any

import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol, watched_stages
from inspect_sentinel._protocols import concurrent, observe_only, sequential, threshold
from inspect_sentinel._report import Action, Decision, Observation
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Monitor, MonitorGroup, Protocol, ProtocolGroup
from inspect_sentinel._validate import validate_decision_shape
from tests._fakes import (
    ListRecorder,
    after_generate_step,
    after_step,
    before_generate_step,
    before_step,
    host_context,
)


@monitor
def sees_input() -> Monitor:
    async def check(context: Context, step: BeforeGenerate) -> Observation | None:
        return Observation.score(0.1, f"{len(step.input)} messages")

    return check


@monitor
def sees_output() -> Monitor:
    async def check(context: Context, step: AfterGenerate) -> Observation | None:
        return Observation.score(0.2, step.output.completion)

    return check


@monitor
def sees_call() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.3)

    return check


@monitor
def sees_both_generate_stages() -> MonitorGroup:
    async def before(context: Context, step: BeforeGenerate) -> Observation | None:
        return Observation.score(0.4)

    async def after(context: Context, step: AfterGenerate) -> Observation | None:
        return Observation.score(0.5)

    return MonitorGroup(before, after)


@protocol
def decides(action: Action) -> Protocol:
    async def decide(context: Context, step: AfterGenerate) -> Decision | None:
        return Decision(action=action)

    return decide


@protocol
def every_stage() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return None

    return decide


@protocol
def tool_stages() -> ProtocolGroup:
    async def call(context: Context, step: BeforeToolCall) -> Decision | None:
        return None

    async def result(context: Context, step: AfterToolCall) -> Decision | None:
        return None

    return ProtocolGroup(call, result)


@protocol
def wraps_a_composition(monitors: Monitor) -> Protocol:
    return concurrent([observe_only(monitors), decides("terminate")])


@pytest.mark.anyio
@pytest.mark.parametrize(
    "step, expected",
    [
        (before_generate_step(), ["sees_input"]),
        (after_generate_step(), ["sees_output"]),
        (before_step(), ["sees_call"]),
        (after_step(), []),
    ],
)
async def test_each_monitor_runs_only_at_its_stage(
    step: Step, expected: list[str]
) -> None:
    recorder = ListRecorder()
    root = observe_only([sees_input(), sees_output(), sees_call()])
    await run_sentinel(root, host_context(recorder=recorder), step)
    assert [r.reported.name for r in recorder.records] == expected


@monitor
def fails_at_generate() -> Monitor:
    async def check(context: Context, step: BeforeGenerate) -> Observation | None:
        raise ValueError("model unavailable")

    return check


@pytest.mark.anyio
async def test_a_monitor_failing_at_a_generate_stage_is_recorded_and_observe_only_proceeds() -> (
    None
):
    recorder = ListRecorder()
    root = observe_only({"broken": fails_at_generate(), "ok": sees_input()})
    decision = await run_sentinel(
        root, host_context(recorder=recorder), before_generate_step()
    )
    assert decision is None
    assert [f.name for f in recorder.failures] == ["broken"]
    assert [r.reported.name for r in recorder.records] == ["ok"]


@pytest.mark.parametrize("action", ["continue", "terminate"])
@pytest.mark.parametrize("step", [before_generate_step(), after_generate_step()])
def test_continue_and_terminate_are_legal_at_the_generate_stages(
    action: Action, step: Step
) -> None:
    validate_decision_shape(Decision(action=action), step)


@pytest.mark.parametrize(
    "decision",
    [
        Decision.reject(message="no"),
        Decision.escalate("unsure"),
        Decision(
            action="modify",
            modified=before_step().call,
        ),
    ],
)
@pytest.mark.parametrize("step", [before_generate_step(), after_generate_step()])
def test_other_actions_are_not_supported_at_the_generate_stages_yet(
    decision: Decision, step: Step
) -> None:
    with pytest.raises(ValueError, match="only 'continue' and 'terminate'"):
        validate_decision_shape(decision, step)


@pytest.mark.anyio
async def test_an_unsupported_generate_decision_fails_the_protocol() -> None:
    with pytest.raises(ValueError, match="decides.*cannot reject yet"):
        await run_sentinel(decides("reject"), host_context(), after_generate_step())


@pytest.mark.parametrize(
    "configure, expected",
    [
        (lambda: observe_only([sees_call()]), {BeforeToolCall}),
        (
            lambda: observe_only({"a": sees_input(), "b": sees_output()}),
            {BeforeGenerate, AfterGenerate},
        ),
        (
            lambda: observe_only(sees_both_generate_stages()),
            {BeforeGenerate, AfterGenerate},
        ),
        (lambda: concurrent([threshold(sees_call(), reject_at=0.5)]), {BeforeToolCall}),
        (
            lambda: concurrent([sequential([sees_output(), tool_stages()])]),
            {AfterGenerate, BeforeToolCall, AfterToolCall},
        ),
        (
            lambda: concurrent([every_stage()]),
            {BeforeGenerate, AfterGenerate, BeforeToolCall, AfterToolCall},
        ),
        (lambda: decides("continue"), {AfterGenerate}),
        (lambda: wraps_a_composition(sees_call()), {AfterGenerate, BeforeToolCall}),
    ],
)
def test_watched_stages_are_what_the_tree_watches(
    configure: Any, expected: set[type[Any]]
) -> None:
    assert watched_stages(configure()) == expected
