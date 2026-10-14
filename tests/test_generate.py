from collections import UserList
from typing import Any

import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol, watched_stages
from inspect_sentinel._protocols import (
    concurrent,
    handle_escalation,
    observe_only,
    sequential,
    threshold,
)
from inspect_sentinel._report import (
    Action,
    Decision,
    Escalations,
    Observation,
)
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
    async def decide(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision(action=action)

    return decide


@protocol
def every_stage() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        return None

    return decide


@protocol
def tool_stages() -> ProtocolGroup:
    async def call(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return None

    async def result(
        context: Context,
        step: AfterToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return None

    return ProtocolGroup(call, result)


@monitor
def returns_a_configured_monitor(through: int = 1) -> Monitor:
    # a factory that returns another factory's configured instance (#75)
    if through > 1:
        return returns_a_configured_monitor(through - 1)
    return sees_input()


@protocol
def returns_a_configured_protocol() -> Protocol:
    return decides("terminate")


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


@pytest.mark.parametrize("action", ["proceed", "escalate", "terminate"])
@pytest.mark.parametrize("step", [before_generate_step(), after_generate_step()])
def test_proceed_escalate_and_terminate_are_legal_at_the_generate_stages(
    action: Action, step: Step
) -> None:
    validate_decision_shape(Decision(action=action), step)


@pytest.mark.parametrize(
    "decision",
    [
        Decision.reject(message="no"),
        Decision(
            action="modify",
            modified=before_step().call,
        ),
    ],
)
@pytest.mark.parametrize("step", [before_generate_step(), after_generate_step()])
def test_reject_and_modify_are_not_supported_at_the_generate_stages_yet(
    decision: Decision, step: Step
) -> None:
    with pytest.raises(ValueError, match="only 'proceed', 'escalate' and 'terminate'"):
        validate_decision_shape(decision, step)


@pytest.mark.anyio
async def test_an_unsupported_generate_decision_fails_the_protocol() -> None:
    with pytest.raises(ValueError, match="decides.*cannot reject yet"):
        await run_sentinel(decides("reject"), host_context(), after_generate_step())


@protocol
def escalates_at_generates() -> ProtocolGroup:
    async def before(
        context: Context,
        step: BeforeGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.escalate("unsure about the request")

    async def after(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.escalate("unsure about the reply")

    return ProtocolGroup(before, after)


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_generate_step(), after_generate_step()])
async def test_an_unhandled_escalate_at_a_generate_stage_terminates(
    step: Step,
) -> None:
    decision = await run_sentinel(
        sequential([escalates_at_generates()]), host_context(), step
    )
    assert decision is not None
    assert decision.action == "terminate"
    assert decision.explanation is not None
    assert decision.explanation.startswith("unhandled escalation from")


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["proceed", "terminate"])
@pytest.mark.parametrize("step", [before_generate_step(), after_generate_step()])
async def test_handle_escalation_answers_an_escalate_at_a_generate_stage(
    action: Any, step: Step
) -> None:
    root = sequential([escalates_at_generates(), handle_escalation(action)])
    decision = await run_sentinel(root, host_context(), step)
    assert decision is not None
    assert decision.action == action
    assert decision.explanation is not None
    assert decision.explanation.startswith("escalated by escalates_at_generates")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "step, expected",
    [
        (before_generate_step(), "escalate"),
        (after_generate_step(), "terminate"),
    ],
)
async def test_threshold_decides_at_the_generate_stages(
    step: Step, expected: str
) -> None:
    root = threshold(
        {"input": sees_input(), "output": sees_output()},
        escalate_at=0.1,
        terminate_at=0.2,
    )
    handled = await run_sentinel(
        sequential([root, handle_escalation("proceed")]), host_context(), step
    )
    alone = await run_sentinel(sequential([root]), host_context(), step)
    assert handled is not None and alone is not None
    assert handled.explanation is not None and alone.explanation is not None
    if expected == "escalate":
        # the threshold escalated: the handler answered it, and alone it is unhandled
        assert handled.action == "proceed"
        assert handled.explanation.startswith("escalated by threshold")
        assert alone.action == "terminate"
        assert alone.explanation.startswith("unhandled escalation from")
    else:
        assert handled.action == alone.action == "terminate"
        assert not alone.explanation.startswith("unhandled escalation")


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
        (lambda: decides("proceed"), {AfterGenerate}),
        (lambda: wraps_a_composition(sees_call()), {AfterGenerate, BeforeToolCall}),
        (lambda: handle_escalation("terminate"), set[type[Any]]()),
        (
            lambda: sequential(
                [threshold(sees_call(), escalate_at=0.5), handle_escalation("proceed")]
            ),
            {BeforeToolCall},
        ),
        (
            lambda: sequential(
                [
                    threshold(sees_output(), escalate_at=0.5),
                    handle_escalation("proceed"),
                ]
            ),
            {AfterGenerate},
        ),
        (
            lambda: threshold({"a": sees_input(), "b": sees_call()}, terminate_at=0.5),
            {BeforeGenerate, BeforeToolCall},
        ),
        (lambda: returns_a_configured_monitor(), {BeforeGenerate}),
        (lambda: returns_a_configured_monitor(through=2), {BeforeGenerate}),
        (lambda: returns_a_configured_protocol(), {AfterGenerate}),
        (
            lambda: observe_only(UserList([sees_input(), sees_call()])),
            {BeforeGenerate, BeforeToolCall},
        ),
    ],
)
def test_watched_stages_are_what_the_tree_watches(
    configure: Any, expected: set[type[Any]]
) -> None:
    assert watched_stages(configure()) == expected
