import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.tool import ToolCall

from inspect_sentinel._context import Context
from inspect_sentinel._monitor import (
    ControlProtocol,
    Monitor,
    monitor,
    protocol,
    step_types,
)
from inspect_sentinel._protocols import concurrent, observe, threshold
from inspect_sentinel._report import Action, Decision, Observation, Suspicion
from inspect_sentinel._runner import run_protocol
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from tests._fakes import ListRecorder, after_step, before_step, runner_context


@monitor
def graded(value: Suspicion = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value)

    return check


@monitor
def silent() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    return check


@monitor
def afterwards() -> Monitor:
    async def check(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.9)

    return check


@protocol
def says(action: Action = "continue") -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action=action)

    return decide


@protocol
def rewrites() -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(
            action="modify",
            modified=ToolCall(id="c1", function="bash", arguments={"cmd": "echo hi"}),
        )

    return decide


async def _run(
    instance: ControlProtocol, step: Step, recorder: ListRecorder
) -> Decision | None:
    reported = await run_protocol(instance, runner_context(recorder=recorder), step)
    return reported.report if reported is not None else None


@pytest.mark.anyio
async def test_observe_records_every_observation_and_never_acts() -> None:
    recorder = ListRecorder()
    decision = await _run(
        observe({"low": graded(0.2), "high": graded(0.8)}), before_step(), recorder
    )
    assert decision is None
    assert sorted(r.reported.name for r in recorder.records) == ["high", "low"]


@pytest.mark.anyio
async def test_observe_records_nothing_for_a_monitor_of_another_stage() -> None:
    recorder = ListRecorder()
    assert await _run(observe([afterwards()]), before_step(), recorder) is None
    assert recorder.records == []


@pytest.mark.anyio
async def test_concurrent_over_monitors_only_does_not_decide() -> None:
    recorder = ListRecorder()
    assert await _run(concurrent([graded(0.3)]), before_step(), recorder) is None
    assert [r.reported.name for r in recorder.records] == ["graded"]


@pytest.mark.anyio
async def test_concurrent_reads_all_escalate_as_continue() -> None:
    decision = await _run(
        concurrent({"a": says("escalate"), "b": says("escalate")}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "continue"


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["continue", "reject"], "reject"),
        (["reject", "terminate"], "terminate"),
        (["escalate", "continue"], "continue"),
    ],
)
@pytest.mark.anyio
async def test_concurrent_takes_the_strongest_decision(
    actions: list[Action], expected: Action
) -> None:
    children = {f"p{index}": says(action) for index, action in enumerate(actions)}
    decision = await _run(concurrent(children), before_step(), ListRecorder())
    assert decision is not None and decision.action == expected


@pytest.mark.anyio
async def test_concurrent_honours_a_lone_modify() -> None:
    decision = await _run(concurrent([rewrites()]), before_step(), ListRecorder())
    assert decision is not None and decision.action == "modify"
    assert decision.modified is not None


@pytest.mark.anyio
async def test_concurrent_turns_a_contested_modify_into_a_reject() -> None:
    decision = await _run(
        concurrent({"rewriter": rewrites(), "approver": says("continue")}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation is not None and "rewriter" in decision.explanation


@pytest.mark.anyio
async def test_concurrent_records_monitors_and_protocols_alike() -> None:
    recorder = ListRecorder()
    await _run(
        concurrent({"m": graded(0.4), "p": says("continue")}), before_step(), recorder
    )
    assert sorted(r.reported.path for r in recorder.records) == [
        "concurrent",
        "concurrent/m",
        "concurrent/p",
    ]


@pytest.mark.parametrize(
    ("value", "expected"), [(0.2, "continue"), (0.5, "reject"), (0.7, "reject")]
)
@pytest.mark.anyio
async def test_threshold_rejects_at_or_above_its_constant(
    value: float, expected: Action
) -> None:
    decision = await _run(
        threshold([graded(value)], reject_at=0.5), before_step(), ListRecorder()
    )
    assert decision is not None and decision.action == expected


@pytest.mark.anyio
async def test_threshold_terminates_at_or_above_its_second_constant() -> None:
    decision = await _run(
        threshold([graded(0.95)], reject_at=0.5, terminate_at=0.9),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "terminate"


@pytest.mark.anyio
async def test_threshold_abstains_when_nobody_observed() -> None:
    decision = await _run(
        threshold([silent()], reject_at=0.5), before_step(), ListRecorder()
    )
    assert decision is None


@pytest.mark.anyio
async def test_threshold_does_not_run_after_a_tool_call() -> None:
    recorder = ListRecorder()
    decision = await _run(
        threshold([graded(0.9)], reject_at=0.5), after_step(), recorder
    )
    assert decision is None
    assert recorder.records == []


@pytest.mark.anyio
async def test_threshold_thresholds_the_highest_dimension() -> None:
    decision = await _run(
        threshold([graded({"x": 0.1, "y": 0.8})], reject_at=0.5),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation == "suspicion 0.80"


def test_the_shipped_protocols_register_under_the_package() -> None:
    assert [registry_info(f).name for f in (observe, concurrent, threshold)] == [
        "inspect_sentinel/observe",
        "inspect_sentinel/concurrent",
        "inspect_sentinel/threshold",
    ]
    assert {registry_info(f).type for f in (observe, concurrent, threshold)} == {
        "protocol"
    }


def test_step_types_come_from_the_annotations() -> None:
    both = frozenset({BeforeToolCall, AfterToolCall})
    assert step_types(threshold([graded()], reject_at=0.5)) == frozenset(
        {BeforeToolCall}
    )
    assert step_types(concurrent([graded()])) == both
    assert step_types(observe([graded()])) == both
