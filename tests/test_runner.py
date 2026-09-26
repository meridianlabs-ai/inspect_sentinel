from typing import Any, cast

import anyio
import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.tool import ToolCall, ToolCallView

from inspect_sentinel._context import Context
from inspect_sentinel._monitor import ControlProtocol, Monitor, monitor, protocol
from inspect_sentinel._report import Action, Decision, Observation, Reported
from inspect_sentinel._runner import (
    Decisions,
    Observations,
    Reports,
    run_children,
    run_monitor,
    run_monitors,
    run_protocol,
    run_protocols,
)
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from tests._fakes import ListRecorder, runner_context


def _obs(name: str, suspicion: float | dict[str, float]) -> Reported[Observation]:
    return Reported(name=name, path=name, report=Observation.score(suspicion))


def _dec(name: str, action: Action) -> Reported[Decision]:
    return Reported(name=name, path=name, report=Decision(action=action))


def test_observations_is_a_sequence() -> None:
    observations = Observations([_obs("a", 0.1), _obs("b", 0.2)])
    assert len(observations) == 2
    assert observations[1].name == "b"
    assert [o.name for o in observations] == ["a", "b"]


def test_max_suspicion_over_scalars_and_dicts() -> None:
    assert Observations([]).max_suspicion() is None
    assert Observations([_obs("a", 0.3), _obs("b", 0.7)]).max_suspicion() == 0.7
    assert (
        Observations([_obs("a", {"x": 0.2, "y": 0.9}), _obs("b", 0.5)]).max_suspicion()
        == 0.9
    )


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["continue", "reject", "modify"], "reject"),
        (["modify", "terminate", "reject"], "terminate"),
        (["continue", "modify"], "modify"),
        (["continue"], "continue"),
        (["escalate", "escalate"], None),
        ([], None),
    ],
)
def test_strongest_by_precedence(actions: list[Action], expected: str | None) -> None:
    decisions = Decisions([_dec(f"d{i}", a) for i, a in enumerate(actions)])
    strongest = decisions.strongest()
    assert (strongest.report.action if strongest else None) == expected


def test_strongest_prefers_the_first_on_ties() -> None:
    decisions = Decisions([_dec("first", "reject"), _dec("second", "reject")])
    strongest = decisions.strongest()
    assert strongest is not None and strongest.name == "first"


def test_reports_holds_both_families() -> None:
    reports = Reports(
        Observations([_obs("a", 0.1)]), Decisions([_dec("d", "continue")])
    )
    assert reports.observations.max_suspicion() == 0.1
    assert reports.decisions.strongest() is not None


def test_report_sequences_compare_by_class_and_items() -> None:
    assert Observations([]) == Observations([])
    assert Observations([]) != Decisions([])


def test_report_sequence_repr_shows_class_and_items() -> None:
    text = repr(Observations([_obs("a", 0.1)]))
    assert "Observations(" in text
    assert "a" in text


def _before() -> BeforeToolCall:
    return BeforeToolCall(
        conversation="c",
        message="",
        call=ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}),
        view=ToolCallView(),
        input=[ChatMessageUser(content="go")],
        history=[ChatMessageUser(content="go")],
    )


@monitor
def scores(value: float = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value)

    return check


@monitor
def abstains() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    return check


@monitor
def after_only() -> Monitor:
    async def check(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.9)

    return check


@monitor
def raises() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        raise RuntimeError("monitor exploded")

    return check


@protocol
def decides(action: Action = "continue") -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action=action)

    return decide


@protocol
def records_path() -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action="continue", explanation=context.path)

    return decide


@pytest.mark.anyio
async def test_run_monitor_records_and_returns_the_report() -> None:
    recorder = ListRecorder()
    context = runner_context(recorder=recorder)
    reported = await run_monitor(scores(0.4), context, _before())
    assert reported is not None
    assert reported.report.suspicion == 0.4
    assert reported.name == "scores" and reported.path == "scores"
    assert [r.reported for r in recorder.records] == [reported]
    assert recorder.records[0].context.path == "scores"


@pytest.mark.anyio
async def test_run_monitor_skips_a_child_for_another_stage() -> None:
    recorder = ListRecorder()
    assert (
        await run_monitor(after_only(), runner_context(recorder=recorder), _before())
        is None
    )
    assert recorder.records == []


@pytest.mark.anyio
async def test_abstention_records_nothing() -> None:
    recorder = ListRecorder()
    assert (
        await run_monitor(abstains(), runner_context(recorder=recorder), _before())
        is None
    )
    assert recorder.records == []


@pytest.mark.anyio
async def test_run_monitor_uses_the_given_name_and_extends_the_path() -> None:
    reported = await run_monitor(
        scores(), runner_context(path="attempt"), _before(), name="judge"
    )
    assert reported is not None
    assert (reported.name, reported.path) == ("judge", "attempt/judge")


@pytest.mark.anyio
async def test_child_context_is_derived_under_the_layer() -> None:
    reported = await run_protocol(
        records_path(), runner_context(path="attempt"), _before(), name="rule"
    )
    assert reported is not None
    assert reported.report.explanation == "attempt/rule"


@pytest.mark.anyio
async def test_run_monitor_rejects_a_protocol() -> None:
    with pytest.raises(TypeError, match="monitor"):
        await run_monitor(cast(Any, decides()), runner_context(), _before())


@pytest.mark.anyio
async def test_runner_requires_a_runner_context() -> None:
    parent = runner_context()
    bare = Context(
        task=None,
        task_description=None,
        sample_id=None,
        epoch=None,
        sample_description=None,
        input="p",
        metadata={},
        path="",
        store=parent.store,
        host=parent.host,
    )
    with pytest.raises(TypeError, match="RunnerContext"):
        await run_monitor(scores(), bare, _before())


@pytest.mark.anyio
async def test_exceptions_propagate() -> None:
    with pytest.raises(RuntimeError, match="exploded"):
        await run_monitors([raises()], runner_context(), _before())


@pytest.mark.anyio
async def test_run_monitors_names_from_a_mapping_and_keeps_order() -> None:
    recorder = ListRecorder()
    observations = await run_monitors(
        {"low": scores(0.1), "high": scores(0.8)},
        runner_context(recorder=recorder),
        _before(),
    )
    assert [o.name for o in observations] == ["low", "high"]
    assert observations.max_suspicion() == 0.8
    assert sorted(r.reported.name for r in recorder.records) == ["high", "low"]


@pytest.mark.anyio
async def test_run_monitors_records_reports_the_caller_ignores() -> None:
    recorder = ListRecorder()
    await run_monitors(
        {"a": scores(0.1), "b": abstains(), "c": scores(0.2)},
        runner_context(recorder=recorder),
        _before(),
    )
    assert len(recorder.records) == 2


@pytest.mark.anyio
async def test_duplicate_names_in_a_layer_are_an_error() -> None:
    with pytest.raises(ValueError, match="scores"):
        await run_monitors([scores(0.1), scores(0.2)], runner_context(), _before())


@pytest.mark.anyio
async def test_run_monitors_is_concurrent() -> None:
    first_started = anyio.Event()
    second_started = anyio.Event()

    @monitor
    def waits_for_second() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            first_started.set()
            await second_started.wait()
            return Observation.score(0.1)

        return check

    @monitor
    def waits_for_first() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            second_started.set()
            await first_started.wait()
            return Observation.score(0.2)

        return check

    with anyio.fail_after(5):
        observations = await run_monitors(
            {"a": waits_for_second(), "b": waits_for_first()},
            runner_context(),
            _before(),
        )
    assert observations.max_suspicion() == 0.2


@pytest.mark.anyio
async def test_terminate_cancels_siblings() -> None:
    started = anyio.Event()
    finished = anyio.Event()

    @protocol
    def slow() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            started.set()
            await anyio.sleep_forever()
            finished.set()
            return Decision(action="continue")

        return decide

    @protocol
    def terminates() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            return Decision(action="terminate")

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decisions = await run_protocols(
            {"slow": slow(), "stop": terminates()},
            runner_context(recorder=recorder),
            _before(),
        )
    strongest = decisions.strongest()
    assert strongest is not None and strongest.report.action == "terminate"
    assert [d.name for d in decisions] == ["stop"]
    assert not finished.is_set()
    assert recorder.cancellations == [("slow", "slow")]
    assert [r.reported.name for r in recorder.records] == ["stop"]


@pytest.mark.anyio
async def test_run_children_splits_by_kind() -> None:
    recorder = ListRecorder()
    reports = await run_children(
        {"m": scores(0.3), "p": decides("reject"), "skip": after_only()},
        runner_context(recorder=recorder),
        _before(),
    )
    assert reports.observations.max_suspicion() == 0.3
    strongest = reports.decisions.strongest()
    assert strongest is not None and strongest.report.action == "reject"
    assert len(recorder.records) == 2


@pytest.mark.anyio
async def test_wrong_report_type_at_runtime_is_an_error() -> None:
    async def lies(context: Context, step: BeforeToolCall) -> Observation | None:
        return cast(Any, Decision(action="continue"))

    def factory() -> Any:
        return lies

    wrong = monitor(cast(Any, factory))()
    with pytest.raises(TypeError, match="must return Observation"):
        await run_monitor(wrong, runner_context(), _before())


@pytest.mark.anyio
async def test_sequence_children_are_named_by_registry_name() -> None:
    recorder = ListRecorder()
    observations = await run_monitors(
        [scores(0.1), abstains()], runner_context(recorder=recorder), _before()
    )
    assert [o.name for o in observations] == ["scores"]
    assert [r.reported.path for r in recorder.records] == ["scores"]


@pytest.mark.anyio
async def test_run_children_keeps_configuration_order_per_family() -> None:
    reports = await run_children(
        {
            "p1": decides("continue"),
            "m1": scores(0.1),
            "p2": decides("modify"),
            "m2": scores(0.2),
        },
        runner_context(),
        _before(),
    )
    assert [o.name for o in reports.observations] == ["m1", "m2"]
    assert [d.name for d in reports.decisions] == ["p1", "p2"]


@pytest.mark.anyio
async def test_run_protocols_rejects_a_monitor() -> None:
    with pytest.raises(TypeError, match="protocol"):
        await run_protocols(cast(Any, [scores()]), runner_context(), _before())


@pytest.mark.anyio
async def test_run_monitors_names_an_uncalled_factory_error() -> None:
    with pytest.raises(TypeError, match="call it"):
        await run_monitors([cast(Any, scores)], runner_context(), _before())


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["", "a/b"])
async def test_invalid_instance_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await run_monitors({name: scores()}, runner_context(), _before())


@pytest.mark.anyio
async def test_child_rejects_invalid_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        runner_context().child("")


@pytest.mark.anyio
async def test_wrong_kind_is_rejected_before_any_child_runs() -> None:
    recorder = ListRecorder()
    with pytest.raises(TypeError, match="monitor"):
        await run_monitors(
            [scores(0.1), cast(Any, decides())],
            runner_context(recorder=recorder),
            _before(),
        )
    assert recorder.records == []


@pytest.mark.anyio
async def test_uncalled_factory_is_rejected_before_any_child_runs() -> None:
    recorder = ListRecorder()
    with pytest.raises(TypeError, match="call it"):
        await run_protocols(
            {"a": decides(), "b": cast(Any, decides)},
            runner_context(recorder=recorder),
            _before(),
        )
    assert recorder.records == []


@pytest.mark.anyio
async def test_iterators_are_rejected_rather_than_exhausted() -> None:
    with pytest.raises(TypeError, match="Mapping or a Sequence"):
        await run_protocols(
            cast(Any, (p for p in [decides("reject")])), runner_context(), _before()
        )


@pytest.mark.anyio
async def test_a_raising_child_surfaces_while_a_sibling_is_mid_await() -> None:
    started = anyio.Event()

    @protocol
    def waits() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return decide

    @monitor
    def explodes_after_start() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            await started.wait()
            raise RuntimeError("boom")

        return check

    with anyio.fail_after(5), pytest.raises(RuntimeError, match="boom"):
        await run_children(
            {"w": waits(), "e": explodes_after_start()}, runner_context(), _before()
        )


@pytest.mark.anyio
async def test_external_cancellation_propagates_through_the_runner() -> None:
    entered = anyio.Event()

    @monitor
    def hangs() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            entered.set()
            await anyio.sleep_forever()
            return None

        return check

    completed = False
    recorder = ListRecorder()

    async with anyio.create_task_group() as tg:

        async def run() -> None:
            nonlocal completed
            await run_monitors([hangs()], runner_context(recorder=recorder), _before())
            completed = True

        tg.start_soon(run)
        await entered.wait()
        tg.cancel_scope.cancel()
    assert tg.cancel_scope.cancel_called
    assert not completed
    assert recorder.cancellations == [("hangs", "hangs")]
    assert recorder.records == []


@pytest.mark.anyio
async def test_terminate_keeps_results_collected_before_it() -> None:
    started = anyio.Event()

    @protocol
    def quick() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            return Decision(action="continue")

        return decide

    @protocol
    def slow() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return decide

    @protocol
    def terminates() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            return Decision(action="terminate")

        return decide

    with anyio.fail_after(5):
        reports = await run_children(
            {"quick": quick(), "slow": slow(), "stop": terminates()},
            runner_context(),
            _before(),
        )
    assert [d.name for d in reports.decisions] == ["quick", "stop"]
