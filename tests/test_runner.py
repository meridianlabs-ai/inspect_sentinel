from collections.abc import Awaitable, Callable
from typing import Any, cast

import anyio
import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.tool import ToolCall

from inspect_sentinel._context import Context
from inspect_sentinel._monitor import (
    ControlProtocol,
    Monitor,
    Protocols,
    monitor,
    protocol,
)
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
from tests._fakes import ListRecorder, after_step, before_step, runner_context


def _obs(name: str, suspicion: float | dict[str, float]) -> Reported[Observation]:
    return Reported(name=name, path=name, report=Observation.score(suspicion))


def _dec(name: str, action: Action, binding: bool = False) -> Reported[Decision]:
    return Reported(
        name=name,
        path=name,
        report=Decision(action=action, binding=binding),
    )


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


def test_strongest_prefers_a_binding_decision_on_ties() -> None:
    decisions = Decisions(
        [_dec("advisory", "reject"), _dec("human", "reject", binding=True)]
    )
    strongest = decisions.strongest()
    assert strongest is not None and strongest.name == "human"


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
        return Decision(
            action=action,
            modified=ToolCall(id="c1", function="bash", arguments={"cmd": "echo hi"})
            if action == "modify"
            else None,
        )

    return decide


@protocol
def insists() -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action="reject", binding=True)

    return decide


@protocol
def overrides(action: Action | None = None) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        await run_protocol(insists(), context, step)
        return Decision(action=action) if action is not None else None

    return decide


@protocol
def insists_when(ready: Callable[[], Awaitable[None]]) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        await ready()
        return Decision(action="reject", binding=True)

    return decide


@protocol
def launders(child: ControlProtocol) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        reported = await run_protocol(child, context, step)
        return None if reported is None else Decision(action=reported.report.action)

    return decide


@protocol
def weakens(child: ControlProtocol) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        await run_protocol(child, context, step)
        return Decision.clear()

    return decide


@protocol
def gathers(children: Protocols) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        await run_protocols(children, context, step)
        return Decision.clear()

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
    reported = await run_monitor(scores(0.4), context, before_step())
    assert reported is not None
    assert reported.report.suspicion == 0.4
    assert reported.name == "scores" and reported.path == "scores"
    assert [r.reported for r in recorder.records] == [reported]
    assert recorder.records[0].context.path == "scores"


@pytest.mark.anyio
async def test_run_monitor_skips_a_child_for_another_stage() -> None:
    recorder = ListRecorder()
    assert (
        await run_monitor(
            after_only(), runner_context(recorder=recorder), before_step()
        )
        is None
    )
    assert recorder.records == []


@pytest.mark.anyio
async def test_abstention_records_nothing() -> None:
    recorder = ListRecorder()
    assert (
        await run_monitor(abstains(), runner_context(recorder=recorder), before_step())
        is None
    )
    assert recorder.records == []


@pytest.mark.anyio
async def test_run_monitor_uses_the_given_name_and_extends_the_path() -> None:
    reported = await run_monitor(
        scores(), runner_context(path="attempt"), before_step(), name="judge"
    )
    assert reported is not None
    assert (reported.name, reported.path) == ("judge", "attempt/judge")


@pytest.mark.anyio
async def test_child_context_is_derived_under_the_layer() -> None:
    reported = await run_protocol(
        records_path(), runner_context(path="attempt"), before_step(), name="rule"
    )
    assert reported is not None
    assert reported.report.explanation == "attempt/rule"


@pytest.mark.anyio
async def test_run_monitor_rejects_a_protocol() -> None:
    with pytest.raises(TypeError, match="monitor"):
        await run_monitor(cast(Any, decides()), runner_context(), before_step())


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
        await run_monitor(scores(), bare, before_step())


@pytest.mark.anyio
async def test_exceptions_propagate() -> None:
    with pytest.raises(RuntimeError, match="exploded"):
        await run_monitors([raises()], runner_context(), before_step())


@pytest.mark.anyio
async def test_run_monitors_names_from_a_mapping_and_keeps_order() -> None:
    recorder = ListRecorder()
    observations = await run_monitors(
        {"low": scores(0.1), "high": scores(0.8)},
        runner_context(recorder=recorder),
        before_step(),
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
        before_step(),
    )
    assert len(recorder.records) == 2


@pytest.mark.anyio
async def test_duplicate_names_in_a_layer_are_an_error() -> None:
    with pytest.raises(ValueError, match="scores"):
        await run_monitors([scores(0.1), scores(0.2)], runner_context(), before_step())


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
            before_step(),
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
            before_step(),
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
        before_step(),
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
        await run_monitor(wrong, runner_context(), before_step())


@pytest.mark.anyio
async def test_sequence_children_are_named_by_registry_name() -> None:
    recorder = ListRecorder()
    observations = await run_monitors(
        [scores(0.1), abstains()], runner_context(recorder=recorder), before_step()
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
        before_step(),
    )
    assert [o.name for o in reports.observations] == ["m1", "m2"]
    assert [d.name for d in reports.decisions] == ["p1", "p2"]


@pytest.mark.anyio
async def test_run_protocols_rejects_a_monitor() -> None:
    with pytest.raises(TypeError, match="protocol"):
        await run_protocols(cast(Any, [scores()]), runner_context(), before_step())


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["", "a/b"])
async def test_invalid_instance_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await run_monitors({name: scores()}, runner_context(), before_step())


@pytest.mark.anyio
async def test_wrong_kind_is_rejected_before_any_child_runs() -> None:
    recorder = ListRecorder()
    with pytest.raises(TypeError, match="monitor"):
        await run_monitors(
            [scores(0.1), cast(Any, decides())],
            runner_context(recorder=recorder),
            before_step(),
        )
    assert recorder.records == []


@pytest.mark.anyio
async def test_uncalled_factory_is_rejected_before_any_child_runs() -> None:
    recorder = ListRecorder()
    with pytest.raises(TypeError, match="call it"):
        await run_protocols(
            {"a": decides(), "b": cast(Any, decides)},
            runner_context(recorder=recorder),
            before_step(),
        )
    assert recorder.records == []


@pytest.mark.anyio
async def test_iterators_are_rejected_rather_than_exhausted() -> None:
    with pytest.raises(TypeError, match="Mapping or a Sequence"):
        await run_protocols(
            cast(Any, (p for p in [decides("reject")])), runner_context(), before_step()
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

    recorder = ListRecorder()
    with anyio.fail_after(5), pytest.raises(RuntimeError, match="boom"):
        await run_children(
            {"w": waits(), "e": explodes_after_start()},
            runner_context(recorder=recorder),
            before_step(),
        )
    assert recorder.cancellations == [("w", "w")]
    assert recorder.records == []


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
            await run_monitors(
                [hangs()], runner_context(recorder=recorder), before_step()
            )
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
            before_step(),
        )
    assert [d.name for d in reports.decisions] == ["quick", "stop"]


@pytest.mark.anyio
async def test_sets_are_rejected_because_they_have_no_order() -> None:
    with pytest.raises(TypeError, match="Mapping or a Sequence"):
        await run_monitors(cast(Any, {scores()}), runner_context(), before_step())


@pytest.mark.anyio
async def test_undecorated_function_in_a_sequence_gets_the_friendly_error() -> None:
    async def plain(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    with pytest.raises(TypeError, match="@monitor or @protocol"):
        await run_monitors([cast(Any, plain)], runner_context(), before_step())
    with pytest.raises(TypeError, match="@monitor or @protocol"):
        await run_monitor(cast(Any, plain), runner_context(), before_step())


@pytest.mark.anyio
async def test_a_child_exception_keeps_its_own_cause() -> None:
    @monitor
    def chained() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            raise RuntimeError("wrapped") from KeyError("inner")

        return check

    with pytest.raises(RuntimeError, match="wrapped") as info:
        await run_monitors([chained()], runner_context(), before_step())
    assert isinstance(info.value.__cause__, KeyError)


class FailingRecorder(ListRecorder):
    def cancelled(self, context: Context, step: Step, name: str) -> None:
        raise OSError("transcript unavailable")


@pytest.mark.anyio
async def test_a_recorder_failure_during_cancellation_fails_loud() -> None:
    started = anyio.Event()

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

    with anyio.fail_after(5), pytest.raises(OSError, match="transcript unavailable"):
        await run_protocols(
            {"slow": slow(), "stop": terminates()},
            runner_context(recorder=FailingRecorder()),
            before_step(),
        )


@pytest.mark.anyio
async def test_terminate_cancels_monitors_at_their_next_await() -> None:
    started = anyio.Event()

    @monitor
    def no_await() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            return Observation.score(0.1)

        return check

    @monitor
    def hangs() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return check

    @protocol
    def terminates() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            return Decision(action="terminate")

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        reports = await run_children(
            {"quick": no_await(), "slow": hangs(), "stop": terminates()},
            runner_context(recorder=recorder),
            before_step(),
        )
    assert [o.name for o in reports.observations] == ["quick"]
    assert recorder.cancellations == [("slow", "slow")]


@pytest.mark.anyio
async def test_invalid_name_is_rejected_even_when_the_stage_does_not_match() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await run_monitor(after_only(), runner_context(), before_step(), name="a/b")


@pytest.mark.anyio
async def test_a_string_is_not_a_sequence_of_children() -> None:
    with pytest.raises(TypeError, match="Mapping or a Sequence"):
        await run_monitors(cast(Any, "scores"), runner_context(), before_step())


@pytest.mark.anyio
async def test_a_child_error_alongside_a_cancellation_surfaces_on_both_backends() -> (
    None
):
    entered = anyio.Event()

    @monitor
    def converts_cancellation() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            entered.set()
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                raise ValueError("boom") from None
            return None

        return check

    @monitor
    def hangs() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            await anyio.sleep_forever()
            return None

        return check

    seen: list[BaseException] = []
    async with anyio.create_task_group() as tg:

        async def run() -> None:
            try:
                await run_monitors(
                    {"c": converts_cancellation(), "h": hangs()},
                    runner_context(),
                    before_step(),
                )
            except ValueError as ex:
                seen.append(ex)

        tg.start_soon(run)
        await entered.wait()
        tg.cancel_scope.cancel()
    assert [str(ex) for ex in seen] == ["boom"]


@pytest.mark.anyio
@pytest.mark.parametrize("key", [1, None])
async def test_non_string_mapping_keys_are_a_configuration_error(key: Any) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        await run_monitors(cast(Any, {key: scores()}), runner_context(), before_step())


@pytest.mark.anyio
async def test_packaged_children_are_named_without_the_package_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_ai._util.registry as registry

    def packaged(o: object) -> str:
        return "acme"

    monkeypatch.setattr(registry, "get_installed_package_name", packaged)

    @monitor
    def leaf() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            return Observation.score(0.2)

        return check

    assert registry_info(leaf).name == "acme/leaf"
    recorder = ListRecorder()
    observations = await run_monitors(
        [leaf()], runner_context(recorder=recorder), before_step()
    )
    assert [o.name for o in observations] == ["leaf"]
    assert recorder.records[0].context.path == "leaf"


@pytest.mark.anyio
async def test_an_illegal_decision_fails_the_layer_that_returned_it() -> None:
    recorder = ListRecorder()
    with pytest.raises(ValueError, match="rule") as info:
        await run_protocol(
            decides("reject"),
            runner_context(recorder=recorder),
            after_step(),
            name="rule",
        )
    assert "already run" in str(info.value)
    assert recorder.records == []


@pytest.mark.anyio
@pytest.mark.parametrize("action", [None, "continue"])
async def test_a_layer_may_not_weaken_its_binding_child(
    action: Action | None,
) -> None:
    with pytest.raises(ValueError, match="binding") as info:
        await run_protocol(overrides(action), runner_context(), before_step())
    assert "insists" in str(info.value)


@pytest.mark.anyio
async def test_a_layer_at_the_binding_floor_stands() -> None:
    reported = await run_protocol(overrides("reject"), runner_context(), before_step())
    assert reported is not None and reported.report.action == "reject"


@pytest.mark.anyio
async def test_the_root_collects_nothing_so_one_call_does_not_constrain_the_next() -> (
    None
):
    root = runner_context()
    await run_protocol(overrides("reject"), root, before_step(), name="first")
    assert root.decisions == []
    reported = await run_protocol(decides("continue"), root, before_step())
    assert reported is not None and reported.report.action == "continue"
    assert root.decisions == []


@pytest.mark.anyio
async def test_authority_survives_a_layer_that_drops_the_flag() -> None:
    with pytest.raises(ValueError, match="binding") as info:
        await run_protocol(
            weakens(launders(insists())), runner_context(), before_step()
        )
    assert "launders" in str(info.value)


@pytest.mark.parametrize("waits", ["alice", "bob"])
@pytest.mark.anyio
async def test_the_named_culprit_does_not_depend_on_who_finished_first(
    waits: str,
) -> None:
    gate = anyio.Event()

    async def first() -> None:
        gate.set()

    async def second() -> None:
        await gate.wait()

    ready = {name: second if name == waits else first for name in ("alice", "bob")}
    with pytest.raises(ValueError, match="binding") as info:
        await run_protocol(
            gathers({name: insists_when(fn) for name, fn in ready.items()}),
            runner_context(),
            before_step(),
        )
    assert "'alice'" in str(info.value)
