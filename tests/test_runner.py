from collections.abc import Callable, Sequence
from typing import Any, cast

import anyio
import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.tool import ToolCall
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context
from inspect_sentinel._final import Final, final
from inspect_sentinel._monitor import (
    Children,
    ControlProtocol,
    Monitor,
    Monitors,
    Protocols,
    monitor,
    protocol,
)
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Action, Decision, Observation, Reported
from inspect_sentinel._resolve import resolve_sentinel
from inspect_sentinel._runner import (
    Decisions,
    Observations,
    Reports,
    run_children,
    run_monitors,
    run_protocols,
    run_root,
)
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from tests._fakes import ListRecorder, after_step, before_step, runner_context


def _obs(name: str, suspicion: float | dict[str, float]) -> Reported[Observation]:
    return Reported(
        name=name, path=name, report=Observation.score(suspicion), function="check"
    )


def _dec(name: str, action: Action) -> Reported[Decision]:
    return Reported(
        name=name, path=name, report=Decision(action=action), function="decide"
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
def finalizes(action: Action = "reject") -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        final(Decision(action=action))

    return decide


@protocol
def wrapper(child: ControlProtocol) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        await run_protocols(child, context, step)
        return Decision.clear()

    return decide


@protocol
def records_path() -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action="continue", explanation=context.path)

    return decide


@pytest.mark.anyio
async def test_a_single_monitor_is_recorded_and_returned() -> None:
    recorder = ListRecorder()
    context = runner_context(recorder=recorder)
    [reported] = await run_monitors(scores(0.4), context, before_step())
    assert reported.report.suspicion == 0.4
    assert reported.name == "scores" and reported.path == "scores"
    assert [r.reported for r in recorder.records] == [reported]
    assert recorder.records[0].context.path == "scores"


@pytest.mark.anyio
async def test_a_child_for_another_stage_is_skipped() -> None:
    recorder = ListRecorder()
    assert await run_monitors(
        after_only(), runner_context(recorder=recorder), before_step()
    ) == Observations([])
    assert recorder.records == []


@pytest.mark.anyio
async def test_abstention_records_nothing() -> None:
    recorder = ListRecorder()
    assert await run_monitors(
        abstains(), runner_context(recorder=recorder), before_step()
    ) == Observations([])
    assert recorder.records == []


@pytest.mark.anyio
async def test_a_mapping_key_names_the_child_and_extends_the_path() -> None:
    [reported] = await run_monitors(
        {"judge": scores()}, runner_context(path="attempt"), before_step()
    )
    assert (reported.name, reported.path) == ("judge", "attempt/judge")


@pytest.mark.anyio
async def test_child_context_is_derived_under_the_layer() -> None:
    [reported] = await run_protocols(
        {"rule": records_path()}, runner_context(path="attempt"), before_step()
    )
    assert reported.report.explanation == "attempt/rule"


@pytest.mark.anyio
async def test_run_monitors_rejects_a_single_protocol() -> None:
    with pytest.raises(TypeError, match="monitor"):
        await run_monitors(cast(Any, decides()), runner_context(), before_step())


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
        await run_monitors(scores(), bare, before_step())


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
        await run_monitors(wrong, runner_context(), before_step())


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
        await run_monitors(cast(Any, plain), runner_context(), before_step())


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
        await run_monitors({"a/b": after_only()}, runner_context(), before_step())


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
        await run_protocols(
            {"rule": decides("reject")},
            runner_context(recorder=recorder),
            after_step(),
        )
    assert "already run" in str(info.value)
    assert recorder.records == []


@pytest.mark.anyio
async def test_final_ends_the_step_past_every_layer_above() -> None:
    started = anyio.Event()

    @protocol
    def slow() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return decide

    @protocol
    def leaf() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            final(Decision.reject("a person said no"))

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decision = await run_root(
            wrapper(concurrent({"slow": slow(), "leaf": leaf()})),
            runner_context(recorder=recorder),
            before_step(),
        )
    assert decision == Decision.reject("a person said no")
    assert [(r.reported.path, r.reported.report) for r in recorder.records] == [
        ("concurrent/leaf", decision)
    ]
    assert recorder.cancellations == [("concurrent/slow", "slow")]
    assert recorder.bypassed_layers == [("concurrent", "concurrent"), ("", "wrapper")]


@pytest.mark.anyio
async def test_a_final_propagating_below_the_root_is_not_yet_recorded() -> None:
    recorder = ListRecorder()
    with pytest.raises(Final):
        await run_protocols(
            wrapper(finalizes("reject")),
            runner_context(recorder=recorder),
            before_step(),
        )
    assert recorder.records == []
    assert recorder.bypassed_layers == [("wrapper", "wrapper")]


@pytest.mark.anyio
async def test_a_final_decision_illegal_for_the_stage_fails_the_protocol() -> None:
    recorder = ListRecorder()
    with pytest.raises(ValueError, match="'finalizes'") as info:
        await run_protocols(
            wrapper(finalizes("reject")),
            runner_context(recorder=recorder),
            after_step(),
        )
    assert "already run" in str(info.value)
    assert recorder.records == [] and recorder.bypassed_layers == []


@pytest.mark.anyio
async def test_the_loser_of_a_final_race_is_recorded_as_superseded() -> None:
    waiting = anyio.Event()
    gate = anyio.Event()

    @protocol
    def opens() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await waiting.wait()
            gate.set()
            final(Decision.reject("opens"))

        return decide

    @protocol
    def follows() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            waiting.set()
            await gate.wait()
            final(Decision.terminate("follows"))

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decision = await run_root(
            concurrent({"opens": opens(), "follows": follows()}),
            runner_context(recorder=recorder),
            before_step(),
        )
    assert decision is not None
    won = decision.explanation
    assert won in ("opens", "follows")
    assert [(r.reported.name, r.reported.report) for r in recorder.records] == [
        (won, decision)
    ]
    assert [r.reported.name for r in recorder.supersessions] == [
        "follows" if won == "opens" else "opens"
    ]
    assert recorder.cancellations == []


@pytest.mark.anyio
async def test_an_error_beside_a_final_decision_is_not_hidden_by_it() -> None:
    waiting = anyio.Event()
    gate = anyio.Event()

    @protocol
    def opens() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await waiting.wait()
            gate.set()
            final(Decision.reject("opens"))

        return decide

    @monitor
    def explodes() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            waiting.set()
            await gate.wait()
            raise RuntimeError("boom")

        return check

    recorder = ListRecorder()
    with anyio.fail_after(5), pytest.raises(RuntimeError, match="boom"):
        await run_children(
            {"opens": opens(), "explodes": explodes()},
            runner_context(recorder=recorder),
            before_step(),
        )
    assert recorder.records == []
    assert [r.reported.name for r in recorder.supersessions] == ["opens"]


@pytest.mark.anyio
async def test_a_monitor_that_calls_final_is_a_bug() -> None:
    @monitor
    def decides_instead() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            final(Decision.reject())

        return check

    recorder = ListRecorder()
    with pytest.raises(TypeError, match="decides_instead"):
        await run_monitors(
            [decides_instead()], runner_context(recorder=recorder), before_step()
        )
    assert recorder.records == []


@pytest.mark.anyio
async def test_a_final_inside_a_protocols_own_task_group_still_ends_the_step() -> None:
    started = anyio.Event()

    @protocol
    def slow() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return decide

    @protocol
    def leaf() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            final(Decision.reject("fanned out"))

        return decide

    @protocol
    def custom() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_protocols, slow(), context, step)
                tg.start_soon(run_protocols, leaf(), context, step)
            return Decision.clear()

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decision = await run_root(
            custom(), runner_context(recorder=recorder), before_step()
        )
    assert decision == Decision.reject("fanned out")
    assert [(r.reported.path, r.reported.report) for r in recorder.records] == [
        ("leaf", decision)
    ]
    assert recorder.bypassed_layers == [("", "custom")]


@pytest.mark.anyio
async def test_a_final_race_inside_a_protocols_own_task_group_records_one_decision() -> (
    None
):
    waiting = anyio.Event()
    gate = anyio.Event()

    @protocol
    def opens() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await waiting.wait()
            gate.set()
            final(Decision.reject("opens"))

        return decide

    @protocol
    def follows() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            waiting.set()
            await gate.wait()
            final(Decision.terminate("follows"))

        return decide

    @protocol
    def custom() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_protocols, opens(), context, step)
                tg.start_soon(run_protocols, follows(), context, step)
            return Decision.clear()

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decision = await run_root(
            wrapper(custom()), runner_context(recorder=recorder), before_step()
        )
    assert decision is not None
    won = decision.explanation
    assert [(r.reported.path, r.reported.report) for r in recorder.records] == [
        (f"custom/{won}", decision)
    ]
    assert [r.reported.path for r in recorder.supersessions] == [
        f"custom/{'follows' if won == 'opens' else 'opens'}"
    ]
    assert recorder.bypassed_layers == [("custom", "custom"), ("", "wrapper")]


@pytest.mark.anyio
async def test_a_final_propagating_through_a_monitor_names_the_protocol() -> None:
    @monitor
    def consults() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            await run_protocols(finalizes("reject"), context, step)
            return None

        return check

    recorder = ListRecorder()
    decision = await run_root(
        concurrent([consults()]), runner_context(recorder=recorder), before_step()
    )
    assert decision is not None and decision.action == "reject"
    assert [r.reported.path for r in recorder.records] == ["consults/finalizes"]
    assert recorder.bypassed_layers == [("consults", "consults"), ("", "concurrent")]


@pytest.mark.anyio
async def test_run_root_returns_a_final_decision_and_records_the_root_bypassed() -> (
    None
):
    recorder = ListRecorder()
    decision = await run_root(
        resolve_sentinel([finalizes("reject")]),
        runner_context(recorder=recorder),
        before_step(),
    )
    assert decision == Decision(action="reject")
    assert [(r.reported.path, r.reported.report) for r in recorder.records] == [
        ("finalizes", decision)
    ]
    assert recorder.bypassed_layers == [("", "concurrent")]


@pytest.mark.anyio
async def test_run_root_records_the_roots_own_decision_at_the_empty_path() -> None:
    recorder = ListRecorder()
    decision = await run_root(
        resolve_sentinel([decides("reject")]),
        runner_context(recorder=recorder),
        before_step(),
    )
    assert decision == Decision(action="reject")
    assert [(r.reported.name, r.reported.path) for r in recorder.records] == [
        ("decides", "decides"),
        ("concurrent", ""),
    ]


@pytest.mark.anyio
async def test_run_root_checks_the_roots_decision_shape() -> None:
    @protocol
    def illegal() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            return Decision(action="modify")

        return decide

    with pytest.raises(ValueError, match="'illegal'"):
        await run_root(illegal(), runner_context(), before_step())


@pytest.mark.anyio
async def test_run_root_needs_the_top_layer_context() -> None:
    with pytest.raises(ValueError, match="path"):
        await run_root(
            resolve_sentinel([decides()]), runner_context(path="x"), before_step()
        )


@pytest.mark.anyio
async def test_an_error_beside_a_protocols_own_final_in_its_task_group_surfaces() -> (
    None
):
    gate = anyio.Event()

    async def finals_now() -> None:
        await gate.wait()
        final(Decision.reject("own"))

    async def explodes() -> None:
        gate.set()
        raise RuntimeError("boom")

    @protocol
    def custom() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(finals_now)
                tg.start_soon(explodes)
            return None

        return decide

    with anyio.fail_after(5), pytest.raises(RuntimeError, match="boom"):
        await run_children([custom()], runner_context(), before_step())


@pytest.mark.anyio
async def test_an_error_beside_a_claimed_final_in_a_protocols_own_group_surfaces() -> (
    None
):
    waiting = anyio.Event()
    gate = anyio.Event()

    @protocol
    def leaf() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await waiting.wait()
            gate.set()
            final(Decision.reject("leaf"))

        return decide

    async def explodes() -> None:
        waiting.set()
        await gate.wait()
        raise RuntimeError("boom")

    @protocol
    def custom() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_protocols, leaf(), context, step)
                tg.start_soon(explodes)
            return None

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5), pytest.raises(RuntimeError, match="boom"):
        await run_root(
            wrapper(custom()), runner_context(recorder=recorder), before_step()
        )
    assert recorder.records == []
    assert [r.reported.path for r in recorder.supersessions] == ["custom/leaf"]
    assert recorder.bypassed_layers == []


@pytest.mark.anyio
async def test_an_illegal_own_final_supersedes_the_claimed_finals_beside_it() -> None:
    waiting = anyio.Event()
    gate = anyio.Event()

    @protocol
    def leaf() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await waiting.wait()
            gate.set()
            final(Decision.reject("leaf"))

        return decide

    async def illegal() -> None:
        waiting.set()
        await gate.wait()
        final(Decision(action="modify"))

    @protocol
    def custom() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_protocols, leaf(), context, step)
                tg.start_soon(illegal)
            return None

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5), pytest.raises(ValueError, match="'custom'"):
        await run_root(
            wrapper(custom()), runner_context(recorder=recorder), before_step()
        )
    assert recorder.records == []
    assert [r.reported.path for r in recorder.supersessions] == ["custom/leaf"]


@pytest.mark.anyio
async def test_a_monitors_own_final_supersedes_the_claimed_finals_beside_it() -> None:
    waiting = anyio.Event()
    gate = anyio.Event()

    @protocol
    def leaf() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await waiting.wait()
            gate.set()
            final(Decision.reject("leaf"))

        return decide

    async def own() -> None:
        waiting.set()
        await gate.wait()
        final(Decision.reject("own"))

    @monitor
    def meddles() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_protocols, leaf(), context, step)
                tg.start_soon(own)
            return None

        return check

    recorder = ListRecorder()
    with anyio.fail_after(5), pytest.raises(TypeError, match="meddles"):
        await run_root(
            concurrent([meddles()]), runner_context(recorder=recorder), before_step()
        )
    assert recorder.records == []
    assert [r.reported.path for r in recorder.supersessions] == ["meddles/leaf"]


@pytest.mark.parametrize("terminate_first", [True, False])
@pytest.mark.anyio
async def test_a_terminate_outrun_by_a_siblings_final_is_superseded(
    terminate_first: bool,
) -> None:
    children = {"t": decides("terminate"), "f": finalizes("continue")}
    if not terminate_first:
        children = {"f": children["f"], "t": children["t"]}
    recorder = ListRecorder()
    decision = await run_root(
        concurrent(children), runner_context(recorder=recorder), before_step()
    )
    assert decision == Decision(action="continue")
    recorded = sorted(r.reported.path for r in recorder.records)
    superseded = [r.reported.path for r in recorder.supersessions]
    cancelled = [path for path, _ in recorder.cancellations]
    assert (recorded, superseded, cancelled) in (
        (["f", "t"], ["t"], []),
        (["f"], [], ["t"]),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("valid_first", [True, False])
async def test_an_ill_shaped_own_final_supersedes_nothing_it_made_itself(
    valid_first: bool,
) -> None:
    @protocol
    def twice() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            async def valid() -> None:
                final(Decision.terminate())

            async def invalid() -> None:
                final(Decision(action="reject"))

            async with anyio.create_task_group() as tg:
                first, second = (valid, invalid) if valid_first else (invalid, valid)
                tg.start_soon(first)
                tg.start_soon(second)
            return None

        return decide

    recorder = ListRecorder()
    with pytest.raises(ValueError, match="twice"):
        await run_root(twice(), runner_context(recorder=recorder), after_step())
    assert recorder.supersessions == []
    assert recorder.records == []


@pytest.mark.parametrize(
    ("configure", "expected"),
    [
        (lambda: scores(0.3), ["scores"]),
        (lambda: [scores(0.3)], ["scores"]),
        (lambda: {"judge": scores(0.3)}, ["judge"]),
    ],
)
@pytest.mark.anyio
async def test_run_monitors_takes_one_many_or_named(
    configure: Callable[[], Monitor | Monitors], expected: list[str]
) -> None:
    observations = await run_monitors(configure(), runner_context(), before_step())
    assert [o.name for o in observations] == expected


@pytest.mark.parametrize(
    ("configure", "expected"),
    [
        (lambda: decides("reject"), ["decides"]),
        (lambda: [decides("reject")], ["decides"]),
        (lambda: {"rule": decides("reject")}, ["rule"]),
    ],
)
@pytest.mark.anyio
async def test_run_protocols_takes_one_many_or_named(
    configure: Callable[[], ControlProtocol | Protocols], expected: list[str]
) -> None:
    decisions = await run_protocols(configure(), runner_context(), before_step())
    assert [d.name for d in decisions] == expected


@pytest.mark.parametrize(
    ("configure", "expected"),
    [
        (lambda: scores(0.3), (["scores"], list[str]())),
        (lambda: decides("reject"), (list[str](), ["decides"])),
        (lambda: [scores(0.3), decides("reject")], (["scores"], ["decides"])),
        (lambda: {"m": scores(0.3), "p": decides()}, (["m"], ["p"])),
    ],
)
@pytest.mark.anyio
async def test_run_children_takes_one_many_or_named(
    configure: Callable[[], Monitor | ControlProtocol | Children],
    expected: tuple[list[str], list[str]],
) -> None:
    reports = await run_children(configure(), runner_context(), before_step())
    assert (
        [o.name for o in reports.observations],
        [d.name for d in reports.decisions],
    ) == expected


@pytest.mark.anyio
async def test_a_report_names_the_function_that_made_it() -> None:
    recorder = ListRecorder()
    reports = await run_children(
        {"m": scores(), "p": decides()},
        runner_context(recorder=recorder),
        before_step(),
    )
    assert [o.function for o in reports.observations] == ["check"]
    assert [d.function for d in reports.decisions] == ["decide"]
    assert sorted(r.reported.function for r in recorder.records) == ["check", "decide"]


@pytest.mark.anyio
async def test_a_final_decision_names_the_function_that_made_it() -> None:
    recorder = ListRecorder()
    await run_root(
        concurrent([finalizes("reject")]),
        runner_context(recorder=recorder),
        before_step(),
    )
    assert [(r.reported.path, r.reported.function) for r in recorder.records] == [
        ("finalizes", "decide")
    ]


class Seen(StoreModel):
    before: int = 0


@monitor
def paired() -> Sequence[Monitor]:
    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        seen = context.store_as(Seen)
        seen.before += 1
        return Observation.score(seen.before / 10)

    async def after(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(context.store_as(Seen).before / 10)

    return [before, after]


@pytest.mark.anyio
async def test_a_groups_members_report_at_one_path_and_share_its_state() -> None:
    recorder = ListRecorder()
    context = runner_context(recorder=recorder)
    group = paired()
    before = await run_monitors(group, context, before_step())
    after = await run_monitors(group, context, after_step())
    assert [(o.name, o.path, o.function, o.report.suspicion) for o in before] == [
        ("paired", "paired", "before", 0.1)
    ]
    assert [(o.name, o.path, o.function, o.report.suspicion) for o in after] == [
        ("paired", "paired", "after", 0.1)
    ]
    assert [r.reported for r in recorder.records] == [*before, *after]


@pytest.mark.anyio
async def test_two_instances_of_a_group_keep_separate_state() -> None:
    context = runner_context()
    group = {"a": paired(), "b": paired()}
    await run_monitors(group, context, before_step())
    observations = await run_monitors(group, context, before_step())
    assert [(o.path, o.report.suspicion) for o in observations] == [
        ("a", 0.2),
        ("b", 0.2),
    ]


@pytest.mark.anyio
async def test_members_on_one_stage_report_in_factory_order() -> None:
    @monitor
    def twins() -> Sequence[Monitor]:
        async def second(context: Context, step: BeforeToolCall) -> Observation | None:
            await anyio.sleep(0)
            return Observation.score(0.2)

        async def first(context: Context, step: BeforeToolCall) -> Observation | None:
            return Observation.score(0.1)

        return [second, first]

    recorder = ListRecorder()
    reports = await run_children(
        [twins(), decides()], runner_context(recorder=recorder), before_step()
    )
    assert [(o.path, o.function) for o in reports.observations] == [
        ("twins", "second"),
        ("twins", "first"),
    ]
    assert [
        r.reported.function for r in recorder.records if r.reported.name == "twins"
    ] == ["second", "first"]


@pytest.mark.anyio
async def test_a_group_member_calling_final_ends_the_step() -> None:
    ran: list[str] = []

    @protocol
    def gatekeeper() -> Sequence[ControlProtocol]:
        async def screen(context: Context, step: BeforeToolCall) -> Decision | None:
            final(Decision.reject("screened"))

        async def later(context: Context, step: BeforeToolCall) -> Decision | None:
            ran.append("later")
            return Decision.clear()

        return [screen, later]

    recorder = ListRecorder()
    decision = await run_root(
        concurrent([gatekeeper()]), runner_context(recorder=recorder), before_step()
    )
    assert decision == Decision.reject("screened")
    assert ran == []
    assert [(r.reported.path, r.reported.function) for r in recorder.records] == [
        ("gatekeeper", "screen")
    ]
    assert recorder.bypassed_layers == [("", "concurrent")]


@pytest.mark.anyio
async def test_a_cancelled_group_is_recorded_once() -> None:
    started = anyio.Event()

    @monitor
    def quick_then_slow() -> Sequence[Monitor]:
        async def quick(context: Context, step: BeforeToolCall) -> Observation | None:
            return Observation.score(0.1)

        async def slow(context: Context, step: BeforeToolCall) -> Observation | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return [quick, slow]

    @protocol
    def terminates() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            return Decision.terminate()

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        reports = await run_children(
            {"g": quick_then_slow(), "stop": terminates()},
            runner_context(recorder=recorder),
            before_step(),
        )
    assert [o.function for o in reports.observations] == ["quick"]
    assert recorder.cancellations == [("g", "g")]
    assert sorted((r.reported.path, r.reported.function) for r in recorder.records) == [
        ("g", "quick"),
        ("stop", "decide"),
    ]


@pytest.mark.anyio
async def test_the_root_is_one_function() -> None:
    @protocol
    def pair() -> Sequence[ControlProtocol]:
        async def one(context: Context, step: Step) -> Decision | None:
            return None

        async def two(context: Context, step: Step) -> Decision | None:
            return None

        return [one, two]

    with pytest.raises(TypeError, match="root"):
        await run_root(pair(), runner_context(), before_step())


@pytest.mark.anyio
async def test_a_members_terminate_stops_the_group() -> None:
    ran: list[str] = []
    started = anyio.Event()

    @protocol
    def stops() -> Sequence[ControlProtocol]:
        async def stop(context: Context, step: BeforeToolCall) -> Decision | None:
            await started.wait()
            return Decision.terminate()

        async def overrule(context: Context, step: BeforeToolCall) -> Decision | None:
            ran.append("overrule")
            final(Decision.reject("overruled"))

        return [stop, overrule]

    @protocol
    def slow() -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decision = await run_root(
            concurrent({"g": stops(), "slow": slow()}),
            runner_context(recorder=recorder),
            before_step(),
        )
    assert decision == Decision.terminate()
    assert ran == []
    assert [(r.reported.path, r.reported.function) for r in recorder.records] == [
        ("g", "stop"),
        ("", "run"),
    ]
    assert recorder.cancellations == [("slow", "slow")]
    assert recorder.supersessions == []


@pytest.mark.anyio
async def test_a_group_members_error_names_its_function() -> None:
    @monitor
    def bad_second() -> Sequence[Monitor]:
        async def fine(context: Context, step: BeforeToolCall) -> Observation | None:
            return None

        async def wrong(context: Context, step: BeforeToolCall) -> Observation | None:
            return cast(Any, Decision.clear())

        return [fine, wrong]

    @monitor
    def meddles() -> Sequence[Monitor]:
        async def fine(context: Context, step: BeforeToolCall) -> Observation | None:
            return None

        async def ends(context: Context, step: BeforeToolCall) -> Observation | None:
            final(Decision.reject())

        return [fine, ends]

    @protocol
    def ill_shaped() -> Sequence[ControlProtocol]:
        async def fine(context: Context, step: AfterToolCall) -> Decision | None:
            return None

        async def rejects(context: Context, step: AfterToolCall) -> Decision | None:
            return Decision.reject()

        return [fine, rejects]

    with pytest.raises(
        TypeError,
        match=r"monitor 'bad_second' \(function 'wrong'\) returned a Decision",
    ):
        await run_monitors(bad_second(), runner_context(), before_step())
    with pytest.raises(
        ValueError, match=r"protocol 'ill_shaped' \(function 'rejects'\)"
    ):
        await run_protocols(ill_shaped(), runner_context(), after_step())
    with pytest.raises(
        TypeError, match=r"monitor 'meddles' \(function 'ends'\) called"
    ):
        await run_monitors(meddles(), runner_context(), before_step())
