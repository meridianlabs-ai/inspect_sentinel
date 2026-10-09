import logging
from collections.abc import Callable

import anyio
import pytest
from inspect_ai.core import LimitExceededError
from inspect_ai.core._registry import registry_info

from inspect_sentinel import (
    Failed,
    MonitorFailedError,
    Observations,
    _results,
    concurrent,
    observe_only,
    sequential,
    threshold,
)
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._report import Action, Decision, Observation
from inspect_sentinel._runner import run_monitors, run_sentinel
from inspect_sentinel._step import BeforeToolCall, Step
from inspect_sentinel._types import Monitor, MonitorGroup, Protocol
from tests._fakes import ListRecorder, before_step, host_context, in_step

_ERRORS: dict[str, type[Exception]] = {"value": ValueError, "os": OSError}


@monitor
def fails(error: str = "value") -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        raise _ERRORS[error]("model unavailable")

    return check


@monitor
def scores(value: float = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value)

    return check


@protocol
def says(action: Action = "continue") -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action=action)

    return decide


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    # the warning is once per process, and each test runs once per backend
    monkeypatch.setattr(_results, "_warned", set[tuple[str, type[Exception]]]())


def _warnings(caplog: pytest.LogCaptureFixture, path: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and f"Monitor {path!r}" in r.getMessage()
    ]


@pytest.mark.anyio
async def test_a_raising_monitor_is_recorded_as_failed_and_its_siblings_report() -> (
    None
):
    recorder = ListRecorder()
    with in_step(path="layer", recorder=recorder) as context:
        observations = await run_monitors(
            {"broken": fails(), "ok": scores(0.3)},
            context,
            before_step(),
        )
    [failed] = observations.failed
    assert (failed.name, failed.path, failed.function) == (
        "broken",
        "layer/broken",
        "check",
    )
    assert isinstance(failed.error, ValueError)
    assert recorder.failures == [failed]
    assert [(c.path, c.factory) for c in recorder.failed_instances] == [
        ("layer/broken", registry_info(fails).name)
    ]
    assert [r.reported.name for r in recorder.records] == ["ok"]
    assert [o.name for o in observations.succeeded] == ["ok"]


def _first(observations: Observations) -> object:
    return observations[0]


def _peak(observations: Observations) -> object:
    return observations.max_suspicion()


def _contains(observations: Observations) -> object:
    return None in observations


@pytest.mark.parametrize("read", [len, bool, list, _first, _peak, _contains])
def test_reading_observations_with_a_failure_raises(
    read: Callable[[Observations], object],
) -> None:
    error = ValueError("model unavailable")
    observations = Observations(
        failed=[Failed(name="m", path="a/m", function="check", error=error)]
    )
    with pytest.raises(MonitorFailedError, match="'a/m'.*model unavailable") as info:
        read(observations)
    assert info.value.__cause__ is error
    assert observations.succeeded.max_suspicion() is None


@pytest.mark.anyio
async def test_a_failing_function_of_a_group_does_not_stop_the_others() -> None:
    @monitor
    def half_failing() -> MonitorGroup:
        async def first(context: Context, step: BeforeToolCall) -> Observation | None:
            raise ValueError("first failed")

        async def second(context: Context, step: BeforeToolCall) -> Observation | None:
            return Observation.score(0.4)

        return MonitorGroup(first, second)

    with in_step() as context:
        observations = await run_monitors([half_failing()], context, before_step())
    assert [f.function for f in observations.failed] == ["first"]
    assert [o.function for o in observations.succeeded] == ["second"]


@pytest.mark.anyio
async def test_a_cancelled_monitor_is_not_recorded_as_failed() -> None:
    started = anyio.Event()

    @monitor
    def slow_monitor() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            started.set()
            await anyio.sleep_forever()
            return None

        return check

    @protocol
    def stops() -> Protocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            return Decision.terminate()

        return decide

    recorder = ListRecorder()
    with anyio.fail_after(5):
        decision = await run_sentinel(
            concurrent({"slow": slow_monitor(), "stops": stops()}),
            host_context(recorder=recorder),
            before_step(),
        )
    assert decision is not None and decision.action == "terminate"
    assert recorder.cancellations == [("slow", "slow")]
    assert recorder.failures == []


@pytest.mark.anyio
async def test_a_limit_reached_in_a_monitor_ends_the_step() -> None:
    @monitor(portable=False)
    def over_budget() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            raise LimitExceededError("cost", value=2.0, limit=1.0)

        return check

    recorder = ListRecorder()
    with pytest.raises(LimitExceededError):
        with in_step(recorder=recorder) as context:
            await run_monitors([over_budget()], context, before_step())
    assert recorder.failures == []


@pytest.mark.anyio
async def test_a_monitor_that_turns_its_cancellation_into_an_error_is_cancelled() -> (
    None
):
    entered = anyio.Event()

    @monitor
    def monitor_converts_cancellation() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            entered.set()
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                raise ValueError("connection closed") from None
            return None

        return check

    recorder = ListRecorder()
    returned: list[Observations] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def run() -> None:
                with in_step(recorder=recorder) as context:
                    returned.append(
                        await run_monitors(
                            {"c": monitor_converts_cancellation()},
                            context,
                            before_step(),
                        )
                    )

            tg.start_soon(run)
            await entered.wait()
            tg.cancel_scope.cancel()
    assert returned == []
    assert recorder.failures == []
    assert recorder.cancellations == [("c", "c")]


@pytest.mark.anyio
async def test_a_limit_beside_another_error_in_a_monitors_own_group_ends_the_step() -> (
    None
):
    async def over_budget() -> None:
        raise LimitExceededError("cost", value=2.0, limit=1.0)

    async def broken() -> None:
        raise ValueError("model unavailable")

    @monitor
    def fans_out() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(broken)
                tg.start_soon(over_budget)
            return None

        return check

    recorder = ListRecorder()
    with pytest.raises(LimitExceededError):
        with in_step(recorder=recorder) as context:
            await run_monitors([fans_out()], context, before_step())
    assert recorder.failures == []


def test_a_monitor_failed_error_names_at_least_one_failure() -> None:
    with pytest.raises(ValueError, match="at least one failure"):
        MonitorFailedError([])


@pytest.mark.anyio
async def test_observe_only_records_a_failure_warns_once_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = ListRecorder()
    root = observe_only({"observe_flaky": fails(), "steady": scores(0.2)})
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            decision = await run_sentinel(
                root, host_context(recorder=recorder), before_step()
            )
            assert decision is None
    assert [f.path for f in recorder.failures] == ["observe_flaky", "observe_flaky"]
    assert [r.reported.name for r in recorder.records] == ["steady", "steady"]
    [warning] = _warnings(caplog, "observe_flaky")
    assert "ValueError: model unavailable" in warning


@pytest.mark.anyio
async def test_the_warning_is_once_per_instance_and_exception_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        for error in ("value", "os", "value"):
            for name in ("dedupe_a", "dedupe_b"):
                await run_sentinel(
                    observe_only({name: fails(error)}), host_context(), before_step()
                )
    for name in ("dedupe_a", "dedupe_b"):
        warnings = _warnings(caplog, name)
        assert [w.split(" failed with ")[1].split(":")[0] for w in warnings] == [
            "ValueError",
            "OSError",
        ]


@pytest.mark.anyio
async def test_a_failed_monitor_under_threshold_fails_the_step() -> None:
    recorder = ListRecorder()
    with pytest.raises(MonitorFailedError, match="'broken'.*model unavailable"):
        await run_sentinel(
            threshold({"broken": fails(), "ok": scores(0.1)}, reject_at=0.5),
            host_context(recorder=recorder),
            before_step(),
        )
    assert [f.path for f in recorder.failures] == ["broken"]
    assert [r.reported.path for r in recorder.records] == ["ok"]


@pytest.mark.anyio
async def test_concurrent_records_a_failed_monitor_and_still_votes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = ListRecorder()
    with caplog.at_level(logging.WARNING):
        decision = await run_sentinel(
            concurrent({"concurrent_flaky": fails(), "rule": says("reject")}),
            host_context(recorder=recorder),
            before_step(),
        )
    assert decision is not None and decision.action == "reject"
    assert [f.path for f in recorder.failures] == ["concurrent_flaky"]
    assert len(_warnings(caplog, "concurrent_flaky")) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("links", "expected"),
    [
        ({"sequential_flaky": fails(), "rule": says("reject")}, "reject"),
        ({"sequential_alone": fails()}, "continue"),
    ],
)
async def test_sequential_records_a_failed_monitor_and_falls_through(
    links: dict[str, Monitor | Protocol],
    expected: Action,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = ListRecorder()
    with caplog.at_level(logging.WARNING):
        decision = await run_sentinel(
            sequential(links), host_context(recorder=recorder), before_step()
        )
    assert decision is not None and decision.action == expected
    assert len(recorder.failures) == 1
    assert len(_warnings(caplog, recorder.failures[0].path)) == 1


@protocol
def tolerant(monitors: dict[str, Monitor], reject_at: float) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        observations = await run_monitors(monitors, context, step)
        if observations.failed and not observations.succeeded:
            return Decision.escalate("every monitor failed")
        peak = observations.succeeded.max_suspicion()
        if peak is not None and peak >= reject_at:
            return Decision.reject(f"suspicion {peak}")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("monitors", "expected"),
    [
        ({"broken": fails(), "high": scores(0.9)}, "reject"),
        ({"broken": fails(), "low": scores(0.1)}, "continue"),
        ({"broken": fails()}, "escalate"),
    ],
)
async def test_a_protocol_can_decide_from_the_monitors_that_did_not_fail(
    monitors: dict[str, Monitor], expected: Action
) -> None:
    recorder = ListRecorder()
    decision = await run_sentinel(
        tolerant(monitors, reject_at=0.5),
        host_context(recorder=recorder),
        before_step(),
    )
    assert decision is not None and decision.action == expected
    assert [f.name for f in recorder.failures] == ["broken"]
