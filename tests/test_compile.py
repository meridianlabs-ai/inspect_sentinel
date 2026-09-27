import logging
from typing import Any, cast

import pytest
from inspect_ai._util.registry import registry_info

from inspect_sentinel._compile import compile_sentinel
from inspect_sentinel._context import Context
from inspect_sentinel._monitor import ControlProtocol, Monitor, monitor, protocol
from inspect_sentinel._protocols import concurrent, observe, threshold
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._runner import run_protocol
from inspect_sentinel._step import BeforeToolCall, Step
from tests._fakes import ListRecorder, before_step, runner_context


@monitor
def noisy() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.4)

    return check


@monitor
def quieter() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    return check


@protocol
def blocks() -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision.reject("no")

    return decide


def test_monitors_only_compile_to_observe() -> None:
    assert registry_info(compile_sentinel(noisy())).name == "inspect_sentinel/observe"
    assert registry_info(compile_sentinel([noisy()])).name == "inspect_sentinel/observe"
    assert (
        registry_info(compile_sentinel({"m": noisy()})).name
        == "inspect_sentinel/observe"
    )


def test_anything_containing_a_protocol_compiles_to_concurrent() -> None:
    assert (
        registry_info(compile_sentinel(blocks())).name == "inspect_sentinel/concurrent"
    )
    assert (
        registry_info(compile_sentinel([noisy(), blocks()])).name
        == "inspect_sentinel/concurrent"
    )
    assert (
        registry_info(compile_sentinel({"m": noisy(), "p": blocks()})).name
        == "inspect_sentinel/concurrent"
    )


def test_a_monitor_nothing_acts_on_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._compile"):
        compile_sentinel(noisy())
    assert len(caplog.records) == 1
    assert "nothing is configured to act on it" in caplog.text
    assert "noisy" in caplog.text


def test_every_unwatched_monitor_is_named(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._compile"):
        compile_sentinel([noisy(), quieter()])
    assert {record.getMessage().split(" ")[0] for record in caplog.records} == {
        "noisy",
        "quieter",
    }


def test_observe_written_explicitly_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._compile"):
        compile_sentinel(observe([noisy()]))
    assert caplog.records == []


def test_a_protocol_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._compile"):
        compile_sentinel([noisy(), blocks()])
    assert caplog.records == []


@pytest.mark.anyio
async def test_a_compiled_monitor_records_and_does_not_act() -> None:
    recorder = ListRecorder()
    reported = await run_protocol(
        compile_sentinel(noisy()), runner_context(recorder=recorder), before_step()
    )
    assert reported is None
    assert [record.reported.name for record in recorder.records] == ["noisy"]


@pytest.mark.anyio
async def test_a_compiled_protocol_decides() -> None:
    reported = await run_protocol(
        compile_sentinel([blocks()]), runner_context(), before_step()
    )
    assert reported is not None and reported.report.action == "reject"


def test_duplicate_names_are_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        compile_sentinel([noisy(), noisy()])


def test_an_uncalled_factory_is_a_configuration_error() -> None:
    with pytest.raises(TypeError, match="call it"):
        compile_sentinel(cast(Any, noisy))


@pytest.mark.parametrize("empty", [[], {}])
def test_an_empty_configuration_is_an_error(empty: Any) -> None:
    with pytest.raises(ValueError, match="at least one"):
        compile_sentinel(empty)


def test_a_protocol_handed_to_threshold_fails_under_compile() -> None:
    with pytest.raises(TypeError, match="monitor"):
        compile_sentinel(threshold(cast(Any, [blocks()]), reject_at=0.5))


def test_duplicate_names_nested_in_a_protocol_fail_under_compile() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        compile_sentinel([concurrent([noisy(), noisy()])])
