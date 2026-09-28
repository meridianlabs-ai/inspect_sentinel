import logging
from collections.abc import Callable
from typing import Any, cast

import pytest
from inspect_ai._util.registry import registry_info

from inspect_sentinel._context import Context
from inspect_sentinel._monitor import ControlProtocol, Monitor, monitor, protocol
from inspect_sentinel._protocols import concurrent, observe, threshold
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._resolve import Sentinels, resolve_sentinel
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


def test_monitors_only_resolve_to_observe() -> None:
    assert registry_info(resolve_sentinel(noisy())).name == "inspect_sentinel/observe"
    assert registry_info(resolve_sentinel([noisy()])).name == "inspect_sentinel/observe"
    assert (
        registry_info(resolve_sentinel({"m": noisy()})).name
        == "inspect_sentinel/observe"
    )


def test_anything_containing_a_protocol_resolves_to_concurrent() -> None:
    assert (
        registry_info(resolve_sentinel([noisy(), blocks()])).name
        == "inspect_sentinel/concurrent"
    )
    assert (
        registry_info(resolve_sentinel({"m": noisy(), "p": blocks()})).name
        == "inspect_sentinel/concurrent"
    )


def test_a_monitor_nothing_acts_on_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._resolve"):
        resolve_sentinel(noisy())
    assert len(caplog.records) == 1
    assert "nothing is configured to act on it" in caplog.text
    assert "noisy" in caplog.text


def test_every_unwatched_monitor_is_named(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._resolve"):
        resolve_sentinel([noisy(), quieter()])
    assert {record.getMessage().split(" ")[0] for record in caplog.records} == {
        "noisy",
        "quieter",
    }


@pytest.mark.parametrize(
    "instance", [blocks(), concurrent([blocks()]), observe([noisy()])]
)
def test_a_lone_protocol_is_wrapped_in_concurrent(instance: ControlProtocol) -> None:
    resolved = resolve_sentinel(instance)
    assert resolved is not instance
    assert registry_info(resolved).name == "inspect_sentinel/concurrent"


def test_observe_written_explicitly_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._resolve"):
        resolve_sentinel(observe([noisy()]))
    assert caplog.records == []


@pytest.mark.parametrize(
    "specs",
    [
        (lambda: blocks(), lambda: [blocks()]),
        (lambda: noisy(), lambda: [noisy()]),
    ],
)
@pytest.mark.anyio
async def test_a_lone_child_and_a_list_of_one_record_the_same_paths(
    specs: tuple[Callable[[], Sentinels], Callable[[], Sentinels]],
) -> None:
    paths: list[list[str]] = []
    for spec in specs:
        recorder = ListRecorder()
        await run_protocol(
            resolve_sentinel(spec()), runner_context(recorder=recorder), before_step()
        )
        paths.append([r.reported.path for r in recorder.records])
    assert paths[0] == paths[1]


def test_a_protocol_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._resolve"):
        resolve_sentinel([noisy(), blocks()])
    assert caplog.records == []


@pytest.mark.anyio
async def test_a_resolved_monitor_records_and_does_not_act() -> None:
    recorder = ListRecorder()
    reported = await run_protocol(
        resolve_sentinel(noisy()), runner_context(recorder=recorder), before_step()
    )
    assert reported is None
    assert [record.reported.name for record in recorder.records] == ["noisy"]


@pytest.mark.anyio
async def test_a_resolved_protocol_decides() -> None:
    reported = await run_protocol(
        resolve_sentinel([blocks()]), runner_context(), before_step()
    )
    assert reported is not None and reported.report.action == "reject"


def test_duplicate_names_are_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_sentinel([noisy(), noisy()])


def test_an_uncalled_factory_is_a_configuration_error() -> None:
    with pytest.raises(TypeError, match="call it"):
        resolve_sentinel(cast(Any, noisy))


@pytest.mark.parametrize("empty", [[], {}])
def test_an_empty_configuration_is_an_error(empty: Any) -> None:
    with pytest.raises(ValueError, match="at least one"):
        resolve_sentinel(empty)


def test_a_protocol_handed_to_threshold_fails_under_resolve() -> None:
    with pytest.raises(TypeError, match="monitor"):
        resolve_sentinel(threshold(cast(Any, [blocks()]), reject_at=0.5))


def test_duplicate_names_nested_in_a_protocol_fail_under_resolve() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_sentinel([concurrent([noisy(), noisy()])])


@pytest.mark.parametrize("spec", [{noisy()}, "noisy"])
def test_a_spec_that_is_neither_a_mapping_nor_a_sequence_is_an_error(spec: Any) -> None:
    with pytest.raises(TypeError, match="Mapping or a Sequence"):
        resolve_sentinel(spec)


def test_each_configured_instance_is_named_in_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="inspect_sentinel._resolve"):
        resolve_sentinel({"first": noisy(), "second": noisy()})
    assert {record.getMessage().split(" ")[0] for record in caplog.records} == {
        "first",
        "second",
    }
