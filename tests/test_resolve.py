from collections.abc import Callable
from typing import Any, cast

import pytest
from inspect_ai.core._registry import registry_info

from inspect_sentinel import Sentinels
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import (
    monitor,
    protocol,
)
from inspect_sentinel._protocols import concurrent, observe_only, threshold
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._resolve import resolve_sentinel
from inspect_sentinel._runner import (
    run_protocols,
    run_sentinel,
)
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import (
    Monitor,
    MonitorGroup,
    Protocol,
    ProtocolGroup,
)
from tests._fakes import ListRecorder, before_step, host_context, in_step


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


@monitor
def watches_both() -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.2)

    async def after(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.3)

    return MonitorGroup(before, after)


@protocol
def blocks() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision.reject("no")

    return decide


@protocol
def pair() -> ProtocolGroup:
    async def one(context: Context, step: Step) -> Decision | None:
        return Decision.proceed()

    async def two(context: Context, step: Step) -> Decision | None:
        return Decision.reject("no")

    return ProtocolGroup(one, two)


MONITORS_ONLY = (
    "A sentinel needs a protocol to decide each step, but it was given only monitors"
)


@pytest.mark.parametrize(
    ("spec", "names"),
    [
        (lambda: noisy(), "noisy"),
        (lambda: [noisy(), quieter()], "noisy, quieter"),
        (lambda: {"first": noisy(), "second": quieter()}, "first, second"),
        (lambda: watches_both(), "watches_both"),
    ],
)
def test_monitors_alone_are_a_configuration_error_naming_observe_only(
    spec: Callable[[], Sentinels], names: str
) -> None:
    with pytest.raises(ValueError) as raised:
        resolve_sentinel(spec())
    message = str(raised.value)
    assert message.startswith(f"{MONITORS_ONLY}: {names}.")
    assert "observe_only()" in message and "threshold()" in message


def test_anything_containing_a_protocol_resolves_to_concurrent() -> None:
    assert (
        registry_info(resolve_sentinel([noisy(), blocks()])).name
        == "inspect_sentinel/concurrent"
    )
    assert (
        registry_info(resolve_sentinel({"m": noisy(), "p": blocks()})).name
        == "inspect_sentinel/concurrent"
    )


@pytest.mark.parametrize(
    "instance", [blocks(), concurrent([blocks()]), observe_only([noisy()])]
)
def test_a_lone_protocol_is_the_root(instance: Protocol) -> None:
    assert resolve_sentinel(instance) is instance


def test_a_lone_protocol_group_is_wrapped_in_concurrent() -> None:
    resolved = resolve_sentinel(pair())
    assert registry_info(resolved).name == "inspect_sentinel/concurrent"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            lambda: threshold(noisy(), reject_at=0.5),
            [("noisy", "noisy"), ("threshold", "")],
        ),
        (lambda: [blocks()], [("blocks", "blocks"), ("concurrent", "")]),
        (lambda: observe_only(noisy()), [("noisy", "noisy")]),
        (
            lambda: [noisy(), blocks()],
            [("noisy", "noisy"), ("blocks", "blocks"), ("concurrent", "")],
        ),
    ],
)
@pytest.mark.anyio
async def test_the_root_records_at_the_empty_path_and_its_children_bare(
    spec: Callable[[], Sentinels], expected: list[tuple[str, str]]
) -> None:
    recorder = ListRecorder()
    await run_sentinel(
        resolve_sentinel(spec()), host_context(recorder=recorder), before_step()
    )
    recorded = [(r.reported.name, r.reported.path) for r in recorder.records]
    assert sorted(recorded[:-1]) == sorted(expected[:-1])
    assert recorded[-1] == expected[-1]


@pytest.mark.anyio
async def test_a_resolved_protocol_decides() -> None:
    with in_step() as context:
        [reported] = await run_protocols(
            resolve_sentinel([blocks()]), context, before_step()
        )
    assert reported.report.action == "reject"


def test_duplicate_names_are_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_sentinel([noisy(), noisy()])


@pytest.mark.parametrize("factory", [noisy, blocks])
def test_an_uncalled_factory_is_a_configuration_error(factory: Any) -> None:
    with pytest.raises(TypeError, match="call it"):
        resolve_sentinel(factory)


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
