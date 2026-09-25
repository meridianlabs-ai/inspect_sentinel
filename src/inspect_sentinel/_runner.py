from __future__ import annotations

import sys
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast, overload

import anyio
from anyio.abc import TaskGroup
from inspect_ai._util.registry import (
    is_registry_object,
    registry_info,
    registry_unqualified_name,
)

from ._context import Context, RunnerContext
from ._monitor import (
    STEP_TYPES_ATTR,
    Children,
    ControlProtocol,
    Monitor,
    Monitors,
    Protocols,
    step_types,
)
from ._report import Action, Decision, Observation, R_co, Report, Reported
from ._step import Step

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

_PRECEDENCE: dict[Action, int] = {
    "terminate": 4,
    "reject": 3,
    "modify": 2,
    "continue": 1,
}


class _ReportSequence(Sequence[Reported[R_co]], Generic[R_co]):
    def __init__(self, items: Iterable[Reported[R_co]] = ()) -> None:
        self._items = tuple(items)

    @overload
    def __getitem__(self, index: int) -> Reported[R_co]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Reported[R_co]]: ...
    def __getitem__(
        self, index: int | slice
    ) -> Reported[R_co] | Sequence[Reported[R_co]]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Reported[R_co]]:
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._items == cast("_ReportSequence[R_co]", other)._items

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._items)!r})"


class Observations(_ReportSequence[Observation]):
    """What a layer's monitors reported, in configuration order."""

    def max_suspicion(self) -> float | None:
        """The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed."""
        scores = [
            max(o.report.suspicion.values())
            if isinstance(o.report.suspicion, dict)
            else o.report.suspicion
            for o in self._items
        ]
        return max(scores) if scores else None


class Decisions(_ReportSequence[Decision]):
    """What a layer's protocols decided, in configuration order."""

    def strongest(self) -> Reported[Decision] | None:
        """The strongest decision by `terminate > reject > modify > continue`; `escalate` does not count; `None` if nobody decided."""
        ranked = [d for d in self._items if d.report.action in _PRECEDENCE]
        if not ranked:
            return None
        return max(ranked, key=lambda d: _PRECEDENCE[d.report.action])


@dataclass(frozen=True)
class Reports:
    """Both families of report from one layer."""

    observations: Observations
    """From the layer's monitors."""

    decisions: Decisions
    """From the layer's protocols."""


R = TypeVar("R", bound=Report)
C = TypeVar("C")


@asynccontextmanager
async def _task_group() -> AsyncGenerator[TaskGroup]:
    try:
        async with anyio.create_task_group() as tg:
            yield tg
    except ExceptionGroup as ex:
        raise ex.exceptions[0] from None


async def run_monitor(
    monitor: Monitor, context: Context, step: Step, *, name: str | None = None
) -> Reported[Observation] | None:
    """Invoke one monitor if it accepts this step's payload.

    Derives the child's context under this layer's path, records the observation, and returns it. Returns `None` if the monitor abstained or does not watch this stage.

    Args:
        monitor: The monitor instance.
        context: This layer's context, as the dispatcher provided it.
        step: The step being examined.
        name: The child's instance name; the registry name when omitted.
    """
    return await _run_child(monitor, "monitor", Observation, context, step, name)


async def run_protocol(
    protocol: ControlProtocol, context: Context, step: Step, *, name: str | None = None
) -> Reported[Decision] | None:
    """The same as `run_monitor`, for one protocol.

    Args:
        protocol: The protocol instance.
        context: This layer's context, as the dispatcher provided it.
        step: The step being examined.
        name: The child's instance name; the registry name when omitted.
    """
    return await _run_child(protocol, "protocol", Decision, context, step, name)


async def run_monitors(
    monitors: Monitors, context: Context, step: Step
) -> Observations:
    """Run monitors concurrently and collect their observations in configuration order.

    Args:
        monitors: A sequence of monitors, or a mapping of instance names to monitors.
        context: This layer's context.
        step: The step being examined.
    """
    named = _named(monitors, "monitor")
    observed: list[tuple[int, Reported[Observation]]] = []

    async def run_one(index: int, name: str, child: Monitor) -> None:
        result = await run_monitor(child, context, step, name=name)
        if result is not None:
            observed.append((index, result))

    async with _task_group() as tg:
        for index, (name, child) in enumerate(named):
            tg.start_soon(run_one, index, name, child)

    return Observations(o for _, o in sorted(observed, key=lambda t: t[0]))


async def run_protocols(
    protocols: Protocols, context: Context, step: Step
) -> Decisions:
    """Run protocols concurrently and collect their decisions in configuration order, cancelling the rest when one returns `terminate`.

    Args:
        protocols: A sequence of protocols, or a mapping of instance names to protocols.
        context: This layer's context.
        step: The step being examined.
    """
    named = _named(protocols, "protocol")
    return (await _run_named(named, context, step)).decisions


async def run_children(children: Children, context: Context, step: Step) -> Reports:
    """Run monitors and protocols together in one task group, cancelling the rest when a protocol returns `terminate`.

    Args:
        children: A sequence of monitors and protocols, or a mapping of instance names to them.
        context: This layer's context.
        step: The step being examined.
    """
    named = _named(children, None)
    return await _run_named(named, context, step)


async def _run_named(
    named: Sequence[tuple[str, Monitor | ControlProtocol]], context: Context, step: Step
) -> Reports:
    observations: list[tuple[int, Reported[Observation]]] = []
    decisions: list[tuple[int, Reported[Decision]]] = []

    async def run_one(
        index: int,
        name: str,
        child: Monitor | ControlProtocol,
        cancel: Callable[[], None],
    ) -> None:
        if registry_info(child).type == "monitor":
            observed = await run_monitor(cast(Monitor, child), context, step, name=name)
            if observed is not None:
                observations.append((index, observed))
        else:
            decided = await run_protocol(
                cast(ControlProtocol, child), context, step, name=name
            )
            if decided is not None:
                decisions.append((index, decided))
                if decided.report.action == "terminate":
                    cancel()

    async with _task_group() as tg:
        for index, (name, child) in enumerate(named):
            tg.start_soon(run_one, index, name, child, tg.cancel_scope.cancel)

    return Reports(
        Observations(o for _, o in sorted(observations, key=lambda t: t[0])),
        Decisions(d for _, d in sorted(decisions, key=lambda t: t[0])),
    )


async def _run_child(
    child: Monitor | ControlProtocol,
    kind: Literal["monitor", "protocol"],
    report_type: type[R],
    context: Context,
    step: Step,
    name: str | None,
) -> Reported[R] | None:
    if not isinstance(context, RunnerContext):
        raise TypeError(
            "The runner needs the RunnerContext the dispatcher provided; a Context constructed elsewhere cannot record reports."
        )
    info = registry_info(child)
    if info.type != kind:
        raise TypeError(f"Expected a {kind}, got the {info.type} {info.name!r}.")
    if not isinstance(step, tuple(step_types(child))):
        return None
    child_name = name if name is not None else registry_unqualified_name(info)
    child_context = context.child(child_name)
    invoke = cast(Callable[[Context, Step], Awaitable[Report | None]], child)
    report = await invoke(child_context, step)
    if report is None:
        return None
    if not isinstance(report, report_type):
        raise TypeError(
            f"{kind} {child_name!r} returned a {type(report).__name__}; a {kind} must return {report_type.__name__} or None."
        )
    reported = Reported(name=child_name, path=child_context.path, report=report)
    child_context.recorder.record(child_context, step, reported)
    return reported


def _named(
    children: Mapping[str, C] | Sequence[C],
    expected: Literal["monitor", "protocol"] | None,
) -> list[tuple[str, C]]:
    if isinstance(children, Mapping):
        named = list(children.items())
    elif isinstance(children, Iterator):
        raise TypeError("children must be a Mapping or a Sequence, not an iterator")
    else:
        named = [(registry_unqualified_name(registry_info(c)), c) for c in children]
    seen: set[str] = set()
    for name, child in named:
        if name == "" or "/" in name:
            raise ValueError(
                f"Instance name {name!r} must be non-empty and must not contain '/'."
            )
        if not hasattr(child, STEP_TYPES_ATTR):
            if is_registry_object(child):
                raise TypeError(
                    f"{registry_info(child).name!r} is the factory, not a configured instance; call it to configure one."
                )
            raise TypeError(f"{child!r} is not a configured monitor or protocol.")
        if expected is not None and registry_info(child).type != expected:
            raise TypeError(
                f"Expected a {expected}, got the {registry_info(child).type} {registry_info(child).name!r}."
            )
        if name in seen:
            raise ValueError(
                f"Duplicate instance name {name!r} in one layer. Give the children distinct names with a mapping."
            )
        seen.add(name)
    return named
