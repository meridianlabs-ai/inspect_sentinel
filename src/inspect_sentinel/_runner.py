from __future__ import annotations

import logging
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
from inspect_ai._util.registry import registry_info, registry_unqualified_name

from ._context import Context, RunnerContext, check_instance_name
from ._monitor import (
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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _task_group() -> AsyncGenerator[TaskGroup]:
    try:
        async with anyio.create_task_group() as tg:
            yield tg
    # Only the first failure surfaces, as in inspect_ai's tg_collect; a second
    # concurrent failure and any nested group are not flattened or chained. The
    # group is suppressed from the traceback without touching the child's own
    # __cause__.
    except ExceptionGroup as ex:
        first = ex.exceptions[0]
        first.__suppress_context__ = True
        raise first


async def run_monitor(
    monitor: Monitor, context: Context, step: Step, *, name: str | None = None
) -> Reported[Observation] | None:
    """Invoke one monitor if it accepts this step's payload.

    Derives the child's context under this layer's path, records the observation, and returns it. Returns `None` if the monitor abstained or does not watch this stage.

    Args:
        monitor: The monitor instance.
        context: This layer's context, as the dispatcher provided it.
        step: The step being examined.
        name: The child's instance name; the registry name without its package prefix when omitted.
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
        name: The child's instance name; the registry name without its package prefix when omitted.
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
    return (await _run_named(named, context, step)).observations


async def run_protocols(
    protocols: Protocols, context: Context, step: Step
) -> Decisions:
    """Run protocols concurrently and collect their decisions in configuration order, cancelling the rest at their next await when one returns `terminate`.

    Args:
        protocols: A sequence of protocols, or a mapping of instance names to protocols.
        context: This layer's context.
        step: The step being examined.
    """
    named = _named(protocols, "protocol")
    return (await _run_named(named, context, step)).decisions


async def run_children(children: Children, context: Context, step: Step) -> Reports:
    """Run monitors and protocols together in one task group, cancelling the rest at their next await when a protocol returns `terminate`.

    A child cancelled this way is recorded through `Recorder.cancelled`; one that finishes without awaiting is recorded normally.

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
    accepted = step_types(child)
    info = registry_info(child)
    if info.type != kind:
        raise TypeError(f"Expected a {kind}, got the {info.type} {info.name!r}.")
    if not isinstance(step, tuple(accepted)):
        return None
    child_name = name if name is not None else registry_unqualified_name(info)
    child_context = context.child(child_name)
    invoke = cast(Callable[[Context, Step], Awaitable[Report | None]], child)
    try:
        report = await invoke(child_context, step)
    except anyio.get_cancelled_exc_class():
        # bookkeeping must not replace the cancellation, or a terminate is lost
        try:
            child_context.recorder.cancelled(child_context, step, child_name)
        except Exception:
            logger.exception(
                "Recorder failed while recording the cancellation of %r",
                child_context.path,
            )
        raise
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
    children: Mapping[str, Monitor | ControlProtocol]
    | Iterable[Monitor | ControlProtocol],
    expected: Literal["monitor", "protocol"] | None,
) -> list[tuple[str, Monitor | ControlProtocol]]:
    pairs: list[tuple[str | None, Monitor | ControlProtocol]]
    if isinstance(children, Mapping):
        mapping = cast(Mapping[str, Monitor | ControlProtocol], children)
        pairs = [(key, child) for key, child in mapping.items()]
    elif isinstance(children, Sequence):
        pairs = [(None, child) for child in children]
    else:
        raise TypeError(
            "children must be a Mapping or a Sequence; a set or an iterator has no configuration order"
        )
    named: list[tuple[str, Monitor | ControlProtocol]] = []
    seen: set[str] = set()
    for given, child in pairs:
        step_types(child)  # rejects an uncalled factory or an undecorated function
        info = registry_info(child)
        if expected is not None and info.type != expected:
            raise TypeError(f"Expected a {expected}, got the {info.type} {info.name!r}.")
        name = check_instance_name(
            given if given is not None else registry_unqualified_name(info)
        )
        if name in seen:
            raise ValueError(
                f"Duplicate instance name {name!r} in one layer. Give the children distinct names with a mapping."
            )
        seen.add(name)
        named.append((name, child))
    return named
