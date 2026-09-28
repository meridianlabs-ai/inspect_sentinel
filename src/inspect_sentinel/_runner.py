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
from typing import Any, Generic, Literal, NamedTuple, TypeVar, cast, overload

import anyio
from anyio.abc import TaskGroup
from inspect_ai._util.registry import (
    RegistryInfo,
    registry_info,
    registry_unqualified_name,
)

from ._check import validate_decision_shape
from ._context import Context, RunnerContext, validate_instance_name
from ._final import Final, Origin
from ._monitor import (
    Children,
    ControlProtocol,
    Monitor,
    Monitors,
    Protocols,
    step_types,
)
from ._report import (
    PRECEDENCE,
    Decision,
    Observation,
    R_co,
    Report,
    Reported,
    deciding,
)
from ._step import Step

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


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
        """The strongest decision by `terminate > reject > modify > continue`, the first in configuration order on a tie; `escalate` does not count; `None` if nobody decided."""
        ranked = deciding(self._items)
        if not ranked:
            return None
        return max(ranked, key=lambda d: PRECEDENCE[d.report.action])


@dataclass(frozen=True)
class Reports:
    """Both families of report from one layer."""

    observations: Observations
    """From the layer's monitors."""

    decisions: Decisions
    """From the layer's protocols."""


R = TypeVar("R", bound=Report)


def _leaves(group: BaseExceptionGroup[BaseException]) -> list[BaseException]:
    # cancellations are dropped; a group of only cancellations has no leaves
    leaves: list[BaseException] = []
    for ex in group.exceptions:
        if isinstance(ex, BaseExceptionGroup):
            leaves.extend(_leaves(cast(BaseExceptionGroup[BaseException], ex)))
        elif not isinstance(ex, anyio.get_cancelled_exc_class()):
            leaves.append(ex)
    return leaves


class _Partition(NamedTuple):
    errors: list[BaseException]
    finals: list[Final]


def _partition(group: BaseExceptionGroup[BaseException]) -> _Partition:
    leaves = _leaves(group)
    return _Partition(
        errors=[leaf for leaf in leaves if not isinstance(leaf, Final)],
        finals=[leaf for leaf in leaves if isinstance(leaf, Final)],
    )


def _supersede(finals: Sequence[Final]) -> None:
    for ex in finals:
        origin = ex.origin
        if origin is None:
            raise RuntimeError(
                f"Runner invariant violated: Final({ex.decision.action!r}) reached a task group without an origin; every Final leaving a child is given one by the runner."
            )
        origin.context.recorder.superseded(origin.context, origin.step, origin.reported)


def _surface(errors: Sequence[BaseException], finals: Sequence[Final]) -> BaseException:
    # The one policy for a group's failures. An error outranks every Final, so a
    # bug is not hidden behind a final decision; the Finals that were decided
    # are superseded, and one without an origin came from the failing child's
    # own body, so it is not a decision to record. Otherwise the first Final to
    # arrive propagates and the rest are superseded.
    if errors:
        _supersede([ex for ex in finals if ex.origin is not None])
        return errors[0]
    _supersede(finals[1:])
    return finals[0]


@asynccontextmanager
async def _task_group() -> AsyncGenerator[TaskGroup]:
    # Nested groups are flattened to their leaves, so a Final raised inside a
    # protocol's own task group is still seen. Only the first failure surfaces,
    # as in inspect_ai's tg_collect; it is re-raised outside the handler so its
    # own __cause__ and __context__ survive and the group does not appear in
    # the traceback. On trio a child error can arrive alongside the
    # cancellation in one BaseExceptionGroup; the error is surfaced, as anyio's
    # asyncio backend already does, and a group holding only cancellations
    # propagates untouched.
    first: BaseException | None = None
    try:
        async with anyio.create_task_group() as tg:
            yield tg
    except BaseExceptionGroup as ex:
        errors, finals = _partition(ex)
        if not errors and not finals:
            raise
        first = _surface(errors, finals)
    if first is not None:
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


async def run_root(
    protocol: ControlProtocol, context: Context, step: Step
) -> Decision | None:
    """Invoke the resolved root protocol for one step and return the step's outcome.

    The root is recorded at the empty path under its registry name without the package prefix, so its children's paths are bare. Its decision is shape-checked and recorded like any layer's; a `final()` from below records the root as bypassed, and its decision is recorded here, the one time it is recorded, and returned, so the caller need not catch `Final`.

    Args:
        protocol: The root protocol, as `resolve_sentinel` returned it.
        context: The top layer's context, whose `path` is empty.
        step: The step being examined.
    """
    try:
        reported = await _run_child(
            protocol, "protocol", Decision, context, step, None, root=True
        )
    except Final as ex:
        origin = ex.origin
        if origin is None:
            raise RuntimeError(
                f"Runner invariant violated: Final({ex.decision.action!r}) left the root without an origin."
            ) from ex
        origin.context.recorder.record(origin.context, origin.step, origin.reported)
        return ex.decision
    return reported.report if reported is not None else None


async def run_monitors(
    monitors: Monitors, context: Context, step: Step
) -> Observations:
    """Run monitors concurrently and collect their observations in configuration order.

    Args:
        monitors: A sequence of monitors, or a mapping of instance names to monitors.
        context: This layer's context.
        step: The step being examined.
    """
    named = named_children(monitors, "monitor")
    return (await _run_named(named, context, step)).observations


async def run_protocols(
    protocols: Protocols, context: Context, step: Step
) -> Decisions:
    """Run protocols concurrently and collect their decisions in configuration order, cancelling the rest at their next await when one returns `terminate` or calls `final()`.

    A `final()` from any protocol at any depth below propagates out of this call, so the caller's own decision logic does not run.

    Args:
        protocols: A sequence of protocols, or a mapping of instance names to protocols.
        context: This layer's context.
        step: The step being examined.
    """
    named = named_children(protocols, "protocol")
    return (await _run_named(named, context, step)).decisions


async def run_children(children: Children, context: Context, step: Step) -> Reports:
    """Run monitors and protocols together in one task group, cancelling the rest at their next await when a protocol returns `terminate` or calls `final()`.

    A child cancelled this way is recorded through `Recorder.cancelled`; one that finishes without awaiting is recorded normally.

    Args:
        children: A sequence of monitors and protocols, or a mapping of instance names to them.
        context: This layer's context.
        step: The step being examined.
    """
    named = named_children(children, None)
    return await _run_named(named, context, step)


async def _run_named(
    named: Sequence[tuple[str, Monitor | ControlProtocol]], context: Context, step: Step
) -> Reports:
    observations: list[tuple[int, Reported[Observation]]] = []
    decisions: list[tuple[int, Reported[Decision]]] = []
    terminated: list[Reported[Decision]] = []

    async def run_one(
        index: int,
        name: str,
        child: Monitor | ControlProtocol,
        cancel: Callable[[], None],
    ) -> None:
        if registry_info(child).type == "monitor":
            observed = await _run_child(
                cast(Monitor, child), "monitor", Observation, context, step, name
            )
            if observed is not None:
                observations.append((index, observed))
        else:
            decided = await _run_child(
                cast(ControlProtocol, child),
                "protocol",
                Decision,
                context,
                step,
                name,
            )
            if decided is not None:
                decisions.append((index, decided))
                if decided.report.action == "terminate":
                    terminated.append(decided)
                    cancel()

    try:
        async with _task_group() as tg:
            for index, (name, child) in enumerate(named):
                tg.start_soon(run_one, index, name, child, tg.cancel_scope.cancel)
    except Final:
        # a sibling's final() outran a terminate already recorded as a decision
        for reported in terminated:
            # _run_child accepted this context, so it is a RunnerContext
            child_context = cast(RunnerContext, context).child(reported.name)
            child_context.recorder.superseded(child_context, step, reported)
        raise

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
    *,
    root: bool = False,
) -> Reported[R] | None:
    if not isinstance(context, RunnerContext):
        raise TypeError(
            "The runner needs the RunnerContext the dispatcher provided; a Context constructed elsewhere cannot record reports."
        )
    info, accepted = _check_child(child, kind)
    if root:
        if context.path != "":
            raise ValueError(
                f"The root runs at the top layer, whose path is empty; got path {context.path!r}."
            )
        child_name = registry_unqualified_name(info)
    else:
        child_name = validate_instance_name(
            name if name is not None else registry_unqualified_name(info)
        )
    if not isinstance(step, tuple(accepted)):
        return None
    child_context = context if root else context.child(child_name)
    invoke = cast(Callable[[Context, Step], Awaitable[Report | None]], child)
    report: Report | None = None
    finals: list[Final] = []
    failure: BaseException | None = None
    try:
        report = await invoke(child_context, step)
    except anyio.get_cancelled_exc_class():
        # a recorder that raises here fails the layer, like one that raises
        # from record(); a cancellation record that cannot be written is not
        # something to paper over
        child_context.recorder.cancelled(child_context, step, child_name)
        raise
    except Final as ex:
        finals = [ex]
    except BaseExceptionGroup as ex:
        # a protocol that fanned out with its own task group or tg_collect
        errors, finals = _partition(ex)
        if not errors and not finals:
            raise
        if errors:
            failure = _surface(errors, finals)
    if failure is not None:
        raise failure
    if finals:
        raise _on_final(finals, kind, child_context, step, child_name)
    if report is not None and not isinstance(report, report_type):
        raise TypeError(
            f"{kind} {child_name!r} returned a {type(report).__name__}; a {kind} must return {report_type.__name__} or None."
        )
    if report is None:
        return None
    if isinstance(report, Decision):
        _validate_shape(report, step, child_name)
    reported = Reported(name=child_name, path=child_context.path, report=report)
    child_context.recorder.record(child_context, step, reported)
    return reported


def _on_final(
    finals: Sequence[Final],
    kind: Literal["monitor", "protocol"],
    child_context: RunnerContext,
    step: Step,
    child_name: str,
) -> Final:
    winner = finals[0]
    unclaimed = [ex for ex in finals if ex.origin is None]
    if kind == "monitor" and unclaimed:
        _supersede([ex for ex in finals if ex.origin is not None])
        raise TypeError(
            f"monitor {child_name!r} called final(); a monitor returns observations, and only a protocol may end the step."
        ) from unclaimed[0]
    own = winner.origin is None
    claimed = [ex for ex in finals if ex.origin is not None]
    for ex in unclaimed:
        try:
            _validate_shape(ex.decision, step, child_name)
        except ValueError:
            _supersede(claimed)
            raise
    for ex in unclaimed:
        ex.origin = Origin(
            child_context,
            step,
            Reported(name=child_name, path=child_context.path, report=ex.decision),
        )
    if not own:
        child_context.recorder.bypassed(child_context, step, child_name)
    return cast(Final, _surface([], finals))


def _validate_shape(decision: Decision, step: Step, name: str) -> None:
    try:
        validate_decision_shape(decision, step)
    except ValueError as ex:
        raise ValueError(f"protocol {name!r}: {ex}") from ex


def _check_child(
    child: object, expected: Literal["monitor", "protocol"] | None
) -> tuple[RegistryInfo, frozenset[type[Any]]]:
    accepted = step_types(
        cast(Any, child)
    )  # rejects factories and undecorated functions
    info = registry_info(child)
    if expected is not None and info.type != expected:
        raise TypeError(f"Expected a {expected}, got the {info.type} {info.name!r}.")
    return info, accepted


def named_children(
    children: Mapping[str, Monitor | ControlProtocol]
    | Iterable[Monitor | ControlProtocol],
    expected: Literal["monitor", "protocol"] | None,
) -> list[tuple[str, Monitor | ControlProtocol]]:
    pairs: list[tuple[object, Monitor | ControlProtocol]]
    if isinstance(children, Mapping):
        mapping = cast(Mapping[str, Monitor | ControlProtocol], children)
        pairs = [(key, child) for key, child in mapping.items()]
        keyed = True
    elif isinstance(children, Sequence) and not isinstance(children, str):
        pairs = [(None, child) for child in children]
        keyed = False
    else:
        raise TypeError(
            "children must be a Mapping or a Sequence; a set or an iterator has no configuration order"
        )
    named: list[tuple[str, Monitor | ControlProtocol]] = []
    seen: set[str] = set()
    for given, child in pairs:
        info, _ = _check_child(child, expected)
        name = validate_instance_name(
            given if keyed else registry_unqualified_name(info)
        )
        if name in seen:
            raise ValueError(
                f"Duplicate instance name {name!r} in one layer. Give the children distinct names with a mapping."
            )
        seen.add(name)
        named.append((name, child))
    return named
