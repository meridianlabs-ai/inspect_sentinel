from __future__ import annotations

import sys
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, overload

import anyio
from anyio.abc import TaskGroup
from inspect_ai._util.registry import (
    RegistryInfo,
    registry_info,
    registry_unqualified_name,
)

from ._check import check_decision_shape
from ._context import Context, RunnerContext, check_instance_name
from ._final import Final, Origin, Passed
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


# Recording a final decision waits for the outermost runner frame, since only
# there is it known to have won every race on the way out.
_nested: ContextVar[bool] = ContextVar("inspect_sentinel_nested", default=False)


@contextmanager
def _frame() -> Generator[bool]:
    outermost = not _nested.get()
    token = _nested.set(True)
    try:
        yield outermost
    finally:
        _nested.reset(token)


def _settle(ex: Final, *, won: bool) -> None:
    origin = ex.origin
    if origin is None:
        raise RuntimeError(
            f"Final({ex.decision.action!r}) reached the runner's task group without passing through a protocol."
        )
    recorder = origin.context.recorder
    if won:
        recorder.record(origin.context, origin.step, origin.reported)
    else:
        recorder.superseded(origin.context, origin.step, origin.reported)
    for passed in ex.bypassed:
        passed.context.recorder.bypassed(passed.context, passed.step, passed.name)


def _settle_race(finals: Sequence[Final], *, record_winner: bool) -> None:
    for loser in finals[1:]:
        _settle(loser, won=False)
    if finals and record_winner:
        _settle(finals[0], won=True)


@asynccontextmanager
async def _task_group() -> AsyncGenerator[TaskGroup]:
    # Nested groups are flattened to their leaves, so a Final raised inside a
    # protocol's own task group is still seen. Only the first failure surfaces,
    # as in inspect_ai's tg_collect; it is re-raised outside the handler so its
    # own __cause__ and __context__ survive and the group does not appear in
    # the traceback. On trio a child error can arrive alongside the
    # cancellation in one BaseExceptionGroup; the error is surfaced, as anyio's
    # asyncio backend already does, and a group holding only cancellations
    # propagates untouched. An error outranks a Final, so a bug is not hidden
    # behind a final decision; the first Final is still recorded as a decision.
    # Of several Finals the first to arrive wins and the rest are superseded.
    first: BaseException | None = None
    with _frame() as outermost:
        try:
            async with anyio.create_task_group() as tg:
                yield tg
        except BaseExceptionGroup as ex:
            leaves = _leaves(ex)
            if not leaves:
                raise
            errors = [leaf for leaf in leaves if not isinstance(leaf, Final)]
            finals = [leaf for leaf in leaves if isinstance(leaf, Final)]
            _settle_race(finals, record_winner=outermost or bool(errors))
            first = (errors or leaves)[0]
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
    info, accepted = _check_child(child, kind)
    child_name = check_instance_name(
        name if name is not None else registry_unqualified_name(info)
    )
    if not isinstance(step, tuple(accepted)):
        return None
    child_context = context.child(child_name)
    invoke = cast(Callable[[Context, Step], Awaitable[Report | None]], child)
    report: Report | None = None
    finals: list[Final] = []
    with _frame() as outermost:
        try:
            report = await invoke(child_context, step)
        except anyio.get_cancelled_exc_class():
            # a recorder that raises here fails the layer, like one that raises
            # from record(); a cancellation record that cannot be written is
            # not something to paper over
            child_context.recorder.cancelled(child_context, step, child_name)
            raise
        except Final as ex:
            finals = [ex]
        except BaseExceptionGroup as ex:
            # a protocol that fanned out with its own task group or tg_collect
            leaves = _leaves(ex)
            finals = [leaf for leaf in leaves if isinstance(leaf, Final)]
            if not finals or len(finals) != len(leaves):
                if outermost:
                    _settle_race(
                        [f for f in finals if f.origin is not None],
                        record_winner=True,
                    )
                raise
    if finals:
        raise _on_final(finals, kind, child_context, step, child_name, outermost)
    if report is not None and not isinstance(report, report_type):
        raise TypeError(
            f"{kind} {child_name!r} returned a {type(report).__name__}; a {kind} must return {report_type.__name__} or None."
        )
    if report is None:
        return None
    if isinstance(report, Decision):
        _check_shape(report, step, child_name)
    reported = Reported(name=child_name, path=child_context.path, report=report)
    child_context.recorder.record(child_context, step, reported)
    return reported


def _on_final(
    finals: Sequence[Final],
    kind: Literal["monitor", "protocol"],
    child_context: RunnerContext,
    step: Step,
    child_name: str,
    outermost: bool,
) -> Final:
    winner = finals[0]
    unclaimed = [ex for ex in finals if ex.origin is None]
    if kind == "monitor" and unclaimed:
        raise TypeError(
            f"monitor {child_name!r} called final(); a monitor returns observations, and only a protocol may end the step."
        ) from unclaimed[0]
    own = winner.origin is None
    for ex in finals:
        if ex.origin is None:
            _check_shape(ex.decision, step, child_name)
            ex.origin = Origin(
                child_context,
                step,
                Reported(name=child_name, path=child_context.path, report=ex.decision),
            )
    if not own:
        winner.bypassed.append(Passed(child_context, step, child_name))
    _settle_race(finals, record_winner=outermost)
    return winner


def _check_shape(decision: Decision, step: Step, name: str) -> None:
    try:
        check_decision_shape(decision, step)
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
        name = check_instance_name(given if keyed else registry_unqualified_name(info))
        if name in seen:
            raise ValueError(
                f"Duplicate instance name {name!r} in one layer. Give the children distinct names with a mapping."
            )
        seen.add(name)
        named.append((name, child))
    return named
