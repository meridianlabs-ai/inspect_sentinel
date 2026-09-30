from __future__ import annotations

import sys
from collections.abc import (
    AsyncGenerator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import replace
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
    Child,
    Children,
    Group,
    Monitor,
    MonitorGroup,
    Monitors,
    Protocol,
    ProtocolGroup,
    Protocols,
    SentinelFunction,
    invoke,
    members,
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


def peak_suspicion(observation: Observation) -> float:
    suspicion = observation.suspicion
    return max(suspicion.values()) if isinstance(suspicion, dict) else suspicion


class Observations(_ReportSequence[Observation]):
    """What a layer's monitors reported, in configuration order."""

    def max_suspicion(self) -> float | None:
        """The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed."""
        scores = [peak_suspicion(o.report) for o in self._items]
        return max(scores) if scores else None


class Decisions(_ReportSequence[Decision]):
    """What a layer's protocols decided, in configuration order."""

    def strongest(self) -> Reported[Decision] | None:
        """The strongest decision by `terminate > reject > modify > continue`, the first in configuration order on a tie; `escalate` does not count; `None` if nobody decided."""
        ranked = deciding(self._items)
        if not ranked:
            return None
        return max(ranked, key=lambda d: PRECEDENCE[d.report.action])


class Reports(NamedTuple):
    """Both families of report from one layer, unpackable as `observations, decisions = await run_children(...)`."""

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


async def run_root(protocol: Protocol, context: Context, step: Step) -> Decision | None:
    """Invoke the resolved root protocol for one step and return the step's outcome.

    The root is recorded at the empty path under its registry name without the package prefix, so its children's paths are bare, and its context's `factory` is set to its full registry name. Its decision is shape-checked and recorded like any layer's; a `decide_final()` from below records the root as bypassed, and its decision is recorded here, the one time it is recorded, and returned, so the caller need not catch `Final`.

    Args:
        protocol: The root protocol, as `resolve_sentinel` returned it.
        context: The top layer's context, whose `path` is empty.
        step: The step being examined.
    """
    if isinstance(protocol, Group):
        raise TypeError(
            "The root is one protocol function, as resolve_sentinel returns it; a group of functions cannot be the root."
        )
    found: list[Reported[Decision]] = []
    try:
        await _run_child(
            protocol, "protocol", Decision, context, step, None, found, root=True
        )
    except Final as ex:
        origin = ex.origin
        if origin is None:
            raise RuntimeError(
                f"Runner invariant violated: Final({ex.decision.action!r}) left the root without an origin."
            ) from ex
        origin.context.recorder.record(origin.context, origin.step, origin.reported)
        return ex.decision
    return found[0].report if found else None


async def run_monitors(
    monitors: Monitor | MonitorGroup | Monitors, context: Context, step: Step
) -> Observations:
    """Run monitors concurrently and collect their observations in configuration order.

    Derives each child's context under this layer's path and records every observation, including the ones the caller goes on to ignore. A monitor that abstained or does not watch this stage contributes nothing, so the result may be empty. An instance whose factory returned several functions contributes one observation per function that reported, in the order the factory returned them.

    Args:
        monitors: One monitor or `MonitorGroup`, named by its registry name without the package prefix; a sequence of them, named the same way; or a mapping of instance names to them.
        context: This layer's context.
        step: The step being examined.
    """
    named = named_children(monitors, "monitor")
    return (await _run_named(named, context, step)).observations


async def run_protocols(
    protocols: Protocol | ProtocolGroup | Protocols, context: Context, step: Step
) -> Decisions:
    """Run protocols concurrently and collect their decisions in configuration order, cancelling the rest at their next await when one returns `terminate` or calls `decide_final()`.

    A `decide_final()` from any protocol at any depth below propagates out of this call, so the caller's own decision logic does not run.

    Args:
        protocols: One protocol or `ProtocolGroup`, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.
        context: This layer's context.
        step: The step being examined.
    """
    named = named_children(protocols, "protocol")
    return (await _run_named(named, context, step)).decisions


async def run_children(
    children: Child | Children, context: Context, step: Step
) -> Reports:
    """Run monitors and protocols together in one task group, cancelling the rest at their next await when a protocol returns `terminate` or calls `decide_final()`.

    A child cancelled this way is recorded through `Recorder.cancelled`; one that finishes without awaiting is recorded normally.

    Args:
        children: One monitor, protocol or group, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.
        context: This layer's context.
        step: The step being examined.
    """
    named = named_children(children, None)
    return await _run_named(named, context, step)


async def _run_named(
    named: Sequence[tuple[str, Child]], context: Context, step: Step
) -> Reports:
    observations: list[tuple[int, Reported[Observation]]] = []
    decisions: list[tuple[int, Reported[Decision]]] = []
    terminated: list[tuple[str, Reported[Decision]]] = []

    async def run_one(
        index: int,
        name: str,
        child: Child,
        cancel: Callable[[], None],
    ) -> None:
        # a group's reports are kept as they arrive, so one cancelled part way
        # returns what it recorded, as a single child finishing first does
        if registry_info(child).type == "monitor":
            observed: list[Reported[Observation]] = []
            try:
                await _run_child(
                    child,
                    "monitor",
                    Observation,
                    context,
                    step,
                    name,
                    observed,
                )
            finally:
                observations.extend((index, o) for o in observed)
        else:
            decided: list[Reported[Decision]] = []
            try:
                await _run_child(
                    child,
                    "protocol",
                    Decision,
                    context,
                    step,
                    name,
                    decided,
                )
            finally:
                decisions.extend((index, d) for d in decided)
                terminated.extend(
                    (registry_info(child).name, d)
                    for d in decided
                    if d.report.action == "terminate"
                )
            if any(d.report.action == "terminate" for d in decided):
                cancel()

    try:
        async with _task_group() as tg:
            for index, (name, child) in enumerate(named):
                tg.start_soon(run_one, index, name, child, tg.cancel_scope.cancel)
    except Final:
        # a decide_final() outran a terminate already recorded as a decision
        for factory, reported in terminated:
            # _run_child accepted this context, so it is a RunnerContext
            child_context = cast(RunnerContext, context).child(reported.name, factory)
            child_context.recorder.superseded(child_context, step, reported)
        raise

    return Reports(
        Observations(o for _, o in sorted(observations, key=lambda t: t[0])),
        Decisions(d for _, d in sorted(decisions, key=lambda t: t[0])),
    )


async def _run_child(
    child: Child,
    kind: Literal["monitor", "protocol"],
    report_type: type[R],
    context: Context,
    step: Step,
    name: str | None,
    found: list[Reported[R]],
    *,
    root: bool = False,
) -> None:
    if not isinstance(context, RunnerContext):
        raise TypeError(
            "The runner needs the RunnerContext the dispatcher provided; a Context constructed elsewhere cannot record reports."
        )
    info, _ = _check_child(child, kind)
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
    grouped = isinstance(child, Group)
    running = [m for m in members(child) if isinstance(step, tuple(m.accepted))]
    if not running:
        return
    child_context = (
        replace(context, factory=info.name)
        if root
        else context.child(child_name, info.name)
    )
    try:
        # sequential, since a group's members share one store
        for member in running:
            reported = await _run_member(
                member.function,
                kind,
                report_type,
                child_context,
                step,
                child_name,
                grouped,
            )
            if reported is not None:
                found.append(reported)
                # nothing outranks terminate, so the rest of the group does not run
                if isinstance(reported.report, Decision) and (
                    reported.report.action == "terminate"
                ):
                    break
    except anyio.get_cancelled_exc_class():
        # a recorder that raises here fails the layer, like one that raises
        # from record(); a cancellation record that cannot be written is not
        # something to paper over
        child_context.recorder.cancelled(child_context, step, child_name)
        raise


async def _run_member(
    function: SentinelFunction,
    kind: Literal["monitor", "protocol"],
    report_type: type[R],
    child_context: RunnerContext,
    step: Step,
    child_name: str,
    grouped: bool,
) -> Reported[R] | None:
    label = describe(child_name, function.__name__, grouped)
    report: Report | None = None
    finals: list[Final] = []
    failure: BaseException | None = None
    try:
        report = await invoke(function, child_context, step)
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
        raise _on_final(
            finals, kind, child_context, step, child_name, function.__name__, label
        )
    if report is not None and not isinstance(report, report_type):
        raise TypeError(
            f"{kind} {label} returned a {type(report).__name__}; a {kind} must return {report_type.__name__} or None."
        )
    if report is None:
        return None
    if isinstance(report, Decision):
        _validate_shape(report, step, label)
    reported = Reported(
        name=child_name,
        path=child_context.path,
        report=report,
        function=function.__name__,
    )
    child_context.recorder.record(child_context, step, reported)
    return reported


def _on_final(
    finals: Sequence[Final],
    kind: Literal["monitor", "protocol"],
    child_context: RunnerContext,
    step: Step,
    child_name: str,
    function: str,
    label: str,
) -> Final:
    winner = finals[0]
    unclaimed = [ex for ex in finals if ex.origin is None]
    if kind == "monitor" and unclaimed:
        _supersede([ex for ex in finals if ex.origin is not None])
        raise TypeError(
            f"monitor {label} called decide_final(); a monitor returns observations, and only a protocol may end the step."
        ) from unclaimed[0]
    own = winner.origin is None
    claimed = [ex for ex in finals if ex.origin is not None]
    for ex in unclaimed:
        try:
            _validate_shape(ex.decision, step, label)
        except ValueError:
            _supersede(claimed)
            raise
    for ex in unclaimed:
        ex.origin = Origin(
            child_context,
            step,
            Reported(
                name=child_name,
                path=child_context.path,
                report=ex.decision,
                function=function,
            ),
        )
    if not own:
        child_context.recorder.bypassed(child_context, step, child_name)
    return cast(Final, _surface([], finals))


def describe(name: str, function: str, grouped: bool) -> str:
    return f"{name!r} (function {function!r})" if grouped else repr(name)


def _validate_shape(decision: Decision, step: Step, label: str) -> None:
    try:
        validate_decision_shape(decision, step)
    except ValueError as ex:
        raise ValueError(f"protocol {label}: {ex}") from ex


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
    children: Child | Mapping[str, Child] | Iterable[Child],
    expected: Literal["monitor", "protocol"] | None,
) -> list[tuple[str, Child]]:
    pairs: list[tuple[object, Child]]
    if callable(children) or isinstance(children, Group):
        pairs = [(None, children)]
        keyed = False
    elif isinstance(children, Mapping):
        mapping = cast(Mapping[str, Child], children)
        pairs = [(key, child) for key, child in mapping.items()]
        keyed = True
    elif isinstance(children, Sequence) and not isinstance(children, str):
        pairs = [(None, child) for child in children]
        keyed = False
    else:
        raise TypeError(
            "children must be a monitor, protocol or group, a Mapping or a Sequence; a set or an iterator has no configuration order"
        )
    named: list[tuple[str, Child]] = []
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
