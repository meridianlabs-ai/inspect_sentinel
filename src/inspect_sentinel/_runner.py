from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Literal, NamedTuple, TypeVar, cast

import anyio
from anyio.abc import TaskGroup
from inspect_ai._util.registry import registry_info, registry_unqualified_name
from inspect_ai.util import LimitExceededError

from ._context import Context, validate_instance_name
from ._decorators import invoke, members
from ._final import Final, Origin
from ._host import HostContext, enter_layer, layer, running_step
from ._report import Decision, Failed, Observation, Report, Reported
from ._results import Decisions, Observations, Reports
from ._step import Step
from ._types import (
    Group,
    Monitor,
    MonitorGroup,
    Monitors,
    Protocol,
    ProtocolGroup,
    Protocols,
    Sentinel,
    SentinelFunction,
    Sentinels,
)
from ._validate import check_child, named_children, validate_shape

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


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
        recorder, factory = layer(origin.context, "The runner")
        recorder.superseded(origin.context, factory, origin.step, origin.reported)


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


async def run_sentinel(
    protocol: Protocol, host_context: HostContext, step: Step
) -> Decision | None:
    """Run the sentinel `resolve_sentinel` returned for one step and return the step's outcome.

    The root is recorded at the empty path under its registry name without the package prefix, so its children's paths are bare, and its full registry name is the `factory` its records carry. Its decision is shape-checked and recorded like any layer's; a `decide_final()` from below records the root as bypassed, and its decision is recorded here, the one time it is recorded, and returned, so the caller need not catch `Final`.

    An `escalate` is returned like any other decision, though at the root there is nobody to hand it to; the host decides what an unhandled escalate means, and inspect_ai's host ends the sample and records the step as an unhandled escalation.

    Args:
        protocol: The root protocol, as `resolve_sentinel` returned it.
        host_context: The top layer's context, whose `path` is empty, the recorder and the sample store.
        step: The step being examined.
    """
    if isinstance(protocol, Group):
        raise TypeError(
            "The root is one protocol function, as resolve_sentinel returns it; a group of functions cannot be the root."
        )
    found: list[Reported[Decision]] = []
    with running_step(host_context.recorder, host_context.store):
        try:
            await _run_child(
                protocol,
                "protocol",
                Decision,
                host_context.context,
                step,
                None,
                found,
                [],
                root=True,
            )
        except Final as ex:
            origin = ex.origin
            if origin is None:
                raise RuntimeError(
                    f"Runner invariant violated: Final({ex.decision.action!r}) left the root without an origin."
                ) from ex
            recorder, factory = layer(origin.context, "The runner")
            recorder.record(origin.context, factory, origin.step, origin.reported)
            return ex.decision
    return found[0].report if found else None


async def run_monitors(
    monitors: Monitor | MonitorGroup | Monitors, context: Context, step: Step
) -> Observations:
    """Run monitors concurrently and collect their observations in configuration order.

    A typed shortcut for `run_children` over monitors only: its parameter rejects a protocol and its return type is the observations alone.

    Derives each child's context under this layer's path and records every observation, including the ones the caller goes on to ignore. A monitor that abstained or does not watch this stage contributes nothing, so the result may be empty. An instance whose factory returned several functions contributes one observation per function that reported, in the order the factory returned them.

    A monitor function that raises does not fail the call: its failure is recorded through `Recorder.failed` and listed in the result's `failed`, and reading the result's observations then raises `MonitorFailedError`, so a caller that reads them fails unless it checks `failed` first. A `LimitExceededError` is not a monitor failure; it propagates, as a cancellation does.

    Args:
        monitors: One monitor or `MonitorGroup`, named by its registry name without the package prefix; a sequence of them, named the same way; or a mapping of instance names to them.
        context: This layer's context, the one this protocol was given. Call this while the protocol runs; after it returns this raises `RuntimeError`.
        step: The step being examined.
    """
    named = named_children(monitors, "monitor")
    return (await _run_named(named, context, step, "run_monitors")).observations


async def run_protocols(
    protocols: Protocol | ProtocolGroup | Protocols, context: Context, step: Step
) -> Decisions:
    """Run protocols concurrently and collect their decisions in configuration order, cancelling the rest at their next await when one returns `terminate` or calls `decide_final()`.

    A typed shortcut for `run_children` over protocols only: its parameter rejects a monitor and its return type is the decisions alone. A `decide_final()` from any protocol at any depth below propagates out of this call, so the caller's own decision logic does not run.

    Args:
        protocols: One protocol or `ProtocolGroup`, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.
        context: This layer's context, the one this protocol was given. Call this while the protocol runs; after it returns this raises `RuntimeError`.
        step: The step being examined.
    """
    named = named_children(protocols, "protocol")
    return (await _run_named(named, context, step, "run_protocols")).decisions


async def run_children(children: Sentinels, context: Context, step: Step) -> Reports:
    """Run monitors and protocols together in one task group, cancelling the rest at their next await when a protocol returns `terminate` or calls `decide_final()`.

    Runs any mix of children; `run_monitors` and `run_protocols` are typed shortcuts for one family. A child cancelled this way is recorded through `Recorder.cancelled`; one that finishes without awaiting is recorded normally.

    Args:
        children: One monitor, protocol or group, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.
        context: This layer's context, the one this protocol was given. Call this while the protocol runs; after it returns this raises `RuntimeError`.
        step: The step being examined.
    """
    named = named_children(children, None)
    return await _run_named(named, context, step, "run_children")


async def _run_named(
    named: Sequence[tuple[str, Sentinel]], context: Context, step: Step, caller: str
) -> Reports:
    recorder = layer(context, caller).recorder
    observations: list[tuple[int, Reported[Observation]]] = []
    failures: list[tuple[int, Failed]] = []
    decisions: list[tuple[int, Reported[Decision]]] = []
    terminated: list[tuple[str, Reported[Decision]]] = []

    async def run_one(
        index: int,
        name: str,
        child: Sentinel,
        cancel: Callable[[], None],
    ) -> None:
        # a group's reports are kept as they arrive, so one cancelled part way
        # returns what it recorded, as a single child finishing first does
        if registry_info(child).type == "monitor":
            observed: list[Reported[Observation]] = []
            failed: list[Failed] = []
            try:
                await _run_child(
                    child,
                    "monitor",
                    Observation,
                    context,
                    step,
                    name,
                    observed,
                    failed,
                )
            finally:
                observations.extend((index, o) for o in observed)
                failures.extend((index, f) for f in failed)
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
                    [],
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
            recorder.superseded(
                _child_context(context, reported.name),
                factory,
                step,
                reported,
            )
        raise

    return Reports(
        Observations(
            (o for _, o in sorted(observations, key=lambda t: t[0])),
            (f for _, f in sorted(failures, key=lambda t: t[0])),
        ),
        Decisions(d for _, d in sorted(decisions, key=lambda t: t[0])),
    )


async def _run_child(
    child: Sentinel,
    kind: Literal["monitor", "protocol"],
    report_type: type[R],
    context: Context,
    step: Step,
    name: str | None,
    found: list[Reported[R]],
    failed: list[Failed],
    *,
    root: bool = False,
) -> None:
    info, _ = check_child(child, kind)
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
    child_context = context if root else _child_context(context, child_name)
    # registered before the stage filter, so a path conflict fails at every stage
    enter_layer(child_context.path, info.name)
    grouped = isinstance(child, Group)
    running = [m for m in members(child) if isinstance(step, tuple(m.accepted))]
    if not running:
        return
    recorder = layer(child_context, "The runner").recorder
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
            if isinstance(reported, Failed):
                failed.append(reported)
            elif reported is not None:
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
        recorder.cancelled(child_context, info.name, step, child_name)
        raise


async def _run_member(
    function: SentinelFunction,
    kind: Literal["monitor", "protocol"],
    report_type: type[R],
    child_context: Context,
    step: Step,
    child_name: str,
    grouped: bool,
) -> Reported[R] | Failed | None:
    recorder, factory = layer(child_context, "The runner")
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
            # a limit ends the sample, whichever leaf surfaced
            limits = [e for e in errors if isinstance(e, LimitExceededError)]
            if limits:
                failure = limits[0]
    except Exception as ex:
        failure = ex
    if failure is not None:
        if (
            kind == "monitor"
            and isinstance(failure, Exception)
            and not isinstance(failure, LimitExceededError)
        ):
            # a monitor that turned its cancellation into an error was cancelled
            await anyio.lowlevel.checkpoint_if_cancelled()
            failed = Failed(
                name=child_name,
                path=child_context.path,
                function=function.__name__,
                error=failure,
            )
            recorder.failed(child_context, factory, step, failed)
            return failed
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
        validate_shape(report, step, label)
    reported = Reported(
        name=child_name,
        path=child_context.path,
        report=report,
        function=function.__name__,
    )
    recorder.record(child_context, factory, step, reported)
    return reported


def _on_final(
    finals: Sequence[Final],
    kind: Literal["monitor", "protocol"],
    child_context: Context,
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
            validate_shape(ex.decision, step, label)
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
        recorder, factory = layer(child_context, "The runner")
        recorder.bypassed(child_context, factory, step, child_name)
    return cast(Final, _surface([], finals))


def _child_context(context: Context, name: str) -> Context:
    validate_instance_name(name)
    return replace(context, path=f"{context.path}/{name}" if context.path else name)


def describe(name: str, function: str, grouped: bool) -> str:
    return f"{name!r} (function {function!r})" if grouped else repr(name)
