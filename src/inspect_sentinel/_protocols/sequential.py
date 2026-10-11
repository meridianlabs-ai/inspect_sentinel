from __future__ import annotations

from dataclasses import replace
from typing import Any, cast, get_args, overload

from inspect_ai._util.registry import registry_info, registry_unqualified_name

from .._context import Context
from .._decorators import protocol, step_types
from .._report import Decision, NotEscalate, Proceed, Reported
from .._results import warn_failed
from .._runner import run_children
from .._step import Step
from .._types import D, Protocol, ProtocolGroup, Sentinel, Sentinels
from .._unhandled import composed, undecided_stages
from .._validate import check_child, named_children


@overload
def sequential(children: Sentinels[D]) -> Protocol[D | Proceed | None]: ...
@overload
def sequential(
    children: Sentinels, *, then: Protocol[NotEscalate] | ProtocolGroup[NotEscalate]
) -> Protocol[NotEscalate | None]: ...
@protocol
def sequential(
    children: Sentinels,
    *,
    then: Protocol[NotEscalate] | ProtocolGroup[NotEscalate] | None = None,
) -> Protocol:
    """Run children one at a time, in order; the first that decides anything but `escalate` decides, and later children do not run.

    Each child is run through the runner and recorded under its own name, and a child that does not watch the current stage is not invoked. An `escalate` hands the step to the next child with the escalation appended to `step.escalations`, so a person at the end sees who asked and why. The list starts empty for each run of the chain: escalations do not cross layers, so a nested `sequential` does not see the escalations of the chain around it. The functions of one `ProtocolGroup` run in one call and all see the step before any of their own escalations.

    A monitor is recorded and falls through, since an observation is not a decision; one that raises is recorded as failed, logged as a warning, and falls through the same way. When every child that decided escalated, the chain returns the last escalate, passing it up as `concurrent` does; when only monitors reported, it returns `continue`; when no child took part, `None`. A child that calls `decide_final()` ends the step as usual.

    `then` is a last link that always decides and never escalates wherever the children run, such as `handle_escalation()`, or `human()` asking at those stages, so the chain never escalates: its type is `Protocol[NotEscalate | None]`, `None` where neither a child nor `then` runs, and a type checker can see that the chain needs nothing after it. Without `then`, the chain's type is the union of what its children return, with `Proceed` and `None`, so a chain whose children can escalate is typed as able to.

    Args:
        children: A monitor, protocol or group, or several, to run in order.
        then: A protocol or group run after the children, which must always decide and never escalate at every stage a protocol among the children runs, recorded under its registry name like a child.

    Raises:
        ValueError: If there are no children, or `then` is named like one of them.
        TypeError: If `then` may abstain or escalate at a stage a protocol among the children runs.
    """
    # so a misconfiguration fails here rather than at the first step
    named = named_children(children, None)
    if not named:
        raise ValueError("sequential needs at least one child.")
    if then is not None:
        named.append(_last(then, named))

    async def run(context: Context, step: Step) -> Decision | None:
        escalations: list[Reported[Decision]] = []
        participated = False
        for name, child in named:
            current = replace(step, escalations=tuple(escalations))
            observations, decisions = await run_children(
                {name: child}, context, current
            )
            warn_failed(observations.failed)
            if observations.failed or observations.succeeded:
                participated = True
            strongest = decisions.strongest()
            if strongest is None:
                continue
            participated = True
            if strongest.report.action != "escalate":
                # a group link decides by its strongest function, as concurrent does
                return strongest.report
            escalations.extend(d for d in decisions if d.report.action == "escalate")
        if escalations:
            return escalations[-1].report
        return Decision.proceed() if participated else None

    # its type is built from its children, by the overloads above
    return cast(Protocol, composed(run, "in_order", named))


def _last(then: Sentinel, named: list[tuple[str, Sentinel]]) -> tuple[str, Sentinel]:
    info, _ = check_child(then, "protocol")
    name = registry_unqualified_name(info)
    # wherever a child can escalate, then must answer it
    stages = frozenset[type[Any]]().union(
        *(
            step_types(child)
            for _, child in named
            if registry_info(child).type == "protocol"
        )
    )
    undecided = undecided_stages(then, [s for s in get_args(Step) if s in stages])
    if undecided:
        raise TypeError(
            f"sequential's then must always decide and never escalate wherever its children run, as handle_escalation() does; {name} may abstain or escalate at {', '.join(undecided)}."
        )
    if name in {child for child, _ in named}:
        raise ValueError(
            f"sequential's then is named {name!r}, like one of its children; give the children other names with a mapping."
        )
    return name, then
