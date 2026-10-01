from __future__ import annotations

from dataclasses import replace

from .._context import Context
from .._decorators import protocol
from .._report import Decision, Reported
from .._runner import run_children
from .._step import Step
from .._types import Children, Monitor, MonitorGroup, Protocol, ProtocolGroup
from .._validate import named_children


@protocol
def sequential(
    children: Monitor | MonitorGroup | Protocol | ProtocolGroup | Children,
) -> Protocol:
    """Run children one at a time, in order; the first that decides anything but `escalate` decides, and later children do not run.

    Each child is run through the runner and recorded under its own name, and a child that does not watch the current stage is not invoked. An `escalate` hands the step to the next child with the escalation appended to `step.escalations`, so a person at the end sees who asked and why. The list starts empty for each run of the chain: escalations do not cross layers, so a nested `sequential` does not see the escalations of the chain around it. The functions of one `ProtocolGroup` run in one call and all see the step before any of their own escalations.

    A monitor is recorded and falls through, since an observation is not a decision. When every child that decided escalated, the chain returns the last escalate, passing it up as `concurrent` does; when only monitors reported, it returns `continue`; when no child took part, `None`. A child that calls `decide_final()` ends the step as usual.

    Args:
        children: A monitor, protocol or group, or several, to run in order.
    """
    # so a misconfiguration fails here rather than at the first step
    named = named_children(children, None)
    if not named:
        raise ValueError("sequential needs at least one child.")

    async def run(context: Context, step: Step) -> Decision | None:
        escalations: list[Reported[Decision]] = []
        participated = False
        for name, child in named:
            current = replace(step, escalations=tuple(escalations))
            observations, decisions = await run_children(
                {name: child}, context, current
            )
            if observations:
                participated = True
            for reported in decisions:
                participated = True
                if reported.report.action != "escalate":
                    return reported.report
                escalations.append(reported)
        if escalations:
            return escalations[-1].report
        return Decision.proceed() if participated else None

    return run
