from __future__ import annotations

import math

from ._context import Context
from ._monitor import Children, ControlProtocol, Group, Monitors, members, protocol
from ._report import Decision, Reported, deciding
from ._runner import describe, named_children, run_children, run_monitors
from ._step import BeforeToolCall, Step


@protocol
def observe(monitors: Monitors) -> ControlProtocol:
    """Record every monitor's observation and never act.

    What a bare monitor or a list of monitors resolves to, and how a benign score distribution is collected before anything is configured to act on it.

    Args:
        monitors: The monitors to run at every step they watch.
    """
    # so a misconfiguration fails here rather than at the first step
    if not named_children(monitors, "monitor"):
        raise ValueError("observe needs at least one child.")

    async def run(context: Context, step: Step) -> Decision | None:
        await run_monitors(monitors, context, step)
        return None

    return run


@protocol
def concurrent(children: Children) -> ControlProtocol:
    """Run every child at once; the strictest decision wins.

    What a list containing a protocol resolves to. Monitors are recorded in the transcript; protocols vote by `terminate > reject > modify > continue`, the first in configuration order winning a tie, `escalate` does not count, and if every protocol escalated the result is `continue`. A `modify` when another protocol also decided becomes a `reject` naming the modifier, since the others decided about the call as it stood; votes count by instance, so another function of the modifier's own instance contests it only by also modifying; the rejection carries the modifier's `audit` and `metadata`, and its explanation leads with the modifier's. When more than one protocol decided, the layer's explanation lists each one's decision after the winner's own, labelling a function of a multi-function instance `name.function`. A child that calls `final()` ends the step and no vote is taken.

    Args:
        children: Monitors and protocols to run together.
    """
    # so a misconfiguration fails here rather than at the first step
    named = named_children(children, None)
    if not named:
        raise ValueError("concurrent needs at least one child.")
    grouped = {name for name, child in named if isinstance(child, Group)}

    def label(reported: Reported[Decision]) -> str:
        if reported.name in grouped:
            return f"{reported.name}.{reported.function}"
        return reported.name

    async def run(context: Context, step: Step) -> Decision | None:
        reports = await run_children(children, context, step)
        if not reports.decisions:
            return None
        strongest = reports.decisions.strongest()
        if strongest is None:
            return Decision.clear()
        voters = deciding(reports.decisions)
        if len(voters) == 1:
            return strongest.report
        own = strongest.report.explanation
        update: dict[str, object] = {}
        if strongest.report.action == "modify":
            # a function of the modifier's own instance contests only by modifying
            contesting = [
                d
                for d in voters
                if d is not strongest
                and (d.path != strongest.path or d.report.action == "modify")
            ]
            others = {d.path for d in contesting if d.path != strongest.path}
            count = len(others) + sum(d.path == strongest.path for d in contesting)
            if contesting:
                rewrite = f"{label(strongest)} modified the call but {count} other vote(s) also decided"
                own = f"{rewrite}: {own}" if own else rewrite
                update = {"action": "reject", "modified": None}
        summary = "; ".join(f"{label(d)}: {d.report.action}" for d in voters)
        update["explanation"] = f"{own} ({summary})" if own else summary
        return strongest.report.model_copy(update=update)

    return run


@protocol
def threshold(
    monitors: Monitors, reject_at: float, terminate_at: float | None = None
) -> ControlProtocol:
    """Act on a suspicion score: `reject` at or above one constant, `terminate` at or above another.

    The smallest protocol, and the one calibration feeds: the constants come from a benign score distribution, which is why they live here and not on a monitor. Every monitor, and every function of a monitor whose factory returned several, must watch `BeforeToolCall`, since that is the only stage a `reject` is legal at.

    Args:
        monitors: The monitors whose scores are thresholded; the highest wins.
        reject_at: Suspicion at or above which the call is rejected.
        terminate_at: Suspicion at or above which the sample is ended; `None` never terminates. Must be above `reject_at`.
    """
    named = named_children(monitors, "monitor")
    if not named:
        raise ValueError("threshold needs at least one child.")
    if not math.isfinite(reject_at) or (
        terminate_at is not None and not math.isfinite(terminate_at)
    ):
        raise ValueError(
            f"threshold's constants must be finite; got reject_at={reject_at}, terminate_at={terminate_at}."
        )
    if terminate_at is not None and terminate_at <= reject_at:
        raise ValueError(
            f"threshold's terminate_at must be above reject_at, or reject would never fire; got reject_at={reject_at}, terminate_at={terminate_at}."
        )
    for name, child in named:
        grouped = isinstance(child, Group)
        for member in members(child):
            if BeforeToolCall not in member.accepted:
                label = describe(name, member.function.__name__, grouped)
                raise TypeError(
                    f"threshold acts before tool calls; monitor {label} never watches that stage"
                )

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        score = (await run_monitors(monitors, context, step)).max_suspicion()
        if score is None:
            return None
        if terminate_at is not None and score >= terminate_at:
            return Decision.terminate(f"suspicion {score:.2f}")
        if score >= reject_at:
            return Decision.reject(f"suspicion {score:.2f}")
        return Decision.clear()

    return decide
