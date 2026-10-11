from __future__ import annotations

from typing import cast, overload

from .._context import Context
from .._decorators import protocol
from .._report import Decision, Reject, Reported
from .._results import warn_failed
from .._runner import run_children
from .._step import Step
from .._types import D, Group, Monitor, MonitorGroup, Monitors, Protocol, Sentinels
from .._unhandled import composed
from .._validate import named_children


@overload
def concurrent(children: Monitor | MonitorGroup | Monitors) -> Protocol[None]: ...
@overload
def concurrent(children: Sentinels[D]) -> Protocol[D | Reject | None]: ...
@protocol
def concurrent(
    children: Sentinels,
) -> Protocol:
    """Run every child at once; the strictest decision wins.

    What a list containing a protocol resolves to. Monitors are recorded in the transcript, a monitor that raises recorded as failed and logged as a warning while the others go on; protocols vote by `terminate > reject > modify > escalate > continue`, the first in configuration order winning a tie, so a peer's `continue` does not override an `escalate` and the layer escalates when that is the strongest vote. A `modify` when another protocol also decided anything but `escalate` becomes a `reject` naming the modifier, since the others decided about the call as it stood; votes count by instance, so another function of the modifier's own instance contests it only by also modifying; the rejection carries the modifier's `audit` and `metadata`, and its explanation leads with the modifier's. When more than one protocol decided and not all of them continued, the layer's explanation lists each one's decision after the winner's own, labelling a function of a `ProtocolGroup` or `MonitorGroup` `name.function`. A child that calls `decide_final()` ends the step and no vote is taken.

    Its type is the union of what its children return, with `Reject`, since a contested modify becomes one, and `None`, since it abstains at a stage none of its protocols decides at. A group whose children can escalate is typed as able to, and one of monitors alone is a `Protocol[None]`.

    Args:
        children: A monitor, protocol or group, or several, to run together.
    """
    # so a misconfiguration fails here rather than at the first step
    named = named_children(children, None)
    # run what was checked and recorded, not the caller's container as it is later
    snapshot = dict(named)
    if not named:
        raise ValueError("concurrent needs at least one child.")
    grouped = {name for name, child in named if isinstance(child, Group)}

    def label(reported: Reported[Decision]) -> str:
        if reported.name in grouped:
            return f"{reported.name}.{reported.function}"
        return reported.name

    async def run(context: Context, step: Step) -> Decision | None:
        reports = await run_children(snapshot, context, step)
        warn_failed(reports.observations.failed)
        voters = reports.decisions
        strongest = voters.strongest()
        if strongest is None:
            return None
        if len(voters) == 1 or all(d.report.action == "continue" for d in voters):
            return strongest.report
        own = strongest.report.explanation
        update: dict[str, object] = {}
        if strongest.report.action == "modify":
            # an escalate does not contest, and a function of the modifier's own
            # instance contests only by modifying
            contesting = [
                d
                for d in voters
                if d is not strongest
                and d.report.action != "escalate"
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
        decided = strongest.report.model_copy(update=update)
        if decided.action == "reject":
            # the contested modify, rewritten, is a Reject
            return Reject.model_validate(decided.model_dump())
        return decided

    # its type is built from its children's, by the overloads above
    return cast(Protocol, composed(run, "at_once", named))
