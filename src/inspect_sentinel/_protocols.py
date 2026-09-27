from __future__ import annotations

from ._context import Context
from ._monitor import Children, ControlProtocol, Monitors, protocol
from ._report import Decision
from ._runner import PRECEDENCE, run_children, run_monitors
from ._step import BeforeToolCall, Step


@protocol
def observe(monitors: Monitors) -> ControlProtocol:
    """Record every monitor's observation and never act.

    What a bare monitor or a list of monitors compiles to, and how a benign score distribution is collected before anything is configured to act on it.

    Args:
        monitors: The monitors to run at every step they watch.
    """

    async def run(context: Context, step: Step) -> Decision | None:
        await run_monitors(monitors, context, step)
        return None

    return run


@protocol
def concurrent(children: Children) -> ControlProtocol:
    """Run every child concurrently; the strongest decision wins.

    What a list containing a protocol compiles to. Monitors are recorded and their observations left for a parent to read; protocols vote by `terminate > reject > modify > continue`, an authoritative decision wins a tie, `escalate` does not count, and if every protocol escalated the result is `continue`. A `modify` when more than one protocol decided becomes a `reject` naming the modifier, since the others decided about the call as it stood; the rejection carries the modifier's `authoritative`, `audit` and `metadata`.

    Args:
        children: Monitors and protocols to run together.
    """

    async def run(context: Context, step: Step) -> Decision | None:
        reports = await run_children(children, context, step)
        if not reports.decisions:
            return None
        strongest = reports.decisions.strongest()
        if strongest is None:
            return Decision.clear()
        deciding = [d for d in reports.decisions if d.report.action in PRECEDENCE]
        if strongest.report.action == "modify" and len(deciding) > 1:
            return Decision(
                action="reject",
                authoritative=strongest.report.authoritative,
                audit=strongest.report.audit,
                metadata=strongest.report.metadata,
                explanation=f"{strongest.name} modified the call but {len(deciding) - 1} other protocol(s) also decided",
            )
        return strongest.report

    return run


@protocol
def threshold(
    monitors: Monitors, reject_at: float, terminate_at: float | None = None
) -> ControlProtocol:
    """Act on a suspicion score: `reject` at or above one constant, `terminate` at or above another.

    The smallest protocol, and the one calibration feeds: the constants come from a benign score distribution, which is why they live here and not on a monitor.

    Args:
        monitors: The monitors whose scores are thresholded; the highest wins.
        reject_at: Suspicion at or above which the call is rejected.
        terminate_at: Suspicion at or above which the sample is ended; `None` never terminates.
    """

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
