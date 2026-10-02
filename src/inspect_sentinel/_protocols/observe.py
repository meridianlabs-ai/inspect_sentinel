from __future__ import annotations

from .._context import Context
from .._decorators import forwards, protocol
from .._report import Decision
from .._results import warn_failed
from .._runner import run_monitors
from .._step import Step
from .._types import Monitor, MonitorGroup, Monitors, Protocol
from .._validate import named_children


@protocol
def observe(monitors: Monitor | MonitorGroup | Monitors) -> Protocol:
    """Record every monitor's observation and never act.

    A monitor that raises is recorded as failed and logged as a warning, once per instance and exception type, and the step continues.

    What a bare monitor or a list of monitors resolves to, and how a benign score distribution is collected before anything is configured to act on it.

    Args:
        monitors: The monitor, or monitors, to run at every step they watch.
    """
    # so a misconfiguration fails here rather than at the first step
    if not named_children(monitors, "monitor"):
        raise ValueError("observe needs at least one child.")

    async def run(context: Context, step: Step) -> Decision | None:
        observations = await run_monitors(monitors, context, step)
        warn_failed(observations.failed)
        return None

    return forwards(run)
