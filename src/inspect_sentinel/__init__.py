"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

from ._context import Context, Host
from ._final import final
from ._monitor import (
    Children,
    ControlProtocol,
    Monitor,
    Monitors,
    Protocols,
    monitor,
    protocol,
)
from ._protocols import concurrent, observe, threshold
from ._report import Action, Decision, Observation, Report, Reported, Suspicion
from ._runner import (
    Decisions,
    Observations,
    Reports,
    run_children,
    run_monitors,
    run_protocols,
)
from ._step import AfterToolCall, BeforeToolCall, Step

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "Action",
    "AfterToolCall",
    "BeforeToolCall",
    "Children",
    "Context",
    "ControlProtocol",
    "Decision",
    "Decisions",
    "Host",
    "Monitor",
    "Monitors",
    "Observation",
    "Observations",
    "Protocols",
    "Report",
    "Reported",
    "Reports",
    "Step",
    "Suspicion",
    "concurrent",
    "final",
    "monitor",
    "observe",
    "protocol",
    "run_children",
    "run_monitors",
    "run_protocols",
    "threshold",
    "__version__",
]
