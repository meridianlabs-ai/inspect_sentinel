"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

from ._context import Context, Host
from ._monitor import (
    Children,
    ControlProtocol,
    Monitor,
    Monitors,
    Protocols,
    monitor,
    protocol,
)
from ._report import Action, Decision, Observation, Report, Reported, Suspicion
from ._runner import (
    Decisions,
    Observations,
    Reports,
    run_children,
    run_monitor,
    run_monitors,
    run_protocol,
    run_protocols,
)
from ._step import AfterToolCall, BeforeToolCall, Stage, Step

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
    "Stage",
    "Step",
    "Suspicion",
    "monitor",
    "protocol",
    "run_children",
    "run_monitor",
    "run_monitors",
    "run_protocol",
    "run_protocols",
    "__version__",
]
