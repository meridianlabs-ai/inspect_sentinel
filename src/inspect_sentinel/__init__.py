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
    "Host",
    "Monitor",
    "Monitors",
    "Observation",
    "Protocols",
    "Report",
    "Reported",
    "Stage",
    "Step",
    "Suspicion",
    "monitor",
    "protocol",
    "__version__",
]
