"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

from ._context import Context, Host
from ._decorators import monitor, protocol
from ._final import decide_final
from ._protocols import concurrent, observe, sequential, threshold
from ._report import Action, Decision, Observation, Report, Reported, Suspicion
from ._results import Decisions, Observations, Reports
from ._runner import run_children, run_monitors, run_protocols
from ._step import AfterToolCall, BeforeToolCall, Step
from ._types import (
    Monitor,
    MonitorGroup,
    Monitors,
    Protocol,
    ProtocolGroup,
    Protocols,
    Sentinel,
    Sentinels,
)

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "Action",
    "AfterToolCall",
    "BeforeToolCall",
    "Context",
    "Decision",
    "Decisions",
    "Host",
    "Monitor",
    "MonitorGroup",
    "Monitors",
    "Observation",
    "Observations",
    "Protocol",
    "ProtocolGroup",
    "Protocols",
    "Report",
    "Reported",
    "Reports",
    "Sentinel",
    "Sentinels",
    "Step",
    "Suspicion",
    "concurrent",
    "decide_final",
    "monitor",
    "observe",
    "protocol",
    "run_children",
    "run_monitors",
    "run_protocols",
    "sequential",
    "threshold",
    "__version__",
]
