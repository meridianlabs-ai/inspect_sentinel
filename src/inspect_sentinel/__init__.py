"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

from ._context import Context, EvalContext
from ._decorators import monitor, protocol
from ._excerpt import (
    Excerpt,
    drop_history,
    drop_input,
    drop_reasoning,
    drop_tool_usage,
    keep_system,
    last_messages,
    last_turns,
    new_since_last_report,
    step_as_str,
)
from ._final import decide_final
from ._host import Host, HumanAnswer
from ._portable import PortabilityError
from ._protocols import (
    concurrent,
    handle_escalation,
    human,
    observe_only,
    sequential,
    threshold,
)
from ._report import (
    Action,
    Decision,
    Failed,
    Observation,
    Report,
    Reported,
    Suspicion,
)
from ._results import Decisions, MonitorFailedError, Observations, Reports
from ._rules import (
    call_text,
    find_words,
    result_text,
    tool_matches,
)
from ._runner import run_children, run_monitors, run_protocols
from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step
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
    "AfterGenerate",
    "AfterToolCall",
    "BeforeGenerate",
    "BeforeToolCall",
    "Context",
    "Decision",
    "Decisions",
    "EvalContext",
    "Excerpt",
    "Failed",
    "Host",
    "HumanAnswer",
    "Monitor",
    "MonitorFailedError",
    "MonitorGroup",
    "Monitors",
    "Observation",
    "Observations",
    "PortabilityError",
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
    "call_text",
    "concurrent",
    "decide_final",
    "drop_history",
    "drop_input",
    "drop_reasoning",
    "drop_tool_usage",
    "find_words",
    "handle_escalation",
    "human",
    "keep_system",
    "last_messages",
    "last_turns",
    "monitor",
    "new_since_last_report",
    "observe_only",
    "protocol",
    "result_text",
    "run_children",
    "run_monitors",
    "run_protocols",
    "sequential",
    "step_as_str",
    "threshold",
    "tool_matches",
    "__version__",
]
