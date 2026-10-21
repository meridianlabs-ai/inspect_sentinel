"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

from ._context import Context, EvalContext, InspectSample, ProxyContext
from ._decorators import monitor, protocol
from ._final import decide_final
from ._from_messages import steps_from_messages
from ._host import Host, HumanAnswer
from ._portable import PortabilityError
from ._preprocess import (
    AllMessages,
    LastMessages,
    LastTurns,
    NewSinceLastReport,
    StepPreprocessor,
    Window,
    preprocess_step,
    step_as_str,
)
from ._protocols import (
    concurrent,
    handle_escalation,
    human,
    observe_only,
    sequential,
    threshold,
)
from ._replay import (
    HumanPolicy,
    HumanSource,
    HumanStandIn,
    ReplayReport,
    Split,
    replay_monitor,
)
from ._replay_step import ReplayStep
from ._report import (
    Action,
    Decision,
    Escalations,
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
from ._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    HumanDecision,
    Step,
    Truncation,
    human_decision_for,
)
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
    "Escalations",
    "EvalContext",
    "Failed",
    "Host",
    "HumanAnswer",
    "HumanDecision",
    "HumanPolicy",
    "HumanSource",
    "HumanStandIn",
    "InspectSample",
    "Monitor",
    "MonitorFailedError",
    "MonitorGroup",
    "Monitors",
    "Observation",
    "Observations",
    "PortabilityError",
    "Protocol",
    "ProtocolGroup",
    "ProxyContext",
    "Protocols",
    "Report",
    "ReplayReport",
    "ReplayStep",
    "Reported",
    "Reports",
    "Sentinel",
    "Sentinels",
    "Split",
    "Step",
    "Suspicion",
    "Truncation",
    "call_text",
    "concurrent",
    "decide_final",
    "find_words",
    "handle_escalation",
    "human",
    "human_decision_for",
    "monitor",
    "observe_only",
    "protocol",
    "replay_monitor",
    "result_text",
    "run_children",
    "run_monitors",
    "run_protocols",
    "sequential",
    "step_as_str",
    "steps_from_messages",
    "AllMessages",
    "LastMessages",
    "LastTurns",
    "NewSinceLastReport",
    "StepPreprocessor",
    "Window",
    "preprocess_step",
    "threshold",
    "tool_matches",
    "__version__",
]
