from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from inspect_ai.tool import ToolCall

from ._report import Action, Decision, Reported
from ._runner import PRECEDENCE
from ._step import AfterToolCall, BeforeToolCall, Step

_RANK: dict[Action, int] = {**PRECEDENCE, "escalate": 0}


def validate_decision(
    decision: Decision, step: Step, children: Sequence[Reported[Decision]] = ()
) -> None:
    """Check a layer's decision before anything acts on it, raising `ValueError` on a protocol bug.

    Three rules no type can express: the action is legal for the stage (`reject` and `modify` are not legal after a tool call); `modified` is set exactly when the action is `modify`, and carries the right type for the stage; and the decision is not weaker than any authoritative decision among the layer's own children.

    Args:
        decision: The decision the layer returned.
        step: The step it decided about.
        children: The layer's recorded child decisions, as the dispatcher collected them.
    """
    if isinstance(step, AfterToolCall) and decision.action in ("reject", "modify"):
        raise ValueError(
            f"A decision after a tool call cannot {decision.action}; the call has already run."
        )
    if (decision.action == "modify") != (decision.modified is not None):
        raise ValueError(
            "`modified` must be set exactly when the action is `modify`; "
            f"got action {decision.action!r} with modified={decision.modified!r}."
        )
    if isinstance(step, BeforeToolCall) and decision.modified is not None:
        # a runtime check for decisions built without validation
        if not isinstance(cast(object, decision.modified), ToolCall):
            raise ValueError(
                f"A modify before a tool call must carry a ToolCall, not {type(decision.modified).__name__}."
            )
    floor = max(
        (
            _RANK[child.report.action]
            for child in children
            if child.report.authoritative and child.report.action != "escalate"
        ),
        default=None,
    )
    if floor is not None and _RANK[decision.action] < floor:
        binding = next(
            child
            for child in children
            if child.report.authoritative and _RANK[child.report.action] == floor
        )
        raise ValueError(
            f"Decision {decision.action!r} is weaker than the authoritative {binding.report.action!r} from {binding.name!r}."
        )
