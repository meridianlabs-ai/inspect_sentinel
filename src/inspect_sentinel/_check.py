from __future__ import annotations

from ._report import Decision
from ._step import AfterToolCall, Step


def check_decision_shape(decision: Decision, step: Step) -> None:
    """Check a decision for the deterministic protocol bugs no type can express, raising `ValueError` on one.

    The action must be legal for the stage (`reject` and `modify` are not legal after a tool call), and `modified` must be set exactly when the action is `modify`.

    Args:
        decision: The decision a layer returned.
        step: The step it decided about.
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
