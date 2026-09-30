from __future__ import annotations

from ._report import Decision
from ._step import AfterToolCall, Step


def validate_decision_shape(decision: Decision, step: Step) -> None:
    """Check a decision for the deterministic protocol bugs no type can express, raising `ValueError` on one.

    The action must be legal for the stage (`reject` and `modify` are not legal after a tool call), `modified` must be set exactly when the action is `modify`, a `modify` may rewrite only the call's arguments (its `id` and `function` must match the step's call), and `message` may be set only on a `reject`.

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
    modified = decision.modified
    if modified is not None and (
        modified.id != step.call.id or modified.function != step.call.function
    ):
        raise ValueError(
            "A `modify` may change only the call's arguments; "
            f"got id {modified.id!r} and function {modified.function!r} "
            f"for call {step.call.id!r} to {step.call.function!r}."
        )
    if decision.message is not None and decision.action != "reject":
        raise ValueError(
            f"`message` may be set only on a `reject`; got action {decision.action!r}. "
            "Telling the agent something while the step proceeds needs a way to deliver it, which does not exist yet."
        )
