from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from inspect_ai.tool import ToolCall

from ._report import PRECEDENCE, Decision, Reported, deciding
from ._step import AfterToolCall, BeforeToolCall, Step


def check_decision_shape(decision: Decision, step: Step) -> None:
    """Check a decision for the deterministic protocol bugs no type can express, raising `ValueError` on one.

    The action must be legal for the stage (`reject` and `modify` are not legal after a tool call), and `modified` must be set exactly when the action is `modify`, carrying a `ToolCall` before a tool call.

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
    if isinstance(step, BeforeToolCall) and decision.modified is not None:
        # a runtime check for decisions built without validation
        if not isinstance(cast(object, decision.modified), ToolCall):
            raise ValueError(
                f"A modify before a tool call must carry a ToolCall, not {type(decision.modified).__name__}."
            )


def apply_binding_floor(
    decision: Decision | None, children: Sequence[Reported[Decision]]
) -> tuple[Decision | None, Decision | None]:
    """Resolve a layer's decision against the strongest binding decision among its children, returning `(resolved, overrode)`.

    A decision that abstains, ranks below that floor, or modifies differently from a binding `modify` at it is replaced by the floor child's decision, and returned as `overrode`; binding `modify` decisions at the floor that disagree make the floor a binding `reject`. Never raises.

    Args:
        decision: The decision the layer returned, or `None` if it abstained.
        children: The layer's recorded child decisions, in the order they were collected.
    """
    binding = [child for child in deciding(children) if child.report.binding]
    if not binding:
        return decision, None
    rank = max(PRECEDENCE[child.report.action] for child in binding)
    at_floor = [child for child in binding if PRECEDENCE[child.report.action] == rank]
    floor = at_floor[0].report
    if floor.action == "modify" and any(
        child.report.modified != floor.modified for child in at_floor
    ):
        floor = Decision(
            action="reject",
            binding=True,
            explanation="binding replacements disagree: "
            + ", ".join(child.name for child in at_floor),
        )
    below = (
        decision is None
        or PRECEDENCE.get(decision.action, 0) < PRECEDENCE[floor.action]
        or (
            decision.action == "modify"
            and floor.action == "modify"
            and decision.modified != floor.modified
        )
    )
    if not below:
        return decision, None
    overridden = decision.action if decision is not None else "abstention"
    resolved = floor.model_copy(
        update={
            "explanation": f"{floor.explanation or floor.action} (overrides {overridden})"
        }
    )
    return resolved, decision
