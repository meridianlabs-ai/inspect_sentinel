from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from inspect_ai.tool import ToolCall

from ._report import PRECEDENCE, Decision, Reported, deciding
from ._step import AfterToolCall, BeforeToolCall, Step


def validate_decision(
    decision: Decision | None, step: Step, children: Sequence[Reported[Decision]] = ()
) -> None:
    """Check a layer's decision before anything acts on it, raising `ValueError` on a protocol bug.

    Three rules no type can express: the action is legal for the stage (`reject` and `modify` are not legal after a tool call); `modified` is set exactly when the action is `modify`, and carries the right type for the stage; and the decision is not weaker than any authoritative decision among the layer's own children, where a `modify` that stands on an authoritative `modify` must carry that child's replacement. A layer that abstained is subject to the third rule alone: abstaining after an authoritative child decided drops that decision.

    A layer whose children include an authoritative decision returns an authoritative decision itself; the runner sets the flag, so the floor survives every layer above.

    Args:
        decision: The decision the layer returned, or `None` if it abstained.
        step: The step it decided about.
        children: The layer's recorded child decisions, as the dispatcher collected them.
    """
    if decision is not None:
        _check_shape(decision, step)
    authoritative = [
        child for child in deciding(children) if child.report.authoritative
    ]
    if not authoritative:
        return
    floor = max(PRECEDENCE[child.report.action] for child in authoritative)
    # the alphabetically first child at the floor, so concurrent children
    # finishing in either order name the same culprit
    binding = min(
        (child for child in authoritative if PRECEDENCE[child.report.action] == floor),
        key=lambda child: child.name,
    )
    if (0 if decision is None else PRECEDENCE.get(decision.action, 0)) < floor:
        returned = "Abstaining" if decision is None else f"Decision {decision.action!r}"
        raise ValueError(
            f"{returned} is weaker than the authoritative {binding.report.action!r} from {binding.name!r}."
        )
    if (
        binding.report.action == "modify"
        and decision is not None
        and decision.action == "modify"
        and decision.modified != binding.report.modified
    ):
        raise ValueError(
            f"A modify standing on the authoritative modify from {binding.name!r} must carry that replacement."
        )


def _check_shape(decision: Decision, step: Step) -> None:
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
