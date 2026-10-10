from __future__ import annotations

from typing import Literal

from .._context import Context
from .._decorators import protocol
from .._escalation import STAGES, declare, handles
from .._report import Decision
from .._step import AfterToolCall, BeforeToolCall, Step
from .._types import ProtocolGroup

_ACTIONS = ("terminate", "continue")


@protocol
def handle_escalation(action: Literal["terminate", "continue"]) -> ProtocolGroup:
    """Answer the escalations handed to it with a fixed policy: end the sample, or let the step proceed.

    Put it last in a `sequential()`, after the links that escalate what they are unsure about, as in `sequential([threshold(...), handle_escalation("terminate")])`, or after a `human()` for when the person cannot be reached. It decides only when `step.escalations` is non-empty and abstains otherwise, so at the root, or anywhere nothing escalated, it does nothing. Its explanation names each escalation it handled, `escalated by triage: call deletes files (rm)`, and it carries their references.

    An escalate that reaches the root with nothing to handle it ends the sample as an unhandled escalation; `handle_escalation()` is how a configuration states a fixed policy instead. `"continue"` lets an unsure escalation through, recorded in the configuration and the log. It runs before and after tool calls.

    Args:
        action: What to do when the step was escalated: `"terminate"` to end the sample, or `"continue"` to let the step proceed. Required, so every configuration states its policy.
    """
    if action not in _ACTIONS:
        raise ValueError(
            f"handle_escalation's action must be {' or '.join(repr(a) for a in _ACTIONS)}; got {action!r}."
        )

    async def tool_call(context: Context, step: BeforeToolCall) -> Decision | None:
        return _handle(action, step)

    async def tool_result(context: Context, step: AfterToolCall) -> Decision | None:
        return _handle(action, step)

    return declare(ProtocolGroup(tool_call, tool_result), handles(STAGES.values()))


def _handle(action: Literal["terminate", "continue"], step: Step) -> Decision | None:
    if not step.escalations:
        return None
    explanation = "escalated by " + "; ".join(
        f"{e.name}: {e.report.explanation}" if e.report.explanation else e.name
        for e in step.escalations
    )
    references = [r for e in step.escalations for r in e.report.references]
    if action == "terminate":
        return Decision.terminate(explanation, references=references)
    return Decision.proceed(explanation, references=references)
