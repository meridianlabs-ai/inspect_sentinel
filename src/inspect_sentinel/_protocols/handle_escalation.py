from __future__ import annotations

from typing import Literal

from .._context import Context
from .._decorators import answers_escalations, protocol
from .._report import Decision, Escalations
from .._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall
from .._types import ProtocolGroup

_ACTIONS = ("terminate", "proceed")


@protocol
def handle_escalation(action: Literal["terminate", "proceed"]) -> ProtocolGroup:
    """Answer the escalations handed to it with a fixed policy: end the sample, or let the step proceed.

    Put it last in a `sequential()`, after the links that escalate what they are unsure about, as in `sequential([threshold(...), handle_escalation("terminate")])`, or after a `human()` for when the person cannot be reached. It decides only when it is handed escalations and abstains otherwise, so at the root, or anywhere nothing escalated, it does nothing. Its explanation names each escalation it handled, `escalated by triage: call deletes files (rm)`, and it carries their references.

    An escalate that reaches the root with nothing to handle it ends the sample, `run_sentinel` returning a `terminate` explained as an unhandled escalation; `handle_escalation()` is how a configuration states a fixed policy instead. `"proceed"` lets an unsure escalation through, recorded in the configuration and the log. It runs before and after tool calls and the agent's model calls, but never makes a stage watched on its own: a stage runs only when something else in the configuration watches it.

    Args:
        action: What to do when the step was escalated: `"terminate"` to end the sample, or `"proceed"` to let the step proceed. Required, so every configuration states its policy.
    """
    if action not in _ACTIONS:
        raise ValueError(
            f"handle_escalation's action must be {' or '.join(repr(a) for a in _ACTIONS)}; got {action!r}."
        )

    async def tool_call(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return _handle(action, escalations)

    async def tool_result(
        context: Context,
        step: AfterToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return _handle(action, escalations)

    async def model_input(
        context: Context,
        step: BeforeGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return _handle(action, escalations)

    async def model_output(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return _handle(action, escalations)

    return ProtocolGroup(
        *(
            answers_escalations(function)
            for function in (tool_call, tool_result, model_input, model_output)
        )
    )


def _handle(
    action: Literal["terminate", "proceed"], escalations: Escalations
) -> Decision | None:
    if not escalations:
        return None
    explanation = "escalated by " + "; ".join(
        f"{e.name}: {e.report.explanation}" if e.report.explanation else e.name
        for e in escalations
    )
    references = [r for e in escalations for r in e.report.references]
    if action == "terminate":
        return Decision.terminate(explanation, references=references)
    return Decision.proceed(explanation, references=references)
