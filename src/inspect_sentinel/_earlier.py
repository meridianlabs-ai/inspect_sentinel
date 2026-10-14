from __future__ import annotations

from typing import Literal, cast

from inspect_ai._util.registry import registry_info

from ._protocols.human import human
from ._report import Decision, Report, Reported
from ._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    EarlierDecision,
    Step,
)


def earlier_decision(
    factory: str, step: Step, step_id: str, reported: Reported[Report]
) -> EarlierDecision | None:
    """What a host keeps of a recorded report for `earlier_decisions` at the later steps of the conversation.

    Call it from `Recorder.record` with what the runner passed there, and hand the records kept for `step.conversation` to each later step of that conversation, as they were before that step began. Only a decision of `human()` is kept, so the result is None for an observation, and for a decision of any other protocol, including the layers that pass the person's decision up. When `Recorder.superseded` later reports a kept decision, its report the same object, the decision did not take effect: drop the record kept for it, so later steps do not see it.

    Args:
        factory: Registry name of the instance's factory, as passed to `Recorder.record`.
        step: The step the report was made about.
        step_id: The host's id for that step, as it records `SentinelEvent.step_id`.
        reported: The report with its instance identity.
    """
    if factory != _HUMAN or not isinstance(reported.report, Decision):
        return None
    decision = cast(Reported[Decision], reported)
    if isinstance(step, AfterGenerate):
        message = step.output.message if step.output.choices else None
        calls = (message.tool_calls if message is not None else None) or []
        return EarlierDecision(
            stage="model_output",
            step_id=step_id,
            call_ids=tuple(call.id for call in calls),
            decided=decision,
        )
    if isinstance(step, BeforeGenerate):
        return EarlierDecision(
            stage="model_input", step_id=step_id, call_ids=(), decided=decision
        )
    stage: Literal["tool_call", "tool_result"] = (
        "tool_result" if isinstance(step, AfterToolCall) else "tool_call"
    )
    return EarlierDecision(
        stage=stage, step_id=step_id, call_ids=(step.call.id,), decided=decision
    )


_HUMAN = registry_info(human).name
