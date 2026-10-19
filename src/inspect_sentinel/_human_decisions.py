from __future__ import annotations

from typing import Literal

from inspect_ai._util.registry import registry_info

from ._protocols.human import human
from ._report import Decision, Report, Reported
from ._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    HumanDecision,
    Step,
)


def kept_human_decision(
    factory: str, step: Step, step_id: str, reported: Reported[Report]
) -> HumanDecision | None:
    """What a host keeps of a recorded report for `human_decisions` at the later steps of the conversation.

    Call it from `Recorder.record` with what the runner passed there. Only a decision of `human()` is kept, so the result is None for an observation, and for a decision of any other protocol, including the layers that pass the person's decision up. The record's decision is a detached copy of the person's: it carries nothing of the step it was made at beyond its own fields.

    Hold the records kept at a step until `run_sentinel` returns for it, dropping each that `supersedes()` says a supersession withdrew, then add the rest to the records for `step.conversation`. Hand each later step of that conversation the records added before it began, so a step never sees a decision from a step still running, which may yet be superseded.

    Args:
        factory: Registry name of the instance's factory, as passed to `Recorder.record`.
        step: The step the report was made about.
        step_id: The host's id for that step, as it records `SentinelEvent.step_id`.
        reported: The report with its instance identity.
    """
    if factory != _HUMAN or not isinstance(reported.report, Decision):
        return None
    # a fresh decision: the person's own carries the runner's record of every
    # decision at the step, other protocols' scores included
    decision = Reported(
        name=reported.name,
        path=reported.path,
        report=Decision.model_validate(reported.report.model_dump()),
        function=reported.function,
    )
    if isinstance(step, AfterGenerate):
        message = step.output.message if step.output.choices else None
        calls = (message.tool_calls if message is not None else None) or []
        return HumanDecision(
            stage="model_output",
            step_id=step_id,
            call_ids=tuple(call.id for call in calls),
            decided=decision,
        )
    if isinstance(step, BeforeGenerate):
        return HumanDecision(
            stage="model_input",
            step_id=step_id,
            call_ids=(),
            decided=decision,
        )
    stage: Literal["tool_call", "tool_result"] = (
        "tool_result" if isinstance(step, AfterToolCall) else "tool_call"
    )
    return HumanDecision(
        stage=stage,
        step_id=step_id,
        call_ids=(step.call.id,),
        decided=decision,
    )


def supersedes(path: str, reported: Reported[Decision], kept: HumanDecision) -> bool:
    """Whether a decision `Recorder.superseded` reported withdraws `kept`, a record kept at the same step.

    A decision passes up only through the layers above the one that made it, as is or as a copy, so the person's decision did not take effect when a decision with its action is superseded at its own path or the path of a layer above it. A layer above that overruled the person with a different action cast an ordinary vote, and the person's decision stays. Matching by path and action, rather than by object, holds for the copies compositions pass up, and is what a replay can apply to the logged events.

    Args:
        path: The path of the context passed to `Recorder.superseded`.
        reported: The decision passed to `Recorder.superseded`.
        kept: A record `kept_human_decision()` returned at the same step.
    """
    own = kept.decided.path
    above = path == "" or own == path or own.startswith(path + "/")
    return above and kept.decided.report.action == reported.report.action


_HUMAN = registry_info(human).name
