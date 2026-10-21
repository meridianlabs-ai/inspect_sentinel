from __future__ import annotations

from bisect import bisect_right
from typing import Any

from inspect_ai.event import ModelEvent, SentinelEvent, ToolEvent
from inspect_ai.scorer import Reference
from inspect_scout import Result, Transcript
from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Decision,
    Observation,
    ReplayStep,
    Report,
    Reported,
)

from ._positions import Positions
from ._steps import Conversations, stored_conversations

# the actions a step is flagged for; escalate declines to decide, so it is not
FLAGGED = ("reject", "modify", "terminate")
_PRECEDENCE = ("terminate", "reject", "modify", "escalate", "proceed")


def step_result(
    *,
    step_id: str,
    stage: str,
    label: str,
    report: Report | None,
    citation: Reference | None,
    children: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Result:
    meta: dict[str, Any] = {"step_id": step_id, "stage": stage, **metadata}
    references = [citation] if citation is not None else []
    if report is None:
        return Result(value=None, label=label, metadata=meta, references=references)
    references.extend(r for r in report.references if r not in references)
    if report.metadata is not None:
        meta["metadata"] = report.metadata
    if isinstance(report, Observation):
        return Result(
            value=_suspicion(report),
            explanation=report.explanation,
            label=label,
            metadata=meta,
            references=references,
        )
    meta["audit"] = report.audit
    if report.modified is not None:
        meta["modified"] = to_jsonable_python(report.modified)
    if children:
        meta["reports"] = children
    return Result(
        value=report.action in FLAGGED,
        answer=report.action,
        explanation=report.explanation,
        label=label,
        metadata=meta,
        references=references,
    )


def child_report(reported: Reported[Report]) -> dict[str, Any]:
    report = reported.report
    shown: dict[str, Any] = {"path": reported.path, "function": reported.function}
    if isinstance(report, Observation):
        shown["suspicion"] = _suspicion(report)
    else:
        shown["action"] = report.action
    shown["explanation"] = report.explanation
    return shown


def _suspicion(observation: Observation) -> JsonValue:
    suspicion = observation.suspicion
    return dict(suspicion) if isinstance(suspicion, dict) else suspicion


class Citations:
    """What each replayed step cites in its transcript, from indexes built once."""

    def __init__(self, transcript: Transcript, timeline: str | None = None) -> None:
        self._events = {
            e.uuid
            for e in transcript.events
            if isinstance(e, ModelEvent) and e.uuid is not None
        }
        # each tool call's tool events, by call id and conversation: the
        # conversation the steps were built with, a stored timeline's where
        # one was chosen, else the span tree's, as read mode finds them
        conversations = Conversations(transcript.events, transcript.transcript_id)
        stored = (
            stored_conversations(transcript, timeline) if timeline is not None else {}
        )
        self._tools: dict[tuple[str, str], list[ToolEvent]] = {}
        for event in transcript.events:
            if isinstance(event, ToolEvent):
                conversation = stored.get(event.uuid) or conversations.of(event.span_id)
                self._tools.setdefault((event.id, conversation), []).append(event)
        self._positions = {
            m.id: i for i, m in enumerate(transcript.messages) if m.id is not None
        }
        # generate steps a sentinel recorded, cited as read mode cites them
        self._recorded: dict[str, tuple[SentinelEvent, int]] = {}
        for index, event in enumerate(transcript.events):
            if isinstance(event, SentinelEvent) and event.stage == "model_input":
                self._recorded.setdefault(event.step_id, (event, index))
        self._spans = (
            Positions(transcript.events, transcript.transcript_id)
            if self._recorded
            else None
        )
        self._answers: dict[str, tuple[list[int], list[str]]] = {}
        for index, message in enumerate(transcript.messages):
            call_id = getattr(message, "tool_call_id", None)
            if message.role == "tool" and call_id and message.id:
                positions, ids = self._answers.setdefault(call_id, ([], []))
                positions.append(index)
                ids.append(message.id)

    def cite(self, replay_step: ReplayStep) -> Reference | None:
        step = replay_step.step
        if isinstance(step, BeforeGenerate):
            if replay_step.id in self._events:
                return Reference(type="event", id=replay_step.id)
            recorded = self._recorded.get(replay_step.id)
            if recorded is not None and self._spans is not None:
                # a log from before generate ids: the generate the step's
                # sentinel span preceded, where none overlaps it, else nothing
                return self._spans.cite(*recorded)
            last = step.input[-1].id if step.input else None
            return Reference(type="message", id=last) if last else None
        if isinstance(step, AfterGenerate):
            message = step.output.message.id if step.output.choices else None
            return Reference(type="message", id=message) if message else None
        return self._call(step)

    def _call(self, step: BeforeToolCall | AfterToolCall) -> Reference | None:
        # the call's tool message, as its tool event in the step's conversation
        # records it; else, from messages alone, the result, or the first
        # answer after the message that made the call, since a provider can
        # reuse call ids
        tools = self._tools.get((step.call.id, step.conversation), [])
        if len(tools) == 1:
            found = tools[0].message_id
            return Reference(type="message", id=found) if found else None
        if tools:
            return None
        if isinstance(step, AfterToolCall):
            return (
                Reference(type="message", id=step.result.id) if step.result.id else None
            )
        answers = self._answers.get(step.call.id)
        if answers is None:
            return None
        positions, ids = answers
        after = bisect_right(positions, self._positions.get(step.message_id, -1))
        return Reference(type="message", id=ids[after]) if after < len(ids) else None


def transcript_result(results: list[Result], label: str) -> Result:
    """The step results reduced to one: the highest suspicion, per dimension when structured, or the strongest decision, citing the steps that set it."""
    reported = [r for r in results if r.value is not None]
    if not reported:
        return Result(value=None, label=label, metadata={"steps": len(results)})
    if any(r.answer is not None for r in reported):
        return _strongest(reported, label, len(results))
    return _highest(reported, label, len(results))


def _strongest(results: list[Result], label: str, steps: int) -> Result:
    action = min(
        (r.answer for r in results if r.answer is not None), key=_PRECEDENCE.index
    )
    setting = [r for r in results if r.answer == action]
    return Result(
        value=action in FLAGGED,
        answer=action,
        explanation=setting[0].explanation,
        label=label,
        metadata=_reduced(setting, steps),
        references=_cited(setting),
    )


def _highest(results: list[Result], label: str, steps: int) -> Result:
    dimensioned = [r for r in results if isinstance(r.value, dict)]
    if dimensioned:
        best: dict[str, float] = {}
        setting: list[Result] = []
        for result in dimensioned:
            for dimension, score in _scores(result).items():
                if dimension not in best or score > best[dimension]:
                    best[dimension] = score
        # each result against the best of its own dimensions only, so sparse
        # dimensions cost what the results hold
        for result in dimensioned:
            if any(best[d] == score for d, score in _scores(result).items()):
                setting.append(result)
        value: JsonValue = dict(best)
    else:
        top = max(_score(r) for r in results)
        setting = [r for r in results if _score(r) == top]
        value = top
    return Result(
        value=value,
        explanation=setting[0].explanation,
        label=label,
        metadata=_reduced(setting, steps),
        references=_cited(setting),
    )


def _score(result: Result) -> float:
    return float(result.value)  # type: ignore[arg-type]


def _scores(result: Result) -> dict[str, float]:
    value = result.value
    assert isinstance(value, dict)
    return {k: float(v) for k, v in value.items()}  # type: ignore[arg-type]


def _reduced(setting: list[Result], steps: int) -> dict[str, Any]:
    return {
        "steps": steps,
        "set_by": [
            {"step_id": r.metadata["step_id"], "stage": r.metadata["stage"]}
            for r in setting
            if r.metadata is not None
        ],
    }


def _cited(setting: list[Result]) -> list[Reference]:
    # each step's citation once, in order, found by its fields rather than by
    # comparing it with every one kept
    cited: list[Reference] = []
    seen: set[tuple[str, str, str | None]] = set()
    for result in setting:
        for reference in result.references[:1]:
            key = (reference.type, reference.id, reference.cite)
            if key not in seen:
                seen.add(key)
                cited.append(reference)
    return cited


def decision_of(reported: Reported[Report] | None) -> Decision | None:
    if reported is not None and isinstance(reported.report, Decision):
        return reported.report
    return None
