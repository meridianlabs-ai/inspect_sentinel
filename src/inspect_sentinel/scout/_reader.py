from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from typing import Any

from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.scorer import Reference
from inspect_scout import Result, Scanner, Transcript, scanner

from inspect_sentinel import Decision, Failed, MonitorFailedError, Observation, Report

from ._results import step_result
from ._scanner import Stage

_EVENTS: list[Any] = ["model", "tool", "sentinel", "span_begin", "span_end"]


@scanner(name="read", messages="all", events=_EVENTS, condensed=True)
def as_reader(path: str, *, stage: Stage | None = None) -> Scanner[Transcript]:
    """Read what a monitor or protocol reported in the runs it was attached to, as a transcript scanner: read mode.

    No inference runs. Each `SentinelEvent` the instance at `path` recorded with status `reported`, and each of its decisions recorded `superseded`, becomes one `Result` in the transcript's resultset, mapped as `as_scanner()` maps a replayed report and citing the step's model event or message the same way, so the two modes' rows join on transcript, `step_id` and `stage`. A superseded decision's metadata says so (`status`). A step the instance abstained at recorded nothing, so it has no result; a step that left no model event or tool result cites nothing. A recorded failure makes the transcript's result an error, as in replay.

    Args:
        path: The instance's path: empty for the root protocol, else as `SentinelEvent.path` records it, such as `attempt/suspicion`.
        stage: Only the steps at this stage; all stages when None. A tool call's two steps share its id, so give a stage to join with `as_scanner()`.
    """

    async def scan(transcript: Transcript) -> list[Result]:
        positions = _Positions(transcript.events, transcript.transcript_id)
        children: dict[tuple[str, str], list[SentinelEvent]] = {}
        for event in transcript.events:
            if isinstance(event, SentinelEvent) and _below(event.path, path):
                children.setdefault((event.stage, event.step_id), []).append(event)
        results: list[Result] = []
        for index, event in enumerate(transcript.events):
            if not isinstance(event, SentinelEvent) or event.path != path:
                continue
            if stage is not None and event.stage != stage:
                continue
            if event.status == "error":
                raise MonitorFailedError(
                    [
                        Failed(
                            name=path.rsplit("/", 1)[-1] or _name(event),
                            path=path,
                            function=event.function or "",
                            error=RuntimeError(event.error),
                        )
                    ]
                )
            superseded = event.status == "superseded" and event.kind == "decision"
            if event.status != "reported" and not superseded:
                continue
            report = _report(event)
            shown = (
                [
                    _child(c)
                    for c in children.get((event.stage, event.step_id), [])
                    if c.status == "reported"
                ]
                if isinstance(report, Decision)
                else []
            )
            results.append(
                step_result(
                    step_id=event.step_id,
                    stage=event.stage,
                    label=path or _name(event),
                    report=report,
                    citation=positions.cite(event, index),
                    children=shown,
                    metadata={"status": "superseded"} if superseded else {},
                )
            )
        return results

    return scan


def _below(child: str, parent: str) -> bool:
    if child == parent:
        return False
    return parent == "" or child.startswith(parent + "/")


def _name(event: SentinelEvent) -> str:
    return event.factory.rsplit("/", 1)[-1]


def _report(event: SentinelEvent) -> Report:
    if event.kind == "observation":
        assert event.suspicion is not None
        return Observation(
            suspicion=event.suspicion,
            explanation=event.explanation,
            references=list(event.references),
            metadata=event.metadata,
        )
    assert event.action is not None
    return Decision(
        action=event.action,
        audit=event.audit,
        modified=event.modified,
        explanation=event.explanation,
        references=list(event.references),
        message=event.message,
        metadata=event.metadata,
    )


def _child(event: SentinelEvent) -> dict[str, Any]:
    shown: dict[str, Any] = {"path": event.path, "function": event.function}
    if event.kind == "observation":
        shown["suspicion"] = event.suspicion
    else:
        shown["action"] = event.action
    shown["explanation"] = event.explanation
    return shown


class _Positions:
    # where each step's model event or tool call sits relative to its sentinel
    # span, indexed in one pass so each citation is a lookup
    def __init__(self, events: list[Event], sample: str) -> None:
        self._sample = sample
        self._spans: dict[str, SpanBeginEvent] = {}
        self._begins: dict[str, int] = {}
        self._ends: dict[str, int] = {}
        self._uuids: dict[str, ModelEvent] = {}
        self._tools: dict[str, tuple[list[int], list[ToolEvent]]] = {}
        self._agents: dict[str | None, str] = {}
        # each conversation's model events, by position
        models: dict[str, tuple[list[int], list[ModelEvent]]] = {}
        # the span of each recorded model_input step, and its conversation
        inputs: dict[str, str] = {}
        for index, event in enumerate(events):
            if isinstance(event, SpanBeginEvent):
                self._spans[event.id] = event
                self._begins[event.id] = index
            elif isinstance(event, SpanEndEvent):
                self._ends[event.id] = index
            elif isinstance(event, ModelEvent):
                at, found = models.setdefault(self._agent(event.span_id), ([], []))
                at.append(index)
                found.append(event)
                if event.uuid is not None:
                    self._uuids[event.uuid] = event
            elif isinstance(event, ToolEvent):
                at, tools = self._tools.setdefault(event.id, ([], []))
                at.append(index)
                tools.append(event)
            elif (
                isinstance(event, SentinelEvent)
                and event.stage == "model_input"
                and event.span_id is not None
            ):
                inputs.setdefault(event.span_id, event.conversation)
        self._generates = _preceded(inputs, models, self._begins, self._ends)

    def cite(self, event: SentinelEvent, index: int) -> Reference | None:
        span = event.span_id
        begin = self._begins.get(span, index) if span is not None else index
        end = self._ends.get(span, index) if span is not None else index
        match event.stage:
            case "model_input":
                if event.step_id in self._uuids:
                    return Reference(type="event", id=event.step_id)
                # a log from before generate ids: the generate the span
                # preceded, where no other generate of the conversation overlaps
                found = self._generates.get(span) if span is not None else None
                return Reference(type="event", id=found) if found else None
            case "model_output":
                model = self._uuids.get(event.step_id)
                if model is None:
                    # a log from before generate ids keys the step by its output
                    return (
                        Reference(type="message", id=event.step_id)
                        if event.step_id
                        else None
                    )
                output = model.output
                message = output.message.id if output.choices else None
                return Reference(type="message", id=message) if message else None
            case "tool_call":
                tool = self._tool(event.step_id, end, after=True)
            case _:
                tool = self._tool(event.step_id, begin, after=False)
        if tool is None or tool.message_id is None:
            return None
        return Reference(type="message", id=tool.message_id)

    def _agent(self, span: str | None) -> str:
        # the conversation an event is in: the innermost agent span above it,
        # else the sample, as live; a parent_id cycle ends the walk
        found = self._agents.get(span)
        if found is not None:
            return found
        walked: list[str | None] = []
        current = span
        conversation = self._sample
        while current is not None and current not in walked:
            walked.append(current)
            if current in self._agents:
                conversation = self._agents[current]
                break
            begin = self._spans.get(current)
            if begin is None:
                break
            if begin.type == "agent":
                conversation = begin.id
                break
            current = begin.parent_id
        for visited in walked:
            self._agents[visited] = conversation
        return conversation

    def _tool(self, call_id: str, at: int, *, after: bool) -> ToolEvent | None:
        found = self._tools.get(call_id)
        if found is None:
            return None
        positions, tools = found
        if after:
            index = bisect_right(positions, at)
            return tools[index] if index < len(tools) else None
        index = bisect_left(positions, at) - 1
        return tools[index] if index >= 0 else None


def _preceded(
    inputs: dict[str, str],
    models: dict[str, tuple[list[int], list[ModelEvent]]],
    begins: dict[str, int],
    ends: dict[str, int],
) -> dict[str, str]:
    # each model_input span's generate: the conversation's next model event
    # after the span, kept only where no other model_input span of the
    # conversation runs between the span's start and that event, since
    # concurrent generates can complete out of order
    windows: dict[str, list[tuple[int, int, str, str | None]]] = {}
    for span, conversation in inputs.items():
        begin = begins.get(span)
        if begin is None:
            continue
        at, found = models.get(conversation, ([], []))
        after = bisect_right(at, ends.get(span, begin))
        model = found[after] if after < len(found) else None
        until = at[after] if model is not None else sys.maxsize
        windows.setdefault(conversation, []).append(
            (begin, until, span, model.uuid if model is not None else None)
        )
    preceded: dict[str, str] = {}
    for spans in windows.values():
        spans.sort()
        reached = -1
        for index, (begin, until, span, uuid) in enumerate(spans):
            earlier = reached >= begin
            later = index + 1 < len(spans) and spans[index + 1][0] <= until
            reached = max(reached, until)
            if uuid is not None and not earlier and not later:
                preceded[span] = uuid
    return preceded
