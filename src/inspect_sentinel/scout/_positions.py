from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right

from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.scorer import Reference

from ._steps import Conversations


class Positions:
    # where each step's model event or tool call sits relative to its sentinel
    # span, indexed in one pass so each citation is a lookup
    def __init__(self, events: list[Event], sample: str) -> None:
        self._begins: dict[str, int] = {}
        self._ends: dict[str, int] = {}
        self._uuids: dict[str, ModelEvent] = {}
        # each generate's last attempt, by its first attempt's uuid, and every
        # output message a model event holds
        self._final: dict[str, ModelEvent] = {}
        self._outputs: set[str] = set()
        retrying: dict[str | None, tuple[str, ModelEvent]] = {}
        # each conversation's tool events, by call id and position: two agents'
        # calls can share an id
        self._tools: dict[tuple[str, str], tuple[list[int], list[ToolEvent]]] = {}
        self._conversations = Conversations(events, sample)
        # each conversation's model events, by position
        models: dict[str, tuple[list[int], list[ModelEvent]]] = {}
        # the span of each recorded model_input step, and its conversation
        inputs: dict[str, str] = {}
        for index, event in enumerate(events):
            if isinstance(event, SpanBeginEvent):
                self._begins[event.id] = index
            elif isinstance(event, SpanEndEvent):
                self._ends[event.id] = index
            elif isinstance(event, ModelEvent):
                at, found = models.setdefault(
                    self._conversations.of(event.span_id), ([], [])
                )
                at.append(index)
                found.append(event)
                if event.uuid is not None:
                    self._uuids[event.uuid] = event
                self._attempt(event, retrying)
                if event.output.choices and event.output.message.id:
                    self._outputs.add(event.output.message.id)
            elif isinstance(event, ToolEvent):
                at, tools = self._tools.setdefault(
                    (self._conversations.of(event.span_id), event.id), ([], [])
                )
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
                first = self._uuids.get(event.step_id)
                if first is not None:
                    # the output is the generate's last attempt's
                    output = self._final.get(event.step_id, first).output
                    message = output.message.id if output.choices else None
                elif event.step_id in self._outputs:
                    # a log from before generate ids keys the step by its output
                    message = event.step_id
                else:
                    message = None
                if message:
                    return Reference(type="message", id=message)
                # an output with no message: its generate's event
                return Reference(type="event", id=event.step_id) if first else None
            case "tool_call":
                tool = self._tool(event, end, after=True)
            case _:
                tool = self._tool(event, begin, after=False)
        if tool is None:
            return None
        # the call's tool message, else, where none was recorded, its tool event
        if tool.message_id:
            return Reference(type="message", id=tool.message_id)
        return Reference(type="event", id=tool.uuid) if tool.uuid else None

    def _attempt(
        self, event: ModelEvent, retrying: dict[str | None, tuple[str, ModelEvent]]
    ) -> None:
        # a generate's attempts: errored attempts followed by one with the same
        # input, in the same span ("Which steps ran")
        previous = retrying.pop(event.span_id, None)
        if previous is not None and _same_input(previous[1], event):
            first = previous[0]
        elif event.uuid is not None:
            first = event.uuid
        else:
            return
        self._final[first] = event
        if event.error is not None:
            retrying[event.span_id] = (first, event)

    def _tool(self, event: SentinelEvent, at: int, *, after: bool) -> ToolEvent | None:
        found = self._tools.get((event.conversation, event.step_id))
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


def _same_input(earlier: ModelEvent, later: ModelEvent) -> bool:
    return earlier.input == later.input
