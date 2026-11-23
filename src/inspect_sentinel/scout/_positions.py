from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from collections import Counter

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

# the name of the sentinel span an agent bridge records its checks in
# (inspect_ai's `BRIDGE_CHECK_SPAN`)
_BRIDGE_CHECK_SPAN = "sentinel_bridge"


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
        # the tool events of each execution a tool event records, by position
        self._executions: dict[str, tuple[list[int], list[ToolEvent]]] = {}
        self._conversations = Conversations(events, sample)
        # each conversation's model events, by position
        models: dict[str, tuple[list[int], list[ModelEvent]]] = {}
        # the span of each recorded model_input step, and its conversation
        inputs: dict[str, str] = {}
        # the outermost sentinel span holding each span within one, whose
        # model and tool events are a monitor's own
        self._sentinel_spans: dict[str, str] = {}
        # the spans within one an agent bridge recorded its checks in
        self._bridge_spans: set[str] = set()
        # each native run outside a sentinel's span, in log order: its call id,
        # its execution id, and whether its error left it unbuilt in replay
        runs: list[tuple[str, str | None, bool]] = []
        # the executions some check names
        checked: set[str] = set()
        # each conversation's model calls an agent bridge made, by the id of
        # each call in their output and position: the call's step id in
        # replay, or None where the output holds the id more than once
        proposals: dict[tuple[str, str], tuple[list[int], list[str | None]]] = {}
        # each bridge check's span, with the replayed step it was of, if any
        joins: dict[str, tuple[str, str, str | None]] = {}
        for index, event in enumerate(events):
            if isinstance(event, SpanBeginEvent):
                self._begins[event.id] = index
                outer = self._sentinel_spans.get(event.parent_id or "")
                if outer is not None:
                    self._sentinel_spans[event.id] = outer
                elif event.type == "sentinel":
                    self._sentinel_spans[event.id] = event.id
                if (
                    event.type == "sentinel" and event.name == _BRIDGE_CHECK_SPAN
                ) or event.parent_id in self._bridge_spans:
                    self._bridge_spans.add(event.id)
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
                if event.requested_model is not None and event.output.choices:
                    _propose(
                        event, index, self._conversations.of(event.span_id), proposals
                    )
            elif isinstance(event, ToolEvent):
                at, tools = self._tools.setdefault(
                    (self._conversations.of(event.span_id), event.id), ([], [])
                )
                at.append(index)
                tools.append(event)
                if event.execution_id is not None:
                    at, tools = self._executions.setdefault(
                        event.execution_id, ([], [])
                    )
                    at.append(index)
                    tools.append(event)
                if event.span_id not in self._sentinel_spans and not _bridged(event):
                    unbuilt = event.error is not None and event.error.type in (
                        "parsing",
                        "approval",
                    )
                    runs.append((event.id, event.execution_id, unbuilt))
            elif isinstance(event, SentinelEvent):
                if event.execution_id is not None:
                    checked.add(event.execution_id)
                span = event.span_id
                if (
                    span is not None
                    and span in self._bridge_spans
                    and self._sentinel_spans[span] not in joins
                ):
                    joins[self._sentinel_spans[span]] = (
                        event.conversation,
                        event.stage,
                        _proposed(proposals, event, index),
                    )
                if event.stage == "model_input" and event.span_id is not None:
                    inputs.setdefault(event.span_id, event.conversation)
        self._generates = _preceded(inputs, models, self._begins, self._ends)
        self._renamed = _renamed(runs, checked)
        self._joined = _one_to_one(joins)

    def step_id(self, event: SentinelEvent) -> str:
        # the step a check was of, as replay keys it: the call id, except for a
        # later native execution of a call id an earlier execution's steps took
        # (inspect_ai's `_Driver._tool_step_id`), named by its execution id
        span = event.span_id
        if span is not None and span in self._bridge_spans:
            # an agent bridge's check, of a call it handed to the agent or ran
            # on the host: replay names its step by model call, so its key is
            # never the bare call id, which a native call's steps can hold. It is the
            # execution the check names, else, for a result checked under an id
            # handed out more than once, the id of the sentinel span it was
            # reported in, which every report of the check shares
            return event.execution_id or self._sentinel_spans[span]
        if event.execution_id is not None and event.execution_id in self._renamed:
            return event.execution_id
        return event.step_id

    def replay_step_id(self, event: SentinelEvent) -> str | None:
        # the replayed step a check joins: a native check's own step; an agent
        # bridge's check, the step replay names by the model call that proposed
        # the call and the call id, where that is one step only this check is
        # of, else None
        span = event.span_id
        if span is not None and span in self._bridge_spans:
            return self._joined.get(self._sentinel_spans[span])
        return self.step_id(event)

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
        if event.execution_id is not None:
            runs = self._executions.get(event.execution_id)
            if runs is not None:
                # the runs of the execution the check judged: a native call's
                # one run, or a bridged tool's runs for one handout, which can
                # run under an id other than the call's
                found = runs
            elif found is not None and any(t.execution_id for t in found[1]):
                # the runs under the id are other executions': the check's own
                # never ran (a bridged tool of the handout that it did not use,
                # or a native call whose execution has no tool event)
                return None
        if found is None:
            return None
        positions, tools = found
        if after:
            index = bisect_right(positions, at)
            return tools[index] if index < len(tools) else None
        index = bisect_left(positions, at) - 1
        return tools[index] if index >= 0 else None


def _renamed(runs: list[tuple[str, str | None, bool]], checked: set[str]) -> set[str]:
    # the executions replay names by their execution ids: each native run that
    # replay builds steps for, after the first built under its call id, in any
    # conversation, as step ids are unique in a sample. Replay builds a run's
    # steps where a check names it, else where it ran: an error of type
    # "parsing" or "approval" is of a call rejected before its step was built.
    # A run whose steps replay skips for reasons the events don't show (one no
    # model call proposed, one in a region that checks no tool calls) still
    # counts as built here, and so does a checked run replay skips with a
    # warning, whose checks then keep the call id replay gives a later run
    holders: dict[str, str | None] = {}
    renamed: set[str] = set()
    for call_id, execution, unbuilt in runs:
        if unbuilt and execution not in checked:
            continue
        holder = holders.setdefault(call_id, execution)
        if execution is not None and holder != execution:
            renamed.add(execution)
    return renamed


def _propose(
    event: ModelEvent,
    index: int,
    conversation: str,
    proposals: dict[tuple[str, str], tuple[list[int], list[str | None]]],
) -> None:
    # the calls a bridged model call proposed, each under replay's step id for
    # it (inspect_ai's `_BridgedCall.step_id`), or None where the output holds
    # the id more than once, which replay builds no steps for. Every bridged
    # model call counts, including one replay builds no steps for (an output
    # with no message id, or one made where calls are not checked): a check
    # it proposed then names a step no replayed row has, and is listed
    # unmatched, where skipping it would pair the check with an earlier
    # proposal of the id
    calls = Counter(call.id for call in event.output.message.tool_calls or [])
    for call_id, count in calls.items():
        at, steps = proposals.setdefault((conversation, call_id), ([], []))
        at.append(index)
        unique = count == 1 and event.uuid is not None
        steps.append(f"{event.uuid}/{call_id}" if unique else None)


def _proposed(
    proposals: dict[tuple[str, str], tuple[list[int], list[str | None]]],
    event: SentinelEvent,
    index: int,
) -> str | None:
    # a bridge check's step: that of the latest bridged model call of its
    # conversation before it that proposed the call id
    found = proposals.get((event.conversation, event.step_id))
    if found is None or event.stage not in ("tool_call", "tool_result"):
        return None
    at, steps = found
    before = bisect_left(at, index) - 1
    return steps[before] if before >= 0 else None


def _one_to_one(joins: dict[str, tuple[str, str, str | None]]) -> dict[str, str]:
    # each bridge check's step, where no other check is of it at that stage:
    # the bridge can check one call more than once (a result reported again
    # under a reused id), and position alone cannot say which check is whose
    claims = Counter(joins.values())
    return {
        span: join[2]
        for span, join in joins.items()
        if join[2] is not None and claims[join] == 1
    }


def _bridged(event: ToolEvent) -> bool:
    # a host call a sandbox agent bridge ran, whose steps replay builds from
    # the bridge's model call, not from this event (inspect_ai's `_bridged`)
    return event.metadata is not None and "bridge" in event.metadata


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
