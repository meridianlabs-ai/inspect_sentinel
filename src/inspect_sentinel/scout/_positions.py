from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from typing import cast

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
        # how many model calls outside a sentinel's span have proposed each
        # call id so far, by the id and the output message proposing it
        proposals: dict[tuple[str, str], int] = {}
        # each run outside a sentinel's span, native or bridged, in log order:
        # its call id, its execution id, whether replay can tie it to the
        # call it ran, and whether it can build it without a check naming it
        runs: list[tuple[str, str | None, bool, bool]] = []
        # the executions some check names
        checked: set[str] = set()
        # the call ids some run records an execution under
        self._executed: set[str] = set()
        for index, event in enumerate(events):
            if isinstance(event, SpanBeginEvent):
                self._begins[event.id] = index
                outer = self._sentinel_spans.get(event.parent_id or "")
                if outer is not None:
                    self._sentinel_spans[event.id] = outer
                elif event.type == "sentinel":
                    self._sentinel_spans[event.id] = event.id
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
                    if event.span_id not in self._sentinel_spans:
                        _propose(event, proposals)
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
                    self._executed.add(event.id)
                if event.span_id not in self._sentinel_spans:
                    runs.append(
                        (
                            event.id,
                            event.execution_id,
                            _tied(event, proposals),
                            _buildable(event),
                        )
                    )
            elif isinstance(event, SentinelEvent):
                if event.execution_id is not None:
                    checked.add(event.execution_id)
                if event.stage == "model_input" and event.span_id is not None:
                    inputs.setdefault(event.span_id, event.conversation)
        self._generates = _preceded(inputs, models, self._begins, self._ends)
        self._holders = _holders(runs, checked)

    def step_id(self, event: SentinelEvent) -> str:
        # the step a check was of, as replay keys it (inspect_ai's
        # `_Driver._tool_step_id` and `_host_step_id`): the call id where the
        # check's execution is the run that holds it, else the execution id
        if event.execution_id is not None:
            if self._holders.get(event.step_id) == event.execution_id:
                return event.step_id
            # a later run of the id, a bridged tool's run under an id of its
            # own, or an execution that never ran
            return event.execution_id
        if event.step_id in self._executed:
            # in a log that records executions, the bridge checks a result
            # under an id it handed out more than once under none: which run it
            # answers is unknown, so its key is the id of the sentinel span it
            # was reported in, which every report of the check shares and no
            # replayed step has (every event such a log records has an id)
            if event.span_id is not None and event.span_id in self._sentinel_spans:
                return self._sentinel_spans[event.span_id]
            return event.uuid or event.step_id
        return event.step_id

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


def _holders(
    runs: list[tuple[str, str | None, bool, bool]], checked: set[str]
) -> dict[str, str | None]:
    # each call id's holder, as replay names steps (inspect_ai's
    # `_Driver._tool_step_id`, which `_host_step_id` shares): the execution of
    # the first run of the id, native or bridged, in log order, whose steps
    # replay builds, in any conversation, as step ids are unique in a sample;
    # None for a run that records no execution. Replay builds the steps of a
    # run it can tie to its call (`_tied`) where a check names its execution,
    # else, by what the events show, where `_buildable` says. A run whose
    # steps replay skips for reasons the events don't show (a native call no
    # model call proposed, a native or host run in a region that checks no
    # tool calls (`_checks_tool_calls`, which #6446's scope guard applies to
    # host runs), a host run whose check replay pairs with no run) still
    # counts as built here, so a later run's checks are keyed by their
    # execution, which joins no replay row rather than the wrong one
    holders: dict[str, str | None] = {}
    for call_id, execution, tied, buildable in runs:
        if tied and (buildable or execution in checked):
            holders.setdefault(call_id, execution)
    return holders


def _propose(event: ModelEvent, proposals: dict[tuple[str, str], int]) -> None:
    # the calls a model call outside a sentinel's span proposed, as replay
    # indexes them (inspect_ai's `_Proposals.add`, for a model call
    # `producing_message_id` names an output of)
    message = event.output.message
    if event.role == "monitor" or event.pending or message.id is None:
        return
    for call in message.tool_calls or []:
        key = (call.id, message.id)
        proposals[key] = proposals.get(key, 0) + 1


def _tied(event: ToolEvent, proposals: dict[tuple[str, str], int]) -> bool:
    # whether replay can tie a run to the call it ran: a native call always
    # (as before); a sandbox agent bridge's host call only to the one
    # proposal before it of the model's id for the call (`proposal_id`) in
    # the output message the bridge granted the run to (`call_message_id`),
    # as `_Proposals.exact` finds it for `_links_host_call`, so not where a
    # cache hit returned that message again, or a log has no message ids
    if event.metadata is None or "bridge" not in event.metadata:
        return True
    bridge: object = event.metadata["bridge"]
    proposal_id = (
        cast(dict[str, object], bridge).get("proposal_id")
        if isinstance(bridge, dict)
        else None
    )
    if not isinstance(proposal_id, str) or event.call_message_id is None:
        return False
    return proposals.get((proposal_id, event.call_message_id)) == 1


def _buildable(event: ToolEvent) -> bool:
    # whether replay can build a run's steps with no check naming it: a native
    # call unless its error is of a call rejected before its step was built
    # ("parsing" or "approval"); a sandbox agent bridge's host call
    # (`metadata["bridge"]`, inspect_ai's `_bridged`) where the bridge linked
    # the run to the result the scaffold reported (`message_id`,
    # `_links_host_call`)
    if event.metadata is not None and "bridge" in event.metadata:
        return event.message_id is not None
    return event.error is None or event.error.type not in ("parsing", "approval")


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
