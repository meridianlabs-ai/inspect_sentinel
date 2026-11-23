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
from inspect_ai.model import ChatMessageTool
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
        # each call in their output (`_Proposals`)
        proposals: dict[tuple[str, str], _Proposals] = {}
        # each bridge check's span, with its conversation, stage, call id and
        # the proposal it was of, if one is
        picks: dict[str, tuple[str, str, str, int | None]] = {}
        # each conversation's last bridge call check, by call id: a proposal
        # before it was another check's
        last_checks: dict[tuple[str, str], int] = {}
        # the top-level phase each span is in (`init`, `solvers`, `scorers`)
        phases: dict[str, str | None] = {}
        # the native runs outside a sentinel's span: by the message their
        # call came from, where they record one, else by call id and position
        native: set[tuple[str, str]] = set()
        unsourced: list[tuple[str, int]] = []
        # each conversation's calls its bridged model calls proposed under
        # each call id, with one more for an id its starting history holds
        proposed: Counter[tuple[str, str]] = Counter()
        # each conversation's starting history ids, until it first proposes
        history: dict[str, set[str]] = {}
        proposing: set[str] = set()
        for index, event in enumerate(events):
            if isinstance(event, SpanBeginEvent):
                self._begins[event.id] = index
                phases[event.id] = (
                    event.name
                    if event.parent_id is None
                    and event.name in ("init", "solvers", "scorers")
                    else phases.get(event.parent_id or "")
                )
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
                if (
                    event.requested_model is not None
                    and event.span_id not in self._sentinel_spans
                ):
                    _count(
                        event,
                        self._conversations.of(event.span_id),
                        proposed,
                        history,
                        proposing,
                    )
                if event.requested_model is not None and event.output.choices:
                    _propose(
                        event,
                        index,
                        self._conversations.of(event.span_id),
                        proposals,
                        builds=_builds(
                            event,
                            phases.get(event.span_id or ""),
                            event.span_id in self._sentinel_spans,
                        ),
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
                    if event.call_message_id is not None:
                        native.add((event.call_message_id, event.id))
                    else:
                        unsourced.append((event.id, index))
            elif isinstance(event, SentinelEvent):
                if event.execution_id is not None:
                    checked.add(event.execution_id)
                span = event.span_id
                if (
                    span is not None
                    and span in self._bridge_spans
                    and self._sentinel_spans[span] not in picks
                ):
                    picks[self._sentinel_spans[span]] = (
                        event.conversation,
                        event.stage,
                        event.step_id,
                        _proposed(proposals, last_checks, event, index),
                    )
                if event.stage == "model_input" and event.span_id is not None:
                    inputs.setdefault(event.span_id, event.conversation)
        self._generates = _preceded(inputs, models, self._begins, self._ends)
        self._renamed = _renamed(runs, checked)
        _drop_native(proposals, native, unsourced)
        joins = {
            span: (
                conversation,
                stage,
                proposals[(conversation, call_id)].steps[pick]
                if pick is not None
                else None,
            )
            for span, (conversation, stage, call_id, pick) in picks.items()
        }
        self._joined = _one_to_one(_answerable(joins, proposed))

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
        # the call and the call id, where replay built that step and it is one
        # step only this check is of, else None
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


class _Proposals:
    # one conversation's bridged proposals of one call id, in log order: each
    # proposing model call's position, its output message id, and the step
    # replay builds for the call, or None where it builds none
    def __init__(self) -> None:
        self.at: list[int] = []
        self.messages: list[str | None] = []
        self.steps: list[str | None] = []


# the origins whose model calls' tool calls a sentinel checks (inspect_ai's
# `_NON_FINAL`)
_CHECKED_ORIGINS = ("agent", "tool", "setup")


def _builds(event: ModelEvent, phase: str | None, in_sentinel: bool) -> bool:
    # whether replay builds tool steps for the calls a bridged model call made
    # (inspect_ai's `_bridged_tool_steps`): one that produced a message for its
    # caller (`producing_message_id`: not a monitor's, not pending, with an
    # output message id), in a region whose calls are checked
    # (`_checks_calls_of`: by origin, else outside the scorers' phase and any
    # sentinel span), and whose recorded agent ids, if any, match its calls
    if (
        event.role == "monitor"
        or event.pending
        or not event.output.choices
        or not event.output.message.id
    ):
        return False
    if event.origin is not None:
        checked = event.origin in _CHECKED_ORIGINS
    else:
        checked = phase != "scorers" and not in_sentinel
    calls = event.output.message.tool_calls or []
    recorded = event.agent_tool_call_ids
    return checked and (recorded is None or len(recorded) == len(calls))


def _propose(
    event: ModelEvent,
    index: int,
    conversation: str,
    proposals: dict[tuple[str, str], _Proposals],
    builds: bool,
) -> None:
    # the calls a bridged model call proposed, each under replay's step id for
    # it (inspect_ai's `_BridgedCall.step_id`), or None where replay builds no
    # step for it: the model call is one replay builds none for (`_builds`),
    # the output holds the id more than once, or the bridge handed the call
    # over under another id, which names its steps. Every bridged model call
    # counts, so a check never pairs with an earlier proposal past one replay
    # built no step for
    calls = event.output.message.tool_calls or []
    recorded = event.agent_tool_call_ids
    received = recorded if recorded is not None else [call.id for call in calls]
    counts = Counter(call.id for call in calls)
    handed = Counter(received)
    for call, agent_id in zip(calls, received, strict=False):
        found = proposals.setdefault((conversation, call.id), _Proposals())
        if found.at and found.at[-1] == index:
            continue
        found.at.append(index)
        found.messages.append(event.output.message.id)
        unique = (
            builds
            and counts[call.id] == 1
            and handed[agent_id] == 1
            and agent_id == call.id
            and event.uuid is not None
        )
        found.steps.append(f"{event.uuid}/{call.id}" if unique else None)
    for call in calls[len(received) :]:
        found = proposals.setdefault((conversation, call.id), _Proposals())
        if not (found.at and found.at[-1] == index):
            found.at.append(index)
            found.messages.append(event.output.message.id)
            found.steps.append(None)


def _proposed(
    proposals: dict[tuple[str, str], _Proposals],
    last_checks: dict[tuple[str, str], int],
    event: SentinelEvent,
    index: int,
) -> int | None:
    # the proposal a bridge check was of. A result check's is the latest
    # proposal of the id in its conversation before it. A call check's is the
    # only one since the conversation's last call check of the id: the bridge
    # checks each call it hands over once, so an earlier proposal was another
    # check's, and where two are since, as when a request filter generated
    # candidates and handed one over, the log does not say which, so none
    key = (event.conversation, event.step_id)
    found = proposals.get(key)
    if event.stage == "tool_call":
        since = last_checks.get(key, -1)
        last_checks[key] = index
        if found is None:
            return None
        first = bisect_right(found.at, since)
        last = bisect_left(found.at, index)
        return first if last - first == 1 else None
    if found is None or event.stage != "tool_result":
        return None
    before = bisect_left(found.at, index) - 1
    return before if before >= 0 else None


def _drop_native(
    proposals: dict[tuple[str, str], _Proposals],
    native: set[tuple[str, str]],
    unsourced: list[tuple[str, int]],
) -> None:
    # a call a native `ToolEvent` ran keeps its native steps, and replay builds
    # no bridged step for it: by the message its call came from, else, for a
    # run that records none, every earlier bridged proposal of its id, since
    # the log does not say which it ran
    for (_, call_id), found in proposals.items():
        for at, message in enumerate(found.messages):
            if message is not None and (message, call_id) in native:
                found.steps[at] = None
    for call_id, index in unsourced:
        for (_, proposed), found in proposals.items():
            if proposed == call_id:
                for at in range(bisect_left(found.at, index)):
                    found.steps[at] = None


def _count(
    event: ModelEvent,
    conversation: str,
    proposed: Counter[tuple[str, str]],
    history: dict[str, set[str]],
    proposing: set[str],
) -> None:
    # the calls a conversation's bridged model calls proposed under each id,
    # counted as inspect_ai's replay counts them (`_bridged_tool_steps`):
    # every call each output holds, by the model's id, and each tool message
    # id the conversation's starting history holds, once. The starting
    # history is the input of its first bridged request and of its first
    # that proposed a call, as a side request with its own history can come
    # first. The bridge hands a call under a starting history id over under a
    # new one, which no proposal holds, so its checks name no step anyway; the
    # history counts where a log's bridge kept the model's id
    calls = (
        event.output.message.tool_calls or []
        if event.output.choices
        and event.output.message.id
        and event.role != "monitor"
        and not event.pending
        else []
    )
    if conversation not in proposing and (conversation not in history or calls):
        held = history.setdefault(conversation, set())
        for message in event.input:
            if isinstance(message, ChatMessageTool) and message.tool_call_id:
                if message.tool_call_id not in held:
                    held.add(message.tool_call_id)
                    proposed[(conversation, message.tool_call_id)] += 1
    if calls:
        proposing.add(conversation)
    for call in calls:
        proposed[(conversation, call.id)] += 1


def _answerable(
    joins: dict[str, tuple[str, str, str | None]],
    proposed: Counter[tuple[str, str]],
) -> dict[str, tuple[str, str, str | None]]:
    # a bridge's result check names no step where its conversation proposed
    # the call's id more than once: replay builds no result step for such an
    # id, as the log does not record which call a result under it answers
    return {
        span: (conversation, stage, None)
        if stage == "tool_result"
        and step is not None
        and proposed[(conversation, step.split("/", 1)[1])] > 1
        else (conversation, stage, step)
        for span, (conversation, stage, step) in joins.items()
    }


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
