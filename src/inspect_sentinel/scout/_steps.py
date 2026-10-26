from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Literal, TypeAlias

# inspect_ai's designated boundary for sentinel, not inspect_ai's public API
from inspect_ai._sentinel._integration import EvalInfo, steps_from_events
from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    Timeline,
    TimelineEvent,
    TimelineSpan,
    timeline_build,
)
from inspect_ai.log import condense_events
from inspect_scout import Transcript

from inspect_sentinel import (
    BeforeGenerate,
    HumanAnswer,
    ReplayStep,
)

# sentinel's own messages projection, internal since 2026-10-26: the adapter
# is its only caller
from inspect_sentinel._from_messages import steps_from_messages

Turns: TypeAlias = Literal["recorded", "inferred", "stored"]

_AGENT = "agent"
_HUMAN = "inspect_sentinel/human"


def transcript_steps(
    transcript: Transcript, eval_info: EvalInfo, timeline: str | None
) -> tuple[list[ReplayStep], Turns]:
    # A log that records where each model call was made replays from its
    # events, the driver inspect_ai shares with live. One that does not, an
    # older Inspect log or another source, replays from its messages, each
    # assistant message taken for an agent turn, until turns can be inferred
    # from its timeline (R7 PR 6). Each step gets its conversation's depth in
    # the chosen timeline.
    stored = _timeline(transcript, timeline) if timeline is not None else None
    if not _origins(transcript):
        if stored is not None:
            raise NotImplementedError(
                f"as_scanner(timeline={timeline!r}) needs a log that records where each model call was made (ModelEvent.origin); replaying older logs by their timeline comes later."
            )
        steps = list(
            steps_from_messages(
                transcript.messages, conversation=transcript.transcript_id
            )
        )
        recorded = [e for e in transcript.events if isinstance(e, SentinelEvent)]
        if recorded:
            steps = _as_recorded(steps, recorded)
        turns: Turns = "inferred"
    else:
        # Scout hands every model input expanded, a list of messages it shares
        # between inputs; the driver reads inputs as ranges of a pool, so pool
        # them once here. Condensing keeps each event's uuid, which citations
        # and a stored timeline find events by.
        events, data = condense_events(transcript.events)
        steps = steps_from_events(
            transcript.messages,
            events,
            data["messages"],
            sample_uuid=transcript.transcript_id,
            eval_info=eval_info,
            timeline=stored,
        )
        turns = "stored" if stored is not None else "recorded"
    depths = _depths(transcript, stored)
    return [replace(s, depth=depths.get(s.step.conversation)) for s in steps], turns


def _depths(transcript: Transcript, stored: Timeline | None) -> dict[str, int]:
    # each conversation's depth: the shallowest level Scout's walk, as
    # llm_scanner(depth=) walks, gives a span holding one of its model calls,
    # in the timeline the conversations come from: the stored one chosen,
    # else one built from the span tree, whatever the transcript stores
    tree = stored if stored is not None else timeline_build(transcript.events)
    levels = {
        item.event.uuid: level
        for span, level in _walk_spans(tree.root, None)
        for item in span.content
        if isinstance(item, TimelineEvent)
        and isinstance(item.event, ModelEvent)
        and item.event.uuid is not None
    }
    if not _origins(transcript):
        # the messages projection's steps are all in the transcript's own
        conversations: dict[str | None, str] = dict.fromkeys(
            levels, transcript.transcript_id
        )
    elif stored is not None:
        conversations = _stored(stored)
    else:
        spans = Conversations(transcript.events, transcript.transcript_id)
        conversations = {
            e.uuid: spans.of(e.span_id)
            for e in transcript.events
            if isinstance(e, ModelEvent) and e.uuid is not None
        }
    depths: dict[str, int] = {}
    for uuid, level in levels.items():
        conversation = conversations.get(uuid)
        if conversation is not None:
            depths[conversation] = min(depths.get(conversation, level), level)
    return depths


class Conversations:
    # the conversation each event is in, from the span tree: the innermost
    # agent span above it, else the sample, as live; a parent_id cycle ends
    # the walk

    def __init__(self, events: Sequence[Event], sample: str) -> None:
        self._sample = sample
        self._spans = {e.id: e for e in events if isinstance(e, SpanBeginEvent)}
        self._found: dict[str | None, str] = {}

    def of(self, span: str | None) -> str:
        found = self._found.get(span)
        if found is not None:
            return found
        walked: list[str | None] = []
        current = span
        conversation = self._sample
        while current is not None and current not in walked:
            walked.append(current)
            if current in self._found:
                conversation = self._found[current]
                break
            begin = self._spans.get(current)
            if begin is None:
                break
            if begin.type == _AGENT:
                conversation = begin.id
                break
            current = begin.parent_id
        for visited in walked:
            self._found[visited] = conversation
        return conversation


def _walk_spans(
    span: TimelineSpan, depth: int | None, scannable_depth: int = 0
) -> Iterator[tuple[TimelineSpan, int]]:
    # mirrors Scout's private `_walk_spans`, so `depth` counts levels as
    # `llm_scanner(depth=)` does: a span is scannable when it is not a utility
    # span and holds a model event of its own, and only scannable spans count
    # a level; others are walked through. Each span comes with its level. A
    # test pins it to Scout's.
    if depth is not None and depth <= 0:
        return
    if not span.utility and any(
        isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
        for item in span.content
    ):
        scannable_depth += 1
        if depth is not None and scannable_depth > depth:
            return
        yield span, scannable_depth
    for item in span.content:
        if isinstance(item, TimelineSpan):
            yield from _walk_spans(item, depth, scannable_depth)


def _origins(transcript: Transcript) -> bool:
    return any(
        isinstance(e, ModelEvent) and e.origin is not None for e in transcript.events
    )


def _as_recorded(
    steps: list[ReplayStep], recorded: list[SentinelEvent]
) -> list[ReplayStep]:
    # A log from before recorded origins that ran a sentinel: key each
    # generate's model_input step as live did then, by its last input
    # message's id with a count on repeats, so its rows join read mode's, and
    # give each step the human() answers recorded at it.
    answers: dict[tuple[str, str], dict[str, HumanAnswer]] = {}
    for event in recorded:
        if (
            event.factory == _HUMAN
            and event.kind == "decision"
            and event.status == "reported"
            and event.action is not None
        ):
            answers.setdefault((event.stage, event.step_id), {})[event.path] = (
                HumanAnswer(
                    decision="approve" if event.action == "proceed" else event.action,
                    reason=event.explanation,
                    modified=event.modified,
                )
            )
    seen = Counter[str]()
    keyed: list[ReplayStep] = []
    for step in steps:
        step_id = step.id
        if isinstance(step.step, BeforeGenerate):
            last = (step.step.input[-1].id if step.step.input else None) or ""
            seen[last] += 1
            step_id = last if seen[last] == 1 else f"{last}:{seen[last]}"
        keyed.append(
            replace(
                step,
                id=step_id,
                recorded_answers=answers.get((step.stage, step_id), {}),
            )
        )
    return keyed


def stored_conversations(transcript: Transcript, name: str) -> dict[str | None, str]:
    # each event's conversation in the stored timeline of that name, by uuid
    return _stored(_timeline(transcript, name))


def _stored(timeline: Timeline) -> dict[str | None, str]:
    # each model event's conversation in a stored timeline: its innermost
    # agent span or branch, else the root, as inspect_ai's driver reads it
    found: dict[str | None, str] = {}

    def walk(span: TimelineSpan, conversation: str) -> None:
        for item in span.content:
            if isinstance(item, TimelineEvent):
                found[item.event.uuid] = conversation
            else:
                walk(item, item.id if item.span_type == _AGENT else conversation)
        for branch in span.branches:
            walk(branch, branch.id)

    walk(timeline.root, timeline.root.id)
    return found


def _timeline(transcript: Transcript, name: str) -> Timeline:
    for stored in transcript.timelines:
        if stored.name == name:
            return stored
    raise ValueError(
        f"Transcript {transcript.transcript_id!r} has no timeline named {name!r}; it has {[t.name for t in transcript.timelines]}."
    )
