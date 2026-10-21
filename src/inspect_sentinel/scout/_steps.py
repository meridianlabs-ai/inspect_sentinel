from __future__ import annotations

from typing import Literal, TypeAlias

from inspect_ai._sentinel._integration import EvalInfo, steps_from_events
from inspect_ai.event import ModelEvent, SpanBeginEvent, Timeline, TimelineSpan
from inspect_scout import Transcript

from inspect_sentinel import ReplayStep, steps_from_messages

Turns: TypeAlias = Literal["recorded", "inferred", "stored"]

_AGENT = "agent"


def transcript_steps(
    transcript: Transcript, eval_info: EvalInfo, timeline: str | None
) -> tuple[list[ReplayStep], Turns]:
    # A log that records where each model call was made replays from its
    # events, the driver inspect_ai shares with live. One that does not, an
    # older Inspect log or another source, replays from its messages, each
    # assistant message taken for an agent turn, until turns can be inferred
    # from its timeline (R7 PR 6).
    if not any(
        isinstance(e, ModelEvent) and e.origin is not None for e in transcript.events
    ):
        if timeline is not None:
            raise NotImplementedError(
                f"as_scanner(timeline={timeline!r}) needs a log that records where each model call was made (ModelEvent.origin); replaying older logs by their timeline comes later."
            )
        steps = steps_from_messages(
            transcript.messages, conversation=transcript.transcript_id
        )
        return list(steps), "inferred"
    stored = _timeline(transcript, timeline) if timeline is not None else None
    data = transcript.events_data
    steps = steps_from_events(
        transcript.messages,
        transcript.events,
        data["messages"] if data is not None else [],
        sample_uuid=transcript.transcript_id,
        eval_info=eval_info,
        timeline=stored,
    )
    return steps, "stored" if stored is not None else "recorded"


def conversation_depths(transcript: Transcript, timeline: str | None) -> dict[str, int]:
    # how deep each conversation is, as llm_scanner counts depth: 1 for the
    # sample's own and top-level agents, one more for each agent above
    if timeline is not None:
        depths: dict[str, int] = {}

        def walk(span: TimelineSpan, depth: int) -> None:
            depths[span.id] = depth
            for item in span.content:
                if isinstance(item, TimelineSpan):
                    walk(item, depth + 1 if item.span_type == _AGENT else depth)
            for branch in span.branches:
                walk(branch, depth)

        walk(_timeline(transcript, timeline).root, 1)
        return depths
    spans = {e.id: e for e in transcript.events if isinstance(e, SpanBeginEvent)}
    found: dict[str, int] = {transcript.transcript_id: 1}
    for span in spans.values():
        if span.type != _AGENT:
            continue
        depth = 0
        seen: set[str] = set()
        current: SpanBeginEvent | None = span
        while current is not None and current.id not in seen:
            seen.add(current.id)
            if current.type == _AGENT:
                depth += 1
            current = spans.get(current.parent_id) if current.parent_id else None
        found[span.id] = depth
    return found


def _timeline(transcript: Transcript, name: str) -> Timeline:
    for stored in transcript.timelines:
        if stored.name == name:
            return stored
    raise ValueError(
        f"Transcript {transcript.transcript_id!r} has no timeline named {name!r}; it has {[t.name for t in transcript.timelines]}."
    )
