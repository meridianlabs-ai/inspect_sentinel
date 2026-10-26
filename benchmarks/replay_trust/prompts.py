"""The monitor model's requests at each step, live and in replay, rendered for comparison.

Live, a step's monitor calls are the `ModelEvent`s inside its `sentinel` span, and the `SentinelEvent` there names the step. In a replay through Scout, the scan's events are flat; `every_step_marked`, a step selector that selects every step, records an `InfoEvent` naming each step just before the step runs, so a step's calls are the model events after its mark. The replay runs one step at a time, so the marks cannot interleave.

A request is rendered as text with `Host.score()`'s nonce replaced on its block markers and message headers, since the nonce is fresh in every transcript by design, and without message ids, which are minted for each call. The rendering compared holds every message's whole content: its text as lines, so a difference reads as a line diff, and every other field of every part, images, audio and documents with their data, as canonical JSON. The rendering shown shortens long values in those parts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import tiktoken
from inspect_ai.event import Event, ModelEvent, SentinelEvent, SpanBeginEvent
from inspect_ai.log import EvalSample, transcript
from inspect_ai.model import ChatMessage, ChatMessageAssistant, Content, ModelUsage

from inspect_sentinel import ReplayStep
from inspect_sentinel.scout import step_selector

StepId = tuple[str, str]
"""A step as a transcript keys it: its stage and step id."""

MARK = "replay_trust_step"
"""The `InfoEvent` source of the marks `every_step_marked` records."""

# a nonce on a block marker, as Host.score renders it; once found, it is
# replaced wherever it is in parentheses, as on every marker and header
_NONCE = re.compile(r"\(([0-9a-f]{8})\)\]")

# a value in a non-text content part longer than this is shortened when shown
_SHOWN_LENGTH = 120

# generation settings that govern the client, not what the model is sent;
# a scan sets its own connections
_CLIENT_SIDE = {
    "max_retries",
    "timeout",
    "attempt_timeout",
    "stream_idle_timeout",
    "max_connections",
    "adaptive_connections",
}

_ENCODING = tiktoken.get_encoding("o200k_base")


@dataclass(frozen=True)
class Request:
    """One request to the monitor's model."""

    text: str
    """The request rendered for comparison: what the model is sent, configuration, tools and messages with their whole content, with the nonce replaced, no message ids, and none of the settings that govern only the client, such as connections and timeouts."""

    tokens: int
    """The request's input tokens, counted with `o200k_base`: each message's text and the tools' JSON."""

    usage: ModelUsage | None = None
    """What the provider reported the call used, its uncached input, cache reads and cache writes apart; None where it reported nothing."""

    shown: str | None = None
    """The request rendered for display where that differs from `text`: long values in content that is not text are shortened."""

    @property
    def display(self) -> str:
        """The request rendered for display."""
        return self.text if self.shown is None else self.shown


@step_selector
def every_step_marked(transcript_id: str, step: ReplayStep) -> bool:
    """Select every step, recording a mark naming it in the scan's events first."""
    transcript().info({"stage": step.stage, "step_id": step.id}, source=MARK)
    return True


def live_requests(events: Sequence[Event]) -> dict[StepId, list[Request]]:
    """Each step's monitor requests in a sample's events: the model events inside its `sentinel` span."""
    parents: dict[str, str | None] = {}
    sentinel_spans: set[str] = set()
    for event in events:
        if isinstance(event, SpanBeginEvent):
            parents[event.id] = event.parent_id
            if event.type == "sentinel":
                sentinel_spans.add(event.id)

    def sentinel_span(span_id: str | None) -> str | None:
        while span_id is not None:
            if span_id in sentinel_spans:
                return span_id
            span_id = parents.get(span_id)
        return None

    steps: dict[str, StepId] = {}
    calls: dict[str, list[Request]] = {}
    for event in events:
        span = sentinel_span(event.span_id)
        if span is None:
            continue
        if isinstance(event, SentinelEvent):
            steps[span] = (event.stage, event.step_id)
        elif isinstance(event, ModelEvent):
            calls.setdefault(span, []).append(request(event))
    return {steps[span]: found for span, found in calls.items() if span in steps}


def replay_requests(events: Iterable[Mapping[str, Any]]) -> dict[StepId, list[Request]]:
    """Each step's monitor requests in a replay scan's events for one transcript: the model events after the step's mark."""
    found: dict[StepId, list[Request]] = {}
    step: StepId | None = None
    for event in events:
        if event.get("event") == "info" and event.get("source") == MARK:
            data = event["data"]
            step = (data["stage"], data["step_id"])
        elif event.get("event") == "model" and step is not None:
            found.setdefault(step, []).append(request(ModelEvent.model_validate(event)))
    return found


def request(event: ModelEvent) -> Request:
    """`event`'s request, rendered for comparison and for display, and counted."""
    text = _render(event, shorten=False)
    shown = _render(event, shorten=True)
    tokens = sum(len(_ENCODING.encode(m.text)) for m in event.input) + sum(
        len(_ENCODING.encode(json.dumps(tool.model_dump(exclude_none=True))))
        for tool in event.tools
    )
    return Request(
        text=text,
        tokens=tokens,
        usage=event.output.usage,
        shown=None if shown == text else shown,
    )


def _render(event: ModelEvent, *, shorten: bool) -> str:
    config = event.config.model_dump(exclude_none=True, exclude=_CLIENT_SIDE)
    lines = [
        f"config: {json.dumps(config, sort_keys=True)}",
        f"tool_choice: {json.dumps(_jsonable(event.tool_choice), sort_keys=True)}",
        "tools:",
        *[
            json.dumps(tool.model_dump(exclude_none=True), sort_keys=True)
            for tool in event.tools
        ],
    ]
    for index, message in enumerate(event.input):
        lines.extend(_message(index, message, shorten))
    text = "\n".join(lines)
    for nonce in set(_NONCE.findall(text)):
        text = text.replace(f"({nonce})", "(NONCE)")
    return text


def _message(index: int, message: ChatMessage, shorten: bool) -> list[str]:
    header = message.model_dump(
        exclude={"id", "content", "tool_calls", "source"}, exclude_none=True
    )
    lines = [f"--- message {index}: {json.dumps(header, sort_keys=True)}"]
    if isinstance(message.content, str):
        lines.extend(message.content.splitlines())
    else:
        for part, content in enumerate(message.content):
            lines.extend(_part(part, content, shorten))
    if isinstance(message, ChatMessageAssistant):
        for call in message.tool_calls or []:
            lines.append(
                f"tool call: {json.dumps({'id': call.id, 'function': call.function, 'arguments': call.arguments}, sort_keys=True)}"
            )
    return lines


def _part(index: int, content: Content, shorten: bool) -> list[str]:
    # a part as canonical JSON, with a text part's text after it as lines
    fields = content.model_dump(exclude_none=True)
    text = fields.pop("text", None) if content.type == "text" else None
    if shorten:
        fields = _shortened(fields)
    lines = [f"content {index}: {json.dumps(fields, sort_keys=True)}"]
    if isinstance(text, str):
        lines.extend(text.splitlines())
    return lines


def _shortened(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _SHOWN_LENGTH:
        digest = hashlib.sha256(value.encode()).hexdigest()[:12]
        return f"<{len(value)} characters, sha256 {digest}>"
    if isinstance(value, dict):
        return {k: _shortened(v) for k, v in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_shortened(v) for v in cast(list[Any], value)]
    return value


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else value


def sample_key(sample: EvalSample) -> str:
    """A sample across runs of one eval: its id, and its epoch where there are several."""
    return f"{sample.id}" if sample.epoch == 1 else f"{sample.id}#{sample.epoch}"
