"""The monitor model's requests at each step, live and in replay, rendered for comparison.

Live, a step's monitor calls are the `ModelEvent`s inside its `sentinel` span, and the `SentinelEvent` there names the step. In a replay through Scout, the scan's events are flat; `every_step_marked`, a step selector that selects every step, records an `InfoEvent` naming each step just before the step runs, so a step's calls are the model events after its mark. The replay runs one step at a time, so the marks cannot interleave.

A request is rendered as text with `Host.score()`'s nonce replaced, since the nonce is fresh in every transcript by design, and without message ids, which are minted for each call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tiktoken
from inspect_ai.event import Event, ModelEvent, SentinelEvent, SpanBeginEvent
from inspect_ai.log import EvalSample, transcript
from inspect_ai.model import ChatMessage, ChatMessageAssistant

from inspect_sentinel import ReplayStep
from inspect_sentinel.scout import step_selector

StepId = tuple[str, str]
"""A step as a transcript keys it: its stage and step id."""

MARK = "replay_trust_step"
"""The `InfoEvent` source of the marks `every_step_marked` records."""

# a nonce on a block marker or message header, as Host.score renders it
_NONCE = re.compile(r"\(([0-9a-f]{8})\)\]")

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
    """The request rendered for comparison: what the model is sent, configuration, tools and messages, with the nonce replaced, no message ids, and none of the settings that govern only the client, such as connections and timeouts."""

    tokens: int
    """The request's input tokens, counted with `o200k_base`: each message's text and the tools' JSON."""

    used_input: int = 0
    """The input tokens the provider reported, including any read from or written to its cache; 0 where it reported none."""

    used_output: int = 0
    """The output tokens the provider reported."""


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
    """`event`'s request, rendered for comparison and counted."""
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
        lines.extend(_message(index, message))
    text = "\n".join(lines)
    for nonce in set(_NONCE.findall(text)):
        text = text.replace(nonce, "NONCE")
    tokens = sum(len(_ENCODING.encode(m.text)) for m in event.input) + sum(
        len(_ENCODING.encode(json.dumps(tool.model_dump(exclude_none=True))))
        for tool in event.tools
    )
    usage = event.output.usage
    return Request(
        text=text,
        tokens=tokens,
        used_input=(
            usage.input_tokens
            + (usage.input_tokens_cache_read or 0)
            + (usage.input_tokens_cache_write or 0)
        )
        if usage
        else 0,
        used_output=usage.output_tokens if usage else 0,
    )


def _message(index: int, message: ChatMessage) -> list[str]:
    header = message.model_dump(
        exclude={"id", "content", "tool_calls", "source"}, exclude_none=True
    )
    lines = [f"--- message {index}: {json.dumps(header, sort_keys=True)}"]
    lines.extend(message.text.splitlines())
    if isinstance(message, ChatMessageAssistant):
        for call in message.tool_calls or []:
            lines.append(
                f"tool call: {json.dumps({'id': call.id, 'function': call.function, 'arguments': call.arguments}, sort_keys=True)}"
            )
    return lines


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else value


def sample_key(sample: EvalSample) -> str:
    """A sample across runs of one eval: its id, and its epoch where there are several."""
    return f"{sample.id}" if sample.epoch == 1 else f"{sample.id}#{sample.epoch}"
