"""How much of a streamed reply can be let go of early: its events as they arrive, and what each is."""

from __future__ import annotations

import codecs
import copy
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias, cast

from ._wire import LINE_END_KEPT, Format, StreamEvent, UnreadableError, stream_event


class EventReader:
    """A reply stream's events, read from its bytes as they arrive.

    The events are those `written_events` reads from the whole stream, however the bytes are split: a line goes on until its line end arrives, and a carriage return at the end of what has arrived waits to see whether a line feed follows it.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._rest = ""
        self._held: list[tuple[str, str]] = []

    def push(self, data: bytes) -> list[StreamEvent]:
        """The events that the bytes complete.

        Raises:
            UnreadableError: If the stream is not UTF-8.
        """
        text = self._rest + self._decode(data, final=False)
        parts = LINE_END_KEPT.split(text)
        # the last part has no line end yet, and a line that ends in `\r` at
        # the end of the text may yet end in `\r\n`
        self._rest = parts[-1]
        lines = list(zip(parts[:-1:2], parts[1::2], strict=True))
        if lines and lines[-1][1] == "\r" and not self._rest:
            line, end = lines.pop()
            self._rest = line + end
        return self._fold(lines)

    def finish(self) -> list[StreamEvent]:
        """The events left once the stream has ended.

        Raises:
            UnreadableError: If the stream is not UTF-8.
        """
        text = self._rest + self._decode(b"", final=True)
        self._rest = ""
        parts = LINE_END_KEPT.split(text)
        events = self._fold(list(zip(parts[::2], [*parts[1::2], ""], strict=True)))
        if self._held:
            events.append(stream_event(self._held))
            self._held = []
        return events

    def _decode(self, data: bytes, *, final: bool) -> str:
        try:
            return self._decoder.decode(data, final=final)
        except UnicodeDecodeError as ex:
            raise UnreadableError("The reply stream is not UTF-8.") from ex

    def _fold(self, lines: list[tuple[str, str]]) -> list[StreamEvent]:
        # as `written_events` does
        events: list[StreamEvent] = []
        for line, end in lines:
            self._held.append((line, end))
            if not line and any(written for written, _ in self._held[:-1]):
                events.append(stream_event(self._held))
                self._held = []
        return events


Role: TypeAlias = Literal["opening", "release", "filler", "held"]
"""What an event of a reply stream is, for a reply released early.

- `opening`: the stream's opening, which goes out with the first text or reasoning.
- `release`: text or reasoning, which goes out as it arrives.
- `filler`: an event that says nothing, such as Anthropic's `ping` or an SSE comment, which goes out with what follows it.
- `held`: anything else: a tool call, or anything the sidecar doesn't release early. It and everything after it are held until the turn has ended and is judged.
"""


class Turn(Protocol):
    """Tells what each event of one provider format's reply stream is, in order, until the first that is held."""

    def place(self, event: StreamEvent) -> Role:
        """What an event is."""
        ...


def _skipped(event: StreamEvent) -> bool:
    # what every SSE reader skips: blank lines, and comments, which a provider
    # or a proxy may send as keepalives
    return all(not line or line.startswith(":") for line, _ in event.lines)


class AnthropicTurn:
    """An Anthropic Messages reply stream: text, thinking and redacted thinking blocks go out as they arrive."""

    # each block that goes out as it arrives, and the deltas it is written in
    _RELEASED: dict[str, frozenset[str]] = {
        "text": frozenset({"text_delta", "citations_delta"}),
        "thinking": frozenset({"thinking_delta", "signature_delta"}),
        "redacted_thinking": frozenset(),
    }

    def __init__(self) -> None:
        self._released: dict[int, str] = {}

    def place(self, event: StreamEvent) -> Role:
        if _skipped(event):
            return "filler"
        data = event.data or {}
        kind = data.get("type")
        if kind == "message_start":
            return "opening"
        if kind == "ping":
            return "filler"
        index = data.get("index")
        if kind == "content_block_start":
            block = cast("dict[str, Any]", data.get("content_block") or {})
            if block.get("type") in self._RELEASED and type(index) is int:
                self._released[index] = block["type"]
                return "release"
            return "held"
        block_type = self._released.get(index) if type(index) is int else None
        if block_type is None:
            return "held"
        if kind == "content_block_stop":
            return "release"
        delta = data.get("delta")
        written = (
            cast("dict[str, Any]", delta).get("type")
            if isinstance(delta, dict)
            else None
        )
        if kind == "content_block_delta" and written in self._RELEASED[block_type]:
            return "release"
        return "held"


class ChatTurn:
    """An OpenAI Chat Completions reply stream: an assistant's content and reasoning go out as they arrive."""

    # what an assistant's delta can carry that goes out as it arrives
    _RELEASED = frozenset({"role", "content", "reasoning_content", "reasoning"})

    def place(self, event: StreamEvent) -> Role:
        if _skipped(event):
            return "filler"
        data = event.data
        if data is None:
            return "held"
        choices = data.get("choices")
        if not isinstance(choices, list) or len(cast("list[object]", choices)) != 1:
            return "held"
        choice = cast("list[object]", choices)[0]
        if not isinstance(choice, dict):
            return "held"
        entry = cast("dict[str, Any]", choice)
        if entry.get("index", 0) != 0 or entry.get("finish_reason") is not None:
            return "held"
        delta = cast("dict[str, Any]", entry.get("delta") or {})
        said = {key for key, value in delta.items() if value not in (None, "", [])}
        if said - self._RELEASED:
            return "held"
        return "release" if said - {"role"} else "opening"


class ResponsesTurn:
    """An OpenAI Responses reply stream: message and reasoning items go out as they arrive."""

    # each item that goes out as it arrives, and the events it is written in
    # between its `output_item.added` and its `output_item.done`
    _RELEASED: dict[str, frozenset[str]] = {
        "message": frozenset(
            {
                "response.content_part.added",
                "response.content_part.done",
                "response.output_text.delta",
                "response.output_text.done",
                "response.output_text.annotation.added",
                "response.refusal.delta",
                "response.refusal.done",
            }
        ),
        "reasoning": frozenset(
            {
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
            }
        ),
    }

    def __init__(self) -> None:
        self._released: dict[int, str] = {}

    def place(self, event: StreamEvent) -> Role:
        if _skipped(event):
            return "filler"
        data = event.data or {}
        kind = data.get("type")
        if kind in ("response.created", "response.in_progress", "response.queued"):
            return "opening"
        index = data.get("output_index")
        item = data.get("item")
        entry = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
        if kind == "response.output_item.added":
            if entry.get("type") in self._RELEASED and type(index) is int:
                self._released[index] = entry["type"]
                return "release"
            return "held"
        item_type = self._released.get(index) if type(index) is int else None
        if item_type is None:
            return "held"
        if kind == "response.output_item.done":
            # the item done is the one that went out, not a call in its place
            return "release" if entry.get("type") == item_type else "held"
        return "release" if kind in self._RELEASED[item_type] else "held"


class ClientCheck:
    """Gives each event released early to the provider SDK's own accumulator first, as a client's SDK will take it, so that an event it can't take is never sent."""

    def __init__(self, format: Format) -> None:
        self._format = format
        self._state: Any = None
        self._extra: dict[str, Any] = {}
        if format == "openai-chat":
            from openai.lib.streaming.chat import ChatCompletionStreamState

            self._state = ChatCompletionStreamState()
        elif format == "openai-responses":
            from openai import omit
            from openai.lib.streaming.responses._responses import (
                ResponseStreamState,
            )

            self._state = ResponseStreamState(input_tools=omit, text_format=omit)
        else:
            import inspect

            from anthropic.lib.streaming import (
                _messages,  # pyright: ignore[reportPrivateUsage]
            )

            accumulate = cast(Any, _messages.accumulate_event)
            if "json_bufs" in inspect.signature(accumulate).parameters:
                self._extra = {"json_bufs": {}}

    def check(self, events: Sequence[StreamEvent]) -> None:
        """Take the next events.

        Raises:
            UnreadableError: If the SDK can't take one.
        """
        try:
            for event in events:
                if event.data is not None:
                    self._take(copy.deepcopy(event.data))
        except Exception as ex:
            # the accumulators raise errors of many kinds on events out of order
            raise UnreadableError("The reply stream doesn't hold together.") from ex

    def _take(self, data: dict[str, Any]) -> None:
        if self._format == "openai-chat":
            from openai._models import construct_type
            from openai.types.chat import ChatCompletionChunk

            if "choices" in data:
                self._state.handle_chunk(
                    construct_type(type_=ChatCompletionChunk, value=data)
                )
        elif self._format == "openai-responses":
            from openai._models import construct_type
            from openai.types.responses import ResponseStreamEvent

            self._state.handle_event(
                construct_type(type_=cast(Any, ResponseStreamEvent), value=data)
            )
        elif data.get("type") not in ("ping", "error"):
            from anthropic._models import construct_type
            from anthropic.lib.streaming import (
                _messages,  # pyright: ignore[reportPrivateUsage]
            )
            from anthropic.types import RawMessageStreamEvent

            self._state = cast(Any, _messages.accumulate_event)(
                event=construct_type(type_=RawMessageStreamEvent, value=data),
                current_snapshot=self._state,
                **self._extra,
            )
