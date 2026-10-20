"""Where a streamed reply can be let go of early: its events as they arrive, and what each is part of."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from ._wire import LINE_END_KEPT, StreamEvent, UnreadableError, stream_event


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


Role: TypeAlias = Literal["opening", "release", "call", "hold", "closing", "filler"]
"""What an event of a reply stream is part of.

- `opening`: the stream's opening, which goes out with the first content released.
- `release`: text or reasoning, which goes out as it arrives.
- `call`: a call the client runs, held until it is complete and judged.
- `hold`: anything else in the turn, held until the turn has ended.
- `closing`: the end of the turn, which goes out once the turn is decided.
- `filler`: an event that says nothing, such as Anthropic's `ping`, which goes out in its place once something has been sent.
"""


@dataclass(frozen=True)
class Placement:
    """Where an event belongs in the turn."""

    role: Role
    item: object = None
    """The block, call or item the event is part of, by its place in the turn, if it is part of one."""
    completes: tuple[object, ...] = ()
    """The calls this event completes, as their last event."""
    completed_before: tuple[object, ...] = ()
    """The calls that were complete before this event, which shows they are: a Chat Completions call, once a later call starts."""


class Turn(Protocol):
    """Places each event of one provider format's reply stream in its turn, in order."""

    def place(self, event: StreamEvent) -> Placement:
        """Where an event belongs. An event of blank lines alone is `filler`.

        Raises:
            UnreadableError: If the event changes a call after the call was complete, or names a block or call it can't.
        """
        ...


def _blank(event: StreamEvent) -> bool:
    return not any(line for line, _ in event.lines)


class AnthropicTurn:
    """An Anthropic Messages reply stream. A call is complete at its block's `content_block_stop`."""

    _RELEASED = frozenset({"text", "thinking", "redacted_thinking"})

    def __init__(self) -> None:
        self._roles: dict[int, Role] = {}
        self._stopped: set[int] = set()

    def place(self, event: StreamEvent) -> Placement:
        if _blank(event):
            return Placement("filler")
        data = event.data
        kind = (data or {}).get("type")
        if kind == "message_start":
            return Placement("opening")
        if kind == "ping":
            return Placement("filler")
        if kind in ("message_delta", "message_stop"):
            return Placement("closing")
        if not isinstance(kind, str) or not kind.startswith("content_block_"):
            # an error from the provider, or an event no reader knows
            return Placement("hold")
        assert data is not None
        index = data.get("index")
        if type(index) is not int:
            raise UnreadableError(
                "The reply stream names a block by an index it doesn't have."
            )
        if index in self._stopped:
            raise UnreadableError("The reply stream changes a block after it stopped.")
        if kind == "content_block_start":
            block = cast("dict[str, Any]", data.get("content_block") or {})
            block_kind = block.get("type")
            self._roles[index] = (
                "release"
                if block_kind in self._RELEASED
                else "call"
                if block_kind == "tool_use"
                else "hold"
            )
        role = self._roles.get(index)
        if role is None:
            raise UnreadableError(
                "The reply stream names a block by an index it doesn't have."
            )
        if kind == "content_block_stop":
            self._stopped.add(index)
            if role == "call":
                return Placement(role, index, completes=(index,))
        return Placement(role, index)


class ChatTurn:
    """An OpenAI Chat Completions reply stream.

    The format marks no end for a single call. As the OpenAI SDK does, a call is taken as complete when a call at a later index starts, or when the choice finishes, and a fragment for it after that makes the stream unreadable.
    """

    # what an assistant's delta can carry that goes out as it arrives
    _RELEASED = frozenset({"role", "content", "reasoning_content", "reasoning"})

    def __init__(self) -> None:
        self._open: list[int] = []
        self._complete: set[int] = set()

    def place(self, event: StreamEvent) -> Placement:
        if _blank(event):
            return Placement("filler")
        data = event.data
        if data is None:
            # `[DONE]`, or data that isn't JSON
            return Placement("closing" if _done(event) else "hold")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            # the usage, after the choice has finished, or an error
            return Placement("closing" if "usage" in data else "hold")
        listed = cast("list[object]", choices)
        if len(listed) > 1 or not isinstance(listed[0], dict):
            return Placement("hold")
        choice = cast("dict[str, Any]", listed[0])
        if choice.get("index", 0) != 0:
            return Placement("hold")
        delta = cast("dict[str, Any]", choice.get("delta") or {})
        fragments: object = delta.get("tool_calls") or []
        indexes: list[int] = []
        for fragment in (
            cast("list[object]", fragments) if isinstance(fragments, list) else []
        ):
            index = (
                cast("dict[str, Any]", fragment).get("index")
                if isinstance(fragment, dict)
                else None
            )
            if type(index) is not int:
                raise UnreadableError("A call in the reply stream has no index.")
            if index in self._complete:
                raise UnreadableError(
                    "The reply stream changes a call after it was complete."
                )
            indexes.append(index)
        before: list[int] = []
        with_this: list[int] = []
        if indexes:
            started = [index for index in indexes if index not in self._open]
            if started:
                latest = max(started)
                # a call at a later index completes those before it: with this
                # event if it holds their last fragment, before it if not
                for index in [i for i in self._open if i < latest]:
                    (with_this if index in indexes else before).append(index)
                self._open = [i for i in self._open if i >= latest]
                self._open += [i for i in started if i not in self._open]
        if choice.get("finish_reason") is not None:
            with_this += [i for i in self._open if i not in with_this]
            self._open = []
        self._complete.update(before, with_this)
        completes, completed_before = tuple(with_this), tuple(before)
        if choice.get("finish_reason") is not None:
            return Placement("closing", None, completes, completed_before)
        if indexes:
            return Placement("call", max(indexes), completes, completed_before)
        said = {key for key, value in delta.items() if value not in (None, "", [])}
        if not said - self._RELEASED:
            return Placement(
                "release" if said - {"role"} else "opening",
                None,
                completes,
                completed_before,
            )
        return Placement("hold", None, completes, completed_before)


class ResponsesTurn:
    """An OpenAI Responses reply stream. A call is complete at the `response.output_item.done` event at its output index."""

    def __init__(self) -> None:
        self._roles: dict[int, Role] = {}
        self._done: set[int] = set()

    def place(self, event: StreamEvent) -> Placement:
        if _blank(event):
            return Placement("filler")
        data = event.data
        kind = (data or {}).get("type")
        if kind in ("response.created", "response.in_progress", "response.queued"):
            return Placement("opening")
        if kind in ("response.completed", "response.incomplete"):
            return Placement("closing")
        index = (data or {}).get("output_index")
        if index is None:
            # a failure, an error, or an event no reader knows
            return Placement("hold")
        if type(index) is not int or index < 0:
            raise UnreadableError(
                "The reply stream has an event for an item without its place in the output."
            )
        if index in self._done:
            raise UnreadableError("The reply stream changes an item after it was done.")
        assert data is not None
        if kind in ("response.output_item.added", "response.output_item.done"):
            item = data.get("item")
            self._roles.setdefault(index, _responses_role(item))
        role = self._roles.get(index)
        if role is None:
            raise UnreadableError(
                "The reply stream has an event for an item it hasn't added."
            )
        if kind == "response.output_item.done":
            self._done.add(index)
            if role == "call":
                return Placement(role, index, completes=(index,))
        return Placement(role, index)


def _done(event: StreamEvent) -> bool:
    return any(
        line.startswith("data:") and line[5:].strip() == "[DONE]"
        for line, _ in event.lines
    )


def _responses_role(item: object) -> Role:
    entry = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
    kind = str(entry.get("type", ""))
    if kind in ("message", "reasoning"):
        return "release"
    # a call the client runs; a call's output, and a call the provider ran
    # itself, which has no `call_id`, are held to the end of the turn
    if isinstance(entry.get("call_id"), str) and not kind.endswith("_output"):
        return "call"
    return "hold"
