"""Where a streamed reply can be let go of early: its events as they arrive, and what each is part of."""

from __future__ import annotations

import codecs
import copy
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from ._wire import (
    LINE_END_KEPT,
    Format,
    StreamEvent,
    UnreadableError,
    stream_event,
    write_events,
)


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

    @property
    def open(self) -> frozenset[object]:
        """The calls that have started and are not complete yet, by their place in the turn."""
        ...

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

    @property
    def open(self) -> frozenset[object]:
        return frozenset(
            index
            for index, role in self._roles.items()
            if role == "call" and index not in self._stopped
        )

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

    @property
    def open(self) -> frozenset[object]:
        return frozenset(self._open)

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

    @property
    def open(self) -> frozenset[object]:
        return frozenset(
            index
            for index, role in self._roles.items()
            if role == "call" and index not in self._done
        )

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


def closed_prefix(
    format: Format, events: Sequence[StreamEvent], open_calls: Collection[object]
) -> bytes:
    """A reply stream's events so far, as the provider sent them, closed as if the reply ended there, for the calls complete in it to be read as they would be at the end of the turn.

    Args:
        format: Which API the stream is written in.
        events: The stream's events so far.
        open_calls: The calls that are not complete yet, by their place in the turn, which are left out.
    """
    if format == "anthropic-messages":
        # blocks come one after another, so none is open once a call stops
        closing = [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
            {"type": "message_stop"},
        ]
        return write_events([*events, *(_made(data) for data in closing)])
    if format == "openai-chat":
        return write_events(_chat_closed(events, open_calls))
    return write_events([*events, _made(_responses_completed(events))])


def _made(data: dict[str, Any]) -> StreamEvent:
    end = "\n"
    kind = data.get("type")
    lines = [(f"event: {kind}", end)] if isinstance(kind, str) else []
    return StreamEvent([*lines, (f"data: {json.dumps(data)}", end), ("", end)], data)


def _chat_closed(
    events: Sequence[StreamEvent], open_calls: Collection[object]
) -> list[StreamEvent]:
    closed: list[StreamEvent] = []
    first: dict[str, Any] | None = None
    finished = False
    for event in events:
        data = event.data
        if data is None or not isinstance(data.get("choices"), list):
            closed.append(event)
            continue
        first = first or data
        choices = cast("list[dict[str, Any]]", data["choices"])
        finished = finished or any(
            choice.get("finish_reason") is not None for choice in choices
        )
        fragments = [
            fragment
            for choice in choices
            for fragment in cast(
                "list[dict[str, Any]]",
                cast("dict[str, Any]", choice.get("delta") or {}).get("tool_calls")
                or [],
            )
        ]
        if not any(fragment.get("index") in open_calls for fragment in fragments):
            closed.append(event)
            continue
        # without the fragments of calls still arriving
        data = copy.deepcopy(data)
        for choice in cast("list[dict[str, Any]]", data["choices"]):
            delta = cast("dict[str, Any]", choice.get("delta") or {})
            kept = [
                fragment
                for fragment in cast(
                    "list[dict[str, Any]]", delta.get("tool_calls") or []
                )
                if fragment.get("index") not in open_calls
            ]
            if kept:
                delta["tool_calls"] = kept
            else:
                delta.pop("tool_calls", None)
        closed.append(_made(data))
    if not finished:
        head = first or {}
        closed.append(
            _made(
                {
                    "id": head.get("id"),
                    "object": "chat.completion.chunk",
                    "created": head.get("created", 0),
                    "model": head.get("model", ""),
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                }
            )
        )
    return closed


def _responses_completed(events: Sequence[StreamEvent]) -> dict[str, Any]:
    # the response the stream began with, holding the items done so far
    response: dict[str, Any] = {}
    done: dict[int, object] = {}
    sequence = 0
    for event in events:
        data = event.data or {}
        number = data.get("sequence_number")
        if isinstance(number, int):
            sequence = max(sequence, number)
        if data.get("type") in ("response.created", "response.in_progress"):
            begun = data.get("response")
            if isinstance(begun, dict):
                response = cast("dict[str, Any]", begun)
        index = data.get("output_index")
        if data.get("type") == "response.output_item.done" and isinstance(index, int):
            done[index] = data.get("item")
    return {
        "type": "response.completed",
        "sequence_number": sequence + 1,
        "response": {
            **response,
            "status": "completed",
            "output": [done[index] for index in sorted(done)],
        },
    }


class ClientCheck:
    """Gives each event that goes out to the provider SDK's own accumulator first, as a client's SDK will take it, so that an event it can't take is never sent."""

    def __init__(self, format: Format) -> None:
        self._format = format
        self._state: Any = None
        self._extra: dict[str, Any] = {}
        # a Responses stream's events so far, and each item's events by its
        # place in the output, as the stream as a whole is checked when held
        self._seen: list[StreamEvent] = []
        self._items: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        self._others: dict[int, list[dict[str, Any]]] = {}
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
                if self._format == "openai-responses":
                    self._agree(event)
        except UnreadableError:
            raise
        except Exception as ex:
            # the accumulators raise errors of many kinds on events out of order
            raise UnreadableError(
                "The reply stream as edited doesn't hold together."
            ) from ex

    def _agree(self, event: StreamEvent) -> None:
        # A Responses call is given whole in its done item, which is what is
        # judged, and also in its argument deltas and its arguments-done
        # event, which a client may act on instead. All of them must agree
        # before the call goes out, and every item must be at its place in
        # the response the stream ends with.
        from ._openai import check_responses_stream, check_streamed_call

        self._seen.append(event)
        data = event.data or {}
        kind = data.get("type")
        if kind in ("response.completed", "response.incomplete"):
            check_responses_stream(self._seen)
            return
        index = data.get("output_index")
        if not isinstance(index, int):
            return
        item = data.get("item")
        if kind in ("response.output_item.added", "response.output_item.done"):
            if isinstance(item, dict):
                entry = cast("dict[str, Any]", item)
                self._items.setdefault(index, []).append((kind, entry))
                if kind == "response.output_item.done" and "call_id" in entry:
                    check_streamed_call(
                        entry, self._items[index], self._others.get(index, [])
                    )
        else:
            self._others.setdefault(index, []).append(data)

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


def says_nothing(event: StreamEvent) -> bool:
    """Whether an event, as edited, carries nothing a client acts on: blank lines, Anthropic's `ping`, or a Chat Completions chunk left with an empty delta once a call's fragments were taken out of it."""
    if _blank(event):
        return True
    data = event.data
    if data is None:
        return False
    if data.get("type") == "ping":
        return True
    choices = data.get("choices")
    if data.get("object") != "chat.completion.chunk" or not isinstance(choices, list):
        return False
    if data.get("usage") or len(cast("list[object]", choices)) != 1:
        return False
    choice = cast("list[object]", choices)[0]
    if not isinstance(choice, dict):
        return False
    entry = cast("dict[str, Any]", choice)
    delta = cast("dict[str, Any]", entry.get("delta") or {})
    return entry.get("finish_reason") is None and not any(
        value not in (None, "", []) for value in delta.values()
    )
