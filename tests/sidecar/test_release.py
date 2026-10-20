import json
from typing import Any

import pytest

from inspect_sentinel.sidecar._release import (
    AnthropicTurn,
    ChatTurn,
    EventReader,
    Placement,
    ResponsesTurn,
    Turn,
)
from inspect_sentinel.sidecar._wire import (
    StreamEvent,
    UnreadableError,
    written_events,
)
from tests.sidecar._traffic import DATA

_RECORDINGS = sorted(path.name for path in DATA.glob("*.sse"))


def _read(body: bytes, size: int) -> list[StreamEvent]:
    reader = EventReader()
    events: list[StreamEvent] = []
    for at in range(0, len(body), size):
        events += reader.push(body[at : at + size])
    return events + reader.finish()


def _seen(events: list[StreamEvent]) -> list[tuple[list[tuple[str, str]], Any]]:
    return [(event.lines, event.data) for event in events]


@pytest.mark.parametrize("name", _RECORDINGS)
@pytest.mark.parametrize("end", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 1 << 30])
def test_events_read_as_they_arrive_are_those_of_the_whole_stream(
    name: str, end: str, size: int
) -> None:
    body = (
        (DATA / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", end.encode())
    )

    assert _seen(_read(body, size)) == _seen(written_events(body))


@pytest.mark.parametrize("size", [1, 2, 3])
def test_a_character_split_between_chunks_is_read_whole(size: int) -> None:
    body = 'data: {"text": "café ☕"}\n\ndata: [DONE]\n\n'.encode()

    events = _read(body, size)

    assert _seen(events) == _seen(written_events(body))
    assert events[0].data == {"text": "café ☕"}


def test_a_stream_that_is_not_utf8_is_unreadable() -> None:
    with pytest.raises(UnreadableError, match="not UTF-8"):
        EventReader().push(b"data: \xff\n\n")


def _placed(turn: Turn, name: str) -> list[Placement]:
    # without the empty event a stream's last blank line makes, which says nothing
    placed = [turn.place(event) for event in written_events((DATA / name).read_bytes())]
    return [placement for placement in placed if placement.role != "filler"]


def test_an_anthropic_call_is_complete_when_its_block_stops() -> None:
    placed = _placed(AnthropicTurn(), "parallel-tool-calls.sse")

    roles = [placement.role for placement in placed]
    assert roles[0] == "opening"
    assert roles[-2:] == ["closing", "closing"]
    # the text block goes out as it arrives, and each call is held
    assert {p.item for p in placed if p.role == "release"} == {0}
    assert [p.completes for p in placed if p.completes] == [(1,), (2,), (3,)]
    assert {p.item for p in placed if p.role == "call"} == {1, 2, 3}


def test_a_chat_call_is_complete_when_a_later_one_starts_or_the_choice_finishes() -> (
    None
):
    placed = _placed(ChatTurn(), "openai-chat.sse")

    assert placed[0].role == "opening"
    assert [p.completed_before for p in placed if p.completed_before] == [(0,), (1,)]
    finish = next(p for p in placed if p.completes)
    assert (finish.role, finish.completes) == ("closing", (2,))
    # the usage and `[DONE]` after it
    assert [p.role for p in placed[-2:]] == ["closing", "closing"]


def test_a_responses_call_is_complete_when_its_item_is_done() -> None:
    placed = _placed(ResponsesTurn(), "openai-responses.sse")

    assert [p.role for p in placed[:2]] == ["opening", "opening"]
    assert placed[-1].role == "closing"
    # the reasoning item goes out as it arrives
    assert {p.item for p in placed if p.role == "release"} == {0}
    assert [p.completes for p in placed if p.completes] == [(1,), (2,), (3,)]


def _place(turn: Turn, data: dict[str, Any]) -> Placement:
    return turn.place(
        StreamEvent([(f"data: {json.dumps(data)}", "\n"), ("", "\n")], data)
    )


def _chunk(**delta: Any) -> dict[str, Any]:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def _fragment(index: int, arguments: str, id: str | None = None) -> dict[str, Any]:
    fragment: dict[str, Any] = {"index": index, "function": {"arguments": arguments}}
    if id is not None:
        fragment |= {
            "id": id,
            "type": "function",
            "function": {"name": "f", "arguments": arguments},
        }
    return fragment


def test_a_chat_call_completes_with_the_event_that_holds_its_last_fragment() -> None:
    turn = ChatTurn()
    _place(turn, _chunk(tool_calls=[_fragment(0, "{", "call_0")]))

    # the last fragment of call 0 and the first of call 1, in one chunk
    placed = _place(
        turn, _chunk(tool_calls=[_fragment(0, "}"), _fragment(1, "{", "call_1")])
    )

    assert (placed.role, placed.item) == ("call", 1)
    assert (placed.completes, placed.completed_before) == ((0,), ())


def test_a_chat_call_changed_after_it_was_complete_is_unreadable() -> None:
    turn = ChatTurn()
    _place(turn, _chunk(tool_calls=[_fragment(0, "{", "call_0")]))
    _place(turn, _chunk(tool_calls=[_fragment(1, "{", "call_1")]))

    with pytest.raises(UnreadableError, match="after it was complete"):
        _place(turn, _chunk(tool_calls=[_fragment(0, "}")]))


@pytest.mark.parametrize(
    ("delta", "role"),
    [
        ({"role": "assistant", "content": None}, "opening"),
        ({"content": "Hello"}, "release"),
        ({"reasoning_content": "Thinking"}, "release"),
        ({"refusal": "No"}, "hold"),
        ({"audio": {"id": "a"}}, "hold"),
    ],
)
def test_only_chat_text_and_reasoning_go_out_as_they_arrive(
    delta: dict[str, Any], role: str
) -> None:
    assert _place(ChatTurn(), _chunk(**delta)).role == role


def test_a_chat_reply_of_more_than_one_choice_is_held() -> None:
    chunk = _chunk(content="Hello")
    chunk["choices"].append({"index": 1, "delta": {"content": "Hi"}})

    assert _place(ChatTurn(), chunk).role == "hold"


def test_an_anthropic_block_changed_after_it_stopped_is_unreadable() -> None:
    turn = AnthropicTurn()
    _place(
        turn,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "f",
                "input": {},
            },
        },
    )
    _place(turn, {"type": "content_block_stop", "index": 0})

    with pytest.raises(UnreadableError, match="after it stopped"):
        _place(
            turn,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        )


@pytest.mark.parametrize(
    ("block", "role"),
    [
        ({"type": "text", "text": ""}, "release"),
        ({"type": "thinking", "thinking": ""}, "release"),
        ({"type": "redacted_thinking", "data": "x"}, "release"),
        ({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}, "call"),
        (
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
            "hold",
        ),
        ({"type": "unknown_block"}, "hold"),
    ],
)
def test_only_anthropic_text_and_reasoning_go_out_as_they_arrive(
    block: dict[str, Any], role: str
) -> None:
    start = {"type": "content_block_start", "index": 0, "content_block": block}

    assert _place(AnthropicTurn(), start).role == role


def test_a_responses_item_changed_after_it_was_done_is_unreadable() -> None:
    turn = ResponsesTurn()
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "f",
        "arguments": "{}",
    }
    _place(
        turn, {"type": "response.output_item.added", "output_index": 0, "item": item}
    )
    _place(turn, {"type": "response.output_item.done", "output_index": 0, "item": item})

    with pytest.raises(UnreadableError, match="after it was done"):
        _place(
            turn,
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "x",
            },
        )


@pytest.mark.parametrize(
    ("item", "role"),
    [
        ({"type": "message", "id": "msg_1"}, "release"),
        ({"type": "reasoning", "id": "rs_1"}, "release"),
        ({"type": "function_call", "id": "fc_1", "call_id": "call_1"}, "call"),
        ({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1"}, "call"),
        ({"type": "web_search_call", "id": "ws_1"}, "hold"),
        ({"type": "unknown_item", "id": "x"}, "hold"),
    ],
)
def test_only_responses_text_and_reasoning_go_out_as_they_arrive(
    item: dict[str, Any], role: str
) -> None:
    added = {"type": "response.output_item.added", "output_index": 0, "item": item}

    assert _place(ResponsesTurn(), added).role == role


def test_events_for_an_item_never_added_are_unreadable() -> None:
    with pytest.raises(UnreadableError, match="hasn't added"):
        _place(
            ResponsesTurn(),
            {"type": "response.output_text.delta", "output_index": 3, "delta": "x"},
        )
