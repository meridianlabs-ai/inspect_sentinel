import json
from typing import Any

import pytest

from inspect_sentinel.sidecar._release import (
    AnthropicTurn,
    ChatTurn,
    EventReader,
    ResponsesTurn,
    Role,
    Turn,
)
from inspect_sentinel.sidecar._wire import StreamEvent, UnreadableError, written_events
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


def _place(turn: Turn, data: dict[str, Any]) -> Role:
    return turn.place(
        StreamEvent([(f"data: {json.dumps(data)}", "\n"), ("", "\n")], data)
    )


def _roles(turn: Turn, name: str) -> list[Role]:
    # without the events that say nothing
    events = written_events((DATA / name).read_bytes())
    roles: list[Role] = [turn.place(event) for event in events]
    kept: list[Role] = [role for role in roles if role != "filler"]
    return kept


def test_a_recorded_anthropic_turn_goes_out_until_its_first_call() -> None:
    roles = _roles(AnthropicTurn(), "parallel-tool-calls.sse")

    first = roles.index("held")
    assert roles[0] == "opening" and first > 1
    assert set(roles[1:first]) == {"release"}


def _chunk(**delta: Any) -> dict[str, Any]:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


@pytest.mark.parametrize(
    ("delta", "role"),
    [
        ({"role": "assistant", "content": None}, "opening"),
        ({"content": "Hello"}, "release"),
        ({"reasoning_content": "Thinking"}, "release"),
        ({"tool_calls": [{"index": 0, "id": "call_1"}]}, "held"),
        ({"refusal": "No"}, "held"),
        ({"audio": {"id": "a"}}, "held"),
    ],
)
def test_only_chat_text_and_reasoning_go_out_as_they_arrive(
    delta: dict[str, Any], role: str
) -> None:
    assert _place(ChatTurn(), _chunk(**delta)) == role


def test_a_chat_reply_of_more_than_one_choice_is_held() -> None:
    chunk = _chunk(content="Hello")
    chunk["choices"].append({"index": 1, "delta": {"content": "Hi"}})

    assert _place(ChatTurn(), chunk) == "held"


def test_a_chat_choice_that_finishes_is_held() -> None:
    chunk = _chunk(content="")
    chunk["choices"][0]["finish_reason"] = "stop"

    assert _place(ChatTurn(), chunk) == "held"


@pytest.mark.parametrize(
    ("block", "role"),
    [
        ({"type": "text", "text": ""}, "release"),
        ({"type": "thinking", "thinking": ""}, "release"),
        ({"type": "redacted_thinking", "data": "x"}, "release"),
        ({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}, "held"),
        (
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
            "held",
        ),
        ({"type": "unknown_block"}, "held"),
    ],
)
def test_only_anthropic_text_and_reasoning_go_out_as_they_arrive(
    block: dict[str, Any], role: str
) -> None:
    start = {"type": "content_block_start", "index": 0, "content_block": block}

    assert _place(AnthropicTurn(), start) == role


def test_an_anthropic_delta_for_a_block_not_released_is_held() -> None:
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hi"},
    }

    assert _place(AnthropicTurn(), delta) == "held"


@pytest.mark.parametrize(
    ("block", "delta", "role"),
    [
        ("text", {"type": "text_delta", "text": "Hi"}, "release"),
        ("text", {"type": "citations_delta", "citation": {}}, "release"),
        ("thinking", {"type": "thinking_delta", "thinking": "Hm"}, "release"),
        ("thinking", {"type": "signature_delta", "signature": "s"}, "release"),
        ("text", {"type": "input_json_delta", "partial_json": "{"}, "held"),
        ("thinking", {"type": "text_delta", "text": "Hi"}, "held"),
        ("redacted_thinking", {"type": "text_delta", "text": "Hi"}, "held"),
        ("text", {"type": "unknown_delta"}, "held"),
    ],
)
def test_only_the_deltas_of_a_released_anthropic_block_go_out(
    block: str, delta: dict[str, Any], role: str
) -> None:
    # a provider or a proxy that writes something else into a released block
    # is not trusted to write only what the block holds
    turn = AnthropicTurn()
    _place(
        turn,
        {"type": "content_block_start", "index": 0, "content_block": {"type": block}},
    )

    assert (
        _place(turn, {"type": "content_block_delta", "index": 0, "delta": delta})
        == role
    )
    assert _place(turn, {"type": "content_block_stop", "index": 0}) == "release"


@pytest.mark.parametrize(
    ("item", "event", "role"),
    [
        ("message", {"type": "response.output_text.delta", "delta": "x"}, "release"),
        ("message", {"type": "response.content_part.added"}, "release"),
        ("message", {"type": "response.refusal.delta", "delta": "x"}, "release"),
        (
            "reasoning",
            {"type": "response.reasoning_summary_text.delta", "delta": "x"},
            "release",
        ),
        (
            "reasoning",
            {"type": "response.reasoning_text.delta", "delta": "x"},
            "release",
        ),
        (
            "message",
            {"type": "response.output_item.done", "item": {"type": "message"}},
            "release",
        ),
        (
            "reasoning",
            {"type": "response.output_item.done", "item": {"type": "reasoning"}},
            "release",
        ),
        (
            "message",
            {"type": "response.function_call_arguments.delta", "delta": "{"},
            "held",
        ),
        (
            "message",
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": "call_1"},
            },
            "held",
        ),
        ("reasoning", {"type": "response.output_text.delta", "delta": "x"}, "held"),
        ("message", {"type": "response.reasoning_text.delta", "delta": "x"}, "held"),
        ("message", {"type": "response.unknown.delta"}, "held"),
    ],
)
def test_only_the_events_of_a_released_responses_item_go_out(
    item: str, event: dict[str, Any], role: str
) -> None:
    # a provider or a proxy that reuses a released item's index for anything
    # else, a call above all, is not trusted to write only what the item holds
    turn = ResponsesTurn()
    added = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": item},
    }
    _place(turn, added)

    assert _place(turn, event | {"output_index": 0}) == role


@pytest.mark.parametrize(
    ("item", "role"),
    [
        ({"type": "message", "id": "msg_1"}, "release"),
        ({"type": "reasoning", "id": "rs_1"}, "release"),
        ({"type": "function_call", "id": "fc_1", "call_id": "call_1"}, "held"),
        ({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1"}, "held"),
        ({"type": "web_search_call", "id": "ws_1"}, "held"),
        ({"type": "unknown_item", "id": "x"}, "held"),
    ],
)
def test_only_responses_text_and_reasoning_go_out_as_they_arrive(
    item: dict[str, Any], role: str
) -> None:
    added = {"type": "response.output_item.added", "output_index": 0, "item": item}

    assert _place(ResponsesTurn(), added) == role


def test_responses_events_for_an_item_never_added_are_held() -> None:
    delta = {"type": "response.output_text.delta", "output_index": 0, "delta": "x"}

    assert _place(ResponsesTurn(), delta) == "held"


@pytest.mark.parametrize("turn", [AnthropicTurn, ChatTurn, ResponsesTurn])
@pytest.mark.parametrize(
    "lines", [[(": ping", "\n"), ("", "\n")], [("", "\n")]], ids=["comment", "blank"]
)
def test_what_every_sse_reader_skips_says_nothing(
    turn: type[Turn], lines: list[tuple[str, str]]
) -> None:
    assert turn().place(StreamEvent(lines, None)) == "filler"
