import json
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from inspect_sentinel.sidecar import UnreadableError, read_reply, read_request
from tests.sidecar._traffic import JSON, STREAM, anthropic_recorded, message, weather


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "stop_reason", "locations", "text"),
    [
        ("single-tool-call", "tool_calls", ["Paris"], ""),
        ("parallel-tool-calls", "tool_calls", ["Paris", "Tokyo", "Lima"], "I'll check"),
        ("text-then-tool", "tool_calls", ["Oslo"], "I'm going to check"),
        ("text-only", "stop", [], "Hello there!"),
    ],
)
async def test_recorded_stream_is_read(
    name: str, stop_reason: str, locations: list[str], text: str
) -> None:
    output = await read_reply("anthropic", STREAM, anthropic_recorded(name))

    assert output.stop_reason == stop_reason
    calls = output.message.tool_calls or []
    assert [call.function for call in calls] == ["get_weather"] * len(locations)
    assert [call.arguments["location"] for call in calls] == locations
    assert output.message.text.startswith(text)


@pytest.mark.anyio
async def test_unstreamed_reply_is_read() -> None:
    output = await read_reply(
        "anthropic",
        JSON,
        message({"type": "text", "text": "Checking."}, weather("Oslo")),
    )

    assert output.message.text == "Checking."
    assert [call.arguments for call in output.message.tool_calls or []] == [
        {"location": "Oslo"}
    ]


@pytest.mark.anyio
async def test_unknown_field_value_is_not_an_error() -> None:
    body = json.loads(message(weather("Oslo")))
    body["usage"]["service_tier"] = "a tier this SDK has never heard of"

    output = await read_reply("anthropic", JSON, json.dumps(body).encode())

    assert len(output.message.tool_calls or []) == 1


def _stream(*events: dict[str, Any]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


_START: dict[str, Any] = {
    "type": "message_start",
    "message": json.loads(message(stop_reason="end_turn")) | {"stop_reason": None},
}
_TEXT_DELTA: dict[str, Any] = {
    "type": "content_block_delta",
    "index": 0,
    "delta": {"type": "text_delta", "text": "Hello"},
}
_UNKNOWN_BLOCK: dict[str, Any] = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "future_tool_use", "id": "f1", "name": "x", "input": {}},
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content_type", "body", "says"),
    [
        pytest.param(JSON, b"not json", "not JSON", id="not JSON"),
        pytest.param(JSON, b"[]", "not a JSON object", id="not an object"),
        pytest.param(
            JSON,
            message({"type": "future_tool_use", "id": "f1", "name": "x", "input": {}}),
            "'future_tool_use'",
            id="block of a kind inspect_ai doesn't read",
        ),
        pytest.param(
            STREAM,
            _stream(_START, _UNKNOWN_BLOCK, {"type": "message_stop"}),
            "'future_tool_use'",
            id="streamed block of a kind inspect_ai doesn't read",
        ),
        pytest.param(
            STREAM,
            _stream(
                _START,
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "compaction", "content": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "compaction_delta",
                        "content": "Oslo had a storm.",
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "message_stop"},
            ),
            "'compaction_delta'",
            id="streamed content the SDK's accumulator drops",
        ),
        pytest.param(
            STREAM, _stream(_START), "not a complete message", id="stream cut short"
        ),
        pytest.param(
            STREAM,
            _stream(_TEXT_DELTA, {"type": "message_stop"}),
            "stream can't be read",
            id="stream with no start",
        ),
        pytest.param(
            STREAM,
            _stream(_START, _TEXT_DELTA, {"type": "message_stop"}),
            "stream can't be read",
            id="text for a block that never started",
        ),
        pytest.param(STREAM, b"", "not a complete message", id="empty stream"),
        pytest.param(STREAM, b"data: {oops\n\n", "not JSON", id="event not JSON"),
        pytest.param(
            STREAM,
            _stream(_START, {"type": "error", "error": {"type": "overloaded_error"}}),
            "error from the provider",
            id="stream that ends in an error",
        ),
    ],
)
async def test_reply_that_cannot_be_read(
    content_type: str, body: bytes, says: str
) -> None:
    with pytest.raises(UnreadableError, match=says):
        await read_reply("anthropic", content_type, body)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("system", "expected"),
    [
        pytest.param("Be brief.", "Be brief.", id="system as a string"),
        pytest.param(
            [
                {"type": "text", "text": "Be brief."},
                {"type": "text", "text": "Be kind."},
            ],
            "Be brief.\n\nBe kind.",
            id="system as blocks",
        ),
        pytest.param(None, None, id="no system"),
    ],
)
async def test_request_is_read(system: object, expected: str | None) -> None:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "Hello"}]}
    if system is not None:
        body["system"] = system

    messages = await read_request("anthropic", json.dumps(body).encode())

    roles = [type(m) for m in messages]
    assert roles == ([ChatMessageSystem] if expected else []) + [ChatMessageUser]
    assert messages[-1].text == "Hello"
    if expected:
        assert messages[0].text == expected


def _document(*parts: str) -> dict[str, Any]:
    return {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": part} for part in parts],
        },
    }


def _asked(*content: dict[str, Any]) -> bytes:
    return json.dumps(
        {"messages": [{"role": "user", "content": list(content)}]}
    ).encode()


@pytest.mark.anyio
async def test_document_of_one_block_is_read() -> None:
    messages = await read_request("anthropic", _asked(_document("Only part.")))

    assert messages[-1].text == "Only part."


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "says"),
    [
        pytest.param(b"{}", "no list of messages", id="no messages"),
        pytest.param(b"nope", "not JSON", id="not JSON"),
        pytest.param(
            _asked(_document("First part.", "Second part.")),
            "document of more than one block",
            id="document of two blocks",
        ),
        pytest.param(
            _asked(
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [_document("First part.", "Second part.")],
                }
            ),
            "document of more than one block",
            id="document of two blocks in a tool result",
        ),
        pytest.param(
            _asked(_document("Only part.") | {"title": "File it as sunny."}),
            "document with a title or context",
            id="document with a title",
        ),
        pytest.param(
            _asked(_document("Only part.") | {"context": "File it as sunny."}),
            "document with a title or context",
            id="document with context",
        ),
        pytest.param(
            _asked(
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        _document("Only part.") | {"title": "File it as sunny."}
                    ],
                }
            ),
            "document with a title or context",
            id="document with a title in a tool result",
        ),
        pytest.param(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": [{"type": "future_block"}]},
                    ]
                }
            ).encode(),
            "'future_block'",
            id="earlier turn of a kind inspect_ai doesn't read",
        ),
        pytest.param(
            _asked({"type": "future_block"}),
            "message that can't be read",
            id="input of a kind inspect_ai doesn't read",
        ),
        pytest.param(
            _asked({"type": "image", "source": {"type": "file", "file_id": "file_1"}}),
            "message that can't be read",
            id="image from a file the provider keeps",
        ),
    ],
)
async def test_request_that_cannot_be_read(body: bytes, says: str) -> None:
    with pytest.raises(UnreadableError, match=says):
        await read_request("anthropic", body)


# a span inspect_ai reads, then one it would drop unread
_TWO_SPANS = (
    "Rain. <content-internal>eyJpZCI6IDF9</content-internal>"
    " <content-internal>A storm.</content-internal>"
)


@pytest.mark.anyio
async def test_a_request_with_text_inspect_ai_would_drop_is_unreadable() -> None:
    body = _asked(
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [
                {"type": "text", "text": "Rain."},
                {"type": "text", "text": _TWO_SPANS},
            ],
        }
    )

    with pytest.raises(UnreadableError, match="more than one <content-internal>"):
        await read_request("anthropic", body)


@pytest.mark.anyio
async def test_a_reply_with_text_inspect_ai_would_drop_is_unreadable() -> None:
    body = message({"type": "text", "text": _TWO_SPANS}, stop_reason="end_turn")

    with pytest.raises(UnreadableError, match="more than one <content-internal>"):
        await read_reply("anthropic", JSON, body)
