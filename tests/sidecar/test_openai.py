import json
from typing import Any

import pytest
from inspect_ai.model import ContentReasoning

from inspect_sentinel.sidecar import UnreadableError, read_reply, read_request
from tests.sidecar._traffic import (
    DATA,
    JSON,
    OPENAI_RECORDED,
    STREAM,
    openai_ran,
    openai_recorded,
    openai_request,
)


def _half(body: bytes) -> bytes:
    frames = body.split(b"\n\n")
    return b"\n\n".join(frames[: len(frames) // 2]) + b"\n\n"


def _changed(name: str, **changes: Any) -> bytes:
    return json.dumps(json.loads(openai_recorded(name)[1]) | changes).encode()


# a call to a custom tool, whose input is free text and not JSON arguments
_CUSTOM_CALL: dict[str, Any] = {
    "id": "call_9",
    "type": "custom",
    "custom": {"name": "apply_patch", "input": "*** Begin Patch"},
}


def _chat_with_custom_call() -> bytes:
    reply = json.loads(openai_recorded("chat")[1])
    reply["choices"][0]["message"]["tool_calls"].append(_CUSTOM_CALL)
    return json.dumps(reply).encode()


def _chat_stream_with_custom_call() -> bytes:
    chunk: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [_CUSTOM_CALL | {"index": 0}],
                },
                "finish_reason": None,
            }
        ],
    }
    end: dict[str, Any] = chunk | {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    }
    return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\n".encode()


def _new_finish_reason_in_stream() -> bytes:
    lines = openai_recorded("chat stream")[1].decode().split("\n")
    changed = [
        line.replace('"finish_reason":"tool_calls"', '"finish_reason":"a_new_reason"')
        for line in lines
    ]
    assert changed != lines
    return "\n".join(changed).encode()


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(OPENAI_RECORDED))
async def test_recorded_reply_is_read(name: str) -> None:
    output = await read_reply("openai", *openai_recorded(name))

    assert output.stop_reason == "tool_calls"
    calls = output.message.tool_calls or []
    assert [call.function for call in calls] == ["get_weather"] * 3
    # the model wrote "Tokyo" in one reply and "Tokyo, Japan" in another
    assert [str(call.arguments["location"]).split(",")[0] for call in calls] == [
        "Paris",
        "Tokyo",
        "Lima",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        pytest.param(
            JSON,
            _changed(
                "chat",
                choices=[
                    choice | {"finish_reason": "a_new_reason"}
                    for choice in json.loads(openai_recorded("chat")[1])["choices"]
                ],
            ),
            id="chat",
        ),
        pytest.param(STREAM, _new_finish_reason_in_stream(), id="chat stream"),
        pytest.param(
            JSON, _changed("responses", service_tier="a_new_tier"), id="responses"
        ),
    ],
)
async def test_value_this_sdk_does_not_know_is_not_an_error(
    content_type: str, body: bytes
) -> None:
    output = await read_reply("openai", content_type, body)

    assert len(output.message.tool_calls or []) == 3


@pytest.mark.anyio
async def test_thinking_written_loosely_by_a_proxy_is_read() -> None:
    # A reply LiteLLM 1.105 wrote for Claude through the Responses API. Its
    # reasoning item has no `summary` and its parts are typed `output_text`,
    # which OpenAI's SDK types don't allow.
    body = (DATA / "litellm-claude-responses.json").read_bytes()

    output = await read_reply("openai", JSON, body)

    content = output.message.content
    assert isinstance(content, list)
    reasoning = [part for part in content if isinstance(part, ContentReasoning)]
    assert len(reasoning) == 1
    assert (reasoning[0].summary or "").startswith(
        "The user is asking for the weather in Philadelphia"
    )
    assert [call.arguments for call in output.message.tool_calls or []] == [
        {"city": "Philadelphia"}
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content_type", "body", "match"),
    [
        pytest.param(
            STREAM,
            _half(openai_recorded("chat stream")[1]),
            "not a complete message",
            id="chat stream cut short",
        ),
        pytest.param(
            STREAM,
            _half(openai_recorded("responses stream")[1]),
            "not a complete response",
            id="responses stream cut short",
        ),
        pytest.param(
            STREAM,
            b'data: {"type": "error", "message": "overloaded"}\n\n',
            "error from the provider",
            id="stream that ends in an error",
        ),
        pytest.param(
            JSON,
            _changed(
                "chat", choices=json.loads(openai_recorded("chat")[1])["choices"] * 2
            ),
            "more than one choice",
            id="two choices",
        ),
        pytest.param(
            JSON,
            _chat_with_custom_call(),
            "tool call of a kind that can't be read: 'custom'",
            id="chat, a custom tool call",
        ),
        pytest.param(
            STREAM,
            _chat_stream_with_custom_call(),
            "tool call of a kind that can't be read: 'custom'",
            id="chat stream, a custom tool call",
        ),
        pytest.param(
            JSON,
            _changed("responses", output=[{"type": "not_known_yet", "id": "x"}]),
            "can't be read",
            id="output of a kind inspect_ai doesn't read",
        ),
        pytest.param(
            JSON, _changed("chat", object="list"), "'list' reply", id="another API"
        ),
        pytest.param(JSON, b"[]", "not a JSON object", id="not an object"),
    ],
)
async def test_reply_that_cannot_be_read(
    content_type: str, body: bytes, match: str
) -> None:
    with pytest.raises(UnreadableError, match=match):
        await read_reply("openai", content_type, body)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "roles"),
    [
        pytest.param(
            openai_request("chat", then=openai_ran("chat", "rain")),
            ["system", "user", "assistant", "tool"],
            id="chat",
        ),
        pytest.param(
            openai_request("responses", then=openai_ran("responses", "rain")),
            ["system", "user", "assistant", "tool"],
            id="responses",
        ),
        pytest.param(
            {"model": "gpt-test", "input": "What is the weather?"},
            ["user"],
            id="responses, a lone message as a string",
        ),
    ],
)
async def test_request_is_read(body: dict[str, Any], roles: list[str]) -> None:
    messages = await read_request("openai", json.dumps(body).encode())

    assert [message.role for message in messages] == roles
    assert messages[-1].text == ("rain" if roles[-1] == "tool" else body["input"])


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            {"input": "And then?", "previous_response_id": "resp_1"},
            "the provider keeps",
            id="continues a stored response",
        ),
        pytest.param(
            {"input": "And then?", "conversation": "conv_1"},
            "the provider keeps",
            id="continues a stored conversation",
        ),
        pytest.param(
            {"input": "Take your time.", "background": True},
            "fetched later",
            id="answered in the background",
        ),
        pytest.param(
            openai_request(
                "chat",
                then=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_CUSTOM_CALL],
                    },
                    {"role": "tool", "tool_call_id": "call_9", "content": "done"},
                ],
            ),
            "tool call of a kind that can't be read: 'custom'",
            id="chat, a custom tool call in the history",
        ),
        pytest.param(
            {"input": [{"type": "item_reference", "id": "msg_1"}]},
            "input that can't be read",
            id="responses, an item the provider keeps",
        ),
        pytest.param({"prompt": "Once upon a time"}, "no reader", id="completions"),
        pytest.param({"messages": "hello"}, "no list of messages", id="no messages"),
    ],
)
async def test_request_that_cannot_be_read(body: dict[str, Any], match: str) -> None:
    with pytest.raises(UnreadableError, match=match):
        await read_request("openai", json.dumps(body | {"model": "gpt-test"}).encode())
