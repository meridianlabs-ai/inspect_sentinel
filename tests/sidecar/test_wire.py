import json

import pytest

from inspect_sentinel.sidecar import UnreadableError, read_reply, read_request
from inspect_sentinel.sidecar._wire import stream_events
from tests.sidecar._traffic import (
    OPENAI_RECORDED,
    STREAM,
    anthropic_recorded,
    openai_recorded,
    openai_request,
)


@pytest.mark.anyio
@pytest.mark.parametrize("line_end", [b"\r\n", b"\r"], ids=["CRLF", "CR"])
async def test_stream_is_read_whatever_ends_its_lines(line_end: bytes) -> None:
    as_recorded = anthropic_recorded("parallel-tool-calls")
    assert b"\r" not in as_recorded

    body = as_recorded.replace(b"\n", line_end)
    output = await read_reply("anthropic", STREAM, body)

    calls = output.message.tool_calls or []
    assert [call.arguments["location"] for call in calls] == ["Paris", "Tokyo", "Lima"]


@pytest.mark.parametrize("code_point", [0x2028, 0x2029, 0x85])
def test_line_separator_inside_a_string_does_not_end_the_line(code_point: int) -> None:
    event = {"type": "text", "text": f"before{chr(code_point)}after"}
    data = json.dumps(event, ensure_ascii=False)
    assert chr(code_point) in data

    assert list(stream_events(f"data: {data}\n\n".encode())) == [event]


@pytest.mark.anyio
async def test_anthropic_body_is_read_as_the_format_the_proxy_names() -> None:
    # Claude served by another provider is still asked in the Messages API
    request = json.dumps({"messages": [{"role": "user", "content": "Hello"}]})

    messages = await read_request("bedrock", request.encode(), "anthropic-messages")
    output = await read_reply(
        "bedrock", STREAM, anthropic_recorded("single-tool-call"), "anthropic-messages"
    )

    assert messages[-1].text == "Hello"
    assert [call.function for call in output.message.tool_calls or []] == [
        "get_weather"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(OPENAI_RECORDED))
async def test_openai_reply_is_read_as_the_format_the_proxy_names(name: str) -> None:
    # a proxy can serve another provider's model through OpenAI's APIs
    format = "openai-chat" if name.startswith("chat") else "openai-responses"

    output = await read_reply("anthropic", *openai_recorded(name), format)

    assert len(output.message.tool_calls or []) == 3


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("chat", "can't be read"),
        ("chat stream", "not a complete response"),
    ],
)
async def test_reply_that_is_not_in_the_format_the_proxy_names_is_not_read(
    name: str, match: str
) -> None:
    # the proxy's word is taken over the reply's own `object` field
    with pytest.raises(UnreadableError, match=match):
        await read_reply("openai", *openai_recorded(name), "openai-responses")


@pytest.mark.anyio
async def test_openai_request_is_read_as_the_format_the_proxy_names() -> None:
    chat = json.dumps(openai_request("chat")).encode()

    messages = await read_request("anthropic", chat, "openai-chat")

    assert [message.role for message in messages] == ["system", "user"]
    with pytest.raises(UnreadableError, match="no input"):
        await read_request("openai", chat, "openai-responses")


@pytest.mark.anyio
async def test_provider_with_no_reader() -> None:
    with pytest.raises(UnreadableError, match="no reader for provider 'somebody'"):
        await read_request("somebody", b"{}")
    with pytest.raises(UnreadableError, match="no reader for provider 'somebody'"):
        await read_reply("somebody", STREAM, b"")
