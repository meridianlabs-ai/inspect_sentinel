import base64
import json
from dataclasses import replace
from typing import Any, cast

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ContentReasoning
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import AfterToolCall, BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    UnreadableError,
    read_reply,
    read_request,
)
from tests.sidecar._traffic import (
    DATA,
    JSON,
    OPENAI_RECORDED,
    STREAM,
    openai_ran,
    openai_recorded,
    openai_request,
    storm_rule,
    weather_rule,
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


_PDF = "data:application/pdf;base64,JVBERi0xLjQK"


def _result_parts(*parts: dict[str, Any]) -> list[dict[str, Any]]:
    # a Responses conversation whose tool result is given as parts
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "Rain."}, *parts],
        },
    ]


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


_AUDIO = {"id": "audio_1", "data": "UklGRg==", "expires_at": 1, "transcript": "Rain."}


def _chat_with_audio() -> bytes:
    reply = json.loads(openai_recorded("chat")[1])
    reply["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "audio": _AUDIO,
    }
    reply["choices"][0]["finish_reason"] = "stop"
    return json.dumps(reply).encode()


def _chat_stream_with_audio() -> bytes:
    chunk: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "audio": _AUDIO},
                "finish_reason": None,
            }
        ],
    }
    end: dict[str, Any] = chunk | {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
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
            _chat_with_audio(),
            "reply holds audio",
            id="chat, an audio reply",
        ),
        pytest.param(
            STREAM,
            _chat_stream_with_audio(),
            "reply holds audio",
            id="chat stream, an audio reply",
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
        pytest.param(
            {"input": _result_parts({"type": "input_file", "file_data": _PDF})},
            "a tool result with an 'input_file' part",
            id="responses, a file in a tool result",
        ),
        pytest.param(
            {
                "input": _result_parts(
                    {"type": "input_image", "image_url": None, "file_id": "file_1"}
                )
            },
            "a tool result with an 'input_image' part",
            id="responses, an image by file id in a tool result",
        ),
        pytest.param(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is this?"},
                            {"type": "input_image", "file_id": "file_1"},
                        ],
                    }
                ]
            },
            "a message with an 'input_image' part",
            id="responses, an image by file id in a message",
        ),
        pytest.param({"prompt": "Once upon a time"}, "no reader", id="completions"),
        pytest.param({"messages": "hello"}, "no list of messages", id="no messages"),
    ],
)
async def test_request_that_cannot_be_read(body: dict[str, Any], match: str) -> None:
    with pytest.raises(UnreadableError, match=match):
        await read_request("openai", json.dumps(body | {"model": "gpt-test"}).encode())


@pytest.mark.anyio
async def test_a_responses_tool_result_of_text_and_images_is_read_whole() -> None:
    image = {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K"}
    body = {"model": "gpt-test", "input": _result_parts(image)}

    messages = await read_request("openai", json.dumps(body).encode())

    result = messages[-1]
    assert isinstance(result, ChatMessageTool)
    assert [part.type for part in cast("list[Any]", result.content)] == [
        "text",
        "image",
    ]


def _request(api: str, id: str = "x", then: Any = (), **fields: Any) -> Request:
    return Request(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        body=json.dumps(openai_request(api, then=then) | fields).encode(),
    )


def _reply(name: str, id: str = "x") -> Reply:
    content_type, body = openai_recorded(name)
    return Reply(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        status=200,
        content_type=content_type,
        body=body,
    )


def _conversation(api: str, body: bytes) -> list[dict[str, Any]]:
    return cast(
        "list[dict[str, Any]]",
        json.loads(body)["messages" if api == "chat" else "input"],
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api", "kinds", "told"),
    [
        (
            "chat",
            ["assistant", "tool", "tool", "tool"],
            # OpenAI's formats have no flag for a failed call, so inspect_ai
            # says it in the text
            [
                "Error: " + text
                for text in ("not run", "Tokyo is not allowed.", "not run")
            ],
        ),
        (
            "responses",
            ["reasoning", *["function_call"] * 3, *["function_call_output"] * 3],
            ["not run", "Tokyo is not allowed.", "not run"],
        ),
    ],
)
async def test_rejection_is_added_in_the_format_of_the_request(
    api: str, kinds: list[str], told: list[str]
) -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), reject_status=503)
    first = _request(api, "a")
    await handler.request(first)

    refused = await handler.reply(_reply(api, "a"))
    again = await handler.request(_request(api, "b"))

    assert refused == Refuse(
        503, "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )
    assert isinstance(again, Replace)
    sent, added = _conversation(api, first.body), _conversation(api, again.body)
    assert added[: len(sent)] == sent
    entries = added[len(sent) :]
    assert [entry.get("type") or entry["role"] for entry in entries] == kinds
    results = [entry.get("content") or entry.get("output") for entry in entries][-3:]
    not_run = (
        "This call was not run, because another call in the same turn was rejected."
    )
    assert results == [text.replace("not run", not_run) for text in told]
    # everything else in the request is as the client sent it
    field = "messages" if api == "chat" else "input"
    assert {k: v for k, v in json.loads(again.body).items() if k != field} == {
        k: v for k, v in json.loads(first.body).items() if k != field
    }


@pytest.mark.anyio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            "storm",
            Refuse(
                400,
                "A sentinel ended this run at the result of the call to get_weather.",
            ),
        ),
        ("rain", Pass()),
    ],
)
async def test_result_is_judged(api: str, result: str, expected: Pass | Refuse) -> None:
    handler = Handler(storm_rule())

    answer = await handler.request(_request(api, then=openai_ran(api, result)))

    assert answer == expected


@protocol
def reports_tools() -> Protocol:
    async def decide(context: Context, step: BeforeGenerate) -> Decision | None:
        return Decision.proceed(
            f"{[tool.name for tool in step.tools]}|{step.config.max_tokens}"
        )

    return decide


_FUNCTION = {
    "name": "get_weather",
    "description": "Look up the weather.",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api", "fields", "expected"),
    [
        (
            "chat",
            {"tools": [{"type": "function", "function": _FUNCTION}], "max_tokens": 64},
            "['get_weather']|64",
        ),
        (
            "responses",
            {
                "tools": [
                    {"type": "function", **_FUNCTION},
                    # a tool whose input is free text, which Codex uses
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Apply a patch.",
                        "format": {"type": "text"},
                    },
                    # a tool the provider runs
                    {"type": "web_search"},
                ],
                "max_output_tokens": 64,
            },
            "['get_weather', 'apply_patch', 'web_search']|64",
        ),
    ],
)
async def test_generate_stage_is_given_the_tools_the_request_offers(
    api: str, fields: dict[str, Any], expected: str
) -> None:
    handler = Handler(reports_tools())

    await handler.request(_request(api, **fields))

    records = await handler.records("x")
    assert records is not None
    assert [cast(Any, record.detail).report.explanation for record in records] == [
        expected
    ]


@pytest.mark.anyio
async def test_generate_stage_is_given_the_tools_a_responses_input_declares() -> None:
    handler = Handler(reports_tools())
    declared = {"type": "function", **_FUNCTION}
    other = declared | {"name": "file_report"}

    await handler.request(
        _request(
            "responses",
            then=[{"type": "additional_tools", "tools": [declared, other]}],
            tools=[declared],
        )
    )

    records = await handler.records("x")
    assert records is not None
    # declared in both places, it is offered once
    assert [cast(Any, record.detail).report.explanation for record in records] == [
        "['get_weather', 'file_report']|None"
    ]


@pytest.mark.anyio
async def test_a_chat_tool_result_read_as_less_than_was_sent_is_unreadable() -> None:
    # inspect_ai's reader would keep only "sun", and the model is sent it all
    body = openai_request(
        "chat",
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "<think>storm</think>sun",
            },
        ],
    )

    with pytest.raises(UnreadableError, match="less than the model is sent"):
        await read_request("openai", json.dumps(body).encode())


@pytest.mark.anyio
async def test_a_chat_tool_result_with_tags_in_parts_is_read_whole() -> None:
    body = openai_request(
        "chat",
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "<think>storm</think>sun"}],
            },
        ],
    )

    messages = await read_request("openai", json.dumps(body).encode())

    assert messages[-1].text == "<think>storm</think>sun"


@pytest.mark.anyio
async def test_a_chat_custom_tool_makes_the_request_unreadable() -> None:
    # inspect_ai's Chat Completions reader has no form for a custom tool, and
    # a step is never told the model is offered fewer tools than it is
    tools = [
        {"type": "function", "function": _FUNCTION},
        {"type": "custom", "custom": {"name": "apply_patch"}},
    ]

    with pytest.raises(UnreadableError, match="openai-chat custom"):
        await Handler(reports_tools()).request(_request("chat", tools=tools))


@protocol
def rejects_tokyo_and_ends_at_any_result() -> ProtocolGroup:
    async def before(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.terminate("a result")

    return ProtocolGroup(before, after)


@pytest.mark.anyio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_what_the_model_is_told_of_a_rejection_is_not_judged_as_a_result(
    api: str,
) -> None:
    handler = Handler(rejects_tokyo_and_ends_at_any_result())
    await handler.request(_request(api, "a"))
    await handler.reply(_reply(api, "a"))

    again = await handler.request(_request(api, "b"))

    assert isinstance(again, Replace)
    assert await handler.records("b") == []


def _function_calls(body: bytes) -> list[tuple[str, str, str]]:
    return [
        (entry["call_id"], entry.get("id", ""), entry["arguments"])
        for entry in json.loads(body)["input"]
        if entry.get("type") == "function_call"
    ]


@pytest.mark.anyio
async def test_a_rejected_turn_is_put_back_as_its_own_run_was_handed_it() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(replace(_request("responses", "a"), run="run_a"))
    first = _reply("responses", "a")
    await handler.reply(first)
    # another run's model reuses the call ids, with arguments of its own
    other = json.loads(first.body)
    for item in other["output"]:
        if item["type"] == "function_call":
            item["arguments"] = json.dumps({"location": "Tokyo", "note": "run b's"})
    await handler.request(replace(_request("responses", "b"), run="run_b"))
    await handler.reply(replace(first, id="b", body=json.dumps(other).encode()))

    again = await handler.request(replace(_request("responses", "c"), run="run_a"))

    assert isinstance(again, Replace)
    handed = json.loads(first.body)["output"]
    # each call as the provider wrote it, its own id included
    assert _function_calls(again.body) == [
        (item["call_id"], item["id"], item["arguments"])
        for item in handed
        if item["type"] == "function_call"
    ]


class _Asked(StoreModel):
    requests: int = 0


@protocol
def one_request_a_run() -> Protocol:
    async def decide(context: Context, step: BeforeGenerate) -> Decision | None:
        asked = context.store_as(_Asked)
        asked.requests += 1
        if asked.requests > 1:
            return Decision.terminate("A second request.")
        return None

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api", "role"),
    [("chat", "system"), ("chat", "developer"), ("responses", "developer")],
)
async def test_requests_that_open_with_different_instructions_alone_are_runs_apart(
    api: str, role: str
) -> None:
    handler = Handler(one_request_a_run())
    field = "messages" if api == "chat" else "input"

    def asked(id: str, prompt: str) -> Request:
        body = {"model": "gpt-test", field: [{"role": role, "content": prompt}]}
        return Request(
            id=id,
            provider="openai",
            model="gpt-test",
            user="user-1",
            body=json.dumps(body).encode(),
        )

    first = await handler.request(asked("a", "You file weather reports."))
    second = await handler.request(asked("b", "You book travel."))

    assert (first, second) == (Pass(), Pass())


_FUNCTION_CALL = {"name": "get_weather", "arguments": '{"location": "Paris"}'}


def _chat_with_function_call() -> bytes:
    reply = json.loads(openai_recorded("chat")[1])
    reply["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "function_call": _FUNCTION_CALL,
    }
    reply["choices"][0]["finish_reason"] = "function_call"
    return json.dumps(reply).encode()


def _chat_stream_with_function_call() -> bytes:
    chunk: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "function_call": _FUNCTION_CALL},
                "finish_reason": None,
            }
        ],
    }
    end: dict[str, Any] = chunk | {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]
    }
    return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\n".encode()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        pytest.param(JSON, _chat_with_function_call(), id="reply"),
        pytest.param(STREAM, _chat_stream_with_function_call(), id="stream"),
    ],
)
async def test_a_legacy_function_call_reply_is_unreadable(
    content_type: str, body: bytes
) -> None:
    with pytest.raises(UnreadableError, match="legacy `function_call`"):
        await read_reply("openai", content_type, body)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "then",
    [
        pytest.param(
            [{"role": "assistant", "content": None, "function_call": _FUNCTION_CALL}],
            id="a call",
        ),
        pytest.param(
            [{"role": "function", "name": "get_weather", "content": "Rain."}],
            id="its result",
        ),
    ],
)
async def test_a_legacy_function_call_request_is_unreadable(
    then: list[dict[str, Any]],
) -> None:
    body = openai_request("chat", then=then) | {"model": "gpt-test"}

    with pytest.raises(UnreadableError, match="legacy `function_call`"):
        await read_request("openai", json.dumps(body).encode())


def _internal(value: Any) -> str:
    # as inspect_ai's providers write an item's `internal` into its text
    encoded = base64.b64encode(json.dumps(value).encode()).decode()
    return f"<content-internal>{encoded}</content-internal>"


# a span inspect_ai reads, then one it would drop unread
_TWO_SPANS = (
    f"Rain. {_internal({'id': 1})} <content-internal>A storm.</content-internal>"
)


def _chat_result(*parts: str) -> dict[str, Any]:
    return openai_request(
        "chat",
        then=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": _FUNCTION_CALL,
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": part} for part in parts],
            },
        ],
    ) | {"model": "gpt-test"}


def _chat_said(*then: dict[str, Any]) -> dict[str, Any]:
    return openai_request("chat", then=list(then)) | {"model": "gpt-test"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_chat_result("Rain.", _TWO_SPANS), id="chat, a tool result"),
        pytest.param(
            _chat_said({"role": "assistant", "content": _TWO_SPANS}),
            id="chat, an assistant's text",
        ),
        pytest.param(
            _chat_said(
                {"role": "user", "content": [{"type": "text", "text": _TWO_SPANS}]}
            ),
            id="chat, a user's text part",
        ),
        pytest.param(
            _chat_said({"role": "user", "content": [{"text": _TWO_SPANS}]}),
            id="chat, a user's text part written as text alone",
        ),
        pytest.param(
            {"model": "gpt-test", "input": [{"role": "user", "content": "Hi"}]}
            | {
                "input": [
                    {"role": "user", "content": "Hi"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _TWO_SPANS}],
                    },
                ]
            },
            id="responses, an assistant's text",
        ),
    ],
)
async def test_a_request_with_text_inspect_ai_would_drop_is_unreadable(
    body: dict[str, Any],
) -> None:
    with pytest.raises(UnreadableError, match="more than one <content-internal>"):
        await read_request("openai", json.dumps(body).encode())


def _write_file(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"text": text}),
                },
            }
        ],
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            _chat_said({"role": "user", "content": _TWO_SPANS}),
            id="chat, a user's text",
        ),
        pytest.param(
            _chat_said(
                _write_file(_TWO_SPANS),
                {"role": "tool", "tool_call_id": "call_1", "content": "Written."},
            ),
            id="chat, a call's arguments",
        ),
        pytest.param(
            {
                "model": "gpt-test",
                "input": _result_parts({"type": "input_text", "text": _TWO_SPANS}),
            },
            id="responses, a tool result",
        ),
    ],
)
async def test_text_inspect_ai_reads_whole_is_read(body: dict[str, Any]) -> None:
    messages = await read_request("openai", json.dumps(body).encode())

    read = [m.text for m in messages] + [
        json.dumps(call.arguments)
        for m in messages
        if isinstance(m, ChatMessageAssistant)
        for call in m.tool_calls or []
    ]
    assert any("A storm." in text for text in read)


@pytest.mark.anyio
async def test_a_chat_reply_is_read_whole() -> None:
    reply = json.loads(openai_recorded("chat")[1])
    reply["choices"][0]["message"]["content"] = _TWO_SPANS

    output = await read_reply("openai", JSON, json.dumps(reply).encode())

    assert "A storm." in output.message.text


@pytest.mark.anyio
async def test_an_item_written_into_its_text_by_inspect_ai_is_read() -> None:
    body = _chat_result(f"Rain. {_internal({'id': 1})}")

    messages = await read_request("openai", json.dumps(body).encode())

    assert messages[-1].text == "Rain."


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content",
    [pytest.param(None, id="audio alone"), pytest.param("Rain.", id="with text")],
)
async def test_earlier_audio_from_the_model_makes_the_request_unreadable(
    content: str | None,
) -> None:
    # inspect_ai's reader drops the reference, and a step would be told the
    # model had said nothing, or only the text
    then = [
        {"role": "assistant", "content": content, "audio": {"id": "audio_1"}},
        {"role": "user", "content": "And tomorrow?"},
    ]

    with pytest.raises(UnreadableError, match="audio"):
        await Handler(storm_rule()).request(_request("chat", then=then))


@pytest.mark.anyio
async def test_audio_from_the_user_is_read() -> None:
    audio = {
        "type": "input_audio",
        "input_audio": {"data": "UklGRg==", "format": "wav"},
    }
    then = [
        {"role": "user", "content": [{"type": "text", "text": "Hear this."}, audio]}
    ]

    assert await Handler(storm_rule()).request(_request("chat", then=then)) == Pass()


@protocol
def rejects_the_computer() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("clicks", message="Clicking is not allowed.")

    return decide


@pytest.mark.anyio
async def test_a_rejected_computer_call_is_put_back_with_why() -> None:
    handler = Handler(rejects_the_computer())
    await handler.request(_request("responses", "a"))
    computer = {
        "type": "computer_call",
        "id": "cu_1",
        "call_id": "call_c",
        "actions": [{"type": "click", "x": 1, "y": 2, "button": "left"}],
        "pending_safety_checks": [],
        "status": "completed",
    }
    await handler.reply(
        replace(
            _reply("responses", "a"),
            body=_changed("responses", output=[computer]),
        )
    )

    again = await handler.request(_request("responses", "b"))

    assert isinstance(again, Replace)
    added = _conversation("responses", again.body)[
        len(_conversation("responses", _request("responses").body)) :
    ]
    # a computer call's output can hold only a screenshot, so why it didn't
    # run follows in a message of its own
    assert [entry.get("type") or entry["role"] for entry in added] == [
        "computer_call",
        "computer_call_output",
        "user",
    ]
    assert "Clicking is not allowed." in json.dumps(added[-1])


def _calls_with(arguments: str) -> bytes:
    # the recorded reply's calls, each with `arguments`
    reply = json.loads(openai_recorded("chat")[1])
    for call in reply["choices"][0]["message"]["tool_calls"]:
        call["function"]["arguments"] = arguments
    return json.dumps(reply).encode()


def _function_calls_with(arguments: str) -> bytes:
    reply = json.loads(openai_recorded("responses")[1])
    for item in reply["output"]:
        if item["type"] == "function_call":
            item["arguments"] = arguments
    return json.dumps(reply).encode()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_calls_with('"Tokyo"'), id="chat"),
        pytest.param(_function_calls_with('"Tokyo"'), id="responses"),
    ],
)
async def test_a_reply_whose_call_arguments_arent_an_object_is_unreadable(
    body: bytes,
) -> None:
    with pytest.raises(UnreadableError, match="aren't a JSON object"):
        await read_reply("openai", JSON, body)


@pytest.mark.anyio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_a_request_whose_call_arguments_arent_an_object_is_unreadable(
    api: str,
) -> None:
    then = openai_ran(api, "rain")
    for entry in then:
        if entry.get("type") == "function_call":
            entry["arguments"] = '"Tokyo"'
        for call in cast("list[dict[str, Any]]", entry.get("tool_calls") or []):
            call["function"]["arguments"] = '"Tokyo"'
    body = openai_request(api, then=then) | {"model": "gpt-test"}

    with pytest.raises(UnreadableError, match="aren't a JSON object"):
        await read_request("openai", json.dumps(body).encode())


@pytest.mark.anyio
async def test_a_call_with_no_arguments_is_read() -> None:
    output = await read_reply("openai", JSON, _calls_with(""))

    assert [call.arguments for call in output.message.tool_calls or []] == [{}] * 3


def _assistant_said(*parts: str | dict[str, Any]) -> bytes:
    body = {
        "model": "gpt-test",
        "input": [
            {"role": "user", "content": "How is the weather?"},
            *(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": part}
                        if isinstance(part, str)
                        else part
                    ],
                }
                for part in parts
            ),
            {"role": "user", "content": "And tomorrow?"},
        ],
    }
    return json.dumps(body).encode()


def _encoded(json_text: str) -> str:
    # a span holding exactly this JSON, however it is spaced
    return f"<content-internal>{base64.b64encode(json_text.encode()).decode()}</content-internal>"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("first", "second", "refusal"),
    [
        pytest.param('{"id": "msg_1"}', '{"id": "msg_1"}', False, id="one encoding"),
        pytest.param('{"id":"msg_1"}', '{"id": "msg_1"}', False, id="two encodings"),
        pytest.param('{"id": "msg_1"}', '{"id": "msg_1"}', True, id="a refusal"),
    ],
)
async def test_two_assistant_texts_with_one_internal_value_are_unreadable(
    first: str, second: str, refusal: bool
) -> None:
    dangerous = f"A dangerous storm. {_encoded(first)}"
    body = _assistant_said(
        {"type": "refusal", "refusal": dangerous} if refusal else dangerous,
        f"A quiet day. {_encoded(second)}",
    )

    with pytest.raises(UnreadableError, match="same <content-internal> value"):
        await read_request("openai", body)


@pytest.mark.anyio
async def test_one_assistant_text_sent_twice_is_read_once() -> None:
    once = "A quiet day. " + _encoded('{"id": "msg_1"}')
    again = "A quiet day. " + _encoded('{"id":"msg_1"}')

    messages = await read_request("openai", _assistant_said(once, again))

    assert [m.text for m in messages if m.role == "assistant"] == ["A quiet day."]


@pytest.mark.anyio
async def test_a_refusal_merged_into_an_earlier_turn_is_unreadable() -> None:
    # inspect_ai merges assistant(calls) -> tool -> assistant(content), a turn
    # some scaffolds split, keeping only the second assistant's content
    body = _chat_said(
        _write_file("Rain."),
        {"role": "tool", "tool_call_id": "call_1", "content": "Written."},
        {"role": "assistant", "content": None, "refusal": "A dangerous storm."},
    )

    with pytest.raises(UnreadableError, match="refusal"):
        await read_request("openai", json.dumps(body).encode())


@pytest.mark.anyio
async def test_a_refusal_that_is_not_merged_is_read() -> None:
    body = _chat_said(
        {"role": "assistant", "content": None, "refusal": "A dangerous storm."}
    )

    messages = await read_request("openai", json.dumps(body).encode())

    assert messages[-1].text == "A dangerous storm."


@protocol
def ends_at_a_custom_result() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        if step.call.type == "custom":
            return Decision.terminate("a custom tool")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_custom_call_is_judged_apart_from_a_function_call_like_it() -> None:
    handler = Handler(ends_at_a_custom_result())
    # read alike, as a call to apply_patch with {"input": "hello"}, apart from
    # the kind of call
    function = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "arguments": json.dumps({"input": "hello"}),
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Done."},
    ]
    custom = [
        {
            "type": "custom_tool_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "input": "hello",
        },
        {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Done."},
    ]

    first = await handler.request(_request("responses", "a", then=function))
    second = await handler.request(_request("responses", "b", then=custom))

    assert (first, isinstance(second, Refuse)) == (Pass(), True)


@pytest.mark.anyio
async def test_texts_with_one_internal_value_in_different_turns_are_read() -> None:
    body = json.loads(
        _assistant_said(
            f"A dangerous storm. {_internal({'id': 'msg_1'})}",
            f"A quiet day. {_internal({'id': 'msg_1'})}",
        )
    )
    # a user message between the two ends the first turn
    body["input"].insert(2, {"role": "user", "content": "Go on."})

    messages = await read_request("openai", json.dumps(body).encode())

    assert [m.text for m in messages if m.role == "assistant"] == [
        "A dangerous storm.",
        "A quiet day.",
    ]
