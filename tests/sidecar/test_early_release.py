import json
import re
from dataclasses import replace
from typing import Any, Literal

import anthropic
import anyio
import httpx2
import openai
import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import AfterGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import (
    ErrorEvent,
    Finished,
    Handler,
    Output,
    Refuse,
    Release,
    Replace,
    Reply,
    ReplyStart,
    ReplyStream,
    Request,
)
from inspect_sentinel.sidecar._errors import TimeLimitError
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._reply_stream import KEEPALIVE
from inspect_sentinel.sidecar._wire import Format, UnreadableError
from tests.sidecar._traffic import (
    JSON,
    STREAM,
    anthropic_recorded,
    inspect_headers,
    message,
    openai_recorded,
    openai_request,
    reply,
    request,
    weather,
    weather_rule,
)

_FORMATS = ["anthropic", "chat", "responses"]


_FORMAT_OF: dict[str, Format] = {
    "anthropic": "anthropic-messages",
    "chat": "openai-chat",
    "responses": "openai-responses",
}


def _exchange(name: str, *, inspect: bool) -> tuple[Request, Reply]:
    # a recorded reply calling get_weather for Paris, Tokyo and Lima at once
    if name == "anthropic":
        asked = request(sample="sample_123") if inspect else request()
        provider, body = "anthropic", anthropic_recorded("parallel-tool-calls")
    else:
        provider, (_, body) = "openai", openai_recorded(f"{name} stream")
        asked = Request(
            id="x",
            provider="openai",
            model="gpt-test",
            user="user-1",
            body=json.dumps(openai_request(name)).encode(),
            headers=inspect_headers() if inspect else {},
        )
    return asked, Reply(
        id="x",
        provider=provider,
        model=asked.model,
        user="user-1",
        status=200,
        content_type=STREAM,
        body=body,
        format=_FORMAT_OF[name],
    )


@protocol
def noting(seen: list[tuple[str, str]], action: str, city: str) -> Protocol:
    """Decide `action` for a weather lookup of `city`, or of every city, noting each call and the message it was given."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        seen.append((step.call.id, step.message))
        # the Responses recording names a city with its country, "Tokyo, Japan"
        location = str(step.call.arguments.get("location", ""))
        if city != "every" and not location.startswith(city):
            return Decision.proceed()
        if action == "modify":
            changed = replace(step.call, arguments={"location": "Oslo"})
            return Decision(action="modify", modified=changed)
        if action == "terminate":
            return Decision.terminate("ended")
        return Decision.reject(f"lookup of {city}", message=f"{city} is not allowed.")

    return decide


def _start(reply: Reply) -> ReplyStart:
    return ReplyStart(
        id=reply.id,
        provider=reply.provider,
        model=reply.model,
        user=reply.user,
        status=reply.status,
        content_type=reply.content_type,
        format=reply.format,
    )


async def _streamed(stream: ReplyStream, body: bytes, size: int) -> list[Output]:
    outputs: list[Output] = []

    async def read() -> None:
        async for output in stream:
            outputs.append(output)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            group.start_soon(read)
            for at in range(0, len(body), size):
                await stream.feed(body[at : at + size])
            await stream.end()
    return outputs


def _sent(outputs: list[Output]) -> bytes:
    return b"".join(o.data for o in outputs if isinstance(o, Release | ErrorEvent))


def _comparable(body: bytes) -> bytes:
    # inspect_ai gives a message a new id each time it is read
    return re.sub(rb'("id": )"[A-Za-z0-9]{22}"', rb'\1"id"', body)


@pytest.mark.anyio
@pytest.mark.parametrize("name", _FORMATS)
@pytest.mark.parametrize(
    ("action", "city"),
    [
        ("reject", "Tokyo"),
        ("reject", "Cairo"),
        ("modify", "Tokyo"),
        ("terminate", "Tokyo"),
        ("reject", "every"),
    ],
    ids=["one rejected", "none rejected", "one changed", "run ended", "all rejected"],
)
@pytest.mark.parametrize("size", [61, 1 << 30], ids=["chunks", "whole"])
@pytest.mark.parametrize("inspect", [False, True], ids=["any client", "inspect"])
async def test_a_reply_released_early_comes_to_what_it_does_held_whole(
    name: str, action: str, city: str, size: int, inspect: bool
) -> None:
    asked, reply = _exchange(name, inspect=inspect)
    whole = Handler(noting([], action, city))
    await whole.request(asked)
    expected = await whole.reply(reply)
    early = Handler(noting([], action, city), release="early")
    await early.request(asked)

    outputs = await _streamed(early.reply_stream(_start(reply)), reply.body, size)

    last = outputs[-1]
    if isinstance(expected, Refuse):
        # refused with its status while nothing had gone out, and with an
        # error event once something had
        if isinstance(last, Refuse):
            assert outputs == [last]
            assert (last.status, last.message) == (expected.status, expected.message)
        else:
            assert isinstance(last, ErrorEvent)
            assert expected.message.encode() in last.data
    else:
        assert last == Finished()
        given = expected.body if isinstance(expected, Replace) else reply.body
        assert _comparable(_sent(outputs)) == _comparable(given)


@protocol
def waits_for(go: anyio.Event, asked: anyio.Event) -> Protocol:
    """Let every call go, once `go` is set, setting `asked` when first asked."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        asked.set()
        await go.wait()
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_text_goes_out_as_it_arrives_and_a_call_only_once_the_turn_is_judged() -> (
    None
):
    asked, reply = _exchange("anthropic", inspect=False)
    go, judging = anyio.Event(), anyio.Event()
    handler = Handler(waits_for(go, judging), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))
    events = reply.body.split(b"\n\n")
    # the recording's text block, then its first call
    first_call = next(i for i, e in enumerate(events) if b'"tool_use"' in e)
    call_stop = next(
        i for i in range(first_call, len(events)) if b"content_block_stop" in events[i]
    )

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            outputs = stream.__aiter__()
            await stream.feed(b"\n\n".join(events[: call_stop + 1]) + b"\n\n")
            # the text is out before the turn has ended, and nothing is judged
            released = await outputs.__anext__()
            assert isinstance(released, Release)
            assert b"text_delta" in released.data
            assert b"tool_use" not in released.data
            assert not judging.is_set()
            await stream.feed(b"\n\n".join(events[call_stop + 1 :]))
            await stream.end()
            # the turn has ended and is judged, and the calls wait for it
            await judging.wait()
            go.set()
            rest = [output async for output in outputs]
            assert b"tool_use" in _sent(rest) and rest[-1] == Finished()
            assert _sent([released, *rest]) == reply.body


@pytest.mark.anyio
async def test_a_turn_of_calls_alone_all_rejected_is_refused_with_its_status() -> None:
    asked, reply = _exchange("chat", inspect=False)
    handler = Handler(noting([], "reject", "every"), release="early", reject_status=503)
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    assert len(outputs) == 1 and isinstance(outputs[0], Refuse)
    assert outputs[0].status == 503


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["anthropic", "responses"])
async def test_a_turn_ended_after_content_went_out_raises_in_the_sdk_with_the_field(
    name: str,
) -> None:
    # these recordings open with text or reasoning, which goes out first
    asked, reply = _exchange(name, inspect=True)
    handler = Handler(noting([], "terminate", "Lima"), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    assert isinstance(outputs[0], Release) and isinstance(outputs[-1], ErrorEvent)
    raised = _read_with_sdk(name, _sent(outputs))
    body: Any = raised.body
    field = (
        body["sentinel"]
        if name == "anthropic"
        else body["provider_specific_fields"]["sentinel"]
    )
    assert field["outcome"] == "terminate"


@pytest.mark.anyio
async def test_a_turn_of_calls_alone_ended_is_refused_with_its_status() -> None:
    # the Chat recording holds nothing but calls, so nothing has gone out
    asked, reply = _exchange("chat", inspect=True)
    handler = Handler(noting([], "terminate", "Lima"), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    [refused] = outputs
    assert isinstance(refused, Refuse) and refused.sentinel is not None
    assert refused.sentinel["outcome"] == "terminate"


def _read_with_sdk(name: str, body: bytes) -> Any:
    # a client of the provider's SDK, reading the stream as it was sent
    if name == "anthropic":
        transport = httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, headers={"content-type": STREAM}, content=body
            )
        )
        client = anthropic.Anthropic(
            api_key="x", http_client=httpx2.Client(transport=transport)
        )
        with pytest.raises(anthropic.APIStatusError) as raised:
            for _ in client.messages.create(
                model="m",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ):
                pass
        return raised.value
    openai_transport = httpx2.MockTransport(
        lambda request: httpx2.Response(
            200, headers={"content-type": STREAM}, content=body
        )
    )
    client_ = openai.OpenAI(
        api_key="x", http_client=httpx2.Client(transport=openai_transport)
    )
    with pytest.raises(openai.APIError) as raised_:
        if name == "chat":
            for _ in client_.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}], stream=True
            ):
                pass
        else:
            for _ in client_.responses.create(model="m", input="hi", stream=True):
                pass
    return raised_.value


@pytest.mark.anyio
async def test_a_reply_that_is_not_a_stream_is_held_whole() -> None:
    handler = Handler(noting([], "reject", "Tokyo"), release="early")
    await handler.request(request())
    body = message(weather("Paris", "toolu_1"), weather("Tokyo", "toolu_2"))
    reply = Reply(
        id="x",
        provider="anthropic",
        model="claude-test",
        user="user-1",
        status=200,
        content_type=JSON,
        body=body,
    )

    outputs = await _streamed(handler.reply_stream(_start(reply)), body, 7)

    assert len(outputs) == 2 and isinstance(outputs[0], Release)
    assert b"toolu_2" not in outputs[0].data


@protocol
def finished_turns() -> Protocol:
    """Look at each finished turn."""

    async def decide(
        context: Context, step: AfterGenerate, escalations: Escalations
    ) -> Decision | None:
        return None

    return decide


@pytest.mark.anyio
async def test_every_call_is_judged_with_the_whole_turn() -> None:
    # as in Inspect's own process, whether or not the reply is released early
    asked, reply = _exchange("anthropic", inspect=False)
    whole: list[tuple[str, str]] = []
    early: list[tuple[str, str]] = []
    held = Handler(noting(whole, "reject", "Tokyo"))
    await held.request(asked)
    await held.reply(reply)
    released = Handler(noting(early, "reject", "Tokyo"), release="early")
    await released.request(asked)

    await _streamed(released.reply_stream(_start(reply)), reply.body, 61)

    assert [message for _, message in early] == [message for _, message in whole]
    assert len(early) == 3


@pytest.mark.anyio
async def test_a_sentinel_given_the_finished_turn_still_has_text_released_early() -> (
    None
):
    asked, reply = _exchange("anthropic", inspect=False)
    handler = Handler(finished_turns(), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    # the text went out on its own, before the turn was judged
    assert len(outputs) > 2 and b"tool_use" not in _sent(outputs[:1])
    assert _sent(outputs) == reply.body and outputs[-1] == Finished()


async def _result(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.reason


@pytest.mark.anyio
async def test_a_turn_whose_client_goes_before_it_ends_is_cut_off_unjudged() -> None:
    asked, reply = _exchange("anthropic", inspect=True)
    seen: list[tuple[str, str]] = []
    handler = Handler(noting(seen, "reject", "Tokyo"), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            await stream.feed(reply.body[: len(reply.body) // 2])
            await stream.abort()

    assert seen == []
    assert await _result(handler) == ("not_judged", "cut_off")


@pytest.mark.anyio
async def test_a_turn_whose_client_goes_while_it_is_judged_is_decided() -> None:
    # as when the reply is held whole: the judgment goes on, and is kept
    asked, reply = _exchange("anthropic", inspect=True)
    go, judging = anyio.Event(), anyio.Event()
    handler = Handler(waits_for(go, judging), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            await stream.feed(reply.body)
            await stream.end()
            await judging.wait()
            await stream.abort()
            go.set()

    assert await _result(handler) == ("decided", None)


@pytest.mark.anyio
async def test_a_stream_that_stops_short_after_content_went_out_fails_in_the_stream() -> (
    None
):
    asked, reply = _exchange("anthropic", inspect=True)
    handler = Handler(noting([], "reject", "Cairo"), release="early")
    await handler.request(asked)
    # without its last two events, message_delta and message_stop
    cut = reply.body[: reply.body.rindex(b"event: message_delta")]

    outputs = await _streamed(handler.reply_stream(_start(reply)), cut, 61)

    assert isinstance(outputs[0], Release) and isinstance(outputs[-1], ErrorEvent)
    assert b"couldn't be read" in outputs[-1].data
    assert await _result(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_a_responses_call_whose_events_disagree_is_never_released() -> None:
    asked, reply = _exchange("responses", inspect=False)
    # the first call's deltas and arguments-done event name Tokyo, while its
    # done item, which is what is judged, names Paris
    lines = reply.body.decode().split("\n")
    for at, line in enumerate(lines):
        if not line.startswith("data: {"):
            continue
        data = json.loads(line[6:])
        if data.get("output_index") != 1:
            continue
        if data["type"] == "response.function_call_arguments.delta":
            data["delta"] = '{"location":"Tokyo, Japan"}'
        elif data["type"] == "response.function_call_arguments.done":
            data["arguments"] = '{"location":"Tokyo, Japan"}'
        lines[at] = "data: " + json.dumps(data)
    body = "\n".join(lines).encode()
    handler = Handler(noting([], "reject", "Lima"), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), body, 61)

    assert not isinstance(outputs[-1], Finished)
    assert b"call_7v64mPRsjnlrRF87GSbt0ACx" not in _sent(
        [o for o in outputs if isinstance(o, Release)]
    )


def _chat_chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


@protocol
def slowly(seconds: float, action: str) -> Protocol:
    """Decide `action`, `proceed` or `reject`, for every call, after `seconds`."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        await anyio.sleep(seconds)
        if action == "reject":
            return Decision.reject("every lookup", message="Lookups are not allowed.")
        return Decision.proceed()

    return decide


def _events_read_with_sdk(name: str, body: bytes) -> list[Any]:
    # the events a client of the provider's SDK reads from a stream it is sent
    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": STREAM}, content=body)

    http = httpx2.Client(transport=httpx2.MockTransport(answer))
    if name == "anthropic":
        client = anthropic.Anthropic(api_key="x", http_client=http)
        stream: Any = client.messages.create(
            model="m",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    elif name == "chat":
        stream = openai.OpenAI(api_key="x", http_client=http).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    else:
        stream = openai.OpenAI(api_key="x", http_client=http).responses.create(
            model="m", input="hi", stream=True
        )
    return [event.model_dump() for event in stream]


@pytest.mark.anyio
@pytest.mark.parametrize("name", _FORMATS)
async def test_a_client_waiting_on_a_judgment_is_kept_alive_with_what_its_sdk_skips(
    name: str,
) -> None:
    asked, reply = _exchange(name, inspect=False)
    handler = Handler(slowly(0.2, "proceed"), release="early", keepalive=0.02)
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    assert Release(KEEPALIVE) in outputs
    assert outputs[-1] == Finished()
    sent = _sent(outputs)
    assert _events_read_with_sdk(name, sent) == _events_read_with_sdk(
        name, sent.replace(KEEPALIVE, b"")
    )


@pytest.mark.anyio
async def test_a_turn_refused_after_a_keepalive_went_out_is_an_error_event() -> None:
    asked, reply = _exchange("chat", inspect=False)
    handler = Handler(slowly(0.2, "reject"), release="early", keepalive=0.02)
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    # the keepalive sent the status, so the refusal can't have one of its own
    assert outputs[0] == Release(KEEPALIVE)
    assert isinstance(outputs[-1], ErrorEvent)
    _read_with_sdk("chat", _sent(outputs))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("release", "keepalive", "carried"),
    [("whole", 0.02, True), ("early", 0.02, False), ("early", None, True)],
    ids=["held whole", "not carried", "turned off"],
)
async def test_no_keepalive_goes_out_where_it_isnt_wanted(
    release: Literal["whole", "early"], keepalive: float | None, carried: bool
) -> None:
    asked, reply = _exchange("chat", inspect=False)
    handler = Handler(slowly(0.2, "proceed"), release=release, keepalive=keepalive)
    await handler.request(asked)

    outputs = await _streamed(
        handler.reply_stream(_start(reply), keepalive=carried), reply.body, 61
    )

    assert Release(KEEPALIVE) not in outputs
    assert outputs[-1] == Finished()


@pytest.mark.anyio
async def test_no_keepalive_goes_out_before_the_format_is_known() -> None:
    # without a format, a refusal can be written only as an HTTP error, so the
    # status has to wait for the stream to say which API it is in
    asked, reply = _exchange("chat", inspect=False)
    handler = Handler(slowly(0, "proceed"), release="early", keepalive=0.02)
    await handler.request(asked)
    stream = handler.reply_stream(replace(_start(reply), format=None))
    outputs: list[Output] = []
    raised: list[UnreadableError] = []

    async def read() -> None:
        try:
            async for output in stream:
                outputs.append(output)
        except UnreadableError as ex:
            raised.append(ex)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            group.start_soon(read)
            await anyio.sleep(0.2)
            await stream.end()

    # nothing went out, so the proxy refuses the reply with a real status
    assert outputs == [] and raised


@pytest.mark.anyio
async def test_a_reply_released_early_has_one_time_limit_for_all_its_judging() -> None:
    # each call alone is within the limit, and the three together are not, as
    # when the reply is held whole
    asked, reply = _exchange("anthropic", inspect=True)
    whole = Handler(slowly(0.06, "proceed"), time_limit=0.1)
    await whole.request(asked)
    with pytest.raises(TimeLimitError):
        await whole.reply(reply)
    early = Handler(slowly(0.06, "proceed"), release="early", time_limit=0.1)
    await early.request(asked)

    outputs = await _streamed(early.reply_stream(_start(reply)), reply.body, 61)

    # the text had gone out, so the turn fails in the stream
    assert isinstance(outputs[-1], ErrorEvent)
    result = await early.request_result("request_456")
    assert not isinstance(result, str)
    assert (result.latest.state, result.latest.reason) == ("not_judged", "time_limit")


def _anthropic_stream(*blocks: dict[str, Any], stop_reason: str = "tool_use") -> bytes:
    # an Anthropic stream of these blocks, each whole in its start but a call's
    # arguments and a text's text, which arrive as deltas
    opened = json.loads(message(stop_reason=stop_reason))
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": opened | {"content": [], "stop_reason": None},
        }
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "tool_use":
            events += [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": block | {"input": {}},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            ]
        elif block["type"] == "text":
            events += [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            ]
        else:
            events.append(
                {"type": "content_block_start", "index": index, "content_block": block}
            )
        events.append({"type": "content_block_stop", "index": index})
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


_SEARCH: dict[str, Any] = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {"query": "Paris weather"},
}


@pytest.mark.anyio
async def test_a_server_tool_answered_in_a_reply_released_early_is_read() -> None:
    # Anthropic can start a search in one reply and give its result in the
    # next, which is read with what reading the first noted
    handler = Handler(weather_rule(), release="early")
    await handler.request(request("a"))
    await handler.reply(reply(message(_SEARCH, weather("Paris")), id="a"))
    then = [
        {"role": "assistant", "content": [_SEARCH, weather("Paris")]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Rain."}
            ],
        },
    ]
    await handler.request(request("b", then=then))
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.com/paris",
                "title": "Paris weather",
                "encrypted_content": "abc",
                "page_age": None,
            }
        ],
    }
    body = _anthropic_stream(
        result, {"type": "text", "text": "Rain."}, weather("Lima", "toolu_2")
    )
    second = reply(body, id="b", content_type=STREAM)

    outputs = await _streamed(handler.reply_stream(_start(second)), body, 61)

    assert outputs[-1] == Finished()
    assert _sent(outputs) == body


_TEXT_EVENT = {
    "anthropic": b"text_delta",
    "chat": b'"content": "Hello"',
    # the Responses recording opens with a reasoning item, which goes out as
    # it arrives
    "responses": b'"type":"reasoning"',
}


@pytest.mark.anyio
@pytest.mark.parametrize("name", _FORMATS)
async def test_a_providers_comment_holds_nothing_back(name: str) -> None:
    # a provider may send comments of its own as keepalives, which say nothing
    asked, recorded = _exchange(name, inspect=False)
    if name == "chat":
        body = (
            _chat_chunk({"role": "assistant", "content": ""})
            + _chat_chunk({"content": "Hello"})
            + _chat_chunk({"content": " there."})
            + _chat_chunk({}, finish="stop")
            + b"data: [DONE]\n\n"
        )
        recorded = replace(recorded, body=body)
    events = recorded.body.split(b"\n\n")
    events.insert(1, b": ping")
    body = b"\n\n".join(events)
    text = next(i for i, event in enumerate(events) if _TEXT_EVENT[name] in event)
    handler = Handler(slowly(0, "proceed"), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(recorded))
    first: Output | None = None
    rest: list[Output] = []

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            outputs = stream.__aiter__()
            await stream.feed(b"\n\n".join(events[: text + 1]) + b"\n\n")
            # the text goes out while the rest of the stream is still to come
            with anyio.move_on_after(2):
                first = await outputs.__anext__()
            assert first is not None, "nothing went out before the stream ended"
            assert isinstance(first, Release) and _TEXT_EVENT[name] in first.data
            await stream.feed(b"\n\n".join(events[text + 1 :]))
            await stream.end()
            rest = [output async for output in outputs]

    assert first is not None and rest[-1] == Finished()
    assert _sent([first, *rest]) == body
