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
)
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._reply_stream import KEEPALIVE
from inspect_sentinel.sidecar._wire import UnreadableError
from tests.sidecar._traffic import JSON, STREAM, message, request, weather
from tests.sidecar.test_early_turn import _exchange, noting

_FORMATS = ["anthropic", "chat", "responses"]


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
async def test_text_goes_out_before_the_sentinel_answers_and_a_call_only_after() -> (
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
            await judging.wait()
            # the text is out while the sentinel is still deciding the call
            released = await outputs.__anext__()
            assert isinstance(released, Release)
            assert b"text_delta" in released.data
            assert b"tool_use" not in released.data
            go.set()
            call = await outputs.__anext__()
            assert isinstance(call, Release) and b"tool_use" in call.data
            await stream.feed(b"\n\n".join(events[call_stop + 1 :]))
            await stream.end()
            rest = [output async for output in outputs]
            assert rest[-1] == Finished()


@pytest.mark.anyio
async def test_a_turn_of_calls_alone_all_rejected_is_refused_with_its_status() -> None:
    asked, reply = _exchange("chat", inspect=False)
    handler = Handler(noting([], "reject", "every"), release="early", reject_status=503)
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    assert len(outputs) == 1 and isinstance(outputs[0], Refuse)
    assert outputs[0].status == 503


@pytest.mark.anyio
@pytest.mark.parametrize("name", _FORMATS)
async def test_a_turn_ended_after_content_went_out_raises_in_the_sdk_with_the_field(
    name: str,
) -> None:
    asked, reply = _exchange(name, inspect=True)
    handler = Handler(noting([], "terminate", "Lima"), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    assert isinstance(outputs[-1], ErrorEvent)
    raised = _read_with_sdk(name, _sent(outputs))
    body: Any = raised.body
    field = (
        body["sentinel"]
        if name == "anthropic"
        else body["provider_specific_fields"]["sentinel"]
    )
    assert field["outcome"] == "terminate"


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
async def test_a_sentinel_given_the_finished_turn_keeps_a_stream_held_whole() -> None:
    asked, reply = _exchange("anthropic", inspect=False)
    handler = Handler(finished_turns(), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), reply.body, 61)

    assert outputs == [Release(reply.body), Finished()]


async def _result(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.reason


@pytest.mark.anyio
async def test_a_call_being_judged_when_the_client_goes_is_decided_and_the_turn_cut_off() -> (
    None
):
    asked, reply = _exchange("anthropic", inspect=True)
    go, judging = anyio.Event(), anyio.Event()
    handler = Handler(waits_for(go, judging), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            await stream.feed(reply.body)
            await judging.wait()
            await stream.abort()
            go.set()

    records = await handler.records("x")
    # the call was decided, and kept, and the turn is recorded as cut off
    assert records is not None and records
    assert await _result(handler) == ("not_judged", "cut_off")


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
async def test_a_chat_call_changed_after_it_was_judged_fails_the_turn() -> None:
    asked, reply = _exchange("chat", inspect=False)
    handler = Handler(noting([], "reject", "Cairo"), release="early")
    await handler.request(asked)
    events = reply.body.split(b"\n\n")
    late = json.loads(events[-4].removeprefix(b"data: "))
    # a fragment for the first call, after the third has started
    late["choices"][0]["delta"] = {
        "tool_calls": [{"index": 0, "function": {"arguments": " "}}]
    }
    late["choices"][0]["finish_reason"] = None
    body = b"\n\n".join(
        [*events[:-4], b"data: " + json.dumps(late).encode(), *events[-4:]]
    )

    outputs = await _streamed(handler.reply_stream(_start(reply)), body, 61)

    assert isinstance(outputs[-1], ErrorEvent)
    assert b"couldn't be read" in outputs[-1].data


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


@pytest.mark.anyio
async def test_no_call_is_judged_after_the_client_has_gone() -> None:
    asked, reply = _exchange("anthropic", inspect=True)
    go, judging = anyio.Event(), anyio.Event()
    seen: list[str] = []

    @protocol
    def counts() -> Protocol:
        """Note each call judged, and wait for `go` at the first."""

        async def decide(
            context: Context, step: BeforeToolCall, escalations: Escalations
        ) -> Decision | None:
            seen.append(step.call.id)
            judging.set()
            await go.wait()
            return Decision.proceed()

        return decide

    handler = Handler(counts(), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            # every call arrives in one chunk
            await stream.feed(reply.body)
            await judging.wait()
            await stream.abort()
            go.set()

    # the call being judged was decided, and no other was asked about
    assert len(seen) == 1


@pytest.mark.anyio
async def test_no_call_is_judged_once_the_client_goes_during_a_delivery() -> None:
    asked, reply = _exchange("anthropic", inspect=False)
    seen: list[tuple[str, bool, str]] = []
    handler = Handler(noting(seen, "reject", "Cairo"), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))
    events = reply.body.split(b"\n\n")
    first_text = next(i for i, e in enumerate(events) if b"text_delta" in e)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            # the opening and the first text go out, and wait to be collected
            await stream.feed(b"\n\n".join(events[: first_text + 1]) + b"\n\n")
            await anyio.wait_all_tasks_blocked()
            # the rest of the text waits to be delivered behind it, before the
            # first call is judged
            await stream.feed(b"\n\n".join(events[first_text + 1 :]))
            await anyio.wait_all_tasks_blocked()
            await stream.abort()

    assert seen == []


@pytest.mark.anyio
async def test_a_turn_whose_end_fails_its_check_is_recorded_as_not_judged() -> None:
    asked, reply = _exchange("responses", inspect=True)
    # the response the stream ends with holds two calls in each other's place
    lines = reply.body.decode().split("\n")
    for at, line in enumerate(lines):
        if line.startswith("data: {") and '"response.completed"' in line:
            data = json.loads(line[6:])
            output = data["response"]["output"]
            output[1], output[2] = output[2], output[1]
            lines[at] = "data: " + json.dumps(data)
    body = "\n".join(lines).encode()
    handler = Handler(noting([], "reject", "Cairo"), release="early")
    await handler.request(asked)

    outputs = await _streamed(handler.reply_stream(_start(reply)), body, 61)

    assert isinstance(outputs[-1], ErrorEvent)
    assert await _result(handler) == ("not_judged", "unreadable")


def _chat_chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


@pytest.mark.anyio
async def test_no_call_is_judged_once_the_client_goes_among_calls_complete_together() -> (
    None
):
    asked, reply = _exchange("chat", inspect=False)
    calls = [
        {
            "index": index,
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": city}),
            },
        }
        for index, city in enumerate(["Paris", "Tokyo", "Lima"])
    ]
    # three calls in one chunk, all complete at the choice's finish
    body = (
        _chat_chunk({"role": "assistant", "content": None})
        + _chat_chunk({"tool_calls": calls})
        + _chat_chunk({}, "tool_calls")
        + b"data: [DONE]\n\n"
    )
    go, judging = anyio.Event(), anyio.Event()
    seen: list[str] = []

    @protocol
    def counts() -> Protocol:
        """Note each call judged, and wait for `go` at the first."""

        async def decide(
            context: Context, step: BeforeToolCall, escalations: Escalations
        ) -> Decision | None:
            seen.append(step.call.id)
            judging.set()
            await go.wait()
            return Decision.proceed()

        return decide

    handler = Handler(counts(), release="early")
    await handler.request(asked)
    stream = handler.reply_stream(_start(reply))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            await stream.feed(body)
            await judging.wait()
            await stream.abort()
            go.set()

    assert seen == ["call_0"]


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
