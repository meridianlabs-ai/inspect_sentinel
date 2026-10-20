import json
import re
from typing import Any

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
