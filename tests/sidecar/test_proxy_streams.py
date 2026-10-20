import base64
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse, Replace
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._json_rpc import (
    INVALID_PARAMS,
    UNKNOWN_STREAM,
    UNREADABLE,
)
from inspect_sentinel.sidecar._proxy_endpoint import PATH, proxy_app
from tests.sidecar._traffic import (
    STREAM,
    anthropic_recorded,
    reply,
    request,
    weather_rule,
)

_BODY = anthropic_recorded("parallel-tool-calls")


@asynccontextmanager
async def _serving(
    handler: Handler, **options: Any
) -> AsyncGenerator[httpx.AsyncClient]:
    # the app's lifespan, in which reply streams run, as uvicorn runs it
    app = proxy_app(handler, **options)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
        ) as client:
            yield client


async def _call(
    client: httpx.AsyncClient, method: str, **params: Any
) -> dict[str, Any]:
    answered = await client.post(
        PATH, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    )
    return answered.json()


def _begun(status: int = 200, content_type: str = STREAM) -> dict[str, Any]:
    return {
        "id": "x",
        "provider": "anthropic",
        "model": "claude-test",
        "user": "user-1",
        "status": status,
        "content_type": content_type,
    }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def _feed(client: httpx.AsyncClient, body: bytes, size: int = 97) -> None:
    for seq, at in enumerate(range(0, len(body), size)):
        fed = await _call(
            client, "reply_chunk", id="x", seq=seq, data=_b64(body[at : at + size])
        )
        assert fed == {"jsonrpc": "2.0", "id": 1, "result": {"taken": True}}
    await _call(client, "end_reply", id="x")


async def _collect(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    # what the caller is sent, polled as a proxy hook polls, to the end
    outputs: list[dict[str, Any]] = []
    while not outputs or outputs[-1]["kind"] not in (
        "finished",
        "refuse",
        "error_event",
    ):
        answered = await _call(client, "next_release", id="x", after=len(outputs))
        outputs += answered["result"]["outputs"]
        assert answered["result"]["position"] == len(outputs)
    return outputs


def _sent(outputs: list[dict[str, Any]]) -> bytes:
    return b"".join(
        base64.b64decode(output["data"])
        for output in outputs
        if output["kind"] in ("release", "error_event")
    )


@pytest.mark.anyio
@pytest.mark.parametrize("release", ["whole", "early"])
@pytest.mark.parametrize(
    ("action", "city"),
    [("reject", "Tokyo"), ("reject", "Cairo"), ("terminate", "Tokyo")],
)
async def test_a_reply_handed_over_as_it_arrives_comes_to_what_judge_reply_does(
    release: str, action: str, city: str
) -> None:
    whole = Handler(weather_rule(action, city))
    await whole.request(request())
    expected = await whole.reply(reply(_BODY, content_type=STREAM))
    handler = Handler(weather_rule(action, city), release=release)  # type: ignore[arg-type]
    await handler.request(request())

    with anyio.fail_after(10):
        async with _serving(handler) as client:
            assert (await _call(client, "begin_reply", **_begun()))["result"] == {
                "taken": True
            }
            outputs: list[dict[str, Any]] = []
            async with anyio.create_task_group() as group:

                async def collect() -> None:
                    outputs.extend(await _collect(client))

                group.start_soon(collect)
                await _feed(client, _BODY)

    last = outputs[-1]
    if isinstance(expected, Refuse):
        if last["kind"] == "refuse":
            assert (last["status"], last["message"]) == (
                expected.status,
                expected.message,
            )
        else:
            assert last["kind"] == "error_event"
            assert expected.message.encode() in _sent([last])
    else:
        assert last["kind"] == "finished"
        given = expected.body if isinstance(expected, Replace) else _BODY
        assert _sent(outputs) == given


@protocol
def waits_for(go: anyio.Event, asked: anyio.Event) -> Protocol:
    """Let every call go once `go` is set, setting `asked` when first asked."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        asked.set()
        await go.wait()
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_text_is_released_to_the_proxy_hook_before_the_sentinel_answers() -> None:
    go, asked = anyio.Event(), anyio.Event()
    handler = Handler(waits_for(go, asked), release="early")
    await handler.request(request())

    with anyio.fail_after(10):
        async with _serving(handler) as client:
            await _call(client, "begin_reply", **_begun())
            await _call(client, "reply_chunk", id="x", seq=0, data=_b64(_BODY))
            await asked.wait()
            answered = await _call(client, "next_release", id="x", after=0)
            released = _sent(answered["result"]["outputs"])
            assert b"text_delta" in released and b"tool_use" not in released
            go.set()
            await _call(client, "end_reply", id="x")
            outputs = (
                answered["result"]["outputs"]
                + (
                    await _call(
                        client,
                        "next_release",
                        id="x",
                        after=answered["result"]["position"],
                    )
                )["result"]["outputs"]
            )
            while outputs[-1]["kind"] != "finished":
                outputs += (
                    await _call(client, "next_release", id="x", after=len(outputs))
                )["result"]["outputs"]

    assert _sent(outputs) == _BODY


@pytest.mark.anyio
async def test_every_call_can_be_sent_again() -> None:
    handler = Handler(weather_rule("reject", "Cairo"))
    await handler.request(request())

    with anyio.fail_after(10):
        async with _serving(handler) as client:
            for _ in range(2):
                assert "result" in await _call(client, "begin_reply", **_begun())
            other = await _call(client, "begin_reply", **(_begun() | {"status": 201}))
            assert other["error"]["code"] == INVALID_PARAMS
            half = len(_BODY) // 2
            for _ in range(2):
                assert "result" in await _call(
                    client, "reply_chunk", id="x", seq=0, data=_b64(_BODY[:half])
                )
            changed = await _call(client, "reply_chunk", id="x", seq=0, data=_b64(b"x"))
            assert changed["error"]["code"] == INVALID_PARAMS
            skipped = await _call(client, "reply_chunk", id="x", seq=2, data=_b64(b"x"))
            assert skipped["error"]["code"] == INVALID_PARAMS
            await _call(client, "reply_chunk", id="x", seq=1, data=_b64(_BODY[half:]))
            for _ in range(2):
                await _call(client, "end_reply", id="x")
            outputs = await _collect(client)
            # the last answer lost, and asked for again from the same place:
            # the same end
            last = len(outputs) - 1
            again = await _call(client, "next_release", id="x", after=last)
            assert again["result"]["outputs"] == outputs[last:]
            # once the proxy hook has the end, the stream is let go of
            done = await _call(client, "next_release", id="x", after=len(outputs))
            assert done["result"]["outputs"] == []
            gone = await _call(client, "next_release", id="x", after=len(outputs))
            assert gone["error"]["code"] == UNKNOWN_STREAM

    assert _sent(outputs) == _BODY


@pytest.mark.anyio
async def test_a_stream_never_begun_is_unknown() -> None:
    async with _serving(Handler(weather_rule())) as client:
        answered = await _call(client, "next_release", id="nowhere", after=0)

    assert answered["error"]["code"] == UNKNOWN_STREAM


@pytest.mark.anyio
async def test_a_wait_with_nothing_ready_is_not_the_end() -> None:
    handler = Handler(weather_rule())
    await handler.request(request())

    async with _serving(handler, poll_seconds=0.01) as client:
        await _call(client, "begin_reply", **_begun())
        answered = await _call(client, "next_release", id="x", after=0)

    assert answered["result"] == {"position": 0, "outputs": []}


@pytest.mark.anyio
async def test_a_reply_that_cant_be_judged_is_an_error_asked_again_and_again() -> None:
    handler = Handler(weather_rule())
    await handler.request(request(sample="sample_123"))

    with anyio.fail_after(10):
        async with _serving(handler) as client:
            await _call(client, "begin_reply", **_begun())
            await _feed(client, b"event: nonsense\ndata: {}\n\n")
            for _ in range(2):
                answered = await _call(client, "next_release", id="x", after=0)
                assert answered["error"]["code"] == UNREADABLE

    assert await _latest(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_an_aborted_stream_is_recorded_as_cut_off_and_let_go_of() -> None:
    handler = Handler(weather_rule())
    await handler.request(request(sample="sample_123"))

    with anyio.fail_after(10):
        async with _serving(handler) as client:
            await _call(client, "begin_reply", **_begun())
            await _call(client, "reply_chunk", id="x", seq=0, data=_b64(_BODY[:100]))
            for _ in range(2):
                assert "result" in await _call(client, "abort_reply", id="x")
            answered = await _call(client, "next_release", id="x", after=0)
            assert answered["error"]["code"] == UNKNOWN_STREAM
            while (await _latest(handler))[0] != "not_judged":
                await anyio.sleep(0.01)

    assert await _latest(handler) == ("not_judged", "cut_off")


@pytest.mark.anyio
async def test_a_stream_its_proxy_hook_forgot_expires() -> None:
    handler = Handler(weather_rule())
    await handler.request(request(sample="sample_123"))

    with anyio.fail_after(10):
        async with _serving(handler, stream_expiry=0.05) as client:
            await _call(client, "begin_reply", **_begun())
            # no call names the stream again, and the sweep aborts it
            while (await _latest(handler))[0] != "not_judged":
                await anyio.sleep(0.01)

    assert await _latest(handler) == ("not_judged", "cut_off")


@pytest.mark.anyio
async def test_a_proxy_hook_that_doesnt_collect_holds_back_its_chunks() -> None:
    handler = Handler(weather_rule())
    await handler.request(request())
    error = json.dumps({"type": "error", "error": {"message": "x" * 400}}).encode()
    fed = 0
    outputs: list[dict[str, Any]] = []

    with anyio.fail_after(10):
        async with _serving(handler, max_pending_bytes=100) as client:
            await _call(
                client,
                "begin_reply",
                **_begun(status=500, content_type="application/json"),
            )

            async def feed() -> None:
                nonlocal fed
                for seq in range(20):
                    await _call(
                        client, "reply_chunk", id="x", seq=seq, data=_b64(error)
                    )
                    fed += 1
                await _call(client, "end_reply", id="x")

            async with anyio.create_task_group() as group:
                group.start_soon(feed)
                await anyio.wait_all_tasks_blocked()
                # nothing collected: the stream, and with it the chunks, wait
                assert fed < 20
                outputs += await _collect(client)

    assert _sent(outputs) == error * 20


async def _latest(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.reason


@pytest.mark.anyio
# asyncio's own tasks are listed by name, as anyio names them
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_an_abort_lets_go_of_a_stream_waiting_for_room() -> None:
    import asyncio

    handler = Handler(weather_rule())
    await handler.request(request())
    error = json.dumps({"type": "error", "error": {"message": "x" * 400}}).encode()

    with anyio.fail_after(10):
        async with _serving(handler, max_pending_bytes=100) as client:
            await _call(
                client,
                "begin_reply",
                **_begun(status=500, content_type="application/json"),
            )
            for seq in range(4):
                await _call(client, "reply_chunk", id="x", seq=seq, data=_b64(error))
            # nothing collected, so the stream waits for room
            await anyio.wait_all_tasks_blocked()
            assert any(
                task.get_name().endswith("._collect") for task in asyncio.all_tasks()
            )
            await _call(client, "abort_reply", id="x")
            await anyio.wait_all_tasks_blocked()
            waiting = [
                task
                for task in asyncio.all_tasks()
                if task.get_name().endswith("._collect") and not task.done()
            ]

    assert waiting == []


@pytest.mark.anyio
async def test_a_stream_is_let_go_of_as_soon_as_its_end_is_acknowledged() -> None:
    handler = Handler(weather_rule("reject", "Cairo"))
    await handler.request(request())

    with anyio.fail_after(10):
        async with _serving(handler, poll_seconds=30) as client:
            await _call(client, "begin_reply", **_begun())
            await _feed(client, _BODY)
            outputs = await _collect(client)
            # answered at once, though the wait for output is 30 seconds
            with anyio.fail_after(5):
                done = await _call(client, "next_release", id="x", after=len(outputs))
            assert done["result"]["outputs"] == []
            gone = await _call(client, "next_release", id="x", after=len(outputs))

    assert gone["error"]["code"] == UNKNOWN_STREAM
