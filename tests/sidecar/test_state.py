import gc
import weakref

import anyio
import pytest

from inspect_sentinel.sidecar import Handler, Pass, UnreadableError, _state
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import message, reply, request, weather, weather_rule


def _clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # The state's own clock, which the test moves on. The state reads it
    # whenever it is used and waits on nothing, so no real time has to pass.
    now = [0.0]

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return now[0]

    monkeypatch.setattr(_state, "time", Clock)
    return now


@pytest.mark.anyio
async def test_a_run_is_kept_while_it_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    state = MemoryState(expiry=0.2)
    run = await state.run("a")

    for _ in range(3):
        now[0] += 0.1
        assert await state.run("a") is run


@pytest.mark.anyio
async def test_a_run_no_request_has_named_for_its_expiry_is_dropped() -> None:
    state = MemoryState(expiry=0.05)
    run = await state.run("a")
    recorder = await state.recorder("x")

    await anyio.sleep(0.1)

    assert await state.run("a") is not run
    assert await state.records("x") is None
    assert await state.recorder("x") is not recorder


@pytest.mark.anyio
async def test_a_request_whose_reply_never_came_is_dropped() -> None:
    handler = Handler(weather_rule(), state=MemoryState(expiry=0.05))
    assert await handler.request(request()) == Pass()

    await anyio.sleep(0.1)

    with pytest.raises(UnreadableError, match="no longer keeps"):
        await handler.reply(reply(message(weather("Oslo"))))


@pytest.mark.anyio
async def test_a_rejection_is_forgotten_with_its_run() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), state=MemoryState(expiry=0.05))
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))

    await anyio.sleep(0.1)

    assert await handler.request(request("b")) == Pass()


@pytest.mark.anyio
async def test_requests_whose_replies_never_come_are_dropped_as_more_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return now[0]

    # the state's own clock only: the event loop keeps the real one
    monkeypatch.setattr(_state, "time", Clock)
    handler = Handler(weather_rule(), state=MemoryState(expiry=10))
    await handler.request(request("a"))
    recorder = weakref.ref(await handler._state.recorder("a"))  # pyright: ignore[reportPrivateUsage]

    # only more requests arrive, and no reply
    now[0] = 11.0
    await handler.request(request("b"))
    gc.collect()

    assert recorder() is None


@pytest.mark.anyio
async def test_what_is_pinned_is_kept_past_its_expiry_until_it_is_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    state = MemoryState(expiry=0.1)
    pinned = _state.Pinned("eval_789", "philadelphia", "run", "irid-1", ("x",))
    run = await state.run("run")
    other = await state.run("other")
    recorder = await state.recorder("x")
    attempt = await state.attempt("irid-1", "x", "eval_789")
    await state.keep_eval_token("eval_789", "the-evals-own-secret", 10)

    await state.pin(pinned)
    now[0] += 0.2

    # what the call needs is still there, and what it doesn't has gone
    assert await state.existing_run("run") is run
    assert await state.existing_run("other") is None
    assert await state.run("other") is not other
    assert await state.attempts("irid-1", "eval_789") == [attempt]
    assert await state.eval_token("eval_789", renew=False) == "the-evals-own-secret"
    assert (await state.recorder("x")) is recorder

    await state.unpin(pinned)

    # it expires again, counted from when it was unpinned
    assert await state.peek_run("run") is run
    now[0] += 0.2
    assert await state.peek_run("run") is None
    assert await state.eval_token("eval_789", renew=False) is None


@pytest.mark.anyio
async def test_a_pin_is_undone_by_as_many_unpins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    state = MemoryState(expiry=0.1)
    first = _state.Pinned("eval_789", "philadelphia", "run", "irid-1", ("x",))
    second = _state.Pinned("eval_789", "philadelphia", "run", "irid-2", ("y",))
    run = await state.run("run")

    # two calls of one run wait for a person, and one of them ends
    await state.pin(first)
    await state.pin(second)
    await state.unpin(first)
    now[0] += 0.2

    assert await state.existing_run("run") is run
