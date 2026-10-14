import anyio
import pytest

from inspect_sentinel.sidecar import Handler, Pass, UnreadableError
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import message, reply, request, weather, weather_rule


@pytest.mark.anyio
async def test_a_run_is_kept_while_it_is_used() -> None:
    state = MemoryState(expiry=0.2)
    run = await state.run("a")

    for _ in range(3):
        await anyio.sleep(0.1)
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
