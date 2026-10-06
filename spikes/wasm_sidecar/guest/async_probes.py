"""Which asyncio/anyio features work on componentize-py's component-model loop."""

import asyncio
import contextvars
import time
from typing import Any

import anyio
import anyio._backends._asyncio  # noqa: F401  (loaded dynamically; must be bundled)
from m1.guest_host import AsyncHost

VAR: contextvars.ContextVar[str] = contextvars.ContextVar("VAR", default="unset")


async def _try(name: str, fn: Any) -> dict[str, Any]:
    t = time.perf_counter()
    try:
        result = await fn()
        return {
            "probe": name,
            "ok": True,
            "result": repr(result)[:160],
            "ms": round((time.perf_counter() - t) * 1000, 1),
        }
    except BaseException as ex:
        import traceback

        while hasattr(ex, "exceptions"):  # unwrap an exception group
            ex = ex.exceptions[0]

        where = [
            f"{f.filename.rsplit('/', 2)[-1]}:{f.lineno} {f.name}"
            for f in traceback.extract_tb(ex.__traceback__)[-3:]
        ]
        return {
            "probe": name,
            "ok": False,
            "error": f"{type(ex).__name__}: {ex}"[:300],
            "at": where,
        }


async def run(host_async: Any, only: str | None = None) -> list[dict[str, Any]]:
    host = AsyncHost(host_async)

    async def gen() -> str:
        out = await host.generate("ping")
        return out["choices"][0]["message"]["content"][-8:]

    async def gather() -> Any:
        return await asyncio.gather(gen(), gen(), gen())

    async def task_group() -> Any:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(gen()) for _ in range(3)]
        return [t.result() for t in tasks]

    async def sleep0() -> Any:
        await asyncio.sleep(0)
        return "ok"

    async def sleep_timer() -> Any:
        await asyncio.sleep(0.01)
        return "ok"

    async def wait_for() -> Any:
        return await asyncio.wait_for(gen(), timeout=5)

    async def contextvar() -> Any:
        VAR.set("outer")

        async def child() -> str:
            await gen()
            return VAR.get()

        return await asyncio.gather(child(), child())

    async def anyio_tg() -> Any:
        results: list[str] = []

        async def one() -> None:
            results.append(await gen())

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(one)
        return results

    async def anyio_cancel_scope() -> Any:
        with anyio.move_on_after(5):
            return await gen()

    async def cancel_task() -> Any:
        task = asyncio.ensure_future(gen())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "not cancelled"

    async def wait_for_fires() -> Any:
        try:
            await asyncio.wait_for(gen(), timeout=0.01)
        except TimeoutError:
            return "timed out"
        return "did not time out"

    async def anyio_fail_after_fires() -> Any:
        try:
            with anyio.fail_after(0.01):
                await gen()
        except TimeoutError:
            return "timed out"
        return "did not time out"

    async def loop_time() -> Any:
        return asyncio.get_running_loop().time()

    probes = {
        "await host.generate": gen,
        "asyncio.gather x3": gather,
        "asyncio.TaskGroup x3": task_group,
        "asyncio.sleep(0)": sleep0,
        "asyncio.sleep(0.01)": sleep_timer,
        "asyncio.wait_for(timeout=5)": wait_for,
        "contextvars across gather": contextvar,
        "cancel an in-flight task": cancel_task,
        "asyncio.wait_for fires (10ms)": wait_for_fires,
        "anyio.fail_after fires (10ms)": anyio_fail_after_fires,
        "loop.time()": loop_time,
        "anyio task group x3": anyio_tg,
        "anyio.move_on_after": anyio_cancel_scope,
    }
    if only == "list":
        return [{"probes": list(probes)}]
    return [await _try(name, fn) for name, fn in probes.items() if only in (None, name)]
