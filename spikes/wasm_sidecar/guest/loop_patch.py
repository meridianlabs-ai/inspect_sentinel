"""Patches to componentize-py's asyncio loop (componentize_py_async_support._Loop).

- `call_soon(context=None)` asserts; anyio's cancellation delivery calls it so.
  Use the caller's context, which carries the loop's per-task future state.
- `time()` raises; use the monotonic clock (wasi:clocks).
- `get_task_factory()` raises; anyio's task groups call it. Return None.
- `call_later`/`call_at` raise; implement them on a host `sleep` import. The
  timer task counts as pending work of the component-model task, because the
  loop drops its waitable set when nothing is pending and that traps while a
  host call is still joined ("resource has children"). A cancelled timer
  therefore still occupies the task until its sleep returns: the runtime
  cannot cancel an in-flight host call yet.
"""

import asyncio
import contextvars
import time
from typing import Any, Callable

import componentize_py_async_support as cpas

_Loop = cpas._Loop
_call_soon = _Loop.call_soon
_sleep: Callable[[int], Any] | None = None


def install(sleep: Callable[[int], Any]) -> None:
    global _sleep
    _sleep = sleep


def _patched_call_soon(self: Any, callback: Any, *args: Any, context: Any = None) -> asyncio.Handle:
    return _call_soon(self, callback, *args, context=context if context is not None else contextvars.copy_context())


def _time(self: Any) -> float:
    return time.monotonic()


def _call_at(self: Any, when: float, callback: Any, *args: Any, context: Any = None) -> asyncio.TimerHandle:
    if _sleep is None:
        raise NotImplementedError("loop_patch.install(sleep) not called")
    ctx = context if context is not None else contextvars.copy_context()
    handle = asyncio.TimerHandle(when, callback, args, self, ctx)
    state = cpas._future_state.get()
    state.pending_count += 1

    async def fire() -> None:
        try:
            delay = max(0.0, when - time.monotonic())
            await _sleep(int(delay * 1000))
            if not handle._cancelled:
                handle._run()
        finally:
            state.pending_count -= 1

    # The handle runs in `ctx`; the task needs its own context to enter it from.
    self.create_task(fire(), context=contextvars.copy_context())
    return handle


def _call_later(self: Any, delay: float, callback: Any, *args: Any, context: Any = None) -> asyncio.TimerHandle:
    return _call_at(self, time.monotonic() + delay, callback, *args, context=context)


def _timer_handle_cancelled(self: Any, handle: Any) -> None:
    pass


_Loop.call_soon = _patched_call_soon
_Loop._timer_handle_cancelled = _timer_handle_cancelled
_Loop.get_task_factory = lambda self: None  # anyio's TaskGroup asks for it
_Loop.time = _time
_Loop.call_at = _call_at
_Loop.call_later = _call_later


# Cancelling a task that is awaiting a host call. The runtime has no
# `subtask.cancel`, and the stock `await_result` leaves the cancelled future
# registered, so the subtask's completion later hits a cancelled future and the
# waitable set is dropped while the subtask is still joined: the instance traps
# ("resource has children"). Detach instead: the host call runs to completion,
# its result is lifted and discarded, and the component-model task stays open
# until then.
import componentize_py_runtime as _rt
from componentize_py_types import Ok as _Ok


async def _await_result(result: Any) -> Any:
    if isinstance(result, _Ok):
        return result.value
    state = cpas._future_state.get()
    waitable, promise = result.value
    future = cpas._loop.create_future()
    state.futures[waitable] = future
    if state.waitable_set is None:
        state.waitable_set = _rt.waitable_set_new()
    _rt.waitable_join(waitable, state.waitable_set)
    try:
        event = await future
    except asyncio.CancelledError:
        drain = cpas._loop.create_future()
        state.futures[waitable] = drain
        state.pending_count += 1

        def finished(f: asyncio.Future[Any]) -> None:
            _rt.promise_get_result(f.result(), promise)
            state.pending_count -= 1

        drain.add_done_callback(finished, context=contextvars.copy_context())
        raise
    return _rt.promise_get_result(event, promise)


cpas.await_result = _await_result
