"""Guest-side `Host` implementations that forward to the WIT imports."""

import json
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class BlockingHost:
    """Over the `host` interface: imports block (the host may suspend us on a fiber)."""

    def __init__(self, imports: Any) -> None:
        self._imports = imports

    async def generate(self, input: str, *, role: str | None = None) -> dict[str, Any]:
        return json.loads(self._imports.generate(json.dumps({"input": input, "role": role})))


class AsyncHost:
    """Over the `host-async` interface: component-model async imports."""

    def __init__(self, imports: Any) -> None:
        self._imports = imports

    async def generate(self, input: str, *, role: str | None = None) -> dict[str, Any]:
        return json.loads(await self._imports.generate(json.dumps({"input": input, "role": role})))


def run_blocking(coro: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine that never suspends (all awaits complete synchronously)."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise RuntimeError("monitor suspended in the blocking world; use the async world")
