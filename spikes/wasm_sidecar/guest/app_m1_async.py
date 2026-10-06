# Import-order constraints: the ssl stub before anything imports anyio, and the
# loop patches before any task runs.
import loop_patch  # noqa: F401
import wasi_shims  # noqa: F401

# isort: split
import asyncio
import json

import async_probes
import wit_world
from m1.guest_host import AsyncHost
from m1.monitor import tool_call_monitor
from wit_world.imports import host_async


class WitWorld(wit_world.WitWorld):
    async def run_monitor(self, step: str) -> str:
        data = json.loads(step)
        if data.get("spin"):
            while True:  # a runaway monitor: the host's epoch deadline must stop it
                pass
        if "alloc_mb" in data:
            try:
                block = bytearray(int(data["alloc_mb"]) * 2**20)
                return json.dumps({"allocated_mb": len(block) // 2**20})
            except MemoryError:
                return json.dumps({"error": "MemoryError"})
        if data.get("asyncprobe"):
            return json.dumps(await async_probes.run(host_async, data.get("only")))
        host = AsyncHost(host_async)
        fanout = int(data.get("fanout", 1))
        if fanout == 1:
            return json.dumps(await tool_call_monitor(data, host))
        # `concurrent()`-style: several monitors in parallel inside one task.
        reports = await asyncio.gather(
            *(tool_call_monitor(data, host) for _ in range(fanout))
        )
        return json.dumps({"reports": reports})
