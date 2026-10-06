import wasi_shims  # noqa: F401  (before anyio)
import asyncio
import json

import wit_world
from wit_world.imports import host_async

import loop_patch

loop_patch.install(host_async.sleep)

from m1.guest_host import AsyncHost
from m1.monitor import tool_call_monitor
import async_probes


class WitWorld(wit_world.WitWorld):
    async def run_monitor(self, step: str) -> str:
        data = json.loads(step)
        if data.get("asyncprobe"):
            return json.dumps(await async_probes.run(host_async, data.get("only")))
        host = AsyncHost(host_async)
        fanout = int(data.get("fanout", 1))
        if fanout == 1:
            return json.dumps(await tool_call_monitor(data, host))
        # `concurrent()`-style: several monitors in parallel inside one task.
        reports = await asyncio.gather(*(tool_call_monitor(data, host) for _ in range(fanout)))
        return json.dumps({"reports": reports})
