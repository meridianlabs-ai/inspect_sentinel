import wasi_shims  # noqa: F401  (before anyio)

import json

import wit_world
from wit_world.imports import host_async

import loop_patch

loop_patch.install(host_async.sleep)

import anyio._backends._asyncio  # noqa: F401  (anyio loads its backend lazily; bundle it)

from inspect_sentinel import concurrent, threshold

from m4.monitors import no_destruction, no_network, suspicion
from m4.runner import run_step

ROOT = concurrent([threshold(suspicion(), reject_at=0.8), no_network(), no_destruction()])


class WitWorld(wit_world.WitWorld):
    async def run_monitor(self, step: str) -> str:
        return json.dumps(await run_step(ROOT, step, host_async.generate))
