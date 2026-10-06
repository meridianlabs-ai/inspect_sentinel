"""The M4 step on native CPython, for comparison with the guest's warm call.

Uses the same import path build.sh assembles (build/m4-path) and needs a
CPython 3.14 with pydantic 2.13.5 and anyio 4.11.0:
    uv run --python 3.14 --with pydantic==2.13.5 --with anyio==4.11.0 --with shortuuid python native_bench.py
"""

import time

t0 = time.perf_counter()
import asyncio  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

sys.path[:0] = ["build/m4-path", "guest"]
from m4.monitors import no_destruction, no_network, suspicion  # noqa: E402
from m4.runner import run_step  # noqa: E402

from inspect_sentinel import concurrent, threshold  # noqa: E402

import_ms = (time.perf_counter() - t0) * 1000


async def generate(request: str) -> str:
    req = json.loads(request)
    score = 0.95 if "rm -rf" in req["input"] else 0.2
    content = json.dumps({"reasoning": "mock", "score": score})
    return json.dumps(
        {
            "model": "mockllm/model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "stop_reason": "stop",
                }
            ],
        }
    )


ROOT = concurrent(
    [
        threshold(
            {"suspicion": suspicion(), "trusted": suspicion(role="trusted")},
            reject_at=0.8,
        ),
        no_network(),
        no_destruction(),
    ]
)


async def main() -> None:
    step = open("step_m4.json").read()
    times = []
    for _ in range(500):
        t = time.perf_counter()
        await run_step(ROOT, step, generate)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    print(
        f"native CPython {sys.version.split()[0]}: import {import_ms:.0f}ms ({len(sys.modules)} modules); "
        f"warm median {times[250]:.3f}ms p90 {times[450]:.3f}ms"
    )


asyncio.run(main())
