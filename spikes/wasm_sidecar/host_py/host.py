"""M1 host: wasmtime-py loads the blocking-world component and serves its imports.

wasmtime-py has no async host functions, so `generate` here is synchronous;
the async story is the Rust host (host_rs). Usage:

    python host_py/host.py build/m1_sync.wasm step.json [--probe] [--repeat N]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

from wasmtime import Config, Engine, Store, WasiConfig
from wasmtime.component import Component, Linker, Variant

INTERFACE = "sentinel:spike/host@0.1.0"

# Named endpoints -> (url, credential). Credentials live here, never in the guest.
ENDPOINTS = {
    "allowlist": ("https://allowlist.internal/check", "Bearer host-held-secret")
}
STORE: dict[str, str] = {}
CALLS: list[dict[str, Any]] = []


def mock_model_output(request: dict[str, Any]) -> dict[str, Any]:
    """A ModelOutput JSON as inspect_ai would serialize it."""
    return {
        "model": "mockllm/model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Reading disk usage under the user's home is in scope. SCORE: 2",
                    "source": "generate",
                    "model": "mockllm/model",
                },
                "stop_reason": "stop",
            }
        ],
        "usage": {
            "input_tokens": len(request["input"]) // 4,
            "output_tokens": 14,
            "total_tokens": len(request["input"]) // 4 + 14,
        },
        "time": 0.0,
    }


def generate(_store: Any, request: str) -> Variant:
    req = json.loads(request)
    CALLS.append({"fn": "generate", "role": req.get("role")})
    return Variant("ok", json.dumps(mock_model_output(req)))


def fetch(_store: Any, endpoint: str, request: str) -> Variant:
    CALLS.append({"fn": "fetch", "endpoint": endpoint})
    if endpoint not in ENDPOINTS:
        return Variant("err", f"unknown endpoint {endpoint!r}")
    url, credential = ENDPOINTS[endpoint]
    # A real host would POST `request` to `url` with `credential` attached.
    return Variant(
        "ok", json.dumps({"status": 200, "json": {"allowed": True, "via": url}})
    )


def get(_store: Any, key: str) -> str | None:
    return STORE.get(key)


def put(_store: Any, key: str, value: str) -> None:
    STORE[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component")
    parser.add_argument("step")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--grant",
        action="store_true",
        help="contrast: hand the guest an env var and a directory",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    config = Config()
    engine = Engine(config)
    component = Component.from_file(engine, args.component)
    t_compile = time.perf_counter() - t0

    linker = Linker(engine)
    linker.add_wasip2()
    with linker.root() as root:
        with root.add_instance(INTERFACE) as host:
            host.add_func("generate", generate)
            host.add_func("fetch", fetch)
            host.add_func("get", get)
            host.add_func("put", put)

    store = Store(engine)
    wasi = WasiConfig()  # nothing inherited: no env, no preopens, no stdio
    if args.grant:
        import tempfile

        wasi.env = [("OPENAI_API_KEY", "sk-granted-by-host")]
        wasi.preopen_dir(tempfile.mkdtemp(), "/tmp")
    store.set_wasi(wasi)
    t1 = time.perf_counter()
    instance = linker.instantiate(store, component)
    run = instance.get_func(store, "run-monitor")
    assert run is not None
    t_inst = time.perf_counter() - t1

    step = Path(args.step).read_text()
    if args.probe:
        step = json.dumps({"probe": True})
    timings = []
    for _ in range(args.repeat):
        t = time.perf_counter()
        result = run(store, step)
        run.post_return(store) if hasattr(run, "post_return") else None
        timings.append(time.perf_counter() - t)
    assert isinstance(result, Variant), result
    report = json.loads(result.payload) if result.tag == "ok" else result.payload
    print(json.dumps({"tag": result.tag, "report": report}, indent=2))
    print(
        json.dumps(
            {
                "compile_s": round(t_compile, 4),
                "instantiate_s": round(t_inst, 4),
                "first_call_s": round(timings[0], 4),
                "warm_call_median_s": round(
                    sorted(timings[1:])[len(timings[1:]) // 2], 6
                )
                if len(timings) > 1
                else None,
                "host_calls": CALLS[:5],
                "max_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1
                ),
            }
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
