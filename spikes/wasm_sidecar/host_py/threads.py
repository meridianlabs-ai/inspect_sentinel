"""Thread per instance with wasmtime-py.

wasmtime-py has no async host functions or component-model async. Can a
Python sidecar still overlap guests? One Store per thread, a blocking
`generate` (time.sleep standing in for HTTP). Usage:

    python host_py/threads.py build/m1_sync.wasm step.json [N]
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from wasmtime import Engine, Store, WasiConfig
from wasmtime.component import Component, Linker, Variant

T0 = time.perf_counter()


def stamp(msg: str) -> None:
    print(
        f"{(time.perf_counter() - T0) * 1000:8.1f}ms {threading.current_thread().name} {msg}"
    )


def generate(_store: Any, request: str) -> Variant:
    stamp("generate start")
    time.sleep(0.5)
    stamp("generate end")
    out = {
        "model": "mock",
        "choices": [
            {
                "message": {"role": "assistant", "content": "SCORE: 2"},
                "stop_reason": "stop",
            }
        ],
    }
    return Variant("ok", json.dumps(out))


def main() -> None:
    path, step_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    engine = Engine()
    component = Component.from_file(engine, path)
    linker = Linker(engine)
    linker.add_wasip2()
    with linker.root() as root:
        with root.add_instance("sentinel:spike/host@0.1.0") as host:
            host.add_func("generate", generate)
            host.add_func("fetch", lambda s, e, r: Variant("err", "no"))
            host.add_func("get", lambda s, k: None)
            host.add_func("put", lambda s, k, v: None)
    step = Path(step_path).read_text()

    def worker() -> None:
        store = Store(engine)
        store.set_wasi(WasiConfig())
        instance = linker.instantiate(store, component)
        run = instance.get_func(store, "run-monitor")
        assert run is not None
        stamp(f"report {run(store, step).payload[:40]}")

    global T0
    T0 = time.perf_counter()
    threads = [threading.Thread(target=worker, name=f"instance-{i}") for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(
        f"{n} instances on {n} threads, 500ms generate each: wall {(time.perf_counter() - T0) * 1000:.0f}ms"
    )


if __name__ == "__main__":
    main()
