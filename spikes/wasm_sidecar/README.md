# Spike: a Python sentinel monitor as a WASM guest in a sidecar

Status: spike, 2026-10-06. Not for merge. Answers the WASM question in `design/proxy-host.md` (section 4, phase 2) and open question 2 of `design/sentinel-deployment.md`.

## Summary

**It works, including the async part.** A host process loads a WebAssembly component that contains CPython 3.14, pydantic, the real `inspect_ai.core` types and the unchanged `inspect_sentinel` runner. It feeds the component a step as JSON. The monitors call back into the host for inference (`generate`, mocked), and the component returns a decision with its records.

- **Toolchain:** componentize-py 0.25.1 builds the component, and a Rust host on wasmtime 46.0.1 runs it. A wasmtime-py host covers the blocking case.
- **Two concurrency models work.**
  - *Blocking world:* `generate` is a plain WIT import that the host implements as an async function. Wasmtime suspends the guest on a fiber while the host awaits, so monitor code needs no asyncio, and instances overlap in one host thread.
  - *Async world:* `generate` is a component-model `async func` (WASIp3). The guest runs a real asyncio loop that componentize-py supplies. `await host.generate()` suspends only that task, so one instance runs many steps at once. Inside a step, `concurrent()`, `asyncio.gather`, `TaskGroup` and anyio task groups overlap their host calls. Measured: 50 concurrent steps × 4 monitors = 200 overlapping 500 ms generates in one instance took 512–523 ms.
- **The design doc's "load-bearing unknown" has an answer.** The guest gets asyncio, but not CPython's stock loop over `wasi:io/poll`. It uses componentize-py's component-model loop instead, and Asyncify is not needed. That loop has gaps (no timers, broken cancellation, some methods anyio needs). A 90-line guest patch, `guest/loop_patch.py`, plus one `sleep` host import fixes them. With the patch, every probe passes: gather, TaskGroup, contextvars, `asyncio.sleep`, `wait_for` (including when it fires), cancelling an in-flight host call, anyio task groups, `move_on_after` and `fail_after`.
- **pydantic-core works, but we have to build it.** No published WASI wheel matches a componentize-py version that has async support. The ones that exist are cp312/cp313; componentize-py embeds CPython 3.12 up to 0.17 and 3.14 from 0.18, and async arrived in 0.19. I cross-compiled pydantic-core 2.46.5 for cp314 `wasm32-wasip1` in about 30 s, with no CPython WASI build (see `build.sh`).
- **Real types: `inspect_ai.core` imports cleanly once `inspect_ai/__init__.py` is kept out.** `inspect_sentinel` imports eight modules beyond core. With small stand-ins for them (`guest/inspect_shim`, about 250 lines), the **unchanged** `run_sentinel(concurrent([threshold({...two LLM monitors...}), no_network(), no_destruction()]))` runs in the guest:
  - `no_network` rejects `curl`.
  - `threshold` rejects at a suspicion of 0.95.
  - `no_destruction`'s `terminate` cancels both LLM monitors while their host calls are in flight. The decision comes back in 3 ms.
- **Sandbox:** with an empty WASI context the guest has no files, no environment variables, no sockets (`PermissionError`), no subprocesses and no threads. Only host-mediated calls work, and `fetch` takes endpoint names, not URLs. The host holds the credentials.
- **Numbers** for the M4 component (real runner, pydantic, two LLM monitors), from a precompiled `.cwasm`:

  | Measure | Value |
  |---|---|
  | Component size | 29.8 MiB raw, 7.8 MiB zstd |
  | Load precompiled | 3 ms |
  | Instantiate | 0.4 ms |
  | First step | 2.2 ms |
  | Warm step | 0.38 ms (native CPython: 0.29 ms) |
  | Linear memory per instance | 22.4 MiB |
  | Host RSS per instance after first call | about 13.6 MiB |

  componentize-py pre-initializes the component: module-level code runs at build time. Natively, those imports take about 100 ms.

## Milestones

| | Result | Evidence |
|---|---|---|
| M1 pure-Python monitor, mocked `generate` | ✅ wasmtime-py host and Rust host | [M1](#m1-a-pure-python-monitor) |
| M2 async host call, overlapping guests | ✅ fibers (blocking world) and component-model async (async world); in-guest fan-out | [M2](#m2-async-host-calls-and-concurrency) |
| M3 pydantic in the guest | ✅ with a self-built cp314 WASI pydantic-core | [M3](#m3-pydantic) |
| M4 real types and a real `@monitor` | ✅ `inspect_ai.core` vendored; unchanged `inspect_sentinel` behind about 250 lines of stand-ins for the inspect_ai modules it imports | [M4](#m4-inspect_aicore-and-inspect_sentinel) |
| M5 sandbox | ✅ | [M5](#m5-sandbox) |
| M6 measurements | ✅ | [M6](#m6-measurements) |
| Extra: Rust host, async host functions, fan-out | ✅ | M2 |
| Extra: resource limits | ✅ epoch deadline, fuel, memory cap | [Limits](#resource-limits) |

## Layout and how to run

```
wit/sentinel.wit        the guest ABI: interfaces host (blocking) and host-async; worlds monitor-sync, monitor-async
guest/                  guest code (componentize-py app modules app_*.py)
  m1/                   rule-plus-LLM monitor in plain Python; guest Host classes over the imports
  m3/                   pydantic step and report models
  m4/                   real @monitor/@protocol factories; thin runner (Host over WIT, Recorder, run_sentinel)
  inspect_shim/         stand-ins for the inspect_ai modules inspect_sentinel imports beyond core
  loop_patch.py         fixes to componentize-py's asyncio loop
  wasi_shims.py         `ssl` stub (anyio imports it; the WASI build has no _ssl)
  probes.py, async_probes.py
host_py/                wasmtime-py hosts: host.py (M1, M5), threads.py (thread per instance)
host_rs/                Rust host (wasmtime 46.0.1): sync, async, bench and limits commands
build.sh                pinned, checksummed, self-contained build into build/ (gitignored)
demo.sh                 runs every milestone (fmt.py formats its output)
native_bench.py         the M4 step on native CPython, for comparison
```

`./build.sh && ./demo.sh`. The build needs `uv`, `curl`, `git` and network access. It installs Rust, wasi-sdk and the Python tools under `build/tools`. A clean build took 2 min 26 s on an M-series Mac, most of it compiling wasmtime and pydantic-core.

Pins: componentize-py 0.25.1 (CPython 3.14 from `dicej/cpython` v3.14.0-wasi-sdk-30), the wasmtime crate 46.0.1 (what componentize-py 0.25.1 tests against), wasmtime-py 49.0.0, Rust 1.99.0, wasi-sdk 30, pydantic 2.13.5 and pydantic-core 2.46.5, anyio 4.11.0, PyYAML 6.0.3 (pure Python), and inspect_ai `8379140fd` (origin/main, 2026-10-06). `build.sh` checks the sha256 of each tarball it downloads.

## The guest ABI

`wit/sentinel.wit` is a JSON-in, JSON-out slice of sentinel's `Host`:

```wit
interface host-async {
  generate: async func(request: string) -> result<string, string>;   // request JSON -> ModelOutput JSON
  fetch: async func(endpoint: string, request: string) -> result<string, string>;  // a named endpoint, never a URL
  sleep: async func(millis: u64);                                     // a timer for the guest's loop
  get: func(key: string) -> option<string>;
  put: func(key: string, value: string);
}
world monitor-async {
  import host-async;
  export run-monitor: async func(step: string) -> result<string, string>;  // step JSON -> report JSON
}
```

The blocking interface, `host`, has the same functions without `async` and without `sleep`.

Payloads are JSON strings. The guest validates the raw bytes once with pydantic: `TypeAdapter(BeforeToolCall).validate_json` for the step and `ModelOutput.model_validate_json` for the model output. That follows `sentinel-deployment.md`'s "one parse beats two" argument; I did not try marshalling `ChatMessage` as WIT variants.

## M1: a pure-Python monitor

`guest/m1/monitor.py` is an `async def` rule-plus-LLM monitor. A regex rule runs first. Otherwise it builds a prompt, calls `host.generate`, parses `SCORE: n` and returns `{"suspicion", "explanation"}`.

In the blocking world, `run_blocking` drives the coroutine with `coro.send(None)`. That works because every `await` completes synchronously, so the same monitor source runs in both worlds.

```
$ python host_py/host.py build/m1_sync.wasm step.json --repeat 20
{"compile_s": 0.7072, "instantiate_s": 0.0045, "first_call_s": 0.0018, "warm_call_median_s": 5.9e-05, ...}
{"tag": "ok", "report": {"suspicion": 0.2, "explanation": "Reading disk usage under the user's home is in scope. SCORE: 2"}}
```

## M2: async host calls and concurrency

**Blocking world plus async host function (fibers).** The guest sees `generate()` as an ordinary blocking call. The Rust host implements it as `async fn` with a `tokio::time::sleep` standing in for HTTP. Wasmtime runs each instance on a fiber and suspends it at the call. Four instances, each in its own `Store`, on a single-threaded tokio runtime:

```
$ sentinel-wasm-host sync build/m1_sync.wasm step.json --instances 4 --delay-ms 500
     2.3ms instance 0 generate start
     3.6ms instance 1 generate start
     4.9ms instance 2 generate start
     6.1ms instance 3 generate start
   504.2ms instance 0 generate end   ...   510.2ms instance 3 generate end
sync world: 4 instances, delay 500ms each: wall 512ms; peak linear memory per instance 16.1 MiB
```

In this model one instance handles one step at a time, and calls inside a step run in sequence. Concurrency comes from a pool of instances, which costs 0.4 ms and about 6 to 13 MiB of RSS each (M6).

**Component-model async (WASIp3).** With `async func` imports and exports, componentize-py generates `async def` bindings. It installs its own `asyncio` event loop, `componentize_py_async_support._Loop`, driven by the component model's waitable sets. Awaiting an import parks the task, and the host's `run_concurrent` can start more export calls into the same instance.

```
$ sentinel-wasm-host async build/m1_async.wasm step.json --calls 3 --delay-ms 500
     0.5ms instance 0 generate #0 start
     0.6ms instance 0 generate #1 start
     0.6ms instance 0 generate #2 start
   502.2ms .. 503.8ms   generate #0..#2 end
async world: 1 instance, 3 concurrent run-monitor calls x fanout 1 = 3 generates, delay 500ms each: wall 504ms
$ sentinel-wasm-host async build/m1_async.wasm step.json --calls 50 --fanout 4 --delay-ms 500
async world: 1 instance, 50 concurrent run-monitor calls x fanout 4 = 200 generates, delay 500ms each: wall 512ms
```

`--fanout` makes each call run the monitor four times under `asyncio.gather`, the shape of `concurrent()`. In M4 the real `concurrent()` and `threshold()` from inspect_sentinel do the same through anyio: two LLM monitors' generates start at the same moment and the step takes one delay.

**What works in the guest loop.** `guest/async_probes.py` probes the loop. It ran once against the stock componentize-py 0.25.1 loop (except the two timeout-firing probes, which I added later) and once with `loop_patch.py`. Each probe's `await host.generate` takes 50 ms.

| Probe | Stock loop | With `loop_patch.py` |
|---|---|---|
| `await host.generate` | ✅ | ✅ |
| `asyncio.gather` ×3, `asyncio.TaskGroup` ×3 | ✅ overlapped (53 ms total) | ✅ |
| contextvars across tasks | ✅ | ✅ |
| `asyncio.sleep(0)` | ✅ | ✅ |
| `asyncio.sleep(0.01)` | ❌ `NotImplementedError` (`call_later`) | ✅ 12 ms |
| `asyncio.wait_for(timeout=5)` | ❌ `NotImplementedError` | ✅ |
| `wait_for` that fires (10 ms over a 50 ms call) | n/a (no timers) | ✅ `TimeoutError` at 12 ms |
| `loop.time()` | ❌ `NotImplementedError` | ✅ |
| cancel a task awaiting a host call | ❌ **instance traps**: `resource has children` from `waitable_set_drop` | ✅ cancelled |
| anyio task group ×3 | ❌ first `ModuleNotFoundError: anyio._backends`; with that bundled, `AssertionError` (`call_soon(context=None)`), then `NotImplementedError` (`get_task_factory`) | ✅ overlapped |
| `anyio.move_on_after`, `fail_after` (including firing) | ❌ `NotImplementedError` (`current_time`) | ✅ |

What `loop_patch.py` does:

1. `call_soon(context=None)` uses the caller's context, which holds the loop's per-task state.
2. `time()` reads the monotonic clock.
3. `call_at` and `call_later` run on the host `sleep` import.
4. `get_task_factory()` returns `None`.
5. `_timer_handle_cancelled` is a no-op.
6. `await_result` detaches on cancellation.

The runtime has no `subtask.cancel`. So when a task is cancelled while it awaits a host call, the call keeps running on the host, the guest discards its result, and the component-model task stays open until the call returns. A timer task also counts as pending work, because otherwise the loop drops its waitable set while a host call is still joined, which traps.

Two other guest fixes: anyio loads its asyncio backend lazily, so the app must `import anyio._backends._asyncio` at top level to get it bundled. anyio also imports `ssl` at import time and the WASI build has no `_ssl`, so `wasi_shims.py` installs a stub `ssl` module.

**The `run_sentinel` context variable** stays isolated per step. componentize-py starts each export call in a fresh `contextvars.Context`. Twenty concurrent M4 steps in one instance (216 ms with 200 ms generates) each recorded their own layers.

**wasmtime-py** (49.0.0) supports components but has no async host functions and no component-model async. A Python host can still overlap guests with one thread per instance. Four instances on four threads with a blocking 500 ms `generate` took 512 ms (`host_py/threads.py`). The host function's `time.sleep` releases the GIL; ctypes also releases it around calls into wasmtime, so guest execution should run in parallel too, but I did not measure that. A Python sidecar would give up in-guest concurrency and run a thread pool of instances.

## M3: pydantic

`app_m3_async.py` validates the step with `BeforeToolCall.model_validate_json` (a pydantic model, `guest/m3/models.py`) and the report with `Report`. Valid step:

```
REPORT 0 {"suspicion":0.2,"explanation":"Reading disk usage is in scope (call 0). SCORE: 2"}
```

Invalid step:

```
REPORT 0 {"error": "invalid step", "detail": [{"type": "dict_type", "loc": ["call", "arguments"], "msg": "Input should be an object", ...},
          {"type": "literal_error", "loc": ["input", 0, "role"], "msg": "Input should be 'system', 'user', 'assistant' or 'tool'", ...}]}
```

**The pydantic-core story:**

- PyPI publishes pydantic-core only as an Emscripten wasm wheel, and nothing for WASI.
- benbrandt/wasi-wheels publishes cp312 and cp313 WASI wheels. The latest is 2.41.5, from 2025-11. dicej/wasi-wheels' `latest` release from 2024-03 includes a `pydantic_core-wasi.tar.gz`, which I did not check.
- componentize-py embedded CPython 3.12 up to 0.17 and has embedded 3.14 since 0.18 (commit `e8b1ab0`). It never shipped 3.13. Async/WASIp3 support arrived in 0.19 (commit `dbb80b6`).
- So **no published wheel works with a componentize-py that has async.** The only published combination is componentize-py 0.17 with a cp312 wheel, which means the blocking world and pydantic-core 2.41.5 or older.
- **Building it ourselves is easy.** `build.sh` cross-compiles pydantic-core 2.46.5 with `cargo rustc --crate-type cdylib --target wasm32-wasip1` against the wasi-sdk 30 sysroot, as a PIC shared library. PyO3 gets the target from a config file (`version=3.14`, `suppress_build_script_link_lines=true`), so no CPython WASI build is needed. componentize-py links the resulting 3.3 MB `_pydantic_core.cpython-314-wasm32-wasi.so` dynamically. One flag goes beyond benbrandt's recipe: `--unresolved-symbols=import-dynamic`. Without it, wasm-ld fails on every `Py*` symbol.
- The wasi-sdk version has to match the one componentize-py's CPython was built with (30).

## M4: inspect_ai.core and inspect_sentinel

**`inspect_ai.core` itself is clean.** Its imports are the stdlib plus pydantic, pydantic_core, typing_extensions and shortuuid, as its own `DEFAULT_ALLOWED` says. The problem is the package around it: `import inspect_ai.core` first runs `inspect_ai/__init__.py`, which imports the eval machinery and so all of inspect_ai. The guest bundle replaces that `__init__` with an empty file and vendors `src/inspect_ai/core` from origin/main. Separate distribution of inspect_core, or a lazy `inspect_ai/__init__`, removes the need.

**What `inspect_sentinel` imports beyond core.** Every package import runs `inspect_sentinel/__init__.py`, which imports every module, so all of these are needed even for a single monitor.

| Sentinel module | Import | In core? | In the guest |
|---|---|---|---|
| `_step`, `_context`, `_host` | `inspect_ai.model.ChatMessage, ChatMessageTool, GenerateConfig, ModelOutput`; `inspect_ai.tool.ToolCall, ToolCallView, ToolResult, ToolInfo`; `inspect_ai.scorer.Target, Reference` | yes | re-exported from core (`inspect_shim/inspect_ai/{model,tool,scorer}`) |
| `_host` | `inspect_ai.model.Model` | **no**; only in `Host.generate`'s signature | empty class |
| `_report` | `inspect_ai.event.SentinelAction, SentinelSuspicion` | **no**; `event/_sentinel.py` imports `event._base` and so the event tree | two type aliases copied |
| `_context`, `_host` | `inspect_ai.util.Store, StoreModel` | **no**; `_store.py` imports jsonpatch and the event machinery | a dict `Store`; `StoreModel` copied verbatim |
| `_runner` | `inspect_ai.util.LimitExceededError` | **no**; `_limit.py` imports the logger, transcript and samples | an `Exception` subclass |
| `_decorators`, `_runner`, `_resolve`, `_validate`, `_config` | `inspect_ai._util.registry` (11 names) | **no**; imports entry points and package metadata, and lazily most of inspect_ai | a small attribute-plus-dict registry (about 100 lines) |
| `_config` | `inspect_ai.log.SentinelConfig, SentinelEntry` | **no** | minimal models (config loading not exercised) |
| `_config` | `inspect_ai.util.resource`, `inspect_ai._util.file.exists, local_path` | **no**; fsspec and s3fs | stubs |
| `_config` | `yaml` | third-party | PyYAML's pure-Python package works; the C accelerator is not built |
| `_runner` | `anyio` | third-party | works with `loop_patch.py` |

**The result.** With those stand-ins, the unchanged `inspect_sentinel` source runs in the guest. It is copied from this checkout at build time, with no edits. The monitors are `examples/llm_suspicion.py`'s `suspicion` and `examples/no_network.py`'s `no_network`, with their `@task` parts removed and `inspect_ai.util.json_schema` replaced by `JSONSchema.model_validate(Verdict.model_json_schema())`. The root is:

```python
ROOT = concurrent(
    [
        threshold(
            {"suspicion": suspicion(), "trusted": suspicion(role="trusted")},
            reject_at=0.8,
        ),
        no_network(),
        no_destruction(),  # terminates on `rm -rf`
    ]
)
```

The output has one report per step, followed by its records:

```
-- benign step (two 200 ms generates, overlapped): wall 205 ms
decision: {'action': 'continue', ...}
   reported no_network       no_network                  continue
   reported no_destruction   no_destruction              continue
   reported threshold/suspicion  suspicion               0.2
   reported threshold/trusted    suspicion               0.2
   reported threshold        inspect_sentinel/threshold  continue
   reported <root>           inspect_sentinel/concurrent continue
-- curl step
decision: {'action': 'reject', 'explanation': 'command uses curl (threshold: continue; no_network: reject; no_destruction: continue)',
           'message': '`curl` needs the network, which this task does not allow.'}
-- rm -rf step: both generates in flight when no_destruction terminates: wall 3 ms
decision: {'action': 'terminate', 'explanation': 'rm -rf (no_network: continue; no_destruction: terminate)'}
   reported no_destruction no_destruction terminate
   cancelled threshold/suspicion, threshold/trusted, threshold
   reported <root> inspect_sentinel/concurrent terminate
```

**Yes, a thin guest-side runner can call a real `@monitor` factory.** `guest/m4/runner.py` is about 80 lines. It has three parts:

- `WitHost`, which implements `Host.generate` by serializing `input`, `model`, `role`, `tools` and `config` to JSON, awaiting the WIT import and parsing the result with `ModelOutput.model_validate_json`.
- `ListRecorder`, which collects records for the host.
- `run_step`, which reads the step JSON into the real `BeforeToolCall` dataclass with `TypeAdapter`, builds `HostContext(Context(path="", host=..., eval=None), recorder, Store())` and calls `run_sentinel`.

`ask_human` raises; an audit queue would be another import.

What would let the guest drop the stand-ins:

1. `inspect_core` as its own distribution, with `SentinelAction` and `SentinelSuspicion` moved in.
2. A sentinel-owned registry, or one in core with no entry-point loading.
3. `Store` as a protocol (already an open question in `proxy-host.md`).
4. `LimitExceededError` and `Model` out of the runner's and `Host`'s import path, for example `TYPE_CHECKING` or a core base class.
5. `_config`, with its yaml, fsspec and `resource` imports, out of `inspect_sentinel/__init__`'s import closure, or importing them lazily.

## M5: sandbox

These run in the blocking world through both hosts, each with an empty WASI context (`WasiCtxBuilder::new()` or `WasiConfig()`: nothing inherited). The modules are imported at build time so that they are in the bundle (see the note below):

| Probe | Result |
|---|---|
| `open('/etc/passwd')` | `FileNotFoundError: [Errno 44] No such file or directory` (no preopened directories) |
| `open('/tmp/x', 'w')` | `FileNotFoundError` |
| `os.listdir('/')` | `FileNotFoundError` |
| `os.environ` | `{}` |
| `socket.connect(...)` | `PermissionError: [Errno 2] Permission denied` (WASI sockets present, network not granted) |
| `urllib.request.urlopen` | `LookupError: unknown encoding: idna`, which fails before reaching a socket |
| `subprocess.run` | `OSError: [Errno 58] wasi does not support processes.` |
| `threading.Thread().start()` | `RuntimeError: can't start new thread` |
| `os.system` | `AttributeError` (not in WASI `os`) |
| `ssl` | `ModuleNotFoundError: No module named '_ssl'` |
| `host.generate(...)` | ✅ |
| `host.fetch("allowlist", ...)` | ✅ (the host resolves the name and would attach the credential) |
| `host.fetch("http://evil.example", ...)` | `Err: unknown endpoint` |

As a contrast, `host.py --grant` gives the guest `OPENAI_API_KEY` and a preopened `/tmp`, and the guest then reads the key and writes the file. Capabilities are exactly what the host's WASI context grants.

**Credentials** live in host configuration (`ENDPOINTS` in `host_py/host.py`, `known_endpoint` in the Rust host). The guest names an endpoint, and the host resolves the URL and attaches the credential. Prompt-injected text cannot reach an arbitrary host or read a key.

**Note:** componentize-py bundles only the modules imported during build-time initialization. A module first imported at run time fails with `ModuleNotFoundError` even when it is in the stdlib. That is a second, accidental layer, not a security boundary. `--stub-wasi` goes further and replaces every WASI import with a trap. It also bakes in the PRNG seed, so it is not suitable here.

## M6: measurements

Measured on an Apple M-series Mac (arm64, macOS 26.5) with the Rust host and the default (on-demand) allocator, 50 instances and 500 warm calls, `generate` returning immediately. "First call" includes the copy-on-write page faults from the memory image. Run `demo.sh` to reproduce.

| | M1 async (plain Python) | M4 async (pydantic + core + sentinel runner, 2 LLM monitors) |
|---|---|---|
| Component `.wasm` | 20.7 MiB (gzip 7.5, zstd-19 5.5) | 29.8 MiB (gzip 10.6, zstd-19 7.8) |
| Precompiled `.cwasm` | 43.0 MiB (zstd-19 10.9) | 60.5 MiB (zstd-19 15.5) |
| Compile with Cranelift | 0.75 s | 1.08 s |
| Load `.cwasm` (mmap) | 2.6 ms | 3.1–3.7 ms |
| Host RSS after load | 28 MiB | 34 MiB |
| Instantiate (from `InstancePre`) | 0.33 ms | 0.38 ms |
| First call | 0.61 ms | 2.24 ms |
| Warm call (median, p90) | 0.026 ms, 0.032 | 0.377 ms, 0.401 |
| Same step natively (CPython 3.14.2, `native_bench.py`) | | 0.285 ms (WASM ≈ 1.3×) |
| Linear memory per instance (peak) | 15.6 MiB | 22.4 MiB; unchanged after 500 calls |
| Host RSS per instance after first call | +6.0 MiB | +13.6 MiB |
| Fuel per call | 666 k | |

**Pre-initialization is confirmed.** componentize-py runs the app's top-level code at build time and snapshots the heap, using `component-init-transform`, a Wizer-like transform. The M4 app records `time.time()` at module level, and at run time that value is the build time, about 30 s earlier: `{"initialized_at": 1791292842.53, "now": 1791292871.04, "modules": 374}`. Instantiation is therefore 0.4 ms, with 374 modules already imported. Natively, importing the same closure takes about 100 ms.

The size difference between M1 and M4 is about 9 MiB. pydantic-core's `.so` is 3.3 MiB of that; the rest is pydantic, core, sentinel and their heap snapshot. The component, not a bundle per module, is the unit to ship, as `sentinel-deployment.md` argues.

### Resource limits

From `sentinel-wasm-host limits`, with epoch interruption and fuel enabled and an epoch tick every 10 ms:

```
normal step:        ok {...} | fuel used 666489 | 0.8ms
spin, 500ms epoch:  trap: wasm trap: interrupt | 617ms
spin, 50M fuel:     trap: wasm trap: all fuel consumed by WebAssembly | 2ms
alloc 16MiB, cap 64MiB:  ok {"allocated_mb": 16}
alloc 200MiB, cap 64MiB: ok {"error": "MemoryError"}     <- the ResourceLimiter refuses memory.grow; Python raises MemoryError
```

The epoch trap came 117–205 ms after the 500 ms deadline across runs: ticks from a sleeping thread drift under load. A trap leaves the instance unusable. The host discards it and instantiates another, which costs 0.4 ms.

## Recommendations for `proxy-host.md` phase 2

1. **Phase 2 should be a sidecar that embeds wasmtime and hosts componentize-py components, not proxy-wasm inside Envoy.** There is no Python proxy-wasm SDK, and proxy-wasm's callback ABI has no component-model async. The path that worked here is Envoy, then `ext_proc`, then a Rust sidecar (or Go with wazero; untested), then CPython-in-WASM guests.
2. **Use the async world, and size instances to workers, not to requests.** One instance served 50 concurrent steps (200 generates). Keep a small pool for isolation and trap recovery. Instances are cheap: 0.4 ms and about 13 MiB of touched memory. Use the blocking world plus fibers only if the guest loop's gaps are unacceptable. It needs no loop patches, but each instance runs one step at a time and generates within a step are sequential.
3. **Keep the WIT ABI JSON-shaped** (`generate`, `fetch(endpoint name)`, `sleep`, `get`/`put`, later `ask_human` and `terminate`), and validate in the guest with pydantic. Add the `sleep` import, or a clock-and-timer interface: the guest loop has no timers without it.
4. **Treat the componentize-py loop gaps as work to upstream.** The changes are `call_soon(context=None)`, timers (could be `wasi:clocks` p3 `wait-for` rather than a custom import), `get_task_factory` and, most importantly, **subtask cancellation**. Until then, carry `loop_patch.py` and test it under the conformance suite. A cancelled host call keeps running on the host, so its cost (tokens, rate limits) is still incurred, and `Host` budgets must count it.
5. **Own a WASI pydantic-core build.** Add a CI job that cross-compiles pydantic-core for each supported pydantic release and the CPython version componentize-py embeds, as `build.sh` does in about 30 s. Pin componentize-py, wasmtime and wasi-sdk together. The component-model async ABI still changes between releases: componentize-py 0.25.1 tests against wasmtime 46.
6. **inspect_core and inspect_sentinel changes** (M4): make `inspect_ai.core` importable without the `inspect_ai` package's `__init__`, and give `inspect_sentinel` an import path to `run_sentinel` that needs only core, pydantic and anyio. Concretely: the registry, `SentinelAction`/`SentinelSuspicion`, `Store`/`StoreModel` as a protocol, `LimitExceededError`, `Model`, and `_config` (yaml, fsspec) out of the eager import closure. The static `portable=True` check could use the table above as its first allowlist test.
7. **Module-level code runs at build time** (pre-initialization). Anything read from the environment, seeded or timestamped at import time is frozen into the artifact. Portable monitors must read configuration at run time. This belongs in the portability guidance.

## Open questions

1. **Instance reuse versus isolation.** Module globals persist across steps in a reused instance, and so across conversations or tenants. Options are a fresh instance per step (0.4 ms plus a 2 ms first call), per conversation, or per worker with a reset discipline.
2. **Cancelled host calls.** Without `subtask.cancel` a `terminate` or timeout cannot stop the host's in-flight request. The host could cancel on its side when the guest drops interest, but that needs an ABI signal.
3. **Memory ratchet with large conversations.** Linear memory stayed at 22.4 MiB over 500 small steps. I did not test a large history.
4. **WASIp3 maturity.** It is not final. componentize-py also ships a WASIp2 `poll_loop.py` over `wasi:io/poll`, which I did not try; it would be the fallback if p3 support lags in a host runtime (for example wazero).
5. **Go host.** Does wazero support components and component-model async? This spike used only wasmtime.
6. **Python host.** wasmtime-py has no async host functions and no component-model async. A Python sidecar is limited to a thread per instance and the blocking world.
7. **Throughput under real payloads.** Warm overhead was 0.38 ms per step with a small step. A many-turn `history` was not measured; sentinel-deployment.md's "feed it the delta" point applies.
