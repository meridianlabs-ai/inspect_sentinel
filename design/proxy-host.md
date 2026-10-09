# Host in a proxy: spike plan

Status: plan, 2026-10-04; section 4's WASM phase updated from the WASM sidecar spike (sentinel #58, `spikes/wasm_sidecar/`) on 2026-10-06. The proxy host itself is not built. Workstream 4 in [workstreams.md](workstreams.md); the design it builds on is [sentinel-deployment.md](sentinel-deployment.md), which this document does not repeat.

The goal is a sentinel host that runs inside, or beside, an HTTP proxy such as Envoy, so the same monitors and protocols that run in an eval can watch an agent's model traffic in deployment. Each section below names what has to exist, and the last lists the spikes in order.

## 1. From the wire to steps

A proxy sees one HTTP exchange at a time, a request and a response, and never a tool running.

| Stage            | In a proxy                                  | Fidelity |
|------------------|---------------------------------------------|----------|
| `BeforeGenerate` | the request body                            | full     |
| `AfterGenerate`  | the response body                           | full     |
| `BeforeToolCall` | `tool_use` blocks in the response           | good     |
| `AfterToolCall`  | `tool_result` blocks in the next request    | lossy: the model already has the result, and the last one is never seen |

Needed:

- **Provider normalization.** OpenAI Chat and Responses, Anthropic and Gemini request and response bodies become `ChatMessage`, `ToolCall` and `ModelOutput`. inspect_ai's converters (`messages_from_openai` and the others) already do this; they move into `inspect_core` (workstream 1, [inspect-core.md](inspect-core.md)) so a proxy does not ship inspect_ai.
- **Step identity across requests.** A conversation key and step ids that let the `tool_use` in one response be matched with its `tool_result` in the next request.
- **The generate stages** (sentinel #42, `design/generate-stages.md` on that branch) as first-class: they are the proxy's native stages.

## 2. From decisions to the wire

| Decision | Before a generate | After a generate, or before a tool call |
|---|---|---|
| `continue` | pass the request | pass the response |
| `modify` | rewrite the request body | rewrite the response, e.g. a `tool_use` block |
| `reject` | pending the generate stages (sentinel #42); the design makes it "do not run this generate, tell the agent why" (`sentinel.md`, "One vocabulary across stages") | rewrite the response so the agent does not run the call: the `tool_use` replaced by the decision's `message` as text, or a refusal. There is no tool-result channel here, so the exact shape needs design. |
| `terminate` | an error response, plus an optional out-of-band signal to whatever owns the session (who owns it is open; deployment doc, open question 3) | same |
| `escalate` at the top, and `human()` | an audit queue that never blocks: the step proceeds with a default and the case is queued for review | same |

Buffering first, since it gives a complete `ModelOutput`; streaming, which means judging partial output after some tokens reached the agent, later (deployment doc, open question 4).

## 3. A proxy implementation of the host interface

The host interface after sentinel #47 is `HostContext(context, recorder, store)` passed to `run_sentinel`, with `Host.generate` and `Host.ask_human` on `context.host`. A proxy host provides:

- **`Context`**: `path` and `host` as in an eval, and `eval=None`: a proxy request is not an Inspect eval and has no task, sample or epoch, so the eval's fields (`context.eval.task`, `sample_id`, `sample_input`, `metadata`, `task_description`) are absent rather than filled with stand-ins. Decided by the maintainer on 2026-10-05. Whether a proxy needs its own equivalents (a deployment id, a conversation key, a charter from deployment config) is for this spike to find out; the conversation key backs the store, below.
- **`Host.generate`**: routes to models with a recursion guard, so a monitor's own model call does not pass back through the proxy it runs in (deployment doc, open question 5). `role` is a model role and `model` a model name; with neither, the role is `monitor` (deployment doc, "The host ABI"). The proxy maps each role to an endpoint and credentials from deployment config, and applies a policy to model names: allow them, map them to configured endpoints, or refuse them. Two decisions to make: whether an unconfigured role is an error in a proxy (there is no agent model to fall back to), and whether arbitrary model names are allowed.
- **`Host.ask_human`**: the audit queue; ties into workstream 13's message queue.
- **`Recorder`**: `SentinelEvent`s to a sink (structured logs, OpenTelemetry, or files a viewer or workstream 12 can read).
- **The store**: a keyed store, keyed by a session header verified against a fingerprint of the conversation's start (deployment doc, "Making the key trustworthy"), behind `store_as()`.
- **`step.history`** across compaction, from the keyed store's accumulation for the conversation.
- **Files**: read-only access to files bundled with the monitor (a read-only preopened bundle data directory, or `importlib.resources`) and nothing else by default; arbitrary host paths are never granted. State goes in the store, since a write is per instance and lost. Decided by the maintainer on 2026-10-06 (deployment doc, "What the guest can reach").
- **New host methods**: `fetch` through named endpoints (designed in the deployment doc, not built), and possibly an out-of-band `terminate()`.

## 4. Where the Python runs, phased

1. **An Envoy external processor (`ext_proc`) sidecar, in ordinary CPython.** Lowest risk. It proves sections 1 to 3 and the real latency and size limits (gRPC's 4 MB default) with no WASM unknowns. `ext_proc` is the Envoy route that sees both directions; `ext_authz` sees only requests. Build it transport-neutral, so phase 3 is the same host behind another transport.
2. **WASM inside the sidecar.** The sidecar embeds wasmtime and runs the sentinel as a CPython-in-WASM component built with componentize-py. Sentinel's `Host` is the guest ABI, written in WIT with JSON payloads (`generate`, `fetch` by endpoint name, `sleep`, `get`/`put`; later `ask_human` and `terminate`); the sidecar implements the imports as async host functions; instance pools are sized to workers, not to requests. proxy-wasm inside Envoy is a fallback only if a deployment cannot run a sidecar: there is no Python proxy-wasm SDK, and its callback ABI has no component-model async.
3. **An RPC interface for other proxies**, such as LiteLLM proxy hooks or a generic JSON-RPC sidecar.

Run the `ext_proc` sidecar as the vehicle for sections 1 to 3; WASM adds isolation, credentials kept by the host, tenants and hard resource limits to the same sidecar.

**What spike #58 proved (2026-10-06).** A Rust host on wasmtime ran the unchanged `inspect_sentinel` runner, real `inspect_ai.core` types and pydantic in a component: two LLM monitors under `threshold()`, `no_network()` and `no_destruction()` in `concurrent()`. Details and numbers are in the deployment doc, "WASM", "pydantic-core for WASI" and "What the guest can reach"; in brief:

- asyncio works in the guest on componentize-py's loop plus a 90-line patch, without Asyncify, and one instance served 50 concurrent steps with 200 overlapping host calls. The blocking world (the host suspends the guest on a fiber) works as well.
- `run_sentinel`'s per-step context variable stays isolated across concurrent steps in one instance (section 6).
- Warm step 0.38 ms (about 1.3× native), instantiation 0.38 ms, about 22 MiB of linear memory per instance, component 7.8 MiB with zstd.
- The empty WASI context leaves the guest no files, environment, sockets, subprocesses or threads; epoch deadlines, fuel and memory caps hold.
- A cancelled host call keeps running on the host, so budgets must count it.

**Next for this phase:**

- A production host: the `ext_proc` sidecar of phase 1 with the component embedded, instance pools, trap recovery, and per-step or per-conversation instance policy (deployment doc, open question 11).
- A CI job that builds pydantic-core for WASI for each supported pydantic release and the CPython componentize-py embeds.
- componentize-py, wasmtime and wasi-sdk pinned together; the component-model async ABI still changes between releases.
- Upstreaming the loop fixes to componentize-py (`call_soon(context=None)`, timers, `get_task_factory`, and subtask cancellation), carrying `loop_patch.py` and testing it under the conformance suite until then.
- The guest-import changes in `inspect_ai` and `inspect_sentinel` that remove the spike's shims, part of workstream 1 ([inspect-core.md](inspect-core.md), "What a WASM guest imports").
- The read-only bundle data directory: preopen it in the production host and give the guest nothing else (section 3, "Files").
- Imports inside function bodies: componentize-py bundles only what build-time initialisation imports, so a module first imported in a function body raises `ModuleNotFoundError` at run time. The bundler also imports what portable functions' bodies import, or bundles whole packages; the portability check does not (deployment doc, "Imports inside function bodies").
- Measure what `time.sleep` does in a guest under the async model, in particular whether it blocks the whole instance; the portability check may make it an error afterwards.

## 5. Packaging and configuration

- **Bundling**: the monitor modules and a manifest (deployment doc, "What gets bundled"), and configuration through `sentinel_from_config`, with the `version` and `meta` each entry records. The host reads and parses its configuration file (YAML or JSON) and passes the `sentinel` value; sentinel does no file access.
- **The `portable=True` check** (workstream 11) runs when a factory is called, per function; it is early feedback, and the sandbox is the enforcement. Bundling adds a separate per-module verdict, the import closure of each bundled module: a module that fails is dropped and named.
- **Dependencies**: `inspect_core`, pydantic and anyio; not inspect_ai.

## 6. Changes to the sentinel interface

- The store as a protocol rather than the concrete `Store` that `HostContext.store` is today (deployment doc, open question 10).
- What `Context`'s fields mean in a proxy, and which may be `None`.
- `fetch`, and whether there is an out-of-band `terminate`.
- `AfterToolCall` marked lossy in a proxy, perhaps as a fidelity field on the step.
- `run_sentinel`'s per-step context variable under the WASM event loop: verified by spike #58, which ran 20 concurrent steps in one instance; it is also tested under CPython asyncio and trio.

## 7. Operations

- Failing open or closed when the processor is down: a safety monitor wants closed, which makes it a hard dependency of the request path (deployment doc, open question 8).
- Latency budgets and per-monitor timeouts.
- Body size limits, and incremental judging: feed monitors the newest turn rather than the whole history, since re-judging everything costs O(n²) inference (deployment doc, "The store is not merely a convenience").
- Tenant isolation and trust in session keys.
- Observability and replay.

## 8. Testing

- **A conformance suite**: the same monitors and steps give the same decisions and events in inspect_ai's host and in the proxy host.
- **A replay harness**: recorded provider traffic, captured and synthetic, played through the proxy in tests.

## 9. Spikes, in order

1. Provider bodies to steps for one provider (Anthropic or OpenAI), on converters moved into `inspect_core`.
2. An `ext_proc` sidecar running `run_sentinel` with an `observe_only()` monitor, then a rule that rejects; measure latency.
3. The asyncio-on-WASI prototype. Done: spike #58, 2026-10-06 (section 4).
4. The keyed store and the audit queue.
5. Wire actions for `modify` and `reject` on tool calls.
6. The WASM sidecar as a production host: the phase 1 sidecar embedding wasmtime and the sentinel component, with a CI build of pydantic-core for WASI and pinned toolchain versions (section 4, "Next for this phase").
