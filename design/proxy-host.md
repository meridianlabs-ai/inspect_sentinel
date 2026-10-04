# Host in a proxy: spike plan

Status: plan, 2026-10-04. Nothing here is built. Workstream 4 in [workstreams.md](workstreams.md); the design it builds on is [sentinel-deployment.md](sentinel-deployment.md), which this document does not repeat.

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
| `reject` | not legal before a generate in v1 | rewrite the response so the agent does not run the call: the `tool_use` replaced by the decision's `message` as text, or a refusal. There is no tool-result channel here, so the exact shape needs design. |
| `terminate` | an error response, plus an optional out-of-band signal to whatever owns the session (who owns it is open; deployment doc, open question 3) | same |
| `escalate` at the top, and `human()` | an audit queue that never blocks: the step proceeds with a default and the case is queued for review | same |

Buffering first, since it gives a complete `ModelOutput`; streaming, which means judging partial output after some tokens reached the agent, later (deployment doc, open question 4).

## 3. A proxy implementation of the host interface

The host interface after sentinel #47 is `HostContext(context, recorder, store)` passed to `run_sentinel`, with `Host.generate` and `Host.ask_human` on `context.host`. A proxy host provides:

- **`Context`**: what the proxy knows. `task` is the agent or deployment id, `sample_id` the conversation key, `sample_input` the first user turn, `metadata` from headers, `task_description` from deployment config. Document which fields may be `None` in a proxy.
- **`Host.generate`**: routes to models with a recursion guard, so a monitor's own model call does not pass back through the proxy it runs in (deployment doc, open question 5). Model roles come from deployment config.
- **`Host.ask_human`**: the audit queue; ties into workstream 11's message queue.
- **`Recorder`**: `SentinelEvent`s to a sink (structured logs, OpenTelemetry, or files a viewer or workstream 10 can read).
- **The store**: a keyed store, keyed by a session header verified against a fingerprint of the conversation's start (deployment doc, "Making the key trustworthy"), behind `store_as()`.
- **`step.history`** across compaction, from the keyed store's accumulation for the conversation.
- **New host methods**: `fetch` through named endpoints (designed in the deployment doc, not built), and possibly an out-of-band `terminate()`.

## 4. Where the Python runs, phased

1. **An Envoy external processor (`ext_proc`) sidecar, in ordinary CPython.** Lowest risk. It proves sections 1 to 3 and the real latency and size limits (gRPC's 4 MB default) with no WASM unknowns. `ext_proc` is the Envoy route that sees both directions; `ext_authz` sees only requests. Build it transport-neutral, so phase 3 is the same host behind another transport.
2. **WASM inside Envoy**: CPython on WASI via proxy-wasm. The first thing to prototype is whether CPython-on-WASI has a working asyncio loop over `wasi:io/poll` (deployment doc, open question 2); the runner relies on asyncio and on `run_sentinel`'s per-step context variable. If not, Asyncify. Then linear memory that only grows, the 10 to 20 MB CPython image, and instance pooling.
3. **An RPC interface for other proxies**, such as LiteLLM proxy hooks or a generic JSON-RPC sidecar.

The workstream says WASM first. Run the `ext_proc` sidecar beside it as the vehicle for sections 1 to 3; the WASM spike then only has to answer the asyncio question.

## 5. Packaging and configuration

- **Bundling**: the monitor modules and a manifest (deployment doc, "What gets bundled"), and configuration through `sentinel_from_config` (YAML), with the `version` and `meta` each entry records.
- **The portability linter** (workstream 9) as a deploy-time check: no captured state, only bundleable imports, no `get_model()` or `sandbox()`.
- **Dependencies**: `inspect_core`, pydantic and anyio; not inspect_ai.

## 6. Changes to the sentinel interface

- The store as a protocol rather than the concrete `Store` that `HostContext.store` is today (deployment doc, open question 10).
- What `Context`'s fields mean in a proxy, and which may be `None`.
- `fetch`, and whether there is an out-of-band `terminate`.
- `AfterToolCall` marked lossy in a proxy, perhaps as a fidelity field on the step.
- `run_sentinel`'s per-step context variable verified under the WASM event loop; it is tested under CPython asyncio and trio.

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
3. The asyncio-on-WASI prototype, in parallel with 1 and 2.
4. The keyed store and the audit queue.
5. Wire actions for `modify` and `reject` on tool calls.
