# Running sentinels outside the eval process

Companion to `sentinel.md`, which defines monitors (which observe) and protocols (which decide). This document covers deploying a sentinel where there is no eval: inside a proxy on the wire in front of a model API.

Status: measured where marked, reasoned elsewhere. The WASM claims were measured by the WASM sidecar spike (sentinel #58, `spikes/wasm_sidecar/`) on 2026-10-06.

## Why bother

A sentinel enforced at the network boundary is tamper-resistant in a way an in-process one is not — the agent cannot bypass it — and it covers all agent traffic rather than only what an eval generates. There is no `Task`, no sample, no `Store`, and the traffic is not necessarily Inspect's.

That deployment restricts the protocol sharply, which is why the constraints belong in the design rather than in a later integration note.

## What a proxy can see

A proxy sits on one HTTP exchange: a request and a response. It never sees a tool execute, because tools execute client-side after the response is delivered.

So the four stages are two, with the tool stages recovered as projections:

| Stage            | In a proxy                                 | Fidelity  |
|------------------|--------------------------------------------|-----------|
| `BeforeGenerate` | the request body                           | full      |
| `AfterGenerate`  | the response body                          | full      |
| `BeforeToolCall` | `tool_use` blocks in the response          | good      |
| `AfterToolCall`  | `tool_result` blocks in the *next* request | **lossy** |

A proxy sees a tool result only if the agent makes another request carrying it. The final step of an episode is never seen, and by the time a result appears the model has already consumed it — verbatim the limitation #5355 wrote the `Reviewer` protocol to escape. No vocabulary choice fixes this; it is a property of where the code runs, and `AfterToolCall` should say so.

## Evidence from an analogous constraint

`agent/_bridge/_approval.py` already enforces a policy with nothing but a model response to work with: a bridged scaffold runs its own tool loop, so Inspect never executes its calls. The deployment differs — that code runs in-process — but the constraint is the proxy's exactly. Five restrictions it arrived at, each of which transfers:

1.  **`reject` cannot be an edit to the response.** Stripping a rejected `tool_use` block "reads as 'turn complete' to most scaffolds, which stalls the agent". Rejection is a *replay*: append the rejected call and a synthetic result to the model's input and generate again. The proxy owns a retry loop.
2.  **Rejection therefore needs a bound.** `MAX_CONSECUTIVE_REJECTIONS = 3`, because a model that keeps proposing rejected calls would loop forever. In-process, `react()` owns the loop and this does not arise.
3.  **`modify` narrows to arguments only** — "the scaffold dispatches on the function name and may have no handler for a substituted one".
4.  **`terminate` is out-of-band**: not in the response at all, but a side call, `bridge.request_terminate(...)`.
5.  **`n = 1` only.** Alternate choices carrying tool calls are dropped with a warning, since returning them would hand the scaffold calls no approver saw.

Plus a payload restriction from the same file: there is **no `ToolCallView`** — "bridged tools reach us as `ToolInfo` from the scaffold's request, not as `ToolDef`, so there is no registered viewer to resolve." A proxy has wire types, not registered Inspect objects.

## The restricted vocabulary

**Every portable decision is an edit to the exchange.**

| Action | As an edit | Portable |
|----|----|----|
| `continue` | no edit | ✓ |
| `modify` | rewrite the body (arguments only, for a call) | ✓ |
| `reject` | append and regenerate, bounded | ✓ |
| `escalate` | defer to the next filter in the chain | ✓ |
| `terminate` | not an edit — an effect | ✓, with caveats below |

**`terminate` travels.** An earlier draft of this document claimed it did not, on the grounds that a production deployment has no sample lifecycle to end. That was wrong, and three pieces of evidence already collected here point the other way:

- **ext_proc has the primitive.** Its vocabulary is continue / modify / **immediate-response** — and immediate-response is precisely "stop, do not forward upstream, return this instead." The earlier draft listed that word and then concluded Envoy had no word for it.
- **`_bridge/_approval.py` already does this out-of-band.** Restriction 4 above: `terminate` is "not in the response at all, but a side call, `bridge.request_terminate(...)`." That is a working pattern for reaching a lifecycle owner that is not the response path.
- **The host ABI is exactly that channel.** `context.host` exists so a monitor can reach the outside world through an enumerable surface. A lifecycle owner is a thing in the outside world.

Three mechanisms, with different blast radii, and the distinction matters more than the word:

| Mechanism | Ends | Notes |
|------------------------|------------------------|------------------------|
| error response | this exchange | the agent's SDK raises; a retry-looping scaffold may retry |
| `host.terminate()` | this session or agent | out-of-band call to whatever owns the lifecycle |
| hanging the request | nothing, usefully | **don't** — see below |

**An error response is not `reject` with more force.** `reject` is a *conversational* move: append a synthetic result, regenerate, let the model learn it was denied. An error is a *transport* move: the SDK raises and the loop breaks. They are genuinely different actions and a proxy can do both, which is what the earlier "refuse, and keep refusing" framing obscured. That framing is still the right answer for a *scaffold* that swallows errors, and `MAX_CONSECUTIVE_REJECTIONS` is still the miniature version of it — but it is a fallback, not the definition.

**Never hang the request.** It is indistinguishable from a network failure, it holds a connection open, and it invites exactly the retry storm a safety mechanism should not cause. It is a denial-of-service pattern pointed at your own infrastructure.

So there is no `PortableAction` subset — the vocabulary crosses whole, and what varies is a host capability rather than a set of words. Two honest caveats remain, and both are about *guarantee*, not expressibility:

- **`host.terminate()` is an optional host capability.** A proxy in front of a public API serving anonymous clients has no session to end, so the ABI has to admit "not supported" and the protocol has to degrade to an error response.
- **It is best-effort and asynchronous where the eval's is total and synchronous.** In an eval, `terminate` cancels sibling tool calls, records an operator limit and ends the epoch, all at once. In a proxy, signalling a control plane takes time, in-flight work elsewhere is not cancelled, and there is no single log to record it in. `terminate` crosses as an *effect* requested, not as a state reached.

## The host ABI

The single most useful structural decision: define a monitor as pure logic plus a small declared effect surface. Then the compilation and transport strategy becomes a deployment detail rather than a design constraint.

``` python
class Host(typing.Protocol):
    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | Model | None = None,  # a model name or instance; used when role is not configured
        role: str | None = None,           # a model role; neither means "monitor"
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput: ...
    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer: ...
    """Ask a person about the step, rendered by its type; a proxy might use a review queue.
    A person who ends the prompt without an offered choice yields `reject` if offered, otherwise `terminate`."""
    async def fetch(
        self,
        endpoint: str,                      # a NAME, not a URL — see below
        *,
        method: str = "POST",
        json: Any = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HostResponse: ...
    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, value: dict) -> None: ...
    async def terminate(self, reason: str) -> bool: ...
    """End this session or agent out-of-band. False if unsupported."""


class HostResponse(typing.Protocol):
    status: int
    headers: Mapping[str, str]
    def json(self) -> Any: ...
    @property
    def text(self) -> str: ...
```

Only `generate` and `ask_human` are built; `fetch`, `get`/`put` and `terminate` are planned.

`ask_human` answers one of the offered choices, or, when the person ends the prompt without one, `reject` if it was offered and `terminate` otherwise, so the step fails closed; `human()` accepts that `terminate` even when it was not offered.

Known limits of inspect_ai's surfaces: ACP cannot withdraw a permission request when the ask is cancelled, so the card stays and a later click on it is discarded; an ACP ask after a call resets the call's card to pending; and inspect_ai's surfaces return fixed reason text and cannot modify a call.

`model` and `role` are separate so a role name is never read as a model name; as with inspect's `get_model(model=, role=)`, a configured role wins, then `model`, then the agent's model with a warning once per eval naming the role to set. With neither, the role is `monitor`.

`generate` returns a `ModelOutput` rather than a string because a protocol that substitutes an action (`resample`) regenerates with the agent's tools and need the whole response; a text-prompt monitor reads `.completion`. Generating as the agent's own model is, in a proxy, a re-issued upstream request.

The same surface whether the host is an `ext_proc` sidecar embedding a WASM runtime, a Go runtime, or ordinary Python in-process where these are normal implementations. In a WASM host it becomes WIT imports (spike #58: `generate`, `fetch`, `sleep`, `get`, `put`, with JSON payloads).

Recording is not on the ABI. The runner records every report, failure, cancellation, bypass and supersession through a separate `Recorder` that the host implements and authors never see (`sentinel-reference.md`, "Context"), so a protocol cannot skip recording by not calling it. The host builds the top layer's `Context` and hands it to `run_sentinel` in a `HostContext` together with its recorder and the sample store. `Context` has no private fields: the runner keeps the recorder, the store and each running layer's factory in a context variable that `run_sentinel` sets for the step, so a monitor or protocol is handed only author data.

A monitor reaches it as `context.host` (`sentinel.md`, "The context"), never by import. That is what keeps a monitor a pure function of its two arguments, and it is why the ABI is small enough to be worth specifying: everything a portable monitor can do to the outside world is enumerable from one interface.

### Why `fetch` is a primitive, not an escape hatch

Monitors routinely need organisational services that are not LLMs: a purpose-built classifier endpoint, a domain allowlist, a threat-intel feed, an internal risk score, a SIEM sink. That is ordinary monitor work, not an edge case.

And it cannot be `httpx`. **A WASM guest has no sockets** — all I/O crosses the host boundary by construction. A monitor that imports an HTTP client is not portable no matter how few other dependencies it has. So the ABI either provides HTTP or portable monitors can only talk to a model.

### JSON-oriented, with JSON-RPC as a helper

`fetch` encodes and decodes JSON and surfaces status, rather than handing back bytes for every monitor to re-implement encode/decode/status/error handling.

JSON-RPC then needs no separate primitive — it is a POST with a `{jsonrpc, method, params, id}` body and an envelope to unwrap, so it is a helper over `fetch` rather than a second ABI function:

``` python
async def rpc(host: Host, endpoint: str, method: str, params: Any) -> Any:
    ...   # POST the envelope, raise on `error`, return `result`
```

**gRPC is the right thing to leave out.** In a guest it would need a protobuf runtime and a generated stub per service compiled into the image, and most internal JSON services do not speak it anyway. It also does not need to be in the guest: because endpoints are *named* rather than addressed (below), a host can expose a gRPC service behind a named endpoint and translate. The guest always speaks JSON; whether the host turns that into REST, JSON-RPC or gRPC is configuration.

### Named endpoints, not URLs

`fetch` takes an endpoint *name* that host configuration resolves to a URL, auth and transport. This is the security-critical part of the design.

A monitor reads model completions, tool arguments and tool output — all agent-controlled text, and a prompt-injection surface (see `sentinel.md`, "Inference, budget, and injection"). A monitor that can then make arbitrary outbound HTTP calls is a confused deputy sitting on the network boundary: **unrestricted egress turns the monitor into an exfiltration channel**, and the payload it would exfiltrate is the conversation it was given privileged access to.

Naming the endpoints closes that:

- injected text cannot steer a monitor to an attacker-controlled host, because the monitor cannot express one
- credentials never enter the guest; the host attaches them
- the deployment operator owns the egress surface and can audit it
- the same monitor runs against different endpoints in dev and prod without code changes, which is a portability win independent of security

A host that wants to allow arbitrary URLs can still register a `"*"`-style endpoint, but that should be a deliberate configuration choice rather than the default shape of the API.

### Other constraints on `fetch`

- **Timeouts and budgets are the host's.** A monitor sits in the request path; an external call that hangs stalls the agent. The host enforces a ceiling regardless of what the monitor asks for.
- **Retries belong to the host too**, for the same reason and so that retry policy is uniform across monitors.
- **It must be non-blocking**, exactly like `generate()`. In a WASM guest both are component-model async imports, which the spike (#58) showed overlap under the guest's asyncio loop ("WASM", below).
- **Record calls for replay.** If a monitor decision is to be reproducible from a transcript, its external calls have to be recorded alongside its inputs. Worth deciding early whether `fetch` results are part of the recorded decision.

### Should `generate` just be `fetch`?

It could be — `fetch("model", ...)` — but keeping it distinct is better. Model calls carry token accounting, model roles, provider retry/backoff, and the limit-suspension semantics approvers already have (`sentinel.md`, "Inference, budget, and injection"). Collapsing them into a generic HTTP call loses all of that, and the host would have to re-derive it from the endpoint name.

### Two further notes

- **Regex belongs in the ABI** if transpilation is ever a target. Python's `re` has backtracking, lookahead and backreferences; Go's RE2 has none and Lua patterns are not regex. A monitor with a lookahead silently is not portable. Routing regex through the host makes semantics consistent by construction.
- **Where the ABI lives differs by path.** In the sidecar model it stays entirely inside Python — the sidecar implements it and the proxy only posts payloads. In the embedded model it becomes a real cross-language interface. So its design barely matters for the sidecar and matters a great deal for embedding: design it now even if the sidecar ships first.

### Where the protocol runs

`sentinel.md` splits the monitor (observes, reports) from the protocol (authorizes, acts). That split survives the trip, and it is mostly good news here: the leaf monitor stays a pure function of one exchange, while everything stateful — thresholds, budget, aggregation — belongs to the protocol, which is also the thing that owns the keyed store described above.

The split is one of types, `@monitor` and `@protocol`, but both are plain functions of `(context, step)`, and that matters here in one specific way. The portability verdict is per function, and each configured instance is checked on its own by its factory (decided 2026-10-05): a protocol is not disqualified by a leaf monitor calling `get_model()`, but a configuration that includes a non-portable instance cannot be deployed, so a composition is deployable only when every member is. A protocol reaching for task-scoped state disqualifies itself while leaving its leaves deployable. The same static check covers both, which would not have been true had protocols been a separate kind of object with their own affordances.

The two compositions cross intact. `concurrent()` and `sequential()` are ordinary code over the runner, and a `sequential`'s hand-off travels as data — `step.escalations` is a field on the payload, not a reference to a `ToolCallView` — so a `sequential` deserializes and runs in a proxy exactly as in-process. What changes is the last link: a `human()` in a proxy is an audit queue rather than a blocking prompt, which is the `audit`-travels-cleanly point below applied to a `sequential`, and it means a proxy `sequential` ending in a person degrades to "record, request review, and apply its default" rather than waiting.

Three notes:

- **`audit` travels most cleanly of all.** Routing a suspicious exchange to a human review queue is an ordinary thing for a proxy to do, and arguably more natural in production than in an eval. Unlike `terminate`, which crosses as a best-effort effect, an audit request is fully honoured on the proxy side — because auditing is a flag on a report rather than an action on the exchange.
- **Budget allocation needs an episode.** Ctrl-Z spends 1% of actions per episode with exponential decay. A proxy has no episode boundary, only a conversation, so the budget becomes per-conversation (keyed as above) or a rate over time. That is a protocol design choice, not a framework one.
- **Observe-only is the right proxy default for scores**, for the same reason it is in an eval, and more so: a threshold nobody calibrated, acting on the network boundary, can break production traffic. The asymmetry `sentinel.md` draws between rules and monitors holds here too — a rule deployed to refuse a call is doing what a proxy filter has always done, and needs no threshold; a monitor's score needs one, and a proxy is the last place to guess it.

## State

A proxy has no ContextVar, no sample identity and no `Store`. It passes `Context.eval=None`, since a request has no task, sample or epoch. Two mechanisms replace the store, and they are complementary.

**The conversation is state.** Every request carries the entire history, so a monitor that is a pure function of the request needs no storage: "has this agent reached the internet" is recomputable by scanning messages. Nothing to key, evict, or forge.

**A keyed store in the proxy layer.** A session header plus a local store, carrying what the conversation does not: the monitor's own accumulated judgment.

The request is `step.input` exactly, which is the one place a proxy beats an eval: nothing sits between the scaffold and the wire. What the request stops carrying once the agent compacts is the pre-compaction history, so `step.history` under a proxy is the keyed store's accumulation of what each request added for that session, with the session key doubling as the step's `conversation` id. Without a store, `history` is `input`, and a monitor that needs the full trajectory should declare it, so that a storeless deployment refuses it at configuration time rather than running it on a truncated view.

### The store is not merely a convenience

Re-judging the whole trajectory on every request is O(n) work at turn n, so O(n²) across a conversation. When judging is an LLM call that is O(n²) *inference* — by turn 40 the monitor has spent more than the agent. The affordable shape is incremental: classify only the newest turn and fold the verdict into running state. Incremental monitoring requires somewhere to fold into.

It also holds what the conversation genuinely does not contain: the monitor's own prior reasoning, a confidence trajectory, which calls it already cleared.

### Making the key trustworthy

- **Per-principal state** (API key, tenant, mTLS identity) is already trustworthy. Derive it from the proxy's authentication context and strip any client-supplied copy of the header on the way in — the ordinary gateway rule.
- **Per-conversation state** is harder: the authenticated principal does not identify a conversation, one key runs many at once, and a client can rotate a discriminator it supplies.

The fix comes from the first mechanism: **derive or verify the conversation key from the conversation itself.** A fingerprint over the stable head of the history — system prompt plus first user turn — cannot be rotated without discarding the context the agent needs the model to have. Use the client header as a fast path and the fingerprint as the check, or key on the fingerprint alone.

So the store is as trustworthy as its key, and the key can be anchored in data the agent must send truthfully.

### What backs `context.store_as()`

`StoreModel` itself is portable, and it should be Inspect's rather than a second one of our own. It is a genuinely odd class — it overrides `__getattribute__` to intercept every attribute read, overrides `__setattr__` to re-validate on every write, and namespaces store keys by class name — but none of that is WASM-hostile. It is pure-Python metaprogramming over Pydantic, and `pydantic-core`, the one native extension in the `inspect_core` floor, compiles to wasm32 (Pyodide ships it). Spike #58 ran `StoreModel` unchanged in a WASI guest (2026-10-06). Its state must be JSON-serializable either way, which the keyed store needs regardless.

The argument against a second `StoreModel` is ecosystem rather than effort. Monitor state is read back by ordinary `store_as()` from a scorer, so both sides must agree on the storage format; two same-named classes with different namespacing would split that. The import cost of reusing Inspect's is negligible — `_store_model.py` imports only `typing`, `pydantic` and `._store`.

**Measured overhead, native CPython, versus a plain `BaseModel`:**

| Operation                | `StoreModel` | plain   | ratio   |
|--------------------------|--------------|---------|---------|
| read a scalar field      | 7.8 µs       | 0.03 µs | \~260x  |
| read a `list[str]` field | 37 µs        | 0.03 µs | \~1200x |
| write a scalar field     | 66 µs        | 0.19 µs | \~350x  |

Two causes. `_coerce_value` constructs a fresh `TypeAdapter` on **every** read of a non-scalar field — the comment claiming this happens "once per field" is wrong, measured at 100 constructions for 100 reads — and `__setattr__` re-runs `model_validate` over the whole model on every write, measured at 100 full validations for 100 writes.

It is a large *constant*, not a complexity problem: read cost is flat in list length (35 µs at 10 elements, 46 µs at 1000), and a 400-step trajectory doing one read and one write per step totals 39 ms. Against 400 inference calls that is invisible, which is why this has never surfaced in an eval.

It is more visible in a proxy, where a deterministic monitor has no inference to hide behind and \~100 µs of store access is a real share of what a rule-based filter should cost. Three local fixes, none WASM-specific and all of which pay off in-process too: cache the `TypeAdapter` per (class, field); validate the single changed field on write rather than the whole model; and skip the store write-back when coercion produced an equivalent value.

What is open is the *backing*. In-process, `store_as()` is backed by the sample `Store`, and that buys transcript recording, checkpoint survival and `store_from_events_as()` reconstruction for free. In a proxy it is the keyed store described above, and none of that machinery applies.

`Store` records itself to the transcript as a *diff stream*: `store_changes()` runs at span boundaries to emit a `StoreEvent`, and `jsonpatch.apply_patch` replays those events in `store_from_events()`. The diff is not an implementation whim — `store.get("items", [])` hands back a mutable container that callers mutate in place, so no write hook on `Store` could be complete, and snapshot-before/compare-after at a boundary is the only reliable way to learn what changed (full snapshots per span would also be O(spans × store size) in the log).

That machinery is therefore coupled to the transcript, and a proxy has no transcript. So the cut is a natural one rather than a nicety: typing the store the host puts in `HostContext` as a Protocol lets a proxy back it with its keyed store and carry none of the diffing. `Store` is thin enough (a dict wrapper; writes are plain dict operations) that a proxy *could* simply use it, in which case the only cost is closure weight — `jsonpatch` and `jsonpointer` are imported at module level in `util/_store.py` but used only on the transcript path, so a lazy import inside those two functions would keep them out of a constrained build.

### Operational consequences

- **No end-of-conversation signal.** Entries expire on a TTL rather than at a lifecycle boundary. Too short and a slow agent loses state mid-trajectory; too long and the store grows with every abandoned conversation.
- **Local means per-replica.** With multiple replicas a local store requires consistent hashing on the session key, or the state moves to shared storage. Sticky routing is cheaper and reuses the key that already exists.
- **Fail open on a store miss.** A cold replica, evicted entry or restart yields a monitor with no history. It should degrade to the pure-function path rather than read absent state as "nothing suspicious has happened".

### Consequence for the protocol

State reaches a monitor as an argument and only as an argument. `context.store_as(...)` survives here — the host builds one context per conversation and hands the monitor whatever it loaded for the key. An ambient accessor would not, since a proxy has nothing to bind to, which is one reason `sentinel.md` declines to provide one.

The same holds for effects. `context.host` is how a monitor reaches inference, keyed storage and outbound JSON, and routing them through an argument rather than an import is what makes the monitor a pure function of `(context, step)` — which is the property everything in this document depends on. Since `sentinel.md` made monitors annotated functions, that is now the literal signature rather than a description of one.

The qualification is the factory closure. `@monitor` decorates a factory, and anything it captures is created once per configuration and shared by every invocation — which is fine for configuration (a threshold, a model name) and wrong for anything accumulated. In an eval that yields a silent cross-sample race; in a proxy, state that a second replica does not have. The keyed store is the only durable place, and it arrives through `context`.

Scopes map cleanly: per-sample becomes per-conversation, per-task becomes per-principal — which a proxy is better placed to hold than the eval process, since it sees every conversation rather than one task's samples.

## Serialization

The intuition that Pydantic is a cost worth avoiding does not survive inspection.

`model_validate_json` hands raw bytes to `jiter` (Rust) and materializes only declared fields. `json.loads` uses CPython's parser and builds a complete dict of the entire payload, including the majority a monitor never reads. If the model is a subset of the payload — and it is — Pydantic can win end to end. A TypedDict-plus-`json.loads` design avoids the cheaper half of the work.

This holds in WASM too, where both are native code compiled to wasm32. Passing raw bytes into guest memory and validating inside the guest is likely the efficient path, and it is the Pydantic one. The Component Model alternative is probably worse: marshaling a deeply nested variant like `ChatMessage` across the component boundary builds intermediate representations and you still construct Python objects on top. One parse beats two.

Where deserialization cost *does* dominate is a pure heuristic monitor whose own work is microseconds — and the fix there is the same as everywhere else in this document: feed it the delta rather than the whole history.

## How the Python runs there

### External processor (Envoy `ext_proc`)

| `ext_proc` phase          | Affordance       |
|---------------------------|------------------|
| `request_body` (BUFFERED) | `BeforeGenerate` |
| `response_body`           | `AfterGenerate`  |

| `ProcessingResponse` | Action                                        |
|----------------------|-----------------------------------------------|
| no mutation          | `continue`                                    |
| `body_mutation`      | `modify`                                      |
| `immediate_response` | `reject`, and the `terminate` wire convention |

`ext_authz` is request-path only and cannot see the response body, so it cannot do `AfterGenerate`. `ext_proc` is the one.

The Python runs unchanged. What is new is an adapter on each side — and under the host-renders-core-JSON contract (`inspect-core.md`) the normalization half lives in the host, so the guest needs only the core types.

### Not transpilation

The objection is not that Python-to-Go and Python-to-Lua transpilers are immature, though they are. It is that **for a monitor the dependency graph is the monitor**: the central operation is an LLM call, meaning an async HTTP client, a provider SDK, and Pydantic whose core is Rust. Transpiling the monitor means transpiling all of it — a reimplementation of the ecosystem plus a permanent obligation to keep it in step with the Python authors test against.

The performance case does not survive arithmetic either. A localhost hop is well under a millisecond; an LLM-based monitor spends 200ms–2s in inference. The transport is \~0.1% of the cost.

There is also no stable resting place: push a Python subset far enough to be reliably portable — no deps, no `re` semantics gaps, no async — and what remains converges on rule evaluation, at which point a declarative format is simpler than shipping an interpreter. Keep it expressive enough to be worth writing in Python and you need CPython in WASM.

### WASM

WASM **does** have async, in several forms — "no stack switching, therefore host calls block" describes core WASM only:

- **Asyncify** (a Binaryen pass) instruments the module to unwind and rewind the stack, giving real suspension on plain WASM. Production-proven since \~2019; **Pyodide uses exactly this to make `await` work**. Costs roughly 2x code size and meaningful slowdown in instrumented paths.
- **WASI 0.2 `pollable` / `wasi:io/poll`** gives the guest non-blocking I/O.
- **Stack switching** (typed continuations) is the clean long-term answer; verify current proposal status rather than assuming.

The concurrency model matters more than the mechanism. One instance per in-flight request is *not* required: if asyncio runs inside the instance and host I/O is non-blocking, one instance serves many concurrent invocations — the ordinary event-loop model, inside WASM. Size instances to worker threads, not to in-flight requests. Compiled code is shared across instances; only linear memory is per-instance.

**Measured (spike #58, 2026-10-06): the guest has asyncio, and Asyncify is not needed.** CPython's stock asyncio loop and threads are unavailable on WASI (the Python docs mark both "Availability: not WASI"), so the question as first posed, a stock loop over `wasi:io/poll`, is answered no. But componentize-py's component runtime supplies its own asyncio loop over component-model async (WASIp3) host calls: `await host.generate()` parks one task, and the host can start more calls into the same instance. That loop has no timers, traps the instance when a task awaiting a host call is cancelled, and lacks methods anyio calls. The spike's guest patch (`loop_patch.py`, about 90 lines, plus a host `sleep` import) fixes all three. With it:

- monitors stay ordinary `async def`, and `concurrent()`, `asyncio.gather`, `TaskGroup` and anyio task groups overlap their host calls;
- timeouts work (`asyncio.wait_for`, anyio's `move_on_after` and `fail_after`), and so does cancelling a task while it awaits a host call: `no_destruction()` terminated a step with two LLM monitors in flight and the decision returned in 3 ms;
- `run_sentinel`'s per-step context variable stays isolated: componentize-py starts each call in a fresh `contextvars.Context`, and 20 concurrent steps in one instance each recorded their own layers;
- one instance served 50 concurrent steps × 4 monitors, 200 overlapping 500 ms `generate` calls, in 512–523 ms.

The blocking model works too. `generate` is a plain import that the host implements as an async function, and wasmtime suspends the guest on a fiber while the host waits. It needs no loop patch, but an instance runs one step at a time and calls within a step run in sequence, so concurrency comes from a pool of instances.

**What remains: a cancelled host call keeps running on the host.** The runtime has no subtask cancellation, so when `terminate` or a timeout cancels a task awaiting `generate`, the guest discards the result but the host's request runs to completion. Its tokens and rate limit are still spent, and host budgets must count it.

Two residual costs:

- **Linear memory only grows.** A large conversation becomes thousands of Python objects and permanently raises that instance's high-water mark; with pooled instances they all ratchet to the worst case. A third independent argument for incremental monitoring, after O(n²) inference and O(n) deserialization. Measured: an instance held a flat 22.4 MiB over 500 small steps; a large conversation was not tested.
- **You are still shipping CPython.** Four dependencies instead of eighty (`inspect-core.md`) is a large improvement, but the runtime dominates the artifact. Lightweight relative to the alternative, not in absolute terms. Genuinely small means a rule DSL, or monitors written in a language that compiles small.

Measured on the spike's component (CPython 3.14, pydantic, `inspect_ai.core`, the unchanged sentinel runner and two LLM monitors), wasmtime, Apple M-series:

| Measure | Value |
|---|---|
| Component | 29.8 MiB raw, 7.8 MiB zstd |
| Precompiled for the host (`.cwasm`) | 60.5 MiB |
| Load precompiled | 3 ms |
| Instantiate | 0.38 ms |
| Warm step | 0.38 ms, about 1.3× native CPython (0.29 ms) |
| Linear memory per instance | about 22 MiB, flat over 500 calls |

**Pre-initialisation is confirmed.** componentize-py runs the guest's module-level code at build time and snapshots the heap, so instantiation is 0.38 ms with 374 modules already imported, against about 100 ms of imports natively. The consequence for authors: anything read from the environment, seeded or timestamped at import time is frozen into the artifact, so portable monitors read configuration at run time.

### pydantic-core for WASI

No published WASI wheel of pydantic-core matches the CPython that componentize-py embeds (measured 2026-10-06). Wheels exist for cp312 and cp313; componentize-py has embedded CPython 3.14 since 0.18, and gained async in 0.19. Building it is easy: the spike cross-compiled pydantic-core 2.46.5 for cp314 `wasm32-wasip1` in about 30 s, with no CPython WASI build and one linker flag beyond the published recipe (`--unresolved-symbols=import-dynamic`). The wasi-sdk version must match the one componentize-py's CPython was built with.

Recommendation: own a CI job that builds pydantic-core for each supported pydantic release and the CPython that componentize-py embeds, and pin componentize-py, wasmtime and wasi-sdk together, since the component-model async ABI still changes between releases.

### What the guest can reach

The WASI context is empty unless the host grants something. Measured in the spike with an empty context: opening or listing files fails (no preopened directories), the environment is empty, sockets raise `PermissionError`, and subprocesses and threads cannot start. Host-mediated calls work: `generate`, and `fetch` by endpoint name, where a URL is refused as an unknown endpoint and the host attaches the credential. Granting a preopened directory or an environment variable gives the guest exactly that.

**File access is a grant, not a prohibition.** Decided by the maintainer on 2026-10-06: the sidecar gives the guest read-only access to files bundled with the monitor (word lists, prompt templates, policy files), as a read-only preopened bundle data directory or through `importlib.resources`, and nothing else by default. Arbitrary host paths are never granted. State belongs in `context.store_as()`, not in files: a write in a proxy is per instance and lost. The `portable=True` check does not check file access; the grant is the boundary.

This is the real enforcement of "effects only through `context`" (mitigation 1 in "Keeping monitors portable"). The static `portable=True` check is early feedback: it fails in an eval, before a monitor reaches a deployment, but the sandbox is what holds in one.

Resource limits work too: an epoch deadline and a fuel limit each trapped a spinning guest, and a memory cap made an oversized allocation raise `MemoryError` in the guest. A trapped instance is discarded and a fresh one costs 0.4 ms.

## What gets bundled

Portability is two axes, and conflating them loses information:

- **Affordance portability is per function** — which `context` members it touches, whether it decides `terminate`.
- **Bundle portability is per module** — the import closure. One `import requests` at the top of a file disqualifies every monitor in it, whatever the functions themselves do.

The class design could not express the first cleanly (a class with one portable method and one unportable one is an all-or-nothing verdict) and obscured the second. Annotated functions in modules give both.

### Monitors are enumerable without importing anything

`_util/decorator.py::parse_decorators` walks top-level `FunctionDef` nodes looking for a named decorator, using nothing but `ast`. A bundler gets two things from a source file with no code executed: which functions are monitors, and — from the file's own import statements — the closure to resolve.

This works *only* because monitors are top-level functions. A class body is an `ast.ClassDef` with its methods nested inside, invisible to that walk.

What is *not* statically available is the stage, since the annotation sits on the inner function a factory returns. That is fine and worth stating plainly so nobody designs around it: the bundler needs enumeration and the closure, both of which it has. Stages resolve at load time, once, inside the bundle.

### Imports inside function bodies

componentize-py bundles only the modules imported during build-time initialisation ("WASM", above), so a module first imported inside a function body is missing from the component and raises `ModuleNotFoundError` at run time. Decided by the maintainer on 2026-10-06: this is the bundler's job, not the portability check's. The bundler imports what portable functions' bodies import as well as module-level imports, or bundles whole packages.

One caveat if the existing helper is reused: `_util/decorator.py` imports `_util.file` for S3 support, which pulls `fsspec` and `s3fs`. The concept is \~40 lines of stdlib `ast`; a bundler-side scanner wants a local-path-only variant rather than the shipped helper.

### The module is a manifest entry, not a bundle boundary

One bundle per module is the wrong instinct. The Python runtime plus Pydantic plus `inspect_core` dominates the artifact (29.8 MiB raw in spike #58, above), so N bundles means N copies of the expensive part while the monitors themselves are kilobytes.

So: one bundle holding every selected module, with the module as the unit of *inclusion*. The `portable=True` check is per function; whether a module's import closure can be bundled is the bundler's per-module verdict. A module whose closure fails is dropped by itself, and the report names it — which it can, because that verdict is computed per module rather than per bundle.

### Selection stays an explicit list

`sentinel.md` declines to provide glob selection over registry names, on the grounds that grouping is rarer than it looks. That has a consequence here worth naming, because it runs the safe direction.

Enablement is an explicit list in both deployments. A bundle may therefore contain a module whose monitors the proxy never wires up — benign, since that is unused code in an artifact rather than an unenabled monitor quietly running. The dangerous direction is the inverse: a selection mechanism that resolves to a monitor the bundle does not contain, so the eval tests five and the deployment runs four with nothing reporting the difference. A pattern language could do that. An explicit list cannot.

## Deployment recap

|   | Monitor runs | Host provides | Codegen | Effort | Main catch |
|------------|------------|------------|------------|------------|------------|
| **Python proxy** | in-process call | nothing — ABI is native | none | hours | none; the reference deployment |
| **Go — sidecar** | Python service | HTTP client | Go structs (optional) | low | second deployable; fail-open policy |
| **Go — embedded** | CPython in wazero | ABI + normalizer | Go structs (required) | high | no sum types; pooling; memory ratchet |
| **Rust — sidecar** | Python service | HTTP client | Rust types (optional) | low | as Go |
| **Rust — embedded** | CPython in wasmtime | ABI + normalizer | Rust types (required) | medium | best tooling of the four |
| **Envoy — native sidecar** | Python `ext_proc` service | gRPC stream | none | low | second deployable; no isolation between monitors and the sidecar |
| **Envoy — WASM sidecar** | CPython in wasmtime, inside an `ext_proc` sidecar | ABI as WIT imports + normalizer | optional | medium | loop patch and pydantic-core build to own (spike #58) |

**Python proxy** — none of this document's discipline applies. Import `inspect_ai`, use `ChatMessage`, call the monitor.

**Go** — sidecar is hours and needs no WASM machinery. Embedded concentrates the Go-specific cost: no sum types means the bespoke emitter (`inspect-core.md`), and you own the async resumption model. `wazero` being pure Go with no CGo is what makes embedded tolerable.

**Rust** — quietly the best target on both paths: `serde` tagged enums fit the discriminated unions, `typify`/`progenitor` generate clean types, `wasmtime` has the strongest component tooling. A Rust shop may prefer writing monitors *in* Rust for a few-hundred-KB module — the genuinely lightweight path, at the cost of the Python ecosystem.

**Envoy** — most constrained, because the host is not yours, so the sentinel runs beside it in an `ext_proc` sidecar. The native-Python sidecar is the simplest option: it runs `run_sentinel` directly. When the deployment needs more, the sidecar embeds wasmtime and hosts the sentinel as a CPython-in-WASM component; spike #58 built that host in Rust (2026-10-06). WASM earns its keep where the sidecar must isolate monitors: from credentials, which the host keeps and attaches to named endpoints; from each other, for tenants; and from the request path's resources, with hard limits (an epoch deadline, fuel and a memory cap, all demonstrated in the spike). proxy-wasm inside Envoy is out: there is no Python proxy-wasm SDK, and its callback ABI is imposed rather than chosen and has no component-model async.

Embedding CPython in Go via CGo is the one option to avoid outright: goroutines and the GIL are a bad marriage, and you inherit libpython plus a venv as deployment artifacts.

## Keeping monitors portable

**The sidecar path works identically across all four deployments and needs none of the WASM work.** That makes it the obvious thing to ship first — and creates a trap: by the time WASM matters, the monitors that exist were written against whatever the sidecar permitted, by people with no incentive to port. "Ship the easy path first" quietly makes it the only path.

The enforcement surface is smaller than it looks, though, and depends on the target. For **WASM with CPython**, arbitrary Python *language* features are fine; the constraint is **dependencies** and the effects the sandbox lacks, so a check of what a function references, against a short denylist of what fails in the guest, is nearly sufficient. Only true transpilation would additionally constrain language semantics.

Mitigations, in order of how much they matter:

1.  **Make the sidecar execute portable monitors through the same restricted host ABI that WASM would use** — not merely permit it. If a portable monitor cannot call `get_model()` and must call `host.generate()`, every production run exercises the constraint and the gap between "we say it is portable" and "it is" goes to zero. This is the difference between a property and an aspiration. In-process there is no runtime guard (decided 2026-10-05): a `portable=True` function is checked from its source when its factory is called, not watched while it runs, so the check catches honest mistakes and the sidecar is where the restricted host is exercised.
2.  **Default to portable; make escaping one keyword.** `@monitor(portable=False)` should be a deliberate, visible choice that appears in code review and the registry, not the accidental result of writing normal Python. Default-permissive loses by construction; default-strict with no escape is hostile to the in-process case. The `portable=True` check is per function, by what the function's code references (decided 2026-10-05), replacing an earlier per-module rule: a `@task` or a `portable=False` monitor beside a portable one does not affect it. Whether a module can be bundled, given everything it imports, is a separate per-module verdict that belongs to the bundler.
3.  **Make the WASM build a CI target before a deployment target.** Compile the portable monitor set and run the conformance corpus against it. Catches drift the day it appears, at a fraction of deploying. The static pass is cheaper still and can run on every commit. The `portable=True` check is that pass for one function at a time: when a factory is called it reads the factory's source with `ast`, resolves each name the code references one level, and fails on a denylist of what fails in a WASM guest (inspect_ai outside `inspect_ai.core`, environment variables, processes and threads, direct networking, compiled packages without a WASI build), without following helpers (decided 2026-10-06; the sandbox is the enforcement and the check is early feedback). It is per function, not an import closure; the bundler's per-module closure check is separate. No build, no runtime, no WASM toolchain.
4.  **Decide `re` explicitly** — route regex through the ABI, or keep it and accept that non-CPython targets are foreclosed. The `portable=True` check allows it today, as it does the rest of the standard library outside its denylist.
5.  **Docs shape this more than checkers.** Whatever the examples use is what people write.
6.  **Ban the ambient escapes.** Chiefly `get_model()`, which bypasses `context.host`; the `portable=True` check refuses it, with the rest of inspect_ai outside `inspect_ai.core`. There is no ambient store accessor to ban, by design.
7.  **Make the portable path not feel like a downgrade.** If it is painful, authors escape and the check becomes an obstacle to route around. This is where `inspect-core.md` pays off twice: portable monitors get the *real* `ChatMessage` and `Content`, not a stripped parallel API. The restriction should be fewer dependencies, not worse ergonomics.

**Module-level code runs at build time** in a pre-initialised WASM component ("WASM", above): anything a module reads from the environment, seeds or timestamps when imported is frozen into the artifact. Portable monitors read configuration at run time, through their parameters.

Expect two tiers regardless, and design for them: portable monitors that run anywhere, and eval monitors using the full ecosystem in-process. An eval-time monitor doing sklearn classification over a trajectory is legitimate work that should not be constrained by a deployment it will never see. What matters is that monitors *intended* to be portable actually are — verified continuously rather than asserted — and that the default pulls toward portable so the choice is conscious.

## Status

**Measured:** `_bridge/_approval.py`'s restrictions and their stated rationales; the bridge provider-normalization layer; the import-closure figures in `inspect-core.md`.

**Measured (spike #58, 2026-10-06):** asyncio in a WASM guest (componentize-py's loop with the spike's patch) and the blocking fiber model; component size, load, instantiation, per-step time and memory per instance; pre-initialisation; building pydantic-core for WASI; the empty WASI sandbox and host-mediated calls; epoch, fuel and memory limits.

**Reasoned, not verified:** serialization throughput, `ext_proc` latency figures, memory with large conversations, and WASM hosts other than wasmtime.

## Open questions

1.  **Is portability a goal or an observation?** If a goal, the portable core should be the shape the protocol is designed around and the in-process extras visibly additive.
2.  **Does CPython-on-WASI have a working asyncio loop over `wasi:io/poll`?** Answered by spike #58 (2026-10-06): not the stock loop, but componentize-py's loop over component-model async host calls works with a small guest patch, and one instance serves many concurrent steps. See "WASM".
3.  **What owns the lifecycle a proxy's `terminate` reaches?** The mechanism is settled (an error response, or `host.terminate()` out-of-band); what is open is who implements the latter — an orchestrator API, a session store, a container control plane — and whether a monitor should be able to tell which it got before deciding.
4.  **Streaming versus buffering.** `BUFFERED` gives a complete `ModelOutput` at the cost of time-to-first-token; `STREAMED` preserves TTFT but forces judgment on partial output, after some tokens have reached the agent. A custom proxy can do better than either — forward text blocks while withholding `tool_use` blocks until cleared — which is a real capability difference from Envoy worth weighing in a build-versus-adopt decision.
5.  **Recursion guard.** The monitor's own LLM call must not traverse the proxy it runs inside; it needs a distinct route, upstream, or skipped header.
6.  **gRPC body limits.** The 4MB default is tight for LLM traffic, which re-sends the whole conversation each request.
7.  **Are `fetch` results part of a recorded decision?** Replaying a monitor from a transcript requires its external calls to be recorded, which means deciding what is safe to record (an allowlist response is fine; a credentialed payload may not be).
8.  **Fail open or closed** when the processor is down. A safety monitor wants closed, which makes it a hard dependency on the request path.
9.  **Does the `portable=True` check run per module or per function?** Decided 2026-10-05: per function, by what its code references. The per-module verdict (a module's import closure) belongs to the bundler.
10. **Should the store behind `store_as()` be a Protocol** rather than the concrete `Store`? The concrete class carries transcript diffing a proxy never uses; an interface lets a proxy back it with the keyed store directly. The cost is one more abstraction on a type authors already know.

From spike #58 (2026-10-06):

11. **Instance reuse versus isolation.** Module globals persist across steps in a reused instance, and so across conversations or tenants. A fresh instance per step costs 0.4 ms plus a 2 ms first call; the alternatives are an instance per conversation, or per worker with a reset discipline.
12. **Cancelling host calls.** Without subtask cancellation, a `terminate` or timeout cannot stop the host's in-flight request. The host could cancel on its side when the guest drops interest, but that needs a signal in the ABI.
13. **Memory with large conversations.** Linear memory stayed flat over 500 small steps; a long history was not tested.
14. **WASIp3 maturity.** Component-model async is not final. componentize-py's WASIp2 loop over `wasi:io/poll` is the untried fallback if a host runtime lags.
15. **A Go host.** Whether wazero supports components and component-model async is untested; the spike used only wasmtime.
