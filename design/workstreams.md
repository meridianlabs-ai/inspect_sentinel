# Workstreams

Areas of sentinel that can be owned separately, as of 2026-10-02. Each names its design document, its scope, and what it touches, so two people can work in parallel without colliding. Smaller agreed items that wait on something are in [pr-series.md](pr-series.md), "Deferred".

Work on your own branches and open PRs into `feature/sentinel`, in inspect_ai and in ts-mono alike. Pushing directly to the shared branch collides.

## 1. Building good, validated monitors

**Design:** [sentinel-development.md](sentinel-development.md).

The development loop: build a monitor, measure it over transcripts, calibrate it, deploy it. Reframed by the maintainer on 2026-10-02 around that goal rather than around Scout: the loop may be model-driven (supporting Claude Code iterating on a monitor) and may offer verbs such as `calibrate()`, using Scout under the hood where it makes scanning efficient. It is related to building good model judges. Scope:

- Reconstruct steps from messages and from events.
- An adapter that runs a monitor as a Scout scanner.
- A read-mode scanner over recorded `SentinelEvent`s.
- `calibrate()`, producing per-dimension thresholds in the key format `threshold` accepts (`"instance.dimension"`, `"dimension"`, `"*"`).
- `Result.subject` and row expansion in Scout.

Lives mostly in inspect_sentinel, with inspect_scout where Scout does the scanning, and consumes the sentinel API without changing it, so it rarely collides with core work. Uses `references` on reports (inspect_ai's `scorer.Reference`, as Scout's `Result` does) and the decorators' `version=`, which calibration records.

## 2. inspect_core

**Design:** [inspect-core.md](inspect-core.md).

Extract Inspect's wire types (`ChatMessage`, `ToolCall`, `ModelOutput` and what they reference) and the registry primitives into a leaf package. What it unblocks:

- The dependency chain becomes `inspect_core ← inspect_sentinel ← inspect_ai`, so inspect_ai depends on sentinel directly. The lazy import, `TYPE_CHECKING`-only types and the temporary `--no-deps` CI install go away.
- `@monitor` and `@protocol` register through the shared registry with no import-order hazard.
- The wire types get a contract, which the proxy and codegen work in [sentinel-deployment.md](sentinel-deployment.md) need.
- Sentinel can release against a version floor of the leaf package rather than an inspect_ai branch.

inspect_core is a second package inside the inspect_ai repository (UK AISI). When names move there, inspect_sentinel and inspect_ai re-export them, so user imports do not change (maintainer, 2026-10-02).

Touches modules that sentinel, Scout and inspect_ai all import. Agree up front which import paths stay stable while it is in flight, or the Scout work churns.

## 3. Generate stages

**Design:** [sentinel.md](sentinel.md), the payloads and "One vocabulary across stages".

`BeforeGenerate` and `AfterGenerate`: the payload types, the dispatcher hooks around model generation in inspect_ai, and the decision about what a generate-stage `modify` may replace (a message, the request, or nothing). That decision unblocks `Decision.modify(step, ...)`, deferred until then.

Lives in inspect_ai's model path and sentinel's step types. Best after inspect_core, or at least not overlapping its moves in `model/`.

## 4. `sequential()` and `human()`

**Design:** [sentinel.md](sentinel.md), "Humans in the loop" and the ordered composition.

The case they serve: a rule that escalates to a person, whose answer decides for the chain. `human()` reuses inspect's human approval surfaces (the approval panel, ACP and the console) and shows the escalations that led to it. Like the human approver, it calls inspect's `notify()` (Apprise, so Slack, email and the rest) before prompting, so the person learns a decision is waiting. `sequential()` is the ordered composition (formerly `chain`), passing escalations from one link to the next.

Moderate in size, but it touches runner semantics (escalation hand-off, `decide_final()`, cancellation), so it needs close review.

## 5. Bridged agents and deployment

**Design:** [sentinel-deployment.md](sentinel-deployment.md).

**Known gap today:** a sentinel never runs for a bridged agent's tool calls; only tool calls that go through `execute_tools` are checked. First step: the before-call hook in `bridge_generate`, next to `apply_bridge_tool_approval`, so bridged agents get sentinel checks on their tool calls. With it, verify that handoff and bridged agents open agent spans consistently, so conversation ids link as designed, with tests. The proxy deployment (a sentinel at the network boundary, the host ABI, sidecar and WASM modes) is a later, larger project.

## 6. Shipped protocols and helpers

**Design:** [sentinel.md](sentinel.md), the shipped protocols, views and "Failure semantics".

Many small, independent tasks, good for onboarding:

- `resample`. `defer_to_trusted` was removed from the plan by the maintainer on 2026-10-02: an eval has no trusted model to defer to.
- Views: helpers that give a monitor author views of the messages and the step, such as the last six messages (maintainer, 2026-10-02).
- A helper that renders a call's arguments as text (`Sequence[str]`), and similar conveniences that make rules shorter (maintainer, 2026-10-02).
- The prompt helpers, such as `monitor_prompt`, unless they move to workstream 9.

## 7. Auto-mode approvers

Support approvers in inspect modelled on the auto modes of Claude Code and Codex, in which a model decides whether each tool call may run without asking a person. Scope to establish first: how each auto mode decides (its inputs, policy and outputs) and what of that can be reproduced or reused. Then offer it in Inspect both as an `@approver`, for `Task(approval=)` users, and as a sentinel protocol, so it composes with monitors and `threshold`.

Open questions: which model judges by default, and how its policy is configured; whether the result is a binary allow/deny or a score that `threshold` calibrates; how it maps to approval's vocabulary (`approve`) and sentinel's (`continue`); and, where an existing product's prompt or policy is reused, whether its terms allow that.

## 8. Remote human surfaces

Let a person decide from outside the eval process: a request with the call, the escalations and the choices, and the answer flowing back to the waiting sample. Decided by the maintainer on 2026-10-02: put human requests on a message queue that other tools consume (Slack, a review app) rather than building the interface ourselves. Lower priority. Inspect's human surfaces today (the panel, ACP and the console) all assume someone at the eval's terminal or client; long-running and remote evals have nobody there. `notify()` already covers telling someone a decision is waiting (workstream 4); this is answering it.

Open questions: how the reply reaches the sample (a callback endpoint, polling, a relay service); how long a sample waits and what happens when nobody answers (proceed, reject, or the escalate default from pr-series.md "Deferred"); who may answer and how that is authenticated and recorded in the log; and whether the same surface serves inspect's existing human approver as well as `human()`.

## 9. LLM affordances for monitors

**Design:** [sentinel.md](sentinel.md), "Views" and "Inference, budget, and injection".

Raised by the maintainer on 2026-10-02, as a peer to workstream 6: what a monitor that calls a model needs, so each author does not rebuild it. Scope:

- Model interaction as in Inspect Scout's LLM scanner: prompt templates, structured answers, and parsing with retries.
- Chunking inputs that exceed the monitor model's context window (the conversation, long tool output) and reducing the chunks to one result.
- Prompt caching: keep a stable prefix across steps and avoid changes that break the cache.
- Helpers that build views for the model, turning the conversation and the call into a prompt.

Prior art: Kai's structured-approval approver in Petri, built on the LLM scanner. Lives in inspect_sentinel over `Host.generate`; shares the view helpers with workstream 6.

## 10. Host in a proxy

**Owner:** Pete. **Design:** [sentinel-deployment.md](sentinel-deployment.md).

Raised by the maintainer on 2026-10-02: a technical spike implementing a monitor host in a proxy, WASM hosting in Envoy first, then an RPC-based interface in another proxy that supports one. Includes the changes to the `Host` interface, `Context` and the rest that proxy deployment needs. It is the later, larger project in workstream 5; coordinate with the bridged-agent work there and with workstream 12.

## 11. Monitoring the monitors

**Owner:** Pete.

Raised by the maintainer on 2026-10-02 and not well defined yet: reporting on what sentinels did across tasks and evals. Ideas so far:

- Append terminations to a shared file (for example in S3).
- A job that reads many logs and summarises each protocol's behaviour.
- A sentinel-activity JSON file inside the `.eval` file.

Reads `SentinelEvent`s and the recorded configuration, so it overlaps the read-mode scanner in workstream 1.

## 12. Portability linter

**Design:** [sentinel-deployment.md](sentinel-deployment.md), "Keeping monitors portable" and "What gets bundled".

Raised by the maintainer on 2026-10-02: an AST-based linter that checks a monitor is portable to the proxy, with no captured state and only imports that can be bundled. It is the static pass that document's mitigations call for, and what a `portable=False` declaration would exempt a monitor from. Consumes the decorators and the module layout without changing them; the proxy work in workstream 10 decides what counts as bundleable.
