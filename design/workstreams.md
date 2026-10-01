# Workstreams

Areas of sentinel that can be owned separately, as of 2026-09-30. Each names its design document, its scope, and what it touches, so two people can work in parallel without colliding. Smaller agreed items that wait on something are in [pr-series.md](pr-series.md), "Deferred".

Work on your own branches and open PRs into `feature/sentinel`, in inspect_ai and in ts-mono alike. Pushing directly to the shared branch collides.

## 1. Scout integration

**Design:** [sentinel-development.md](sentinel-development.md).

The development loop: build a monitor, measure it over transcripts, calibrate it, deploy it. Scope:

- Reconstruct steps from messages and from events.
- An adapter that runs a monitor as a Scout scanner.
- A read-mode scanner over recorded `SentinelEvent`s.
- `calibrate()`, producing per-dimension thresholds in the key format `threshold` accepts (`"instance.dimension"`, `"dimension"`, `"*"`).
- `Result.subject` and row expansion in Scout.

Lives in inspect_scout and inspect_sentinel and consumes the sentinel API without changing it, so it rarely collides with core work. Uses `references` on reports (inspect_ai's `scorer.Reference`, as Scout's `Result` does) and the decorators' `version=`, which calibration records.

## 2. inspect_core

**Design:** [inspect-core.md](inspect-core.md).

Extract Inspect's wire types (`ChatMessage`, `ToolCall`, `ModelOutput` and what they reference) and the registry primitives into a leaf package. What it unblocks:

- The dependency chain becomes `inspect_core ← inspect_sentinel ← inspect_ai`, so inspect_ai depends on sentinel directly. The lazy import, `TYPE_CHECKING`-only types and the temporary `--no-deps` CI install go away.
- `@monitor` and `@protocol` register through the shared registry with no import-order hazard.
- The wire types get a contract, which the proxy and codegen work in [sentinel-deployment.md](sentinel-deployment.md) need.
- Sentinel can release against a version floor of the leaf package rather than an inspect_ai branch.

Touches modules that sentinel, Scout and inspect_ai all import. Agree up front which import paths stay stable while it is in flight, or the Scout work churns.

## 3. Generate stages

**Design:** [sentinel.md](sentinel.md), the payloads and "One vocabulary across stages".

`BeforeGenerate` and `AfterGenerate`: the payload types, the dispatcher hooks around model generation in inspect_ai, and the decision about what a generate-stage `modify` may replace (a message, the request, or nothing). That decision unblocks `Decision.modify(step, ...)`, deferred until then.

Lives in inspect_ai's model path and sentinel's step types. Best after inspect_core, or at least not overlapping its moves in `model/`.

## 4. `sequential()` and `human()`

**Design:** [sentinel.md](sentinel.md), "Humans in the loop" and the ordered composition.

The case they serve: a rule that escalates to a person, who ends the step with `decide_final()`. `human()` reuses inspect's human approval surfaces (the approval panel, ACP and the console) and shows the escalations that led to it. Like the human approver, it calls inspect's `notify()` (Apprise, so Slack, email and the rest) before prompting, so the person learns a decision is waiting. `sequential()` is the ordered composition (formerly `chain`), passing escalations from one link to the next.

Moderate in size, but it touches runner semantics (escalation hand-off, `decide_final()`, cancellation), so it needs close review.

## 5. Bridged agents and deployment

**Design:** [sentinel-deployment.md](sentinel-deployment.md).

**Known gap today:** a sentinel never runs for a bridged agent's tool calls; only tool calls that go through `execute_tools` are checked. First step: the before-call hook in `bridge_generate`, next to `apply_bridge_tool_approval`, so bridged agents get sentinel checks on their tool calls. The proxy deployment (a sentinel at the network boundary, the host ABI, sidecar and WASM modes) is a later, larger project.

## 6. Shipped protocols and helpers

**Design:** [sentinel.md](sentinel.md), the shipped protocols, views and "Failure semantics".

Many small, independent tasks, good for onboarding:

- `defer_to_trusted` and `resample`.
- The prompt and view helpers, such as `monitor_prompt`.
- The failure policy: `@monitor(fail="open")` and `@protocol(fail="open")`, with the per-child hook in the runner.

## 7. Auto-mode approvers

Support approvers in inspect modelled on the auto modes of Claude Code and Codex, in which a model decides whether each tool call may run without asking a person. Scope to establish first: how each auto mode decides (its inputs, policy and outputs) and what of that can be reproduced or reused. Then offer it in Inspect both as an `@approver`, for `Task(approval=)` users, and as a sentinel protocol, so it composes with monitors and `threshold`.

Open questions: which model judges by default, and how its policy is configured; whether the result is a binary allow/deny or a score that `threshold` calibrates; how it maps to approval's vocabulary (`approve`) and sentinel's (`continue`); and, where an existing product's prompt or policy is reused, whether its terms allow that.

## 8. Remote human surfaces

Let a person decide from outside the eval process, Slack first: a message with the call, the escalations and approve/reject buttons, and the answer flowing back to the waiting sample. Inspect's human surfaces today (the panel, ACP and the console) all assume someone at the eval's terminal or client; long-running and remote evals have nobody there. `notify()` already covers telling someone a decision is waiting (workstream 4); this is answering it.

Open questions: how the reply reaches the sample (a callback endpoint, polling, a relay service); how long a sample waits and what happens when nobody answers (proceed, reject, or the escalate default from pr-series.md "Deferred"); who may answer and how that is authenticated and recorded in the log; and whether the same surface serves inspect's existing human approver as well as `human()`.
