# Workstreams

Areas of sentinel that can be owned separately, as of 2026-10-05. Each names its design document, its scope, and what it touches, so two people can work in parallel without colliding. Sections are in priority order, decided by the maintainer on 2026-10-02: high priority (1 to 6), next (7 to 11), lower priority (12 and 13), then done; 5 and 6 were added on 2026-10-05. Smaller agreed items that wait on something are in [pr-series.md](pr-series.md), "Deferred".

Work on your own branches and open PRs into `feature/sentinel`, in inspect_ai and in ts-mono alike. Pushing directly to the shared branch collides.

## 1. inspect_core

**Priority:** high. **Owner:** epatey. **Design:** [inspect-core.md](inspect-core.md).

Extract Inspect's wire types (`ChatMessage`, `ToolCall`, `ModelOutput` and what they reference) and the registry primitives into a leaf package. What it unblocks:

- The dependency chain becomes `inspect_core ← inspect_sentinel ← inspect_ai`, so inspect_ai depends on sentinel directly. The lazy import, `TYPE_CHECKING`-only types and the temporary `--no-deps` CI install go away.
- `@monitor` and `@protocol` register through the shared registry with no import-order hazard.
- The wire types get a contract, which the proxy and codegen work in [sentinel-deployment.md](sentinel-deployment.md) need.
- Sentinel can release against a version floor of the leaf package rather than an inspect_ai branch.

inspect_core is a second package inside the inspect_ai repository (UK AISI). When names move there, inspect_sentinel and inspect_ai re-export them, so user imports do not change (maintainer, 2026-10-02).

Touches modules that sentinel, Scout and inspect_ai all import. Agree up front which import paths stay stable while it is in flight, or the Scout work churns.

## 2. Generate stages

**Priority:** high. **Owner:** cteague. **Design:** [sentinel.md](sentinel.md), the payloads and "One vocabulary across stages".

`BeforeGenerate` and `AfterGenerate`: the payload types, the dispatcher hooks around model generation in inspect_ai, and the decision about what a generate-stage `modify` may replace (a message, the request, or nothing). That decision unblocks `Decision.modify(step, ...)`, deferred until then.

Lives in inspect_ai's model path and sentinel's step types. Best after inspect_core, or at least not overlapping its moves in `model/`.

## 3. LLM affordances for monitors

**Priority:** high. **Owner:** epatey. **Design:** [sentinel.md](sentinel.md), "Views" and "Inference, budget, and injection".

Raised by the maintainer on 2026-10-02, as a peer to workstream 7: what a monitor that calls a model needs, so each author does not rebuild it. Scope:

- Model interaction as in Inspect Scout's LLM scanner: prompt templates, structured answers, and parsing with retries.
- Chunking inputs that exceed the monitor model's context window (the conversation, long tool output) and reducing the chunks to one result.
- Prompt caching: keep a stable prefix across steps and avoid changes that break the cache.
- Helpers that build views for the model, turning the conversation and the call into a prompt.

Prior art: Kai's structured-approval approver in Petri, built on the LLM scanner. Lives in inspect_sentinel over `Host.generate`; shares the view helpers with workstream 7.

## 4. Host in a proxy

**Priority:** high. **Owner:** Pete. **Design:** [sentinel-deployment.md](sentinel-deployment.md); spike plan in [proxy-host.md](proxy-host.md).

Raised by the maintainer on 2026-10-02: a technical spike implementing a monitor host in a proxy, WASM hosting in Envoy first, then an RPC-based interface in another proxy that supports one. Includes the changes to the `Host` interface, `Context` and the rest that proxy deployment needs. It is the later, larger project in workstream 9; coordinate with the bridged-agent work there and with workstream 11.

## 5. Landing and the first release

**Priority:** high. **Design:** [pr-series.md](pr-series.md).

Raised by the maintainer on 2026-10-05: land the open work and ship inspect_sentinel to PyPI. Before the release the public names are settled, since they cannot change freely afterwards. Release Please is paused (inspect_sentinel #4) until then.

Decide first, as a short review of the public surface:

- The package and the concept: `inspect_sentinel`, "sentinel", and `Task(sentinel=)` / `eval(sentinel=)` / `--sentinel` in inspect_ai.
- The two kinds and their decorators: `Monitor` / `@monitor` and `Protocol` / `@protocol` (the clash with `typing.Protocol` was accepted on 2026-09-30), and `MonitorGroup` / `ProtocolGroup`.
- The shipped protocols: `observe_only`, `concurrent`, `sequential`, `threshold`, `human`.
- The report and decision types: `Observation`, `Decision` and its constructors (`proceed`, `reject`, `terminate`, `escalate`), `Action`, `Suspicion`, `Reported`, `Decisions`, `Observations`, `Reports`, `decide_final` (its own revisit is under "Then: design decisions" in pr-series.md "Deferred").
- The step and context types: `Step`, `BeforeToolCall`, `AfterToolCall` (and the generate stages if they ship), `Context` and its fields, `Host`, `HostContext`, `Recorder`, `HumanAnswer`.
- The log contract in inspect_ai: `SentinelEvent` and its fields, `SentinelConfig` / `SentinelEntry`, and the stage names recorded in logs.
- What is public and what is the integration contract (`inspect_sentinel._integration`).

Then land, in order:

1. The open sentinel PRs, and any renames the review decides.
2. ts-mono `feature/sentinel` (#716) into ts-mono `main`.
3. inspect_ai #5514: point the submodule at the merged ts-mono commit and rebuild the viewer (the land-ts-mono flow), un-dark-launch it (the `--sentinel` option visible, the experimental notes on the API revisited, the sentinel docs pages and a CHANGELOG entry), and merge.
4. Release inspect_ai with the sentinel hooks, so sentinel can depend on a released version.
5. Release inspect_sentinel: replace the git dependency on inspect_ai with a version floor (release-pin-deps.yml does this on the Release Please PR, and release-dep-guard.yml blocks a git ref), unpause Release Please, publish to PyPI, and publish the docs.
6. In inspect_ai, replace the temporary pinned `--no-deps` CI install of sentinel with the released package.

Touches both repositories and ts-mono; every rename it decides touches the log contract, so it needs the compatibility and round-trip coverage AGENTS.md asks for before the first release fixes it.

## 6. Completing the docs

**Priority:** high. **Design:** the outline pages in `docs/`.

Raised by the maintainer on 2026-10-05: finish the user documentation before the first release. Today Getting Started (`docs/index.qmd`) is written, and the other pages are outlines with a few real sections: `monitors.qmd` (prompt helpers are planned there), `protocols.qmd` (Helpers for Rules), `composition.qmd`, `final-decisions.qmd`, `approval.qmd`, `configuration.qmd`, `state.qmd`, `host.qmd`, `transcript.qmd`, `calibration.qmd` and `deployment.qmd`.

Scope:

- Write each outline page, using inspect_ai's docs as the model for structure and tone, with examples that teach one idea each.
- The reference (`docs/reference/`) complete for the public surface that workstream 5 settles.
- The inspect_ai side: the sentinel pages and the `Task`/`eval` docs that are hidden while #5514 is dark-launched, written for when workstream 5 un-dark-launches it.
- An examples index that runs end to end, and a check in CI that the documented examples still run.

Depends on workstream 5's naming review, so it follows it page by page rather than waiting for all of it.

## 7. Shipped protocols and helpers

**Priority:** next. **Design:** [sentinel.md](sentinel.md), the shipped protocols, views and "Failure semantics".

Many small, independent tasks, good for onboarding:

- `resample`. `defer_to_trusted` was removed from the plan by the maintainer on 2026-10-02: an eval has no trusted model to defer to.
- Views: helpers that give a monitor author views of the messages and the step, such as the last six messages (maintainer, 2026-10-02).
- A helper that renders a call's arguments as text, and similar conveniences that make rules shorter (maintainer, 2026-10-02). The first set is done (`call_text`, `tool_matches`, `find_words`, `path_matches`, `path_resolves`, `result_text` and step builders; [pr-series.md](pr-series.md), "Helpers for rules"); `urls_in`, `host_matches` and `secrets_in` follow.
- The prompt helpers, such as `monitor_prompt`, unless they move to workstream 3.

## 8. Auto-mode approvers

**Priority:** next.

Support approvers in inspect modelled on the auto modes of Claude Code and Codex, in which a model decides whether each tool call may run without asking a person. Scope to establish first: how each auto mode decides (its inputs, policy and outputs) and what of that can be reproduced or reused. Then offer it in Inspect both as an `@approver`, for `Task(approval=)` users, and as a sentinel protocol, so it composes with monitors and `threshold`.

Open questions: which model judges by default, and how its policy is configured; whether the result is a binary allow/deny or a score that `threshold` calibrates; how it maps to approval's vocabulary (`approve`) and sentinel's (`continue`); and, where an existing product's prompt or policy is reused, whether its terms allow that.

## 9. Bridged agents and deployment

**Priority:** next. **Design:** [sentinel-deployment.md](sentinel-deployment.md).

**Known gap today:** a sentinel never runs for a bridged agent's tool calls; only tool calls that go through `execute_tools` are checked. First step: the before-call hook in `bridge_generate`, next to `apply_bridge_tool_approval`, so bridged agents get sentinel checks on their tool calls. With it, verify that handoff and bridged agents open agent spans consistently, so conversation ids link as designed, with tests. The proxy deployment (a sentinel at the network boundary, the host ABI, sidecar and WASM modes) is a later, larger project.

## 10. Building good, validated monitors

**Priority:** next. **Design:** [sentinel-development.md](sentinel-development.md).

The development loop: build a monitor, measure it over transcripts, calibrate it, deploy it. Reframed by the maintainer on 2026-10-02 around that goal rather than around Scout: the loop may be model-driven (supporting Claude Code iterating on a monitor) and may offer verbs such as `calibrate()`, using Scout under the hood where it makes scanning efficient. It is related to building good model judges. Scope:

- Reconstruct steps from messages and from events.
- An adapter that runs a monitor as a Scout scanner.
- A read-mode scanner over recorded `SentinelEvent`s.
- `calibrate()`, producing per-dimension thresholds in the key format `threshold` accepts (`"instance.dimension"`, `"dimension"`, `"*"`).
- `Result.subject` and row expansion in Scout.

Lives mostly in inspect_sentinel, with inspect_scout where Scout does the scanning, and consumes the sentinel API without changing it, so it rarely collides with core work. Uses `references` on reports (inspect_ai's `scorer.Reference`, as Scout's `Result` does) and the decorators' `version=`, which calibration records.

## 11. Portability check (`portable=True`)

**Priority:** next. **Design:** [sentinel-deployment.md](sentinel-deployment.md), "Keeping monitors portable", "Where the protocol runs" and "What gets bundled".

Raised by the maintainer on 2026-10-02 as an AST-based linter. Decided by the maintainer on 2026-10-05: it is a runtime check, triggered by `portable=True`, the default on `@monitor` and `@protocol`, not a separate lint step a user runs. `@monitor(portable=False)` and `@protocol(portable=False)` opt out, visibly in code review and the registry. Scope:

- **The check.** When a portable monitor or protocol is registered or configured, sentinel checks the import closure of its defining module against the allowlist of bundleable dependencies (the dependency half, per module), and that its body does not reach the ambient escapes: `get_model()`, `sandbox()`, inspect_ai's `store()` and similar (the affordance half, per function). A failure is an error naming the function, the offending import or call, and `portable=False`.
- **The restricted host.** While it runs, a portable function runs through the same restricted host a proxy would use ([sentinel-deployment.md](sentinel-deployment.md), "Keeping monitors portable", mitigation 1), so a banned call raises at runtime too, in every eval. The property is exercised continuously rather than asserted.
- **Compositions.** A composition is as portable as its least portable member: a leaf monitor calling `get_model()` disqualifies the protocol that wraps it.
- **Per function and per module.** The flag is per function, but the dependency half is per module, so a `portable=False` function does not excuse its module: the module cannot be bundled, and the portable functions beside it go with it. The check says so rather than let it be discovered at build time.
- **CI and deployment.** Enumerating the portable set and building it (the WASM build as a CI target, mitigation 3, and the proxy bundle in [proxy-host.md](proxy-host.md), section 5) consume the same check. There is no separate linter.

Adds `portable=` to the decorators and otherwise consumes them and the module layout without changing them; the proxy work in workstream 4 decides what counts as bundleable.

Open questions:

- Where each half runs: at decoration, when the factory is called (configuration), or on first run. The design does not settle it.
- What the check reports when the per-function flag and the per-module verdict disagree ([sentinel-deployment.md](sentinel-deployment.md), open question 9).
- Whether it also checks for captured state, which the 2026-10-02 framing included.

## 12. Monitoring the monitors

**Priority:** lower. **Owner:** Pete.

Raised by the maintainer on 2026-10-02 and not well defined yet: reporting on what sentinels did across tasks and evals. Ideas so far:

- Append terminations to a shared file (for example in S3).
- A job that reads many logs and summarises each protocol's behaviour.
- A sentinel-activity JSON file inside the `.eval` file.

Reads `SentinelEvent`s and the recorded configuration, so it overlaps the read-mode scanner in workstream 10.

## 13. Remote human surfaces

**Priority:** lower.

Let a person decide from outside the eval process: a request with the call, the escalations and the choices, and the answer flowing back to the waiting sample. Decided by the maintainer on 2026-10-02: put human requests on a message queue that other tools consume (Slack, a review app) rather than building the interface ourselves. Inspect's human surfaces today (the panel, ACP and the console) all assume someone at the eval's terminal or client; long-running and remote evals have nobody there. `notify()` already covers telling someone a decision is waiting (see `sequential()` and `human()`, under "Done"); this is answering it.

Open questions: how the reply reaches the sample (a callback endpoint, polling, a relay service); how long a sample waits and what happens when nobody answers (proceed, reject, or the escalate default from pr-series.md "Deferred"); who may answer and how that is authenticated and recorded in the log; and whether the same surface serves inspect's existing human approver as well as `human()`.

## Done: `sequential()` and `human()`

**Status:** done. **Design:** [sentinel.md](sentinel.md), "Humans in the loop" and the ordered composition.

The case they serve: a rule that escalates to a person, whose answer decides for the chain. `human()` reuses inspect's human approval surfaces (the approval panel, ACP and the console) and shows the escalations that led to it. Like the human approver, it calls inspect's `notify()` (Apprise, so Slack, email and the rest) before prompting, so the person learns a decision is waiting. `sequential()` is the ordered composition (formerly `chain`), passing escalations from one link to the next.

Moderate in size, but it touches runner semantics (escalation hand-off, `decide_final()`, cancellation), so it needs close review.
