# Workstreams

Areas of sentinel that can be owned separately, as of 2026-10-05. Each names its design document, its scope, and what it touches, so two people can work in parallel without colliding. Sections are in priority order, decided by the maintainer on 2026-10-02: high priority (1 to 6), next (7 to 11), lower priority (12 to 14), then done; 5 and 6 were added on 2026-10-05. Smaller agreed items that wait on something are in [pr-series.md](pr-series.md), "Deferred".

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
- A helper that renders a call's arguments as text, and similar conveniences that make rules shorter (maintainer, 2026-10-02). The first set is done (`call_text`, `tool_matches`, `find_words` and `result_text`; [pr-series.md](pr-series.md), "Helpers for rules"); `urls_in`, `host_matches` and `secrets_in` follow. The path helpers are workstream 14.
- The prompt helpers, such as `monitor_prompt`, unless they move to workstream 3.

## 8. Auto-mode approvers

**Priority:** next.

Support approvers in inspect modelled on the auto modes of Claude Code and Codex, in which a model decides whether each tool call may run without asking a person. Scope to establish first: how each auto mode decides (its inputs, policy and outputs) and what of that can be reproduced or reused. Then offer it in Inspect both as an `@approver`, for `Task(approval=)` users, and as a sentinel protocol, so it composes with monitors and `threshold`.

Open questions: which model judges by default, and how its policy is configured; whether the result is a binary allow/deny or a score that `threshold` calibrates; how it maps to approval's vocabulary (`approve`) and sentinel's (`continue`); and, where an existing product's prompt or policy is reused, whether its terms allow that.

## 9. Bridged agents and deployment

**Priority:** next. **Design:** [sentinel-deployment.md](sentinel-deployment.md).

**Known gap today:** a sentinel never runs for a bridged agent's tool calls; only tool calls that go through `execute_tools` are checked. First step: the before-call hook in `bridge_generate`, next to `apply_bridge_tool_approval`, so bridged agents get sentinel checks on their tool calls. With it, verify that handoff and bridged agents open agent spans consistently, so conversation ids link as designed, with tests. Tool names differ across agents: inspect_ai's built-in tools are lowercase (`bash`, `text_editor`), Claude Code's capitalised and named differently (`Bash`, `Read`, `Edit`), Codex's different again (`shell`), so a rule written for one agent's tools silently does not fire on another's; for a deny rule that fails open. Decide how rules see a bridged agent's tools, for example a documented mapping or a normalised tool kind on the step (raised by the maintainer on 2026-10-05; `tool_matches(case_sensitive=False)` covers case only). The proxy deployment (a sentinel at the network boundary, the host ABI, sidecar and WASM modes) is a later, larger project.

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

**Priority:** next. **Status:** the check is implemented ([pr-series.md](pr-series.md), "Portable monitors and protocols are checked when configured"); `fetch` remains. **Design:** [sentinel-deployment.md](sentinel-deployment.md), "Keeping monitors portable", "Where the protocol runs" and "What gets bundled".

Raised by the maintainer on 2026-10-02 as an AST-based linter. Decided by the maintainer on 2026-10-05: it is a runtime check, triggered by `portable=True`, the default on `@monitor` and `@protocol`, not a separate lint step a user runs. `@monitor(portable=False)` and `@protocol(portable=False)` opt out, visibly in code review and the registry. Scope as first raised; the decisions below supersede it where they differ (no runtime guard, per function by reference, the per-module verdict at deploy time):

- **The check.** When a portable monitor or protocol is registered or configured, sentinel checks the import closure of its defining module against the allowlist of bundleable dependencies (the dependency half, per module), and that its body does not reach the ambient escapes: `get_model()`, `sandbox()`, inspect_ai's `store()` and similar (the affordance half, per function). A failure is an error naming the function, the offending import or call, and `portable=False`.
- **The restricted host.** While it runs, a portable function runs through the same restricted host a proxy would use ([sentinel-deployment.md](sentinel-deployment.md), "Keeping monitors portable", mitigation 1), so a banned call raises at runtime too, in every eval. The property is exercised continuously rather than asserted.
- **Compositions.** A composition is as portable as its least portable member: a leaf monitor calling `get_model()` disqualifies the protocol that wraps it.
- **Per function and per module.** The flag is per function, but the dependency half is per module, so a `portable=False` function does not excuse its module: the module cannot be bundled, and the portable functions beside it go with it. The check says so rather than let it be discovered at build time.
- **CI and deployment.** Enumerating the portable set and building it (the WASM build as a CI target, mitigation 3, and the proxy bundle in [proxy-host.md](proxy-host.md), section 5) consume the same check. There is no separate linter.

Adds `portable=` to the decorators and otherwise consumes them and the module layout without changing them; the proxy work in workstream 4 decides what counts as bundleable.

Decided by the maintainer on 2026-10-05:

- **What a portable function may do.** Compute anything, and affect the outside world only through `context`: `context.host.generate()`, `context.host.ask_human()`, `context.store_as()`, and `context.host.fetch()` once it exists. It reads the step, the `Context` and its factory's parameters (configuration comes through parameters, which the log records, not the environment), and may return any decision.
- **Imports allowed.** `inspect_core` and its declared dependencies (read from its package metadata, so the list cannot drift; until then one constant mirrors them); `anyio` (added in the rework below: sentinel depends on it and protocols run work concurrently); `inspect_sentinel`'s public API; and, as a stopgap until they move into `inspect_ai.core`, `StoreModel` and `Reference` by identity. Not the rest of inspect_ai, and no other third-party package (no `requests`, `openai`, `anthropic`, `numpy`): models are reached through `context.host.generate()`.
- ~~Standard-library modules not allowed: a deny list of modules with effects.~~ Replaced by the maintainer on 2026-10-05 (rework): **the standard library is an allowlist**, failing closed. About 50 modules of pure computation that CPython provides on WASI (`json`, `re`, `math`, `datetime`, `collections`, `asyncio` without its streams, subprocesses and threads, `os.path` but not `os`, and so on), in one constant with the C modules behind them. A module off the list fails with an error saying it is not on the portable list and to use `portable=False` or ask for it to be added.
- **Builtins not allowed:** `open`, `input`, `exec`, `eval`, `compile`, `__import__`, `breakpoint`, by name (rework: no identity matching of aliases, no checks on dunder attributes, frames or `getattr()` with a literal name).
- **Honest mistakes, not evasion** (rework). The check catches a monitor that reads the environment or calls `get_model()` by accident; it does not try to stop deliberate evasion. Attribute chains resolve only through modules and classes; an instance is judged by its type. The check never imports anything. The mechanism stays `ast` over source: bytecode (opcode churn across 3.10–3.14 could fail open; annotations indistinguishable before 3.14) and `symtable` (about 65 lines saved, but version handling for inlined comprehensions and `__annotate__` tables) were considered and rejected.
- **When it runs: when the factory is called** (configuration), so only instances someone configures are checked, the error comes before the eval starts and names the configuration, and importing a module stays harmless. Checks are cached per factory. Without source (the plain REPL, `exec()`'d code) the check is skipped; Jupyter notebooks are checked.
- **Each instance is checked on its own.** A monitor's or protocol's own module imports and function body; children are passed in as arguments, not imported, so they play no part in their parent's check. Deployment requires every configured instance to be portable; there is no separate composition verdict.
- ~~A portable function in a module with a disallowed import is an error.~~ Replaced by the maintainer on 2026-10-05: **the check is per function, by what it references.** It resolves the names a portable factory's code references (and follows the author's own functions and classes they reach) and judges each object by its defining module, so a `@task` or an inspect_ai import beside the monitor does not count unless the monitor uses it, and examples can keep the task beside the monitor. The per-module verdict, whether a module can be bundled given everything it imports, is a deploy-time concern of the bundler, not this check.
- **No runtime guard to start.** A runtime guard (for example an audit hook raising on effects while a portable function runs) was considered: authors are not adversaries, the allowed libraries do effectful things lazily (`datetime.strptime` importing `_strptime`, pydantic's lazy imports, locale and timezone files), so it would need its own allowlists and would raise confusing errors in ordinary evals, and in a proxy the environment (WASM, a sandboxed sidecar) enforces the same thing anyway. Revisit once the proxy host exists and real portable monitors show which effects are legitimate; a strict mode limited to tests or CI is the cheap middle ground.
- **`fetch` is in scope here.** `Host.fetch` through named endpoints, as designed in [sentinel-deployment.md](sentinel-deployment.md) ("The host ABI", "Named endpoints, not URLs"), built in sentinel and in inspect_ai's host, with endpoints configured per task or deployment. It is a portable function's only network route, so a monitor that needs an HTTP API is not portable until it exists; the proxy host implements the same method (workstream 4).
- **The shipped protocols** (`observe_only`, `concurrent`, `sequential`, `threshold`, `human`) must pass the check themselves.

Later:

- **Captured state.** A factory's closure holding a mutable object that its function mutates keeps state in process memory rather than `store_as()`: shared across samples in an eval, and divergent across instances in a proxy. Telling it apart from a closure over parameters needs more design; for now the docs say to keep state in `store_as()`.

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

## 14. Path matching helpers

**Priority:** lower. **Design:** [pr-series.md](pr-series.md), "Helpers for rules" and Deferred "Finding paths in shell and code text".

Raised by the maintainer on 2026-10-05, when the path helpers were split out of sentinel PR #49.

What exists: `path_matches(path, patterns, *, cwd=None, home=None)` and `path_resolves(path, *, cwd=None, home=None)`, built on wcmatch (`GLOBSTAR`, `DOTGLOB`, `NODOTDIR`, `FORCEUNIX`, `CASE`) and `posixpath`, with 299 tests including Hypothesis properties against `PurePosixPath.full_match`, on the branch `archive/rule-helpers-path-matching`. The earlier `paths_in`, `unresolved_paths` and shell lexer are on `archive/rule-helpers-shell-paths`.

Why it was split out: the matcher is correct for strings, but authors will read it as being about files. Probed gaps:

- `/proc/self/root/etc/passwd`, and a symlink such as `/tmp/link/passwd` with `link` pointing to `/etc`, pass a `/etc/**` deny-list while `path_resolves` is True.
- `/ETC/passwd` passes on a case-insensitive filesystem.
- A leading `//` (a network path) collapses to `/`.
- Windows forms (drive letters, backslashes, UNC paths) are escalated only because they look relative: by accident, and untested.

Scope for the PR:

- State the scope: the path as written, POSIX, in the agent's sandbox, not which file is opened. Real protection is the sandbox's (read-only mounts, permissions); a rule is a first line.
- Fail safe explicitly, with `path_resolves` False, on Windows forms, a leading `//`, and known alias prefixes such as `/proc/*/root`, `/proc/*/cwd` and `/dev/fd`, each tested.
- An option for case-insensitive matching.
- A decision on Windows sandboxes: unsupported and escalated, or a later Windows mode.
- A short threat-model section in the docs.
- One review focused on bypasses. Agent reviewers have declined bypass-hunting, so plan for a human reviewer.

## Done: `sequential()` and `human()`

**Status:** done. **Design:** [sentinel.md](sentinel.md), "Humans in the loop" and the ordered composition.

The case they serve: a rule that escalates to a person, whose answer decides for the chain. `human()` reuses inspect's human approval surfaces (the approval panel, ACP and the console) and shows the escalations that led to it. Like the human approver, it calls inspect's `notify()` (Apprise, so Slack, email and the rest) before prompting, so the person learns a decision is waiting. `sequential()` is the ordered composition (formerly `chain`), passing escalations from one link to the next.

Moderate in size, but it touches runner semantics (escalation hand-off, `decide_final()`, cancellation), so it needs close review.
