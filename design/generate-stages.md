# Generate stages

Design for `BeforeGenerate` and `AfterGenerate`: which model calls they check, where the hook sits in inspect_ai, what each action means at a generate, what the step carries, and how the events link to the `ModelEvent`. It extends [sentinel.md](sentinel.md), "The payloads" and "One vocabulary across stages", and answers the deferred items in [pr-series.md](pr-series.md) that wait on the generate stages.

Status: prototyped. Monitors observe both generate stages and protocols may return `continue` or `terminate` there; every other action is a shape-check error. The rest of this document is the design for review; [Decisions for the maintainer](#decisions) lists what needs a ruling, each with a recommendation.

## Decisions for the maintainer {#decisions}

| # | Decision | Recommendation |
|---|---|---|
| 1 | How an agent generate is identified | A call on the sample's active model, made while the sample runs its solvers, outside a sentinel run, a compaction and a tool's own body ([a](#a-which-calls)). Prototyped. |
| 2 | Sub-agents and role models that are not the active model | Not checked for now. Add a declared opt-in (model roles whose calls count as the agent's) when a control setup needs one. |
| 3 | Compaction summary calls | Not checked. The summary reaches the next `BeforeGenerate` as part of `input`. Prototyped. |
| 4 | Where the hook sits | In `Model.generate`, once per call, outside the connection slot; the input preparation moves ahead of the slot so the step sees what is sent ([b](#b-where-the-hook-sits)). Prototyped. |
| 5 | Cache hits | Both stages run on a cache hit. Prototyped. |
| 6 | A generate that raises | No `AfterGenerate` (provider error, `fail_on_refusal`, a limit). Prototyped. |
| 7 | Dispatching a stage nothing watches | Skip it, at every stage, so a tool-only sentinel opens no span per generate. Prototyped. |
| 8 | `reject` before a generate | Not legal in v1; use `modify` or `terminate`. If kept, define it as a sentinel-authored refusal ([c](#c-before-generate-decisions)). |
| 9 | `modify` before a generate | Replace `input`, narrow `tools` to a subset by name, change `tool_choice`; not `model` or `config`. |
| 10 | `reject` after a generate | Replay: append the rejected message and the `message` as a user turn, regenerate, bounded at three; then terminate ([d](#d-after-generate-decisions)). |
| 11 | `modify` after a generate | Replace `choices[0].message` with a `ChatMessageAssistant`; usage, time and model unchanged; other choices dropped. |
| 12 | `Decision.modify(step, ...)` | One constructor, overloaded on the step type: `arguments=` at `BeforeToolCall`, `input=`/`tools=`/`tool_choice=` at `BeforeGenerate`, `message=` at `AfterGenerate`. |
| 13 | `SentinelEvent.modified` at the generate stages | `ChatMessageAssistant` joins the union for `AfterGenerate`; a `BeforeGenerate` modify records no copy, since the `ModelEvent` holds the request sent. A log contract change, for when `modify` lands. |
| 14 | `Host.generate` and `tool_choice` | Add `tool_choice=`, so `resample` regenerates exactly. |
| 15 | `history` at the generate stages | The caller's messages for now; the agent loops declare their full history later ([e](#e-what-the-step-carries)). |
| 16 | `escalate` at the generate stages | Enable with the next step, with the same root behaviour as the tool stages. |
| 17 | Recording `step.input` at generate time | Follow-up: the after-generate hook records each input by its assistant message id, replacing the transcript scan the tool stages use. |

## a. Which calls are checked {#a-which-calls}

Decided before this design: the generate stages check the agent's own model calls, and not the sentinel's own `Host.generate` calls, monitor or protocol model calls, scorers, or model-graded tools. The question is how to identify those calls robustly. Inspect has no marker for "an agent turn": `react`, `basic_agent`, the `generate()` solver, `generate_loop`, the bridges and custom solvers all call `Model.generate` directly.

### Options

- **A. The active model, with exclusions.** Hook `Model.generate`; a call is the agent's when the model is the sample's active model (`self is active_model()`, the test `Model.generate` already uses to merge the task's generate config and update the message count) and the call is not inside a sentinel run, a scorer, a compaction or a tool's own body. One hook covers every loop, custom solvers included.
- **B. Agent loops opt in.** `react`, `generate()`, `generate_loop` and `bridge_generate` call a sentinel-aware generate. Exact about turns, but a custom agent or solver calling `model.generate` is never checked unless it adopts a new public API, and every loop must remember to.
- **C. Any generate in an agent context.** Hook `Model.generate` and check every call made in the solver phase outside a tool body, whatever the model. Covers sub-agents and role models, but also a solver's helper calls to a classifier or a critique model, which are not agent actions.

**Recommendation: A.** It matches the decided scope, needs no new public API, and covers custom agents. Its gaps (sub-agents and role models on another model) are the ones C would close at the cost of checking calls that are not the agent's, so they wait for a declared opt-in (decision 2).

### The rule, as prototyped

A generate is the agent's when all of these hold:

1. a sentinel is configured for the task and some function in it watches a generate stage;
2. the sample is running its solvers: inspect_ai sets a context variable around `plan(state, generate)`, so init and scoring (the `scorers` span, and `inspect score`) are outside. A task's `setup` solvers are part of the plan and are inside; `score()` called from a solver runs its scorers outside;
3. the model is the active model;
4. the call is not made inside a sentinel run: the dispatcher sets a context variable around `run_root`, which covers `Host.generate` and a monitor that calls `get_model()` itself;
5. the call is not a compaction summary: the summary strategy sets the same variable;
6. the call is not made from a tool's own body: `call_tool` records the current agent span when a tool starts, and a generate in that same agent span is the tool's. An agent started inside the tool (`as_tool()`) opens its own agent span, so its calls are the agent's again.

### What is in and out

| Caller | Checked | Why |
|---|---|---|
| `generate()` solver, `generate_loop` | yes | active model |
| `react()`, `basic_agent()` on the default model | yes | active model |
| `handoff()` to an agent on the default model | yes | active model; its steps carry the sub-agent's conversation id |
| `as_tool()` agent on the default model | yes | its agent span differs from the tool's |
| bridged agents (`agent_bridge()`, `sandbox_agent_bridge()`) on the `inspect` model | yes, untested | `bridge_generate` calls `model.generate` on the model `resolve_inspect_model` returns, `get_model()` for `inspect`; a model alias or another name resolves elsewhere |
| a custom solver or agent calling `get_model().generate()` | yes | active model; includes helper calls such as `self_critique()` with no critique model |
| a sub-agent or handoff with an explicit other model | no | rule 3; decision 2 |
| an agent whose model is a role (`get_model(role="untrusted")`) | no | rule 3, as the role resolves to another instance; decision 2 |
| a model-graded tool calling `get_model()` | no | rule 6 |
| scorers, `model_graded_qa()`, including `score()` called from a solver | no | rule 2 |
| a task's `setup` solvers calling the active model | yes | they are part of the plan |
| `Host.generate`, monitor model calls | no | rule 4 |
| compaction summaries | no | rule 5; decision 3 |
| a bridge filter that returns a `ModelOutput` itself | no | no `Model.generate` call; a gap to close with the bridge hook (workstream 5) |
| MCP sampling (`get_model().generate()` from an MCP server callback) | depends | runs in the MCP session's context; unverified |

Identity has one fragility: a sub-agent given the agent model's *name* rather than `None` is checked only if `get_model()` returns the memoized active instance, which depends on the name, role, config, base URL and model args matching how the eval created it. Comparing by qualified name and an unset role would be more predictable; the prototype keeps identity, the existing precedent.

## b. Where the hook sits {#b-where-the-hook-sits}

### Options

1. **Per attempt, inside `Model._generate`**, where `Hooks.on_before_model_generate` fires. Sees the exact request, but runs once per retry, and runs inside the connection slot: a monitor on the same model (the fallback when no `monitor` role is set) waits for a slot the agent holds, which deadlocks at `max_connections=1`, and `human()` would hold a slot while a person decides.
2. **Once per call, in `Model.generate`, outside the slot.** `BeforeGenerate` runs after the input is prepared and before the slot is acquired (so before the cache lookup); `AfterGenerate` runs after the slot is released, when the output is complete and about to be returned.
3. **In the agent loops** (option B above).

**Recommendation: 2.** The input preparation that was at the top of `_generate` (resolving tools and `tool_choice`, reasoning history, tool `model_input` handlers, media extraction, merging consecutive messages) moves into `Model._prepare_input`, called before the slot, so `step.input` is what is sent. Only `Hooks.on_before_model_generate`, which runs per attempt and may mutate the request, can still change it afterwards. The move is a behaviour change for every caller, sentinel or not: tool resolution (an MCP server's tool listing, for example) no longer holds a connection slot, so `max_connections` no longer bounds it, and `ModelEvent.timestamp` and the fallback `working_time` start after preparation rather than before it.

The step holds copies of the request's message and tool lists and of the config, so a monitor that mutates them cannot change what is sent; `history` is the caller's own list, as at the tool stages, and monitors must not mutate it.

### Interactions

- **Retries.** Provider retries are inside one call and are not checked again. A loop's own retries (`react`'s `retry_refusals`, the bridge's refusal and rejection retries) are new calls, and each is checked; a retried request produces a second `BeforeGenerate` behind the same last message, whose step id takes an ordinal suffix ([f](#f-events-and-the-viewer)).
- **Cache.** `BeforeGenerate` runs before the lookup and `AfterGenerate` after a hit: a cached output is still the agent's action in this sample. A monitor's own calls inherit the eval's `cache` setting and may hit the cache too.
- **Batch.** Nothing special: a slow `BeforeGenerate` delays submission to the batch.
- **Streaming.** `on_stream` deltas reach the caller before `AfterGenerate` decides. Streaming is display-only, so this does not matter for `continue` and `terminate`. When `modify` or `reject` lands, the dispatcher must send a `StreamRetryEvent`-style boundary so a display discards the replaced deltas.
- **`fallback_models`.** Handled inside the provider call, so one call is checked once; `step.model` is the model requested, and `output.model` and `output.fallback` say which answered.
- **`cache_prompt`.** Unaffected by observation. A future `BeforeGenerate` modify that edits an early message invalidates the provider's cached prefix, which costs tokens but is not wrong.
- **Limits.** The token and turn limits are suspended inside a sentinel run, as at the tool stages. The message limit is checked before `BeforeGenerate`, so a generate the limit refuses is not checked. Time and working limits keep running, so a slow monitor counts against the sample's time. A turn or token limit raised inside the call means no `AfterGenerate`, since the output never reaches the agent.
- **Monitor failures.** As at the tool stages ([sentinel.md](sentinel.md), "Failure semantics"): a monitor that raises at a generate stage is recorded as failed, with an `error` `SentinelEvent` in the stage's span, and the generate goes ahead unless a protocol reading its observations without checking `failed` raises `MonitorFailedError` and so fails the sample.
- **`fail_on_refusal`.** The `ModelRefusalError` is raised before `AfterGenerate` runs, so a refusal configured to fail produces no after step (decision 6). A refusal returned as an output is checked like any output.
- **Failures.** A monitor or protocol that raises fails the sample with its own error. The dispatcher wraps it so that, inside an `as_tool()` agent, the tool call does not turn it into a tool error the model sees; the tool call unwraps it, as at the tool stages, and so does the sample's solver phase for a generate outside any tool.
- **Spans.** Each stage runs in its own `sentinel` span: the before span precedes the call's `ModelEvent`, the after span follows it, and monitor model calls sit inside their span. A stage that nothing in the configured tree watches opens no span (decision 7). Under a bridge with a `ModelEventSink`, the `ModelEvent` goes to the sink while the `SentinelEvent`s go to the transcript, so their order in the log depends on the sink; unverified.

### Dispatching only watched stages

Without decision 7, a sentinel of tool-stage monitors under `concurrent` or `observe` (both annotate `Step`) would open two empty spans per generate. The decorators therefore record, on each configured instance, the stages it or any instance passed to its factory watches. A composition that only forwards the step (`concurrent`, `sequential`, `observe`) contributes its children's stages and not its own `Step`; any other function contributes the stages it accepts, so a user's protocol annotated `Step` still runs everywhere. A forwarding composition whose children were not found among its arguments counts every stage it accepts, so the analysis can over-dispatch but never under-dispatch. The host asks `watched_stages(root)` before building a step. Applied to the tool stages too, it removes the empty after-call span a before-call-only sentinel opened before.

## c. Before-generate decisions {#c-before-generate-decisions}

- **`continue`**: the request is sent unchanged.
- **`terminate`**: the request is not sent and the sample ends with an operator limit, as at the tool stages. No `ModelEvent` is recorded. Prototyped.
- **`escalate`**: passed to the next decider; at the root it proceeds and warns, as at the tool stages (decision 16).
- **`modify`**: the protocol replaces the request for this call only. What may change:
  - `input`, freely: redact a secret, annotate ("you have used 8 of 10 commands"), drop a turn. The change is not written back to the agent's history, so a persistent annotation is re-added at every generate.
  - `tools`, narrowed to a subset by name. Adding a tool the loop cannot execute produces a call that fails with "tool not found".
  - `tool_choice`, e.g. `"none"` to force a text answer.
  - not `model`: answering from another model is an `AfterGenerate` `modify` with that model's message.
  - not `config`: a protocol that needs other settings generates through `Host.generate` and substitutes at `AfterGenerate`.

  The replacement is applied to the prepared input, the list `step.input` showed, and the `ModelEvent` records it. The tool-stage policy argument for `AfterToolCall` (#5355: withholding a result makes the model's picture of the sandbox diverge from the sandbox) applies to a `modify` that removes or rewrites a tool result in `input`; the docs should say so rather than the shape check forbid it, since redacting a leaked secret from a result is a legitimate use.

- **`reject`**: the existing table has it legal, meaning "do not run this generate, and tell the agent why". There is no channel to tell the agent anything without producing an output, since the caller is waiting for a `ModelOutput`. The options:
  1. **Not legal** in v1; the use cases are `modify` (strip what should not be sent) and `terminate`. Recommended.
  2. **A sentinel-authored refusal**: return a `ModelOutput` with `stop_reason="content_filter"` and an assistant message carrying `message`, with no usage. Every loop already handles a refusal (`react` retries refusals and stops after three in a row; `fail_on_refusal` raises), and a proxy can return a refusal-shaped response. Its costs: the agent's history gains an assistant message the model did not write, which needs marking as sentinel-authored in its metadata, and a log reader cannot tell it from a provider refusal without that mark.
  3. **An exception** the loop must handle. No loop handles it, so it is `terminate` with a different name.

## d. After-generate decisions {#d-after-generate-decisions}

- **`continue`**: the output is returned unchanged.
- **`terminate`**: the output is discarded and the sample ends with an operator limit; the `ModelEvent` stays in the log, and the assistant message never reaches the agent's messages, so its tool calls do not run. Prototyped.
- **`escalate`**: as elsewhere (decision 16).
- **`reject`** ("discard this completion"). The options:
  1. **Replay**: append the rejected assistant message and a user message carrying `message` (or the default "not approved" text) to this request, regenerate, and check the new output; bounded at three consecutive rejections, after which the sample terminates. This is what `_bridge/_approval.py` does for a rejected tool call in a bridged agent, and what a proxy can do. The agent's loop receives only the accepted output; the rejected exchange is in the log but not in the agent's history, as in the bridge. Each regenerate is a new call with its own two stages and `ModelEvent`. Recommended.
  2. **Regenerate silently**: the same request again. The model learns nothing and may produce the same output; `resample` already covers drawing again on purpose.
  3. **A synthetic refusal**, as for `BeforeGenerate` option 2.
- **`modify`**: `modified` is a `ChatMessageAssistant` replacing `choices[0].message`, per [sentinel.md](sentinel.md), "One vocabulary across stages". Constraints:
  - `usage`, `time` and `model` describe the call that happened and are kept; the replacement's own cost is recorded by the call that produced it (a `Host.generate` under its role).
  - Only `choices[0]` is substitutable; the other choices are dropped, so no call the monitors did not see can run.
  - `stop_reason` is recomputed: `tool_calls` when the replacement has tool calls, `stop` otherwise.
  - The replacement keeps its own message id; the step id stays the original's, and `SentinelEvent.modified` carries the replacement, so both are in the log.
  - Its tool calls run through `BeforeToolCall` like any other, so a protocol that substituted a call sees it again there; tool call ids must be unique within the message.

`resample` is an `AfterGenerate` protocol over these. It draws with `model=step.model` in a loop, scores each candidate through `run_monitors(monitors, context, replace(step, output=candidate))`, and returns `Decision.modify(step, message=best.message)` or `escalate`. Its draws run inside the sentinel run, so they are not themselves checked (rule 4), are not charged to the agent's token limit, and are recorded under the agent model's usage until the `protocol` usage bucket exists (sentinel.md open question 8). It needs `tool_choice` on `Host.generate` (decision 14).

## e. What the step carries {#e-what-the-step-carries}

| Field | `BeforeGenerate` | `AfterGenerate` |
|---|---|---|
| `model` | the requested model, `provider/name` | same |
| `conversation` | the current agent span id, or the sample uuid outside any agent span | same |
| `input` | the prepared request: the caller's messages with the configured system message, resolved reasoning history, tool `model_input` handlers, extracted media and merged consecutive messages; what the `ModelEvent` records | the same list |
| `history` | the messages the caller passed | same |
| `tools` | `ToolInfo`s sent, after `tool_choice` filtering | same |
| `tool_choice` | as sent (`"none"` when there are no tools) | same |
| `config` | the resolved `GenerateConfig`: the task's, the model's and the call's, with `max_tokens` filled | same |
| `output` | — | the `ModelOutput` returned; `output.message` is the subject |
| `escalations` | inside a `sequential` | same |

`AfterGenerate` repeats the request so that `resample` can regenerate with exactly what the agent was given.

**`history`.** `Model.generate` sees only the messages it is passed. For the `generate()` solver and for `react` without compaction those are the agent's whole conversation; with compaction they are the compacted conversation, and after `react`'s overflow recovery replaces `state.messages`, the folded turns are gone from both (pr-series.md, "react overflow recovery rewriting history"). The full history needs the loop that owns the messages to declare them: a context the loops set around their generate, holding their message list, which the dispatcher reads (decision 15). The same declaration is where a loop would mint `conversation`, which sentinel.md open question 22 leaves open.

**`step.input` at the tool stages.** The tool stages rebuild the request that produced a call by scanning the transcript for the `ModelEvent`. With the generate hook in place, the after-generate dispatch can record each input keyed by its assistant message id, scoped to the sample and bounded, whenever the call is an agent call and the sentinel watches a tool stage; the scan remains for a message no agent generate produced (decision 17, pr-series.md "Record step.input instead of rebuilding it").

**Compaction.** A summary compaction's text reaches the next `BeforeGenerate` in `input`, so a monitor sees what the agent's model wrote about its own trajectory. `compaction_summary(step)` stays deferred on inspect_ai documenting the summary message metadata key.

## f. Events and the viewer {#f-events-and-the-viewer}

`SentinelEvent.stage` already declares `model_input` and `model_output` next to `tool_call` and `tool_result`; the prototype emits them, so no event schema or generated type changes. The stage names are the ones `human(stages=)` would take.

`step_id` follows sentinel-development.md, "Step ids":

- `model_input`: the id of the last message in `input`, the one that triggered the generate. A repeat behind the same message in one sample takes an ordinal suffix, `{message_id}:2`, counted per sample by the dispatcher.
- `model_output`: the id of `output.message`, or empty for an output with no choices. A cached output carries the message id it was stored with, so two cache hits on the same request in one sample give two `model_output` events with one `step_id`; suffixing those as for `model_input` is open.

Linking to the `ModelEvent` needs no new field: a `model_output` event's `step_id` is the `ModelEvent`'s `output.message.id`, and a `model_input` event's `step_id`, without its suffix, is the id of the last message of the next `ModelEvent`'s `input`. Event uuids cannot be used, since a `BeforeGenerate` runs before its `ModelEvent` exists. In Inspect View the before span sits directly above the `ModelEvent` and the after span directly below it, so a viewer that renders the sentinel span inline needs no lookup; one that wants a "checked by" badge on the `ModelEvent` matches by message id. Whether ts-mono renders the two new stage values well is unverified. A monitor's own model calls sit in its stage's sentinel span, listed together as at the tool stages; tying each to its instance stays deferred (pr-series.md, "Monitor model calls tied to their row").

Later contract changes, none in the prototype: `SentinelEvent.modified` gains `ChatMessageAssistant` for an `AfterGenerate` modify, and a `BeforeGenerate` modify leaves `modified` unset with the validator relaxed for that stage, since the `ModelEvent` records the request sent (decision 13). If decision 8 adopts the synthetic refusal, the assistant message it produces needs a documented metadata mark.

## Prototype scope

Built:

- `BeforeGenerate` and `AfterGenerate` in `inspect_sentinel`, exported, in `Step`, `Monitor` and `Protocol`; a function annotated with either is registered for that stage.
- The shape check allows `continue` and `terminate` at the generate stages and raises "only 'continue' and 'terminate' are supported at the generate stages" for the rest.
- `watched_stages()` in the integration contract, and the forwarding mark on the three compositions.
- In inspect_ai: `Model._prepare_input`, the two hooks in `Model.generate`, the identification rule, the per-stage events and step ids, and the skip for unwatched stages.

Not built: every action other than `continue` and `terminate`, `Decision.modify`, `Host.generate(tool_choice=)`, the full `history`, recording `step.input`, bridged-agent verification, and `human()` at the generate stages.
