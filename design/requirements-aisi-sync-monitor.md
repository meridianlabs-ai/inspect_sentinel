# Requirements: the AISI sync monitor

External requirements from AISI (UK AI Security Institute), 2026-10-08, for moving their internal sync monitor onto Inspect Sentinel. The first section reproduces the document as received, lightly formatted. The rest is ours: what is met today, where each requirement sits in the plan, where a requirement conflicts with a recorded decision, and a suggested order.

Status: received 2026-10-08; statuses checked against inspect_sentinel `main` and inspect_ai `feature/sentinel` on that date. AISI's follow-up the same day ([PR #66](https://github.com/meridianlabs-ai/inspect_sentinel/pull/66#issuecomment-6064370831) comment, reproduced below) narrows R2, R4, R5, R6 and R7 and adds R10; the sections after it reflect the narrowed requirements. Priorities and the remaining choices are for the maintainer to decide; nothing here changes a recorded decision.

## The document

**Context.** We are refactoring our internal sync monitor to use Inspect Sentinel, but so far we have run into a few limitations that stop us doing it fully without workarounds. We have monitors that run before tool calls and monitors that run after them. The pre-tool-call monitor runs once per step and sees all of that step's tool calls at once. The post-tool-call monitor runs separately after each tool call.

### R1: escalation to human review at AfterGenerate

**What we need:** A protocol at AfterGenerate can return escalate. A monitor there sees the whole assistant turn, before any of its tool calls run. human() can review the turn at that stage. The human reviewer sees the turn's tool calls and step.escalations, and can at least approve, so the turn proceeds unchanged, or terminate.

**Why:** Our monitors judge a turn as a whole, because the combined effect of parallel calls matters. We need to be able to escalate from our monitor to human review.

### R2: exact step.input at AfterToolCall, or an explicit failure

**What we need:** At AfterToolCall, step.input is exactly what was sent to the model call that produced step.call, however long the run and even when older transcript events are no longer resident. When the exact input isn't available, the monitor can tell, rather than silently getting history in its place, and replay (R7) gives the same step.input. This may be worth having at BeforeToolCall too, but we only need it at AfterToolCall.

**Why:** Our tool-result monitors judge each result in the context the agent actually had. After a compaction that is a summary plus recent turns, not the whole history. If that record is lost we fail closed. Showing the monitor the fuller history instead may show it context the agent never had without anyone knowing, which we want to avoid.

### R3: the step's tool definitions and truncation at AfterToolCall

**What we need:** At AfterToolCall, a monitor can directly get the exact tool definitions offered to the model call that produced step.call (as AfterGenerate has in step.tools). It can also tell whether step.result was truncated, through one signal that is the same live and in replay. The tools may be worth having at BeforeToolCall too, but we only need them at AfterToolCall.

**Why:** What a tool does changes how suspicious a call to it is, so our tool-result monitors show the definitions of the tools the agent actually had for that step. At the moment the only way to get them at AfterToolCall is to scan transcript events for the producing model call. Our monitors also judge the result as the model saw it, and mark truncation in a way the agent can't forge. Live, we can work truncation out by comparing step.result with step.output, but the untruncated output isn't in the log, so replay needs another source.

### R4: a monitor can see earlier decisions in its conversation

**What we need:** A monitor can see the decisions recorded at earlier steps of its conversation, for example that the human() link approved the turn or call escalated at step X. This works at AfterGenerate (R1) and the tool stages, and in replay (R7).

**Why:** When a person approves a flagged action, our later prompts mark those calls as approved, so the monitor doesn't flag them again in isolation and the person isn't asked again and again. The approval is a decision by the human() link of a sequential, and a monitor can't see another layer's decisions; step.escalations covers only the current step.

### R5: escalate or terminate on monitor scores, without rejecting

**What we need:** A built-in protocol, whether an extension of threshold or something new, that turns monitors' suspicion into escalate at one score and/or terminate at another, without also requiring a reject score. It works at AfterGenerate (R1), BeforeToolCall and AfterToolCall, and can combine monitors that watch different stages. Each decision's explanation includes every contributing monitor's score and explanation.

**Why:** None of our configurations rejects: a rejection tells the agent no, and the agent keeps trying variations. Ours terminate, or escalate to a person, above a score threshold. Two of our monitors act at AfterToolCall, and one configuration combines a proposed-action monitor with a tool-result monitor. The reviewer needs to see why the action was flagged, and where monitors disagree.

### R6: an unhandled escalate doesn't let the action through

**What we need:** When the root's decision is escalate, at any stage, the action does not proceed, and the log shows what happened.

**Why:** A flagged action must never run without someone deciding. At the moment a root escalate proceeds with a warning, so a misconfigured chain, such as a threshold that escalates with no human() after it, lets the action through. Inspect's approval doesn't in the same situation.

### R7: efficient replay over existing logs

**What we need:** A way to run the exact same @monitor over stored eval logs. Replayed AfterGenerate and AfterToolCall steps carry what the monitor would have seen live (the exact input, tools, and result with its truncation), including after a compaction and in logs from before Sentinel, where approvals are ApprovalEvents. Each step's Observation can be matched to the same step live and in replay, and a chosen subset of steps can be scored. Replay parallelises well and makes good use of provider prompt caching, while each step still sees the store_as state it would have had live. Live and replay stay identical as Sentinel changes.

**Why:** We need to be confident that backtesting results are valid i.e. that the monitor behaves identically and receives identical context whether it is in live or backtesting mode. Backtesting can be slow for long transcripts and we want to make sure we can run backtesting as quickly and efficiently as possible. For reference, our current implementation runs transcripts and compaction segments in parallel (optionally splitting long segments further), and steps in order within each segment so we can make use of provider prefix caching.

### R8: monitor-safe message rendering

**What we need:** The shared message renderer (Scout's messages_as_str and MessageFormatOptions, which pr-series.md plans to move into inspect_ai) meets the needs in the table below, so a monitor's view of the transcript becomes configuration rather than our own ~650 lines of rendering.

**Why:** We need secure transcript rendering that can't be tampered with / injected by the agent, and we need to support certain configuration of what is shown to the monitor from the transcript.

| | Need | Detail |
|---|---|---|
| 8.1 | Agent text can't forge structure | No message content, argument value or tool output can be mistaken for the rendering's own structure. format="json" already protects message boundaries; we also need it inside a message: tool calls and results, errors, \<thinking\> blocks and harness-set markers. |
| 8.2 | An actions-only view | Tool calls can be rendered without the assistant's prose around them, with or without reasoning. exclude_tool_usage does the opposite. |
| 8.3 | Compaction summaries | Readable summaries are shown as summaries rather than \<data /\>; unreadable ones are optionally left out. |
| 8.4 | Tool result status | Whether a result was an error, or was truncated (R3), is shown in a way the agent can't fake. |
| 8.5 | Referring to earlier calls | Tool calls can be identified, so a prompt can say which ones a person approved (R4). |
| 8.6 | Prompt caching | Per-message output, as format="list" gives now, survives the move. We put each turn in its own ContentText block so the prompt prefix stays cacheable. |
| 8.7 | Content it can't show | A monitor can tell when the transcript holds content that can't be rendered as text, such as images or unknown types, so it can fail closed. |
| 8.8 | Tool definitions | Tool definitions (ToolInfo) can be rendered. |
| 8.9 | Withheld messages | A deployment can withhold messages it has flagged from the monitor (for example a jailbreak prompt in a red-team eval), and the monitor can tell something was withheld. In general it would also be useful to have a well thought out escape hatch for message rendering so that it is possible to modify the rendered transcript (but hard to do accidentally or for the model to do) for e.g. ControlArena where we want to test the monitor by giving the agent a red team prompt that the monitor can't see. |

### R9: human() routed to a team's own review tool

Lower priority than the rest.

**What we need:** A task can route human() to its own review surface rather than inspect_ai's built-in human_approver panel, at the tool stages and at AfterGenerate (R1).

**Why:** We may want to use our own approver UI. Currently some teams do and may want to continue using their implementation, and escalated actions need to reach it.

## AISI follow-up (2026-10-08)

Received as a comment on PR #66, lightly formatted. It answers the conflicts listed below and adds one request.

**R6 (root `escalate`).** An opt-in is enough: a task-level or root-level setting that makes an unhandled `escalate` fail closed. The default can stay as recorded. `terminate` is fine as the fail-closed action at every stage; `reject` is never needed here.

**R2 (fallback to history).** Any explicit signal works: `input` being `None`, a flag on the step, or an exception. Keeping a fallback is fine too, as long as the step says it fell back. What must be avoided is a monitor silently judging context the model was never sent.

**R5 (`threshold` explanations).** Each monitor's observation is already a `SentinelEvent`, so the log is covered. This only matters for what the person reviewing an escalation sees: every contributing monitor's score and reasoning should reach `human()`, whether through the explanation, `step.escalations` or metadata.

**R4 (earlier decisions vs monitor independence).** Narrowed: only earlier decisions by protocols, in practice `human()` approvals of escalations, and only at steps before the current one. Other monitors' observations, at the same step or earlier, are not needed, so the independence property can stay as it is.

**R7 (concurrent segments vs `store_as` state).** The only state across steps is the agent's opening prompts as they were sent to the model, recorded at a conversation's first monitored step so they survive a compaction that drops them from later inputs. With R10, monitors would keep no state across steps, the conflict goes away, and step selection is just "only run these steps".

### R10: the history as sent to the model

**What we need:** A second field beside `step.history` holding every message this agent instance has sent to the model, in the order sent. It is scoped to one agent instance, so sub-agents and parallel agents in the same sample are not mixed in. `history` stays as it is. It works live and in replay. Open question: how to show a message the scaffold edits after it was sent; keeping each version in the order it was sent seems right, but any consistent answer works.

**Why:** `step.history` is the scaffold's list, which can differ from what the model was sent, for example when a scaffold edits messages before sending them. The only as-sent view is `step.input`, which covers only the current call. This gives the opening prompts and the pre-compaction context as the model actually saw them, without monitors keeping state, live and in replay, and resolves the R7 conflict. It depends on the same `conversation` id on events that R4 and R7 need; [sentinel-development.md](sentinel-development.md), "From events", already sketches rebuilding history per agent from `ModelEvent`s and `CompactionEvent`s sharing that id.

## Status today

| | Requirement | Status |
|---|---|---|
| R1 | escalate and `human()` at `AfterGenerate` | not met |
| R2 | exact `step.input` at `AfterToolCall`, or an explicit failure | partly met |
| R3 | tool definitions and a truncation signal at `AfterToolCall` | not met |
| R4 | earlier decisions in the conversation | not met |
| R5 | escalate or terminate on scores, without reject | not met |
| R6 | a root escalate does not proceed | decided 2026-10-10 (terminates); not built |
| R7 | replay over existing logs | not met; designed in part |
| R8 | monitor-safe message rendering | partly met, in Scout; the move to inspect_ai is not started |
| 8.1 | agent text cannot forge structure | partly met |
| 8.2 | actions-only view | not met |
| 8.3 | compaction summaries | not met |
| 8.4 | tool result status | not met |
| 8.5 | referring to earlier calls | partly met |
| 8.6 | per-message output for caching | met in Scout; the plan keeps it |
| 8.7 | content it cannot show | partly met |
| 8.8 | tool definitions | not met |
| 8.9 | withheld messages | partly met |
| R9 | `human()` routed to a team's review tool | not met |
| R10 | the history as sent to the model, per agent instance | not met; designed in part for replay |

**R1.** Only the tool stages are wired ([pr-series.md](pr-series.md), Decisions, "Tool stages only, for now"). The generate-stages prototype, sentinel #42 (draft), allows only `continue` and `terminate` at `BeforeGenerate` and `AfterGenerate`; its decision 16 enables `escalate` next, with the tool stages' root behaviour. `human(stages=)` accepts only `tool_call` and `tool_result` (`_protocols/human.py`; [sentinel.md](sentinel.md), "Humans in the loop", open question 14). inspect_ai's `Host.ask_human` passes one `step.call` to `human_approver` (`_sentinel/_dispatch.py`), so there is no rendering of a whole turn. Gap: `escalate` at `AfterGenerate`, `human(stages=["model_output"])`, and a host rendering of a turn's tool calls with `step.escalations`.

**R2** (narrowed: any explicit signal; a flagged fallback is acceptable). inspect_ai rebuilds `step.input` at both tool stages by scanning resident transcript events for the `ModelEvent` whose output carries the assistant message (`_dispatch.py`, `_model_input`). When none is found, for example once the event is no longer resident, it uses the conversation before the assistant message and logs a warning once per sample; nothing on the step says which happened. That fallback is `history`, which after a compaction holds turns the model was not sent. Gap: record the input when the generate finishes ([pr-series.md](pr-series.md), Deferred, "Record `step.input` instead of rebuilding it"; #42 decision 17), and add an explicit signal when the exact input is unavailable (`input` of `None`, a flag on the step, or an exception), keeping the fallback only if the step marks it. Replay: none yet (R7).

**R3.** `BeforeToolCall` and `AfterToolCall` have no `tools` field (`_step.py`); only the generate stages carry `tools` in the design ([sentinel.md](sentinel.md), "The payloads"). Truncation: `step.result` is what the model sees and `step.output` the untruncated value, live only. The log records the truncated byte range on `ToolEvent.truncated`, not on the step. The in-band notice inspect_ai writes into a truncated result ("The output of your call to … was too long to be displayed", `<START_TOOL_OUTPUT>`) is text a tool's output can imitate. Gap: `tools` at the tool stages and one truncation field set the same way live and in replay.

**R4** (narrowed: earlier protocol decisions only, in practice `human()` approvals, at steps before the current one). A monitor reads its own `store_as()` namespace, keyed by its path, and `step.escalations`, which holds only earlier links' escalations at the current step inside a `sequential` ([sentinel.md](sentinel.md), "The payloads"). `SentinelEvent`s are in the transcript with `step_id` and `conversation`, but nothing on `Context` or `Host` exposes them. [sentinel.md](sentinel.md), "The surfaces", lists sibling reports at the same step among what a monitor never sees; earlier steps are not addressed. Gap: a read of earlier protocol decisions in the conversation, keyed by step id, at every stage and in replay. Monitor observations stay out of it, so the independence property is untouched.

**R5** (narrowed: the log is covered; the scores must reach the reviewer). `threshold(monitors, reject_at, terminate_at=None)` requires `reject_at`, returns only `reject`, `terminate` or `continue`, and requires every monitor to watch `BeforeToolCall` (`_protocols/threshold.py`). Its explanation names the one highest-scoring crossing monitor ([pr-series.md](pr-series.md), "Threshold explanations"). Today this needs a custom protocol. Gap: a built-in protocol with `escalate_at` and `terminate_at`, no required `reject_at`, at `AfterGenerate` and both tool stages, combining monitors across stages, with every contributing score and explanation reaching `human()`, through the explanation, `step.escalations` or metadata.

**R6** (narrowed: an opt-in is enough; decided 2026-10-10 as the default). A root `escalate` proceeds, stays recorded as the root's decision, and warns once per process (`_dispatch.py`, `_run`; [pr-series.md](pr-series.md), "Root and escalate"). Decided 2026-10-10 ([pr-series.md](pr-series.md), "Unhandled escalate"): an escalate that reaches the root is a configuration bug, and sentinel's `run_sentinel` turns it into a terminate at every stage, recorded as unhandled; `handle_escalation("terminate" | "continue")` states a policy explicitly, and typed decisions (a follow-up) catch a provably unhandled escalate when the sentinel is built. This meets R6 by default, without the opt-in setting. Gap: building it.

**R7.** [sentinel-development.md](sentinel-development.md) designs replay: steps from messages and from events (`ModelEvent.input` and `tools`, `ToolEvent.truncated`), live step ids, a per-transcript store, compaction segments run concurrently and walked in order, and a read mode whose agreement with replay is "a test worth shipping". Built: `SentinelEvent.step_id` at the tool stages (the tool call id), so a live observation can be matched by step. Not built: any replay. Not in the design: logs from before Sentinel, whose approvals are `ApprovalEvent`s, and scoring a chosen subset of steps ("only run these steps"). The `store_as()` state across concurrent segments drops out for AISI if R10 lands, since their only cross-step state is the opening prompts. Gap: workstream 10, plus those two.

**R8.** The shared renderer is planned, not started ([pr-series.md](pr-series.md), Deferred, "Shared message rendering, then the prompt helpers"): Scout's `messages_as_str`, `message_as_str` and `MessageFormatOptions` move into inspect_ai with output exactly Scout's. What Scout does today (`inspect_scout/_scanner/extract.py`):

- **8.1.** `format="json"` delimits messages. Inside a message, text format writes tool calls as `Tool Call: name` / `Arguments:` lines with raw values, errors as `Error in tool call '…'`, and reasoning as `<thinking>…</thinking>` around raw text, all of which agent text can imitate. Sentinel #48's lessons (a fixed marker for unparsed arguments, quoted non-identifier argument names) are recorded in that Deferred entry, not built.
- **8.2.** `exclude_tool_usage` removes tool calls and results; there is no option that keeps only them.
- **8.3.** Data content renders as `<data />`; the `compaction_summary(step)` helper waits on inspect_ai documenting the `"summary"` metadata key (Deferred, Later).
- **8.4.** Errors are text as above; truncation is not shown. [sentinel.md](sentinel.md), "The helpers", plans `result_as_str(result, truncated=)`.
- **8.5.** `message_numbering()` labels messages `[M1]`; tool call ids are not rendered.
- **8.6.** `format="list"` gives one string per message. The plan requires the moved renderer's output to match Scout's.
- **8.7.** Images, audio, video, data and documents render as `<image />`-style placeholders, with no signal a monitor can test; a content type the renderer does not know is dropped.
- **8.8.** No `ToolInfo` renderer.
- **8.9.** `MessagesPreprocessor.transform` is a free-form escape hatch, which the plan keeps in Scout; there is no marker that something was withheld.

**R9.** `Host.ask_human` in inspect_ai always calls `human_approver` (`_dispatch.py`), so the approval panel, ACP or the console. Nothing routes `human()` to another surface. Workstream 13 decided remote surfaces go through a message queue that other tools consume ([workstreams.md](workstreams.md)); [proxy-host.md](proxy-host.md) maps `ask_human` to an audit queue.

**R10.** `step.history` is the scaffold's message list (`_step.py`), and `step.input` is the as-sent input of the current call only. Nothing live accumulates what one agent instance has sent across calls. [sentinel-development.md](sentinel-development.md), "From events", designs the same reconstruction for replay from `ModelEvent`s and `CompactionEvent`s sharing a `conversation` id, which core does not record yet. Gap: a step field (for example `sent`) built per agent instance from each generate's recorded input (the R2 record), the `conversation` id on events, the same field in replay, and a rule for messages edited after they were sent.

## Where each requirement sits in the plan

| | Covered by | Added | Depends on |
|---|---|---|---|
| R1 | Workstream 2 (generate stages); sentinel #42 decision 16 | Deferred, Later: `human()` at `AfterGenerate` | generate stages; R6's decision for the root behaviour |
| R2 | Deferred, "Record `step.input` instead of rebuilding it"; #42 decision 17 | on that entry: an explicit signal when the input is unavailable; a flagged fallback is allowed | generate-stage hook (#42) for recording at generate time |
| R3 | none | Deferred, tool-stage completeness: tools and a truncation signal at the tool stages | the same record as R2 |
| R4 | none | Deferred, design decisions: earlier protocol decisions (`human()` approvals) visible to a monitor | `conversation` on `ModelEvent` and `CompactionEvent`; step ids at the generate stages (#42); R1; R7 for replay |
| R5 | none | Deferred, design decisions: escalate and terminate on scores without reject, with every contributing score reaching `human()`; workstream 7 bullet | R1 for `AfterGenerate`; R6's decision for an unpaired escalate |
| R6 | pr-series.md, "Unhandled escalate" (decided 2026-10-10); Deferred, "Build the unhandled-escalate decision" | the decision | inspect_ai dispatcher change; `handle_escalation()` in sentinel |
| R7 | Workstream 10; [sentinel-development.md](sentinel-development.md) | workstream 10 note: pre-Sentinel logs, step subsets | R2 and R3 recorded in the log; `conversation` on events; inspect_core if replay moves into sentinel |
| R8 | Deferred, "Shared message rendering"; workstream 3 (views) | the uncovered sub-needs on that entry; workstream 3 note | the inspect_ai and Scout PRs in that entry; R3 for 8.4; R4 for 8.5 |
| R9 | Workstream 13 (remote human surfaces); `Host.ask_human` | workstream 13 note; Deferred, Later: routing `human()` | R1 for `AfterGenerate` |
| R10 | [sentinel-development.md](sentinel-development.md), "From events" (replay only) | Deferred, tool-stage completeness: the history as sent; workstream 2 and 10 notes | the R2 record of each generate's input; `conversation` on `ModelEvent` and `CompactionEvent` |

## Conflicts with recorded decisions

AISI's follow-up resolves or narrows each conflict listed on 2026-10-08. R6 was then decided on 2026-10-10, replacing the recorded root-escalate default; for the others what remains is a smaller choice, and no recorded decision has to change.

1. **R6 and "Root and escalate".** Decided 2026-10-10, going further than AISI's opt-in: an escalate that reaches the root is treated as a configuration bug and terminates at every stage, recorded as unhandled, replacing the recorded proceed-with-warning default. `handle_escalation("terminate" | "continue")` states a policy explicitly; typed decisions, a follow-up, catch a provably unhandled escalate when the sentinel is built ([pr-series.md](pr-series.md), "Unhandled escalate", "Typed decisions").
2. **R2 and the fallback to history.** Resolved: the fallback may stay if the step says it fell back. To decide: the form of the signal (`input` of `None`, a flag on the step, or an exception), which also decides `input`'s type.
3. **R5 and "Threshold explanations".** Resolved without changing the 2026-09-29 decision: the explanation can keep naming the highest-scoring monitor, as long as every contributing monitor's score and reasoning reaches the person reviewing an escalation. To decide: which carrier, `step.escalations` (already what `human()` shows) or metadata. `threshold` stays `BeforeToolCall`-only for `reject`; the new protocol's escalate and terminate need no reject.
4. **R4 and monitor independence.** Resolved: only earlier protocol decisions (in practice `human()` approvals) at earlier steps, never monitor observations, so [sentinel.md](sentinel.md) "The surfaces" stands. To decide: how a monitor reads them (a `Context` or `Host` read keyed by step id) and whether every protocol decision or only `human()`'s is exposed.
5. **R7 and concurrent compaction segments.** Dissolves for AISI with R10: their only cross-step state is the opening prompts as sent, which R10 provides. The general question, open question 8 of [sentinel-development.md](sentinel-development.md), stays open for other stateful monitors.
6. **R1 and `human()` asking without escalation.** Not a conflict: `human()` asks whenever reached (pr-series, "human()"), and inside a `sequential` that means after an escalation, which is what R1 needs.
7. **R10 and edited messages.** New, not a conflict: how the as-sent history shows a message the scaffold edits after sending it. AISI suggests keeping each version in the order sent.

## Suggested priority

A suggestion for the maintainer, not a decision. Updated after AISI's follow-up, which makes R6 and R2 smaller and adds R10.

1. **R6.** Decided 2026-10-10; small to build (the dispatcher change, `handle_escalation()` and a build-time warning).
2. **R2 and R3.** One mechanism: record each generate's input and tools when it finishes, keyed by the assistant message, plus an explicit unavailable signal and a truncation field. R7 and R10 build on it.
3. **R5.** Unblocks their tool-stage configurations without a custom protocol; the scores reach `human()` through `step.escalations`. The `AfterGenerate` part follows R1.
4. **R1.** Rides on the generate-stages workstream, already high priority; adds `escalate` there and a turn rendering for `human()`.
5. **R10**, with the `conversation` id on `ModelEvent` and `CompactionEvent` that R4 and R7 also need. Built on R2's record; removes the cross-step state that made R7 hard.
6. **R8, security sub-needs first (8.1, 8.4, 8.7), then the rest.** Large and shared with Scout; the security items protect every LLM monitor, the others are configuration.
7. **R7.** Depends on R2, R3 and R10 being recorded; workstream 10.
8. **R4.** Narrowed to earlier protocol decisions; needs the `conversation` id and R7 for replay.
9. **R9.** Lowest, as stated by AISI; follows workstream 13.
