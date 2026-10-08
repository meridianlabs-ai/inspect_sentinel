# Requirements: the AISI sync monitor

External requirements from AISI (UK AI Security Institute), 2026-10-08, for moving their internal sync monitor onto Inspect Sentinel. The first section reproduces the document as received, lightly formatted. The rest is ours: what is met today, where each requirement sits in the plan, where a requirement conflicts with a recorded decision, and a suggested order.

Status: received 2026-10-08; statuses checked against inspect_sentinel `main` and inspect_ai `feature/sentinel` on that date. Conflicts and priorities are for the maintainer to decide; nothing here changes a recorded decision.

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

## Status today

| | Requirement | Status |
|---|---|---|
| R1 | escalate and `human()` at `AfterGenerate` | not met |
| R2 | exact `step.input` at `AfterToolCall`, or an explicit failure | partly met |
| R3 | tool definitions and a truncation signal at `AfterToolCall` | not met |
| R4 | earlier decisions in the conversation | not met |
| R5 | escalate or terminate on scores, without reject | not met |
| R6 | a root escalate does not proceed | not met; conflicts with a recorded decision |
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

**R1.** Only the tool stages are wired ([pr-series.md](pr-series.md), Decisions, "Tool stages only, for now"). The generate-stages prototype, sentinel #42 (draft), allows only `continue` and `terminate` at `BeforeGenerate` and `AfterGenerate`; its decision 16 enables `escalate` next, with the tool stages' root behaviour. `human(stages=)` accepts only `tool_call` and `tool_result` (`_protocols/human.py`; [sentinel.md](sentinel.md), "Humans in the loop", open question 14). inspect_ai's `Host.ask_human` passes one `step.call` to `human_approver` (`_sentinel/_dispatch.py`), so there is no rendering of a whole turn. Gap: `escalate` at `AfterGenerate`, `human(stages=["model_output"])`, and a host rendering of a turn's tool calls with `step.escalations`.

**R2.** inspect_ai rebuilds `step.input` at both tool stages by scanning resident transcript events for the `ModelEvent` whose output carries the assistant message (`_dispatch.py`, `_model_input`). When none is found, for example once the event is no longer resident, it uses the conversation before the assistant message and logs a warning once per sample; nothing on the step says which happened. That fallback is `history`, which after a compaction holds turns the model was not sent. Gap: record the input when the generate finishes ([pr-series.md](pr-series.md), Deferred, "Record `step.input` instead of rebuilding it"; #42 decision 17), and define what a step carries when the exact input is unavailable. Replay: none yet (R7).

**R3.** `BeforeToolCall` and `AfterToolCall` have no `tools` field (`_step.py`); only the generate stages carry `tools` in the design ([sentinel.md](sentinel.md), "The payloads"). Truncation: `step.result` is what the model sees and `step.output` the untruncated value, live only. The log records the truncated byte range on `ToolEvent.truncated`, not on the step. The in-band notice inspect_ai writes into a truncated result ("The output of your call to … was too long to be displayed", `<START_TOOL_OUTPUT>`) is text a tool's output can imitate. Gap: `tools` at the tool stages and one truncation field set the same way live and in replay.

**R4.** A monitor reads its own `store_as()` namespace, keyed by its path, and `step.escalations`, which holds only earlier links' escalations at the current step inside a `sequential` ([sentinel.md](sentinel.md), "The payloads"). `SentinelEvent`s are in the transcript with `step_id` and `conversation`, but nothing on `Context` or `Host` exposes them. [sentinel.md](sentinel.md), "The surfaces", lists sibling reports at the same step among what a monitor never sees; earlier steps are not addressed. Gap: a read of earlier decisions in the conversation, keyed by step id, at every stage and in replay.

**R5.** `threshold(monitors, reject_at, terminate_at=None)` requires `reject_at`, returns only `reject`, `terminate` or `continue`, and requires every monitor to watch `BeforeToolCall` (`_protocols/threshold.py`). Its explanation names the one highest-scoring crossing monitor ([pr-series.md](pr-series.md), "Threshold explanations"). Today this needs a custom protocol. Gap: a built-in protocol with `escalate_at` and `terminate_at`, no required `reject_at`, at `AfterGenerate` and both tool stages, combining monitors across stages, with every contributing score and explanation in the decision.

**R6.** A root `escalate` proceeds, stays recorded as the root's decision, and warns once per process (`_dispatch.py`, `_run`; [pr-series.md](pr-series.md), "Root and escalate"). See [Conflicts](#conflicts-with-recorded-decisions).

**R7.** [sentinel-development.md](sentinel-development.md) designs replay: steps from messages and from events (`ModelEvent.input` and `tools`, `ToolEvent.truncated`), live step ids, a per-transcript store, compaction segments run concurrently and walked in order, and a read mode whose agreement with replay is "a test worth shipping". Built: `SentinelEvent.step_id` at the tool stages (the tool call id), so a live observation can be matched by step. Not built: any replay. Not in the design: logs from before Sentinel, whose approvals are `ApprovalEvent`s; scoring a chosen subset of steps; and `store_as()` state across concurrent segments (the design gives each transcript one store, and a split sub-section a fresh store, its open question 8). Gap: workstream 10, plus those three.

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

## Where each requirement sits in the plan

| | Covered by | Added | Depends on |
|---|---|---|---|
| R1 | Workstream 2 (generate stages); sentinel #42 decision 16 | Deferred, Later: `human()` at `AfterGenerate` | generate stages; R6's decision for the root behaviour |
| R2 | Deferred, "Record `step.input` instead of rebuilding it"; #42 decision 17 | the explicit-unavailable requirement on that entry | generate-stage hook (#42) for recording at generate time |
| R3 | none | Deferred, tool-stage completeness: tools and a truncation signal at the tool stages | the same record as R2 |
| R4 | none | Deferred, design decisions: earlier decisions visible to a monitor | `conversation` on `ModelEvent` and `CompactionEvent`; step ids at the generate stages (#42); R1; R7 for replay |
| R5 | none | Deferred, design decisions: escalate and terminate on scores without reject; workstream 7 bullet | R1 for `AfterGenerate`; R6's decision for an unpaired escalate |
| R6 | Deferred, "Revisit an `escalate` that reaches the top"; sentinel.md open question 24 | link only | maintainer decision |
| R7 | Workstream 10; [sentinel-development.md](sentinel-development.md) | workstream 10 note: pre-Sentinel logs, step subsets, store fidelity | R2 and R3 recorded in the log; `conversation` on events; inspect_core if replay moves into sentinel |
| R8 | Deferred, "Shared message rendering"; workstream 3 (views) | the uncovered sub-needs on that entry; workstream 3 note | the inspect_ai and Scout PRs in that entry; R3 for 8.4; R4 for 8.5 |
| R9 | Workstream 13 (remote human surfaces); `Host.ask_human` | workstream 13 note; Deferred, Later: routing `human()` | R1 for `AfterGenerate` |

## Conflicts with recorded decisions

For the maintainer to decide. Each states the recorded position and the requirement.

1. **R6 and "Root and escalate".** Recorded ([pr-series.md](pr-series.md), "Root and escalate", 2026-09-30): a root `escalate` proceeds, is recorded, and the host warns once per process; the alternative (reject, as approval does when every approver escalates, or a setting with a default) is already listed under Deferred, "Revisit an `escalate` that reaches the top", with the trade-off that rejecting turns every unsure rule into a blocked call. #42 decision 16 extends the same behaviour to the generate stages. R6: the action must not proceed, at any stage, and the log must show what happened. Note that "does not proceed" is not yet defined per stage: `reject` is not legal after a tool call or, in #42, at the generate stages, so the fail-closed form there would be `terminate`.
2. **R2 and the fallback to history.** Current behaviour (inspect_ai `_model_input`, not a pr-series decision record): when no `ModelEvent` is found, `step.input` is the conversation before the call, with a warning once per sample, and the step does not say so. The planned replacement (Deferred, "Record `step.input`") still falls back to a scan for a message no inspect generate produced. R2: never substitute history; make unavailability visible to the monitor. Deciding this also decides what a step's `input` type is when the input is unknown.
3. **R5 and "Threshold explanations".** Recorded (2026-09-29): a reject or terminate names the highest-scoring monitor and carries its explanation. R5: every contributing monitor's score and explanation. Related: `threshold` is `BeforeToolCall`-only "as designed", because `reject` is not legal after a call; R5 asks for the tool-result stage and `AfterGenerate` with no reject.
4. **R4 and monitor independence.** [sentinel.md](sentinel.md), "The surfaces", says a monitor never sees sibling monitors' reports at the same step, as a safety property, and `step.escalations` is per step by design ("The payloads"). R4 asks for other layers' decisions at earlier steps, which neither statement covers. Whether earlier decisions are visible, to which instances, and whether that includes other monitors' observations, needs a decision.
5. **R7 and concurrent compaction segments.** [sentinel-development.md](sentinel-development.md), "Cost and parallelism", runs compaction segments concurrently by default and gives a split sub-section a fresh store (open question 8), while a transcript has one store. R7 asks for parallel segments and, at each step, the `store_as()` state it had live. A stateful monitor cannot have both unless state is reconstructed at each segment start, for example from recorded events. A sketch, not a decision.
6. **R1 and `human()` asking without escalation.** Not a conflict: `human()` asks whenever reached (pr-series, "human()"), and inside a `sequential` that means after an escalation, which is what R1 needs.

## Suggested priority

A suggestion for the maintainer, not a decision.

1. **R6.** A small change once decided, and the requirement that keeps a misconfiguration from letting a flagged action run.
2. **R2 and R3.** One mechanism: record each generate's input and tools when it finishes, keyed by the assistant message, plus a truncation field. Correctness of what tool-result monitors see; R7 depends on it.
3. **R5.** Unblocks their tool-stage configurations without a custom protocol; the `AfterGenerate` part follows R1.
4. **R1.** Rides on the generate-stages workstream, already high priority; adds `escalate` there and a turn rendering for `human()`.
5. **R8, security sub-needs first (8.1, 8.4, 8.7), then the rest.** Large and shared with Scout; the security items protect every LLM monitor, the others are configuration.
6. **R7.** Depends on R2 and R3 being recorded; workstream 10.
7. **R4.** Needs a design and depends on R1, R7 and the `conversation` id on events.
8. **R9.** Lowest, as stated by AISI; follows workstream 13.
