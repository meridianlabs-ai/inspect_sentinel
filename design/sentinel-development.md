# Developing sentinels

How the monitors in a sentinel get built, measured and calibrated before they are trusted to act. The mechanism is Inspect Scout: a monitor becomes a scanner over the steps of a transcript, and Scout's validation sets, metrics and viewer become the monitor's feedback loop.

Status: design sketch for the development loop. Replay over stored logs (AISI R7) is planned to build as of 2026-10-19: [Replay over logs (R7)](#replay-over-logs-r7) states the plan and the decisions it needs, and [Building replay: the PR series](#building-replay-the-pr-series) orders the work. The Python is illustrative. Scout facts were measured against `inspect_scout` at 2026-09-09 and rechecked for the replay plan on 2026-10-19 against `main` of inspect_ai (4aa17c03d) and inspect_scout (b7f7805f); file and symbol citations are to those commits.

**Related design documents.** `sentinel.md` defines the protocol this document develops against, and `sentinel-reference.md` is its reference form. `sentinel-deployment.md` defines the proxy deployment, whose view of a transcript turns out to be the same one offline replay has. `inspect-core.md` covers the package split this document depends on. [requirements-aisi-sync-monitor.md](requirements-aisi-sync-monitor.md) is AISI's R7 and the requirements whose replay halves fall under it (R2, R3, R4, R10). [sub-agents.md](sub-agents.md) fixes what `step.conversation` means.

## Goal

A monitor is a classifier, and nobody should trust a classifier they have not measured. Concretely, the things a monitor author needs to do before configuring one to act:

- run it over a corpus of transcripts and see what it flags
- label a set of steps or trajectories as should-flag / should-not-flag, and get precision, recall and F1 against them
- change the prompt and see whether the numbers moved
- collect the score distribution over known-benign runs, and turn a budget ("audit the top 1%") into a threshold
- check how the monitor actually behaved in an eval it was attached to, without re-running it

Scout already does the first four for *scanners* — validation sets with splits, balanced accuracy, precision, recall and F1, Scout View for labelling, a results dataframe, and scanner versioning to compare prompts — over Inspect logs and over transcripts from other sources. The design problem is therefore small: make a monitor look enough like a scanner that all of that applies, without either package growing a second copy of anything.

## The shape

**A monitor is a scanner over steps.** Scout scans a `Transcript` and returns a `Result`; a monitor takes `(Context, Step)` and returns a `Report`. Everything a step needs is already in a transcript, so a transcript can be re-walked into the steps the monitor would have seen and the monitor run against each in order.

``` python
from inspect_scout import scan, transcripts_from
from inspect_sentinel.scout import as_scanner

validation = "exfil-steps.csv"

scan(
    scanners=[as_scanner(suspicion_monitor("openai/gpt-4o-mini"))],
    transcripts=transcripts_from("./logs").for_validation(validation),
    validation=validation,
)
```

`as_scanner()` accepts a protocol too, so `as_scanner(threshold(...))` back-tests a whole configuration the same way, and the recorded per-layer reports go along with it. It takes the sentinel as `Task(sentinel=)` does, and records it in the configuration form the eval log records (`SentinelConfig`, inspect_ai `_sentinel/_config.py`), so Scout can write it in the scan spec (`ScannerSpec.params`), resume the scan, and rebuild the scanner in a worker process. (`as_scanner()` was first placed in Scout as `inspect_scout.sentinel`; it moves to sentinel, 2026-10-19, decision R7-D3 below.)

Three things fall out of this that the rest of the document works through:

- **Offline replay is a third deployment**, beside the eval and the proxy, and it has a fidelity table like the other two. Reconstruction from a messages-only transcript is *the same projection* the proxy and the bridged agent already use, so one function serves all three.
- **The recorded reports finally have a consumer.** `sentinel.md` records every report, including the ones a protocol ignored, on the argument that they are the benign distribution calibration needs. Scout is where that distribution gets read.
- **Threshold calibration is a scan plus a percentile.** `sentinel.md`'s open question 5 — where a protocol's thresholds come from — has an answer that needs no new machinery.

## Packages

`inspect_sentinel` is its own distribution, to depend only on `inspect_core` once that is extracted (see `inspect-core.md`; today it depends on `inspect_ai`). The chain is then:

```
inspect_core  ←  inspect_sentinel  ←  inspect_ai  ←  inspect_scout
```

Today `inspect_sentinel` depends on `inspect_ai` (for the wire and event types), and imports Scout lazily where it uses it, as `step_as_str` already does (`_excerpt.py`; Scout is a dev dependency). What goes where (revised 2026-10-19 for R7):

| Where | What | Why there |
|------------------------|------------------------|------------------------|
| `inspect_sentinel` | `steps_from_messages()`; the `ReplayStep` record; `Context` from a transcript's info; the replay `Host`; the replay runner (units, store, recorder); `as_scanner()` and the read-mode scanner in `inspect_sentinel.scout`; `replay_monitor()`; `calibrate()` | the replay host and runner are built on the runner's integration contract (`_integration`), so they change in step with it; the development loop's verbs live here ([workstreams.md](workstreams.md), workstream 10) |
| `inspect_ai` | the step builder, shared by the live dispatcher and a log driver, `steps_from_events()`: every step field, the as-sent fold, step ids and earlier `human()` decisions, from live facts or from a log's events | the live builder is here, and one function per field is what keeps live and replay from drifting ([One builder for live and replay](#one-builder-for-live-and-replay)); the event types are not in core today |
| `inspect_scout` | the `subject` field on `Result` and row expansion; viewer work | does the scanning; owns validation and the view; knows nothing of sentinel |

The middle row is pressure on `inspect-core.md`'s first open question. If `ModelEvent` and `ToolEvent` move down, the events-based reconstruction moves down with them and `inspect_sentinel` can replay an eval log at full fidelity on its own. Nothing in this document requires that; it only gets tidier.

## Replay over logs (R7)

AISI's R7 asks to run the exact same `@monitor` over stored eval logs, each replayed step carrying what the monitor saw live, including after a compaction and in logs from before Sentinel; to match each step's report to the same step live; to score a chosen subset of steps; to parallelise and use provider prompt caching; and to keep live and replay identical as Sentinel changes. The replay halves of R2 (`input`), R3 (`tools`, `truncated`), R4 (`human_decisions`) and R10 (`history`) are part of it. Parked for design on 2026-10-14 with one direction agreed: build on Scout, which already runs scanners concurrently across transcripts, manages model connections, records results, resumes and validates, rather than a standalone `replay()` engine.

The plan, proposed 2026-10-19:

1. **The log records which model calls were agent turns, and each call's and compaction's conversation** ([What the log records](#what-the-log-records)). This is the one thing replay needs that a log lacks today, and it is inspect_ai PR 1.
2. **One step builder in inspect_ai, two drivers.** The live dispatcher and the log driver gather facts differently and hand them to the same functions; a test runs an eval with a probe that records every step it sees, and asserts that the steps rebuilt from the written log are equal, field by field ([One builder for live and replay](#one-builder-for-live-and-replay)).
3. **Scout scans transcripts; each step is a result row.** `as_scanner()` is a transcript scanner returning one `Result` per step with `subject` set to the step id; Scout records each as its own row, with `input_ids` `[transcript_id, step_id]`, so validation, the dataframe and `for_validation()` work per step unchanged ([The scan unit and the validation unit](#the-scan-unit-and-the-validation-unit)).
4. **Within a transcript, units run concurrently and steps run in order within a unit.** A unit is a conversation's run of steps between two of its compactions by default, optionally split further, or the whole sample for a stateful monitor ([Cost and parallelism](#cost-and-parallelism)).
5. **Every field has a stated fidelity** for new logs, older logs and bridged agents ([Fidelity in replay](#fidelity-in-replay)); logs from before Sentinel get `human_decisions` from `ApprovalEvent`s and, by opt-in, inferred agent turns ([Logs from before Sentinel](#logs-from-before-sentinel)).
6. **Read mode and replay agree**, checked by the same probe, end to end through Scout ([Two modes: replay and read](#two-modes-replay-and-read)).

### Decisions

Settled by this design (2026-10-19), each argued in its section:

- **Record the conversation, not only a kind marker.** `ModelEvent.conversation` and `CompactionEvent.conversation` are recorded beside `ModelEvent.agent_turn`. This revises the earlier note ("From events", and pr-series.md, Deferred, "Replaying the history as sent") that the instance could follow from the span tree once the turns were marked: the span tree gives it only when the replay loads span events, and only if every event's span chain leads to the span the live dispatcher read through `current_agent_span_id()`, a context variable kept apart from the current span (`util/_span.py`; `SpanRotationScope` moves the current span through a shared cell that tasks spawned earlier also see), which nothing checks today. Recording it costs about 40 bytes an event and makes the equality hold by construction. [sub-agents.md](sub-agents.md) already asks for the instance id on these events. `ToolEvent` needs none: a tool step's conversation is its producing `ModelEvent`'s.
- **The step builder lives in inspect_ai**, used by both drivers; the replay host, runner and Scout adapter live in sentinel.
- **Step ids, including `BeforeGenerate`'s repeat ordinal, are computed in one sequential pass** over the whole sample before any monitor runs, so concurrent units cannot change them.
- **A failed monitor fails the transcript's scan.** The runner records a raising monitor as failed, as live; `as_scanner()` reading a failed report for a monitor it is validating raises, so the transcript's Scout result is an error, as AGENTS.md's failure policy requires. A protocol that handles `observations.failed` reports its decision as usual.
- **Read mode compares by step id and instance path**, and the agreement test uses a probe monitor whose report is a digest of every step field.

**Decisions for the maintainer.** None blocks the rest of the plan: each has a recommendation the PR series assumes, and changing one changes only the PR named.

- **R7-D1: record agent turns whether or not a sentinel is configured?** Recommended: yes. `agent_turn` and `conversation` are set on every `ModelEvent` in a sample, so any log written after PR 1 can be replayed exactly, not only logs from evals that ran a sentinel. Alternative: set them only while a sentinel is active (where `record_agent_turn` runs today), which keeps logs from evals without a sentinel unchanged but leaves them with the older-log fidelity. The fields are additive either way. Affects PR 1.
- **R7-D2: add the `agent_messages` fingerprint to `SentinelEvent` now?** Recommended: no. Replay establishes `agent_messages` by cross-checking `sample.messages` (option 1 of [Reconstructing `agent_messages` in replay](#reconstructing-agent_messages-in-replay)) for every log. The fingerprint (option 2) helps only logs that ran a sentinel, and computing it means hashing the scaffold's whole list at every watched step, which is quadratic in a conversation's length. Revisit when a monitor needs `agent_messages` exactly in replay. Alternative: add `agent_messages_len` and `agent_messages_hash` to `SentinelEvent` in PR 1. Affects PR 1 only if chosen.
- **R7-D3: `as_scanner()` in `inspect_sentinel.scout` rather than `inspect_scout.sentinel`?** Recommended: sentinel. The adapter is the replay host and runner around `run_sentinel`, which change with sentinel's integration contract; Scout then needs only the generic `subject` change and never tracks sentinel versions, and workstream 10's reframing (2026-10-02) already puts the loop's verbs in sentinel, with Scout under the hood. Alternative: Scout, as first drawn in [Packages](#packages), which makes Scout depend on sentinel's private contract. Affects PRs 2 and 5.
- **R7-D4: agent turns in logs written before PR 1: infer them, or `None`?** Recommended: `history` is `None` by default for such logs, and `as_scanner(infer_agent_turns=True)` opts into inspect_ai's existing helper-call classification plus a compaction-summary rule ([Logs from before Sentinel](#logs-from-before-sentinel)), with each result recording that the history was inferred. Alternative: infer by default, which serves AISI's old logs without a flag but hands a monitor a guessed `history` it cannot tell from an exact one, against R2's rule that a monitor must not silently judge context the model was not sent. Affects PR 6.
- **R7-D5: `human()` in replay when the log holds no decision for it.** Recommended: `Host.ask_human` may return `None`, meaning no person can be asked here; only the replay host does, and `human()` then abstains, as [What replay cannot do](#what-replay-cannot-do) already says it should. Alternative: the replay host raises, which fails the protocol's layer and so every transcript whose replay reaches a `human()` without a recorded answer, such as a benign corpus scanned for calibration. Affects PR 2.

## Reconstructing steps

### What the log records

Specified 2026-10-19 as inspect_ai PR 1. Live, inspect_ai decides at each model call whether it is one of its conversation's agent turns and folds the input into that conversation's as-sent list (`Model._record_model_interaction` calls `record_agent_turn`, `_sentinel/_dispatch.py`, which calls `record_sent`, `_sentinel/_sent.py`), but writes neither the decision nor the conversation to the log. A span names the innermost scope rather than the agent; a compaction summary call (`model/_compaction/summary.py`) runs inside the agent's span with the agent's system prompt; and `score()` from a solver, an approver or a tool's own model call look like any other call. So a log cannot say which calls make up `step.history`. Three fields fix that:

| Event | Field | Type | Set | Old logs read as |
|---|---|---|---|---|
| `ModelEvent` | `agent_turn` | `bool \| None`, default `None` | when the event is created in `Model._record_model_interaction`, before the request is sent: `True` when the call is an agent turn by the generate stages' test, `False` otherwise | `None`: not recorded |
| `ModelEvent` | `conversation` | `str \| None`, default `None` | at the same point: the conversation the call is made in, the innermost agent span's id (`current_agent_span_id()`) or else the sample's uuid; `None` outside a sample | `None` |
| `CompactionEvent` | `conversation` | `str \| None`, default `None` | when the event is emitted in `model/_compaction/_compaction.py`: the conversation whose input was compacted, by the same function | `None` |

**The test is the live one.** `is_agent_generate()` (`_sentinel/_context.py`) is "a sentinel is active, the sample is running its solvers (`solving()`), and the innermost scope is an agent's (`current_scope() is AgentScope.ON`)". PR 1 splits out `is_agent_turn()`, the last two conditions, and defines `is_agent_generate()` as an active sentinel and `is_agent_turn()`. `sentinel_solving()` wraps the plan unconditionally (`_eval/task/run.py`), and every scope that turns agent turns off is entered whether or not a sentinel runs: the sentinel run, approval and review (`final`), the compaction summary and `score()` (`final`), a tool's own body (`model/_call_tools.py`) and a bridged host tool (`agent/_bridge/sandbox/service.py`), MCP sampling; an agent span turns them on (`util/_span.py`). So the recorded value is the one the live fold uses, with or without a sentinel (decision R7-D1).

**The conversation is the live one.** `_conversation()` moves out of `_dispatch.py` beside `is_agent_turn()`, and the dispatcher, the recording and the compaction event all call it; the events record `None` where it returns `""`, outside a sample. `_record_model_interaction` computes both values once, sets them on the event, and calls `record_sent(event.conversation, input)` when `event.agent_turn` is set and a sentinel is active, so the live fold and the log agree by construction. The conversation of a monitor's own call inside a sentinel run is the step's (the agent span is still current), with `agent_turn` `False`; that is a start on the Deferred "Monitor model calls tied to their row", not a use of it.

**Additive.** The three fields have defaults, so logs written before them load with `None`; events are pydantic models with the default `extra="ignore"` (none in `event/` sets `forbid`), so an older inspect_ai reading a newer log drops them. inspect_ai's OpenAPI schema and ts-mono's generated viewer types are regenerated with the PR (ts-mono's change lands on its `main` first); the viewer need not show them. No span is added, so the viewer and the ACP TUI, which treat span types specially, see no change, as generate-stages.md decision 15 requires.

**Cost.** About 60 bytes of JSON per `ModelEvent` (`"agent_turn":true,"conversation":"<22-character shortuuid>"`) and 40 per `CompactionEvent`, before the log's compression, against model events whose `input`, `output`, `config` and `call` run to kilobytes; PR 1 measures a 300-turn mockllm sample to confirm. At run time, one context-variable read and two lookups per model call, which the live fold already pays while a sentinel runs.

**Not recorded, and why.** `ToolEvent.conversation`: a tool step's conversation is its producing `ModelEvent`'s. The step's `agent_messages`: see decision R7-D2. A per-step input: R2 is met from the producing `ModelEvent` (pr-series.md, Deferred, "Record `step.input` instead of rebuilding it"). The untruncated tool output: never in the log, so `AfterToolCall.output` is `None` in replay.

**Acceptance for PR 1.** Table-driven tests on mockllm, with and without a sentinel configured: `agent_turn` is `True`, with the agent span's id as `conversation`, for the `generate()` solver, `react` turns, sub-agents run through `as_tool()`, `handoff()` and `agent.run()` (each its own conversation), and a bridged agent's calls; it is `False` for the compaction summary call, `score()` from a solver, a tool's own model call, a monitor's call, an approver's, a reviewer's, MCP sampling, the scorers and sample init. `CompactionEvent.conversation` is the compacted agent's. With a probe sentinel watching all four stages, every `SentinelEvent.conversation` equals its step's `ModelEvent.conversation`, and every live `step.history` equals the fold of that conversation's agent-turn `ModelEvent` inputs that precede the step's `sentinel` span, in log order: the fields are sufficient before any replay code exists. A log written before the fields loads with them `None`. The schema and viewer types are regenerated.

### One builder for live and replay

AISI's last R7 line, "live and replay stay identical as Sentinel changes", is a property of code, not of a document: two builders drift. So there is one.

**inspect_ai owns it.** A new module, `_sentinel/_steps.py`, holds one function per stage that takes the facts a step is made from and returns the step, and the pieces those facts need: the as-sent fold (`_Sent`, made a plain object whose `record(input)` both drivers call; `record_sent` keeps its per-transcript map for the live driver), the producing-call predicate (`_produced_by`), step ids with `BeforeGenerate`'s repeat counter, and the conversation's `human()` decisions published as each step settles (`_publish`, `_human_decisions`). The live driver is `_dispatch.py` as it is, gathering facts from its arguments, the transcript's history and context variables. The log driver, `steps_from_events()`, gathers the same facts from a sample's events and messages:

| Fact | Live (`_dispatch.py`) | From the log |
|---|---|---|
| conversation | `_conversation()` | the producing or own `ModelEvent.conversation` |
| agent turns folded so far | `record_sent` as each call is made | the conversation's `agent_turn` `ModelEvent`s before the step's cut (below), in log order |
| `input`, `tools` at the tool stages | `_producing_event()`: `transcript().history.find_last(_produced_by(id))` | the same predicate over the log's events |
| `input`, `tools`, `tool_choice`, `config`, `model` at the generate stages | `Model.generate`'s arguments | the turn's `ModelEvent` |
| `output` | `Model.generate`'s result | `ModelEvent.output` |
| `call`, `message` | `execute_tools`' arguments | the assistant message in the producing `ModelEvent.output`, with an approver's modification from its `ApprovalEvent`, since approval runs before the sentinel |
| `result` | the `ChatMessageTool` built for the model | the message with `ToolEvent.message_id`, from a later input of the conversation or `sample.messages`, else rebuilt from the `ToolEvent` (`result` is the content the model saw) |
| `truncated` | `execute_tools`' `(original, limit)` | `ToolEvent.truncated` |
| `view` | `resolve_tool_call_view(call, viewer)` | `view.call` from `ToolEvent.view` (or the whole view from an `ApprovalEvent`); `context` absent |
| `human_decisions` | published as each step's span ends | the same rules over the log's `SentinelEvent`s ([Human decisions in replay](#human-decisions-in-replay)) |

**The cut.** Live, a step's `history` and `human_decisions` are taken when the step is built, just before its `sentinel` span begins. In a log that ran a sentinel watching the step's stage, the cut is that span's `SpanBeginEvent`, found by the `SentinelEvent`s inside it (their `step_id` and `stage`), so the replayed value is the live one; a span holding none, because every instance in it abstained, falls back to the anchor. Elsewhere the cut is the step's anchor: before its own `ModelEvent` at `BeforeGenerate`, after it at `AfterGenerate`, after the producing `ModelEvent` at the tool stages. The two agree unless another agent turn of the same conversation started between the anchor and the step, which takes two agents sharing a conversation, as sub-agents inside one bridged CLI do.

**Events in log order.** The driver reads events in the order the log stores them, flattening an older log's nested events (`ToolEvent.events`) in place, and makes one sequential pass over the whole sample before any monitor runs: step ids, the fold, the cut and the published decisions all depend on order, and none on monitor results. The pass yields `ReplayStep`s in live order; the runner then groups them into units ([Cost and parallelism](#cost-and-parallelism)).

**What pins it.** Two tests, so a field added to a step without its log form fails CI:

- In inspect_ai (PR 3), **live equals replay, step by step.** Scripted mockllm evals (`react` with summary compaction, parallel tool calls, a truncated result, a sub-agent through `as_tool()` and one through `handoff()`, a retried refusal that repeats a `BeforeGenerate` id, a `GenerateConfig` system message rebuilt each call, an approver that modifies a call, a scripted `human()` answer) run with a probe sentinel watching all four stages that records each step it is handed. The test rebuilds the steps from the written log with `steps_from_events()` and compares them by stage and step id over every field of the step dataclass, iterated with `dataclasses.fields()`, against a short, named list of fields that replay leaves absent (`AfterToolCall.output`, `view.context`, `agent_messages` where [Reconstructing `agent_messages` in replay](#reconstructing-agent_messages-in-replay) gives `None`). A new step field is compared by default, so it fails until its log form is built or it is added to that list on purpose.
- In inspect_sentinel (PR 5), **read mode equals replay, end to end** ([Two modes](#two-modes-replay-and-read)).

### From messages

The canonical reconstruction works from `list[ChatMessage]` alone, because that is what every source has — an eval log, a Scout transcript from another agent framework, a proxy request. It is exactly the projection `agent/_bridge/_approval.py` performs on a model response and `sentinel-deployment.md` describes for a proxy:

| Stage | Reconstructed from | Fidelity |
|------------------------|------------------------|------------------------|
| `BeforeGenerate` | every message before an assistant message is that generate's `input` | `tools`, `tool_choice`, `config` absent |
| `AfterGenerate` | the assistant message is `output.choices[0].message`; `usage`, `time` absent | good |
| `BeforeToolCall` | each `tool_calls[i]` on an assistant message, with the message text and history to that point | `view` absent |
| `AfterToolCall` | the `ChatMessageTool` whose `tool_call_id` matches, as `result`; history to that point | `output` (untruncated) absent; `truncated` is `None` (unknown) |

Messages give the two histories unevenly. A transcript's messages are `agent_messages` at the end of the run; at a step, `agent_messages` is a prefix of them only where replay can establish that the scaffold had not rewritten its list since (`react`'s overflow recovery replaces it, and a sub-agent's list is not the sample's), and `None` otherwise, as under a proxy (see "Reconstructing `agent_messages` in replay" below). At the tool stages `input` (and with it `tools`) is `None` from messages alone: the history to that point is right until the first compaction and wrong after it (the model was shown a summary plus recent turns), so it is not offered as the model's input. A live step without a recorded model call gets `None` the same way, so a monitor handles both alike.

`view` and `output` being absent, and `agent_messages` being `None`, is the case `sentinel.md` already requires monitors to tolerate for proxies, so a portable monitor replays from messages unchanged. Note the last row is *better* than the proxy's: a proxy never sees the final step's result because no further request carries it, whereas a transcript is complete.

``` python
@dataclass(frozen=True)
class ReplayStep:
    id: str                 # the step id — see below
    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]  # as SentinelEvent.stage
    step: Step
    segment: tuple[str, int]  # the conversation, and how many of its compactions came before


def steps_from_messages(messages: list[ChatMessage]) -> Iterator[ReplayStep]: ...
```

The `Context` is the sample's and constant, so the runner holds it rather than each step (revised 2026-10-19). `ReplayStep` is sentinel's; inspect_ai's `steps_from_events()` builds it, as the live dispatcher builds sentinel's step types.

### From events

An eval log's events restore what messages lose. `ModelEvent` carries `input`, `tools`, `tool_choice`, `config` and `output` — `BeforeGenerate` and `AfterGenerate` field for field. `ToolEvent` carries `function`, `arguments`, `view`, `result` and `truncated` (the output's size and the limit it was cut to), which restores `view` and lets `result` be exactly what the model saw. It gives `AfterToolCall.truncated` as the live host set it: `Truncation(*event.truncated)` when set, and `False` when absent in a log new enough to record it, that is, one whose `eval.packages["inspect_ai"]` is 0.3.31 or later (`max_tool_output`, inspect_ai #478, which added the field); in an older log, `None`. At the tool stages the producing `ModelEvent` gives `input` and `tools`, the same lookup the live dispatcher makes. The one thing an eval log does not hold is the untruncated tool `output`, so `AfterToolCall.output` stays optional in replay too.

Events also restore `input` exactly, from `ModelEvent.input`, and let `history`, the as-sent view of an agent's run (AISI R10), be rebuilt per agent; it is not `agent_messages`, the scaffold's list. Live, inspect_ai folds each agent turn's input in as the call is made and writes nothing to the log (built 2026-10-14; `sentinel.md`, "The history as sent"). Replay runs the same fold, the same `_Sent.record()`, over the inputs of the conversation's `ModelEvent`s with `agent_turn` set, up to the step's cut ([One builder for live and replay](#one-builder-for-live-and-replay)). What the fold needs that a log lacked is which model calls were the instance's agent turns, and which instance; [What the log records](#what-the-log-records) adds both. Logs written before that get the fallback in [Logs from before Sentinel](#logs-from-before-sentinel). `CompactionEvent`s are not needed for the fold, since a message stays in the list once sent; they mark where a conversation's input stopped extending its previous input, which is where replay splits units.

`steps_from_events()` is an *enrichment* of the messages reconstruction rather than a second algorithm: it walks the same steps, in the order the log records them, and attaches the matching event's fields where one exists. That keeps step identity (below) independent of whether events were available, which is what lets one validation set serve both kinds of source.

### Fidelity in replay

What each field of a replayed step holds, by source. "Exact" means equal to what the live step held, or would have held had a sentinel watched that stage, and is what the live-equals-replay test checks. "Older log" is one written before [What the log records](#what-the-log-records); "bridged" is an agent run through `agent_bridge()` or `sandbox_agent_bridge()`, whose tool calls inspect_ai does not yet check live (inspect_ai #5759).

| Field | New log | Older log | Bridged agent |
|---|---|---|---|
| step id, `stage` | exact | exact | exact |
| `conversation` | exact (`ModelEvent.conversation`) | from the span tree: the innermost `span_begin` of type `agent` enclosing the event, else the sample's uuid; exact where the span chain leads to the live agent span | exact |
| `model`, `input`, `tools`, `tool_choice`, `config` (generate stages) | exact (`ModelEvent`) | exact | exact |
| `output` (`AfterGenerate`) | exact | exact | exact |
| `input`, `tools` (tool stages) | exact from the producing `ModelEvent`; `None` when there is none, as live. Live also gives `None` when the transcript's history is unavailable, which a complete log never is | exact | no live tool stages; replay builds them where a `ModelEvent` produced the assistant message, else from messages, with `input` and `tools` `None` |
| `call`, `message` | exact | exact | from messages |
| `result` (`AfterToolCall`) | exact | exact | from messages |
| `output` (`AfterToolCall`) | `None`: the untruncated output is not in the log | `None` | `None` |
| `truncated` | exact (`ToolEvent.truncated`, `False` when unset) | exact from inspect_ai 0.3.31; `None` before | `None` |
| `view` | `view.call` exact (`ToolEvent.view`, or the whole view where an `ApprovalEvent` recorded it); `view.context` absent | same | from messages: absent |
| `history` | exact | `None`, or inferred by opt-in (decision R7-D4) | as live: the bridge's conversation, with the CLI's helper calls and sub-agents in it; exact where a sentinel span gives the cut, else approximate where sub-agents overlap |
| `agent_messages` | approximate: a verified prefix of `sample.messages` for the sample's top-level agent, else `None` (R7-D2) | same | `None` |
| `human_decisions` | exact, from `SentinelEvent`s when the eval ran a sentinel; from human-approver `ApprovalEvent`s when it did not | from `ApprovalEvent`s | as for its source log |
| `store_as()` state | as live when the unit is the whole sample (`split=None`), except between steps that ran concurrently live, which replay runs in log order; fresh at each unit otherwise | same | same |

`Context` follows [`Context`, `Host` and `Store` in replay](#context-host-and-store-in-replay). Read mode is exact by definition: it reads what was recorded.

### Reconstructing `agent_messages` in replay

Noted 2026-10-13 with the maintainer, alongside making `step.agent_messages` optional (`None` when the host can't provide the caller's list: replay from a log, proxy hosts, steps with no scaffold conversation).

`agent_messages` is the scaffold's list *at that step*. An eval log holds only the list at the end of the sample (`sample.messages`) and, per model call, what the model was sent (`ModelEvent.input`, already `step.input`). For an append-only scaffold the step's history is a prefix of `sample.messages`, cut at the step's anchor (tool stages: the assistant message holding the call; generate stages: the last message of the call's input), found by message id.

`react` keeps `state.messages` uncompacted: compaction shapes a copy for each model call, so a compaction does not break the prefix. `react` does change its list in two places: `_remove_submit_tool` at the end (default `keep_in_messages=False`) strips the submit call and result, and `on_continue` returning an `AgentState` can replace the list. Multi-solver chains, handoffs and custom solvers can also rewrite it, and bridged agents' `sample.messages` is not the CLI's conversation. Replay therefore has to verify a prefix, not assume it.

Options:

1. **Cross-check the log (works on existing logs).** For the agent instance, every `ModelEvent` output message and every tool result must appear in `sample.messages` with the same id, in order, with the same content, up to the step's anchor; outputs are never compacted, so this holds through compaction. If it does, the prefix is the step's history; steps after a divergence (e.g. the final submit turn) get `None`. Heuristic: an edit to a message that is neither a model output nor a tool result (a user message a custom solver rewrote) is not detected.
2. **Record a history fingerprint at run time (exact, any scaffold).** Each `SentinelEvent` stores `history_len` and a hash of the step's history (tens of bytes, never the messages). Replay takes the first `history_len` messages of `sample.messages` and compares hashes: equal means exactly what the live step saw, otherwise `None`. Needs two fields on inspect_ai's `SentinelEvent` and a canonical message serialization for the hash.

Proposed: 2 for logs written after the fields exist, 1 as the fallback for older logs. Not built; part of workstream 10 / R7.

Revised 2026-10-19 (decision R7-D2, for the maintainer): option 1 for every log, option 2 deferred. Option 2 helps only logs from evals that ran a sentinel, since only they have `SentinelEvent`s, and it means hashing the scaffold's whole list at every watched step, quadratic in the conversation's length where `history` is built incrementally for the same reason. The scaffold's list is Inspect-only and `None` under a proxy, so a portable monitor already handles `None`; AISI's need is `history` (R10), which replay gives exactly. The cross-check runs in the log driver: at the tool stages the anchor is the assistant message holding the call, at the generate stages the last message of the call's input that `sample.messages` holds (the input itself is the prepared input, after `Model._prepare_input()` resolves reasoning history and collapses messages, so it is not the caller's list). It applies only to the conversation whose messages `sample.messages` is, the sample's top-level agent; a sub-agent's steps get `None`.

### Human decisions in replay

Decided 2026-10-14 (AISI R4); not built. A replayed step's `human_decisions` holds what the live step held: the decisions a person made through `human()` at earlier steps of the conversation. Replay reads them from the log rather than making them, which is the only way, since a replay cannot ask the person: for each step, the `SentinelEvent`s with `factory == "inspect_sentinel/human"`, `kind == "decision"`, `status == "reported"` and the step's `conversation`, from steps whose `sentinel` span ended before the `sentinel` span of the step itself began, in event order (live, the host publishes a step's decisions as its span ends, and snapshots the list when it builds a step, before its span begins), less each one withdrawn by a `superseded` event at its own step: one with the person's action at the person's `path` or the path of a layer above it, the rule `supersedes()` applies live. A `terminate` that a sibling's `decide_final()` outran is recorded `reported` and then `superseded`, possibly as a copy a composition passed up, at a layer above the person. Both events fall inside the step's span, so a later step sees the same result live and in replay, including a parallel step that began while that span was open. Each becomes an `HumanDecision`: `stage` and `step_id` from the event; `decided` a `Reported[Decision]` rebuilt from `path` (its last segment the name), `function`, `action`, `explanation`, `message`, `modified`, `references` and `metadata`; `call_ids` the step id at a tool stage, and at `model_output` the tool calls of the assistant message with that id (its `ModelEvent` output, or `sample.messages`). Because the records are what was decided live, segments replayed concurrently each see the same list.

A log from before Sentinel records a person's approval as an `ApprovalEvent`; replay could map one from a human approver to the same record (stage `tool_call`, step id the call id, `approve` to `proceed`), but this is not planned; until then such a log gives `None`, since replay cannot tell whether a person decided anything. *Revised 2026-10-19 for AISI's R7: planned, in [Logs from before Sentinel](#logs-from-before-sentinel).*

The log driver applies these rules with the same functions the live host uses: it rebuilds each `Reported[Decision]` from the event and calls sentinel's `kept_human_decision()` and `supersedes()` (`_human_decisions.py`), as inspect_ai's `_Recorder` does live, and publishes a step's kept decisions at its span's end with the shared `_publish`, so a field added to `HumanDecision` changes both paths at once.

### Logs from before Sentinel

AISI's R7 includes logs from before Sentinel, where a person's approvals are `ApprovalEvent`s. Planned 2026-10-19 as PR 6. Everything a tool or generate step reads from `ModelEvent`, `ToolEvent` and messages is as exact in these logs as in new ones (the fidelity table's middle column). Three things differ.

**Conversation.** Without `ModelEvent.conversation`, the log driver takes the innermost `span_begin` event of type `agent` enclosing the event, following `parent_id`, or else the sample's uuid: what `current_agent_span_id()` returned live wherever the span chain leads to that span. So the replay must load span events (`as_scanner()` asks Scout for `span_begin` and `span_end`).

**Agent turns** (decision R7-D4). Without `ModelEvent.agent_turn`, `history` is `None` by default: a monitor is not handed a guess it cannot tell from the model's input. `as_scanner(infer_agent_turns=True)` opts in to a classification from what such logs hold, and records `"history": "inferred"` in each result's metadata:

- outside the solvers (before the `solvers` span or after it), or with `role` `"monitor"`, `"sentinel"` or another role the eval did not give its agents: not a turn;
- inside a `sentinel` span: not a turn;
- the model call that ends just before a `CompactionEvent` with `source == "inspect"` and `type == "summary"`, in the same agent span: the summary call, not a turn;
- a call whose innermost enclosing span of type `tool` or `agent` is a `tool` span (`model/_call_tools.py` runs each call in one): the tool's own call, not a turn;
- otherwise inspect_ai's own helper-call classifier, built for the viewer's timelines (`event/_timeline.py`: `_wrap_utility_events` treats a call with a different system prompt from the span's first tool-calling call, or a one-token warm-up call, as a helper; `_classify_utility_agents` does the same for a single-turn agent span with a different prompt from a parent that runs a tool loop). What it calls a helper is not a turn; the rest are.

The classification is heuristic: a scaffold's own model call with the agent's system prompt counts as a turn, as live (pr-series.md, "The history as sent", known limits), and `score()` from a solver or a reviewer's call that shares the agent's prompt is counted wrongly. The live-equals-replay test runs the classifier over its own new logs with the recorded fields removed, and reports how often it disagrees with them, so the error rate is a measured number in the PR.

**Earlier decisions.** When the eval ran no sentinel (`eval.config.sentinel` unset), each `ApprovalEvent` from a person becomes a `HumanDecision`, published once its tool call's step settles, as live: stage `tool_call`, `step_id` and `call_ids` the call's id, `decided` a `Reported[Decision]` with the approver's name as its `name` and `path`, `approve` as `proceed` with the explanation, `reject` as `reject` with it as `message` and `explanation`, `terminate` as `terminate`, and `modify` as `modify` with the `modified` call. An `escalate` is passed on, not decided, and is skipped. Which approvers are people is a parameter, `human_approvers=("human",)`, the registry name inspect_ai's `human_approver` records (`approval/_call.py`, `record_approval`); a team with its own approval UI (AISI R9) names theirs. A person who decides through approval always leaves an `ApprovalEvent`, so a log without one gives `()`, not `None`. A log from an eval that ran a sentinel gets `human_decisions` from its `SentinelEvent`s only, as live, where approval's decisions were never in it.

### `Context`, `Host` and `Store` in replay

| Member | In replay |
|------------------------|------------------------|
| `eval` | an `EvalContext` when the source has a task and sample id; None otherwise, as in a proxy |
| `eval.task`, `eval.task_description` | from the log's eval spec; from Scout's `TranscriptInfo.task_set` and `metadata` for other sources |
| `eval.sample_id`, `eval.epoch` | `TranscriptInfo.task_id`, `task_repeat` |
| `eval.sample_input` | the sample input from the log; the first user turn otherwise |
| `eval.metadata` | task and sample metadata merged, as in an eval |
| `eval.target` | available from the log; **still opt-in**, and the adapter records that it was requested, for the same validity reason as in an eval |
| `store_as()` | backed by a fresh `Store` per unit ([Cost and parallelism](#cost-and-parallelism)), so incremental monitors accumulate across a unit's steps and nothing leaks between units or transcripts; with `split=None` the unit is the sample, which is the live scope (`store_as()` is per sample, [sub-agents.md](sub-agents.md), decision 3) |
| `host` | sentinel's replay host: `generate()` resolves models as inspect_ai's host does live (`_HostNative.generate`: a named model, else the `monitor` role or the role asked for, from the scan's `model_roles`, else the scan's model with the same warning), and its usage lands in the scan's `model_usage`; `ask_human()` answers from the log (below); `fetch()` resolves named endpoints from scan config; `terminate()` returns `False` |

### What replay cannot do

Replay reproduces the monitor's *reports*, not its *effects*. A `reject` in replay does not change the transcript; the agent already did what it did. This is the right limitation for measuring a classifier — precision and recall are about reports — but it means two things cannot be back-tested offline:

- **Interventions.** Whether rejecting call 7 would have led the agent somewhere better is a question for a live eval, not a replay.
- **Humans.** `human()`, a protocol with no children, cannot be replayed. The useful substitute is to *replay the recorded human decision* when the transcript carries a `SentinelEvent` from that `human()` instance for that step, so a `sequential` can be back-tested against the answers a person actually gave; otherwise `human()` abstains in replay. The replay host does this in `ask_human()`: it reads the asking instance's path from the runner's layer state and answers with that instance's recorded decision at the step (`approve` for a `proceed`), or with `None` when there is none, which `human()` takes as abstaining (decision R7-D5; today `ask_human()` must return a `HumanAnswer`, so this is a sentinel change in PR 2). A log from before Sentinel has no `human()` instances, so a `human()` replayed over it always abstains; its `ApprovalEvent`s reach monitors as `human_decisions`, not as answers.

## Step ids

Per-step validation needs each step to have an id that exists in a messages-only transcript, in an events transcript, and **live, at the moment the monitor runs** — so that a label made once serves replay and also the read mode over recorded events. Message and tool-call ids satisfy that. Event uuids do not: a `BeforeGenerate` monitor runs before its `ModelEvent` exists.

| Stage | Step id |
|------------------------|------------------------|
| `BeforeGenerate` | id of the last input message — the one that triggered the generate |
| `AfterGenerate` | id of the assistant message produced |
| `BeforeToolCall` | the tool call id |
| `AfterToolCall` | the tool call id |

One wrinkle: a rejected call followed by a regenerate produces two `BeforeGenerate` steps behind the same last message. That stage takes an ordinal suffix on repeat — `{message_id}:2` — and the other three stages are clean. Within one monitor's stage, ids are unique per transcript; a `BeforeToolCall` monitor and an `AfterToolCall` monitor both key by tool call id, but validation sets are per scanner so there is no collision.

**Consequence for `sentinel.md`:** `SentinelEvent` gains `step_id: str`, recorded by the runner live, so the read-mode scanner emits results with the same subjects the replay scanner does. *Built: `SentinelEvent.step_id` at all four stages (inspect_ai `_dispatch.py`, `_step_id`).*

**Computing them in replay** (2026-10-19). The repeat ordinal is a count per sample, across its conversations: live, `_step_id` counts each `BeforeGenerate` step's last input message id per transcript (`_model_inputs`), and it runs for every agent generate whenever `BeforeGenerate` is watched. The log driver keeps the same counter in its sequential pass, counting every agent-turn `ModelEvent` in log order, so the ids match whenever live computed them, and concurrent units cannot reorder them. A message without an id counts under `""`, as live.

## The scan unit and the validation unit

Three grains of label are in play, and they do not need three scan units:

| Grain | The label answers | Keyed by |
|------------------------|------------------------|------------------------|
| step | did this tool call or generate deserve a flag | step id |
| transcript | did this trajectory contain an attack | transcript id |
| protocol | did the configuration reach the right decision at this step | step id; the result is the outermost layer's |

Four ways of getting the step grain into Scout were considered:

- **A `Step` scanner input type.** First-class, and the view could render it natively. But `sentinel.md` says monitors should be incremental, so nearly every real monitor is stateful across steps and needs its transcript's steps *in order with one store*. Scout's input model does not promise sequential in-order invocation within a transcript, so the new type would also need a new ordering guarantee. It buys less than it looks.
- **A `Sequence[Step]` input per transcript.** Preserves order, but the ids on a sequence input are the whole list, so validation cannot address one step.
- **A resultset per transcript, validated by label.** No Scout change, but label validation is presence-or-absence per label, so "step 7 was flagged and should not have been" is not expressible.
- **Keep the transcript as the scan unit; make the result the validation unit.** Chosen.

### The chosen shape

The adapter is a `Scanner[Transcript]`. It builds the steps in one pass, runs them in units ([Cost and parallelism](#cost-and-parallelism)), in order within a unit with the unit's own store, and returns one `Result` per eligible step. Each result carries a **subject** — the step id. A validation case whose id is a step id matches the result with that subject.

``` python
as_scanner(monitor, unit="step")        # one Result per step; subject = step id
as_scanner(monitor, unit="transcript")  # one Result; transcript id, as today
```

State stays where `sentinel.md` already puts it; Scout's parallelism across transcripts is untouched; and the transcript-grain form is the same adapter reducing to one result by max suspicion or strongest decision, with `references` to the flagged steps.

### What Scout changes

Two things, both small, and deliberately not a new input type or a new matching path.

**`subject` on `Result`.** A reference says what a result *cites as evidence*; a subject says what a result *is about*. A step result routinely has both — it is about tool call X and cites messages M3 and M7. They differ in cardinality (references are a list; a subject is exactly one thing, and validation must match exactly one), in id kind (`Reference.type` is `message` or `event`; a tool call id is neither), and in what the viewer does with them (references render as links from cite text; a subject drives row identity and navigation). So `references` stays for what an LLM monitor's explanation points at, and `subject` is new:

``` python
class Result(BaseModel):
    ...
    subject: str | None = Field(default=None)
    """Id of the thing this result is about, when narrower than the scanner
    input. Each subject-bearing result is recorded and validated as its own row."""
```

**Expansion into rows.** Today `_validate_scan` (`_scan.py`) matches a case id against the input ids of one `ResultReport`, and a scanner returning a list is folded into a single resultset report (`as_resultset`). When a scanner returns results carrying a `subject`, each becomes its own `ResultReport` with `input_ids=[transcript_id, subject]` (settled 2026-10-19). Matching, the dataframe, the summary metrics and the results view all already work per report, so per-step behaviour falls out. A result with no subject is recorded exactly as today; a list mixing the two is an error.

The transcript id goes first in `input_ids` for two reasons found in Scout's code. Step ids are unique within a transcript, not across a corpus: message ids are shortuuids, but a tool call id is whatever the provider or a scripted output gives, and nothing promises it is unique across samples. And `Transcripts.for_validation()` selects transcripts by every id in every case (`sample_id IN ...`), so a case with id `["<transcript>", "<step>"]` selects its transcript with no change, where a bare step id would select nothing. A per-step validation set therefore keys cases by the pair; Scout already accepts a list id on `ValidationCase`.

One rule keeps matching unambiguous: **subjects are unique within one scanner's results for one input.** The adapter emits one result per step for the monitor being validated; a protocol's child reports go in that result's metadata. To validate a leaf inside a protocol, wrap the leaf directly. A tool call's two steps share its id, so an instance watching more than one stage is scanned one stage at a time: `as_scanner(..., stage="tool_result")`, required when the instance watches several stages and defaulting to its only one otherwise.

What the expanded rows share (settled 2026-10-19, answering open question 2): each row's `input` is the transcript, recorded once through the pooled-dedup path that already avoids a copy per row; the scan's `events` and `model_usage` for the transcript go on the transcript's first row and are empty on the others, so sums over the dataframe stay right, and a step's own usage, when the adapter measures it, is in that result's metadata. A transcript's rows are returned together from one scanner call, so a resumed scan never holds half a transcript. Whether the view needs a step's messages materialised for rendering stays a viewer question.

`"sentinel"` is already in Scout's `EventType` (`_transcript/types.py`), so a scanner can ask for these events; the earlier note that Scout must add it is done.

## Mapping `Report` to `Result`

| Report | `value` | `answer` | `metadata` | `label` | `subject` |
|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| `Observation` | `suspicion` — float or dict | — | monitor metadata | instance path | step id |
| `Decision` | flagged: `action` in `reject`, `modify`, `terminate` | `action` | `audit`, `modified`, child reports | instance path | step id |
| abstained | `None` | `None` | — | instance path | step id |
| failed (a monitor raised) | the transcript's scan raises: no row, the transcript's result is an error | | | | |

Three things the table encodes:

- **A dict suspicion validates per key** through Scout's existing dict-target path, so a multi-dimension monitor gets per-dimension precision and recall without any adapter logic.
- **The boolean flagged value is what makes binary labels work.** A per-step validation set will label steps the monitor abstained on, and abstention produces no report. The adapter emits a result for every step at the monitor's stage, `None` on abstain, and a label of `false` compares equal to "not flagged" only if the value is boolean. Teaching the `eq` predicate that `None` is negative was the alternative; mapping to a boolean keeps the predicate honest and leaves the action in `answer` for anyone sweeping on it.
- **`escalate` is not flagged.** It is "I decline to decide", not a verdict. Inside a `sequential` it also means "worth a person's look", which a reviewer might want to count; `answer` carries it for that. This is a default that is cheap to get wrong, and it is recorded as a decision rather than left implicit.

The `explanation` and `references` map directly from the report's own fields — an LLM monitor using Scout's `message_numbering` in its prompt gets cites for free.

**`unit="transcript"`** (settled 2026-10-19, answering open question 3) reduces the step results to one: for a monitor, the maximum suspicion, per dimension when it is a dict; for a protocol, the root's strongest decision across the steps (`terminate`, then `reject`, `modify`, `escalate`, `proceed`); `references` the steps that set it. Taking the root's decision is what a protocol result already means at the step grain ("the result is the outermost layer's").

**Replaying a chosen subset of steps.** `as_scanner(..., steps=...)` takes a predicate over `ReplayStep`, and `steps_in(validation)` builds one from a validation set's `[transcript, step]` ids. The pass still builds every step, which costs no inference; the runner invokes the sentinel only for the chosen ones, so a unit's store sees only those. With R10 a monitor that needs the earlier conversation reads `history`, so AISI's subset is "only run these steps", as their follow-up said.

## Two modes: replay and read

Both consume the same validation set, because both key results by step id.

| Mode | What runs | When |
|------------------------|------------------------|------------------------|
| **replay** | reconstruct steps, run the monitor, one result per step | you changed the prompt, the model, or the monitor's logic |
| **read** | walk the transcript's recorded `SentinelEvent`s, one result per event | you want to know how the monitor behaved in the run it was attached to |

Read mode costs no inference and is what makes "run the eval observe-only, then look" a real workflow rather than a slogan. It is also the only mode that sees what the protocol actually did with each report, from the decision events at the parent paths, so "how often did `resample` substitute a draw" is a read-mode query. Scout's `EventType` literal gains `"sentinel"` so the transcript loader can filter for these events.

Read mode also gives the honest check on replay itself: replay the monitor over a log it was attached to, and the two modes should agree wherever the monitor is deterministic. Where they do not, the reconstruction is wrong, and that is a test worth shipping.

**The read-mode scanner** (designed 2026-10-19, PR 5): `as_reader(path, *, stage=None)` in `inspect_sentinel.scout`, a transcript scanner asking Scout only for `sentinel` events. For the instance at `path` it emits one result per `SentinelEvent` with status `reported` (and `superseded` for a decision, with the status in metadata), `subject` the event's `step_id`, mapped as [Mapping `Report` to `Result`](#mapping-report-to-result) maps a live report. An `error` event is a failed monitor, and the scan raises as replay does. A step the instance watched but did not report on (it abstained) has no event; the reader emits nothing for it, and the agreement test below compares only steps both modes report.

**The agreement test** (PR 5). A probe monitor, kept in sentinel's tests, watches all four stages and reports `suspicion` `0.0` with metadata holding a digest of every field of the step it was handed: per field, a hash of its JSON form, with the fields replay leaves absent (the list the live-equals-replay test names) left out. A test runs a set of scripted mockllm evals with the probe attached, then scans the logs through Scout, at each stage once with `as_scanner(probe(), stage=...)` and once with `as_reader("probe", stage=...)`, and asserts that for every `[transcript, step]` both report, the metadata is equal; and that replay reports every step read mode does. The live-equals-replay test in inspect_ai compares the steps themselves; this one checks the whole path a user takes, through Scout's loading, the units, the replay host and the row expansion. A monitor author can run the same check on their own logs with any deterministic monitor.

## The workflow

1. **Write** the monitor as `sentinel.md` describes. Its unit tests use `replay_monitor()` from `inspect_sentinel` over a handful of message lists — no Scout, no eval.
2. **Replay** it over a corpus with `as_scanner()` and look at what it flags in Scout View.
3. **Label** from the results view — each step is a row — building a validation set with `dev` and `test` splits. Scout's existing case editor does this; steps appear as results rows, so no new labelling gesture is required for the first version.
4. **Validate**: precision, recall, F1, balanced accuracy over the `dev` split. Change the prompt; bump the monitor's version; re-scan; compare. This is Scout's existing loop.
5. **Calibrate** thresholds (below) if the monitor is an `Observation`.
6. **Configure** the protocol with those thresholds and attach it to the eval — observe-only first.
7. **Read** the resulting logs in read mode to confirm the deployed behaviour matches the replayed one, then let it act.

Steps 2–4 are exactly what a scanner author does today. The monitor-specific parts are the adapter in step 2 and calibration in step 5.

## Calibration

`sentinel.md` insists thresholds live in protocols, not monitors, because a threshold needs a benign score distribution and a monitor looking at one step has never seen one. This is where the distribution comes from.

**From a benign corpus.** Replay an `Observation` monitor over transcripts known to be benign. The results dataframe's `value` column *is* the benign distribution. A budget becomes a threshold by percentile: "audit the top 1%" is the 99th percentile of that column, per dimension when suspicion is structured.

**From a labelled corpus.** Replay over a corpus with numeric or boolean step labels. Scout's `gte` predicate against a numeric target is a threshold; sweeping it gives the precision–recall curve, and the operating point is a choice the author makes with the curve in front of them rather than a constant in a prompt.

A helper closes the loop:

``` python
thresholds = calibrate(results, budget=0.01)          # percentile per dimension
thresholds = calibrate(results, target_recall=0.95)   # from labels
Task(sentinel=threshold(suspicion_monitor(), reject_at=thresholds["sabotage"]))
```

It is ordinary pandas over a Scout results dataframe. It lives in inspect_sentinel, as one of the development loop's verbs, and uses Scout for the scanning (workstream 10, reframed by the maintainer on 2026-10-02). What matters more than the helper is that the *inputs* to it — recorded scores, including ones a protocol ignored — are produced by the runner as a matter of course, which is why `sentinel.md` records them.

The calibration corpus should be held out from the validation set's `test` split, and a threshold derived from one model's scores is not a threshold for another model's; both are ordinary ML hygiene rather than framework concerns, but the docs should say them.

## Monitors as scores

`sentinel.md`'s open question 11 asks whether a monitor can produce a `Score` beside the task's own scores. It can, without touching scoring:

``` python
Task(scorer=[my_task_scorer(), as_scorer(as_scanner(suspicion_monitor(), unit="transcript"))])
```

`as_scorer()` already turns a transcript scanner into an Inspect scorer. The transcript-grain adapter reduces the monitor's per-step reports to one value, so the monitor's suspicion appears in the log as a score with the usual metrics, and the flagged steps travel as `scanner_references` in the score metadata, which Inspect View already renders. Nothing in `inspect_ai.scorer` changes.

## Cost and parallelism

An LLM monitor over a corpus is one inference call per eligible step, which is many more calls than one per transcript. What keeps it tolerable:

- **Incremental monitors are cheaper to replay**, not just to run — the design's insistence on classifying the newest turn and folding into state means replay cost is linear in steps rather than quadratic, and the adapter's per-unit store is what makes that work.
- **Stage filtering happens before loading.** The adapter declares the content it needs from the stage it scans: messages, and the event types the log driver reads (`model`, `tool`, `compaction`, `approval`, `sentinel`, and `span_begin` and `span_end` for the cut and for older logs), so Scout loads nothing else.
- **Parallelism is across transcripts**, as for any transcript scanner (Scout's `max_transcripts`, default 25, also the default `max_connections`), **and across units within a transcript.**

**Units** (designed 2026-10-19; AISI describes their current implementation in R7). After the step pass, the runner groups a transcript's steps into units, runs units concurrently in an anyio task group, and runs the steps of a unit in order, each unit with its own `Store`, recorder and replay host. `as_scanner(split=...)` chooses the unit:

| `split` | Unit | Store | For |
|---|---|---|---|
| `"compaction"` (default) | a conversation's steps between two of its compactions (the `segment` on `ReplayStep`: the conversation, and how many of its `CompactionEvent`s precede the step) | fresh per unit | monitors without cross-step state, which with R10 includes AISI's |
| `"conversation"` | a conversation's steps | fresh per unit | state kept per agent instance, as [sub-agents.md](sub-agents.md) recommends keying it until `store_as(scope="conversation")` exists |
| `None` | the whole sample, in log order | one, as live | a monitor whose state crosses conversations, which is the live scope |

`max_unit_steps=` splits a unit further into runs of at most that many steps, each starting with a fresh store; the docs say a stateful monitor sees less at each cut. The runner bounds how many units of one transcript run at once (`max_units`, whose default PR 5 sets by measuring), and every monitor call still passes through inspect_ai's per-model connection limit, so a wide transcript queues on the model rather than flooding it.

**Prompt caching.** A unit runs its steps in order so that each step's monitor prompt can extend the previous one's, which is what provider prefix caching rewards. Whether it does is the monitor's prompt: one rendering `step.input` breaks its prefix at a compaction, which is why the default unit ends there; one rendering `step.history`, the as-sent list, which only grows, keeps its prefix across compactions and is cached best with `split="conversation"`. Splitting further trades cache hits for wall-clock time, so it is a knob, not something the framework infers.

**State across a split** (answering open question 8, 2026-10-19). Each unit starts with a fresh store; no seeding from the previous unit, no refusal to split. A monitor cannot declare that it keeps state, so the adapter cannot refuse for it, and seeding a store from a summary is the monitor's business. The docs say what each `split` does to state, and the scan records `split` and `max_unit_steps` in the scanner's parameters, so a result can be read knowing it. With R10 the commonest state, the opening prompts as sent, is in `step.history`.

**Resume.** Scout records and resumes a scan by transcript (`_recorder/recorder.py`, `resume`): a transcript's rows are written together, so an interrupted scan re-runs only transcripts it had not finished, and a long transcript is re-run whole.

Monitor inference in replay is charged to the scan, never to anything resembling an agent limit, which is the replay form of the budget exemption approvers have in an eval.

## Building replay: the PR series

Proposed 2026-10-19 for AISI R7. Six PRs across the three repositories, in dependency order; 4 can go in parallel with 1 to 3. Each lands on `main`; sentinel tracks inspect_ai's `main` by git reference and Scout's dev install pins inspect_ai's `main`, so none waits on a release, but a PR in one repository that needs another's must wait for that one to merge. Each PR updates the R7, R4 and R10 rows of [requirements-aisi-sync-monitor.md](requirements-aisi-sync-monitor.md) and the Deferred list in [pr-series.md](pr-series.md) as it lands.

**PR 1 (inspect_ai): the log records agent turns and conversations.** The spec is [What the log records](#what-the-log-records). Scope: `ModelEvent.agent_turn`, `ModelEvent.conversation` and `CompactionEvent.conversation`; `is_agent_turn()` split out of `is_agent_generate()`; `_conversation()` moved beside it and shared by the dispatcher, the recording and the compaction; `_record_model_interaction` setting the fields and calling `record_sent` from them; the OpenAPI schema and ts-mono's viewer types regenerated. Acceptance: that section's tests, including the sufficiency test (every live `step.history` equals the fold of the recorded agent turns before the step's span), a pre-PR log loading with `None`, and a measured size change on a 300-turn sample. Decision R7-D1 sets whether the fields are written without a sentinel; R7-D2 adds two `SentinelEvent` fields only if chosen.

**PR 2 (inspect_sentinel): the types replay needs.** Scope: `AfterToolCall.output: ToolResult | None`, `None` where the host cannot see the untruncated output (replay; the sidecar, which passes `""` today; pr-series.md, Deferred, "`AfterToolCall.output` is `None` where the host can't see it"); the `ReplayStep` record; `HostNative.ask_human` returning `HumanAnswer | None`, with `human()` abstaining on `None` (R7-D5). No compatibility kept. Acceptance: tests that `human()` abstains, recording no decision, when its host returns `None`, and that the shipped protocols and helpers take an `AfterToolCall` with `output` `None` (nothing in the package reads it today; the excerpt helpers render `result`); docs say when `output` is `None`. inspect_ai's host is unaffected: it always passes a `ToolResult` and never answers `None`.

**PR 3 (inspect_ai): one step builder, two drivers.** Needs PR 1, and PR 2 on sentinel's `main`. Scope: `_sentinel/_steps.py` with the per-stage builders, the fold as a plain object, the producing-call predicate, step ids and the published decisions ([One builder for live and replay](#one-builder-for-live-and-replay)); `_dispatch.py` reduced to gathering live facts; `steps_from_events(messages, events, *, sample_uuid)`, the log driver, yielding `ReplayStep`s in live order with their segments, `agent_messages` by the cross-check, and `human_decisions` from `SentinelEvent`s (and `ApprovalEvent`s, which PR 6 completes). It is offered to sentinel as an integration function, as sentinel offers `_integration` to inspect_ai, not as author-facing API. Acceptance: the live-equals-replay test over the scripted evals named there, with the absent-field list as short as the fidelity table allows; the existing sentinel dispatch tests unchanged.

**PR 4 (inspect_scout): `Result.subject` and one row per subject.** Independent of 1 to 3. Scope: the `subject` field; expansion of a subject-bearing list into one `ResultReport` each, with `input_ids=[transcript_id, subject]`; the transcript's events and usage on its first row; an error for a list mixing results with and without a subject; Scout's generated schema and types regenerated. Acceptance: a scanner returning three subject results records three rows, each validated against a `[transcript, step]` case, with `for_validation()` selecting their transcript; sums of usage over the rows equal the scan's; a list without subjects is recorded as a resultset as today; resume never leaves a transcript's rows half written.

**PR 5 (inspect_sentinel): replay through Scout.** Needs PRs 3 and 4. Scope: the replay runner (units by `split`, `max_unit_steps`, `max_units`, the `steps=` predicate, a store and recorder per unit); the replay host (model resolution as live, `ask_human()` from recorded decisions); `Context` from the log; `inspect_sentinel.scout` with `as_scanner()` (`unit`, `stage`, `split`, `steps`), `as_reader()` and `steps_in()`; the report-to-result mapping and the transcript reduction; a docs page on replaying a monitor over logs. Acceptance: the replay-equals-read agreement test through Scout, at every stage; a scan resumed after interruption gives the same rows; a protocol replayed with recorded `human()` answers decides as live; a failed monitor makes the transcript's result an error; replaying a sample with two compactions under `split="compaction"` runs its three units concurrently (a test with a host that blocks until all three have started).

**PR 6 (inspect_ai, then inspect_sentinel): logs from before Sentinel.** Needs PR 3 (and PR 5 for the sentinel half). Scope, in inspect_ai's log driver: the conversation from the span tree when `conversation` is unset; agent turns inferred when `agent_turn` is unset and the caller opts in; `ApprovalEvent`s from the approvers named as people mapped to `human_decisions` ([Logs from before Sentinel](#logs-from-before-sentinel)). In sentinel: `as_scanner(infer_agent_turns=..., human_approvers=...)`, and the `"history": "inferred"` marker in result metadata. Acceptance: a log written by inspect_ai before PR 1 replays, with `history` `None` by default and inferred by opt-in; the classifier's disagreement with the recorded fields, measured over the PR 3 test logs with the fields removed, is stated in the PR; an `approval=` human approver's approve, reject, terminate and modify reach a later step's `human_decisions`; an escalate does not.

Not in this series: transcripts from other sources (Scout's non-Inspect transcripts, messages only, through `steps_from_messages()`), `calibrate()`, step-level labelling in Scout View, and the `agent_messages` fingerprint unless R7-D2 chooses it. They stay in workstream 10.

## What this asks of the other documents

- **`sentinel.md` and `sentinel-reference.md`:** `SentinelEvent.step_id`, recorded live by the runner. Open question 5 (calibration) and open question 11 (monitors as scores) are answered here and retire there.
- **Core:** a recorded classification of agent turns on `ModelEvent`, so the events reconstruction can rebuild one agent's `history` across compactions without a heuristic. Live, inspect_ai needs none (2026-10-14). *Specified 2026-10-19 as `agent_turn` and `conversation` on `ModelEvent` and `conversation` on `CompactionEvent` ([What the log records](#what-the-log-records)), recording the conversation rather than deriving it from the span tree.*
- **`inspect-core.md`:** the events-based reconstruction wants `ModelEvent` and `ToolEvent` in core. Not required, but it is a concrete consumer for the first open question.
- **Scout:** `Result.subject`; expansion of subject-bearing results into rows; and, later, a step-level labelling gesture in the transcript view. *`"sentinel"` is already in `EventType`, and the read-mode scanner moves to sentinel with `as_scanner()` (R7-D3), 2026-10-19.*

## Open questions

1. **Does `subject` belong on `Result` or on `ResultReport`?** The result is what a scanner author constructs, so it is the natural home; but expansion is a recorder concern, and a subject the recorder then copies into `input_ids` is a small duplication. Result, unless the recorder has a reason. *Answered 2026-10-19: `Result`, with the recorder writing `[transcript_id, subject]` as `input_ids` ("What Scout changes").*
2. **What does an expanded row's `input` hold?** Parent transcript id plus subject is the proposal; whether the view needs the step's messages materialised for rendering is a viewer question. *Answered 2026-10-19: the transcript, recorded once through pooled dedup, with `[transcript_id, subject]` as `input_ids`; the view question stays ("What Scout changes").*
3. **Should `unit="transcript"` reduce by max or by strongest?** Max suspicion for observations and strongest decision are the obvious defaults and match `sentinel.md`'s composition rules; a protocol may want the outermost layer's decision instead, which the adapter can take from its own report. *Answered 2026-10-19: maximum suspicion per dimension for a monitor, the root's strongest decision for a protocol ("Mapping `Report` to `Result`").*
4. **Replaying a bridged agent's log.** A `claude_code` or `codex` transcript has no `ToolEvent`s of Inspect's own, so replay is messages-only even from an eval log. That is fine — it is the deployment those monitors face anyway — but the fidelity table should say so. *Answered 2026-10-19: the fidelity table's bridged column. The generate stages replay as live; the tool stages, which inspect_ai does not yet run for bridged agents, come from messages.*
5. **Does replay honour `portable=True`?** The check runs when a monitor's factory is called, before it runs, with no restricted host or runtime guard (`workstreams.md`, workstream 11), so a monitor configured for replay is checked as in an eval. Open: whether replay should also refuse a `portable=False` monitor when the replay stands in for a proxy deployment.
6. **Step-level labelling in the transcript view.** Results rows suffice for a first version; a "this tool call was bad" gesture on the transcript itself is the natural way to build step sets and needs Scout View work.
7. **Splits and calibration.** Whether `calibrate()` should refuse to read a split that is also used for validation, or merely document the hygiene.
8. **Sub-section splitting and stateful monitors.** Splitting a compaction segment resets the per-transcript store at each cut. Whether the adapter should refuse to split for a monitor that declares state, seed each sub-section with a summary of the prior one, or simply document the fidelity loss. *Answered 2026-10-19: document it; each unit starts with a fresh store, and `split=None` replays the sample whole with one store, the live scope ("Cost and parallelism").*
