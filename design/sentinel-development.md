# Developing sentinels

How the monitors in a sentinel get built, measured and calibrated before they are trusted to act. The mechanism is Inspect Scout: a monitor becomes a scanner over the steps of a transcript, and Scout's validation sets, metrics and viewer become the monitor's feedback loop.

Status: design sketch. The Python is illustrative. Scout facts are measured against `inspect_scout` at 2026-09-09; Inspect facts against this branch.

**Related design documents.** `sentinel.md` defines the protocol this document develops against, and `sentinel-reference.md` is its reference form. `sentinel-deployment.md` defines the proxy deployment, whose view of a transcript turns out to be the same one offline replay has. `inspect-core.md` covers the package split this document depends on.

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
from inspect_scout.sentinel import as_scanner

validation = "exfil-steps.csv"

scan(
    scanners=[as_scanner(suspicion_monitor("openai/gpt-4o-mini"))],
    transcripts=transcripts_from("./logs").for_validation(validation),
    validation=validation,
)
```

`as_scanner()` accepts a protocol too, so `as_scanner(threshold(...))` back-tests a whole configuration the same way, and the recorded per-layer reports go along with it.

Three things fall out of this that the rest of the document works through:

- **Offline replay is a third deployment**, beside the eval and the proxy, and it has a fidelity table like the other two. Reconstruction from a messages-only transcript is *the same projection* the proxy and the bridged agent already use, so one function serves all three.
- **The recorded reports finally have a consumer.** `sentinel.md` records every report, including the ones a protocol ignored, on the argument that they are the benign distribution calibration needs. Scout is where that distribution gets read.
- **Threshold calibration is a scan plus a percentile.** `sentinel.md`'s open question 5 — where a protocol's thresholds come from — has an answer that needs no new machinery.

## Packages

`inspect_sentinel` is its own distribution, to depend only on `inspect_core` once that is extracted (see `inspect-core.md`; today it depends on `inspect_ai`). The chain is then:

```
inspect_core  ←  inspect_sentinel  ←  inspect_ai  ←  inspect_scout
```

Scout already depends on Inspect, so it gets `inspect_sentinel` transitively and the adapter lives in Scout with no new dependency. What goes where:

| Where | What | Why there |
|------------------------|------------------------|------------------------|
| `inspect_sentinel` | `steps_from_messages()`; the `ReplayStep` record; `Context` from a transcript's info; an in-process `Host`; `replay_monitor()`; `calibrate()` | needs only core types, and is also what a monitor's unit tests want; the development loop's verbs live here ([workstreams.md](workstreams.md), workstream 10) |
| `inspect_ai` | `steps_from_events()` — enrichment of the messages-only reconstruction with `view`, truncation, `tools`, `config` from `ModelEvent` and `ToolEvent` | the event types are not in core today |
| `inspect_scout` | `as_scanner()`; the `subject` field on `Result` and row expansion; the read-mode scanner over recorded `SentinelEvent`s; viewer work | does the scanning; owns validation and the view |

The middle row is pressure on `inspect-core.md`'s first open question. If `ModelEvent` and `ToolEvent` move down, the events-based reconstruction moves down with them and `inspect_sentinel` can replay an eval log at full fidelity on its own. Nothing in this document requires that; it only gets tidier.

## Reconstructing steps

### From messages

The canonical reconstruction works from `list[ChatMessage]` alone, because that is what every source has — an eval log, a Scout transcript from another agent framework, a proxy request. It is exactly the projection `agent/_bridge/_approval.py` performs on a model response and `sentinel-deployment.md` describes for a proxy:

| Stage | Reconstructed from | Fidelity |
|------------------------|------------------------|------------------------|
| `BeforeGenerate` | every message before an assistant message is that generate's `input` | `tools`, `tool_choice`, `config` absent |
| `AfterGenerate` | the assistant message is `output.choices[0].message`; `usage`, `time` absent | good |
| `BeforeToolCall` | each `tool_calls[i]` on an assistant message, with the message text and history to that point | `view` absent |
| `AfterToolCall` | the `ChatMessageTool` whose `tool_call_id` matches, as `result`; history to that point | `output` (untruncated) absent; `truncated` is `None` (unknown) |

Messages give the two histories unevenly. A transcript's messages are `history` at the end of the run; at a step, `history` is a prefix of them only where replay can establish that the scaffold had not rewritten its list since (`react`'s overflow recovery replaces it, and a sub-agent's list is not the sample's), and `None` otherwise, as under a proxy (see "Reconstructing `history` in replay" below). At the tool stages `input` (and with it `tools`) is `None` from messages alone: the history to that point is right until the first compaction and wrong after it (the model was shown a summary plus recent turns), so it is not offered as the model's input. A live step without a recorded model call gets `None` the same way, so a monitor handles both alike.

`view` and `output` being absent, and `history` being `None`, is the case `sentinel.md` already requires monitors to tolerate for proxies, so a portable monitor replays from messages unchanged. Note the last row is *better* than the proxy's: a proxy never sees the final step's result because no further request carries it, whereas a transcript is complete.

``` python
@dataclass(frozen=True)
class ReplayStep:
    id: str                 # the step id — see below
    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]  # as SentinelEvent.stage
    context: Context
    step: Step


def steps_from_messages(
    messages: list[ChatMessage], *, context: Context
) -> Iterator[ReplayStep]: ...
```

### From events

An eval log's events restore what messages lose. `ModelEvent` carries `input`, `tools`, `tool_choice`, `config` and `output` — `BeforeGenerate` and `AfterGenerate` field for field. `ToolEvent` carries `function`, `arguments`, `view`, `result` and `truncated` (the output's size and the limit it was cut to), which restores `view` and lets `result` be exactly what the model saw. It gives `AfterToolCall.truncated` as the live host set it: `Truncation(*event.truncated)` when set, and `False` when absent in a log new enough to record it, that is, one whose `eval.packages["inspect_ai"]` is 0.3.31 or later (`max_tool_output`, inspect_ai #478, which added the field); in an older log, `None`. At the tool stages the producing `ModelEvent` gives `input` and `tools`, the same lookup the live dispatcher makes. The one thing an eval log does not hold is the untruncated tool `output`, so `AfterToolCall.output` stays optional in replay too.

Events also restore `input` exactly, from `ModelEvent.input`, and let an accumulated as-sent view of an agent's run (related to AISI R10) be rebuilt per agent; it is not `history`, the scaffold's list. Walk the `ModelEvent`s and `CompactionEvent`s that share one `conversation` id in order, accumulating what each generate added and marking each compaction. That id is what `sentinel.md` asks core to record; without it the link between a pre- and a post-compaction event is a guess, because a span identifies the innermost scope rather than the agent, and a sample with several agents interleaves their events. That guess is the reconstruction difficulty the current approval-based monitor runs into, and it is why the id is a requirement rather than a convenience.

`steps_from_events()` is therefore an *enrichment* of the messages reconstruction rather than a second algorithm: it walks the same message sequence and attaches the matching event's fields where one exists. That keeps step identity (below) independent of whether events were available, which is what lets one validation set serve both kinds of source.

### Reconstructing `history` in replay

Noted 2026-10-13 with the maintainer, alongside making `step.history` optional (`None` when the host can't provide the caller's list: replay from a log, proxy hosts, steps with no scaffold conversation).

`history` is the scaffold's list *at that step*. An eval log holds only the list at the end of the sample (`sample.messages`) and, per model call, what the model was sent (`ModelEvent.input`, already `step.input`). For an append-only scaffold the step's history is a prefix of `sample.messages`, cut at the step's anchor (tool stages: the assistant message holding the call; generate stages: the last message of the call's input), found by message id.

`react` keeps `state.messages` uncompacted: compaction shapes a copy for each model call, so a compaction does not break the prefix. `react` does change its list in two places: `_remove_submit_tool` at the end (default `keep_in_messages=False`) strips the submit call and result, and `on_continue` returning an `AgentState` can replace the list. Multi-solver chains, handoffs and custom solvers can also rewrite it, and bridged agents' `sample.messages` is not the CLI's conversation. Replay therefore has to verify a prefix, not assume it.

Options:

1. **Cross-check the log (works on existing logs).** For the agent instance, every `ModelEvent` output message and every tool result must appear in `sample.messages` with the same id, in order, with the same content, up to the step's anchor; outputs are never compacted, so this holds through compaction. If it does, the prefix is the step's history; steps after a divergence (e.g. the final submit turn) get `None`. Heuristic: an edit to a message that is neither a model output nor a tool result (a user message a custom solver rewrote) is not detected.
2. **Record a history fingerprint at run time (exact, any scaffold).** Each `SentinelEvent` stores `history_len` and a hash of the step's history (tens of bytes, never the messages). Replay takes the first `history_len` messages of `sample.messages` and compares hashes: equal means exactly what the live step saw, otherwise `None`. Needs two fields on inspect_ai's `SentinelEvent` and a canonical message serialization for the hash.

Proposed: 2 for logs written after the fields exist, 1 as the fallback for older logs. Not built; part of workstream 10 / R7.

### Earlier decisions in replay

Decided 2026-10-14 (AISI R4); not built. A replayed step's `earlier_decisions` holds what the live step held: the decisions a person made through `human()` at earlier steps of the conversation. Replay reads them from the log rather than making them, which is the only way, since a replay cannot ask the person: for each step, the `SentinelEvent`s with `factory == "inspect_sentinel/human"`, `kind == "decision"`, `status == "reported"` and the step's `conversation`, from steps whose `sentinel` span ended before the `sentinel` span of the step itself began, in event order (live, the host publishes a step's decisions as its span ends, and snapshots the list when it builds a step, before its span begins), less each one withdrawn by a `superseded` event at its own step: one with the person's action at the person's `path` or the path of a layer above it, the rule `supersedes()` applies live. A `terminate` that a sibling's `decide_final()` outran is recorded `reported` and then `superseded`, possibly as a copy a composition passed up, at a layer above the person. Both events fall inside the step's span, so a later step sees the same result live and in replay, including a parallel step that began while that span was open. Each becomes an `EarlierDecision`: `stage` and `step_id` from the event; `decided` a `Reported[Decision]` rebuilt from `path` (its last segment the name), `function`, `action`, `explanation`, `message`, `modified`, `references` and `metadata`; `call_ids` the step id at a tool stage, and at `model_output` the tool calls of the assistant message with that id (its `ModelEvent` output, or `sample.messages`). Because the records are what was decided live, segments replayed concurrently each see the same list.

A log from before Sentinel records a person's approval as an `ApprovalEvent`; replay could map one from a human approver to the same record (stage `tool_call`, step id the call id, `approve` to `proceed`), but this is not planned; until then such a log gives `None`, since replay cannot tell whether a person decided anything.

### `Context`, `Host` and `Store` in replay

| Member | In replay |
|------------------------|------------------------|
| `eval` | an `EvalContext` when the source has a task and sample id; None otherwise, as in a proxy |
| `eval.task`, `eval.task_description` | from the log's eval spec; from Scout's `TranscriptInfo.task_set` and `metadata` for other sources |
| `eval.sample_id`, `eval.epoch` | `TranscriptInfo.task_id`, `task_repeat` |
| `eval.sample_input` | the sample input from the log; the first user turn otherwise |
| `eval.metadata` | task and sample metadata merged, as in an eval |
| `eval.target` | available from the log; **still opt-in**, and the adapter records that it was requested, for the same validity reason as in an eval |
| `store_as()` | backed by a fresh `Store` per transcript, so incremental monitors accumulate across the transcript's steps and nothing leaks between transcripts |
| `host` | in-process: `generate()` uses the scan's model and its usage lands in the scan's `model_usage`; `fetch()` resolves named endpoints from scan config; `terminate()` returns `False` |

### What replay cannot do

Replay reproduces the monitor's *reports*, not its *effects*. A `reject` in replay does not change the transcript; the agent already did what it did. This is the right limitation for measuring a classifier — precision and recall are about reports — but it means two things cannot be back-tested offline:

- **Interventions.** Whether rejecting call 7 would have led the agent somewhere better is a question for a live eval, not a replay.
- **Humans.** `human()`, a protocol with no children, cannot be replayed. The useful substitute is to *replay the recorded human decision* when the transcript carries a `SentinelEvent` from that `human()` instance for that step, so a `sequential` can be back-tested against the answers a person actually gave; otherwise `human()` abstains in replay.

## Step ids

Per-step validation needs each step to have an id that exists in a messages-only transcript, in an events transcript, and **live, at the moment the monitor runs** — so that a label made once serves replay and also the read mode over recorded events. Message and tool-call ids satisfy that. Event uuids do not: a `BeforeGenerate` monitor runs before its `ModelEvent` exists.

| Stage | Step id |
|------------------------|------------------------|
| `BeforeGenerate` | id of the last input message — the one that triggered the generate |
| `AfterGenerate` | id of the assistant message produced |
| `BeforeToolCall` | the tool call id |
| `AfterToolCall` | the tool call id |

One wrinkle: a rejected call followed by a regenerate produces two `BeforeGenerate` steps behind the same last message. That stage takes an ordinal suffix on repeat — `{message_id}:2` — and the other three stages are clean. Within one monitor's stage, ids are unique per transcript; a `BeforeToolCall` monitor and an `AfterToolCall` monitor both key by tool call id, but validation sets are per scanner so there is no collision.

**Consequence for `sentinel.md`:** `SentinelEvent` gains `step_id: str`, recorded by the runner live, so the read-mode scanner emits results with the same subjects the replay scanner does.

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

The adapter is a `Scanner[Transcript]`. It walks the steps itself, in order, with its own store, and returns one `Result` per eligible step. Each result carries a **subject** — the step id. A validation case whose id is a step id matches the result with that subject.

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

**Expansion into rows.** Today `_validate_scan` matches a case id against the input ids of one `ResultReport`, and a scanner returning a list is folded into a single resultset report. When a scanner returns results carrying a `subject`, each becomes its own `ResultReport` with `input_ids=[subject]`. Matching, the dataframe, the summary metrics and the results view all already work per report, so per-step behaviour falls out. A result with no subject is recorded exactly as today.

One rule keeps matching unambiguous: **subjects are unique within one scanner's results for one input.** The adapter emits one result per step for the monitor being validated; a protocol's child reports go in that result's metadata. To validate a leaf inside a protocol, wrap the leaf directly.

What `input` holds on an expanded row is a recorder decision rather than a design one: the parent transcript id plus the subject is the honest content, the view navigates to the transcript and highlights the message or tool call, and the pooled-dedup path already exists to avoid copying a transcript per step.

## Mapping `Report` to `Result`

| Report | `value` | `answer` | `metadata` | `label` | `subject` |
|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| `Observation` | `suspicion` — float or dict | — | monitor metadata | instance path | step id |
| `Decision` | flagged: `action` in `reject`, `modify`, `terminate` | `action` | `audit`, `modified`, child reports | instance path | step id |
| abstained | `None` | `None` | — | instance path | step id |

Three things the table encodes:

- **A dict suspicion validates per key** through Scout's existing dict-target path, so a multi-dimension monitor gets per-dimension precision and recall without any adapter logic.
- **The boolean flagged value is what makes binary labels work.** A per-step validation set will label steps the monitor abstained on, and abstention produces no report. The adapter emits a result for every step at the monitor's stage, `None` on abstain, and a label of `false` compares equal to "not flagged" only if the value is boolean. Teaching the `eq` predicate that `None` is negative was the alternative; mapping to a boolean keeps the predicate honest and leaves the action in `answer` for anyone sweeping on it.
- **`escalate` is not flagged.** It is "I decline to decide", not a verdict. Inside a `sequential` it also means "worth a person's look", which a reviewer might want to count; `answer` carries it for that. This is a default that is cheap to get wrong, and it is recorded as a decision rather than left implicit.

The `explanation` and `references` map directly from the report's own fields — an LLM monitor using Scout's `message_numbering` in its prompt gets cites for free.

## Two modes: replay and read

Both consume the same validation set, because both key results by step id.

| Mode | What runs | When |
|------------------------|------------------------|------------------------|
| **replay** | reconstruct steps, run the monitor, one result per step | you changed the prompt, the model, or the monitor's logic |
| **read** | walk the transcript's recorded `SentinelEvent`s, one result per event | you want to know how the monitor behaved in the run it was attached to |

Read mode costs no inference and is what makes "run the eval observe-only, then look" a real workflow rather than a slogan. It is also the only mode that sees what the protocol actually did with each report, from the decision events at the parent paths, so "how often did `resample` substitute a draw" is a read-mode query. Scout's `EventType` literal gains `"sentinel"` so the transcript loader can filter for these events.

Read mode also gives the honest check on replay itself: replay the monitor over a log it was attached to, and the two modes should agree wherever the monitor is deterministic. Where they do not, the reconstruction is wrong, and that is a test worth shipping.

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

An LLM monitor over a corpus is one inference call per eligible step, which is many more calls than one per transcript. Three things keep it tolerable:

- **Incremental monitors are cheaper to replay**, not just to run — the design's insistence on classifying the newest turn and folding into state means replay cost is linear in steps rather than quadratic, and the adapter's per-transcript store is what makes that work.
- **Stage filtering happens before loading.** A `BeforeToolCall` monitor's scanner declares the content it needs from the monitor's stage annotation, so Scout loads only what the stage reads.
- **Parallelism is across transcripts** by default, as it is for any transcript scanner, and within a transcript the walk is sequential. Sequential is not only a statefulness constraint; it is also how the monitor's calls get provider prefix caching, since each step's prompt extends the previous one's. Fanning out within a transcript trades that cache hit rate for wall-clock time, so it should be a knob rather than a property the framework infers.
- **Split on compaction by default; split further on request.** Compaction rewrites the prefix, so nothing before a compaction boundary is cacheable against anything after it, and the segments between boundaries are the natural units to run concurrently at no caching cost. Each segment is then walked sequentially. An option on `as_scanner()` (a maximum segment length, or a target number of sub-sections) splits a segment into sub-sections that run concurrently, buying more concurrency at the cost of prefix-cache reuse; the author picks the point on that curve for their corpus and budget. A sub-section starts with a fresh store, which is the same statefulness caveat as before: a stateful monitor sees less history at each cut, and the option should say so.

Monitor inference in replay is charged to the scan, never to anything resembling an agent limit, which is the replay form of the budget exemption approvers have in an eval.

## What this asks of the other documents

- **`sentinel.md` and `sentinel-reference.md`:** `SentinelEvent.step_id`, recorded live by the runner. Open question 5 (calibration) and open question 11 (monitors as scores) are answered here and retire there.
- **Core:** `conversation` on `ModelEvent` and `CompactionEvent`, so the events reconstruction can rebuild one agent's as-sent history across compactions without a heuristic.
- **`inspect-core.md`:** the events-based reconstruction wants `ModelEvent` and `ToolEvent` in core. Not required, but it is a concrete consumer for the first open question.
- **Scout:** `Result.subject`; expansion of subject-bearing results into rows; `"sentinel"` in `EventType`; the read-mode scanner; and, later, a step-level labelling gesture in the transcript view.

## Open questions

1. **Does `subject` belong on `Result` or on `ResultReport`?** The result is what a scanner author constructs, so it is the natural home; but expansion is a recorder concern, and a subject the recorder then copies into `input_ids` is a small duplication. Result, unless the recorder has a reason.
2. **What does an expanded row's `input` hold?** Parent transcript id plus subject is the proposal; whether the view needs the step's messages materialised for rendering is a viewer question.
3. **Should `unit="transcript"` reduce by max or by strongest?** Max suspicion for observations and strongest decision are the obvious defaults and match `sentinel.md`'s composition rules; a protocol may want the outermost layer's decision instead, which the adapter can take from its own report.
4. **Replaying a bridged agent's log.** A `claude_code` or `codex` transcript has no `ToolEvent`s of Inspect's own, so replay is messages-only even from an eval log. That is fine — it is the deployment those monitors face anyway — but the fidelity table should say so.
5. **Does replay honour `portable=True`?** The check runs when a monitor's factory is called, before it runs, with no restricted host or runtime guard (`workstreams.md`, workstream 11), so a monitor configured for replay is checked as in an eval. Open: whether replay should also refuse a `portable=False` monitor when the replay stands in for a proxy deployment.
6. **Step-level labelling in the transcript view.** Results rows suffice for a first version; a "this tool call was bad" gesture on the transcript itself is the natural way to build step sets and needs Scout View work.
7. **Splits and calibration.** Whether `calibrate()` should refuse to read a split that is also used for validation, or merely document the hygiene.
8. **Sub-section splitting and stateful monitors.** Splitting a compaction segment resets the per-transcript store at each cut. Whether the adapter should refuse to split for a monitor that declares state, seed each sub-section with a summary of the prior one, or simply document the fidelity loss.
