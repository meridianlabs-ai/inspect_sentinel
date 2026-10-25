# Inspect Sentinel

## Overview

Inspect Sentinel is a package for defining, developing, and deploying sentinels: functions that watch agents and intervene in their execution. Key design elements include:

1.  A single interface that encompasses monitoring tool calls and model generations, with execution either before or after model generations and tool calls.
2.  A division between monitors, which observe and report a suspicion score, and protocols, which decide on an action similar to the current Inspect tool call approval interface (e.g. proceed, modify, reject, escalate, terminate). A protocol with no monitors is a simple rule.
3.  Support for layering control protocols (thresholds, audit budgets, resampling, human review, etc.) on top of monitors.
4.  Core interfaces decoupled from the Inspect runtime so that monitors can be deployed elsewhere (e.g. in network proxies).
5.  Development tools for validating, calibrating, and backtesting monitors on test or production transcripts.

The purpose of this document is to solicit feedback on the core concepts and Python API used to implement monitors and protocols.

On naming: *sentinel* is the umbrella, for the feature and for a configured tree passed as `Task(sentinel=)`, which is always a protocol with its monitors. Inside it, *monitor* and *protocol* keep the senses the AI-control literature (and ControlArena) give them: the monitor scores, the protocol decides. An earlier draft used "monitor" for both the umbrella and the scorer, and readers took `Task(monitor=)` to mean "the monitor plus the protocol"; renaming the umbrella keeps the literature's pair intact.

## Monitors and Protocols

There are two registerable types. A **monitor** observes: it returns an `Observation` (a suspicion score) or nothing, and can never act. A **protocol** decides: it returns a `Decision` (an action), optionally after consulting monitors. The simplest protocol is a rule with no monitors. For example, here is a rule that prohibits the use of `curl` in any tool call:

``` python
@protocol
def no_curl() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall, escalations: Escalations) -> Decision | None:
        if "curl" in str(step.call.arguments):
            return Decision.reject("uses curl", message="network access is not allowed")
        return None

    return decide
```

A protocol's third parameter holds the escalations of the links before it in a `sequential()` (see Composition below); it is empty here. A monitor takes only `(context, step)`, so it never sees what another monitor or protocol decided.

And here is a monitor that scores each tool call, wrapped in the `threshold()` protocol that decides what the score means:

``` python
@monitor
def suspicion_monitor(model: str | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        ...
        return Observation.score(score, explanation)

    return check

Task(sentinel=threshold(suspicion_monitor(), reject_at=0.8))
```

Both run before the tool call executes (the `step` they take is `BeforeToolCall`). There are four stages, the step types a monitor or protocol could choose to target; the two tool stages are built, and the generate stages are built (with `proceed`, `escalate` and `terminate`; `reject` and `modify` come later):

| Stage | Fires |
|------------------------------------|------------------------------------|
| `BeforeGenerate` | Before each generate, before the cache lookup |
| `AfterGenerate` | After each generate, before the output reaches the agent |
| `BeforeToolCall` | Before a tool call executes |
| `AfterToolCall` | After a tool call executes, before the model sees the result |

Each function handles one stage and never dispatches on the payload type; a protocol that only forwards the step, such as `concurrent`, annotates `Step` to run at every stage.

Every step carries both `input`, exactly what the model was sent for that generate (at the tool stages `None` when no model call produced the call or its record is gone), and `agent_messages`, the outermost running `@agent` agent function's own `AgentState.messages`, read live by reference at all four stages. The scaffold may curate its list to leave hidden side-task instructions out and may keep retries in it; it is `None` without a running decorated agent function, for bridged agents, in replay, and under sidecar or proxy hosts. A compaction does not drop turns from the list and adds its summary message (overflow recovery also drops the reply that overflowed); `input` carries the compacted request. With `truncation` set, `react`'s overflow filter trims this list when forced compaction is not available or does not help (no compaction, a model agent as `model=`, a failed forced compaction, or a second overflow right after one), while `history` keeps the turns. Monitors must treat a present list as read-only. Scaffolds edit what they send, so neither list is derivable from the other. A third field, `history`, holds everything the model was sent over the agent instance's run (AISI R10): each message once in the order sent, edits kept as versions, `None` when the host cannot produce it (replay, for now); beside a proxy the sidecar builds it from the requests it sees, with the differences in [sidecar.md](sidecar.md#8-what-a-sentinel-is-given). The tool stages also carry the `tools` that producing call offered, `None` exactly when `input` is, and `AfterToolCall.truncated`, the cut the host made to the result as the log's `ToolEvent` records it: a `Truncation`, `False` for no cut, or `None` when the host cannot tell (AISI R3). Each step also carries a `conversation` id that links one agent's steps across compactions when several agents share a sample. And each carries `human_decisions`, the answers a person gave through `human()` at earlier steps of that conversation, so a monitor can tell which calls a person already approved (AISI R4); monitors' scores and other protocols' decisions are never in it, so monitors stay independent.

The sentinel API will include a variety of view functions that help with presenting task context and agent actions to the monitor (e.g. like the `messages_as_str()` function in Inspect Scout).

### Actions

A protocol's `Decision` carries one of the following actions:

| Action      | Effect                                                       |
|------------------------------------|------------------------------------|
| `proceed`  | No change to the exchange                                    |
| `modify`    | Replace the step's subject with `modified`                   |
| `reject`    | Reject the action, notifying the agent that it was rejected. |
| `terminate` | End the sample                                               |
| `escalate`  | Pass the decision to the next decider                        |

A decision's `explanation` is for the log only; the agent never sees it, since an agent told why it was flagged can learn to evade. The one thing the agent reads is `message`, which only a `reject` may carry for now (`Decision.reject(explanation, message=...)`); a reject without one gets the host's default "not approved" text.

A monitor never returns an action. It records an `Observation` for a protocol to act on (see [Protocols](#protocols) below), which keeps every threshold in the protocol where it can be calibrated, and makes every monitor dry-runnable and backtestable before anything acts on it.

### Context

In addition to the data associated with a tool call or generation, monitors and protocols are passed a `Context` that includes a place to store state, an interface to the sentinel host (which could either be Inspect AI or a proxy server), and, in an eval, task and sample information under `eval`. A proxy answers requests that have no task, sample or epoch, so there `eval` is None:

``` python
@dataclass(frozen=True, kw_only=True)
class EvalContext:
    task: str
    """Task name."""

    task_description: str | None
    """What agents in this setting are asked to do: the task's `description`. None when the task sets none or the host does not provide it."""

    sample_id: str | int
    """Dataset sample identifier."""

    epoch: int
    """Epoch of the sample: 1 unless the task runs several."""

    sample_description: str | None
    """What is distinctive about this instance: the sample's `description`. None when the sample sets none or the host does not provide it."""

    sample_input: str | list[ChatMessage]
    """The sample's input: the assignment the agent was given."""

    metadata: dict[str, Any]
    """Task metadata merged with sample metadata."""

    target: Target | None = None
    """The expected answer (opt-in, not built yet; always None for now)."""


@dataclass(frozen=True, kw_only=True)
class Context:
    path: str
    """Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer."""

    host: Host
    """Interface to the host; inference and asking a person for now, `fetch` planned."""

    eval: EvalContext | None
    """The task and sample being run. None outside an Inspect eval."""

    def store_as(self, model_cls: Type[SMT]) -> SMT:
        """Typed view of this instance's per-sample state, namespaced by its path.

        Other scopes (across epochs, the task, global) are planned."""
```

## Protocols {#protocols}

A protocol decides what happens at a step. It may consult monitors, other protocols, or nothing at all: `no_curl` above is a protocol with no children. Protocols own:

- **Thresholds**, calibrated against benign score distributions.
- **Escalation** to more capable judges or humans for review.
- **Aggregation** of reports from several monitors.
- **Control actions**: accept, audit, trusted edit, resample. These are things a protocol does with a score, not values a monitor can return.
- **Pipeline stages** such as paraphrasing agent output before monitors see it, or injecting honeypot inputs to catch a colluding monitor.

Built-in protocols cover the common compositions and basic control scenarios. `concurrent()`, `sequential()`, `observe_only()`, `threshold()` and `human()` ship today; `resample()` is designed but not yet built:

| Protocol | Description |
|------------------------|------------------------------------------------|
| `concurrent()` | Runs every child concurrently; the strongest decision wins (`terminate` \> `reject` \> `modify` \> `escalate` \> `proceed`). What a list containing a protocol resolves to. Use for independent guards that must all be satisfied. |
| `sequential()` | Runs children in order until one decides; first non-`escalate` wins, and escalations are handed to the next link so a human at the end sees who asked and why. Use for a cheap rule ahead of an expensive judge. |
| `observe_only()` | Records every observation and acts on none. How monitors run with nothing acting, since `Task(sentinel=)` takes no bare monitors, and how a benign score distribution gets collected. |
| `threshold()` | Acts on scores: `escalate_at`, `reject_at` and `terminate_at` are floors, and the strongest action any score reaches decides; each is one number or a mapping per dimension (`{"audit.sabotage": 0.6, "*": 0.9}`). Escalations carry every monitor's score to a `human()` after it, in the `escalations` it is handed. The smallest protocol, and the one calibration feeds. |
| `human()` | Asks a person whenever it is reached, at the stages named in its required `stages` (`tool_call`, `tool_result`, `model_output`), through `Host.ask_human(step, escalations, choices)`, passing on the escalations it was handed. The answer is an ordinary decision: it ends a `sequential()`, and beside other protocols in a `concurrent()` it is one vote; a reject's typed reason is the `message` the agent reads. Put it at the end of a `sequential()` to ask when the links before it escalate, or abstain, as a `threshold()` does when none of its monitors reported. |
| `resample()` | Above a threshold, draws further completions, scores each, and executes the least suspicious; escalates if none clears. |

### Writing a Protocol

A protocol's body is ordinary code around the runner (`run_monitors()`, `run_protocols()`, `run_children()`), which runs the children concurrently and records their reports. Thresholds, budget arithmetic, and aggregation are up to the protocol.

For example, `threshold`, the smallest shipped protocol: it reads its children's scores and acts when one reaches its threshold. In outline, with configuration-time validation left out:

``` python
@protocol
def threshold(
    monitors: Monitor | MonitorGroup | Monitors,
    *,
    escalate_at: float | Mapping[str, float] | None = None,
    reject_at: float | Mapping[str, float] | None = None,
    terminate_at: float | Mapping[str, float] | None = None,
) -> Protocol | ProtocolGroup:
    # strongest first; a float is {"*": float}; the ones not given are left out
    levels = [
        ("terminate", thresholds(terminate_at)),
        ("reject", thresholds(reject_at)),
        ("escalate", thresholds(escalate_at)),
    ]

    async def decide(context: Context, step: Step, escalations: Escalations) -> Decision | None:
        observations = await run_monitors(monitors, context, step)
        if not observations:
            return None
        for action, limits in levels:
            if crossed := crossing(observations, limits):
                return Decision(
                    action=action,
                    explanation=crossed.explain(),
                    references=references(observations),
                    metadata={"observations": scores(observations)},
                )
        return Decision.proceed()

    # one function per stage its monitors watch; before tool calls only, with
    # reject_at
    return per_stage(decide, monitors)
```

## Deployment

Sentinels can be developed, refined, and deployed within Inspect AI, and for many applications this will be sufficient. However, in some scenarios you may want to run sentinels in a network proxy in front of the model API, where the agent cannot bypass them and where they cover all agent traffic (e.g. coding agents). A proxy has no `Task`, no sample, and no `Store`, and the traffic is not necessarily Inspect's, so the sentinel interface is designed so that the same monitor runs in both places.

### Proxy Steps

A proxy sits on one HTTP exchange: a request and a response. It never sees a tool execute, because tools run client-side after the response is delivered. So two of the four stages are exact, and the tool stages are projections:

| Stage | In a proxy | Fidelity |
|------------------------|------------------------|------------------------|
| `BeforeGenerate` | The request body | Full |
| `AfterGenerate` | The response body | Full |
| `BeforeToolCall` | `tool_use` blocks in the response | Good (no `view`) |
| `AfterToolCall` | `tool_result` blocks in the *next* request | Lossy (the last step is never seen) |

### Execution

The goal is to enable two deployment modes for proxy servers:

1.  Sidecar (a Python service beside the proxy); or
2.  Embedded (CPython compiled to WASM inside the proxy)

Both modes depend on sentinels being lightweight. `inspect_sentinel` is to depend only on `inspect_ai.core`, a subpackage of inspect_ai holding the wire types (`ChatMessage`, `ModelOutput`, `ToolCall`, `ToolInfo`) and, later, the registry primitives, so that a monitor author, a proxy, or a WASM build never pulls in the full Inspect framework. Today it imports the wire types from `inspect_ai.core` and the registry, store and configuration from the rest of `inspect_ai`, and depends on `inspect_scout` for message rendering until that moves into inspect_ai.

Even if you are deploying sentinels inside Inspect AI to start with, sentinel will check that your code travels well to a proxy without Inspect dependencies. Monitors and protocols are portable by default, and `@monitor(portable=False)` opts out. Calling the factory checks its code for common mistakes that would fail in a proxy, such as calling `get_model()` or reading environment variables; there is no check while it runs.

## Development

Inspect Scout will provide a feedback loop for monitor development (validation sets with labelled cases, precision/recall/F1, a results data frame, and a viewer for labelling). This will work with both Inspect logs and transcripts from other sources (e.g. proxy servers, coding agent sessions, etc.). The `as_scanner()` function turns any monitor into a scanner that replays it over the steps of each transcript.

``` python
from inspect_scout import scan, transcripts_from
from inspect_scout.sentinel import as_scanner

scan(
    scanners=[as_scanner(suspicion_monitor("openai/gpt-4o-mini"))],
    transcripts=transcripts_from("./logs"),
    validation="exfil-steps.csv",
)
```

The typical workflow:

1.  Replay the monitor over a corpus and review what it flags in Scout View.
2.  Label steps as should-flag or should-not-flag directly from the results view, building a validation set with dev and test splits. Each step has a stable id (the tool call id, or the message id), so labels apply to replayed steps and to the `SentinelEvent`s an eval recorded.
3.  Validate using precision, recall, F1, and balanced accuracy over the dev split. Change the prompt, bump the version, re-scan, compare.
4.  Calibrate by replaying an `Observation` monitor over known-benign transcripts. The score column of the results is the benign distribution, so a budget (e.g. "audit the top 1%") becomes a threshold by percentile.


