# Log Viewer – Inspect Sentinel

## Overview

Every observation and decision a sentinel makes is recorded in the sample’s transcript as a `SentinelEvent`, next to the tool call or model call it was about. Inspect View shows these events inside the tool call they belong to, so as you read a sample you can see:

1.  Which tool calls the sentinel checked, and what decision took effect.
2.  The score and explanation from every monitor and protocol that reported.
3.  What the agent was told when a tool call was rejected.
4.  The prompts the monitor sent to the monitor model, and its answers.

The same events are available in code through [read_eval_log()](https://inspect.aisi.org.uk/reference/inspect_ai.log.html#read_eval_log), for analysis across samples and logs.

## Viewing a Checked Tool Call

Run an eval with a sentinel, then open the log viewer:

``` bash
inspect view
```

Open a sample and look at its tool calls. Each tool call the sentinel checked shows the results inside the tool call’s card. Results from before the tool call ran appear with its input, and results from after it returned appear below its output.

### Summary Row

The first row names the action that took effect: **Proceeded**, **Rejected**, **Modified**, **Terminated**, or **Escalated**. It also shows the explanation, the suspicion score if there is one, and the instance path of the protocol that decided, when that protocol is not at the top. A step where only monitors reported, as under `observe_only()`, shows **Observed** with the monitors’ scores.

A rejected tool call is tinted, so it stands out as you scroll. A decision made with `audit=True` shows a **flagged** marker.

### Checks

When more than one monitor or protocol reported, the summary row ends in an **N checks** pill. Expand it for a tree of every monitor and protocol by instance path, such as `network`, `llm` and `llm/suspicion`, with the top-level decision shown as **(top)**. Each row shows its decision or score. A score with several dimensions shows its first dimension and a count of the rest. Click a row to see its full explanation and every score.

Rows for monitors and protocols that did not report are muted and labelled with their status: **cancelled**, **bypassed** or **superseded** (see [Statuses](#statuses)). A monitor that raised an exception shows as a failed row with the error.

References such as `[M22]` in an explanation link to the message or event they cite.

### Rejected Tool Calls

When a tool call is rejected, the tool does not run. In place of its output, the card says **Did not run**, followed by the message the agent received as the tool result. For a `terminate`, the card says that the sample was terminated.

For a `modify`, the card shows the tool call that ran, with the agent’s original tool call struck through.

### Monitor Model Calls

Below the checks, a line such as **1 monitor model call** expands to the model calls that LLM monitors made at this step: the prompt sent to the monitor model and its answer. These calls do not appear in the agent’s conversation.

## SentinelEvent

`SentinelEvent` is part of Inspect, in `inspect_ai.event`. A step produces one event for every monitor’s observation and every protocol’s decision, including decisions that did not take effect. Within a step, the monitors and protocols under a protocol are recorded before it, so the top-level decision comes last.

| Field | Description |
|----|----|
| `factory` | Registry name of the factory that made the monitor or protocol, such as `no_network` or `inspect_sentinel/threshold`. |
| `path` | Instance path, such as `llm/suspicion`. The top of the sentinel is the empty string. |
| `function` | Name of the function that reported, for a monitor or protocol whose factory returns several. `None` for `cancelled` and `bypassed` events. |
| `step_id` | The step examined. For a tool stage, the tool call’s id. For a generate stage, the id of the model call’s first [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent), shared by both generate stages. |
| `conversation` | The agent conversation the step belongs to. |
| `stage` | `"model_input"`, `"model_output"`, `"tool_call"` or `"tool_result"`: the `BeforeGenerate`, `AfterGenerate`, `BeforeToolCall` and `AfterToolCall` stages. |
| `kind` | `"observation"` from a monitor, `"decision"` from a protocol. |
| `status` | What happened to the report (see [Statuses](#statuses)). |
| `suspicion` | The suspicion score of an observation: a number from 0.0 to 1.0, or a dict of scores by dimension. |
| `action` | The action of a decision: `"proceed"`, `"reject"`, `"modify"`, `"terminate"` or `"escalate"`. |
| `audit` | Whether the report flagged this step for extra oversight. |
| `modified` | The replacement tool call, for a `modify` decision. |
| `message` | What the agent was told, for a `reject` decision. |
| `explanation` | Why, for the log. The agent never sees it. |
| `references` | Messages and events the report cites, linked from references such as `[M22]` in the explanation. |
| `error` | The exception a monitor raised, as its type and message, for an `error` event. |
| `metadata` | The report’s own metadata. |

## Statuses

The `status` field says what happened to a report:

| Status | Meaning |
|----|----|
| `reported` | The monitor or protocol reported. The event carries its observation or decision. |
| `cancelled` | The monitor or protocol was cancelled before it finished, because another one beside it decided `terminate`, called `decide_final()`, or raised. The event records no report. |
| `bypassed` | A `decide_final()` from a protocol below passed this protocol without running its decision logic. The event records no report. |
| `superseded` | A decision from `decide_final()` that lost to another final decision, or to an exception, in the same layer. It did not take effect. |
| `error` | A monitor raised an exception instead of reporting, and `error` holds it. At the empty path, it records that the sentinel itself raised at this step. |

A decision with status `reported` did not necessarily take effect. In a `concurrent()`, every protocol’s decision is `reported`, and only the strongest takes effect, through the decision at the top. When a protocol calls `decide_final()`, the step ends with its decision. The protocols above it are recorded as `bypassed`, and then the final decision is recorded at its own path.

## Reading Events

To work with the events in code, read the log with [read_eval_log()](https://inspect.aisi.org.uk/reference/inspect_ai.log.html#read_eval_log) and select the `SentinelEvent`s from each sample’s events. This script prints every observation and decision from a run of the [Tutorial](./tutorial.html.md)’s `llm_suspicion.py`, grouped by step:

    sentinel_events.py

``` python
import sys

from inspect_ai.event import SentinelEvent
from inspect_ai.log import read_eval_log

log = read_eval_log(sys.argv[1])
for sample in log.samples or []:
    steps: dict[tuple[str, str], list[SentinelEvent]] = {}
    for event in sample.events:
        if isinstance(event, SentinelEvent):
1            steps.setdefault((event.stage, event.step_id), []).append(event)
    for (stage, step_id), events in steps.items():
        print(f"{sample.id} {stage} {step_id}")
        for event in events:
2            result = event.action if event.kind == "decision" else event.suspicion
            print(f"  {event.path or '(top)'}: {event.status} {result}")
            if event.explanation:
                print(f"    {event.explanation}")
```

1  
A step is identified by its stage and step id. Tool calls that run in parallel can interleave their events, so the script groups events by that pair rather than by position.

2  
A decision has an `action` and an observation has a `suspicion`. Events with status `cancelled` or `bypassed` have neither, so they print `None`.

Run it on a log file:

``` bash
python sentinel_events.py logs/<file>.eval
```

For a rejected tool call in the `cleanup` sample, the output looks like this:

``` text
cleanup tool_call call_8Fq2
  suspicion: reported {'files': 0.95, 'network': 0.0}
    The command deletes /var/log/app, outside /work.
  (top): reported reject
    suspicion 0.95 from suspicion: The command deletes /var/log/app, outside /work.
```

Note that reading `sample.events` loads whole samples into memory. For large logs, use [read_eval_log_samples()](https://inspect.aisi.org.uk/reference/inspect_ai.log.html#read_eval_log_samples) to read one sample at a time; see [Log Files](https://inspect.aisi.org.uk/eval-logs.html) in the Inspect documentation.

## Monitor Token Usage

Model calls made through `context.host.generate()` or `context.host.score()` are recorded separately from the agent’s. They do not count toward the sample’s token, turn or working-time limits, but cost limits still apply.

The eval summary reports token usage by model role, so the `monitor` role’s usage appears apart from the agent’s. In code, read it from `role_usage` on the eval and on each sample:

``` python
from inspect_ai.log import read_eval_log

log = read_eval_log("logs/<file>.eval")
print(log.stats.role_usage.get("monitor"))
for sample in log.samples or []:
    print(sample.id, sample.role_usage.get("monitor"))
```

A monitor’s model call is recorded under the role it used, `monitor` by default. A call that passes a `model` and no `role` is recorded under the role `sentinel`.
