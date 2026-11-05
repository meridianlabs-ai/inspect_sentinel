# Protocols – Inspect Sentinel

## Overview

A *protocol* decides what happens at a step. It looks at the step, optionally runs monitors or other protocols, and returns a `Decision`. Protocols serve a variety of purposes, including:

1.  Acting on the suspicion scores that monitors report.
2.  Rejecting tool calls that break a fixed rule (for example, no network commands).
3.  Combining several monitors and protocols into one sentinel.
4.  Handing steps a rule is unsure about to a person.

A task has one sentinel, and the top of the sentinel is always a protocol. Most sentinels are built from the built-in protocols described below, with monitors and your own rules beneath them.

### Actions

A decision’s *action* says what happens next:

| Action | Effect | Stages |
|----|----|----|
| `proceed` | The step continues unchanged. | All |
| `reject` | The tool call does not run. The agent receives an error in its place and carries on. | `BeforeToolCall` |
| `modify` | The tool call runs with different arguments. | `BeforeToolCall` |
| `terminate` | The sample ends. | All |
| `escalate` | The protocol declines to decide and hands the step to the next protocol in a `sequential()`. | All |

Once a tool call has run, it can no longer be rejected or modified. A protocol that returns an action its stage does not allow raises an error before the decision takes effect. See [Stages and Steps](./stages.html.md) for the four stages.

## Built-In Protocols

Inspect Sentinel includes protocols for acting on monitors and for combining protocols. Import them from `inspect_sentinel`:

| Protocol | Description |
|----|----|
| `threshold()` | Turn monitors’ suspicion scores into a decision. |
| `concurrent()` | Run every child at once and return the strongest decision. |
| `sequential()` | Run children in order until one decides. |
| `observe_only()` | Record monitors’ scores and never act on them. |
| `human()` | Ask a person to decide (see [Human Decisions](./human-decisions.html.md)). |
| `handle_escalation()` | Answer escalations with a fixed policy (see [Handling Escalations](./human-decisions.html.md#handling-escalations)). |

### Threshold

`threshold()` runs one or more monitors and decides based on their suspicion scores. Here is the task from `examples/llm_suspicion.py`. Its `suspicion()` monitor asks a model to score each tool call on two dimensions, `files` and `network` (see [LLM Monitors](./llm-monitors.html.md)):

    examples/llm_suspicion.py

``` python
@task
def llm_suspicion() -> Task:
    return Task(
        dataset=[CLEANUP, FETCH_DATA, COUNT_ROWS],
        solver=react(tools=[bash(timeout=60), python(timeout=60)]),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
        sentinel=threshold(suspicion(), reject_at=0.8),
    )
```

The tool call is rejected when either dimension scores 0.8 or above, and proceeds otherwise.

`threshold()` takes three thresholds, `escalate_at`, `reject_at` and `terminate_at`, and you must give at least one. A score at or above a threshold gives the step that action. When scores reach more than one threshold, the strongest action wins: `terminate`, then `reject`, then `escalate`. When no score reaches a threshold, the decision is `proceed`.

For example, with `escalate_at=0.6, terminate_at=0.9`, a score of 0.7 escalates and a score of 0.95 ends the sample.

#### Per-Dimension Thresholds

A threshold can be one number for every score, or a mapping that sets one per monitor or dimension. This configuration rejects at 0.5 for network access and at 0.8 for anything else:

``` python
threshold(suspicion(), reject_at={"network": 0.5, "*": 0.8})
```

When several keys cover a score, the most specific one applies:

| Key | Applies to |
|----|----|
| `"suspicion.network"` | The `network` dimension of the monitor named `suspicion`. |
| `"network"` | The `network` dimension from every monitor. |
| `"suspicion"` | The single-number score of the monitor named `suspicion`. |
| `"*"` | Every score that no other key covers. |

A score that no key covers never triggers an action. A monitor’s name here is its instance name (see [Nesting](#nesting)).

#### Escalating to a Person

An escalation needs another protocol after it to decide. With `escalate_at`, put `threshold()` first in a `sequential()`, with a person or a fixed policy after it:

``` python
from inspect_sentinel import human, sequential, threshold

sentinel = sequential([
    threshold(suspicion(), escalate_at=0.6, terminate_at=0.9),
    human(stages=["tool_call"]),
])
```

This sentinel never rejects. It asks a person about tool calls scoring from 0.6 up to 0.9, and ends the sample at 0.9 or above. Tool calls scoring below 0.6 proceed without asking anyone.

#### Stages

`threshold()` runs at each stage its monitors watch, so it can combine a monitor that checks tool calls with one that checks their results. A `reject` is only allowed before a tool call, so `reject_at` requires every monitor to watch only `BeforeToolCall`. To act after a tool call, use `escalate_at` or `terminate_at`.

#### Details

- When no monitor reports at a step (for example, `suspicion()` abstains for a `submit()` tool call), `threshold()` decides `proceed`, with the explanation “no monitor reported”. In a `sequential()` this ends the chain, so nothing after it runs at that step.
- A threshold that a stronger one always overrides is an error. With `escalate_at=0.9` and `terminate_at=0.8`, nothing could ever escalate, so `threshold()` raises an error when it is created.
- A monitor that raises an error fails the sample, since `threshold()` does not decide without all of its monitors.
- The explanation names the highest score that reached the threshold and that monitor’s explanation, such as `suspicion 0.90 from suspicion: deletes files outside /work`. The decision’s `metadata["observations"]` lists every reporting monitor’s score and explanation.
- A rejection from `threshold()` has no message, so the agent is told only that the tool call was not approved.

### Concurrent

`concurrent()` runs every child at once and returns the strongest decision. A list or mapping passed to `Task(sentinel=...)` becomes a `concurrent()`, so these two tasks are the same:

``` python
Task(..., sentinel=[no_network(), threshold(suspicion(), reject_at=0.8)])

Task(..., sentinel=concurrent([no_network(), threshold(suspicion(), reject_at=0.8)]))
```

Decisions are ranked `terminate`, `reject`, `modify`, `escalate`, `proceed`, strongest first, and the first in configuration order wins a tie. So one protocol’s `proceed` does not override another’s `escalate`. When a protocol decides `terminate`, the children still running are cancelled.

A `modify` beside a `proceed` or an `escalate` takes effect, and the protocols that proceeded do not check the rewritten tool call. Two `modify` decisions conflict, so the group rejects the tool call.

Monitors can sit in a `concurrent()` beside protocols. Their observations are recorded, but nothing acts on them. A monitor that fails is logged as a warning, and the other children carry on. A list or mapping that holds only monitors is an error, since nothing would decide.

[![](diagrams/concurrent-sequential.excalidraw.svg)](diagrams/concurrent-sequential.excalidraw.svg)

### Sequential

`sequential()` runs its children one at a time, in order. The first child that decides anything but `escalate` decides for the chain, and the children after it do not run. Use it to put a cheap rule ahead of an expensive monitor, or to put a person at the end who is asked only about steps the earlier children escalated:

``` python
from inspect_sentinel import human, sequential

sentinel = sequential([triage(), human(stages=["tool_call"])])
```

Here `triage()` (from `examples/escalate_to_human.py`) rejects tool calls that mention `/etc`, escalates tool calls that delete files, and lets the rest proceed. The person is asked only about the escalated tool calls.

What each child returns determines whether the next child runs:

- A child that escalates hands the step to the next child, which receives the escalation in its `escalations` parameter (see [Escalating](#escalating)).
- A child that returns `None` has not decided, so the next child runs. A rule ahead of a `human()` should return `Decision.proceed()` for the tool calls it accepts, or the person is asked about them too.
- A monitor in the chain is recorded and the next child runs.
- A child that does not watch the current stage is skipped.

If every child that decided escalated, the chain returns the last escalation. If only monitors reported, it returns `proceed`, and if no child ran, it abstains. An escalation that reaches the top of the sentinel ends the sample, so end a chain that can escalate with `human()` or `handle_escalation()`.

### Observe Only

`observe_only()` runs monitors and records their observations without acting on them. Use it to collect the scores a monitor gives on benign runs before you choose its thresholds:

``` python
from inspect_sentinel import observe_only

Task(..., sentinel=observe_only(suspicion()))
```

A monitor that fails is logged as a warning, and the step continues.

### Nesting

Any child of `concurrent()` or `sequential()` can itself be a protocol with children. Each monitor and protocol in a sentinel has an *instance name*: its mapping key, or its function name when it is passed in a list or on its own. Its *instance path* joins the instance names from the top of the sentinel down to it, separated by `/`.

For example, this sentinel runs a rule and a thresholded monitor side by side:

``` python
Task(..., sentinel={
    "network": no_network(),
    "llm": threshold(suspicion(), reject_at=0.8),
})
```

The mapping becomes a `concurrent()` at the top of the sentinel, which has the empty path. Its children are recorded at `network` and `llm`, and the monitor inside `threshold()` at `llm/suspicion`:

[![](diagrams/protocol-tree.excalidraw.svg)](diagrams/protocol-tree.excalidraw.svg)

The log uses instance paths to say which monitor or protocol reported what. Names in one group must be distinct, so use a mapping to give two instances of the same monitor or protocol different names.

## Rules

A *rule* is a protocol that runs no monitors. It reads the step itself, usually a tool call’s arguments, and decides. Rules make no model calls, so they cost almost nothing. Here is the rule from `examples/no_network.py`, which rejects [bash()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#bash) commands that use the network:

    examples/no_network.py

``` python
from example_samples import COUNT_ROWS, FETCH_DATA, FIX_SCRIPT
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    Protocol,
    call_text,
    find_words,
    protocol,
)

NETWORK_COMMANDS = ["curl", "wget", "pip install"]


1@protocol
def no_network() -> Protocol:
    async def decide(
        context: Context,
2        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        if step.call.function != "bash":
3            return None
        used = find_words(call_text(step.call), NETWORK_COMMANDS)
        if used:
4            return Decision.reject(
                f"command uses {', '.join(used)}",
                message=f"`{used[0]}` needs the network, which this task does not allow.",
            )
        return Decision.proceed()

    return decide


@task
def network_rule() -> Task:
    return Task(
        dataset=[FIX_SCRIPT, FETCH_DATA, COUNT_ROWS],
        solver=react(tools=[bash(timeout=60), python(timeout=60)]),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
5        sentinel=no_network(),
    )
```

1  
`@protocol` registers the *factory*, the function that returns the protocol function. Factory parameters configure the protocol.

2  
The type of `step` sets the stage the protocol runs at: here, before each tool call.

3  
Returning `None` abstains. This rule has no opinion about tool calls to other tools.

4  
The first argument is the explanation, which is recorded in the log and never shown to the agent. The `message` is what the agent is told.

5  
A protocol passed on its own is the top of the sentinel.

Run it from the repository root:

``` bash
inspect eval examples/no_network.py --model openai/gpt-5-mini
```

When the agent runs `curl`, the tool call does not run. The agent receives the message as the tool call’s error and carries on.

A `message` is only allowed on a `reject`. A `reject` without one tells the agent only that the tool call was not approved. This is often what you want, since a detailed message tells the agent what to avoid next time.

### Helpers for Rules

These functions handle the common parts of reading a tool call. They are plain functions, so a rule built on them is easy to test:

| Function | Returns |
|----|----|
| `call_text(call)` | Every string in the tool call’s arguments, in order, joined with newlines. For [bash()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#bash) it is the command; for [text_editor()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#text_editor) it is the command name, the path and the text. |
| `tool_matches(call, *patterns, case_sensitive=True)` | Whether the tool’s name matches one of the names or globs. `"bash"` matches only `bash`, and `"web_browser*"` matches every browser tool. `case_sensitive=False` also matches a bridged agent’s `Bash`. |
| `find_words(text, words)` | The words that appear in the text as whole words, in the order they appear, so an explanation can name them. `rm` does not match `platform`; `-delete` and `pip install` work as written. |
| `result_text(step)` | After a tool call: its result as text, or `Error (<type>): <message>` when it failed. |

Note that `tool_matches()` matches the whole name. This differs from the `tools` of an approval policy in Inspect, where `"bash"` also matches `bash_session` and a comma separates several names.

### Limits of Rules

Rules match a tool call’s text and arguments, and cannot prove a tool call safe. `no_network()` catches `curl`, but not a command that builds the word `curl` from variables at run time, and not a network request made through the [python()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#python) tool. Anything that needs an understanding of shell or code, such as which files a command touches, belongs in an [LLM monitor](./llm-monitors.html.md) or should be escalated to a person.

## Custom Protocols

This section covers the other actions, running monitors and protocols from inside a protocol, and decisions that no protocol above can change.

### Protocol Functions

A protocol factory is decorated with `@protocol` and returns an async function that takes three parameters, as `no_network()` does above:

| Parameter | Description |
|----|----|
| `context` | The protocol’s context: the host (for model calls), the sample being run, and the protocol’s instance path. |
| `step` | The step being examined. Its type annotation (`BeforeGenerate`, `AfterGenerate`, `BeforeToolCall` or `AfterToolCall`) is the stage the function runs at. |
| `escalations` | The escalations of the children before this one in a `sequential()`. Empty anywhere else. |

To run at several stages, return a `ProtocolGroup` with one function per stage. To run at every stage, annotate `step` as `Step`. Return `None` to abstain. The factory must return new functions each time it is called.

### Modifying Tool Calls

A `modify` decision runs the tool call with new arguments. Build the replacement tool call with `dataclasses.replace()` on `step.call`, and pass it as `modified`. This rule runs every [bash()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#bash) command from `/work`:

``` python
from dataclasses import replace

@protocol
def in_work_dir() -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        if step.call.function != "bash":
            return None
        command = step.call.arguments.get("command", "")
        return Decision(
            action="modify",
            modified=replace(
                step.call,
                arguments={**step.call.arguments, "command": f"cd /work && {command}"},
            ),
            explanation="ran the command from /work",
        )

    return decide
```

A `modify` may change only the tool call’s arguments: the replacement’s `id` and `function` must match the original’s.

### Terminating

`Decision.terminate()` ends the sample. It is allowed at every stage, so it is the action to use after a tool call has run:

``` python
@protocol
def leaked_keys() -> Protocol:
    async def decide(
        context: Context,
        step: AfterToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        if "AWS_SECRET_ACCESS_KEY" in result_text(step):
            return Decision.terminate("tool result contains AWS credentials")
        return Decision.proceed()

    return decide
```

### Escalating

`Decision.escalate()` declines to decide and hands the step to the next child of a `sequential()`. That child receives every escalation before it in its `escalations` parameter, oldest first. Each escalation is a `Reported[Decision]`: `name` and `path` identify the protocol that escalated, and `report` holds its decision.

This protocol answers escalations. It ends the sample when the protocol named `network` escalated, rejects the tool call when another protocol escalated, and abstains when nothing escalated:

``` python
@protocol
def escalation_policy() -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        if not escalations:
            return None
        reasons = "; ".join(f"{e.name}: {e.report.explanation}" for e in escalations)
        if any(e.name == "network" for e in escalations):
            return Decision.terminate(f"escalated by {reasons}")
        return Decision.reject(f"escalated by {reasons}")

    return decide
```

A protocol receives escalations only from the children before it in the same `sequential()`, so a nested `sequential()` starts with none. Monitors never receive escalations.

### Consulting Monitors

A protocol runs monitors with `run_monitors()`, passing its own `context` and the step. Taking the monitor as a factory parameter lets each task choose which monitor to use. This protocol rejects tool calls with a high score, escalates tool calls with a middling score, and lets the rest proceed:

``` python
@protocol
def screened(
    monitor: Monitor, escalate_at: float = 0.5, reject_at: float = 0.8
) -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        observations = await run_monitors({"judge": monitor}, context, step)
        score = observations.max_suspicion()
        if score is None:
            return Decision.proceed("no monitor reported")
        if score >= reject_at:
            return Decision.reject(f"suspicion {score:.2f}")
        if score >= escalate_at:
            return Decision.escalate(f"suspicion {score:.2f}")
        return Decision.proceed()

    return decide
```

Use it like any other protocol, for example `sequential([screened(suspicion()), human(stages=["tool_call"])])`. The mapping passed to `run_monitors()` gives the monitor the instance name `judge`, so its observations are recorded at `screened/judge`. `max_suspicion()` returns the highest score across monitors and dimensions, or `None` when no monitor reported.

The protocol runs only at the stage its `step` annotation names, so a monitor that watches other stages never runs inside it. To consult monitors at several stages, return a `ProtocolGroup` with one function per stage, as `threshold()` does.

### Consulting Protocols

Three functions run children from inside a protocol. Each runs its children concurrently and returns their reports in configuration order:

| Function | Runs | Returns |
|----|----|----|
| `run_monitors()` | Monitors | `Observations` |
| `run_protocols()` | Protocols | `Decisions` |
| `run_children()` | Monitors and protocols | `Reports`, which unpacks as `observations, decisions` |

Each item is a `Reported` holding the instance’s `name`, `path` and report. `Decisions.strongest()` returns the strongest decision, ranked as in `concurrent()`. When a child decides `terminate`, the other children are cancelled. The protocols you run receive no escalations unless you pass them with the `escalations` argument of `run_protocols()` or `run_children()`.

For example, a protocol that takes its children as a `Protocols` factory parameter can run them with `run_protocols(children, context, step)`, and reject the tool call when `decisions.strongest()` is an escalation instead of passing the escalation up.

### When a Monitor Fails

A monitor that raises an error does not stop the other monitors: `run_monitors()` records the failure and returns. Reading the observations then raises `MonitorFailedError`, so a protocol that uses the scores fails rather than deciding without the failed monitor.

To decide anyway, check `observations.failed`, and read `observations.succeeded` for the observations of the monitors that did not fail:

``` python
observations = await run_monitors(monitors, context, step)
if observations.failed:
    names = ", ".join(f.name for f in observations.failed)
    return Decision.escalate(f"monitor failed: {names}")
score = observations.max_suspicion()
```

Note that a protocol that reads only the decisions from `run_children()` gets no error for a failed monitor.

### Final Decisions

The protocols above a protocol can combine or override the decision it returns. `decide_final()` instead ends the step with a decision at once. Children still running beside the protocol are cancelled, and the protocols above it do not run their decision logic.

Let’s look at `examples/nested.py`. Its `protected()` rule rejects any [bash()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#bash) command that mentions `/etc` or `~/.aws`, using `decide_final()`:

    examples/nested.py

``` python
PROTECTED_PATHS = ["/etc", "~/.aws"]


@protocol
def protected() -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        if step.call.function != "bash":
            return None
        command = call_text(step.call)
        for path in PROTECTED_PATHS:
            if path in command:
                decide_final(
                    Decision.reject(
                        f"command mentions {path}",
                        message=f"`{path}` is off limits; work only under /work.",
                    )
                )
        return Decision.proceed()

    return decide


Task(
    ...,
    sentinel={
        "guard": concurrent({"network": no_network(), "protected": protected()}),
        "audit": suspicion(),
    },
)
```

When `protected()` calls `decide_final()`, the log records `guard` and the top of the sentinel as bypassed, the `audit` monitor as cancelled, and the `reject` from `guard/protected`. Run it with a model for the monitor:

``` bash
inspect eval examples/nested.py --model openai/gpt-5-mini --model-role monitor=anthropic/claude-haiku-4-5
```

Use `decide_final()` when no protocol above should be able to change the decision, or to stop monitors that are still running. Under the built-in protocols, returning a `reject` or `terminate` is usually enough, since nothing outranks them. If two protocols call `decide_final()` at once, the first one wins and the other is recorded as superseded.
