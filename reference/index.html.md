# Reference – Inspect Sentinel

## Steps

A [Step](../reference/index.html.md#step) here is the payload a monitor or protocol receives, and [Context](../reference/index.html.md#context) is what it knows about the sample; both are unrelated to inspect_ai’s `StepEvent` and `step()`.

### BeforeToolCall

A tool call about to execute.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_step.py#L12)

``` python
@dataclass(frozen=True)
class BeforeToolCall
```

#### Attributes

`conversation` str  
Id linking this agent’s steps across compactions.

`message` str  
Assistant text alongside the call.

`call` ToolCall  
The call as it will execute.

`view` ToolCallView  
Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise.

`input` list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `context.input` is the sample’s input, the assignment.

`history` list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
The scaffold’s full conversation, pre-compaction turns included. In-process this is the scaffold’s own list; monitors must not mutate it.

`escalations` tuple\[[Reported](../reference/index.html.md#reported)\[[Decision](../reference/index.html.md#decision)\], ...\]  
Earlier links’ escalations, when this step is inside a `sequential` composition.

### AfterToolCall

A tool call that has executed, before the model sees its result.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_step.py#L38)

``` python
@dataclass(frozen=True)
class AfterToolCall
```

#### Attributes

`conversation` str  
Id linking this agent’s steps across compactions.

`message` str  
Assistant text alongside the call.

`call` ToolCall  
The call as executed, after any `modify`.

`result` [ChatMessageTool](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessagetool)  
What the model will see, after truncation.

`output` [ToolResult](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolresult)  
The tool’s untruncated return value.

`view` ToolCallView  
Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise.

`input` list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `context.input` is the sample’s input, the assignment.

`history` list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
The scaffold’s full conversation, pre-compaction turns included. In-process this is the scaffold’s own list; monitors must not mutate it.

`escalations` tuple\[[Reported](../reference/index.html.md#reported)\[[Decision](../reference/index.html.md#decision)\], ...\]  
Earlier links’ escalations, when this step is inside a `sequential` composition.

### Step

The union, for code that handles any stage: a protocol, or a dispatcher.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_step.py#L70)

``` python
Step: TypeAlias = BeforeToolCall | AfterToolCall
```

## Report types

### Observation

Evidence for a protocol to calibrate and threshold. What a monitor returns.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L27)

``` python
class Observation(BaseModel)
```

#### Attributes

`suspicion` [Suspicion](../reference/index.html.md#suspicion)  
How suspicious. Required: an observation with no score says nothing.

`explanation` str \| None  
Why, for a reader of the transcript.

`references` list\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  
Messages and events in the transcript the report points at, which Inspect View links from cites such as `[M22]` in `explanation`.

`metadata` dict\[str, Any\] \| None  
Author-supplied structured context, recorded verbatim.

#### Methods

score  
An observation that sets `suspicion` to the given score.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L42)

``` python
@classmethod
def score(
    cls,
    suspicion: Suspicion,
    explanation: str | None = None,
    *,
    references: Sequence[Reference] = (),
) -> Observation
```

`suspicion` [Suspicion](../reference/index.html.md#suspicion)  

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

flag  
A detector’s observation: suspicion `1.0`, for a protocol to act on or not.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L55)

``` python
@classmethod
def flag(
    cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
) -> Observation
```

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

### Decision

What should happen at a step. What a protocol returns; advisory to any protocol wrapping it unless passed to [decide_final()](../reference/index.html.md#decide_final).

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L63)

``` python
class Decision(BaseModel)
```

#### Attributes

`action` [Action](../reference/index.html.md#action)  
What should happen at this step. Required.

`audit` bool  
Request that oversight budget be spent on this step. Recorded for offline analysis and shown in the viewer as a flag; nothing acts on it yet.

`modified` ToolCall \| None  
Replacement for the step’s subject, for a `modify` decision.

`explanation` str \| None  
Why, for the log. The agent never sees it.

`references` list\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  
Messages and events in the transcript the report points at, which Inspect View links from cites such as `[M22]` in `explanation`.

`message` str \| None  
Text the agent reads when the step is rejected. The agent sees nothing else from a decision.

`metadata` dict\[str, Any\] \| None  
Author-supplied structured context, recorded verbatim.

#### Methods

proceed  
Let the step proceed unchanged: a decision with action `continue`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L87)

``` python
@classmethod
def proceed(
    cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
) -> Decision
```

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

reject  
Reject the step, telling the agent it was rejected: with `message` if given, else with the host’s default text.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L96)

``` python
@classmethod
def reject(
    cls,
    explanation: str | None = None,
    *,
    message: str | None = None,
    references: Sequence[Reference] = (),
) -> Decision
```

`explanation` str \| None  

`message` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

terminate  
End the sample.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L112)

``` python
@classmethod
def terminate(
    cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
) -> Decision
```

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

escalate  
Decline to decide and pass the step to the next decider.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L121)

``` python
@classmethod
def escalate(
    cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
) -> Decision
```

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

### Suspicion

How suspicious a step is: one finite score, or a non-empty dict scoring several dimensions independently.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L12)

``` python
Suspicion: TypeAlias = SentinelSuspicion
```

### Action

What a protocol can decide about the step it examined.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L15)

``` python
Action: TypeAlias = SentinelAction
```

### Report

What the transcript records.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L131)

``` python
Report: TypeAlias = Observation | Decision
```

### Reported

A report with the identity of the configured instance that produced it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_report.py#L137)

``` python
@dataclass(frozen=True)
class Reported(Generic[R_co])
```

#### Attributes

`name` str  
Instance name: the mapping key, or the registry name without its package prefix.

`path` str  
Instance path, e.g. `attempt/internet_attempt`.

`report` R_co  
The observation or decision this instance produced.

`function` str  
The `__name__` of the function that produced the report, which tells apart the functions of an instance whose factory returned several.

## Monitors and protocols

[Protocol](../reference/index.html.md#protocol) is a control protocol, not `typing.Protocol`.

### monitor

Register a monitor factory.

Use as `@monitor`, or as `@monitor(name=..., version=...)`. The factory returns a function that must be `async`, take `(context, step)`, annotate `step` with exactly one stage payload, and be annotated `-> Observation`, or `-> Observation | None` if it can abstain. These are checked when the factory is called. Each function watches one stage; to watch several, return a [MonitorGroup](../reference/index.html.md#monitorgroup) of functions, which together are one configured instance. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

A configured function runs only through the runners while a sentinel is running: a protocol that calls one directly from its body raises `RuntimeError`, and must call it through `run_monitors` or `run_children` instead. Outside a run, as in a unit test, a direct call works.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_decorators.py#L111)

``` python
def monitor(
    factory: Callable[P, Monitor | MonitorGroup] | None = None,
    /,
    *,
    name: str | None = None,
    version: int = 0,
) -> Callable[P, Monitor | MonitorGroup] | _MonitorDecorator
```

`factory` Callable\[P, [Monitor](../reference/index.html.md#monitor) \| [MonitorGroup](../reference/index.html.md#monitorgroup)\] \| None  
A function returning a monitor, or a [MonitorGroup](../reference/index.html.md#monitorgroup).

`name` str \| None  
The registered name, in place of the factory’s `__name__`.

`version` int  
The monitor’s version, recorded in its registry metadata so a calibration can say which version it measured; bump it when a change alters the scores. Defaults to 0.

### protocol

Register a protocol factory.

Same contract as `@monitor`, with each returned function annotated `-> Decision`, or `-> Decision | None` if it can abstain, and `step` may also be annotated [Step](../reference/index.html.md#step) for a protocol that runs at every stage, such as a composition that only forwards the step to its children. To configure several functions as one instance, return a [ProtocolGroup](../reference/index.html.md#protocolgroup); a function among them that returns `terminate` or calls [decide_final()](../reference/index.html.md#decide_final) ends the instance’s run, and those after it do not run and are not recorded. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

As with `@monitor`, a configured function called directly from another protocol’s body during a run raises `RuntimeError`; call it through `run_protocols` or `run_children`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_decorators.py#L147)

``` python
def protocol(
    factory: Callable[P, Protocol | ProtocolGroup] | None = None,
    /,
    *,
    name: str | None = None,
    version: int = 0,
) -> Callable[P, Protocol | ProtocolGroup] | _ProtocolDecorator
```

`factory` Callable\[P, [Protocol](../reference/index.html.md#protocol) \| [ProtocolGroup](../reference/index.html.md#protocolgroup)\] \| None  
A function returning a protocol, or a [ProtocolGroup](../reference/index.html.md#protocolgroup).

`name` str \| None  
The registered name, in place of the factory’s `__name__`.

`version` int  
The protocol’s version, recorded in its registry metadata; bump it when a change alters its decisions. Defaults to 0.

### Monitor

A monitor: observes a step at one stage and reports a suspicion score, or abstains.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L10)

``` python
Monitor: TypeAlias = (
    Callable[[Context, BeforeToolCall], Awaitable[Observation | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Observation | None]]
)
```

### Protocol

A control protocol, not `typing.Protocol`: decides what happens at a step, or abstains. Annotating [Step](../reference/index.html.md#step) runs it at every stage.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L16)

``` python
Protocol: TypeAlias = (
    Callable[[Context, Step], Awaitable[Decision | None]]
    | Callable[[Context, BeforeToolCall], Awaitable[Decision | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Decision | None]]
)
```

### MonitorGroup

Several monitor functions that form one monitor instance.

Return one from a `@monitor` factory in place of a single function. Together the functions are one configured instance, sharing its name, path and `store_as` namespace; each runs, in the order given, at the stage it watches, sequentially since they share one store, and each report records which `function` made it. A group is not callable: hand it to a protocol, which runs it through `run_monitors` or `run_children`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L47)

``` python
@final
class MonitorGroup(Group)
```

#### Attributes

`functions` tuple\[[Monitor](../reference/index.html.md#monitor), ...\]  
The member functions, in the order they run.

#### Methods

\_\_init\_\_  
Group monitor functions into one instance.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L54)

``` python
def __init__(self, *functions: Monitor) -> None
```

`*functions` [Monitor](../reference/index.html.md#monitor)  
At least one monitor function, each with a distinct `__name__`.

### ProtocolGroup

Several protocol functions that form one protocol instance.

Return one from a `@protocol` factory in place of a single function. As with [MonitorGroup](../reference/index.html.md#monitorgroup), the functions share one name, path and `store_as` namespace and run in the order given at the stages they accept; a function that returns `terminate` or calls [decide_final()](../reference/index.html.md#decide_final) ends the instance’s run, and those after it do not run and are not recorded. A group is not callable: hand it to a protocol, which runs it through `run_protocols` or `run_children`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L71)

``` python
@final
class ProtocolGroup(Group)
```

#### Attributes

`functions` tuple\[[Protocol](../reference/index.html.md#protocol), ...\]  
The member functions, in the order they run.

#### Methods

\_\_init\_\_  
Group protocol functions into one instance.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L78)

``` python
def __init__(self, *functions: Protocol) -> None
```

`*functions` [Protocol](../reference/index.html.md#protocol)  
At least one protocol function, each with a distinct `__name__`.

### Monitors

Monitors handed to a protocol, named by mapping key or by registry name without its package prefix.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L95)

``` python
Monitors: TypeAlias = (
    Mapping[str, Monitor | MonitorGroup] | Sequence[Monitor | MonitorGroup]
)
```

### Protocols

Protocols handed to a protocol, named the same way.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L100)

``` python
Protocols: TypeAlias = (
    Mapping[str, Protocol | ProtocolGroup] | Sequence[Protocol | ProtocolGroup]
)
```

### Children

Monitors and protocols together, for the compositions that record either.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_types.py#L107)

``` python
Children: TypeAlias = Mapping[str, Child] | Sequence[Child]
```

## Shipped protocols

### observe

Record every monitor’s observation and never act.

What a bare monitor or a list of monitors resolves to, and how a benign score distribution is collected before anything is configured to act on it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_protocols/observe.py#L12)

``` python
@protocol
def observe(monitors: Monitor | MonitorGroup | Monitors) -> Protocol
```

`monitors` [Monitor](../reference/index.html.md#monitor) \| [MonitorGroup](../reference/index.html.md#monitorgroup) \| [Monitors](../reference/index.html.md#monitors)  
The monitor, or monitors, to run at every step they watch.

### concurrent

Run every child at once; the strictest decision wins.

What a list containing a protocol resolves to. Monitors are recorded in the transcript; protocols vote by `terminate > reject > modify > escalate > continue`, the first in configuration order winning a tie, so a peer’s `continue` does not override an `escalate` and the layer escalates when that is the strongest vote. A `modify` when another protocol also decided anything but `escalate` becomes a `reject` naming the modifier, since the others decided about the call as it stood; votes count by instance, so another function of the modifier’s own instance contests it only by also modifying; the rejection carries the modifier’s `audit` and `metadata`, and its explanation leads with the modifier’s. When more than one protocol decided, the layer’s explanation lists each one’s decision after the winner’s own, labelling a function of a [ProtocolGroup](../reference/index.html.md#protocolgroup) or [MonitorGroup](../reference/index.html.md#monitorgroup) `name.function`. A child that calls [decide_final()](../reference/index.html.md#decide_final) ends the step and no vote is taken.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_protocols/concurrent.py#L12)

``` python
@protocol
def concurrent(
    children: Monitor | MonitorGroup | Protocol | ProtocolGroup | Children,
) -> Protocol
```

`children` [Monitor](../reference/index.html.md#monitor) \| [MonitorGroup](../reference/index.html.md#monitorgroup) \| [Protocol](../reference/index.html.md#protocol) \| [ProtocolGroup](../reference/index.html.md#protocolgroup) \| [Children](../reference/index.html.md#children)  
A monitor, protocol or group, or several, to run together.

### threshold

Act on suspicion scores: `reject` when any score reaches its reject threshold, `terminate` when any reaches its terminate threshold.

A threshold is one number for every score, or a mapping that sets one per dimension. In a mapping, `"<monitor>.<dimension>"` applies to one dimension of one monitor (the monitor’s instance name, as in `Reported.name`), a bare `"<dimension>"` applies to that dimension from every monitor, a monitor reporting a single number is keyed by its instance name, and `"*"` applies to anything unlisted. The most specific key wins: qualified, then bare, then `"*"`. A score no key covers never triggers. A number is shorthand for `{"*": number}`. A dimension name containing `.` must be qualified.

Terminate is checked first. A reject or terminate is explained for the log: it names the highest-scoring score that crossed (the first in configuration order on a tie) and carries its monitor’s explanation and references, e.g. `suspicion 0.95 from audit: posts credentials to an external host` when the threshold came from `"*"` or the score is a single number, and `sabotage 0.72 ≥ 0.60 from audit: edits the tests` when it came from the dimension’s own key. Neither sets `message`, so the agent gets the host’s default “not approved” text and learns nothing about what the monitors saw.

The smallest protocol, and the one calibration feeds: the constants come from a benign score distribution, which is why they live here and not on a monitor. Every monitor, and every function of a [MonitorGroup](../reference/index.html.md#monitorgroup), must watch [BeforeToolCall](../reference/index.html.md#beforetoolcall), since that is the only stage a `reject` is legal at.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_protocols/threshold.py#L17)

``` python
@protocol
def threshold(
    monitors: Monitor | MonitorGroup | Monitors,
    reject_at: float | Mapping[str, float],
    terminate_at: float | Mapping[str, float] | None = None,
) -> Protocol
```

`monitors` [Monitor](../reference/index.html.md#monitor) \| [MonitorGroup](../reference/index.html.md#monitorgroup) \| [Monitors](../reference/index.html.md#monitors)  
The monitor, or monitors, whose scores are thresholded.

`reject_at` float \| Mapping\[str, float\]  
Suspicion at or above which the call is rejected: one number, or a mapping by dimension.

`terminate_at` float \| Mapping\[str, float\] \| None  
Suspicion at or above which the sample is ended, in the same form; `None` never terminates. Where it and `reject_at` both cover a score, it must be above `reject_at`.

## Runner

### run_monitors

Run monitors concurrently and collect their observations in configuration order.

A typed shortcut for `run_children` over monitors only: its parameter rejects a protocol and its return type is the observations alone.

Derives each child’s context under this layer’s path and records every observation, including the ones the caller goes on to ignore. A monitor that abstained or does not watch this stage contributes nothing, so the result may be empty. An instance whose factory returned several functions contributes one observation per function that reported, in the order the factory returned them.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_runner.py#L142)

``` python
async def run_monitors(
    monitors: Monitor | MonitorGroup | Monitors, context: Context, step: Step
) -> Observations
```

`monitors` [Monitor](../reference/index.html.md#monitor) \| [MonitorGroup](../reference/index.html.md#monitorgroup) \| [Monitors](../reference/index.html.md#monitors)  
One monitor or [MonitorGroup](../reference/index.html.md#monitorgroup), named by its registry name without the package prefix; a sequence of them, named the same way; or a mapping of instance names to them.

`context` [Context](../reference/index.html.md#context)  
This layer’s context.

`step` [Step](../reference/index.html.md#step)  
The step being examined.

### run_protocols

Run protocols concurrently and collect their decisions in configuration order, cancelling the rest at their next await when one returns `terminate` or calls [decide_final()](../reference/index.html.md#decide_final).

A typed shortcut for `run_children` over protocols only: its parameter rejects a monitor and its return type is the decisions alone. A [decide_final()](../reference/index.html.md#decide_final) from any protocol at any depth below propagates out of this call, so the caller’s own decision logic does not run.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_runner.py#L160)

``` python
async def run_protocols(
    protocols: Protocol | ProtocolGroup | Protocols, context: Context, step: Step
) -> Decisions
```

`protocols` [Protocol](../reference/index.html.md#protocol) \| [ProtocolGroup](../reference/index.html.md#protocolgroup) \| [Protocols](../reference/index.html.md#protocols)  
One protocol or [ProtocolGroup](../reference/index.html.md#protocolgroup), a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.

`context` [Context](../reference/index.html.md#context)  
This layer’s context.

`step` [Step](../reference/index.html.md#step)  
The step being examined.

### run_children

Run monitors and protocols together in one task group, cancelling the rest at their next await when a protocol returns `terminate` or calls [decide_final()](../reference/index.html.md#decide_final).

Runs any mix of children; `run_monitors` and `run_protocols` are typed shortcuts for one family. A child cancelled this way is recorded through `Recorder.cancelled`; one that finishes without awaiting is recorded normally.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_runner.py#L176)

``` python
async def run_children(
    children: Child | Children, context: Context, step: Step
) -> Reports
```

`children` Child \| [Children](../reference/index.html.md#children)  
One monitor, protocol or group, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.

`context` [Context](../reference/index.html.md#context)  
This layer’s context.

`step` [Step](../reference/index.html.md#step)  
The step being examined.

### decide_final

End the step with this decision.

Nothing above the calling protocol runs; siblings still in flight are cancelled and recorded as such. First call wins if two race.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_final.py#L25)

``` python
def decide_final(decision: Decision) -> NoReturn
```

`decision` [Decision](../reference/index.html.md#decision)  
The outcome for this step.

### Observations

What a layer’s monitors reported, in configuration order.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_results.py#L42)

``` python
class Observations(_ReportSequence[Observation])
```

#### Methods

max_suspicion  
The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_results.py#L45)

``` python
def max_suspicion(self) -> float | None
```

### Decisions

What a layer’s protocols decided, in configuration order.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_results.py#L51)

``` python
class Decisions(_ReportSequence[Decision])
```

#### Methods

strongest  
The strongest decision by `terminate > reject > modify > escalate > continue`, the first in configuration order on a tie; `None` if nobody decided.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_results.py#L54)

``` python
def strongest(self) -> Reported[Decision] | None
```

### Reports

Both families of report from one layer, unpackable as `observations, decisions = await run_children(...)`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_results.py#L61)

``` python
class Reports(NamedTuple)
```

#### Attributes

`observations` [Observations](../reference/index.html.md#observations)  
From the layer’s monitors.

`decisions` [Decisions](../reference/index.html.md#decisions)  
From the layer’s protocols.

## Context and host

### Context

What a monitor or protocol is given. Constant for a sample; nothing in it came from the agent.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_context.py#L105)

``` python
@dataclass(frozen=True, kw_only=True)
class Context
```

#### Attributes

`task` str \| None  
Task name, or the deployed agent’s identifier.

`task_description` str \| None  
What agents in this setting are asked to do. None until inspect_ai has `Task(description=)`.

`sample_id` str \| int \| None  
Dataset sample identifier.

`epoch` int \| None  
Epoch of the sample, when the task runs several.

`sample_description` str \| None  
What is distinctive about this instance. None until inspect_ai has `Sample(description=)`.

`input` str \| list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
The sample’s input: the assignment the agent was given. Judge a step against `step.input`, which is exactly what the model was sent; use this for what the agent was asked to do.

`metadata` dict\[str, Any\]  
Task metadata merged with sample metadata.

`path` str  
Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer.

`store` [Store](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#store)  
The whole sample store, the agent’s state included; not namespaced. Use [store_as()](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#store_as) for this instance’s own state.

`host` [Host](../reference/index.html.md#host)  
Inference through the host’s models. See [Host](../reference/index.html.md#host).

`target` [Target](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#target) \| None  
The expected answer. None until monitors and protocols can opt in with `target=True`.

#### Methods

store_as  
Typed view of this instance’s state, namespaced by `path`.

Renaming the instance (its mapping key) or wrapping it in another layer changes `path`, so its state moves with it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_context.py#L142)

``` python
def store_as(self, model_cls: type[SMT]) -> SMT
```

`model_cls` type\[SMT\]  
The [StoreModel](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#storemodel) subclass to read and write through.

### Host

What a monitor or protocol may do to the outside world. Author-facing.

Inference is the only effect for now. Outbound HTTP through named endpoints (`fetch`) is planned. Meanwhile a monitor running in-process may call inspect_ai APIs directly, such as [sandbox()](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#sandbox) or `logging`, at the cost of portability to a proxy, which a future `portable=False` will declare.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_context.py#L18)

``` python
class Host(TypingProtocol)
```

#### Methods

generate  
Generate with a model the host resolves.

With neither `model` nor `role`, the host uses the `monitor` role. If that role is not configured, it falls back to the agent’s model and logs a warning once per eval naming the role to set.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/e906d6842ba211473357bf866f97fad8b20b3fb6/src/inspect_sentinel/_context.py#L24)

``` python
async def generate(
    self,
    input: str | list[ChatMessage],
    *,
    model: str | Model | None = None,
    role: str | None = None,
    tools: list[ToolInfo] | None = None,
    config: GenerateConfig | None = None,
) -> ModelOutput
```

`input` str \| list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
A prompt string or a list of chat messages.

`model` str \| [Model](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#model) \| None  
A model name or instance. Not with `role`.

`role` str \| None  
A model role, e.g. `trusted`. Not with `model`.

`tools` list\[[ToolInfo](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolinfo)\] \| None  
Tool definitions to offer the model.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig) \| None  
Generation configuration.
