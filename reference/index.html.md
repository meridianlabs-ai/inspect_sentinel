# Reference – Inspect Sentinel

## Steps

A `Step` here is the payload a monitor or protocol receives, and `Context` is what it knows about the sample; both are unrelated to inspect_ai’s `StepEvent` and `step()`.

### BeforeGenerate

A generate by the agent’s model, before the request is sent and before the cache lookup.

`input`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. `input`, `agent_messages`, `tools` and `config` are the host’s own objects, borrowed for the step, not copies, and `history` holds the logged copies of what was sent: monitors and protocols must not mutate any of them (copy first to edit), since a change could alter the agent state, the real request or the log.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L38)

``` python
@dataclass(frozen=True)
class BeforeGenerate
```

#### Attributes

`model` str  
The model the request is for, as `provider/name`.

`conversation` str  
Id linking this agent’s steps across compactions.

`input` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
Exactly what the model is sent: the caller’s messages after the model’s own preparation (the configured system message, reasoning history, merged consecutive messages). In-process these are the host’s own messages; monitors must not mutate them.

`agent_messages` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
The outermost running `@agent` agent function’s own message list: the `messages` of the [AgentState](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#agentstate) object it was called with, read live at every stage. A scaffold may curate this list, for example to keep hidden side-task instructions out of a control monitor’s view, and it may contain retries. For the actual model traffic, including instructions sent but kept out of this list, use `input` and `history`. To omit the request lists and show the agent’s own list, use `StepPreprocessor(input_window=None, history_window=None, agent_messages_window=AllMessages())`. Other stage content remains: every stage renders the request’s tools and tool choice while `input` is known (a tool stage’s are the producing request’s); `AfterGenerate` renders the reply, including reasoning by default before the scaffold has curated it; both tool stages render the assistant text and call; and `AfterToolCall` renders the result. Strict blindness also needs a stage-specific `transform` or a custom prompt that curates these surfaces; a transform can clear `tools` and `tool_choice`. The list is never approximated from `input` or `history`. A compaction does not drop turns from the list and adds its summary message (overflow recovery also drops the reply that overflowed); `input` carries the compacted request. With `truncation` set, `react`’s overflow filter trims this list when forced compaction is not available or does not help (no compaction, a model agent as `model=`, a failed forced compaction, or a second overflow right after one), while `history` keeps the turns.

The same source is read by reference at all four stages. An agent that continues with another state object shows the object it was called with; `react` keeps its own object when a model agent returns a new one. Sub-agents run through handoff or [as_tool()](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#as_tool) have their own conversations and lists. Hidden input stays out only when the scaffold that owns the conversation is the outermost `@agent` agent function.

`None` when no `@agent` agent function is running (for example a solver calling [generate()](https://inspect.aisi.org.uk/reference/inspect_ai.solver.html#generate)); a callable-object or `partial` agent does not register this list. Also `None` for bridged agents, in replay, and under a sidecar or proxy host. Monitors must treat a present list as read-only and must not mutate it.

`history` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
Every message this agent instance has sent to the model so far, in the order sent: the inputs of all its model calls taken together, each message once, where it was first sent. The opening prompts as the model saw them come first, a message the scaffold rewrote before sending shows as it was sent, and turns a compaction later dropped are still here. What was already sent is decided by content and position, not by message id: each call’s input is aligned against the previous call’s. So a message the host rebuilds with a fresh id for every call, such as a configured system message, is not repeated where it was, while a message sent again changed appears again as its new version, where that version was sent, as do a message moved behind a later one, a message one call dropped and a later call restored, and an equal message sent again at a later turn. inspect_ai sends an earlier assistant turn without its reasoning under `reasoning_history="last"`, so such a turn appears once with its reasoning and once without. Each message shows as it was sent: a message the scaffold edits or elides in place after sending it (as `react`’s `on_continue` can elide an old tool result) keeps its sent content here, and appears again as its new version if it is sent again. The model’s outputs appear once a later call sends them back. It does not yet include `input`, which is about to be sent. Only this agent instance’s model calls count: a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer or a monitor is not an agent turn. A model call the scaffold makes itself inside its agent loop does count, unless the scaffold marks it with `inspect_ai.model.model_origin("helper")`, which also leaves it unchecked, or runs it as an agent of its own, through `inspect_ai.agent.run()`. Sub-agents inside one bridged CLI share its conversation. `None` when the host cannot produce it: in a replay from a log, until replay is built. In-process Inspect always provides it. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, a conversation is told apart by how it opens, so agents whose conversations open alike share one history, every model call of the conversation counts as an agent turn, and a sidecar that joins a conversation midway starts the history there. Unlike `agent_messages`, it is what was sent: a rebuilt or compacted input, and instructions a scaffold sends but keeps out of its own list, are here; outputs and results the model was never sent back are not. In-process the messages are the copies of what was sent that the transcript’s [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)s hold, not the host’s own objects; monitors must not mutate them.

`human_decisions` Sequence\[HumanDecision\] \| None  
Decisions a person made through `human()` at earlier steps of this agent’s conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent’s conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls’ tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling’s `decide_final()` superseded is dropped); it never holds a monitor’s observation or another protocol’s decision. A prompt the person cancelled or closed without choosing is recorded as the surface’s fail-closed answer, with an explanation that says so; render it, never as the person’s judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them.

`tools` list\[[ToolInfo](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolinfo)\]  
The tools the model is offered, after `tool_choice` filtering. In-process these are the host’s own objects; monitors must not mutate them.

`tool_choice` [ToolChoice](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolchoice)  
The tool choice the model is sent.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig)  
The generation configuration the model is sent, with the task’s and the model’s settings merged in. In-process this is the host’s own object; monitors must not mutate it.

### AfterGenerate

A generate by the agent’s model that has completed, before its output reaches the agent.

`input`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. `input`, `agent_messages`, `tools`, `config` and `output` are the host’s own objects, borrowed for the step, not copies, and `history` holds the logged copies of what was sent: monitors and protocols must not mutate any of them (copy first to edit), since a change could alter the agent state, the real request, the response the agent receives or the log.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L78)

``` python
@dataclass(frozen=True)
class AfterGenerate
```

#### Attributes

`model` str  
The model the request was for, as `provider/name`; `output.model` names the model that answered.

`conversation` str  
Id linking this agent’s steps across compactions.

`input` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
Exactly what the model was sent, as for `BeforeGenerate`; monitors must not mutate it.

`agent_messages` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
The outermost running agent function’s own message list, read live by reference, or `None`, as for `BeforeGenerate.agent_messages`. The agent may not have added `output` yet. Monitors must not mutate it.

`history` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
Every message this agent instance has sent to the model so far, in the order sent, or `None` when the host cannot produce it, as for `BeforeGenerate.history`. It includes `input`; `output` is not in it yet. Monitors must not mutate them.

`human_decisions` Sequence\[HumanDecision\] \| None  
Decisions a person made through `human()` at earlier steps of this agent’s conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent’s conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls’ tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling’s `decide_final()` superseded is dropped); it never holds a monitor’s observation or another protocol’s decision. A prompt the person cancelled or closed without choosing is recorded as the surface’s fail-closed answer, with an explanation that says so; render it, never as the person’s judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them.

`tools` list\[[ToolInfo](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolinfo)\]  
The tools the model was offered; monitors must not mutate them.

`tool_choice` [ToolChoice](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolchoice)  
The tool choice the model was sent.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig)  
The generation configuration the model was sent; monitors must not mutate it.

`output` [ModelOutput](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#modeloutput)  
What the model returned; the first choice’s message is what the agent receives. Some providers return an output with no choices, so check `output.choices` before reading `output.message`. In-process this is the host’s own output, which the agent receives; monitors must not mutate it.

### BeforeToolCall

A tool call about to execute.

`input`, `tools`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps for the conversation rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. Monitors and protocols must not mutate them or the messages and tool definitions in them.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L127)

``` python
@dataclass(frozen=True)
class BeforeToolCall
```

#### Attributes

`conversation` str  
Id linking this agent’s steps across compactions.

`message` str  
Assistant text alongside the call.

`message_id` str  
Id of the assistant message that made the call, so a monitor model can cite it.

`call` ToolCall  
The call as it will execute.

`view` ToolCallView  
Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise.

`input` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when no model call produced the assistant message (a pre-filled message, a human agent, or one the scaffold built itself) or when the transcript event that recorded the model call is no longer available; it is never approximated from `agent_messages`. `context.eval.sample_input` is the sample’s input, the assignment.

`agent_messages` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
The outermost running agent function’s own message list, read live by reference, or `None`, as for `BeforeGenerate.agent_messages`. For `react` and other scaffolds that append to the state they were called with, it includes the assistant message holding the call. Monitors must not mutate it.

`history` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
Every message this agent instance has sent to the model so far, in the order sent: the inputs of all its model calls taken together, each message once, where it was first sent. The opening prompts as the model saw them come first, a message the scaffold rewrote before sending shows as it was sent, and turns a compaction later dropped are still here. What was already sent is decided by content and position, not by message id: each call’s input is aligned against the previous call’s. So a message the host rebuilds with a fresh id for every call, such as a configured system message, is not repeated where it was, while a message sent again changed appears again as its new version, where that version was sent, as do a message moved behind a later one, a message one call dropped and a later call restored, and an equal message sent again at a later turn. inspect_ai sends an earlier assistant turn without its reasoning under `reasoning_history="last"`, so such a turn appears once with its reasoning and once without. Each message shows as it was sent: a message the scaffold edits or elides in place after sending it (as `react`’s `on_continue` can elide an old tool result) keeps its sent content here, and appears again as its new version if it is sent again. The model’s outputs appear once a later call sends them back. It includes the input of the call that produced this tool call. Only this agent instance’s model calls count: a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer or a monitor is not an agent turn. A model call the scaffold makes itself inside its agent loop does count, unless the scaffold marks it with `inspect_ai.model.model_origin("helper")`, which also leaves it unchecked, or runs it as an agent of its own, through `inspect_ai.agent.run()`. Sub-agents inside one bridged CLI share its conversation. `None` when the host cannot produce it: in a replay from a log, until replay is built. In-process Inspect always provides it. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, a conversation is told apart by how it opens, so agents whose conversations open alike share one history, every model call of the conversation counts as an agent turn, and a sidecar that joins a conversation midway starts the history there. Unlike `agent_messages`, it is what was sent: a rebuilt or compacted input, and instructions a scaffold sends but keeps out of its own list, are here; outputs and results the model was never sent back are not. In-process the messages are the copies of what was sent that the transcript’s [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)s hold, not the host’s own objects; monitors must not mutate them.

`human_decisions` Sequence\[HumanDecision\] \| None  
Decisions a person made through `human()` at earlier steps of this agent’s conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent’s conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls’ tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling’s `decide_final()` superseded is dropped); it never holds a monitor’s observation or another protocol’s decision. A prompt the person cancelled or closed without choosing is recorded as the surface’s fail-closed answer, with an explanation that says so; render it, never as the person’s judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them.

`tools` Sequence\[[ToolInfo](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolinfo)\] \| None  
The tool definitions offered to the model call that produced the call, as the transcript’s [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent) recorded them, after any model hooks (so they can differ from that call’s `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` exactly when `input` is `None`; it is never approximated from the tools available now. Monitors must not mutate them.

`tool_choice` [ToolChoice](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolchoice) \| None  
The tool choice of the model call that produced the call, as the transcript’s [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent) recorded it. `None` exactly when `input` is `None`.

### AfterToolCall

A tool call that has executed, before the model sees its result.

`input`, `tools`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps for the conversation rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. Monitors and protocols must not mutate them or the messages and tool definitions in them.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L168)

``` python
@dataclass(frozen=True)
class AfterToolCall
```

#### Attributes

`conversation` str  
Id linking this agent’s steps across compactions.

`message` str  
Assistant text alongside the call.

`message_id` str  
Id of the assistant message that made the call, so a monitor model can cite it.

`call` ToolCall  
The call as executed, after any `modify`.

`result` [ChatMessageTool](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessagetool)  
What the model will see, after truncation; `truncated` says whether it was cut.

`output` [ToolResult](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolresult) \| None  
The tool’s untruncated return value, before any cut `truncated` reports. `None` when the host cannot see it: beside a proxy, which sees only the `result` the model is sent, and in a replay from a log, which records only that `result`. In-process Inspect always provides it. Judge what the model saw by `result`, which every host provides.

`view` ToolCallView  
Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise.

`input` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when no model call produced the assistant message (a pre-filled message, a human agent, or one the scaffold built itself) or when the transcript event that recorded the model call is no longer available; it is never approximated from `agent_messages`. `context.eval.sample_input` is the sample’s input, the assignment.

`agent_messages` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
The outermost running agent function’s own message list, read live by reference, or `None`, as for `BeforeToolCall.agent_messages`. The agent may not have added `result` yet. Monitors must not mutate it.

`history` Sequence\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\] \| None  
Every message this agent instance has sent to the model so far, in the order sent, or `None` when the host cannot produce it, as for `BeforeToolCall.history`. It includes the input of the call that produced this tool call; `result` is not in it yet. Monitors must not mutate them.

`human_decisions` Sequence\[HumanDecision\] \| None  
Decisions a person made through `human()` at earlier steps of this agent’s conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent’s conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls’ tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling’s `decide_final()` superseded is dropped); it never holds a monitor’s observation or another protocol’s decision. A prompt the person cancelled or closed without choosing is recorded as the surface’s fail-closed answer, with an explanation that says so; render it, never as the person’s judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them.

`tools` Sequence\[[ToolInfo](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolinfo)\] \| None  
The tool definitions offered to the model call that produced the call, as the transcript’s [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent) recorded them, after any model hooks (so they can differ from that call’s `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` exactly when `input` is `None`; it is never approximated from the tools available now. Monitors must not mutate them.

`tool_choice` [ToolChoice](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolchoice) \| None  
The tool choice of the model call that produced the call, as the transcript’s [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent) recorded it. `None` exactly when `input` is `None`.

`truncated` Truncation \| Literal\[False\] \| None  
Whether the host cut the tool’s output before the model saw it in `result`. A `Truncation` when it did; `False` when it did not, which is always so for structured content; `None` when this host cannot tell (a proxy, a replay from messages alone, or a log older than the record of cuts). `if step.truncated:` is true only for a cut. It is built from the value the transcript’s `ToolEvent.truncated` records (`Truncation(*event.truncated)` when that is set), so a monitor reads the same answer live and from a log. A cut made before the host’s limit applies, by a sandbox output limit or a timeout, is not reported here: `result.error` reports it, with type `"limit"` or `"timeout"`. Use this field, not the “output was too long” notice in `result`, to tell whether the host cut the result: that notice is text a tool’s output can contain, so it can be forged.

### Truncation

A cut the host made to a tool’s output before the model saw it: `AfterToolCall.truncated` when there was one.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L116)

``` python
@dataclass(frozen=True)
class Truncation
```

#### Attributes

`original_bytes` int  
The size of the tool’s output, in UTF-8 bytes of its text.

`limit_bytes` int  
The limit the output was cut to, in bytes.

### HumanDecision

A decision a person made through `human()` at an earlier step of the agent’s conversation. `human_decisions` on a step holds them, and `human_decision_for()` finds the one covering a call.

Use it to annotate the past call it covers when rendering the conversation into a prompt, such as “reviewed by a person: approved”, so a monitor does not flag already-reviewed history again. It is not a standing permission: never lower suspicion of a new call because it resembles one a person approved. The same command can mean something else once the state has changed (an `rm -rf build/` approved in one working directory deletes something else after a `cd`), and an agent can get a benign call approved and then issue a similar harmful one.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L13)

``` python
@dataclass(frozen=True)
class HumanDecision
```

#### Attributes

`stage` Literal\['model_input', 'model_output', 'tool_call', 'tool_result'\]  
The stage of the step it was made at, as `SentinelEvent.stage` records it: `tool_call`, `tool_result` or `model_output`, the stages `human()` asks at.

`step_id` str  
The id of the step it was made at, as `SentinelEvent.step_id` records it: the tool call id at a tool stage, the generate’s id (the `uuid` of its first [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)) at `model_output`.

`call_ids` tuple\[str, ...\]  
The tool calls the decision covered: the step’s call at a tool stage, and every call the turn makes at `model_output`. How a prompt says which calls a person approved. Empty for a review of a turn with no tool calls, which has no key to annotate a message by: `step_id` is then the generate’s id, which no message in the conversation carries.

`decided` Reported\[Decision\]  
The `human()` instance that asked (`name`, `path`, `function`) and a detached copy of its decision, which carries nothing of the step it was made at beyond its own fields: `approve` is a `proceed`, and the person’s reason is the `explanation` (and, for a `reject`, the `message`). The reason is untrusted text, which can quote the agent, so quote it when rendering it into a prompt.

It is what the person decided, not what happened to the step: beside other protocols in a `concurrent()`, a peer’s `reject` outranks the person’s `approve`, so a call a person approved may not have run. Mark only calls that ran, which have a result without an error in the history (a call rejected before it ran also gets a tool message, with an `approval` error, so a tool message alone does not prove the call ran).

A prompt the person cancelled or closed without choosing is recorded as the surface’s fail-closed answer, a `reject` or `terminate`, and its `explanation` says so (for example “ACP client cancelled the permission request.”). Render the explanation, and do not present such a decision as the person’s judgment.

### human_decision_for

The latest decision a person made through `human()` that covers the tool call `call_id`, from `step.human_decisions`.

A call can be decided about more than once: its turn at `model_output` covers each of its calls, then the call itself before it runs and its result after, so the latest is the most specific. Look decisions up by call id, which the model’s provider assigns and which stays the same however a scaffold copies or rebuilds its messages; message ids do not, so there is no lookup by message, and a review of a turn with no tool calls, which stays in `step.human_decisions`, is found by none. None when no decision covers the call, and also when `step.human_decisions` is None because the host cannot tell; check that field to tell the two apart. Use the result to annotate that exact call, not to judge a new call that resembles it (see `HumanDecision`).

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L222)

``` python
def human_decision_for(step: Step, call_id: str) -> HumanDecision | None
```

`step` Step  
The step being examined.

`call_id` str  
The id of a tool call earlier in the conversation, as `ToolCall.id`.

### Step

The union, for code that handles any stage: a protocol, or a host.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_step.py#L218)

``` python
Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
```

### ReplayStep

A step rebuilt from a stored transcript, for replaying a sentinel over it: the step a monitor is given, with what identifies it in the transcript and what was recorded at it.

Replay builds them in the order the steps ran live. The sample’s `Context` is the same for every step of a transcript, so the replay holds it rather than each step. Only `step` reaches monitors and protocols; the other fields are the replay’s own.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_replay_step.py#L12)

``` python
@dataclass(frozen=True)
class ReplayStep
```

#### Attributes

`id` str  
The step id, as `SentinelEvent.step_id` records it live: the tool call id at a tool stage, and the generate’s id (the `uuid` of its first [ModelEvent](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)) at both generate stages. Unique within its conversation for a stage, so a label made against it serves replay and the live record alike; two agents’ tool calls can share an id, since a provider makes it.

`stage` Literal\['model_input', 'model_output', 'tool_call', 'tool_result'\]  
The stage of `step`, as `SentinelEvent.stage` records it: `model_input` for a `BeforeGenerate`, `model_output` for an `AfterGenerate`, `tool_call` for a `BeforeToolCall` and `tool_result` for an `AfterToolCall`.

`step` Step  
The step, as the live host would have built it, less what the transcript does not hold: for a step from a log, `AfterToolCall.output` is `None`.

`segment` tuple\[str, int\]  
Where the step falls in its conversation: the conversation (`step.conversation`), and how many of that conversation’s compactions came before the step. Steps with the same segment ran between the same two compactions.

`recorded_answers` Mapping\[str, HumanAnswer\]  
The answers a person gave through `human()` at this step when it ran, keyed by the path of the `human()` instance that asked, for the replay host to give again when the same instance asks. Empty when none were recorded. It is the replay’s own record and never part of `step`, so monitors and protocols do not see it. Read-only: a copy of the mapping given.

`depth` int \| None  
How deep the step’s conversation is in the transcript’s timeline, counted as Inspect Scout’s `llm_scanner(depth=)` counts: only spans that make model calls of their own and are not utility spans count a level, so 1 is the outermost such span on each branch. It is the shallowest level of any span holding one of the conversation’s model calls. `None` where none of them is in such a span, such as a lone call Scout’s timeline leaves out of every span, or where the step’s builder had no timeline.

## Report types

### Observation

Evidence for a protocol to calibrate and threshold. What a monitor returns.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L27)

``` python
class Observation(BaseModel)
```

#### Attributes

`suspicion` Suspicion  
How suspicious. Required: an observation with no score says nothing.

`explanation` str \| None  
Why, for a reader of the transcript.

`references` list\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  
Messages and events in the transcript the report points at, which Inspect View links from cites such as `[M22]` in `explanation`.

`metadata` dict\[str, Any\] \| None  
Author-supplied structured context, recorded verbatim. `Host.score` adds a `left_out` entry here when it cuts the oldest shown messages to fit the prompt.

#### Methods

score  
An observation that sets `suspicion` to the given score.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L42)

``` python
@classmethod
def score(
    cls,
    suspicion: Suspicion,
    explanation: str | None = None,
    *,
    references: Sequence[Reference] = (),
    metadata: dict[str, Any] | None = None,
) -> Observation
```

`suspicion` Suspicion  

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

`metadata` dict\[str, Any\] \| None  

flag  
A detector’s observation: suspicion `1.0`, for a protocol to act on or not.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L59)

``` python
@classmethod
def flag(
    cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
) -> Observation
```

`explanation` str \| None  

`references` Sequence\[[Reference](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#reference)\]  

### Decision

What should happen at a step. What a protocol returns; advisory to any protocol wrapping it unless passed to `decide_final()`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L91)

``` python
class Decision(BaseModel)
```

#### Attributes

`action` Action  
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
Let the step proceed unchanged: a decision with action `proceed`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L117)

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

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L126)

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

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L142)

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

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L151)

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

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L12)

``` python
Suspicion: TypeAlias = SentinelSuspicion
```

### Action

What a protocol can decide about the step it examined.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L15)

``` python
Action: TypeAlias = SentinelAction
```

### Report

What the transcript records.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L176)

``` python
Report: TypeAlias = Observation | Decision
```

### Reported

A report with the identity of the configured instance that produced it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L182)

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

### Failed

A monitor function that raised instead of reporting, with the identity of its configured instance.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L203)

``` python
@dataclass(frozen=True)
class Failed
```

#### Attributes

`name` str  
Instance name, as in `Reported.name`.

`path` str  
Instance path, e.g. `attempt/internet_attempt`.

`function` str  
The `__name__` of the function that raised.

`error` Exception  
What it raised.

## Monitors and protocols

`Protocol` is a control protocol, not `typing.Protocol`.

### monitor

Register a monitor factory.

Use as `@monitor`, or as `@monitor(name=..., version=..., portable=...)`. The factory returns a function that must be `async`, take `(context, step)`, annotate `step` with exactly one stage payload, and be annotated `-> Observation`, or `-> Observation | None` if it can abstain. These are checked when the factory is called. Each function watches one stage, so `Step` or any other union is rejected; to watch several, return a `MonitorGroup` of functions, which together are one configured instance. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A factory may instead return an instance of its kind that another factory configured, such as `sequential([...])` from a `@protocol` factory, to register that composition under its own name: it runs as configured, its records and configuration entry carry this factory’s name and params, and a configuration rebuilds it by calling this factory. The returned instance is guarded afresh, not cloned: it keeps its attributes other than its registry name and params and its step types, which this factory replaces, and it shares its closure and children with the instance returned, so a factory that returns one shared instance shares that state across its configurations. Since the configuration records only this factory, bump this factory’s `version` when the composition it returns changes behaviour. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

A configured function runs only through the runners while a sentinel is running: a protocol that calls one directly from its body raises `RuntimeError`, and must call it through `run_monitors` or `run_children` instead. Outside a run, as in a unit test, a direct call works.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_decorators.py#L141)

``` python
def monitor(
    factory: Callable[P, Monitor | MonitorGroup] | None = None,
    /,
    *,
    name: str | None = None,
    version: int = 0,
    portable: bool = True,
) -> Callable[P, Monitor | MonitorGroup] | _MonitorDecorator
```

`factory` Callable\[P, Monitor \| MonitorGroup\] \| None  
A function returning a monitor, or a `MonitorGroup`.

`name` str \| None  
The registered name, in place of the factory’s `__name__`.

`version` int  
The monitor’s version, recorded in its registry metadata so a calibration can say which version it measured; bump it when a change alters the scores. Defaults to 0.

`portable` bool  
Whether the monitor can run outside an eval, in a proxy, recorded in its registry metadata. When True, the default, calling the factory first checks the names its code references, in its body and the functions defined inside it, and raises for inspect_ai other than a fixed list of its message, model output, config, tool and content types, environment variables, processes and threads, direct networking, and packages with compiled code. Helpers defined outside the factory are not followed. A factory with no source, as in the plain REPL before Python 3.13 or `exec()`’d code, is not checked; one whose source cannot be parsed or located is reported as not checked. False opts out, for a monitor that only runs in an eval.

### protocol

Register a protocol factory.

Same contract as `@monitor`, with each returned function taking `(context, step, escalations)` and annotated `-> Decision`, or `-> Decision | None` if it can abstain. `escalations`, annotated `Escalations` (`tuple[Reported[Decision], ...]`), holds the escalations of the links before this one in a `sequential()`, in the order they escalated, so a protocol after them sees who escalated and why; it is empty anywhere else, since escalations do not cross layers. A monitor is never handed them, so its observation does not depend on what another monitor or protocol decided. `step` may also be annotated `Step` for a protocol that runs at every stage, such as a composition that only forwards the step to its children. A function never dispatches on the payload type: a union of some stages but not all is rejected, so write a `ProtocolGroup` with one function per stage instead. A union of every stage written out is `Step`. To configure several functions as one instance, return a `ProtocolGroup`; a function among them that returns `terminate` or calls `decide_final()` ends the instance’s run, and those after it do not run and are not recorded. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

As with `@monitor`, a configured function called directly from another protocol’s body during a run raises `RuntimeError`; call it through `run_protocols` or `run_children`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_decorators.py#L190)

``` python
def protocol(
    factory: Callable[P, Protocol | ProtocolGroup] | None = None,
    /,
    *,
    name: str | None = None,
    version: int = 0,
    portable: bool = True,
) -> Callable[P, Protocol | ProtocolGroup] | _ProtocolDecorator
```

`factory` Callable\[P, Protocol \| ProtocolGroup\] \| None  
A function returning a protocol, or a `ProtocolGroup`.

`name` str \| None  
The registered name, in place of the factory’s `__name__`.

`version` int  
The protocol’s version, recorded in its registry metadata; bump it when a change alters its decisions. Defaults to 0.

`portable` bool  
Whether the protocol can run outside an eval, checked when the factory is called as for `@monitor`. Children, passed in as arguments or configured inside the factory, are checked by their own factories, not as part of this one. Defaults to True; False opts out.

### Monitor

A monitor: observes a step at one stage and reports a suspicion score, or abstains.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L10)

``` python
Monitor: TypeAlias = (
    Callable[[Context, BeforeGenerate], Awaitable[Observation | None]]
    | Callable[[Context, AfterGenerate], Awaitable[Observation | None]]
    | Callable[[Context, BeforeToolCall], Awaitable[Observation | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Observation | None]]
)
```

### Protocol

A control protocol, not `typing.Protocol`: decides what happens at a step, or abstains. Annotating `Step` runs it at every stage. Its third parameter is the escalations earlier links of a `sequential` handed it, empty anywhere else.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L18)

``` python
Protocol: TypeAlias = (
    Callable[[Context, Step, Escalations], Awaitable[Decision | None]]
    | Callable[
        [Context, BeforeGenerate, Escalations],
        Awaitable[Decision | None],
    ]
    | Callable[
        [Context, AfterGenerate, Escalations],
        Awaitable[Decision | None],
    ]
    | Callable[
        [Context, BeforeToolCall, Escalations],
        Awaitable[Decision | None],
    ]
    | Callable[
        [Context, AfterToolCall, Escalations],
        Awaitable[Decision | None],
    ]
)
```

### Escalations

The escalations a protocol is handed as its third parameter: those of the links before it in a `sequential()`, in the order they escalated, each with the escalating instance’s name and its decision. Empty anywhere else, since escalations do not cross layers. A monitor is never handed them.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_report.py#L199)

``` python
Escalations: TypeAlias = tuple[Reported[Decision], ...]
```

### MonitorGroup

Several monitor functions that form one monitor instance.

Return one from a `@monitor` factory in place of a single function. Together the functions are one configured instance, sharing its name, path and `store_as` namespace; each runs, in the order given, at the stage it watches, sequentially since they share one store, and each report records which `function` made it. A group is not callable: hand it to a protocol, which runs it through `run_monitors` or `run_children`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L64)

``` python
@final
class MonitorGroup(Group)
```

#### Attributes

`functions` tuple\[Monitor, ...\]  
The member functions, in the order they run.

#### Methods

\_\_init\_\_  
Group monitor functions into one instance.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L71)

``` python
def __init__(self, *functions: Monitor) -> None
```

`*functions` Monitor  
At least one monitor function, each with a distinct `__name__`.

### ProtocolGroup

Several protocol functions that form one protocol instance.

Return one from a `@protocol` factory in place of a single function. As with `MonitorGroup`, the functions share one name, path and `store_as` namespace and run in the order given at the stages they accept; a function that returns `terminate` or calls `decide_final()` ends the instance’s run, and those after it do not run and are not recorded. A group is not callable: hand it to a protocol, which runs it through `run_protocols` or `run_children`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L88)

``` python
@final
class ProtocolGroup(Group)
```

#### Attributes

`functions` tuple\[Protocol, ...\]  
The member functions, in the order they run.

#### Methods

\_\_init\_\_  
Group protocol functions into one instance.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L95)

``` python
def __init__(self, *functions: Protocol) -> None
```

`*functions` Protocol  
At least one protocol function, each with a distinct `__name__`.

### Monitors

Monitors handed to a protocol, named by mapping key or by registry name without its package prefix.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L112)

``` python
Monitors: TypeAlias = (
    Mapping[str, Monitor | MonitorGroup] | Sequence[Monitor | MonitorGroup]
)
```

### Protocols

Protocols handed to a protocol, named the same way.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L117)

``` python
Protocols: TypeAlias = (
    Mapping[str, Protocol | ProtocolGroup] | Sequence[Protocol | ProtocolGroup]
)
```

### Sentinel

A monitor or protocol, or a group of either.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L122)

``` python
Sentinel: TypeAlias = Monitor | MonitorGroup | Protocol | ProtocolGroup
```

### Sentinels

What `Task(sentinel=)` and the compositions accept: one sentinel, or a sequence or mapping of them, named by mapping key or by registry name without its package prefix.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_types.py#L125)

``` python
Sentinels: TypeAlias = Sentinel | Mapping[str, Sentinel] | Sequence[Sentinel]
```

### PortabilityError

A portable monitor or protocol references something a portable function cannot use.

Raised when the factory of a `@monitor` or `@protocol` declared with `portable=True` (the default) is called, before it runs. The message lists each reference in the factory’s code that a portable function cannot use: the file and line, the function, the name, what it resolved to and why. A factory whose source exists but cannot be parsed or located is reported as not checked. It subclasses `TypeError`, like the factory’s other configuration checks.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_portable.py#L134)

``` python
class PortabilityError(TypeError)
```

## Shipped protocols

### observe_only

Record every monitor’s observation and never act.

A monitor that raises is recorded as failed and logged as a warning, once per instance and exception type, and the step continues.

How a monitor runs when nothing should act on its scores, such as while a benign score distribution is collected; `Task(sentinel=)` takes no bare monitors, so the configuration says that the sentinel enforces nothing.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_protocols/observe_only.py#L15)

``` python
@protocol
def observe_only(monitors: Monitor | MonitorGroup | Monitors) -> Protocol
```

`monitors` Monitor \| MonitorGroup \| Monitors  
The monitor, or monitors, to run at every step they watch.

### concurrent

Run every child at once; the strictest decision wins.

What a list containing a protocol resolves to. Monitors are recorded in the transcript, a monitor that raises recorded as failed and logged as a warning while the others go on; protocols vote by `terminate > reject > modify > escalate > proceed`, the first in configuration order winning a tie, so a peer’s `proceed` does not override an `escalate` and the layer escalates when that is the strongest vote. A `modify` is honoured beside a `proceed` or an `escalate`, so a peer with no objection to the call does not turn a rewrite into a block; the rewritten call is not checked again by the protocols that proceeded, as with a lone modifier. Only another `modify` contests it, from any instance or from another function of the modifier’s own: two rewrites conflict, so the layer returns a `reject` naming the modifiers, carrying the first modifier’s `audit` and `metadata`, its explanation leading with that modifier’s. A `reject` or `terminate` outranks a `modify` anyway. When more than one protocol decided and not all of them proceeded, the layer’s explanation lists each one’s decision after the winner’s own, labelling a function of a `ProtocolGroup` or `MonitorGroup` `name.function`. A child that calls `decide_final()` ends the step and no vote is taken. Its protocol children are handed no escalations, even inside a `sequential()`: escalations do not cross layers.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_protocols/concurrent.py#L13)

``` python
@protocol
def concurrent(
    children: Sentinels,
) -> Protocol
```

`children` Sentinels  
A monitor, protocol or group, or several, to run together.

### sequential

Run children one at a time, in order; the first that decides anything but `escalate` decides, and later children do not run.

Each child is run through the runner and recorded under its own name, and a child that does not watch the current stage is not invoked. An `escalate` hands the step to the next child with the escalation appended to the `escalations` that protocol is handed, so a person at the end sees who asked and why; a monitor is never handed them. The list starts empty for each run of the chain: escalations do not cross layers, so a nested `sequential` does not see the escalations of the chain around it, and the chain ignores any it is handed itself. The functions of one `ProtocolGroup` run in one call and are all handed the escalations from before any of their own.

A monitor is recorded and falls through, since an observation is not a decision; one that raises is recorded as failed, logged as a warning, and falls through the same way. When every child that decided escalated, the chain returns the last escalate, passing it up as `concurrent` does; when only monitors reported, it returns `proceed`; when no child took part, `None`. A child that calls `decide_final()` ends the step as usual.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_protocols/sequential.py#L13)

``` python
@protocol
def sequential(
    children: Sentinels,
) -> Protocol
```

`children` Sentinels  
A monitor, protocol or group, or several, to run in order.

### human

Ask a person to decide, every time the step reaches this protocol.

The person is asked through `Host.ask_human`, which is handed the step and the escalations this protocol was handed and shows them together, so at the end of a `sequential()` the person sees who escalated and why. On its own, `human(stages=["tool_call"])` asks about every call. After a generate, the person sees the whole assistant turn, its text and every tool call it makes, before any of the calls run. The answer maps to a decision: `approve` to `proceed` explained by the person’s reason, so the log records why the step went ahead, `reject` to `reject` with the person’s reason as both the `message` the agent reads and the `explanation`, `terminate` to `terminate` explained by the reason, and `modify` to a `modify` with the person’s replacement call.

The answer is an ordinary decision: it ends a `sequential()` as any link’s decision does, and beside other protocols in a `concurrent()` it is one vote, so another protocol’s `reject` outranks the person’s `approve`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_protocols/human.py#L30)

``` python
@protocol
def human(
    stages: Sequence[str],
    choices: Sequence[str] | None = None,
) -> Protocol | ProtocolGroup
```

`stages` Sequence\[str\]  
The stages to ask at, as `SentinelEvent.stage` records them: `"tool_call"` (before a tool call), `"tool_result"` (after it, with the result the model is about to receive) and `"model_output"` (after a generate by the agent’s model, before its output reaches the agent).

`choices` Sequence\[str\] \| None  
What the person may pick, from `approve`, `reject`, `terminate` and `modify`. Defaults to `approve`, `reject` and `terminate` before a call, and `approve` and `terminate` after a call and after a generate, where `reject` and `modify` are not legal.

### handle_escalation

Answer the escalations handed to it with a fixed policy: end the sample, or let the step proceed.

Put it last in a `sequential()`, after the links that escalate what they are unsure about, as in `sequential([threshold(...), handle_escalation("terminate")])`, or after a `human()` for when the person cannot be reached. It decides only when it is handed escalations and abstains otherwise, so at the root, or anywhere nothing escalated, it does nothing. Its explanation names each escalation it handled, `escalated by triage: call deletes files (rm)`, and it carries their references.

An escalate that reaches the root with nothing to handle it ends the sample, `run_sentinel` returning a `terminate` explained as an unhandled escalation; `handle_escalation()` is how a configuration states a fixed policy instead. `"proceed"` lets an unsure escalation through, recorded in the configuration and the log. It runs before and after tool calls and the agent’s model calls, but never makes a stage watched on its own: a stage runs only when something else in the configuration watches it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_protocols/handle_escalation.py#L14)

``` python
@protocol
def handle_escalation(action: Literal["terminate", "proceed"]) -> ProtocolGroup
```

`action` Literal\['terminate', 'proceed'\]  
What to do when the step was escalated: `"terminate"` to end the sample, or `"proceed"` to let the step proceed. Required, so every configuration states its policy.

### threshold

Act on suspicion scores: each threshold is a floor, so at or above `escalate_at` the step is at least escalated, at or above `reject_at` at least rejected, and at or above `terminate_at` the sample is ended.

The decision is the strongest action whose threshold any score reaches, `terminate` over `reject` over `escalate`, and `proceed` when none is reached. When no monitor reported at a step it also decides `proceed`, explained as “no monitor reported”, rather than abstaining. Give at least one threshold; to record scores without acting on them, use `observe_only()`. An escalation needs a `human()` or `handle_escalation()` after it in a `sequential()`, as in `sequential([threshold(monitors, escalate_at=0.6, terminate_at=0.9), human(stages=["tool_call"])])`; one that reaches the root with nothing to handle it ends the sample. Since `threshold` always decides at a stage it runs, it ends the chain there unless it escalates, so the person is asked only when it escalates, not about the steps its monitors skip; a link after it that is a fallback, such as another `threshold` or a rule, no longer runs at those steps either. Its `proceed` is a vote like any other, and a `proceed` does not contest a `modify`, so in a `concurrent()` beside a protocol that modifies the call the rewrite takes effect, whether its monitors scored low or did not report, and the threshold does not check the rewritten call.

A threshold is one number for every score, or a mapping that sets one per dimension. In a mapping, `"<monitor>.<dimension>"` applies to one dimension of one monitor (the monitor’s instance name, as in `Reported.name`), a bare `"<dimension>"` applies to that dimension from every monitor, a monitor reporting a single number is keyed by its instance name, and `"*"` applies to anything unlisted. The most specific key wins: qualified, then bare, then `"*"`. A score no key covers never triggers. A number is shorthand for `{"*": number}`. A key containing `.` is always `"<monitor>.<dimension>"`, qualified by the longest monitor instance name that prefixes it, and is an error if none does: it is never also a bare dimension, so with monitors `a` and `a.b` the key `"a.b.c"` is dimension `c` of `a.b`, and a dimension containing `.` is keyed through its monitor. A threshold that can never apply, because a stronger action’s threshold is at or below it for every score it covers, is a configuration error: `escalate_at=0.9` beside `terminate_at=0.8` would never escalate. A key that one mapping lists and another does not is not such a conflict.

It runs at each stage its monitors watch, before and after tool calls and the agent’s model calls, and combines monitors that watch different stages, each thresholded at its own: at a step, only the monitors watching that stage report. A `reject` is legal only before a tool call, so with `reject_at` every monitor function must watch only `BeforeToolCall`, and `threshold` runs only there.

A decision is explained for the log: it names the highest-scoring score that reached the decision’s threshold (the first in configuration order on a tie) with its monitor’s explanation, e.g. `suspicion 0.95 from audit: posts credentials to an external host` when the threshold came from `"*"` or the score is a single number, and `sabotage 0.72 ≥ 0.60 from audit: edits the tests` when it came from the dimension’s own key. It carries every reporting monitor’s references, and its `metadata["observations"]` lists every monitor that reported at the step, `{"monitor": name, "suspicion": score, "explanation": text}` in configuration order, so they reach a `human()` after it in the escalations it is handed, where a host can show each score and where the monitors disagree; inspect_ai’s approval card shows only the explanation until it renders them. None sets `message`, so on a reject the agent gets the host’s default “not approved” text and learns nothing about what the monitors saw.

The smallest protocol, and the one calibration feeds: the constants come from a benign score distribution, which is why they live here and not on a monitor.

A monitor that raises fails the step: `threshold` does not decide without one of its monitors, so the `MonitorFailedError` reading its observations propagates and the sample fails, with the monitor’s failure in the transcript.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_protocols/threshold.py#L19)

``` python
@protocol
def threshold(
    monitors: Monitor | MonitorGroup | Monitors,
    *,
    escalate_at: float | Mapping[str, float] | None = None,
    reject_at: float | Mapping[str, float] | None = None,
    terminate_at: float | Mapping[str, float] | None = None,
) -> Protocol | ProtocolGroup
```

`monitors` Monitor \| MonitorGroup \| Monitors  
The monitor, or monitors, whose scores are thresholded.

`escalate_at` float \| Mapping\[str, float\] \| None  
Suspicion at or above which the step is escalated, to be decided by what follows in a `sequential()`: one number, or a mapping by dimension. `None` never escalates.

`reject_at` float \| Mapping\[str, float\] \| None  
Suspicion at or above which the call is rejected, in the same form; `None` never rejects. Legal only when every monitor function watches only `BeforeToolCall`.

`terminate_at` float \| Mapping\[str, float\] \| None  
Suspicion at or above which the sample is ended, in the same form; `None` never terminates.

## Runner

### run_monitors

Run monitors concurrently and collect their observations in configuration order.

A typed shortcut for `run_children` over monitors only: its parameter rejects a protocol and its return type is the observations alone.

Derives each child’s context under this layer’s path and records every observation, including the ones the caller goes on to ignore. A monitor that abstained or does not watch this stage contributes nothing, so the result may be empty. An instance whose factory returned several functions contributes one observation per function that reported, in the order the factory returned them.

A monitor function that raises does not fail the call: its failure is recorded through `Recorder.failed` and listed in the result’s `failed`, and reading the result’s observations then raises `MonitorFailedError`, so a caller that reads them fails unless it checks `failed` first. A [LimitExceededError](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#limitexceedederror) is not a monitor failure; it propagates, as a cancellation does.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_runner.py#L172)

``` python
async def run_monitors(
    monitors: Monitor | MonitorGroup | Monitors, context: Context, step: Step
) -> Observations
```

`monitors` Monitor \| MonitorGroup \| Monitors  
One monitor or `MonitorGroup`, named by its registry name without the package prefix; a sequence of them, named the same way; or a mapping of instance names to them.

`context` Context  
This layer’s context, the one this protocol was given. Call this while the protocol runs; after it returns this raises `RuntimeError`.

`step` Step  
The step being examined.

### run_protocols

Run protocols concurrently and collect their decisions in configuration order, cancelling the rest at their next await when one returns `terminate` or calls `decide_final()`.

A typed shortcut for `run_children` over protocols only: its parameter rejects a monitor and its return type is the decisions alone. A `decide_final()` from any protocol at any depth below propagates out of this call, so the caller’s own decision logic does not run.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_runner.py#L192)

``` python
async def run_protocols(
    protocols: Protocol | ProtocolGroup | Protocols,
    context: Context,
    step: Step,
    *,
    escalations: Sequence[Reported[Decision]] = (),
) -> Decisions
```

`protocols` Protocol \| ProtocolGroup \| Protocols  
One protocol or `ProtocolGroup`, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.

`context` Context  
This layer’s context, the one this protocol was given. Call this while the protocol runs; after it returns this raises `RuntimeError`.

`step` Step  
The step being examined.

`escalations` Sequence\[Reported\[Decision\]\]  
The escalations to hand each protocol, as `sequential` hands a link those of the links before it. Defaults to none: escalations do not cross layers unless a protocol passes them on.

### run_children

Run monitors and protocols together in one task group, cancelling the rest at their next await when a protocol returns `terminate` or calls `decide_final()`.

Runs any mix of children; `run_monitors` and `run_protocols` are typed shortcuts for one family. A child cancelled this way is recorded through `Recorder.cancelled`; one that finishes without awaiting is recorded normally.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_runner.py#L215)

``` python
async def run_children(
    children: Sentinels,
    context: Context,
    step: Step,
    *,
    escalations: Sequence[Reported[Decision]] = (),
) -> Reports
```

`children` Sentinels  
One monitor, protocol or group, a sequence of them, or a mapping of instance names to them, named as for `run_monitors`.

`context` Context  
This layer’s context, the one this protocol was given. Call this while the protocol runs; after it returns this raises `RuntimeError`.

`step` Step  
The step being examined.

`escalations` Sequence\[Reported\[Decision\]\]  
The escalations to hand each protocol among the children, as for `run_protocols`; a monitor is never handed them. Defaults to none.

### decide_final

End the step with this decision.

Nothing above the calling protocol runs; siblings still in flight are cancelled and recorded as such. First call wins if two race.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_final.py#L25)

``` python
def decide_final(decision: Decision) -> NoReturn
```

`decision` Decision  
The outcome for this step.

### Observations

What a layer’s monitors reported, in configuration order.

When a monitor failed, reading the observations raises, so a protocol that reads them fails closed rather than deciding without the failed monitor. A protocol that decides anyway checks `failed` and reads `succeeded`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_results.py#L45)

``` python
class Observations(_ReportSequence[Observation])
```

#### Attributes

`failed` tuple\[Failed, ...\]  
The monitor functions that raised instead of reporting, in configuration order.

`succeeded` Observations  
The observations of the monitors that did not fail, readable whether or not any failed.

#### Methods

max_suspicion  
The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_results.py#L98)

``` python
def max_suspicion(self) -> float | None
```

### Decisions

What a layer’s protocols decided, in configuration order.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_results.py#L141)

``` python
class Decisions(_ReportSequence[Decision])
```

#### Methods

strongest  
The strongest decision by `terminate > reject > modify > escalate > proceed`, the first in configuration order on a tie; `None` if nobody decided.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_results.py#L144)

``` python
def strongest(self) -> Reported[Decision] | None
```

### Reports

Both families of report from one layer, unpackable as `observations, decisions = await run_children(...)`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_results.py#L151)

``` python
class Reports(NamedTuple)
```

#### Attributes

`observations` Observations  
From the layer’s monitors.

`decisions` Decisions  
From the layer’s protocols.

### MonitorFailedError

Raised on reading the observations of a layer in which a monitor failed. Check `Observations.failed` first to decide without the failed monitors.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_results.py#L109)

``` python
class MonitorFailedError(RuntimeError)
```

## Context and host

### Context

What a monitor or protocol is given. Nothing in it came from the agent, except what `proxy` holds.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_context.py#L95)

``` python
@dataclass(frozen=True, kw_only=True)
class Context
```

#### Attributes

`path` str  
Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer.

`host` Host  
Inference through the host’s models, and asking a person. See `Host`.

`proxy` ProxyContext \| None  
What a proxy says about the model request the step was found in, when the sentinel runs beside a proxy. None in an eval’s own process.

`eval` EvalContext \| None  
The task and sample being run, in an Inspect eval: the host’s `Host.eval`. None outside an eval, for example when a proxy runs the sentinel on requests that have no task, sample or epoch.

#### Methods

store_as  
Typed view of this instance’s state, namespaced by `path`.

Renaming the instance (its mapping key) or wrapping it in another layer changes `path`, so its state moves with it. The store is the one the host supplied for the running step, so call this while the monitor or protocol runs; outside a step it raises `RuntimeError`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_context.py#L113)

``` python
def store_as(self, model_cls: type[SMT]) -> SMT
```

`model_cls` type\[SMT\]  
The [StoreModel](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#storemodel) subclass to read and write through.

### EvalContext

The Inspect eval sample a monitor or protocol is running in. Constant for the sample; nothing in it came from the agent.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_context.py#L16)

``` python
@dataclass(frozen=True, kw_only=True)
class EvalContext
```

#### Attributes

`task` str  
Task name.

`task_description` str \| None  
What agents in this setting are asked to do: the task’s `description`. None when the task sets none or the host does not provide it.

`sample_id` str \| int  
Dataset sample identifier.

`epoch` int  
Epoch of the sample: 1 unless the task runs several.

`sample_description` str \| None  
What is distinctive about this instance: the sample’s `description`. None when the sample sets none or the host does not provide it.

`sample_input` str \| list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
The sample’s input: the assignment the agent was given. Judge a step against `step.input`, which is exactly what the model was sent (`None` at a tool stage no model call produced); use this for what the agent was asked to do.

`metadata` dict\[str, Any\]  
Task metadata merged with sample metadata.

`target` [Target](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#target) \| None  
The expected answer. None until monitors and protocols can opt in with `target=True`.

`sample_input_text` str  
`sample_input` as one string: `sample_input` itself if a string, else its messages’ text joined with newlines.

### ProxyContext

What a proxy says about the model request a step was found in. The caller set the headers, so with an agent that makes its own model calls, they came from the agent.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_context.py#L75)

``` python
@dataclass(frozen=True, kw_only=True)
class ProxyContext
```

#### Attributes

`provider` str  
Whose API the request is written for, e.g. `anthropic`.

`model` str  
The model, by the name the proxy shows for it.

`user` str \| None  
Who the proxy signed the caller in as, if it says.

`headers` Mapping\[str, str\]  
The request headers the proxy passed on, by lowercase name.

`inspect` InspectSample \| None  
The sample the request says it is from, when an Inspect eval sent it. Read it when `Context.eval` is None because the sample’s task was not registered.

### InspectSample

The Inspect sample a request says it is from, as its `x-inspect-context` header says. The client set the header, so this is its claim.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_context.py#L52)

``` python
@dataclass(frozen=True, kw_only=True)
class InspectSample
```

#### Attributes

`eval_id` str  
The id of the eval, as its log records it.

`task` str  
The task’s name.

`sample_id` str \| int  
The sample’s id in the dataset.

`epoch` int  
The epoch, from 1.

`sample_uuid` str  
The uuid the eval log gives the sample. It stays the same when Inspect retries the sample after an error.

`attempt` str  
Inspect’s id for this attempt at the sample. A retry has a new one.

### Host

What a monitor or protocol may do to the outside world. Author-facing.

Inference and asking a person are the only effects for now. Outbound HTTP through named endpoints (`fetch`) is planned. Meanwhile a monitor running in-process may call inspect_ai APIs directly, such as [sandbox()](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#sandbox), if it is declared `portable=False`.

The host builds it around its `HostNative`; `score` is built on `generate`, so it is the same for every host.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L88)

``` python
class Host(HostNative)
```

#### Attributes

`eval` EvalContext \| None  
The task and sample being run, in an Inspect eval; monitors read it as `Context.eval`. None outside an eval, for example when a proxy runs the sentinel on requests that have no task, sample or epoch.

#### Methods

\_\_init\_\_  
Wrap a host’s own implementation.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L96)

``` python
def __init__(self, native: HostNative) -> None
```

`native` HostNative  
The host’s implementation of the eval, inference and asking a person.

generate  
Generate with a model the host resolves.

`model` is a model name, e.g. `openai/gpt-4o-mini`; `role` is a model role, e.g. `monitor` or `trusted`, which resolves through the host’s role configuration (in an eval, `Task(model_roles=...)` or `--model-role`). A configured `role` wins; otherwise `model`; with neither, the role is `monitor`. When the role is not configured and no `model` is given, what happens is up to the host, and inspect_ai falls back to the agent’s model with a warning. A role is portable, since the deployer picks its model; a model name suits quick experiments.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L109)

``` python
async def generate(
    self,
    input: str | list[ChatMessage],
    *,
    model: str | None = None,
    role: str | None = None,
    tools: list[ToolInfo] | None = None,
    tool_choice: ToolChoice | None = None,
    config: GenerateConfig | None = None,
) -> ModelOutput
```

`input` str \| list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
A prompt string or a list of chat messages.

`model` str \| None  
A model name, used when `role` is not given or not configured.

`role` str \| None  
A model role. Defaults to `monitor` when `model` is None.

`tools` list\[[ToolInfo](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolinfo)\] \| None  
Tool definitions to offer the model.

`tool_choice` [ToolChoice](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#toolchoice) \| None  
Which of `tools` the model may or must call.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig) \| None  
Generation configuration.

input_budget  
How many input tokens a `generate` with the same `model`, `role` and `config` may send: the resolved model’s context window minus the output tokens inspect_ai reserves for the reply. An unknown model falls back to 128k, with a warning. `Host.score` reads it to decide whether a prompt fits.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L140)

``` python
async def input_budget(
    self,
    *,
    model: str | None = None,
    role: str | None = None,
    config: GenerateConfig | None = None,
) -> int
```

`model` str \| None  
A model name, as for `generate`.

`role` str \| None  
A model role, as for `generate`.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig) \| None  
Generation configuration, as for `generate`; its `max_tokens` sets the reply reserve.

count_tokens  
The resolved model’s token count for `input`, the same count a `generate` with this `model`, `role` and `config` would send. `Host.score` counts a prompt against `input_budget` with it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L156)

``` python
async def count_tokens(
    self,
    input: str | list[ChatMessage],
    *,
    model: str | None = None,
    role: str | None = None,
    config: GenerateConfig | None = None,
) -> int
```

`input` str \| list\[[ChatMessage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#chatmessage)\]  
A prompt string or a list of chat messages.

`model` str \| None  
A model name, as for `generate`.

`role` str \| None  
A model role, as for `generate`.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig) \| None  
Generation configuration, as for `generate`.

score  
Ask a model to score a step from 0.0 to 1.0, with an explanation.

The model is asked through `generate`, with the same `model`, `role` and `config`. It is sent a prompt built from `template`, and answers by calling an answer tool whose arguments are checked, so a missing or out-of-range score, or a reply with no answer (or no reply at all), goes back to the model to correct. When the model gives no valid answer within three attempts, `score` raises, and the runner records the monitor as failed. A reply the provider refuses (stop reason `content_filter`) is asked again without using an attempt, up to `retry_refusals` times, after which `score` raises. Call it while the monitor or protocol runs, since it keeps the sample’s nonce in the sample store; outside a step it raises `RuntimeError`.

What the model is shown of the step is set by `preprocessor`: by default all of the model’s input, without system messages, and the step itself; pass a `StepPreprocessor` to window the conversations (`input_window=LastTurns(6)`, `NewSinceLastReport(context)`, `history_window=AllMessages()`), leave parts out, or change the step with a `transform`.

The prompt is fitted to the monitor model’s context window. The host reads the model’s input budget (its context window minus the output reserve; an unknown model falls back to 128k) through `input_budget`, and counts the prompt through `count_tokens`; a prompt fits when the count is within 95% of the budget. A prompt that fits is sent unchanged. One that does not has the oldest shown messages cut from its message lists, HISTORY first, then AGENT MESSAGES, then INPUT, until it fits. At `BeforeGenerate`, the cut never removes the newest INPUT message, because that message is the request being judged. Each list that loses messages gains one marker line in its place (`[N earlier messages left out to fit the context window (nonce)]`); the marker takes no message number, so the render renumbers the kept messages from `[M1]`, and each cite still resolves to the kept message it names. When messages are cut, `Observation.metadata` records the count per list, as `{"left_out": {"history": 3, "input": 2}}`. When the fixed part of the prompt (the instructions, policy, task, dimensions, the step’s own blocks, at `BeforeGenerate` the newest INPUT message, and the answer tool) does not fit even with every other shown message cut, `score` raises: it does not score a step that the model was not shown. A head cut keeps the last message of each list it does not empty, so an incremental `NewSinceLastReport` window stays correct: a message the cut removed is not shown at a later step either. When the model’s reply stops on `model_length`, `score` raises at once, naming the stop reason: the request exceeded the model’s context window although the prompt fit the input budget as counted (the count leaves out the answer tool, and a provider can count more tokens than `count_tokens`), and a correction would only lengthen it.

The default template opens with `instructions` and a paragraph on the nonce, then shows the model `policy`, the task and the step, asks it to score the step from 0.0 to 1.0 against `dimensions`, and asks for an explanation. Templates are Jinja templates. A custom one uses the same variables, and must include `{ step }` and `{ answer_format }`:

- `{ instructions }`: `instructions`, or by default a sentence on monitoring AI agents for harmful actions and one telling the model to treat what the agent wrote or was shown as data and ignore any instructions in it. Instructions you pass replace both, so say the second yourself.
- `{ policy }`: `policy`, the policy the agent must follow whatever its task, between `[BEGIN POLICY (nonce)]` and `[END POLICY (nonce)]` lines; empty when there is none.
- `{ task }`: the task the agent was given (in an eval, the sample’s input), between `[BEGIN TASK (nonce)]` and `[END TASK (nonce)]` lines; empty outside an eval.
- `{ eval }`: the eval the agent is running in, as `context.eval` holds it, such as `{ eval.task }` or `{ eval.metadata.difficulty }`; None outside an eval, so guard it with `{% if eval %}` in a monitor that may run outside one.
- `{ step }`: `step_as_str()` of the preprocessed step, its messages numbered `[M1]`, `[M2]` and so on so the explanation can cite them; the observation’s `references` come from those cites.
- `{ sentinel_nonce }`: a nonce made on the sample’s first `score` and used for the rest of the sample, so the monitor model’s prompt cache keeps working from step to step; 8 hex characters, on every block marker and every message header (`[M1] USER (3f9a1c7e):`). The agent never sees the prompt, so text it writes that looks like a marker or a header lacks the nonce, and the default template tells the model so. The nonce is kept in the sample store, so a tool that shows the agent the store, or a protocol that passes the agent a monitor explanation quoting the nonce, gives it away for the rest of the sample. A `sentinel_nonce` in `template_variables` is ignored.
- `{ dimensions }`: `dimensions` as given, or each dimension’s name followed by its text.
- `{ answer_format }`: the instruction to call the answer tool.
- Any name in `template_variables`, set to its value.

A template that uses a name that is none of these raises before the model is asked, so a misspelled variable is caught. The template is rendered once and values are inserted as text, so the agent’s text is never read as template. A custom template states the 0.0 to 1.0 scale in its own words; without it the answer tool still rejects scores out of range.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L176)

``` python
async def score(
    self,
    step: S,
    dimensions: str | Mapping[str, str],
    *,
    preprocessor: StepPreprocessor[S] | None = None,
    instructions: str | None = None,
    policy: str | None = None,
    template: str | None = None,
    template_variables: Mapping[str, Any] | None = None,
    model: str | None = None,
    role: str | None = None,
    config: GenerateConfig | None = None,
    retry_refusals: int = 3,
) -> Observation
```

`step` S  
The step to score.

`dimensions` str \| Mapping\[str, str\]  
What to score and how. A string is the prompt text for a single score, and the observation’s `suspicion` is one float. A mapping is each dimension’s name to its prompt text, and `suspicion` is a dict keyed by name; a name is 1 to 64 letters, digits, `_`, `.` or `-` (what providers accept as a tool parameter name), must not start with `_`, and must not be `explanation` or an attribute of pydantic’s `BaseModel`.

`preprocessor` StepPreprocessor\[S\] \| None  
What the model is shown of the step; `StepPreprocessor()` when None.

`instructions` str \| None  
The monitor’s own instructions, such as its role and the agent’s setting, in place of the default instructions.

`policy` str \| None  
The policy the agent must follow whatever its task says, such as which files it may change; shown before the task.

`template` str \| None  
A replacement for the default prompt template, a Jinja template.

`template_variables` Mapping\[str, Any\] \| None  
More variables for the template, by name; a name must not be one of the variables above.

`model` str \| None  
A model name, as for `generate`.

`role` str \| None  
A model role, as for `generate`.

`config` [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig) \| None  
Generation configuration.

`retry_refusals` int  
How many times to ask again when the provider refuses to answer.

ask_human  
Ask a person to decide about a step, and wait for the answer.

The host is handed the step itself and renders it by its type, with the `escalations` (who escalated and why): before a tool call the call, its view and the conversation; after a call the same with the result the model is about to receive; after a generate the whole assistant turn, its text and every tool call it makes. A host raises a clear error for a stage it cannot render. When the person ends the prompt without one of the `choices`, the host answers `reject` if it was offered and `terminate` otherwise, so the step fails closed; `human()` accepts that `terminate` even when it was not offered. It notifies the person and marks the sample as awaiting a person while it waits. `human()` calls this; most protocols never need to.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L242)

``` python
async def ask_human(
    self,
    step: Step,
    escalations: Escalations,
    choices: Sequence[str],
) -> HumanAnswer
```

`step` Step  
The step to decide about.

`escalations` Escalations  
The escalations the asking protocol was handed, from the links before it in a `sequential()`, in order; empty when nothing escalated.

`choices` Sequence\[str\]  
What the person may pick, from `approve`, `reject`, `terminate` and `modify`, in the order to present them.

### HumanAnswer

What a person answered when `Host.ask_human` asked them about a step.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_host.py#L25)

``` python
@dataclass(frozen=True)
class HumanAnswer
```

#### Attributes

`decision` str  
The choice the person picked, one of the `choices` they were offered (`approve`, `reject`, `terminate` or `modify`), or, when the person ended the prompt without one of them, `reject` if it was offered and `terminate` otherwise.

`reason` str \| None  
The reason the surface gives for the answer, if any: the person’s typed reason, or fixed text such as “Human operator rejected the tool call.”.

`modified` ToolCall \| None  
The replacement call, when the person chose `modify` at a tool stage.

## Helpers for rules

Pure functions for rules that check tool calls, and builders of steps for testing them.

### call_text

The text of a tool call: every string in its arguments.

The string argument values, in the order the arguments were given, joined with newlines. Strings inside lists and dicts are included, in order (a dict’s values in insertion order); empty strings and values that are not strings, such as numbers, are left out. This knows nothing about particular tools: for [bash()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#bash) it is the `command`, and for [text_editor()](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#text_editor) it is the `command` (such as `"create"`) as well as the path and the file’s text.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_rules.py#L13)

``` python
def call_text(call: ToolCall) -> str
```

`call` ToolCall  
The tool call.

### tool_matches

Whether the call’s tool name matches any of the patterns.

A pattern is a tool name or a glob (`*`, `?`, `[seq]`) that must match the whole name: `"bash"` matches `bash` and not `bash_session`, while `"bash*"` matches both. Matching is case-sensitive by default and the same on every platform; pass `case_sensitive=False` for tools whose names differ only in case, such as inspect_ai’s `bash` and a bridged agent’s `Bash`. Unlike the `tools` of an [ApprovalPolicy](https://inspect.aisi.org.uk/reference/inspect_ai.approval.html#approvalpolicy) in inspect_ai, a pattern is not extended with a trailing `*` and is not split on commas.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_rules.py#L38)

``` python
def tool_matches(call: ToolCall, *patterns: str, case_sensitive: bool = True) -> bool
```

`call` ToolCall  
The tool call.

`*patterns` str  
Tool names or globs.

`case_sensitive` bool  
Whether case must match.

### find_words

The words that appear in the text as whole words.

A word matches where it is not part of a longer word: the characters on either side of it are not letters, digits or underscores. So `"rm"` matches `rm -rf x` and `/bin/rm x` but not `platform`, and a word that starts or ends with punctuation works too: `"-delete"` matches `find . -delete`. A space in a word matches any run of whitespace, so `"pip install"` also matches `pip` and `install` separated by a tab or several spaces.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_rules.py#L58)

``` python
def find_words(
    text: str, words: Iterable[str], *, case_sensitive: bool = True
) -> list[str]
```

`text` str  
The text to search, such as `call_text(step.call)`.

`words` Iterable\[str\]  
The words to look for.

`case_sensitive` bool  
Whether case must match.

### result_text

The tool call’s result as text, as inspect_ai’s human review shows it.

Content other than text, such as an image, appears as its type in brackets (`[image]`). A call that failed reads `"Error (<type>): <message>"`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_rules.py#L85)

``` python
def result_text(step: AfterToolCall) -> str
```

`step` AfterToolCall  
The step after the tool call.

## Replay

Running a sentinel over steps rebuilt from stored transcripts is done through Inspect Scout: `as_scanner()`, `as_reader()`, `replay_check()`, `replay_check_differences()`, `@step_selector` and `@human_stand_in` are in `inspect_sentinel.scout`; see [Replay](../replay.html.md). A stand-in for the person `human()` asks has this type, and is handed a `ReplayStep`. A sentinel’s configuration and the stages it watches are in [`inspect_sentinel.config`](../reference/config.html.md).

### HumanStandIn

A stand-in for the person `human()` asks, in a replay: handed the step, the escalations and the choices, as `Host.ask_human` is, and answering as the person would. An oracle built from labels, or a model.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_replay.py#L35)

``` python
HumanStandIn: TypeAlias = Callable[
    [ReplayStep, Escalations, Sequence[str]], Awaitable[HumanAnswer]
]
```

## Preprocessing

What a monitor shows a model about a step: a `StepPreprocessor` says how much of its conversations to show and what to leave out, `preprocess_step()` applies it, and `step_as_str()` renders the result.

### StepPreprocessor

What a monitor shows a model about a step.

`preprocess_step()` applies it, in this order: the windows to `input`, `history` and `agent_messages`, then the omissions to every message the step holds, then `transform`, which sees the result. By default `input` is shown whole, while `history` and `agent_messages` are omitted, even when the host could not provide them, so a default prompt is the same live and in replay. A scaffold may keep hidden instructions out of its own list but send them in `input` and `history`. To omit the request lists and show the agent’s own list, use `StepPreprocessor(input_window=None, history_window=None, agent_messages_window=AllMessages())`; this requires the conversation-owning scaffold to be the outermost `@agent` agent function. Other stage content remains: every stage renders the request’s tools and tool choice while `input` is known (a tool stage’s are the producing request’s); `AfterGenerate` renders the reply, including reasoning by default before the scaffold has curated it; both tool stages render the assistant text and call; and `AfterToolCall` renders the result. Strict blindness also needs a stage-specific `transform` or a custom prompt that curates these surfaces; a transform can clear `tools` and `tool_choice`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L93)

``` python
@dataclass(frozen=True, kw_only=True)
class StepPreprocessor(Generic[S])
```

#### Attributes

`input_window` Window \| None  
How much of `input` to show: all of it by default; None for none.

Before a generate, `input` is the request being judged, so a window that leaves none of it (None, `LastTurns(0)`, `LastMessages(0)`, or `NewSinceLastReport` when nothing is new) shows the model only the tools offered when the other message windows also leave none. At a tool stage the call is judged against only the tools of the request that produced it.

`history_window` Window \| None  
How much of `history`, every message the agent has sent its model, to show: none by default.

`agent_messages_window` Window \| None  
How much of `agent_messages`, the agent’s own message list, to show: none by default.

`omit_system` bool  
Leave system messages out.

`omit_reasoning` bool  
Leave the model’s reasoning out.

`omit_tool_usage` bool  
Leave out a provider’s own tool use (such as web search) recorded in a message.

`transform` Callable\[\[S\], Awaitable\[S\]\] \| None  
Any further change, applied last to the thinned step. It returns a step of the same type. The messages it is given are the host’s own objects: copy one before changing it.

### Window

How much of one conversation, `input`, `history` or `agent_messages`, to show.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L89)

``` python
Window: TypeAlias = AllMessages | LastTurns | LastMessages | NewSinceLastReport
```

### AllMessages

Keep all of a conversation.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L44)

``` python
@dataclass(frozen=True)
class AllMessages
```

### LastTurns

Keep the last `n` turns of a conversation: everything from the `n`th assistant message from the end. A conversation with no more than `n` assistant messages is kept whole. `n=0` keeps nothing.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L49)

``` python
@dataclass(frozen=True)
class LastTurns
```

#### Attributes

`n` int  
How many turns to keep.

### LastMessages

Keep the last `n` messages of a conversation.

The cut can fall between a tool call and its result, so the kept messages can start with a result whose call is not shown, and the monitor model sees output without the command that produced it. Prefer `LastTurns`, which keeps each call with its result, unless the count of messages matters more than that.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L61)

``` python
@dataclass(frozen=True)
class LastMessages
```

#### Attributes

`n` int  
How many messages to keep.

### NewSinceLastReport

Keep the messages after the last one this monitor reported on, so an incremental monitor does not re-read its conversations.

An incremental prompt chooses `input_window` as well: `StepPreprocessor` shows all of `input` by default, so `StepPreprocessor(history_window=NewSinceLastReport(context))` still sends the monitor model the whole request at every step. To judge what was already sent, at `AfterGenerate` or a tool stage, where `history` holds the request, window `history` and omit or bound `input`: `StepPreprocessor(input_window=None, history_window=NewSinceLastReport(context))`. At `BeforeGenerate`, `history` does not yet hold the request about to go out, so a monitor that judges that request keeps `input`: `StepPreprocessor(input_window=NewSinceLastReport(context))` shows only what is new in it. `history` only grows and is resumed where it was; over `input` or `agent_messages`, which can be rewritten, each step rereads the whole list to find what is new, though it shows only that.

The last message shown is recorded by id and position in the monitor’s store (by `context.path`), separately for each conversation, for each list it windows, and for each function of the monitor. It is recorded only when the function returns or decides final and the runner accepts what it returned, so a function that raises or is cancelled is shown those messages again at the next step. `history`, which only grows, is resumed where it was without reading what was shown before; `input` and `agent_messages` can be rewritten, as by a compaction, so for them the whole list is read to find the message shown last, a cost linear in the list at each step. An edited version of a message, which `history` holds again under the same id, is new. A conversation whose recorded message is gone, as after a compaction, is kept whole. Preprocess with it from the monitor or protocol function the runner is running; elsewhere preprocessing raises `RuntimeError`.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L76)

``` python
@dataclass(frozen=True)
class NewSinceLastReport
```

#### Attributes

`context` Context  
The monitor’s context, whose store holds what was shown.

### preprocess_step

`step`, thinned as `preprocessor` says (`StepPreprocessor()` when None).

Each window cuts its list as it is. A list with no window becomes `[]`, even one the host could not provide; a windowed list the host could not provide stays None. The omissions apply to `input`, `history`, `agent_messages`, a tool call’s result and a generate’s reply.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L125)

``` python
async def preprocess_step(
    step: S, preprocessor: StepPreprocessor[S] | None = None
) -> S
```

`step` S  
The step a monitor was given.

`preprocessor` StepPreprocessor\[S\] \| None  
What to show; `StepPreprocessor()` when None.

### step_as_str

Render `step` for a monitor’s prompt, numbering its messages `[M1]`, `[M2]` and so on, and return the text with a function that turns those cites in a model’s answer into references.

Everything is in one block, marked `[BEGIN AGENT STEP - ...]` and `[END AGENT STEP - ...]` and named for the stage: `TOOL CALL REQUEST` before a tool call, `TOOL CALL RESULT` after one, `MODEL REQUEST` before a generate, and `MODEL REPLY` after one. Inside it are blocks between `[BEGIN ...]` and `[END ...]` lines, each left out when it is empty. At every stage: `HISTORY` (`step.history`, every message the agent has sent its model), `AGENT MESSAGES` (`step.agent_messages`, the outermost running agent function’s own message list, which its scaffold may curate), then the model request, as `INPUT`, `TOOLS` (the tools offered, as JSON) and a one-line `[TOOL CHOICE] "auto"`; at a tool stage, the request that produced the call. Then after a generate, `REPLY`; at a tool stage, `CALL` (the assistant’s text and the call) and, after the call, `RESULT`. A list the host could not provide (None) is named in its block (`preprocess_step()` leaves a list with no window empty, so only a windowed one is named); at a tool stage whose input is not known, the tools and tool choice are left out. With a `nonce`, every `[BEGIN ...]` and `[END ...]` line, and the `[TOOL CHOICE]` line, carries it in parentheses, as `[BEGIN INPUT (3f9a1c7e)]`, and so does every message header. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/b5469a943eda0545f6f17f2f19de8ec96c5922dd/src/inspect_sentinel/_preprocess.py#L180)

``` python
async def step_as_str(
    step: Step, *, nonce: str | None = None
) -> tuple[str, ExtractReferences]
```

`step` Step  
The step to render, usually from `preprocess_step()`.

`nonce` str \| None  
A code the agent cannot guess, such as `secrets.token_hex(4)`, for every block marker and message header; none by default. Scout needs at least 8 characters.
