from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from inspect_ai.model import ChatMessage, ChatMessageTool, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolCallView, ToolChoice, ToolInfo, ToolResult

from ._report import Decision, Reported


@dataclass(frozen=True)
class HumanDecision:
    """A decision a person made through `human()` at an earlier step of the agent's conversation. `human_decisions` on a step holds them, and `human_decision_for()` finds the one covering a call.

    Use it to annotate the past call it covers when rendering the conversation into a prompt, such as "reviewed by a person: approved", so a monitor does not flag already-reviewed history again. It is not a standing permission: never lower suspicion of a new call because it resembles one a person approved. The same command can mean something else once the state has changed (an `rm -rf build/` approved in one working directory deletes something else after a `cd`), and an agent can get a benign call approved and then issue a similar harmful one.
    """

    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]
    """The stage of the step it was made at, as `SentinelEvent.stage` records it: `tool_call`, `tool_result` or `model_output`, the stages `human()` asks at."""

    step_id: str
    """The id of the step it was made at, as `SentinelEvent.step_id` records it: the tool call id at a tool stage, the assistant message id at `model_output`."""

    call_ids: tuple[str, ...]
    """The tool calls the decision covered: the step's call at a tool stage, and every call the turn makes at `model_output`. How a prompt says which calls a person approved. Empty for a review of a turn with no tool calls, which has no stable key to annotate by: `step_id` is then the turn's message id, which is not stable across scaffolds that copy or rebuild their messages."""

    decided: Reported[Decision]
    """The `human()` instance that asked (`name`, `path`, `function`) and a detached copy of its decision, which carries nothing of the step it was made at beyond its own fields: `approve` is a `proceed`, and the person's reason is the `explanation` (and, for a `reject`, the `message`). The reason is untrusted text, which can quote the agent, so quote it when rendering it into a prompt.

    It is what the person decided, not what happened to the step: beside other protocols in a `concurrent()`, a peer's `reject` outranks the person's `approve`, so a call a person approved may not have run. Mark only calls that ran, which have a result without an error in the history (a call rejected before it ran also gets a tool message, with an `approval` error, so a tool message alone does not prove the call ran).

    A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, a `reject` or `terminate`, and its `explanation` says so (for example "ACP client cancelled the permission request."). Render the explanation, and do not present such a decision as the person's judgment.
    """


@dataclass(frozen=True)
class BeforeGenerate:
    """A generate by the agent's model, before the request is sent and before the cache lookup.

    `input`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. They, `tools` and `config` are the host's own objects, borrowed for the step: monitors and protocols must not mutate them or the messages in them (copy first to edit), since a change would alter the real request.
    """

    model: str
    """The model the request is for, as `provider/name`."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: Sequence[ChatMessage]
    """Exactly what the model is sent: the caller's messages after the model's own preparation (the configured system message, reasoning history, merged consecutive messages). In-process these are the host's own messages; monitors must not mutate them."""

    agent_messages: Sequence[ChatMessage] | None
    """The agent loop's own message list: the conversation as the caller passed it to the model (`react()`'s `AgentState.messages`, a sub-agent's own list, or the list a solver passed to `generate`). For a scaffold that compacts its conversation this is the compacted conversation. `None` when the host cannot know the caller's list: under a proxy, which sees only the request; in a replay from a log that cannot establish the list at this step, since a log records the sample's final messages and what the model was sent, not the caller's list at each step; or at a step with no caller conversation behind it. It is never approximated from `input` or `history`; for what the model saw, use `input` and `history`. In-process Inspect always provides it, as the caller's own list; monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent: the inputs of all its model calls taken together, each message once, where it was first sent. The opening prompts as the model saw them come first, a message the scaffold rewrote before sending shows as it was sent, and turns a compaction later dropped are still here. A message sent again changed (the same id with different content) appears again as its new version, where that version was sent; a message the host rebuilds with a fresh id for every call, such as a configured system message, is not repeated while it is unchanged and at the same position as in the call before, nor when its latest rebuilt copy is sent again later; a scaffold that alternates between two rebuilt copies of one message can see it repeated. inspect_ai sends an earlier assistant turn without its reasoning under `reasoning_history="last"`, so such a turn appears once with its reasoning and once without. A message the scaffold changes in place after sending it shows its current content. The model's outputs appear once a later call sends them back. It does not yet include `input`, which is about to be sent. Only this agent instance's model calls count: a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer or a monitor is not an agent turn. A model call the scaffold makes itself inside its agent loop does count; a scaffold keeps one out by running it as an agent of its own, through `inspect_ai.agent.run()`. Sub-agents inside one bridged CLI share its conversation. `None` when the host cannot produce it: in a replay from a log, which does not yet record which model calls were agent turns. In-process Inspect always provides it. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, a conversation is told apart by how it opens, so agents whose conversations open alike share one history, every model call of the conversation counts as an agent turn, and a sidecar that joins a conversation midway starts the history there. The messages are the host's own, mostly shared with `agent_messages`; monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: list[ToolInfo]
    """The tools the model is offered, after `tool_choice` filtering. In-process these are the host's own objects; monitors must not mutate them."""

    tool_choice: ToolChoice
    """The tool choice the model is sent."""

    config: GenerateConfig
    """The generation configuration the model is sent, with the task's and the model's settings merged in. In-process this is the host's own object; monitors must not mutate it."""


@dataclass(frozen=True)
class AfterGenerate:
    """A generate by the agent's model that has completed, before its output reaches the agent.

    `input`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. They, `tools`, `config` and `output` are the host's own objects, borrowed for the step: monitors and protocols must not mutate them or the messages in them (copy first to edit), since a change would alter the real request or the response the agent receives.
    """

    model: str
    """The model the request was for, as `provider/name`; `output.model` names the model that answered."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: Sequence[ChatMessage]
    """Exactly what the model was sent, as for `BeforeGenerate`; monitors must not mutate it."""

    agent_messages: Sequence[ChatMessage] | None
    """The agent loop's own message list, or `None`, as for `BeforeGenerate.agent_messages`; monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent, or `None` when the host cannot produce it, as for `BeforeGenerate.history`. It includes `input`; `output` is not in it yet. Monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: list[ToolInfo]
    """The tools the model was offered; monitors must not mutate them."""

    tool_choice: ToolChoice
    """The tool choice the model was sent."""

    config: GenerateConfig
    """The generation configuration the model was sent; monitors must not mutate it."""

    output: ModelOutput
    """What the model returned; the first choice's message is what the agent receives. Some providers return an output with no choices, so check `output.choices` before reading `output.message`. In-process this is the host's own output, which the agent receives; monitors must not mutate it."""


@dataclass(frozen=True)
class Truncation:
    """A cut the host made to a tool's output before the model saw it: `AfterToolCall.truncated` when there was one."""

    original_bytes: int
    """The size of the tool's output, in UTF-8 bytes of its text."""

    limit_bytes: int
    """The limit the output was cut to, in bytes."""


@dataclass(frozen=True)
class BeforeToolCall:
    """A tool call about to execute.

    `input`, `tools`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps for the conversation rather than copy them, and their messages and tool definitions are the host's own: monitors and protocols must not mutate them. A monitor that wants a list calls `list()` on one, at its own cost.
    """

    conversation: str
    """Id linking this agent's steps across compactions."""

    message: str
    """Assistant text alongside the call."""

    message_id: str
    """Id of the assistant message that made the call, so a monitor model can cite it."""

    call: ToolCall
    """The call as it will execute."""

    view: ToolCallView
    """Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise."""

    input: Sequence[ChatMessage] | None
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when no model call produced the assistant message (a pre-filled message, a human agent, or one the scaffold built itself) or when the transcript event that recorded the model call is no longer available; it is never approximated from `agent_messages`. `context.eval.sample_input` is the sample's input, the assignment."""

    agent_messages: Sequence[ChatMessage] | None
    """The agent loop's own message list (`react()`'s `AgentState.messages`, a sub-agent's own list): its full conversation, pre-compaction turns included. `None` when the host cannot know the scaffold's list: under a proxy, which sees only the request; in a replay from a log that cannot establish the list at this step, since a log records the sample's final messages and what the model was sent, not the scaffold's list at each step; or at a step with no scaffold conversation behind it. It is never approximated from `input` or `history`; for what the model saw, use `input` and `history`. In-process Inspect always provides it, as the scaffold's own list; monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent: the inputs of all its model calls taken together, each message once, where it was first sent. The opening prompts as the model saw them come first, a message the scaffold rewrote before sending shows as it was sent, and turns a compaction later dropped are still here. A message sent again changed (the same id with different content) appears again as its new version, where that version was sent; a message the host rebuilds with a fresh id for every call, such as a configured system message, is not repeated while it is unchanged and at the same position as in the call before, nor when its latest rebuilt copy is sent again later; a scaffold that alternates between two rebuilt copies of one message can see it repeated. inspect_ai sends an earlier assistant turn without its reasoning under `reasoning_history="last"`, so such a turn appears once with its reasoning and once without. A message the scaffold changes in place after sending it shows its current content. The model's outputs appear once a later call sends them back. It includes the input of the call that produced this tool call. Only this agent instance's model calls count: a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer or a monitor is not an agent turn. A model call the scaffold makes itself inside its agent loop does count; a scaffold keeps one out by running it as an agent of its own, through `inspect_ai.agent.run()`. Sub-agents inside one bridged CLI share its conversation. `None` when the host cannot produce it: in a replay from a log, which does not yet record which model calls were agent turns. In-process Inspect always provides it. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, a conversation is told apart by how it opens, so agents whose conversations open alike share one history, every model call of the conversation counts as an agent turn, and a sidecar that joins a conversation midway starts the history there. The messages are the host's own, mostly shared with `agent_messages`; monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: Sequence[ToolInfo] | None = None
    """The tool definitions offered to the model call that produced the call, as the transcript's `ModelEvent` recorded them, after any model hooks (so they can differ from that call's `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` exactly when `input` is `None`; it is never approximated from the tools available now. Monitors must not mutate them."""

    tool_choice: ToolChoice | None = None
    """The tool choice of the model call that produced the call, as the transcript's `ModelEvent` recorded it. `None` exactly when `input` is `None`."""


@dataclass(frozen=True)
class AfterToolCall:
    """A tool call that has executed, before the model sees its result.

    `input`, `tools`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps for the conversation rather than copy them, and their messages and tool definitions are the host's own: monitors and protocols must not mutate them. A monitor that wants a list calls `list()` on one, at its own cost.
    """

    conversation: str
    """Id linking this agent's steps across compactions."""

    message: str
    """Assistant text alongside the call."""

    message_id: str
    """Id of the assistant message that made the call, so a monitor model can cite it."""

    call: ToolCall
    """The call as executed, after any `modify`."""

    result: ChatMessageTool
    """What the model will see, after truncation; `truncated` says whether it was cut."""

    output: ToolResult | None
    """The tool's untruncated return value, before any cut `truncated` reports. `None` when the host cannot see it: beside a proxy, which sees only the `result` the model is sent, and in a replay from a log, which records only that `result`. In-process Inspect always provides it. Judge what the model saw by `result`, which every host provides."""

    view: ToolCallView
    """Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise."""

    input: Sequence[ChatMessage] | None
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when no model call produced the assistant message (a pre-filled message, a human agent, or one the scaffold built itself) or when the transcript event that recorded the model call is no longer available; it is never approximated from `agent_messages`. `context.eval.sample_input` is the sample's input, the assignment."""

    agent_messages: Sequence[ChatMessage] | None
    """The agent loop's own message list (`react()`'s `AgentState.messages`, a sub-agent's own list): its full conversation, pre-compaction turns included. `None` when the host cannot know the scaffold's list: under a proxy, which sees only the request; in a replay from a log that cannot establish the list at this step, since a log records the sample's final messages and what the model was sent, not the scaffold's list at each step; or at a step with no scaffold conversation behind it. It is never approximated from `input` or `history`; for what the model saw, use `input` and `history`. In-process Inspect always provides it, as the scaffold's own list; monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent, or `None` when the host cannot produce it, as for `BeforeToolCall.history`. It includes the input of the call that produced this tool call; `result` is not in it yet. Monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: Sequence[ToolInfo] | None = None
    """The tool definitions offered to the model call that produced the call, as the transcript's `ModelEvent` recorded them, after any model hooks (so they can differ from that call's `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` exactly when `input` is `None`; it is never approximated from the tools available now. Monitors must not mutate them."""

    tool_choice: ToolChoice | None = None
    """The tool choice of the model call that produced the call, as the transcript's `ModelEvent` recorded it. `None` exactly when `input` is `None`."""

    truncated: Truncation | Literal[False] | None = None
    """Whether the host cut the tool's output before the model saw it in `result`. A `Truncation` when it did; `False` when it did not, which is always so for structured content; `None` when this host cannot tell (a proxy, a replay from messages alone, or a log older than the record of cuts). `if step.truncated:` is true only for a cut. It is built from the value the transcript's `ToolEvent.truncated` records (`Truncation(*event.truncated)` when that is set), so a monitor reads the same answer live and from a log. A cut made before the host's limit applies, by a sandbox output limit or a timeout, is not reported here: `result.error` reports it, with type `"limit"` or `"timeout"`. Use this field, not the "output was too long" notice in `result`, to tell whether the host cut the result: that notice is text a tool's output can contain, so it can be forged."""


Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a host."""


def human_decision_for(step: Step, call_id: str) -> HumanDecision | None:
    """The latest decision a person made through `human()` that covers the tool call `call_id`, from `step.human_decisions`.

    A call can be decided about more than once: its turn at `model_output` covers each of its calls, then the call itself before it runs and its result after, so the latest is the most specific. Look decisions up by call id, which the model's provider assigns and which stays the same however a scaffold copies or rebuilds its messages; message ids do not, so there is no lookup by message, and a review of a turn with no tool calls, which stays in `step.human_decisions`, is found by none. None when no decision covers the call, and also when `step.human_decisions` is None because the host cannot tell; check that field to tell the two apart. Use the result to annotate that exact call, not to judge a new call that resembles it (see `HumanDecision`).

    Args:
        step: The step being examined.
        call_id: The id of a tool call earlier in the conversation, as `ToolCall.id`.
    """
    for decision in reversed(step.human_decisions or ()):
        if call_id in decision.call_ids:
            return decision
    return None
