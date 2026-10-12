from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from inspect_ai.model import ChatMessage, ChatMessageTool, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolCallView, ToolChoice, ToolInfo, ToolResult

from ._report import Decision, Reported


@dataclass(frozen=True)
class BeforeGenerate:
    """A generate by the agent's model, before the request is sent and before the cache lookup."""

    model: str
    """The model the request is for, as `provider/name`."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage]
    """Exactly what the model is sent: the caller's messages after the model's own preparation (the configured system message, reasoning history, merged consecutive messages)."""

    history: list[ChatMessage]
    """The model's view of this agent instance's run, ending with this request: every message the instance has sent to its model or received from it, in order, deduplicated by id, an edited message kept in each version. Where the instance made no model call, the scaffold's conversation."""

    tools: list[ToolInfo]
    """The tools the model is offered, after `tool_choice` filtering."""

    tool_choice: ToolChoice
    """The tool choice the model is sent."""

    config: GenerateConfig
    """The generation configuration the model is sent, with the task's and the model's settings merged in."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


@dataclass(frozen=True)
class AfterGenerate:
    """A generate by the agent's model that has completed, before its output reaches the agent."""

    model: str
    """The model the request was for, as `provider/name`; `output.model` names the model that answered."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage]
    """Exactly what the model was sent, as for `BeforeGenerate`."""

    history: list[ChatMessage]
    """The model's view of this agent instance's run, as for `BeforeGenerate`, ending with `output.message`."""

    tools: list[ToolInfo]
    """The tools the model was offered."""

    tool_choice: ToolChoice
    """The tool choice the model was sent."""

    config: GenerateConfig
    """The generation configuration the model was sent."""

    output: ModelOutput
    """What the model returned; the first choice's message is what the agent receives. Some providers return an output with no choices, so check `output.choices` before reading `output.message`."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


@dataclass(frozen=True)
class BeforeToolCall:
    """A tool call about to execute."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    message: str
    """Assistant text alongside the call."""

    call: ToolCall
    """The call as it will execute."""

    view: ToolCallView
    """Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise."""

    input: list[ChatMessage]
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `context.eval.sample_input` is the sample's input, the assignment."""

    history: list[ChatMessage]
    """The model's view of this agent instance's run, as for `BeforeGenerate`, ending with the assistant message that made the call. Where the instance made no model call, the scaffold's conversation."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


@dataclass(frozen=True)
class AfterToolCall:
    """A tool call that has executed, before the model sees its result."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    message: str
    """Assistant text alongside the call."""

    call: ToolCall
    """The call as executed, after any `modify`."""

    result: ChatMessageTool
    """What the model will see, after truncation."""

    output: ToolResult
    """The tool's untruncated return value."""

    view: ToolCallView
    """Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise."""

    input: list[ChatMessage]
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `context.eval.sample_input` is the sample's input, the assignment."""

    history: list[ChatMessage]
    """The model's view of this agent instance's run, as for `BeforeGenerate`, ending with the assistant message that made the call. Where the instance made no model call, the scaffold's conversation."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a host."""
