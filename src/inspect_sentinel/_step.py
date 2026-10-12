from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from inspect_ai.model import ChatMessage, ChatMessageTool, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolCallView, ToolChoice, ToolInfo, ToolResult

from ._report import Decision, Reported


@dataclass(frozen=True)
class BeforeGenerate:
    """A generate by the agent's model, before the request is sent and before the cache lookup.

    `input`, `history`, `tools` and `config` are the host's own objects, borrowed for the step, not copies: monitors and protocols must not mutate them (copy first to edit), since a change would alter the real request.
    """

    model: str
    """The model the request is for, as `provider/name`."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage]
    """Exactly what the model is sent: the caller's messages after the model's own preparation (the configured system message, reasoning history, merged consecutive messages). In-process these are the host's own messages; monitors must not mutate them."""

    history: list[ChatMessage]
    """The conversation as the caller passed it to the model. For a scaffold that compacts its conversation this is the compacted conversation. In-process this is the caller's own list; monitors must not mutate it."""

    tools: list[ToolInfo]
    """The tools the model is offered, after `tool_choice` filtering. In-process these are the host's own objects; monitors must not mutate them."""

    tool_choice: ToolChoice
    """The tool choice the model is sent."""

    config: GenerateConfig
    """The generation configuration the model is sent, with the task's and the model's settings merged in. In-process this is the host's own object; monitors must not mutate it."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


@dataclass(frozen=True)
class AfterGenerate:
    """A generate by the agent's model that has completed, before its output reaches the agent.

    `input`, `history`, `tools`, `config` and `output` are the host's own objects, borrowed for the step, not copies: monitors and protocols must not mutate them (copy first to edit), since a change would alter the real request or the response the agent receives.
    """

    model: str
    """The model the request was for, as `provider/name`; `output.model` names the model that answered."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage]
    """Exactly what the model was sent, as for `BeforeGenerate`; monitors must not mutate it."""

    history: list[ChatMessage]
    """The conversation as the caller passed it to the model, as for `BeforeGenerate`; monitors must not mutate it."""

    tools: list[ToolInfo]
    """The tools the model was offered; monitors must not mutate them."""

    tool_choice: ToolChoice
    """The tool choice the model was sent."""

    config: GenerateConfig
    """The generation configuration the model was sent; monitors must not mutate it."""

    output: ModelOutput
    """What the model returned; the first choice's message is what the agent receives. Some providers return an output with no choices, so check `output.choices` before reading `output.message`. In-process this is the host's own output, which the agent receives; monitors must not mutate it."""

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

    input: list[ChatMessage] | None
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when no model call produced the assistant message (a pre-filled message, a human agent, or one the scaffold built itself) or when the transcript event that recorded the model call is no longer available; it is never approximated from `history`. `context.eval.sample_input` is the sample's input, the assignment."""

    history: list[ChatMessage]
    """The scaffold's full conversation, pre-compaction turns included. In-process this is the scaffold's own list; monitors must not mutate it."""

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

    input: list[ChatMessage] | None
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when no model call produced the assistant message (a pre-filled message, a human agent, or one the scaffold built itself) or when the transcript event that recorded the model call is no longer available; it is never approximated from `history`. `context.eval.sample_input` is the sample's input, the assignment."""

    history: list[ChatMessage]
    """The scaffold's full conversation, pre-compaction turns included. In-process this is the scaffold's own list; monitors must not mutate it."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a host."""
