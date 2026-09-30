from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from inspect_ai.model import ChatMessage, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallView, ToolResult

from ._report import Decision, Reported


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
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `context.input` is the sample's input, the assignment."""

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

    input: list[ChatMessage]
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `context.input` is the sample's input, the assignment."""

    history: list[ChatMessage]
    """The scaffold's full conversation, pre-compaction turns included. In-process this is the scaffold's own list; monitors must not mutate it."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside a `sequential` composition."""


Step: TypeAlias = BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a dispatcher."""
