"""The step payloads: what exists at each point in the agent loop a sentinel can watch.

Tool stages only for now; the generate payloads arrive with the generate-side dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from inspect_ai.model import ChatMessage, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallView, ToolResult

from ._report import Decision, Reported

Stage: TypeAlias = Literal["tool_call", "tool_result"]
"""Which point in the loop a step belongs to: `tool_call` for `BeforeToolCall`, `tool_result` for `AfterToolCall`. Grows with the generate stages."""


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
    """Exactly what the model was sent for the generate that produced the call."""

    history: list[ChatMessage]
    """The scaffold's full conversation, pre-compaction turns included. In-process this is the scaffold's own list; monitors must not mutate it."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside an ordered chain."""


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
    """Exactly what the model was sent for the generate that produced the call."""

    history: list[ChatMessage]
    """The scaffold's full conversation, pre-compaction turns included. In-process this is the scaffold's own list; monitors must not mutate it."""

    escalations: tuple[Reported[Decision], ...] = ()
    """Earlier links' escalations, when this step is inside an ordered chain."""


Step: TypeAlias = BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a dispatcher."""


STAGE_OF_TYPE: dict[type[Any], Stage] = {
    BeforeToolCall: "tool_call",
    AfterToolCall: "tool_result",
}
"""Payload type to stage. Add here when a stage is added; `stage_of` and the decorators both read it."""


def stage_of(step: Step) -> Stage:
    """The stage a step payload belongs to.

    Args:
        step: The payload.
    """
    return STAGE_OF_TYPE[type(step)]
