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

    tools: list[ToolInfo] | None = None
    """The tool definitions offered to the model call that produced the call, as the transcript's `ModelEvent` recorded them, after any model hooks (so they can differ from that call's `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` exactly when `input` is `None`; it is never approximated from the tools available now. Monitors must not mutate them."""


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
    """What the model will see, after truncation; `truncated` says whether it was cut."""

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

    tools: list[ToolInfo] | None = None
    """The tool definitions offered to the model call that produced the call, as the transcript's `ModelEvent` recorded them, after any model hooks (so they can differ from that call's `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` exactly when `input` is `None`; it is never approximated from the tools available now. Monitors must not mutate them."""

    truncated: tuple[int, int] | None = None
    """When `result` is a cut-down form of `output`, the cut as `(from, to)` in bytes: the size of the tool's output and the limit it was cut to. `None` when the host did not cut it, which is always so for structured content. `None` does not prove the model saw everything the tool produced: a tool's output can be cut before the host's limit applies (a sandbox output limit or a timeout, which `result.error` reports with type `"limit"` or `"timeout"`), and a host that cannot observe the cut (a proxy, or replay from messages alone) leaves it `None`. It is the value the transcript's `ToolEvent.truncated` records, so a monitor reads one field the same way live and from a log. Use it, not the "output was too long" notice in `result`, to tell whether the host cut the result: that notice is text a tool's output can contain, so it can be forged."""


Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a host."""
