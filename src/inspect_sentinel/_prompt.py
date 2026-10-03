from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    Content,
)
from inspect_ai.tool import ToolCall

from ._step import AfterToolCall, BeforeToolCall


def messages_as_str(
    messages: Sequence[ChatMessage],
    *,
    exclude_system: bool = True,
    exclude_reasoning: bool = False,
    exclude_tool_usage: bool = False,
) -> str:
    """Render messages as text for a monitor prompt.

    Messages are rendered oldest first, so successive steps of a conversation share a prefix and a model provider can cache it. Each message is rendered by `message_as_str()`, and messages are separated by a blank line. The format matches Inspect Scout's `messages_as_str()`, so monitors and scanners read alike.

    Args:
        messages: The messages to render, typically `step.input` or a view of it such as `last_turns()`.
        exclude_system: Leave out system messages.
        exclude_reasoning: Leave out reasoning content.
        exclude_tool_usage: Leave out tool calls, tool results and server-side tool use.

    Returns:
        The rendered messages; an empty string when there are none to render.
    """
    return "\n".join(
        message_as_str(
            message,
            exclude_reasoning=exclude_reasoning,
            exclude_tool_usage=exclude_tool_usage,
        )
        for message in messages
        if not (exclude_system and message.role == "system")
        and not (exclude_tool_usage and isinstance(message, ChatMessageTool))
    )


def message_as_str(
    message: ChatMessage,
    *,
    exclude_reasoning: bool = False,
    exclude_tool_usage: bool = False,
) -> str:
    """Render one message as text for a monitor prompt.

    The message is headed by its role (`USER:`, `ASSISTANT:`, `TOOL:`, `SYSTEM:`). An assistant message lists its tool calls after its text as `Tool Call: ` followed by `call_as_str()`; a tool message ends with the error when the call failed. Reasoning is wrapped in `<thinking>` tags, and images and other media are shown as placeholders such as `<image />`.

    Args:
        message: The message to render, e.g. `step.result`.
        exclude_reasoning: Leave out reasoning content.
        exclude_tool_usage: Leave out an assistant message's tool calls and server-side tool use. A tool message is rendered regardless; `messages_as_str()` is where tool messages are left out.

    Returns:
        The rendered message, ending in a newline.
    """
    parts = [
        text
        for content in (
            [message.content] if isinstance(message.content, str) else message.content
        )
        if (text := _content_as_str(content, exclude_reasoning, exclude_tool_usage))
    ]
    if (
        isinstance(message, ChatMessageAssistant)
        and message.tool_calls
        and not exclude_tool_usage
    ):
        parts.extend(f"Tool Call: {call_as_str(call)}" for call in message.tool_calls)
    if isinstance(message, ChatMessageTool) and message.error:
        parts.append(
            f"Error in tool call '{message.function or 'unknown'}':\n{message.error.message}"
        )
    body = "\n\n".join(parts)
    return f"{message.role.upper()}:\n{body}\n"


def call_as_str(call: ToolCall) -> str:
    """Render a tool call as text, in the form of a Python call.

    For example, `bash(command="ls -la /etc", timeout=30)`. String arguments are double-quoted, with double quotes and backslashes escaped and newlines kept, so a multi-line script stays readable; other arguments are rendered as JSON.

    Args:
        call: The call to render, e.g. `step.call`.

    Returns:
        The call as one string.
    """
    arguments = ", ".join(
        f"{name}={_argument_as_str(value)}" for name, value in call.arguments.items()
    )
    return f"{call.function}({arguments})"


def last_turns(messages: Sequence[ChatMessage], n: int) -> list[ChatMessage]:
    """The last `n` turns of a conversation.

    A turn starts at an assistant message and runs to the next one, so it carries the assistant's tool calls together with their results and any user message that followed; a call is never separated from its result. Earlier messages, including the system message and the task, are left out unless the conversation has no more than `n` turns, in which case every message is returned.

    Args:
        messages: The conversation, typically `step.input`.
        n: How many turns to keep.

    Returns:
        The messages of the last `n` turns, oldest first.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError(f"last_turns() needs n >= 0, got {n}")
    if n == 0:
        return []
    starts = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ChatMessageAssistant)
    ]
    if len(starts) <= n:
        return list(messages)
    return list(messages[starts[-n] :])


def step_as_str(
    step: BeforeToolCall | AfterToolCall, *, turns: int | None = None
) -> str:
    """Render a tool-call step as the body of a monitor prompt.

    The conversation the model was sent (`step.input`, rendered by `messages_as_str()` without system messages), then the assistant's message with the call being checked, then, after the call, its result. The call being checked is always the last `Tool Call:` in the text. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it.

    Generate stages are not supported yet.

    Args:
        step: The step to render.
        turns: Render only the last `turns` turns of `step.input` (see `last_turns()`); all of it when None.

    Returns:
        The rendered step.
    """
    input = step.input if turns is None else last_turns(step.input, turns)
    call = ChatMessageAssistant(content=step.message, tool_calls=[step.call])
    rendered = [messages_as_str(input), message_as_str(call)]
    if isinstance(step, AfterToolCall):
        rendered.append(message_as_str(step.result))
    return "\n".join(part for part in rendered if part)


def _content_as_str(
    content: str | Content, exclude_reasoning: bool, exclude_tool_usage: bool
) -> str | None:
    if isinstance(content, str):
        return content
    match content.type:
        case "text":
            return content.text
        case "reasoning":
            if exclude_reasoning:
                return None
            if content.redacted:
                return (
                    f"<thinking_summary>{content.summary}</thinking_summary>"
                    if content.summary
                    else "<thinking_redacted/>"
                )
            return (
                f"<thinking>{content.reasoning}</thinking>"
                if content.reasoning
                else None
            )
        case "tool_use":
            if exclude_tool_usage:
                return None
            error = f" {content.error}" if content.error else ""
            return f"Tool Use: {content.name}({content.arguments}) -> {content.result}{error}"
        case _:
            return f"<{content.type} />"


def _argument_as_str(value: Any) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return json.dumps(value, ensure_ascii=False)
