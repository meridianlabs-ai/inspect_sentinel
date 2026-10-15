from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    messages_from_anthropic,
    model_output_from_anthropic,
)

from ._inspect_ai import anthropic_message
from ._wire import UnreadableError, json_object, stream_events

# The content kinds that inspect_ai's Anthropic converter reads, in a reply or
# in an assistant turn of a request. It drops any other kind without an error,
# so a body holding one is unreadable here, rather than judged with part of it
# unseen. Kept in step with content_and_tool_calls_from_assistant_content_blocks()
# in inspect_ai's src/inspect_ai/model/_providers/anthropic.py.
_BLOCKS = frozenset(
    {
        "text",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "compaction",
        "fallback",
    }
)

# The kinds of content delta that the SDK's stream accumulator applies. It
# drops any other kind without an error, such as the delta that carries a
# compaction block's summary, so a stream holding one is unreadable here.
# Kept in step with accumulate_event() in the SDK's
# anthropic/lib/streaming/_messages.py.
_DELTAS = frozenset(
    {
        "text_delta",
        "input_json_delta",
        "citations_delta",
        "thinking_delta",
        "signature_delta",
    }
)


async def read_reply(content_type: str, body: bytes) -> ModelOutput:
    if content_type.startswith("text/event-stream"):
        message = _stream(body)
    else:
        message = _message(json_object(body, "reply"))
    _check_blocks(getattr(message, "content", None), "reply")
    try:
        # an object, not a dict, so that the converter doesn't validate it again
        return await model_output_from_anthropic(cast(Any, message))
    except Exception as ex:
        # the converter raises errors of many kinds on content it can't read
        raise UnreadableError("The reply holds content that can't be read.") from ex


async def read_request(request: dict[str, Any]) -> list[ChatMessage]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise UnreadableError("The request has no list of messages.")
    for entry in cast("list[object]", messages):
        if not isinstance(entry, dict):
            continue
        entry = cast("dict[str, Any]", entry)
        if entry.get("role") == "assistant" and isinstance(entry.get("content"), list):
            _check_blocks(entry["content"], "request")
        _check_documents(entry.get("content"))
    try:
        return await messages_from_anthropic(
            cast(Any, messages), _system(request.get("system"))
        )
    except Exception as ex:
        # the converter raises errors of many kinds on content it can't read
        raise UnreadableError("The request holds a message that can't be read.") from ex


async def rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    # written by inspect_ai, so a thinking block keeps its signature and a call
    # its id
    blocks: list[Any] = []
    for result in results:
        blocks.extend(cast("list[Any]", (await anthropic_message(result))["content"]))
    return [await anthropic_message(message), {"role": "user", "content": blocks}]


def settled(entry: object) -> object:
    # inspect_ai's Anthropic provider moves its cache marks to the latest
    # messages on every request, which writes a text it marks as a block
    if not isinstance(entry, dict):
        return entry
    message = cast("dict[str, Any]", entry)
    content: object = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = [
            {
                key: value
                for key, value in cast("dict[str, Any]", block).items()
                if key != "cache_control"
            }
            if isinstance(block, dict)
            else block
            for block in cast("list[object]", content)
        ]
    return {**message, "content": content}


def _kind(block: object) -> object:
    if isinstance(block, dict):
        return cast("dict[str, Any]", block).get("type")
    return getattr(block, "type", None)


def _check_blocks(content: object, what: str) -> None:
    for block in cast("list[object]", content or []):
        kind = _kind(block)
        if kind not in _BLOCKS:
            raise UnreadableError(
                f"The {what} holds content of a kind that can't be read: {kind!r}."
            )


def _check_documents(content: object) -> None:
    # inspect_ai's Anthropic converter reads only the first block of a document
    # whose source is a list of blocks, and drops the rest, and drops a
    # document's title and context, which the model is sent too, all without
    # an error. See content_block_to_content() in inspect_ai's
    # src/inspect_ai/agent/_bridge/anthropic_api_impl.py.
    if not isinstance(content, list):
        return
    for block in cast("list[object]", content):
        if not isinstance(block, dict):
            continue
        block = cast("dict[str, Any]", block)
        if block.get("type") == "tool_result":
            _check_documents(block.get("content"))
        if block.get("type") != "document":
            continue
        if block.get("title") or block.get("context"):
            raise UnreadableError(
                "The request holds a document with a title or context, which would not be read."
            )
        source = block.get("source")
        if not isinstance(source, dict):
            continue
        parts = cast("dict[str, Any]", source).get("content")
        if isinstance(parts, list) and len(cast("list[object]", parts)) > 1:
            raise UnreadableError(
                "The request holds a document of more than one block, of which only the first would be read."
            )


def _message(reply: dict[str, Any]) -> object:
    # built as the SDK builds a reply it receives, so that a field value this
    # SDK version doesn't know yet is kept rather than an error
    from anthropic._models import construct_type
    from anthropic.types import Message

    return construct_type(type_=Message, value=reply)


def _system(system: object) -> str | None:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        texts = [
            str(cast("dict[str, Any]", block).get("text", ""))
            for block in cast("list[object]", system)
            if isinstance(block, dict)
        ]
        return "\n\n".join(texts) or None
    return None


def _stream(body: bytes) -> object:
    # The SDK's own accumulator, which inspect_ai's Anthropic provider also
    # relies on. inspect_ai has no public function that turns a stream into a
    # ModelOutput, so the provider's own repairs on top of the accumulator
    # (compaction content and the code execution container) are not applied
    # here. The accumulator is in a private module of the SDK, and newer
    # releases take the caller's `json_bufs`.
    from anthropic._models import construct_type
    from anthropic.lib.streaming import (
        _messages,  # pyright: ignore[reportPrivateUsage]
    )
    from anthropic.types import RawMessageStreamEvent

    accumulate = cast(Any, _messages.accumulate_event)
    extra: dict[str, Any] = (
        {"json_bufs": {}}
        if "json_bufs" in inspect.signature(accumulate).parameters
        else {}
    )
    snapshot: Any = None
    stopped = False
    for event in stream_events(body):
        kind = event.get("type")
        if kind == "ping":
            continue
        if kind == "error":
            raise UnreadableError(
                "The reply stream ended in an error from the provider."
            )
        stopped = stopped or kind == "message_stop"
        if kind == "content_block_delta":
            delta: object = event.get("delta")
            delta_kind = _kind(delta)
            if delta_kind not in _DELTAS:
                raise UnreadableError(
                    f"The reply stream holds content of a kind that can't be read: {delta_kind!r}."
                )
        try:
            snapshot = accumulate(
                event=construct_type(type_=RawMessageStreamEvent, value=event),
                current_snapshot=snapshot,
                **extra,
            )
        except Exception as ex:
            # the accumulator raises errors of many kinds on events out of order
            raise UnreadableError("The reply stream can't be read.") from ex
    if snapshot is None or not stopped:
        raise UnreadableError("The reply stream is not a complete message.")
    return cast(object, snapshot)
