from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ModelOutput,
    messages_from_openai,
    messages_from_openai_responses,
    messages_to_openai,
    messages_to_openai_responses,
    model_output_from_openai,
    model_output_from_openai_responses,
)

from ._wire import UnreadableError, json_object, stream_events


async def read_reply(
    content_type: str, body: bytes, chat: bool | None = None
) -> ModelOutput:
    if content_type.startswith("text/event-stream"):
        events = list(stream_events(body))
        if chat is None:
            chat = any(
                event.get("object") == "chat.completion.chunk" for event in events
            )
        reply = _chat_stream(events) if chat else _responses_stream(events)
    else:
        parsed = json_object(body, "reply")
        if chat is None:
            kind = parsed.get("object")
            if kind not in ("chat.completion", "response"):
                raise UnreadableError(
                    f"There is no reader for an OpenAI {kind!r} reply."
                )
            chat = kind == "chat.completion"
        reply = _completion(parsed) if chat else _response(parsed)
    if chat:
        for choice in cast("list[Any]", getattr(reply, "choices", None) or []):
            _check_calls(getattr(getattr(choice, "message", None), "tool_calls", None))
    try:
        # objects, not dicts, so that the converters don't validate them again
        output = await (
            model_output_from_openai(cast(Any, reply))
            if chat
            else model_output_from_openai_responses(cast(Any, reply))
        )
    except Exception as ex:
        # the converters raise errors of many kinds on content they can't read
        raise UnreadableError("The reply holds content that can't be read.") from ex
    if len(output.choices) > 1:
        raise UnreadableError(
            "The reply holds more than one choice, and only the first would be judged."
        )
    return output


async def read_chat_request(request: dict[str, Any]) -> list[ChatMessage]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise UnreadableError("The request has no list of messages.")
    for entry in cast("list[object]", messages):
        if isinstance(entry, dict):
            _check_calls(cast("dict[str, Any]", entry).get("tool_calls"))
    try:
        return await messages_from_openai(cast(Any, messages), _model(request))
    except Exception as ex:
        # the converters raise errors of many kinds on content they can't read
        raise UnreadableError("The request holds a message that can't be read.") from ex


async def read_responses_request(request: dict[str, Any]) -> list[ChatMessage]:
    if (
        request.get("previous_response_id") is not None
        or request.get("conversation") is not None
    ):
        raise UnreadableError(
            "The request continues a conversation the provider keeps, which this sidecar is not shown."
        )
    if request.get("background"):
        raise UnreadableError(
            "The request asks for a reply that is fetched later, which this sidecar would not be shown."
        )
    entries: object = request.get("input")
    if isinstance(entries, str):
        # the Responses API takes a lone user message as a string
        entries = [{"role": "user", "content": entries}]
    if not isinstance(entries, list):
        raise UnreadableError("The request has no input.")
    try:
        messages = await messages_from_openai_responses(
            cast(Any, entries), _model(request)
        )
    except Exception as ex:
        # the converters raise errors of many kinds on content they can't read
        raise UnreadableError("The request holds input that can't be read.") from ex
    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions:
        return [ChatMessageSystem(content=instructions), *messages]
    return messages


async def chat_rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    return cast("list[Any]", await messages_to_openai([message, *results]))


async def responses_rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    # written by inspect_ai, so a reasoning item goes back with the calls it
    # led to
    return cast("list[Any]", await messages_to_openai_responses([message, *results]))


def _check_calls(calls: object) -> None:
    # inspect_ai's Chat Completions converters skip a call that is not a function
    # call, such as a custom tool's, without an error. See
    # chat_tool_calls_from_openai() and messages_from_openai() in inspect_ai's
    # src/inspect_ai/model/_openai.py.
    for call in cast("list[object]", calls or []):
        kind = (
            cast("dict[str, Any]", call).get("type")
            if isinstance(call, dict)
            else getattr(call, "type", None)
        )
        if kind != "function":
            raise UnreadableError(
                f"The body holds a tool call of a kind that can't be read: {kind!r}."
            )


def _model(request: dict[str, Any]) -> str | None:
    model = request.get("model")
    return model if isinstance(model, str) else None


# Each reply is built as the SDK builds one it receives, so that a field value
# this SDK version doesn't know yet, such as a new `finish_reason`, is kept
# rather than an error. inspect_ai's public converters run pydantic's strict
# `model_validate` when they are given a dict, which would make such a reply
# unreadable; they take the built object as it is.


def _completion(reply: dict[str, Any]) -> object:
    from openai._models import construct_type
    from openai.types.chat import ChatCompletion

    return construct_type(type_=ChatCompletion, value=reply)


def _response(reply: dict[str, Any]) -> object:
    from openai._models import construct_type
    from openai.types.responses import Response

    return construct_type(type_=Response, value=reply)


def _chat_stream(events: list[dict[str, Any]]) -> object:
    # the SDK's own accumulator, which inspect_ai's provider relies on
    from openai._models import construct_type
    from openai.lib.streaming.chat import ChatCompletionStreamState
    from openai.types.chat import ChatCompletionChunk

    if any("error" in event for event in events):
        raise UnreadableError("The reply stream ended in an error from the provider.")
    state = ChatCompletionStreamState()
    try:
        for event in events:
            state.handle_chunk(
                cast(Any, construct_type(type_=ChatCompletionChunk, value=event))
            )
        snapshot = state.current_completion_snapshot
    except Exception as ex:
        # the accumulator raises errors of many kinds on chunks out of order
        raise UnreadableError("The reply stream can't be read.") from ex
    # a choice has no finish reason until the chunk that ends it, whatever the
    # SDK's types say
    if not snapshot.choices or any(
        getattr(choice, "finish_reason", None) is None for choice in snapshot.choices
    ):
        raise UnreadableError("The reply stream is not a complete message.")
    return cast(object, snapshot)


def _responses_stream(events: list[dict[str, Any]]) -> object:
    for event in events:
        kind = event.get("type")
        if kind in ("response.failed", "error"):
            raise UnreadableError(
                "The reply stream ended in an error from the provider."
            )
        # the event that ends the stream holds the whole response
        if kind in ("response.completed", "response.incomplete"):
            response = event.get("response")
            if isinstance(response, dict):
                return _response(cast("dict[str, Any]", response))
    raise UnreadableError("The reply stream is not a complete response.")
