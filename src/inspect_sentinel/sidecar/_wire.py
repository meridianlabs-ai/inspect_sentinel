from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, Literal, TypeAlias, cast

from inspect_ai.model import ChatMessage, ModelOutput

Format: TypeAlias = Literal["anthropic-messages", "openai-chat", "openai-responses"]
"""Which API a request or a reply is written in. A provider can have more than one, and a proxy can serve one provider's model through another provider's API."""

# the line endings an event stream may use. str.splitlines breaks at more,
# some of which JSON allows unescaped inside a string
_LINE_END = re.compile(r"\r\n|\r|\n")


class UnreadableError(Exception):
    """A request or reply couldn't be read into Inspect's types, so it can't be judged."""


async def read_request(
    provider: str, body: bytes, format: Format | None = None
) -> list[ChatMessage]:
    """Read a request body into the messages the model is being sent.

    Args:
        provider: Whose API the body is written for, e.g. `anthropic`.
        body: The request body, in the provider's own format.
        format: Which API the body is written in, if the proxy said. It is then read as that, whatever the provider.

    Raises:
        UnreadableError: If there is no reader for the provider's API, or the body isn't what that API takes.
    """
    request = json_object(body, "request")
    match _request_format(provider, request, format):
        case "anthropic-messages":
            from . import _anthropic

            return await _anthropic.read_request(request)
        case "openai-chat":
            from . import _openai

            return await _openai.read_chat_request(request)
        case "openai-responses":
            from . import _openai

            return await _openai.read_responses_request(request)


async def read_reply(
    provider: str, content_type: str, body: bytes, format: Format | None = None
) -> ModelOutput:
    """Read a reply body, streamed or not, into the model's output.

    Args:
        provider: Whose API the body is written in, e.g. `anthropic`.
        content_type: The reply's `content-type`, which says whether the body is a stream.
        body: The whole reply body, in the provider's own format.
        format: Which API the body is written in, if the proxy said. It is then read as that, whatever the provider.

    Raises:
        UnreadableError: If there is no reader for the provider's API, the body isn't a complete reply, or it holds content that can't be read.
    """
    if format == "anthropic-messages" or (format is None and provider == "anthropic"):
        from . import _anthropic

        return await _anthropic.read_reply(content_type, body)
    if format is not None or provider == "openai":
        from . import _openai

        return await _openai.read_reply(
            content_type, body, None if format is None else format == "openai-chat"
        )
    raise UnreadableError(f"There is no reader for provider {provider!r}.")


def json_object(body: bytes, what: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(body)
    except ValueError as ex:
        raise UnreadableError(f"The {what} body is not JSON.") from ex
    if not isinstance(parsed, dict):
        raise UnreadableError(f"The {what} body is not a JSON object.")
    return cast("dict[str, Any]", parsed)


def stream_events(body: bytes) -> Iterator[dict[str, Any]]:
    held: list[str] = []
    # a blank line ends an event, and so does the end of the stream
    for line in [*_LINE_END.split(body.decode("utf-8", errors="replace")), ""]:
        if line.startswith("data:"):
            held.append(line[5:].lstrip(" "))
            continue
        if line:
            continue
        data, held = "\n".join(held), []
        # OpenAI's chat stream ends with an event that is not JSON
        if not data or data == "[DONE]":
            continue
        try:
            event: object = json.loads(data)
        except ValueError as ex:
            raise UnreadableError(
                "The reply stream holds an event that is not JSON."
            ) from ex
        if not isinstance(event, dict):
            raise UnreadableError(
                "The reply stream holds an event that is not a JSON object."
            )
        yield cast("dict[str, Any]", event)


def _request_format(
    provider: str, request: dict[str, Any], format: Format | None
) -> Format:
    if format is not None:
        return format
    if provider == "anthropic":
        return "anthropic-messages"
    if provider == "openai":
        # the proxy named the provider, and not which of its APIs a request is for
        if "messages" in request:
            return "openai-chat"
        if "input" in request:
            return "openai-responses"
        raise UnreadableError("There is no reader for this OpenAI API.")
    raise UnreadableError(f"There is no reader for provider {provider!r}.")
