"""What a client is told when the sidecar refuses: errors in each provider's own format, which its SDK raises as one of its own API errors."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, cast

from ._exchange import Refuse
from ._registration import UnregisteredSampleError
from ._wire import Format, StreamEvent, UnreadableError

logger = logging.getLogger(__name__)


class TimeLimitError(TimeoutError):
    """The sentinel didn't finish judging a request or a reply within the handler's time limit, and was cancelled."""


_ANTHROPIC_ERRORS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def refusal_for(ex: Exception, what: str, id: str, *, reply: bool) -> Refuse:
    """The refusal for a request or reply that couldn't be judged, with the status a proxy hook's failure would bring and a reason that never quotes the traffic.

    Args:
        ex: Why it couldn't be judged.
        what: `request` or `reply`.
        id: The proxy's id for the request, for the log.
        reply: Whether it is a reply, which a provider sent rather than the client.
    """
    if isinstance(ex, TimeLimitError):
        return Refuse(504, str(ex))
    if isinstance(ex, UnregisteredSampleError):
        return Refuse(400, str(ex))
    if isinstance(ex, UnreadableError):
        # why is kept here, since it may quote the request or the reply
        logger.warning("The %s %r was not judged: %s", what, id, ex)
        return Refuse(
            502 if reply else 400,
            f"The {what} couldn't be read, so it was not judged.",
        )
    logger.error("The %s %r was not judged.", what, id, exc_info=ex)
    return Refuse(500, f"The sentinel failed, so the {what} was not judged.")


def bounded(status: int) -> int:
    """A refusal's status, which is an error: one outside 400 to 599 is a 500."""
    return status if 400 <= status <= 599 else 500


def error_body(format: Format | None, status: int, message: str) -> dict[str, Any]:
    """An error in the provider's own format, as the body of an HTTP error."""
    if format == "anthropic-messages":
        kind = _ANTHROPIC_ERRORS.get(
            status, "api_error" if status >= 500 else "invalid_request_error"
        )
        return {"type": "error", "error": {"type": kind, "message": message}}
    return {
        "error": {
            "message": message,
            "type": _openai_code(status),
            "param": None,
            "code": None,
        }
    }


def error_event(
    format: Format, refused: Refuse, events: Sequence[StreamEvent]
) -> bytes:
    """An error event that refuses a turn once some of its stream has gone out, which the provider's SDK raises an exception on.

    For an Inspect client, the `sentinel` field goes where the SDK keeps it: beside the error in Anthropic's event, whose whole data is the exception's body, and inside the error, at `error.provider_specific_fields.sentinel`, for OpenAI's, whose exception keeps only the error.

    Args:
        format: Which API the stream is written in.
        refused: The refusal.
        events: The stream's events so far, as the provider sent them.
    """
    status, message = bounded(refused.status), refused.message
    field = dict(refused.sentinel) if refused.sentinel is not None else None
    if format == "anthropic-messages":
        data = error_body(format, status, message)
        if field is not None:
            data["sentinel"] = field
        return _written(data, event="error")
    code = _openai_code(status)
    error: dict[str, Any] = {
        "message": message,
        "type": code,
        "param": None,
        "code": code,
    }
    if field is not None:
        error["provider_specific_fields"] = {"sentinel": field}
    if format == "openai-chat":
        return _written({"error": error})
    # The SDK raises on an event with an error at its top, and a client that
    # reads the stream's events sees the response fail: one error, in both.
    begun: dict[str, Any] = {}
    sequence = 0
    for event in events:
        data = event.data or {}
        number = data.get("sequence_number")
        if isinstance(number, int):
            sequence = max(sequence, number)
        if data.get("type") in ("response.created", "response.in_progress"):
            response = data.get("response")
            if isinstance(response, dict):
                begun = cast("dict[str, Any]", response)
    failed = {
        "type": "response.failed",
        "sequence_number": sequence + 1,
        "response": {
            **begun,
            "status": "failed",
            "error": {"code": code, "message": message},
        },
        "error": error,
    }
    return _written(failed, event="response.failed")


def _openai_code(status: int) -> str:
    # what a client tries again after, as for the provider's own errors: a
    # server error, and not a refusal of what was asked
    return "server_error" if status >= 500 else "invalid_request_error"


def _written(data: dict[str, Any], *, event: str | None = None) -> bytes:
    lines = [f"event: {event}"] if event is not None else []
    lines.append(f"data: {json.dumps(data)}")
    return ("\n".join(lines) + "\n\n").encode()
