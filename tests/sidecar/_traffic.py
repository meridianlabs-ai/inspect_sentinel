import hashlib
import hmac
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import AfterToolCall, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Reply, Request

DATA = Path(__file__).parent / "data"

STREAM = "text/event-stream; charset=utf-8"
JSON = "application/json"

# Replies recorded from the OpenAI API, each calling get_weather for Paris,
# Tokyo and Lima at once.
OPENAI_RECORDED = {
    "chat": "openai-chat.json",
    "chat stream": "openai-chat.sse",
    "responses": "openai-responses.json",
    "responses stream": "openai-responses.sse",
}


def anthropic_recorded(name: str) -> bytes:
    """A reply stream recorded from the Anthropic API."""
    return (DATA / f"{name}.sse").read_bytes()


def openai_recorded(name: str) -> tuple[str, bytes]:
    """The content type and body of a reply recorded from the OpenAI API."""
    file = OPENAI_RECORDED[name]
    return (STREAM if file.endswith(".sse") else JSON), (DATA / file).read_bytes()


def message(*content: dict[str, Any], stop_reason: str = "tool_use") -> bytes:
    """An Anthropic reply that was not streamed."""
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()


def weather(location: str, id: str = "toolu_1") -> dict[str, Any]:
    """An Anthropic tool call to get_weather."""
    return {
        "type": "tool_use",
        "id": id,
        "name": "get_weather",
        "input": {"location": location},
    }


def ran(
    location: str, result: str, id: str = "toolu_1", *, error: bool = False
) -> list[dict[str, Any]]:
    """A weather lookup the model made and the client's result for it: the two Anthropic messages a later request holds."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": id,
        "content": result,
    }
    if error:
        block["is_error"] = True
    return [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Looking."}, weather(location, id)],
        },
        {"role": "user", "content": [block]},
    ]


def request(
    id: str = "x",
    user_text: str = "What is the weather?",
    then: Sequence[dict[str, Any]] = (),
    *,
    sample: str | None = None,
    **fields: Any,
) -> Request:
    """An Anthropic request as a proxy hands it over, its conversation followed by `then`, from the Inspect sample whose uuid is `sample` if given."""
    return Request(
        id=id,
        headers=inspect_headers(sample_uuid=sample) if sample is not None else {},
        provider="anthropic",
        model="claude-test",
        user="user-1",
        body=json.dumps(
            {
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": user_text}, *then],
                **fields,
            }
        ).encode(),
    )


SAMPLE: dict[str, Any] = {
    "version": 1,
    "eval_id": "eval_789",
    "task": "philadelphia",
    "sample_id": 1,
    "epoch": 2,
    "sample_uuid": "sample_123",
    "attempt": "attempt_1",
    "attempt_number": 1,
}


TOKEN = "eval-789-token-that-inspect-made-for-these-tests"
"""The token the eval `SAMPLE` names registered with, as Inspect makes one: at least 32 characters."""


def signed(token: str, values: Sequence[object]) -> str:
    """An HMAC as Inspect makes one: SHA-256, in hex, over a compact JSON list of the values as strings."""
    message = json.dumps([str(value) for value in values], separators=(",", ":"))
    return hmac.new(token.encode(), message.encode(), hashlib.sha256).hexdigest()


def inspect_headers(
    *,
    irid: str = "request_456",
    token: str = TOKEN,
    timestamp: int | None = None,
    **changes: Any,
) -> dict[str, str]:
    """The headers an Inspect eval sends with each model request, for `SAMPLE` with `changes` and the request id `irid`, signed with `token` at `timestamp`, now if not given."""
    fields = SAMPLE | changes
    fields["timestamp"] = int(time.time()) if timestamp is None else timestamp
    fields["mac"] = signed(
        token,
        [
            "model-request",
            *(
                fields[name]
                for name in (
                    "task",
                    "sample_id",
                    "epoch",
                    "sample_uuid",
                    "attempt",
                    "attempt_number",
                )
            ),
            irid,
            fields["timestamp"],
        ],
    )
    return {"x-inspect-context": json.dumps(fields), "x-irid": irid}


def eval_call(
    method: str,
    eval_id: str = "eval_789",
    *,
    token: str = TOKEN,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """What a call to the endpoint for evals presents for the eval, signed with `token` at `timestamp`, now if not given."""
    at = int(time.time()) if timestamp is None else timestamp
    return {
        "eval_id": eval_id,
        "timestamp": at,
        "mac": signed(token, [method, eval_id, at]),
    }


async def holding(
    handler: Handler, eval_id: str = "eval_789", token: str = TOKEN
) -> Handler:
    """The handler, once it holds the eval's token, as the eval's first registration leaves it, so that it can check the eval's requests."""
    await handler.keep_eval_token(eval_id, token)
    return handler


def reply(
    body: bytes, *, id: str = "x", content_type: str = JSON, status: int = 200
) -> Reply:
    """An Anthropic reply as a proxy hands it over."""
    return Reply(
        id=id,
        provider="anthropic",
        model="claude-test",
        user="user-1",
        status=status,
        content_type=content_type,
        body=body,
    )


@protocol
def weather_rule(action: str = "reject", city: str = "Tokyo") -> Protocol:
    """Decide `action` for a weather lookup of `city`, and let everything else go."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") != city:
            return Decision.proceed()
        if action == "reject":
            return Decision.reject(
                f"lookup of {city}", message=f"{city} is not allowed."
            )
        if action == "terminate":
            return Decision.terminate(f"lookup of {city}")
        if action == "escalate":
            return Decision.escalate(f"lookup of {city}")
        if action == "modify":
            changed = replace(step.call, arguments={"location": "Oslo"})
            return Decision(action="modify", modified=changed)
        raise RuntimeError("the rule broke")

    return decide


@protocol
def every_call_rule() -> Protocol:
    """Reject every weather lookup, each with a reason naming its city, so that a turn is refused whole."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        city = step.call.arguments.get("location")
        return Decision.reject(f"lookup of {city}", message=f"{city} is not allowed.")

    return decide


@protocol
def storm_rule(action: str = "terminate") -> Protocol:
    """Decide `action` for a result that reports a storm, and let everything else go."""

    async def decide(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        if "storm" not in step.result.text:
            return Decision.proceed()
        if action == "terminate":
            return Decision.terminate("a storm")
        if action == "escalate":
            return Decision.escalate("a storm")
        if action == "reject":
            return Decision.reject("a storm")
        raise RuntimeError("the rule broke")

    return decide


def openai_request(api: str, then: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """An OpenAI request body for `chat` or `responses`, its conversation followed by `then`."""
    opening = {"role": "user", "content": "What is the weather?"}
    if api == "chat":
        return {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                opening,
                *then,
            ],
        }
    return {
        "model": "gpt-test",
        "instructions": "You are helpful.",
        "input": [opening, *then],
        "store": False,
    }


def openai_ran(api: str, result: str) -> list[dict[str, Any]]:
    """A weather lookup the model made and the client's result for it, as a later OpenAI request holds them."""
    arguments = json.dumps({"location": "Oslo"})
    if api == "chat":
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": arguments},
        }
        return [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": result},
        ]
    return [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": arguments,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": result},
    ]
