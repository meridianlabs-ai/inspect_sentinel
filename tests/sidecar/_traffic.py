import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
