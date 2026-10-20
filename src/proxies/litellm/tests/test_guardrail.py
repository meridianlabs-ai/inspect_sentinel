"""The guardrail, handed LiteLLM's own classes as LiteLLM's proxy hands them over.

These tests run in an environment of their own, with one LiteLLM version, and import nothing of inspect_sentinel, as the guardrail doesn't (`make test-litellm`). A test that needs a real sidecar starts one from the repository's own environment.
"""

import asyncio
import base64
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx
import jsonschema
import openai
import pytest
import uvicorn
from fastapi import HTTPException
from openai.types.responses import Response, ResponseReasoningItem
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse
from starlette.responses import Response as HttpResponse
from starlette.routing import Route
from starlette.types import ASGIApp

_PACKAGE = Path(__file__).parents[1]
_REPOSITORY = Path(__file__).parents[4]
# the replies the sidecar's own tests replay
DATA = _REPOSITORY / "tests/sidecar/data"
# the one thing the guardrail and the sidecar share: the description of the
# methods and their results
_SCHEMA: dict[str, Any] = json.loads(
    (
        _REPOSITORY / "src/inspect_sentinel/sidecar/proxy_endpoint.schema.json"
    ).read_text()
)
_METHODS: dict[str, Any] = _SCHEMA["methods"]
PATH = "/proxy/v1"

# LiteLLM's own classes, which it hands a guardrail on its OpenAI routes
_litellm: Any = importlib.import_module("litellm.types.utils")
_openai: Any = importlib.import_module("litellm.types.llms.openai")
_USER: Any = importlib.import_module("litellm.proxy._types").UserAPIKeyAuth(
    user_id="user-1"
)

_QUESTION = {"role": "user", "content": "What is the weather?"}
_REFUSAL = {"action": "refuse", "status": 403, "message": "Not allowed."}
_UNANSWERED = (503, "The sentinel sidecar gave no usable answer.")


def stream_events(body: bytes) -> list[dict[str, Any]]:
    """The JSON object of each event in a stream."""
    return [
        json.loads(line[5:].strip())
        for line in body.decode().splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]


def anthropic_recorded(name: str) -> bytes:
    """A reply stream recorded from the Anthropic API."""
    return (DATA / f"{name}.sse").read_bytes()


def message(*content: dict[str, Any]) -> bytes:
    """An Anthropic reply that was not streamed."""
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": list(content),
            "stop_reason": "tool_use",
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


def _recorded_events(name: str) -> list[dict[str, Any]]:
    return stream_events((DATA / name).read_bytes())


def _response_event(event: dict[str, Any]) -> Any:
    # LiteLLM has a class for each kind of event in a Responses stream
    for name in dir(_openai):
        value: Any = getattr(_openai, name)
        fields = getattr(value, "model_fields", None)
        if isinstance(value, type) and fields and "type" in fields:
            kinds = getattr(fields["type"].annotation, "__args__", ())
            if event["type"] in [getattr(kind, "value", kind) for kind in kinds]:
                return value(**event)
    raise AssertionError(f"LiteLLM has no class for a {event['type']} event.")


def _notes() -> dict[str, Any]:
    return {"additional_headers": {"x-litellm-model-group": "a-model"}}


def _messages_whole() -> Any:
    return json.loads(message(weather("Paris"))) | {"_hidden_params": _notes()}


def _messages_stream() -> list[Any]:
    # LiteLLM passes on a provider's bytes in whatever pieces they came in
    sent = anthropic_recorded("single-tool-call")
    return [sent[:100], sent[100:]]


def _chat_whole() -> Any:
    reply = _litellm.ModelResponse(
        **json.loads((DATA / "openai-chat.json").read_text())
    )
    reply._hidden_params = _notes()
    return reply


def _chat_stream() -> list[Any]:
    return [
        _litellm.ModelResponseStream(**event)
        for event in _recorded_events("openai-chat.sse")
    ]


def _responses_whole() -> Any:
    reply = _openai.ResponsesAPIResponse(
        **json.loads((DATA / "openai-responses.json").read_text())
    )
    reply._hidden_params = _notes()
    return reply


def _responses_stream() -> list[Any]:
    return [
        _response_event(event) for event in _recorded_events("openai-responses.sse")
    ]


@dataclass(frozen=True)
class _On:
    """One of LiteLLM's routes: what it calls the route, what a request to it holds, and the reply it hands a guardrail. Every reply here looks up the weather in Paris."""

    call_type: str
    provider: str
    format: str
    body: dict[str, Any]
    whole: Callable[[], Any]
    stream: Callable[[], list[Any]]
    # the field LiteLLM keeps its own notes on the caller in, on this route
    notes: str = "litellm_metadata"


_MESSAGES = _On(
    "anthropic_messages",
    "anthropic",
    "anthropic-messages",
    {"model": "a-model", "max_tokens": 100, "messages": [_QUESTION]},
    _messages_whole,
    _messages_stream,
)
_CHAT = _On(
    "acompletion",
    "openai",
    "openai-chat",
    {"model": "a-model", "max_tokens": 100, "messages": [_QUESTION]},
    _chat_whole,
    _chat_stream,
    notes="metadata",
)
_RESPONSES = _On(
    "aresponses",
    "openai",
    "openai-responses",
    {"model": "a-model", "max_output_tokens": 100, "input": [_QUESTION]},
    _responses_whole,
    _responses_stream,
)
_ROUTES = [
    pytest.param(
        route, streamed, id=f"{route.format}{', streamed' if streamed else ''}"
    )
    for route in (_MESSAGES, _CHAT, _RESPONSES)
    for streamed in (False, True)
]


def _data(on: _On, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """A request as LiteLLM hands it to a guardrail: its own fields among LiteLLM's working state."""
    notes = {"user_api_key_hash": "a-hash", "headers": headers or {}}
    return {
        **on.body,
        on.notes: notes,
        "litellm_call_id": "call-1",
        "litellm_logging_obj": object(),
        "secret_fields": object(),
        "proxy_server_request": {
            "url": "http://litellm/",
            "method": "POST",
            "headers": headers or {},
            "body": {**on.body, on.notes: notes},
        },
    }


def _load(monkeypatch: pytest.MonkeyPatch) -> Any:
    # as LiteLLM has it: installed, and importable by its own name
    return importlib.import_module("inspect_sentinel_litellm")


def _make(monkeypatch: pytest.MonkeyPatch, url: str | None, **settings: Any) -> Any:
    """The guardrail as LiteLLM makes it: its own settings for a guardrail, and those from the config."""
    told: dict[str, Any] = {
        "guardrail_name": "sentinel",
        "event_hook": ["pre_call", "post_call"],
        "default_on": True,
        "unreachable_fallback": "fail_closed",
        "api_base": url,
    }
    return _load(monkeypatch).LiteLLMGuardrail(**(told | settings))


async def _chunks(held: list[Any]) -> AsyncIterator[Any]:
    for chunk in held:
        yield chunk


def _exchange(
    guardrail: Any,
    on: _On,
    data: dict[str, Any],
    streamed: bool,
    out: list[Any] | None = None,
) -> tuple[Any, Any]:
    """Put a request and its reply through the guardrail as LiteLLM does. A streamed reply's chunks are also added to `out` as they are let go."""
    let_go: list[Any] = [] if out is None else out

    async def run() -> tuple[Any, Any]:
        try:
            sent = await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            if not streamed:
                return sent, await guardrail.async_post_call_success_hook(
                    data, _USER, on.whole()
                )
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                _USER, _chunks(on.stream()), data
            ):
                let_go.append(chunk)
            return sent, let_go
        finally:
            await guardrail.aclose()

    return asyncio.run(run())


@contextmanager
def _serving(app: ASGIApp) -> Generator[str]:
    """The address of a real server for an app, since the guardrail makes real HTTP calls."""
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        serving.join()


_RULE = """
from inspect_sentinel import BeforeToolCall, Context, Decision, Escalations, protocol


@protocol
def weather_rule():
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        location = step.call.arguments.get("location")
        if {city!r} in (None, location):
            if {action!r} == "terminate":
                return Decision.terminate("a lookup")
            return Decision.reject("a lookup", message=f"{{location}} is not allowed.")
        return Decision.proceed()

    return decide
"""


def _sidecar_python() -> Path:
    # the repository's own environment, which has the sidecar and no LiteLLM
    given = os.environ.get("INSPECT_SENTINEL_PYTHON")
    python = Path(given) if given else _REPOSITORY / ".venv/bin/python"
    if not python.exists():
        if os.environ.get("GITHUB_ACTIONS"):
            raise FileNotFoundError(f"No sidecar environment at {python}.")
        pytest.skip(
            f"No sidecar environment at {python}: set one up as CONTRIBUTING.md describes."
        )
    return python


@contextmanager
def _sidecar(
    city: str | None,
    tmp_path: Path,
    *,
    release: str = "whole",
    action: str = "reject",
) -> Generator[str]:
    """The address of a real sidecar, whose sentinel rejects a weather lookup of `city`, or every lookup for None, or ends the run there, releasing replies as `release` says."""
    rule = tmp_path / "weather_rule.py"
    rule.write_text(_RULE.format(city=city, action=action))
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    serving = subprocess.Popen(
        [
            str(_sidecar_python()),
            "-c",
            # the rule is registered by importing it, as an installed package's
            # are by its entry point, and the sidecar is then started as usual
            "import weather_rule\n"
            "from inspect_sentinel.sidecar.__main__ import main\n"
            "main()",
            "--sentinel",
            "weather_rule",
            "--port",
            str(port),
            "--release",
            release,
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=1):
                    break
            except OSError:
                if serving.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"The sidecar did not start: {serving.stderr.read().decode() if serving.stderr else ''}"
                    ) from None
                time.sleep(0.1)
        yield url
    finally:
        serving.terminate()
        serving.communicate(timeout=30)


def _result(result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "result": result}


_KEEPALIVE = b": keepalive\n\n"

# LiteLLM passes on the bytes a guardrail yields on Chat and Responses from
# 1.105, and writes them as JSON before
_PASSES_BYTES_ON = tuple(
    int(part) for part in importlib.metadata.version("litellm").split(".")[:2]
) >= (1, 105)

_STREAM_METHODS = frozenset(
    {"begin_reply", "reply_chunk", "end_reply", "abort_reply", "next_release"}
)


def _scripted(
    calls: list[Any],
    *answers: Any,
    status: int = 200,
    streamed: list[Any] | None = None,
) -> Starlette:
    """A stand-in for the sidecar that keeps the calls made to it and gives these answers in turn.

    A reply handed over as a reply stream is kept in `calls` as the `judge_reply` that would have held it whole, and answered from the next answer as the stream's outputs: the reply as handed over, then its end, for `pass`; the replacement, then its end, for `replace`; and a refusal, or the error, as it is. The calls of each stream are kept in `streamed`.
    """
    left = list(answers)
    streams: dict[str, dict[str, Any]] = {}

    async def call(http: HttpRequest) -> JSONResponse:
        asked = await http.json()
        method, params = asked["method"], asked.get("params", {})
        if method not in _STREAM_METHODS:
            calls.append(asked)
            return JSONResponse(left.pop(0), status_code=status)
        if streamed is not None:
            streamed.append(asked)
        if status != 200:
            return JSONResponse({}, status_code=status)
        taken = _result({"taken": True})
        if method == "begin_reply":
            streams[params["id"]] = {"begun": params, "data": b"", "ended": False}
            return JSONResponse(taken)
        stream = streams[params["id"]]
        if method == "reply_chunk":
            stream["data"] += base64.b64decode(params["data"])
            return JSONResponse(taken)
        if method == "end_reply":
            stream["ended"] = True
            return JSONResponse(taken)
        if method == "abort_reply":
            return JSONResponse(taken)
        if not stream["ended"]:
            await asyncio.sleep(0.01)
            return JSONResponse(_result({"position": params["after"], "outputs": []}))
        if "answer" not in stream:
            body = base64.b64encode(stream["data"]).decode()
            calls.append(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "judge_reply",
                    "params": stream["begun"] | {"body": body},
                }
            )
            stream["answer"] = _as_outputs(
                left.pop(0), body, stream["begun"].get("keepalive", True)
            )
        answer = stream["answer"]
        if "error" in answer:
            return JSONResponse(answer)
        outputs = answer["outputs"]
        return JSONResponse(
            _result({"position": len(outputs), "outputs": outputs[params["after"] :]})
        )

    return Starlette(routes=[Route(PATH, call, methods=["POST"])])


def _as_outputs(answer: dict[str, Any], body: str, carried: bool) -> dict[str, Any]:
    # a scripted answer to `judge_reply`, as a reply stream's outputs, which a
    # pass `kept_alive` opens with a keepalive where the hook can send one
    if "error" in answer:
        return answer
    result = answer["result"]
    action = result.get("action")
    if action == "pass":
        kept_alive = (
            [{"kind": "release", "data": base64.b64encode(_KEEPALIVE).decode()}]
            if result.get("kept_alive") and carried
            else []
        )
        return {
            "outputs": [
                *kept_alive,
                {"kind": "release", "data": body},
                {"kind": "finished"},
            ]
        }
    if action == "replace":
        return {
            "outputs": [
                {"kind": "release", "data": result["body"]},
                {"kind": "finished"},
            ]
        }
    if action == "refuse":
        refusal = {key: value for key, value in result.items() if key != "action"}
        return {"outputs": [{"kind": "refuse", **refusal}]}
    return {"outputs": [result]}


def _fits(part: dict[str, Any], value: Any) -> None:
    jsonschema.validate(value, {**_SCHEMA, **part}, cls=jsonschema.Draft202012Validator)


def _written(reply: Any) -> Any:
    """A reply, or a chunk of one, as LiteLLM writes it for the caller."""
    return json.loads(reply.model_dump_json(exclude_none=True, exclude_unset=True))


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_exchange_the_sentinel_allows_goes_on_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    data = _data(on)
    before = dict(data)

    with _sidecar("Oslo", tmp_path) as url:
        sent, reply = _exchange(_make(monkeypatch, url), on, data, streamed)

    assert sent is data
    assert {name: data[name] for name in before} == before
    if not streamed:
        # LiteLLM sends on the reply it has
        assert reply is None
    elif on is _MESSAGES:
        # the bytes as the provider sent them, in the parts the sidecar
        # released them in
        assert b"".join(reply) == b"".join(on.stream())
    else:
        assert [_written(chunk) for chunk in reply] == [
            _written(chunk) for chunk in on.stream()
        ]


def _error_event(chunk: Any) -> dict[str, Any]:
    """The error in the event a refused Chat Completions or Messages stream is answered with, written as LiteLLM looks for an error in a stream's first event."""
    assert isinstance(chunk, str)
    assert chunk.startswith("data: ")
    return json.loads(chunk.removeprefix("data: "))["error"]


# The Messages route's replies above make one call; these make three, Paris,
# Tokyo and Lima, as the recorded OpenAI replies do.
_THREE_CALLS = replace(
    _MESSAGES,
    whole=lambda: (
        json.loads(
            message(
                weather("Paris", "toolu_1"),
                weather("Tokyo", "toolu_2"),
                weather("Lima", "toolu_3"),
            )
        )
        | {"_hidden_params": _notes()}
    ),
    stream=lambda: [anthropic_recorded("parallel-tool-calls")],
)


def _events_out(sent: list[Any]) -> list[dict[str, Any]]:
    # each event of a stream as the caller is sent it: bytes as they are, and
    # LiteLLM's own objects as LiteLLM writes them
    events: list[dict[str, Any]] = []
    for chunk in sent:
        if isinstance(chunk, bytes):
            events += stream_events(chunk)
        else:
            events.append(_written(chunk))
    return events


def _call_ids(on: _On, reply: Any, streamed: bool) -> list[str]:
    """The ids of the calls a reply makes, in order, as the caller is sent it."""
    if not streamed:
        whole: Any = cast(Any, reply) if isinstance(reply, dict) else _written(reply)
        if on.format == "anthropic-messages":
            return [b["id"] for b in whole["content"] if b["type"] == "tool_use"]
        if on.format == "openai-chat":
            return [c["id"] for c in whole["choices"][0]["message"]["tool_calls"]]
        return [i["call_id"] for i in whole["output"] if i["type"] == "function_call"]
    ids: list[str] = []
    event: Any
    for event in _events_out(reply):
        if event.get("type") == "content_block_start":
            block: Any = event["content_block"]
            if block["type"] == "tool_use":
                ids.append(block["id"])
        choice: Any
        for choice in cast(list[Any], event.get("choices") or []):
            delta: Any = choice.get("delta") or {}
            call: Any
            for call in cast(list[Any], delta.get("tool_calls") or []):
                if call.get("id"):
                    ids.append(call["id"])
        if event.get("type") == "response.output_item.added":
            if event["item"]["type"] == "function_call":
                ids.append(event["item"]["call_id"])
    return ids


@pytest.mark.parametrize(
    ("on", "streamed"),
    [
        pytest.param(
            route, streamed, id=f"{route.format}{', streamed' if streamed else ''}"
        )
        for route in (_THREE_CALLS, _CHAT, _RESPONSES)
        for streamed in (False, True)
    ],
)
@pytest.mark.parametrize("release", ["whole", "early"])
def test_a_reply_with_a_rejected_call_goes_on_without_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    on: _On,
    streamed: bool,
    release: str,
) -> None:
    proposed = _call_ids(on, on.stream() if streamed else on.whole(), streamed)
    # the recorded Responses stream is the one reply that names the country too
    city = "Tokyo, Japan" if on is _RESPONSES and streamed else "Tokyo"

    with _sidecar(city, tmp_path, release=release) as url:
        _, reply = _exchange(_make(monkeypatch, url), on, _data(on), streamed)

    # the second call, Tokyo, is taken out, and the others go on
    assert _call_ids(on, reply, streamed) == [proposed[0], proposed[2]]


def _refused(
    guardrail: Any, on: _On, data: dict[str, Any], streamed: bool
) -> tuple[int, Any, Any, list[Any]]:
    """How the guardrail refused an exchange: the status and message the caller is given, the sentinel field, and what else went out.

    A streamed reply is refused with an event in place of the reply: on the Responses route a `response.failed` event with an error at its top, which holds the field and which LiteLLM sends on with status 200, and elsewhere an error event, which LiteLLM sends as an error under the status it names. A reply that isn't streamed is refused with an `HTTPException`, whose `provider_specific_fields` LiteLLM writes into the error.
    """
    let_go: list[Any] = []
    if streamed:
        _exchange(guardrail, on, data, streamed, let_go)
        if on is _RESPONSES:
            event = _written(let_go.pop())
            assert event["type"] == "response.failed"
            assert event["response"]["status"] == "failed"
            message = event["response"]["error"]["message"]
            # the error the OpenAI SDK raises on and keeps, which holds the field
            error = event["error"]
            assert error["message"] == message
            extra = error.get("provider_specific_fields") or {}
            return 200, message, extra.get("sentinel"), let_go
        error = _error_event(let_go.pop())
        extra: dict[str, Any] = error.get("provider_specific_fields") or {}
        return error["code"], error["message"], extra.get("sentinel"), let_go
    with pytest.raises(HTTPException) as refused:
        _exchange(guardrail, on, data, streamed, let_go)
    extra = getattr(refused.value, "provider_specific_fields", None) or {}
    return (
        refused.value.status_code,
        refused.value.detail,
        extra.get("sentinel"),
        let_go,
    )


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_the_sentinel_rejects_is_refused_in_its_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    # the recorded Responses stream is the one reply that names the country too
    city = "Paris, France" if on is _RESPONSES and streamed else "Paris"

    # every call, so that the turn is refused whole; the first is Paris
    with _sidecar(None, tmp_path) as url:
        status, message, _, let_go = _refused(
            _make(monkeypatch, url), on, _data(on), streamed
        )

    assert status == (200 if on is _RESPONSES and streamed else 400)
    assert message == (
        f"A sentinel rejected the call to get_weather: {city} is not allowed."
    )
    # none of a streamed reply had gone out
    assert let_go == []


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_calls_the_guardrail_makes_fit_the_schema_file(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    calls: list[Any] = []
    headers = {
        "X-Inspect-Context": '{"version": 1}',
        "x-irid": "request-1",
        "x-inspect-sample-uuid": "sample-1",
        "x-hawk-job-type": "eval-set",
        "user-agent": "a client",
        "x-api-key": "a provider's key",
    }
    passed = _result({"action": "pass"})
    stream_calls: list[Any] = []

    with _serving(_scripted(calls, passed, passed, streamed=stream_calls)) as url:
        guardrail = _make(monkeypatch, url, sidecar_headers=["x-hawk-*"])
        _exchange(guardrail, on, _data(on, headers), streamed)

    # a streamed reply is handed over as a reply stream, whose every call fits
    # its method
    assert bool(stream_calls) == streamed
    for each in stream_calls:
        _fits(_METHODS[each["method"]]["params"], each["params"])
    assert {each["method"] for each in stream_calls} >= (
        {"begin_reply", "reply_chunk", "end_reply", "next_release"}
        if streamed
        else set()
    )
    asked, held = calls
    assert (asked["jsonrpc"], asked["method"]) == ("2.0", "judge_request")
    assert (held["jsonrpc"], held["method"]) == ("2.0", "judge_reply")
    request, reply = asked["params"], held["params"]
    _fits(_METHODS["judge_request"]["params"], request)
    _fits(_METHODS["judge_reply"]["params"], reply)
    told = {
        "provider": on.provider,
        "format": on.format,
        "model": "a-model",
        "user": "user-1",
    }
    assert {name: request[name] for name in told} == told
    assert {name: reply[name] for name in told} == told
    # the guardrail's own id for the exchange, never LiteLLM's, which a
    # caller can set
    assert reply["id"] == request["id"] != "call-1"
    # LiteLLM names no runs of its own: the sidecar finds the run in
    # Inspect's header
    assert "run" not in request
    # Inspect's own, unread, and the one that is listed
    assert request["headers"] == {
        "x-inspect-context": '{"version": 1}',
        "x-irid": "request-1",
        "x-hawk-job-type": "eval-set",
    }
    # the request's own fields, and none of LiteLLM's notes on the caller
    assert request["body"] == on.body
    body = base64.b64decode(reply["body"])
    assert reply["status"] == 200
    if not streamed:
        assert reply["content_type"] == "application/json"
        whole = on.whole()
        assert json.loads(body) == (
            json.loads(message(weather("Paris")))
            if on is _MESSAGES
            else json.loads(whole.model_dump_json())
        )
    else:
        assert reply["content_type"] == "text/event-stream"
        if on is _MESSAGES:
            assert body == anthropic_recorded("single-tool-call")
        else:
            assert list(stream_events(body)) == [
                _written(chunk) for chunk in on.stream()
            ]


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_a_request_whose_inspect_header_cant_be_read_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fallback: str
) -> None:
    # anyone can write the header, so it can't be a way past the sentinel
    data = _data(_CHAT, {"x-inspect-context": "not json", "x-irid": "request-1"})

    with _sidecar("Oslo", tmp_path) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(
                _make(monkeypatch, url, unreachable_fallback=fallback),
                _CHAT,
                data,
                False,
            )

    assert refused.value.status_code == 400
    assert "x-inspect-context" in str(refused.value.detail)


def test_requests_that_share_litellms_call_id_are_told_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    passed = _result({"action": "pass"})
    # LiteLLM takes its call id from the caller's `x-litellm-call-id`
    first, second = _data(_CHAT), _data(_CHAT)

    async def both(guardrail: Any) -> None:
        try:
            await guardrail.async_pre_call_hook(_USER, None, first, "acompletion")
            await guardrail.async_pre_call_hook(_USER, None, second, "acompletion")
            for data in (first, second):
                await guardrail.async_post_call_success_hook(data, _USER, _CHAT.whole())
        finally:
            await guardrail.aclose()

    with _serving(_scripted(calls, *[passed] * 4)) as url:
        asyncio.run(both(_make(monkeypatch, url)))

    ids = [call["params"]["id"] for call in calls]
    # each reply is asked about under its own request's id
    assert ids[0] != ids[1]
    assert ids[2:] == ids[:2]


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
@pytest.mark.parametrize("on", [_MESSAGES, _CHAT, _RESPONSES])
def test_a_request_that_sets_extra_body_is_refused_unasked(
    monkeypatch: pytest.MonkeyPatch, on: _On, fallback: str
) -> None:
    calls: list[Any] = []
    data = _data(on)
    # LiteLLM would have this replace the messages the sidecar was shown
    data["extra_body"] = {"messages": [{"role": "user", "content": "Other."}]}

    with _serving(_scripted(calls)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(
                _make(monkeypatch, url, unreachable_fallback=fallback),
                on,
                data,
                False,
            )

    assert refused.value.status_code == 400
    assert "extra_body" in str(refused.value.detail)
    assert calls == []


# a value LiteLLM would rewrite the request from, for each such field
_REWRITING: dict[str, Any] = {
    "extra_body": {"messages": [{"role": "user", "content": "Other."}]},
    "ensure_alternating_roles": True,
    "user_continue_message": {"role": "user", "content": "Other."},
    "assistant_continue_message": {"role": "assistant", "content": "Other."},
    "prompt_id": "a-stored-prompt",
    "prompt_variables": {"city": "Tokyo"},
    "prompt_label": "production",
    "prompt_version": 2,
    "prompt_environment": "production",
    "langfuse_prompt_version": 2,
    "litellm_system_prompt": "Other.",
    "roles": {"user": {"pre_message": "Other."}},
    "initial_prompt_value": "Other.",
    "final_prompt_value": "Other.",
    "bos_token": "Other.",
    "eos_token": "Other.",
    "custom_prompt_dict": {"a-model": {"roles": {}}},
    "supports_system_message": False,
}


def test_every_field_litellm_rewrites_a_request_from_is_tested() -> None:
    from inspect_sentinel_litellm import _guardrail

    assert set(_REWRITING) == _guardrail._REWRITTEN_FROM  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
@pytest.mark.parametrize("name", sorted(_REWRITING))
def test_a_request_that_sets_a_field_litellm_rewrites_it_from_is_refused_unasked(
    monkeypatch: pytest.MonkeyPatch, name: str, fallback: str
) -> None:
    calls: list[Any] = []
    data = _data(_CHAT)
    data[name] = _REWRITING[name]

    # refused, even where LiteLLM is set to let through what wasn't judged
    with _serving(_scripted(calls)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(
                _make(monkeypatch, url, unreachable_fallback=fallback),
                _CHAT,
                data,
                False,
            )

    assert refused.value.status_code == 400
    assert f"`{name}`" in str(refused.value.detail)
    assert calls == []


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
@pytest.mark.parametrize(
    "server_url",
    ["litellm_proxy", "litellm_proxy/mcp/github", "https://litellm.example/mcp/github"],
    ids=["gateway", "gateway server", "gateway url"],
)
@pytest.mark.parametrize("on", [_MESSAGES, _CHAT, _RESPONSES])
def test_a_request_with_an_mcp_tool_litellms_gateway_would_run_is_refused_unasked(
    monkeypatch: pytest.MonkeyPatch, on: _On, server_url: str, fallback: str
) -> None:
    calls: list[Any] = []
    data = _data(on)
    # LiteLLM would run the model's calls to it, and send the provider the
    # results in requests no guardrail is shown
    data["tools"] = [
        {"type": "mcp", "server_label": "github", "server_url": server_url}
    ]

    with _serving(_scripted(calls)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(
                _make(monkeypatch, url, unreachable_fallback=fallback),
                on,
                data,
                False,
            )

    assert refused.value.status_code == 400
    assert "MCP gateway" in str(refused.value.detail)
    assert calls == []


def test_a_request_with_an_mcp_tool_the_provider_runs_is_asked_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    data = _data(_RESPONSES)
    # the provider calls this server itself, which LiteLLM leaves alone
    tool = {
        "type": "mcp",
        "server_label": "wiki",
        "server_url": "https://mcp.example/mcp",
    }
    data["tools"] = [tool]
    data["proxy_server_request"]["body"]["tools"] = [tool]

    async def ask(guardrail: Any) -> Any:
        try:
            return await guardrail.async_pre_call_hook(_USER, None, data, "aresponses")
        finally:
            await guardrail.aclose()

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        sent = asyncio.run(ask(_make(monkeypatch, url)))

    assert sent is data
    assert [call["method"] for call in calls] == ["judge_request"]


def test_a_litellm_without_the_gateways_test_has_every_mcp_tool_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler: Any = importlib.import_module(
        "litellm.responses.mcp.litellm_proxy_mcp_handler"
    ).LiteLLM_Proxy_MCP_Handler
    monkeypatch.delattr(handler, "_should_use_litellm_mcp_gateway")
    calls: list[Any] = []
    data = _data(_RESPONSES)
    # one the gateway would leave to the provider, were LiteLLM to say so
    data["tools"] = [
        {"type": "mcp", "server_label": "wiki", "server_url": "https://mcp.example/mcp"}
    ]

    with _serving(_scripted(calls)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), _RESPONSES, data, False)

    assert "MCP gateway" in str(refused.value.detail)
    assert calls == []


def test_a_request_with_an_empty_extra_body_is_asked_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    data = _data(_CHAT)
    data["extra_body"] = {}

    async def ask(guardrail: Any) -> Any:
        try:
            return await guardrail.async_pre_call_hook(_USER, None, data, "acompletion")
        finally:
            await guardrail.aclose()

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        sent = asyncio.run(ask(_make(monkeypatch, url)))

    assert sent is data
    assert [call["method"] for call in calls] == ["judge_request"]


def test_request_is_put_as_it_will_go_to_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    data = _data(_CHAT)
    # an earlier guardrail changed the request after LiteLLM recorded it
    data["messages"] = [{"role": "user", "content": "What is the weather, [NAME]?"}]

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url)
        asyncio.run(guardrail.async_pre_call_hook(_USER, None, data, "acompletion"))

    assert calls[0]["params"]["body"]["messages"] == data["messages"]


def test_request_is_replaced_in_place_and_refused_as_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = {
        "model": "a-model",
        "messages": [_QUESTION, {"role": "user", "content": "A call was rejected."}],
        "temperature": 0,
    }
    replace = {"action": "replace", "body": replacement}
    _fits(_METHODS["judge_request"]["result"], replace)
    _fits(_METHODS["judge_request"]["result"], _REFUSAL)
    data = _data(_CHAT)
    kept = {name: data[name] for name in data if name not in _CHAT.body}

    async def run(guardrail: Any) -> Any:
        try:
            sent = await guardrail.async_pre_call_hook(_USER, None, data, "acompletion")
            with pytest.raises(HTTPException) as refused:
                await guardrail.async_pre_call_hook(
                    _USER, None, _data(_CHAT), "acompletion"
                )
            return sent, refused
        finally:
            await guardrail.aclose()

    with _serving(_scripted([], _result(replace), _result(_REFUSAL))) as url:
        sent, refused = asyncio.run(run(_make(monkeypatch, url)))

    assert sent is data
    # a field that was changed, one that was added and one that was left out
    assert {name: data[name] for name in replacement} == replacement
    assert "max_tokens" not in data
    # LiteLLM's own state is as it was
    assert {name: data[name] for name in kept} == kept
    assert (refused.value.status_code, refused.value.detail) == (403, "Not allowed.")


def _changed(on: _On, streamed: bool) -> tuple[bytes, Callable[[Any], None]]:
    """A reply body to replace the route's with, and a check that what the guardrail hands LiteLLM is that reply in the form LiteLLM had it."""
    if on is _MESSAGES and streamed:
        body = b"event: ping\ndata: {}\n\n"
        return body, lambda reply: _same(reply, [body])
    if on is _MESSAGES:
        whole = json.loads(message(weather("Oslo")))
        return json.dumps(whole).encode(), lambda reply: _same(
            reply, whole | {"_hidden_params": _notes()}
        )
    if streamed:
        # the first and the last event of the reply
        events = [_written(chunk) for chunk in on.stream()]
        ends = [events[0], events[-1]]
        kinds: list[Any] = [type(chunk) for chunk in on.stream()]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in ends)

        def check(reply: Any) -> None:
            assert [_written(chunk) for chunk in reply] == ends
            assert [type(chunk) for chunk in reply] == [kinds[0], kinds[-1]]

        return (body + "data: [DONE]\n\n").encode(), check
    whole = json.loads(on.whole().model_dump_json()) | {"id": "another-id"}

    def check_whole(reply: Any) -> None:
        assert type(reply) is type(on.whole())
        assert json.loads(reply.model_dump_json()) == whole
        # LiteLLM's own notes on the call go with the replacement
        assert reply._hidden_params == _notes()

    return json.dumps(whole).encode(), check_whole


def _same(reply: Any, expected: Any) -> None:
    assert reply == expected


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_is_replaced_in_the_form_litellm_had_it(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    body, check = _changed(on, streamed)
    replace = {"action": "replace", "body": base64.b64encode(body).decode()}
    _fits(_METHODS["judge_reply"]["result"], replace)

    with _serving(_scripted([], _result({"action": "pass"}), _result(replace))) as url:
        _, reply = _exchange(_make(monkeypatch, url), on, _data(on), streamed)

    check(reply)


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_is_refused_as_asked(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    answers = (_result({"action": "pass"}), _result(_REFUSAL))

    with _serving(_scripted([], *answers)) as url:
        status, message, field, let_go = _refused(
            _make(monkeypatch, url), on, _data(on), streamed
        )

    assert (status, message) == (
        200 if on is _RESPONSES and streamed else 403,
        "Not allowed.",
    )
    # a client that isn't Inspect is sent no field
    assert field is None
    assert let_go == []


_FIELD: dict[str, Any] = {"version": 1, "outcome": "reject", "steps": []}


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_a_refusal_carries_the_sentinel_field(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    answers = (_result({"action": "pass"}), _result(_REFUSAL | {"sentinel": _FIELD}))

    with _serving(_scripted([], *answers)) as url:
        _, message, field, _ = _refused(
            _make(monkeypatch, url), on, _data(on), streamed
        )

    assert (message, field) == ("Not allowed.", _FIELD)


@pytest.mark.parametrize("on", [_MESSAGES, _CHAT, _RESPONSES])
def test_a_refused_request_carries_the_sentinel_field(
    monkeypatch: pytest.MonkeyPatch, on: _On
) -> None:
    with _serving(_scripted([], _result(_REFUSAL | {"sentinel": _FIELD}))) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), on, _data(on), False)

    extra = getattr(refused.value, "provider_specific_fields", None)
    assert extra == {"sentinel": _FIELD}


def test_a_responses_stream_is_failed_as_the_response_it_began() -> None:
    from inspect_sentinel_litellm import _guardrail

    begun = _RESPONSES.stream()
    refused = HTTPException(403, "Not allowed.")
    setattr(refused, "provider_specific_fields", {"sentinel": _FIELD})  # noqa: B010

    event = _written(_guardrail._response_failed(begun, refused))  # pyright: ignore[reportPrivateUsage]

    created = _written(begun[0])["response"]
    assert event["response"]["id"] == created["id"]
    assert event["sequence_number"] == _written(begun[-1])["sequence_number"] + 1
    assert event["response"]["error"] == {
        "code": "permission_error",
        "message": "Not allowed.",
    }
    # the error the OpenAI SDK raises on, the same one, with the field in it
    assert event["error"] == {
        "code": "permission_error",
        "message": "Not allowed.",
        "type": "permission_error",
        "param": None,
        "provider_specific_fields": {"sentinel": _FIELD},
    }
    assert "sentinel" not in event


def test_a_replaced_chat_stream_sends_the_chunk_with_the_field_as_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [_written(chunk) for chunk in _CHAT.stream()]
    events[-1] = events[-1] | {"sentinel": _FIELD}
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    answers = (
        _result({"action": "pass"}),
        _result({"action": "replace", "body": base64.b64encode(body).decode()}),
    )

    with _serving(_scripted([], *answers)) as url:
        _, let_go = _exchange(_make(monkeypatch, url), _CHAT, _data(_CHAT), True)

    # LiteLLM writes its own chunks without a field they don't have, so only
    # the chunk with the field is in a class that keeps every field
    *chunks, last = let_go
    assert all(isinstance(chunk, _litellm.ModelResponseStream) for chunk in chunks)
    assert not isinstance(last, _litellm.ModelResponseStream)
    assert _written(last) == events[-1]


_EVENT_STREAM = "text/event-stream"
# where OpenAI's SDK sends a streamed request, under the base URL it is given
_SDK_PATHS = ("/chat/completions", "/responses")


def _as_litellm_writes(on: _On, chunks: list[Any], data: dict[str, Any]) -> bytes:
    """What LiteLLM's proxy writes for the caller from the chunks a guardrail lets go, with the LiteLLM version installed."""
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    asked: dict[str, Any] = {}
    taken = inspect.signature(proxy.async_data_generator).parameters
    if on is _RESPONSES and "responses_stream_errors" in taken:
        # as the Responses route asks it to
        asked["responses_stream_errors"] = True

    async def write() -> bytes:
        written = proxy.async_data_generator(_chunks(chunks), _USER, data, **asked)
        return b"".join(
            [
                part if isinstance(part, bytes) else part.encode()
                async for part in written
            ]
        )

    return asyncio.run(write())


def _read_by_the_sdk(
    on: _On, written: bytes, status: int = 200, content_type: str = _EVENT_STREAM
) -> list[Any]:
    """The events OpenAI's SDK reads from what LiteLLM answered with."""

    async def answer(_: HttpRequest) -> HttpResponse:
        return HttpResponse(written, status_code=status, media_type=content_type)

    async def read(url: str) -> list[Any]:
        client = openai.AsyncOpenAI(api_key="sk-caller", base_url=url)
        if on is _CHAT:
            stream: Any = await client.chat.completions.create(
                model="a-model",
                messages=[{"role": "user", "content": "What is the weather?"}],
                stream=True,
            )
        else:
            stream = await client.responses.create(
                model="a-model", input="What is the weather?", stream=True
            )
        events: list[Any] = [event async for event in stream]
        await client.close()
        return events

    routes = [Route(path, answer, methods=["POST"]) for path in _SDK_PATHS]
    with _serving(Starlette(routes=routes)) as url:
        return asyncio.run(read(url))


def test_a_reply_litellm_converts_for_the_messages_route_is_read_whole_by_anthropics_sdk() -> (
    None
):
    # LiteLLM writes the Anthropic events itself when it converts another
    # provider's reply for the Messages route. Anthropic's SDK dispatches an
    # event only by its `event:` label, and the sidecar reads a stream as the
    # SDK does, so an event LiteLLM wrote without one would reach neither.
    litellm: Any = importlib.import_module("litellm")
    anthropic: Any = importlib.import_module("anthropic")

    async def upstream(request: HttpRequest) -> HttpResponse:
        name = (
            "openai-chat.sse"
            if request.url.path.endswith("/chat/completions")
            else "openai-responses.sse"
        )
        return HttpResponse((DATA / name).read_bytes(), media_type=_EVENT_STREAM)

    async def convert(url: str) -> bytes:
        stream = await litellm.anthropic.messages.acreate(
            # the model the replies were recorded from, which LiteLLM streams
            model="openai/gpt-5-mini",
            api_key="sk-upstream",
            api_base=f"{url}/v1",
            max_tokens=100,
            messages=[_QUESTION],
            tools=[
                {
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
            stream=True,
        )
        return b"".join(
            [
                part if isinstance(part, bytes) else part.encode()
                async for part in stream
            ]
        )

    routes = [Route(f"/v1{path}", upstream, methods=["POST"]) for path in _SDK_PATHS]
    with _serving(Starlette(routes=routes)) as url:
        written = asyncio.run(convert(url))

    async def answer(_: HttpRequest) -> HttpResponse:
        return HttpResponse(written, media_type=_EVENT_STREAM)

    async def read(url: str) -> tuple[list[Any], Any]:
        client = anthropic.AsyncAnthropic(api_key="sk-caller", base_url=url)
        asked: dict[str, Any] = {
            "model": "a-model",
            "max_tokens": 100,
            "messages": [_QUESTION],
        }
        events = [
            event async for event in await client.messages.create(**asked, stream=True)
        ]
        async with client.messages.stream(**asked) as stream:
            message = await stream.get_final_message()
        await client.close()
        return events, message

    with _serving(
        Starlette(routes=[Route("/v1/messages", answer, methods=["POST"])])
    ) as url:
        events, message = asyncio.run(read(url))

    written_events = [
        event for event in stream_events(written) if event["type"] != "ping"
    ]
    assert [event.type for event in events] == [
        event["type"] for event in written_events
    ]
    assert [block.input for block in message.content if block.type == "tool_use"] == [
        {"location": "Paris, France"},
        {"location": "Tokyo, Japan"},
        {"location": "Lima, Peru"},
    ]


def _as_litellm_answers(
    on: _On, chunks: list[Any], data: dict[str, Any]
) -> tuple[int, str, bytes]:
    """The status, content type and body LiteLLM's route answers the caller with, from the chunks a guardrail lets go of a stream, with the LiteLLM version installed."""
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    processing: Any = importlib.import_module("litellm.proxy.common_request_processing")

    async def answer() -> tuple[int, str, bytes]:
        if on is _MESSAGES:
            written = processing.ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                _chunks(chunks), _USER, data, proxy.proxy_logging_obj
            )
        else:
            written = proxy.async_data_generator(_chunks(chunks), _USER, data)
        response = await processing.create_response(written, _EVENT_STREAM, {})
        body = getattr(response, "body", b"")
        if hasattr(response, "body_iterator"):
            body = b"".join(
                [
                    part if isinstance(part, bytes) else part.encode()
                    async for part in response.body_iterator
                ]
            )
        return response.status_code, response.headers["content-type"], body

    return asyncio.run(answer())


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        pytest.param(_result(_REFUSAL | {"sentinel": _FIELD}), 403, id="refused"),
        pytest.param(
            _result({"action": "replace", "body": "not base64!"}),
            503,
            id="no usable answer",
        ),
    ],
)
@pytest.mark.parametrize("on", [_MESSAGES, _CHAT], ids=["messages", "chat"])
def test_a_refused_stream_is_answered_as_an_error_with_the_field_by_litellm(
    monkeypatch: pytest.MonkeyPatch, on: _On, answer: Any, status: int
) -> None:
    data = _data(on)

    with _serving(_scripted([], _result({"action": "pass"}), answer)) as url:
        _, let_go = _exchange(_make(monkeypatch, url), on, data, True)
    answered, content_type, body = _as_litellm_answers(on, let_go, data)

    # an error of its own, as JSON, which every provider SDK reads as one
    assert (answered, content_type) == (status, "application/json")
    error = json.loads(body)["error"]
    if status == 403:
        assert error["message"] == "Not allowed."
        assert error["provider_specific_fields"] == {"sentinel": _FIELD}
    else:
        assert error["message"] == _UNANSWERED[1]


def test_a_refused_chat_stream_gives_the_sdk_the_field_as_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = (_result({"action": "pass"}), _result(_REFUSAL | {"sentinel": _FIELD}))
    data = _data(_CHAT)

    with _serving(_scripted([], *answers)) as url:
        _, let_go = _exchange(_make(monkeypatch, url), _CHAT, data, True)
    answered, content_type, body = _as_litellm_answers(_CHAT, let_go, data)
    with pytest.raises(openai.PermissionDeniedError) as refused:
        _read_by_the_sdk(_CHAT, body, answered, content_type)

    body: Any = refused.value.body
    assert isinstance(body, dict)
    error = cast("dict[str, Any]", body)
    assert error["provider_specific_fields"] == {"sentinel": _FIELD}


def test_a_chat_stream_with_the_field_is_read_by_the_sdk_from_what_litellm_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [_written(chunk) for chunk in _CHAT.stream()]
    events[-1] = events[-1] | {"sentinel": _FIELD}
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    answers = (
        _result({"action": "pass"}),
        _result({"action": "replace", "body": base64.b64encode(body).decode()}),
    )
    data = _data(_CHAT)

    with _serving(_scripted([], *answers)) as url:
        _, let_go = _exchange(_make(monkeypatch, url), _CHAT, data, True)
    chunks = _read_by_the_sdk(_CHAT, _as_litellm_writes(_CHAT, let_go, data))

    assert len(chunks) == len(events)
    extra: dict[str, Any] = chunks[-1].model_extra or {}
    assert extra.get("sentinel") == _FIELD
    # LiteLLM names the model the caller asked for on every chunk
    assert {chunk.model for chunk in chunks} == {"a-model"}


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "invalid_request_error"),
        (403, "permission_error"),
        (503, "server_error"),
    ],
)
def test_a_refused_responses_stream_is_read_by_the_sdk_from_what_litellm_writes(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    refusal = _REFUSAL | {"status": status, "sentinel": _FIELD}
    answers = (_result({"action": "pass"}), _result(refusal))
    data = _data(_RESPONSES)

    with _serving(_scripted([], *answers)) as url:
        _, let_go = _exchange(_make(monkeypatch, url), _RESPONSES, data, True)

    # the SDK raises on the failed response, rather than handing it over as
    # one more event a client could take for the reply's end
    with pytest.raises(openai.APIError) as raised:
        _read_by_the_sdk(_RESPONSES, _as_litellm_writes(_RESPONSES, let_go, data))

    error: Any = raised.value.body
    # only a server error is one a client tries again
    assert (raised.value.code, error["message"]) == (code, "Not allowed.")
    assert error["provider_specific_fields"]["sentinel"] == _FIELD


_NOT_BASE64 = {"action": "replace", "body": "not base64!"}
_NOT_A_REPLY = {"action": "replace", "body": base64.b64encode(b"[]").decode()}


@pytest.mark.parametrize(
    ("phase", "answer", "status"),
    [
        pytest.param("request", _result({"action": "allow"}), 200, id="no such action"),
        pytest.param("request", ["pass"], 200, id="not an object"),
        pytest.param("request", {"jsonrpc": "2.0", "id": 1}, 200, id="no result"),
        pytest.param(
            "request",
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32003, "message": "No."}},
            200,
            id="request not judged",
        ),
        pytest.param("request", _result({"action": "pass"}), 500, id="a server error"),
        pytest.param(
            "request",
            _result({"action": "replace", "body": "text"}),
            200,
            id="a replaced request not JSON",
        ),
        pytest.param(
            "request",
            _result({"action": "replace", "body": {"litellm_call_id": "another"}}),
            200,
            id="a replaced request that sets LiteLLM's own state",
        ),
        pytest.param(
            "reply",
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32005, "message": "No."}},
            200,
            id="reply not judged",
        ),
        pytest.param(
            "reply", _result(_NOT_BASE64), 200, id="a replaced reply not base64"
        ),
        pytest.param(
            "reply",
            _result(_NOT_A_REPLY),
            200,
            id="a replaced reply LiteLLM can't take",
        ),
    ],
)
@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_no_usable_answer_refuses_or_lets_through_as_litellm_is_set(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    answer: Any,
    status: int,
    fallback: str,
    streamed: bool,
) -> None:
    # a replaced stream on the Messages route is bytes, which are taken as they are
    on = _CHAT if answer == _result(_NOT_A_REPLY) else _MESSAGES
    calls: list[Any] = []
    passed = _result({"action": "pass"})
    answers = [answer] if phase == "request" else [passed, answer]
    data = _data(on)
    before = dict(data)

    with _serving(_scripted(calls, *answers, status=status)) as url:
        guardrail = _make(monkeypatch, url, unreachable_fallback=fallback)
        if fallback == "fail_closed" and streamed and phase == "reply":
            # the stream is answered with the error in place of the reply
            _, reply = _exchange(guardrail, on, data, streamed)
            [event] = reply
            error = _error_event(event)
            assert (error["code"], error["message"]) == _UNANSWERED
            return
        if fallback == "fail_closed":
            with pytest.raises(HTTPException) as refused:
                _exchange(guardrail, on, data, streamed)
            assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
            return
        sent, reply = _exchange(guardrail, on, data, streamed)

    # it goes on as it was
    assert sent is data
    assert {name: data[name] for name in before} == before
    if streamed and on is _MESSAGES:
        assert reply == on.stream()
    elif streamed:
        assert [_written(chunk) for chunk in reply] == [
            _written(chunk) for chunk in on.stream()
        ]
    else:
        assert reply is None
    # the sidecar is not asked about the reply to a request it gave no answer on
    assert len(calls) == (1 if phase == "request" else 2)


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_sidecar_that_cannot_be_reached_refuses_or_lets_through(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    # nothing listens here
    guardrail = _make(monkeypatch, "http://127.0.0.1:9", unreachable_fallback=fallback)
    data = _data(_CHAT)

    if fallback == "fail_closed":
        with pytest.raises(HTTPException) as refused:
            _exchange(guardrail, _CHAT, data, False)
        assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
    else:
        assert _exchange(guardrail, _CHAT, data, False) == (data, None)


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_refusal_that_cannot_be_read_still_refuses(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    unreadable = _result({"action": "refuse", "status": 200, "message": "No."})

    with _serving(_scripted([], unreadable)) as url:
        guardrail = _make(monkeypatch, url, unreachable_fallback=fallback)
        with pytest.raises(HTTPException) as refused:
            _exchange(guardrail, _CHAT, _data(_CHAT), False)

    assert refused.value.status_code == 503
    assert "refusal could not be read" in str(refused.value.detail)


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_reply_too_large_to_hold_is_not_put_to_the_sidecar(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    calls: list[Any] = []
    let_go: list[Any] = []
    chunks = _MESSAGES.stream()
    settings = {"unreachable_fallback": fallback, "max_reply_bytes": len(chunks[0])}

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url, **settings)
        _exchange(guardrail, _MESSAGES, _data(_MESSAGES), True, let_go)

    # only the request was put, and the reply went out whole or not at all
    assert len(calls) == 1
    if fallback == "fail_closed":
        [event] = let_go
        error = _error_event(event)
        assert (error["code"], error["message"]) == _UNANSWERED
    else:
        assert let_go == chunks


def test_request_on_a_route_the_guardrail_does_not_read_goes_on_unasked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    embedding = _On(
        "aembedding",
        "openai",
        "",
        {"model": "a-model", "input": "hello"},
        _chat_whole,
        _chat_stream,
    )
    data = _data(embedding)

    with _serving(_scripted(calls)) as url:
        guardrail = _make(monkeypatch, url)
        whole = _exchange(guardrail, embedding, data, False)
        streamed = _exchange(guardrail, embedding, data, True)

    assert calls == []
    assert whole == (data, None)
    assert [_written(chunk) for chunk in streamed[1]] == [
        _written(chunk) for chunk in _chat_stream()
    ]


def test_sidecar_that_takes_longer_than_the_time_set_gives_no_usable_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow(http: HttpRequest) -> JSONResponse:
        await asyncio.sleep(1)
        return JSONResponse(_result({"action": "pass"}))

    app = Starlette(routes=[Route(PATH, slow, methods=["POST"])])
    with _serving(app) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url, timeout=0.05), _CHAT, _data(_CHAT), False)
        unhurried = _make(monkeypatch, url)

    assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
    # no limit unless one is set: a sentinel may ask a person
    assert unhurried._timeout.total is None


@pytest.mark.parametrize(
    ("settings", "says"),
    [
        pytest.param({"api_base": None}, "api_base", id="no address"),
        pytest.param({"unreachable_fallback": "maybe"}, "fail_closed", id="fallback"),
        pytest.param({"event_hook": "pre_call"}, "post_call", id="shown no replies"),
        pytest.param({"event_hook": ["post_call"]}, "pre_call", id="shown no requests"),
        pytest.param({"sidecar_headers": ["a b"]}, "'a b'", id="not a header name"),
        pytest.param(
            {"sidecar_headers": "x-irid, Authorization"},
            "credential",
            id="a credential",
        ),
        pytest.param(
            {"sidecar_headers": ["x-*"]}, "credential", id="a prefix over a credential"
        ),
    ],
)
def test_setting_that_makes_no_sense_stops_the_guardrail_being_made(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        _make(monkeypatch, "http://sidecar", **settings)


def test_litellm_finds_the_stream_method_on_the_class_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LiteLLM looks in the class's own namespace, so an inherited one is not called
    guardrail = _load(monkeypatch).LiteLLMGuardrail
    assert "async_post_call_streaming_iterator_hook" in vars(guardrail)


def test_aclose_ends_the_connection_and_can_be_asked_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serving(_scripted([], _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url)

        async def run() -> bool:
            await guardrail.async_pre_call_hook(
                _USER, None, _data(_CHAT), "acompletion"
            )
            session = guardrail._session
            await guardrail.aclose()
            await guardrail.aclose()
            return bool(session.closed)

        assert asyncio.run(run())


def test_guardrail_imports_nothing_of_this_package() -> None:
    # it runs in LiteLLM's process, where this package is not installed
    files = list((_PACKAGE / "inspect_sentinel_litellm").glob("*.py"))
    assert files
    for file in files:
        assert not re.search(
            r"^\s*(from|import) inspect_sentinel(\.|\s)", file.read_text(), re.M
        )


# A reply as LiteLLM 1.105 wrote it for a Claude model that was asked through
# the Responses API and thought first. Its `reasoning` item is the one OpenAI's
# types reject: it has no `summary`, and its parts are typed `output_text`.
_THINKING: dict[str, Any] = json.loads(
    (DATA / "litellm-claude-responses.json").read_text()
)
_THOUGHT: dict[str, Any] = _THINKING["output"][0]


def _thinking_whole() -> Any:
    return _openai.ResponsesAPIResponse(**_THINKING)


def _thinking_stream() -> list[Any]:
    # the event that ends a stream holds the whole reply, which is what is read
    return [_response_event({"type": "response.completed", "response": _THINKING})]


_CLAUDE_RESPONSES = _On(
    "aresponses",
    "openai",
    "openai-responses",
    {"model": "a-model", "input": [_QUESTION]},
    _thinking_whole,
    _thinking_stream,
)
_WHOLE_OR_STREAMED = pytest.mark.parametrize(
    "streamed", [False, True], ids=["whole", "streamed"]
)


def test_the_reply_the_tests_use_is_one_openais_types_reject() -> None:
    # the sidecar reads a reply as leniently as OpenAI's SDK does, so it reads
    # this one as it is; OpenAI's types, held strictly, do not
    with pytest.raises(ValidationError):
        Response.model_validate(_THINKING)


@_WHOLE_OR_STREAMED
def test_claude_thinking_asked_for_through_responses_is_judged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, streamed: bool
) -> None:
    with _sidecar("Oslo", tmp_path) as url:
        _, reply = _exchange(
            _make(monkeypatch, url),
            _CLAUDE_RESPONSES,
            _data(_CLAUDE_RESPONSES),
            streamed,
        )

    # the caller is sent the reply as LiteLLM wrote it
    if streamed:
        assert [_written(chunk) for chunk in reply] == [
            _written(chunk) for chunk in _thinking_stream()
        ]
    else:
        assert reply is None


@_WHOLE_OR_STREAMED
def test_sidecar_is_shown_claude_thinking_as_litellm_wrote_it(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    # the sidecar reads the item as it is, as the test above shows end to end
    calls: list[Any] = []
    passed = _result({"action": "pass"})

    with _serving(_scripted(calls, passed, passed)) as url:
        _exchange(
            _make(monkeypatch, url),
            _CLAUDE_RESPONSES,
            _data(_CLAUDE_RESPONSES),
            streamed,
        )

    shown = calls[1]["params"]
    body = base64.b64decode(shown["body"])
    written = stream_events(body)[-1]["response"] if streamed else json.loads(body)
    thought = written["output"][0]
    # unrepaired: no summary, and its part typed as LiteLLM typed it
    assert not thought.get("summary")
    assert thought["content"] == _THOUGHT["content"]


def _echoes_with_a_note(shown: list[Any]) -> Starlette:
    """A stand-in for the sidecar that keeps the request it is shown, and replaces the request with the same one and a note added to its input."""

    async def call(http: HttpRequest) -> JSONResponse:
        body = (await http.json())["params"]["body"]
        shown.append(body)
        note = {"role": "user", "content": "A call was rejected."}
        return JSONResponse(
            _result(
                {"action": "replace", "body": body | {"input": [*body["input"], note]}}
            )
        )

    return Starlette(routes=[Route(PATH, call, methods=["POST"])])


def test_request_that_sends_litellms_thinking_back_is_read_and_goes_on_as_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a client that hands an item back as it got it. One that OpenAI's types
    # take, as inspect_ai sends, is beside it.
    valid: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [],
        "encrypted_content": "kept",
    }
    sent_input: list[Any] = [_QUESTION, _THOUGHT, valid, *_THINKING["output"][1:]]
    on = _On(
        "aresponses",
        "openai",
        "openai-responses",
        {"model": "a-model", "input": sent_input},
        _thinking_whole,
        _thinking_stream,
    )
    data = _data(on)
    shown: list[Any] = []

    with _serving(_echoes_with_a_note(shown)) as url:
        guardrail = _make(monkeypatch, url)

        async def run() -> None:
            try:
                await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            finally:
                await guardrail.aclose()

        asyncio.run(run())

    # OpenAI's own types take the item the sidecar was shown, as they don't the
    # one the client sent
    ResponseReasoningItem.model_validate(shown[0]["input"][1])
    with pytest.raises(ValidationError):
        ResponseReasoningItem.model_validate(_THOUGHT)
    # an item OpenAI's types take is shown as it is
    assert shown[0]["input"][2] == valid
    # LiteLLM is sent each item as the client sent it, and what the sidecar added
    assert data["input"][:-1] == sent_input
    assert data["input"][-1] == {"role": "user", "content": "A call was rejected."}


_INSPECT = {"X-Inspect-Context": '{"version": 1}', "x-irid": "request-1"}


@pytest.mark.parametrize("call_type", ["acompletion", "aembedding"])
@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_inspect_headers_never_reach_the_provider(
    monkeypatch: pytest.MonkeyPatch, call_type: str, fallback: str
) -> None:
    # LiteLLM sends a caller's `x-` headers on to the provider, in `headers`,
    # when its operator says to. Whether or not the sidecar answers, and
    # whether or not the route is read, Inspect's own are not sent on.
    data = _data(_CHAT, _INSPECT)
    data["headers"] = {**_INSPECT, "x-other": "kept"}
    data["extra_headers"] = {"x-irid": "request-1", "x-other": "kept"}
    # nothing listens here
    guardrail = _make(monkeypatch, "http://127.0.0.1:9", unreachable_fallback=fallback)

    async def run() -> None:
        try:
            await guardrail.async_pre_call_hook(_USER, None, data, call_type)
        except HTTPException:
            pass
        finally:
            await guardrail.aclose()

    asyncio.run(run())

    assert data["headers"] == {"x-other": "kept"}
    assert data["extra_headers"] == {"x-other": "kept"}


def test_guardrail_made_outside_litellms_proxy_adds_no_route() -> None:
    # a guardrail made by a test, or by anything but LiteLLM's proxy, loads
    # none of the proxy's server
    made = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from inspect_sentinel_litellm import LiteLLMGuardrail\n"
            "LiteLLMGuardrail(api_base='http://sidecar', guardrail_name='sentinel',"
            " event_hook=['pre_call', 'post_call'])\n"
            "print('litellm.proxy.proxy_server' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert made.stdout.strip() == "False"


def _sidecar_for_evals(seen: list[dict[str, Any]]) -> Starlette:
    """A stand-in for the sidecar's endpoint for evals that says what it was sent."""

    async def answer(http: HttpRequest) -> JSONResponse:
        seen.append(
            {
                "method": http.method,
                "path": http.url.path,
                "query": http.url.query,
                "headers": dict(http.headers),
                "body": (await http.body()).decode(),
            }
        )
        return JSONResponse({"answered": http.url.path}, status_code=201)

    return Starlette(
        routes=[Route("/sentinel/{rest:path}", answer, methods=["GET", "POST"])]
    )


def _without_eval_route(proxy: Any) -> None:
    # a guardrail made earlier in this process, while LiteLLM's proxy was
    # loaded, added a route that passes calls to another sidecar
    proxy.app.router.routes[:] = [
        route
        for route in proxy.app.router.routes
        if getattr(route, "path", None) != "/sentinel/{rest:path}"
    ]


def test_eval_routes_reach_the_sidecar_through_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    _without_eval_route(proxy)
    routes: Any = importlib.import_module("litellm.proxy._types").LiteLLMRoutes
    seen: list[dict[str, Any]] = []

    async def call(url: str) -> tuple[Any, Any]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://litellm"
        ) as client:
            probe = await client.get(
                "/sentinel/info?x=1", headers={"authorization": "Bearer sk-caller"}
            )
            asked = await client.post(
                "/sentinel/v1",
                content=b'{"jsonrpc": "2.0"}',
                headers={"authorization": "Bearer sk-caller"},
            )
            return probe, asked

    with _serving(_sidecar_for_evals(seen)) as url:
        # LiteLLM's proxy makes its guardrails once its app exists
        guardrail = _make(monkeypatch, url)
        guardrail.serve_eval_routes()
        probe, asked = asyncio.run(call(url))
        asyncio.run(guardrail.aclose())

    # ahead of LiteLLM's own routes, once however often it is asked
    paths = [getattr(route, "path", None) for route in proxy.app.router.routes]
    assert paths[0] == "/sentinel/{rest:path}"
    assert paths.count("/sentinel/{rest:path}") == 1
    assert {"/sentinel/info", "/sentinel/v1"} <= set(routes.openai_routes.value)
    assert (probe.status_code, probe.json()) == (201, {"answered": "/sentinel/info"})
    assert (asked.status_code, asked.json()) == (201, {"answered": "/sentinel/v1"})
    assert [(s["method"], s["path"], s["query"]) for s in seen] == [
        ("GET", "/sentinel/info", "x=1"),
        ("POST", "/sentinel/v1", ""),
    ]
    assert seen[1]["body"] == '{"jsonrpc": "2.0"}'
    # the caller's key was LiteLLM's to check, and the sidecar never sees it
    assert not any("authorization" in s["headers"] for s in seen)


def test_eval_route_answers_502_when_the_sidecar_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    # nothing listens here
    guardrail = _make(monkeypatch, "http://127.0.0.1:9")
    _without_eval_route(proxy)
    guardrail.serve_eval_routes()

    async def call() -> Any:
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy.app), base_url="http://litellm"
            ) as client:
                return await client.get("/sentinel/info")
        finally:
            await guardrail.aclose()

    answered = asyncio.run(call())

    assert answered.status_code == 502
    assert answered.json() == {"error": "The sentinel sidecar could not be reached."}


@pytest.mark.parametrize(
    "raw_path",
    [
        b"/sentinel/%2e%2e/proxy/v1",
        b"/sentinel/../proxy/v1",
        b"/sentinel/v1/%2E%2E/%2e%2e/proxy/v1",
        b"/sentinel/./v1",
        b"/sentinel//v1",
    ],
)
def test_eval_route_passes_on_nothing_outside_sentinel(
    monkeypatch: pytest.MonkeyPatch, raw_path: bytes
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    seen: list[dict[str, Any]] = []

    async def asgi_call(app: Any) -> tuple[int, bytes]:
        # straight to the app, so no client tidies the path on the way
        from urllib.parse import unquote

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": unquote(raw_path.decode()),
            "raw_path": raw_path,
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"litellm"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("litellm", 80),
        }
        sent: list[dict[str, Any]] = []
        received = False

        async def receive() -> dict[str, Any]:
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        body = b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        )
        return status, body

    with _serving(_sidecar_for_evals(seen)) as url:
        guardrail = _make(monkeypatch, url)
        proxy.app.router.routes[:] = [
            route
            for route in proxy.app.router.routes
            if getattr(route, "path", None) != "/sentinel/{rest:path}"
        ]
        guardrail.serve_eval_routes()
        try:
            status, body = asyncio.run(asgi_call(proxy.app))
        finally:
            asyncio.run(guardrail.aclose())

    assert (status, json.loads(body)) == (404, {"error": "Not found."})
    assert seen == []


def test_credentials_among_a_requests_fields_never_reach_the_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a caller may send LiteLLM a provider's key, or where to call, in the body
    sent_along = {
        "api_key": "sk-a-providers-key",
        "api_base": "https://elsewhere",
        "client_secret": "a-secret",
    }
    on = _On(
        "acompletion",
        "openai",
        "openai-chat",
        {**_CHAT.body, **sent_along},
        _chat_whole,
        _chat_stream,
        notes="metadata",
    )
    data = _data(on)
    calls: list[Any] = []

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url)

        async def run() -> None:
            try:
                await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            finally:
                await guardrail.aclose()

        asyncio.run(run())

    assert calls[0]["params"]["body"] == _CHAT.body
    # LiteLLM still has them, to call the provider with
    assert {name: data[name] for name in sent_along} == sent_along


def test_a_replaced_stream_may_hold_any_line_separator_json_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # U+2028 and U+2029 end a line for str.splitlines, and JSON allows them
    # unescaped inside a string
    events = _recorded_events("openai-chat.sse")
    first = json.loads(json.dumps(events[0]))
    first["choices"][0]["delta"]["content"] = "Rain and wind"
    body = "".join(
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in [first, *events[1:]]
    ).encode()
    assert " ".encode() in body
    replaced = _result({"action": "replace", "body": base64.b64encode(body).decode()})

    with _serving(_scripted([], _result({"action": "pass"}), replaced)) as url:
        _, reply = _exchange(_make(monkeypatch, url), _CHAT, _data(_CHAT), True)

    assert _written(reply[0])["choices"][0]["delta"]["content"] == "Rain and wind"
    assert len(reply) == len(events)


def test_text_reaches_the_caller_before_the_reply_has_ended_when_released_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = anthropic_recorded("parallel-tool-calls").split(b"\n\n")
    first_call = next(i for i, event in enumerate(events) if b'"tool_use"' in event)
    text = [event + b"\n\n" for event in events[:first_call]]
    rest = [event + b"\n\n" for event in events[first_call:] if event]
    seen_text = asyncio.Event()

    async def provider() -> AsyncIterator[Any]:
        for chunk in text:
            yield chunk
        # the calls come only once the caller has the text, which a reply
        # held whole would never let it have
        await asyncio.wait_for(seen_text.wait(), 10)
        for chunk in rest:
            yield chunk

    async def run(url: str) -> bytes:
        guardrail = _make(monkeypatch, url)
        data = _data(_MESSAGES)
        try:
            await guardrail.async_pre_call_hook(_USER, None, data, "anthropic_messages")
            out = b""
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                _USER, provider(), data
            ):
                out += chunk
                if b"text_delta" in out:
                    seen_text.set()
            return out
        finally:
            await guardrail.aclose()

    with _sidecar("Cairo", tmp_path, release="early") as url:
        out = asyncio.run(run(url))

    # every call let through, so the caller has what the provider sent
    assert out == b"".join(text + rest)


def test_a_turn_ended_after_a_call_went_out_raises_in_the_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the recorded Chat reply is calls alone: Paris goes out before Tokyo ends
    # the run
    data = _data(_CHAT)

    with _sidecar("Tokyo", tmp_path, release="early", action="terminate") as url:
        _, let_go = _exchange(_make(monkeypatch, url), _CHAT, data, True)

    with pytest.raises(openai.APIError) as raised:
        _read_by_the_sdk(_CHAT, _as_litellm_writes(_CHAT, let_go, data))

    assert "ended this run" in str(raised.value)


def _stand_in(answer: Callable[[str, dict[str, Any]], Any]) -> Starlette:
    """A stand-in for the sidecar that answers each call with what `answer` gives for its method and parameters: a result, or a JSON-RPC error as `{"error": ...}`."""

    async def call(http: HttpRequest) -> JSONResponse:
        asked = await http.json()
        given: Any = answer(asked["method"], asked.get("params", {}))
        if isinstance(given, Awaitable):
            given = await cast(Awaitable[Any], given)
        if isinstance(given, dict) and "error" in given:
            return JSONResponse({"jsonrpc": "2.0", "id": 1, **given})
        return JSONResponse(_result(given))

    return Starlette(routes=[Route(PATH, call, methods=["POST"])])


async def _never_ready(after: int) -> dict[str, Any]:
    # a release that never comes, as from a stream whose input never ended
    await asyncio.sleep(0.05)
    return {"position": after, "outputs": []}


def _within(seconds: float, exchange: Callable[[], tuple[Any, Any]]) -> tuple[Any, Any]:
    # an exchange that hangs would leave the test waiting, so it fails fast
    done: list[tuple[Any, Any]] = []
    raised: list[BaseException] = []

    def run() -> None:
        try:
            done.append(exchange())
        except BaseException as ex:
            raised.append(ex)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "the exchange didn't end"
    if raised:
        raise raised[0]
    return done[0]


def test_a_reply_whose_chunks_the_sidecar_wont_take_is_refused_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        if method == "reply_chunk":
            return {"error": {"code": -32603, "message": "No."}}
        if method == "next_release":
            return _never_ready(params["after"])
        return {"taken": True}

    with _serving(_stand_in(answer)) as url:
        _, reply = _within(
            10, lambda: _exchange(_make(monkeypatch, url), _CHAT, _data(_CHAT), True)
        )

    [event] = reply
    assert (_error_event(event)["code"], _error_event(event)["message"]) == _UNANSWERED


def test_a_reply_whose_stream_cant_begin_is_let_through_when_set_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        return {"error": {"code": -32603, "message": "No."}}

    with _serving(_stand_in(answer)) as url:
        guardrail = _make(monkeypatch, url, unreachable_fallback="fail_open")
        _, reply = _within(10, lambda: _exchange(guardrail, _CHAT, _data(_CHAT), True))

    assert [_written(chunk) for chunk in reply] == [
        _written(chunk) for chunk in _CHAT.stream()
    ]


def test_a_reply_too_large_to_judge_is_let_through_whole_when_set_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        if method == "next_release":
            return _never_ready(params["after"])
        return {"taken": True}

    with _serving(_stand_in(answer)) as url:
        guardrail = _make(
            monkeypatch, url, unreachable_fallback="fail_open", max_reply_bytes=100
        )
        _, reply = _within(10, lambda: _exchange(guardrail, _CHAT, _data(_CHAT), True))

    # every chunk, those after the limit too
    assert [_written(chunk) for chunk in reply] == [
        _written(chunk) for chunk in _CHAT.stream()
    ]


def test_a_reply_stream_s_end_is_acknowledged(monkeypatch: pytest.MonkeyPatch) -> None:
    passed = _result({"action": "pass"})
    stream_calls: list[Any] = []

    with _serving(_scripted([], passed, passed, streamed=stream_calls)) as url:
        _exchange(_make(monkeypatch, url), _CHAT, _data(_CHAT), True)

    # the outputs are the reply and its end, both of which the sidecar is told
    # the guardrail has, so that it lets go of the stream
    last = stream_calls[-1]
    assert (last["method"], last["params"]["after"]) == ("next_release", 2)


def test_the_session_to_the_sidecar_is_made_one_way_whoever_makes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    seen: list[dict[str, Any]] = []
    _without_eval_route(proxy)

    async def call(guardrail: Any) -> int:
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy.app), base_url="http://litellm"
            ) as client:
                await client.get(
                    "/sentinel/info", headers={"authorization": "Bearer sk-caller"}
                )
            # an eval's probe made it, and a reply stream shares it
            return guardrail._session.connector.limit
        finally:
            await guardrail.aclose()

    with _serving(_sidecar_for_evals(seen)) as url:
        guardrail = _make(monkeypatch, url)
        guardrail.serve_eval_routes()
        limit = asyncio.run(call(guardrail))

    assert seen and limit == 0


def test_nothing_is_left_running_when_the_caller_goes_mid_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polled = threading.Event()

    async def slow(after: int) -> dict[str, Any]:
        polled.set()
        await asyncio.sleep(5)
        return {"position": after, "outputs": []}

    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        if method == "next_release":
            return slow(params["after"])
        return {"taken": True}

    async def run(url: str) -> list[asyncio.Task[Any]]:
        guardrail = _make(monkeypatch, url)
        data = _data(_CHAT)
        await guardrail.async_pre_call_hook(_USER, None, data, _CHAT.call_type)
        released = guardrail.async_post_call_streaming_iterator_hook(
            _USER, _chunks(_CHAT.stream()), data
        )

        async def read() -> None:
            async for _ in released:
                pass

        reading = asyncio.ensure_future(read())
        while not polled.is_set():
            await asyncio.sleep(0.01)
        # the caller goes while the guardrail waits for a release
        reading.cancel()
        await asyncio.wait({reading})
        await released.aclose()
        await guardrail.aclose()
        for _ in range(10):
            await asyncio.sleep(0)
        return [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]

    with _serving(_stand_in(answer)) as url:
        left = asyncio.run(run(url))

    assert left == []


def test_a_reply_let_through_is_read_no_faster_than_the_caller_takes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled = 0
    chunk = _CHAT.stream()[1]

    async def provider() -> AsyncIterator[Any]:
        nonlocal pulled
        for _ in range(50):
            pulled += 1
            yield chunk

    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        if method == "next_release":
            return _never_ready(params["after"])
        return {"taken": True}

    async def run(url: str) -> tuple[int, int]:
        guardrail = _make(
            monkeypatch, url, unreachable_fallback="fail_open", max_reply_bytes=500
        )
        data = _data(_CHAT)
        try:
            await guardrail.async_pre_call_hook(_USER, None, data, _CHAT.call_type)
            released = guardrail.async_post_call_streaming_iterator_hook(
                _USER, provider(), data
            )
            await released.__anext__()
            # the caller is slow: what is read meanwhile stays bounded
            await asyncio.sleep(0.5)
            read_while_slow = pulled
            given = 1 + len([each async for each in released])
            return read_while_slow, given
        finally:
            await guardrail.aclose()

    with _serving(_stand_in(answer)) as url:
        read_while_slow, given = asyncio.run(run(url))

    assert read_while_slow < 20
    assert given == 50


def test_a_reply_let_through_after_its_stream_cant_begin_is_read_to_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # more than may wait to be handed over, which nothing hands over once the
    # stream couldn't begin
    monkeypatch.setattr(
        importlib.import_module("inspect_sentinel_litellm._guardrail"),
        "_PENDING_BYTES",
        200,
    )
    chunk = _CHAT.stream()[1]

    async def provider() -> AsyncIterator[Any]:
        for _ in range(50):
            yield chunk

    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        return {"error": {"code": -32603, "message": "No."}}

    async def run(url: str) -> int:
        guardrail = _make(monkeypatch, url, unreachable_fallback="fail_open")
        data = _data(_CHAT)
        try:
            await guardrail.async_pre_call_hook(_USER, None, data, _CHAT.call_type)
            released = guardrail.async_post_call_streaming_iterator_hook(
                _USER, provider(), data
            )
            return len([each async for each in released])
        finally:
            await guardrail.aclose()

    with _serving(_stand_in(answer)) as url:
        given, _ = _within(10, lambda: (asyncio.run(run(url)), None))

    assert given == 50


@pytest.mark.parametrize(
    "on", [_MESSAGES, _CHAT, _RESPONSES], ids=["messages", "chat", "responses"]
)
def test_a_keepalive_reaches_the_caller_where_litellm_passes_it_on(
    monkeypatch: pytest.MonkeyPatch, on: _On
) -> None:
    passed = _result({"action": "pass"})
    stream_calls: list[Any] = []
    with _serving(
        _scripted(
            [],
            passed,
            _result({"action": "pass", "kept_alive": True}),
            streamed=stream_calls,
        )
    ) as url:
        _, let_go = _exchange(_make(monkeypatch, url), on, _data(on), True)
    with _serving(_scripted([], passed, passed)) as url:
        _, plain = _exchange(_make(monkeypatch, url), on, _data(on), True)

    [begun] = [call for call in stream_calls if call["method"] == "begin_reply"]
    carried = on is _MESSAGES or _PASSES_BYTES_ON
    assert begun["params"]["keepalive"] is carried
    written = _as_litellm_answers(on, let_go, _data(on))[2]
    assert (_KEEPALIVE in written) is carried
    assert (
        written.replace(_KEEPALIVE, b"") == _as_litellm_answers(on, plain, _data(on))[2]
    )


def test_a_reply_kept_alive_is_still_let_through_when_set_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        if method == "next_release":
            if params["after"] == 0:
                kept_alive = base64.b64encode(_KEEPALIVE).decode()
                return {
                    "position": 1,
                    "outputs": [{"kind": "release", "data": kept_alive}],
                }
            return {"error": {"code": -32603, "message": "No."}}
        return {"taken": True}

    with _serving(_stand_in(answer)) as url:
        guardrail = _make(monkeypatch, url, unreachable_fallback="fail_open")
        _, reply = _within(
            10, lambda: _exchange(guardrail, _MESSAGES, _data(_MESSAGES), True)
        )

    # a keepalive holds nothing, so the reply can go on unjudged after it
    assert b"".join(reply) == _KEEPALIVE + b"".join(_MESSAGES.stream())


def test_a_reply_kept_alive_is_refused_in_the_stream_when_the_sidecar_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(method: str, params: dict[str, Any]) -> Any:
        if method == "judge_request":
            return {"action": "pass"}
        if method == "next_release":
            if params["after"] == 0:
                kept_alive = base64.b64encode(_KEEPALIVE).decode()
                return {
                    "position": 1,
                    "outputs": [{"kind": "release", "data": kept_alive}],
                }
            return {"error": {"code": -32603, "message": "No."}}
        return {"taken": True}

    with _serving(_stand_in(answer)) as url:
        _, reply = _within(
            10,
            lambda: _exchange(
                _make(monkeypatch, url), _MESSAGES, _data(_MESSAGES), True
            ),
        )
    status, _, written = _as_litellm_answers(_MESSAGES, reply, _data(_MESSAGES))

    # the keepalive sent the status, so only an error event refuses the reply
    async def sent(_: HttpRequest) -> HttpResponse:
        return HttpResponse(written, status_code=status, media_type=_EVENT_STREAM)

    async def read(url: str) -> None:
        client = anthropic.AsyncAnthropic(api_key="sk-caller", base_url=url)
        try:
            asked: dict[str, Any] = {
                "model": "a-model",
                "max_tokens": 100,
                "messages": [_QUESTION],
            }
            async for _ in await client.messages.create(**asked, stream=True):
                pass
        finally:
            await client.close()

    routes = [Route("/v1/messages", sent, methods=["POST"])]
    with _serving(Starlette(routes=routes)) as url:
        with pytest.raises(anthropic.APIStatusError) as raised:
            asyncio.run(read(url))

    assert status == 200
    assert "gave no usable answer" in str(raised.value)


@pytest.mark.parametrize("on", [_CHAT, _RESPONSES], ids=["chat", "responses"])
def test_a_streamed_reply_is_not_judged_again_once_whole(
    monkeypatch: pytest.MonkeyPatch, on: _On
) -> None:
    # LiteLLM 1.83 shows a guardrail a streamed reply again once it has gone
    # out, put together whole, which the stream's own judging has covered
    passed = _result({"action": "pass"})
    calls: list[Any] = []
    data = _data(on) | {"stream": True}

    async def run(url: str) -> Any:
        guardrail = _make(monkeypatch, url)
        try:
            await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            async for _ in guardrail.async_post_call_streaming_iterator_hook(
                _USER, _chunks(on.stream()), data
            ):
                pass
            return await guardrail.async_post_call_success_hook(
                data, _USER, on.whole()
            )
        finally:
            await guardrail.aclose()

    with _serving(_scripted(calls, passed, passed, passed)) as url:
        answered = asyncio.run(run(url))

    assert answered is None
    assert [call["method"] for call in calls] == ["judge_request", "judge_reply"]
