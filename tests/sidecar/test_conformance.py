"""The sidecar's work against Inspect's own process.

Each test runs a real Inspect eval twice against a stand-in provider. Once with the sentinel in the eval's own process, and once without it, with the provider's replies first put through a sidecar, as a model proxy would, and the eval's requests after them too. The request the model is sent after a turn with a call rejected or changed must be the same either way. The tests of what each step is given run the sidecar both behind a stand-in for a proxy hook and behind a real Envoy, which needs Docker.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    CompactionTrim,
)
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, tool
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import Response
from starlette.routing import Route

from inspect_sentinel import (
    AfterToolCall,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    Protocol,
    Step,
    protocol,
)
from inspect_sentinel.sidecar import Handler, Pass, Refuse, Replace, Reply, Request
from inspect_sentinel.sidecar._anthropic import settled
from tests.sidecar._envoy import (
    Received,
    Scripted,
    Stack,
    docker_missing,
    envoy_stack,
)
from tests.sidecar._traffic import JSON, STREAM

_ASKED = "What is the weather in Tokyo and Paris?"


@tool
def get_weather() -> Tool:
    async def execute(location: str) -> str:
        """Get the weather in a city.

        Args:
            location: The city.
        """
        return f"Sunny in {location}."

    return execute


@protocol
def tokyo_rule(action: str) -> Protocol:
    """Reject a lookup of Tokyo, or change it to Oslo."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") != "Tokyo":
            return Decision.proceed()
        if action == "reject":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        oslo = replace(step.call, arguments={"location": "Oslo"})
        return Decision(action="modify", modified=oslo)

    return decide


# what each tool the stand-in model calls is given
_ARGUMENTS = {"get_weather": "location", "submit": "answer"}


def _anthropic_stream(message: dict[str, Any]) -> bytes:
    # a reply as the Messages API streams it, which Inspect's provider asks for
    def event(data: dict[str, Any]) -> str:
        return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"

    start: dict[str, Any] = {**message, "content": [], "stop_reason": None}
    written = [event({"type": "message_start", "message": start})]
    for index, block in enumerate(message["content"]):
        if block["type"] == "tool_use":
            empty: dict[str, Any] = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        else:
            empty = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        written.append(
            event(
                {"type": "content_block_start", "index": index, "content_block": empty}
            )
        )
        written.append(
            event({"type": "content_block_delta", "index": index, "delta": delta})
        )
        written.append(event({"type": "content_block_stop", "index": index}))
    stop = {"stop_reason": message["stop_reason"], "stop_sequence": None}
    written.append(
        event({"type": "message_delta", "delta": stop, "usage": {"output_tokens": 1}})
    )
    written.append(event({"type": "message_stop"}))
    return "".join(written).encode()


def _anthropic(
    calls: list[tuple[str, str]] | None, used: int = 1, function: str = "get_weather"
) -> bytes:
    content: list[dict[str, Any]] = (
        [
            {
                "type": "tool_use",
                "id": id,
                "name": function,
                "input": {_ARGUMENTS[function]: value},
            }
            for id, value in calls
        ]
        if calls
        else [{"type": "text", "text": "Done."}]
    )
    return _anthropic_stream(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": used, "output_tokens": 1},
        }
    )


def _chat(
    calls: list[tuple[str, str]] | None, used: int = 1, function: str = "get_weather"
) -> bytes:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None if calls else "Done.",
    }
    if calls:
        message["tool_calls"] = [
            {
                "id": id,
                "type": "function",
                "function": {
                    "name": function,
                    "arguments": json.dumps({_ARGUMENTS[function]: value}),
                },
            }
            for id, value in calls
        ]
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": used,
                "completion_tokens": 1,
                "total_tokens": used + 1,
            },
        }
    ).encode()


def _responses(
    calls: list[tuple[str, str]] | None, used: int = 1, function: str = "get_weather"
) -> bytes:
    output: list[dict[str, Any]] = (
        [
            {
                "type": "function_call",
                "id": f"fc_{id}",
                "call_id": id,
                "name": function,
                "arguments": json.dumps({_ARGUMENTS[function]: value}),
                "status": "completed",
            }
            for id, value in calls
        ]
        if calls
        else [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
            }
        ]
    )
    return json.dumps(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-test",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": used,
                "output_tokens": 1,
                "total_tokens": used + 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    ).encode()


@dataclass(frozen=True)
class _Api:
    """One API Inspect's providers speak: how Inspect is pointed at it, and how the stand-in replies."""

    model: str
    model_args: dict[str, Any]
    provider: str
    content_type: str
    path: str
    reply: Callable[..., bytes]
    # what Inspect changes between requests while the conversation stays the
    # same: the Anthropic provider moves its cache marks to the latest messages
    settle: Callable[[dict[str, Any]], dict[str, Any]] = lambda body: body


def _settled(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "messages": [settled(entry) for entry in body["messages"]]}


_APIS = {
    "anthropic": _Api(
        "anthropic/claude-test",
        {},
        "anthropic",
        STREAM,
        "/v1/messages",
        _anthropic,
        _settled,
    ),
    "openai chat": _Api(
        "openai/gpt-test",
        {"responses_api": False},
        "openai",
        JSON,
        "/v1/chat/completions",
        _chat,
    ),
    "openai responses": _Api(
        "openai/gpt-test",
        {"responses_api": True},
        "openai",
        JSON,
        "/v1/responses",
        _responses,
    ),
}


async def _count(http: HttpRequest) -> Response:
    # Anthropic's token count, which Inspect asks for to decide when to
    # compact: a token for every eight bytes the request would send, so that
    # the tools and the opening fit under the threshold the test sets
    return Response(
        json.dumps({"input_tokens": len(await http.body()) // 8}),
        media_type="application/json",
    )


_COUNT = Route("/v1/messages/count_tokens", _count, methods=["POST"])


@contextmanager
def _provider(api: _Api, replies: list[bytes]) -> Generator[tuple[str, list[bytes]]]:
    """A stand-in provider that gives these replies in turn, and keeps each request it was sent."""
    asked: list[bytes] = []
    left = list(replies)

    async def answer(http: HttpRequest) -> Response:
        asked.append(await http.body())
        return Response(left.pop(0), media_type=api.content_type)

    app = Starlette(routes=[Route(api.path, answer, methods=["POST"]), _COUNT])
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", asked
    finally:
        server.should_exit = True
        serving.join()


def _run(
    api: _Api, replies: list[bytes], log_dir: Path, sentinel: Protocol | None = None
) -> list[bytes]:
    """Run an eval whose model looks up the weather, and give back what it sent the provider."""
    with _provider(api, replies) as (url, asked):
        log = eval(
            Task(
                dataset=[Sample(input=_ASKED)],
                solver=[use_tools(get_weather()), generate()],
                sentinel=sentinel,
            ),
            model=api.model,
            model_args=api.model_args,
            model_base_url=url if api.provider == "anthropic" else f"{url}/v1",
            display="none",
            log_dir=str(log_dir),
        )[0]
    assert log.status == "success", log.error
    return asked


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stand-in")
    monkeypatch.setenv("OPENAI_API_KEY", "stand-in")


@pytest.mark.parametrize("name", _APIS)
@pytest.mark.parametrize("action", ["reject", "modify"])
def test_the_model_is_sent_what_inspect_sends_in_its_own_process(
    name: str, action: str, tmp_path: Path
) -> None:
    api = _APIS[name]
    turn = api.reply([("call_1", "Tokyo"), ("call_2", "Paris")])
    done = api.reply(None)

    # the sentinel in the eval's own process
    in_process = _run(api, [turn, done], tmp_path, tokyo_rule(action))

    # the sentinel in a sidecar, beside an eval that knows nothing of it
    handler = Handler(tokyo_rule(action))

    def judged(id: str, body: bytes) -> Any:
        return asyncio.run(
            handler.request(
                Request(id=id, provider=api.provider, model="m", user=None, body=body)
            )
        )

    assert judged("1", in_process[0]) == Pass()
    edited = asyncio.run(
        handler.reply(
            Reply(
                id="1",
                provider=api.provider,
                model="m",
                user=None,
                status=200,
                content_type=api.content_type,
                body=turn,
            )
        )
    )
    assert isinstance(edited, Replace)
    beside = _run(api, [edited.body, done], tmp_path)
    amended = judged("2", beside[1])

    assert isinstance(amended, Replace)
    assert api.settle(json.loads(amended.body)) == api.settle(json.loads(in_process[1]))


@contextmanager
def _proxied(api: _Api, replies: list[bytes], handler: Handler) -> Generator[str]:
    """A stand-in provider behind a sidecar: each request and its reply are put to the handler, as a model proxy would."""
    left = list(replies)
    count = iter(range(1_000))

    async def answer(http: HttpRequest) -> Response:
        id = str(next(count))
        asked = await handler.request(
            Request(
                id=id,
                provider=api.provider,
                model="m",
                user=None,
                body=await http.body(),
            )
        )
        assert not isinstance(asked, Refuse), asked
        given = left.pop(0)
        replied = await handler.reply(
            Reply(
                id=id,
                provider=api.provider,
                model="m",
                user=None,
                status=200,
                content_type=api.content_type,
                body=given,
            )
        )
        if isinstance(replied, Refuse):
            return Response(replied.message, status_code=replied.status)
        body = replied.body if isinstance(replied, Replace) else given
        return Response(body, media_type=api.content_type)

    app = Starlette(routes=[Route(api.path, answer, methods=["POST"]), _COUNT])
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        serving.join()


@pytest.fixture(scope="module")
def _envoy() -> Iterator[Stack]:
    missing = docker_missing()
    if missing is not None:
        pytest.skip(f"Envoy runs in Docker, and {missing}")
    with envoy_stack() as stack:
        yield stack


@contextmanager
def _sidecar(
    behind: str,
    api: _Api,
    replies: list[bytes],
    handler: Handler,
    request: pytest.FixtureRequest,
) -> Generator[str]:
    """A stand-in provider behind a sidecar, with the sidecar behind a stand-in for a proxy hook or behind a real Envoy, and the URL a client reaches it at."""
    if behind == "proxy hook":
        with _proxied(api, replies, handler) as url:
            yield url
        return
    stack: Stack = request.getfixturevalue("_envoy")
    stack.use(handler)
    left = list(replies)

    def answer(received: Received) -> Scripted:
        if received.path == "/v1/messages/count_tokens":
            counted = {"input_tokens": len(received.body) // 8}
            return Scripted(body=[json.dumps(counted).encode()])
        return Scripted(
            body=[left.pop(0)],
            content_type=api.content_type,
            chunked=api.content_type == STREAM,
        )

    stack.provider.script = answer
    yield stack.url


_BEHIND = ["proxy hook", "envoy"]


def _told(message: ChatMessage) -> str:
    # What the model is told of a message. Inspect's own process holds a
    # rejected call's result as an error, which the wire carries as text, and
    # the Anthropic provider writes with "Error: " before it, so the sidecar
    # reads it back as that text.
    if isinstance(message, ChatMessageTool) and message.error is not None:
        return message.error.message
    return message.text.removeprefix("Error: ")


def _said(message: ChatMessage) -> tuple[object, ...]:
    # what a message says that the history must keep as Inspect does: its
    # role and what the model is told of it, each call it makes, and the call
    # a result answers
    calls = [
        (call.id, call.function, call.arguments)
        for call in (
            message.tool_calls if isinstance(message, ChatMessageAssistant) else None
        )
        or []
    ]
    answers = message.tool_call_id if isinstance(message, ChatMessageTool) else None
    return (message.role, _told(message), calls, answers)


# each step's history, as the sentinels below saw it, by the step
_HISTORIES: dict[str, list[tuple[str, list[tuple[object, ...]]]]] = {}


@protocol
def records_histories(seen: str, action: str = "proceed") -> Protocol:
    """Keep each step's history, as `_said` gives each message, and reject a lookup of Tokyo, or change it to Oslo, if asked."""

    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        call = step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else ""
        history = [_said(m) for m in step.history or []]
        _HISTORIES.setdefault(seen, []).append(
            (f"{type(step).__name__}{call}", history)
        )
        if (
            isinstance(step, BeforeToolCall)
            and step.call.arguments.get("location") == "Tokyo"
        ):
            if action == "reject":
                return Decision.reject("Tokyo", message="Tokyo is not allowed.")
            if action == "modify":
                oslo = replace(step.call, arguments={"location": "Oslo"})
                return Decision(action="modify", modified=oslo)
        return Decision.proceed()

    return decide


def _by_step(seen: str) -> dict[str, list[list[tuple[object, ...]]]]:
    # the order of the steps differs, as a sidecar judges a result only when
    # the request that holds it arrives, so they are compared by step
    found: dict[str, list[list[tuple[object, ...]]]] = {}
    for step, history in _HISTORIES.pop(seen, []):
        found.setdefault(step, []).append(history)
    return found


@pytest.mark.parametrize("behind", _BEHIND)
@pytest.mark.parametrize("name", _APIS)
@pytest.mark.parametrize("action", ["proceed", "reject", "modify"])
def test_each_step_is_given_the_history_inspect_gives_it_in_its_own_process(
    name: str, action: str, behind: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    api = _APIS[name]
    replies = [api.reply([("call_1", "Tokyo"), ("call_2", "Paris")]), api.reply(None)]

    _run(api, replies, tmp_path, records_histories("in process", action))
    handler = Handler(records_histories("sidecar", action))
    with _sidecar(behind, api, replies, handler, request) as url:
        log = eval(
            Task(
                dataset=[Sample(input=_ASKED)],
                solver=[use_tools(get_weather()), generate()],
            ),
            model=api.model,
            model_args=api.model_args,
            model_base_url=url if api.provider == "anthropic" else f"{url}/v1",
            display="none",
            log_dir=str(tmp_path),
        )[0]
    assert log.status == "success", log.error

    in_process, sidecar = _by_step("in process"), _by_step("sidecar")
    assert sidecar == in_process


@pytest.mark.parametrize("behind", _BEHIND)
@pytest.mark.parametrize("name", _APIS)
def test_the_history_keeps_what_inspects_own_compaction_drops(
    name: str, behind: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # A compaction that keeps the conversation's opening, as Inspect's own
    # do, leaves the sidecar the same conversation, and what it drops stays
    # in the history. The stand-in reports more of the context used with
    # each turn, so that the agent compacts before its last lookup.
    api = _APIS[name]
    cities = ["Tokyo", "Paris", "Lima", "Oslo"]
    replies = [
        api.reply([(f"call_{index}", city)], used=150 + 60 * index)
        for index, city in enumerate(cities)
    ] + [api.reply([("call_9", "done")], used=150, function="submit")]

    def task(sentinel: Protocol | None) -> Task:
        return Task(
            dataset=[Sample(input=_ASKED)],
            solver=react(
                tools=[get_weather()],
                compaction=CompactionTrim(threshold=300, memory=False, preserve=0.5),
            ),
            sentinel=sentinel,
        )

    def run(url: str, sentinel: Protocol | None) -> None:
        log = eval(
            task(sentinel),
            model=api.model,
            model_args=api.model_args,
            model_base_url=url if api.provider == "anthropic" else f"{url}/v1",
            display="none",
            log_dir=str(tmp_path),
        )[0]
        assert log.status == "success", log.error

    with _provider(api, replies) as (url, asked):
        run(url, records_histories("in process"))
    # the agent compacted: the request after its last lookup holds the first no more
    assert b"call_0" not in asked[-1]
    handler = Handler(records_histories("sidecar"))
    with _sidecar(behind, api, replies, handler, request) as url:
        run(url, None)

    in_process, sidecar = _by_step("in process"), _by_step("sidecar")
    # the result of the agent's last call, its submission, reaches no proxy,
    # as no request follows it
    del in_process["AfterToolCallcall_9"]
    assert sidecar == in_process
