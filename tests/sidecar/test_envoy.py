"""The sidecar behind a real Envoy, through the Envoy shim.

Each test sends model requests to Envoy, with a provider's SDK or by hand, and Envoy sends them through the shim to a stand-in provider (`tests/sidecar/_envoy.py`). They need Docker, and are skipped without it.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler
from inspect_sentinel.sidecar.envoy import envoy_yaml
from tests.sidecar._envoy import IMAGE, Scripted, Stack, docker_missing, envoy_stack
from tests.sidecar._traffic import (
    JSON,
    STREAM,
    anthropic_recorded,
    every_call_rule,
    inspect_headers,
    message,
    openai_recorded,
    weather,
    weather_rule,
)

_MISSING = docker_missing()

pytestmark = pytest.mark.skipif(
    _MISSING is not None, reason=f"Envoy runs in Docker, and {_MISSING}"
)


@protocol
def rejects(city: str) -> Protocol:
    """Reject each weather lookup of `city`, however the model wrote it, as "Tokyo" or "Tokyo, Japan"."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if _city(step.call.arguments.get("location")) != city:
            return Decision.proceed()
        return Decision.reject(f"lookup of {city}", message=f"{city} is not allowed.")

    return decide


def _city(location: object) -> str:
    return str(location).split(",")[0]


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    with envoy_stack() as started:
        yield started


_APIS = ("anthropic", "chat", "responses")


def _recorded(api: str, stream: bool) -> tuple[str, bytes]:
    # a reply that calls get_weather for Paris, Tokyo and Lima at once
    if api == "anthropic":
        if stream:
            return STREAM, anthropic_recorded("parallel-tool-calls")
        return JSON, message(
            weather("Paris"), weather("Tokyo", "toolu_2"), weather("Lima", "toolu_3")
        )
    return openai_recorded(f"{api} stream" if stream else api)


def _split(body: bytes, size: int = 37) -> list[bytes]:
    # a stream in parts that split its events anywhere
    return [body[start : start + size] for start in range(0, len(body), size)]


def _answer_with(stack: Stack, api: str, stream: bool) -> bytes:
    content_type, body = _recorded(api, stream)
    stack.provider.script = lambda _: Scripted(
        body=_split(body) if stream else [body],
        content_type=content_type,
        chunked=stream,
    )
    return body


# what a client waits for an answer, in seconds, so that a shim that never
# answers fails a test at once, where the SDKs would wait ten minutes
_TIMEOUT = 10.0

_TOOL = {
    "name": "get_weather",
    "description": "The weather in a city.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


def _ask(stack: Stack, api: str, stream: bool) -> list[str]:
    """Ask for the weather with the provider's SDK, and return the cities the reply called get_weather for."""
    question = "What is the weather?"
    # the SDKs' overloads are not what these tests are about
    client: Any
    if api == "anthropic":
        client = anthropic.Anthropic(
            base_url=stack.url, api_key="key", max_retries=0, timeout=_TIMEOUT
        )
        asked = {
            "model": "claude-test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": question}],
            "tools": [
                {
                    "name": _TOOL["name"],
                    "description": _TOOL["description"],
                    "input_schema": _TOOL["parameters"],
                }
            ],
        }
        if stream:
            with client.messages.stream(**asked) as streamed:
                reply = streamed.get_final_message()
        else:
            reply = client.messages.create(**asked)
        return [
            _city(block.input["location"])
            for block in reply.content
            if block.type == "tool_use"
        ]
    client = openai.OpenAI(
        base_url=f"{stack.url}/v1", api_key="key", max_retries=0, timeout=_TIMEOUT
    )
    if api == "chat":
        asked = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": question}],
            "tools": [{"type": "function", "function": _TOOL}],
        }
        if stream:
            with client.chat.completions.stream(**asked) as streamed:
                completion = streamed.get_final_completion()
        else:
            completion = client.chat.completions.create(**asked)
        calls: list[Any] = completion.choices[0].message.tool_calls or []
        return [
            _city(json.loads(call.function.arguments)["location"]) for call in calls
        ]
    asked = {
        "model": "gpt-test",
        "input": question,
        "tools": [{"type": "function", **_TOOL}],
    }
    if stream:
        with client.responses.stream(**asked) as streamed:
            response = streamed.get_final_response()
    else:
        response = client.responses.create(**asked)
    return [
        _city(json.loads(item.arguments)["location"])
        for item in response.output
        if item.type == "function_call"
    ]


def _path(api: str) -> str:
    return {
        "anthropic": "/v1/messages",
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
    }[api]


def _body(api: str, stream: bool) -> dict[str, Any]:
    if api == "anthropic":
        asked: dict[str, Any] = {
            "model": "claude-test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is the weather?"}],
        }
    elif api == "chat":
        asked = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "What is the weather?"}],
        }
    else:
        asked = {"model": "gpt-test", "input": "What is the weather?"}
    return {**asked, "stream": stream}


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("api", _APIS)
def test_reply_that_passes_reaches_the_client_as_the_provider_sent_it(
    stack: Stack, api: str, stream: bool
) -> None:
    stack.use(Handler(weather_rule(city="Oslo")))
    sent = _answer_with(stack, api, stream)

    answered = httpx.post(f"{stack.url}{_path(api)}", json=_body(api, stream))

    assert answered.status_code == 200
    assert answered.content == sent


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("api", _APIS)
def test_rejected_call_is_taken_out_of_the_reply(
    stack: Stack, api: str, stream: bool
) -> None:
    stack.use(Handler(rejects("Tokyo")))
    _answer_with(stack, api, stream)

    assert _ask(stack, api, stream) == ["Paris", "Lima"]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("api", _APIS)
def test_turn_whose_every_call_is_rejected_is_refused_with_the_providers_error(
    stack: Stack, api: str, stream: bool
) -> None:
    stack.use(Handler(every_call_rule()))
    _answer_with(stack, api, stream)

    error = anthropic.BadRequestError if api == "anthropic" else openai.BadRequestError
    with pytest.raises(error) as refused:
        _ask(stack, api, stream)

    assert refused.value.status_code == 400
    # the model wrote "Paris" in one reply and "Paris, France" in another
    assert "Paris" in refused.value.message
    assert "is not allowed." in refused.value.message


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("api", _APIS)
def test_inspect_client_is_told_what_was_decided(
    stack: Stack, api: str, stream: bool
) -> None:
    stack.use(Handler(rejects("Tokyo")))
    _answer_with(stack, api, stream)

    answered = httpx.post(
        f"{stack.url}{_path(api)}", json=_body(api, stream), headers=inspect_headers()
    )

    assert answered.status_code == 200
    assert b'"sentinel"' in answered.content
    assert b'"reject"' in answered.content
    # Inspect's headers name the sample to the sidecar, and never reach the
    # provider
    assert not {"x-inspect-context", "x-irid"} & set(stack.provider.received[0].headers)


@pytest.mark.parametrize("api", _APIS)
def test_refusal_carries_the_sentinel_field_for_an_inspect_client(
    stack: Stack, api: str
) -> None:
    stack.use(Handler(every_call_rule()))
    _answer_with(stack, api, False)

    answered = httpx.post(
        f"{stack.url}{_path(api)}", json=_body(api, False), headers=inspect_headers()
    )

    assert answered.status_code == 400
    assert answered.headers["content-type"] == "application/json"
    refused = answered.json()
    assert refused["sentinel"]["outcome"] == "reject"
    if api == "anthropic":
        assert refused["type"] == "error"
        assert refused["error"]["type"] == "invalid_request_error"
    else:
        assert refused["error"]["type"] == "invalid_request_error"


def test_provider_is_asked_for_an_uncompressed_reply(stack: Stack) -> None:
    stack.use(Handler(weather_rule()))
    _answer_with(stack, "anthropic", False)

    httpx.post(
        f"{stack.url}/v1/messages",
        json=_body("anthropic", False),
        headers={"accept-encoding": "gzip, br"},
    )

    assert stack.provider.received[0].headers["accept-encoding"] == "identity"


def test_compressed_reply_is_refused(stack: Stack) -> None:
    stack.use(Handler(weather_rule()))
    _, body = _recorded("anthropic", False)
    stack.provider.script = lambda _: Scripted(
        body=[gzip.compress(body)], headers={"content-encoding": "gzip"}
    )

    answered = httpx.post(f"{stack.url}/v1/messages", json=_body("anthropic", False))

    assert answered.status_code == 502
    assert "gzip" in answered.json()["error"]["message"]


def test_request_the_sidecar_replaces_reaches_the_provider_with_its_own_length(
    stack: Stack,
) -> None:
    stack.use(Handler(rejects("Tokyo")))
    _answer_with(stack, "anthropic", False)
    opening = _body("anthropic", False)
    first = httpx.post(f"{stack.url}/v1/messages", json=opening).json()
    # the client runs the calls it was given, and asks again
    kept = [block for block in first["content"] if block["type"] == "tool_use"]
    then = {
        **opening,
        "messages": [
            *opening["messages"],
            {"role": "assistant", "content": first["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Sunny.",
                    }
                    for block in kept
                ],
            },
        ],
    }
    stack.provider.script = lambda _: Scripted(
        body=[message({"type": "text", "text": "Done."}, stop_reason="end_turn")]
    )

    answered = httpx.post(f"{stack.url}/v1/messages", json=then)

    assert answered.status_code == 200
    sent = stack.provider.received[-1]
    # the rejected call is put back, with its rejection as its result
    assert b"Tokyo is not allowed." in sent.body
    assert int(sent.headers["content-length"]) == len(sent.body)
    assert len(sent.body) > len(json.dumps(then))


def test_request_larger_than_envoys_default_buffer_is_judged(stack: Stack) -> None:
    stack.use(Handler(weather_rule()))
    _answer_with(stack, "anthropic", False)
    asked = _body("anthropic", False)
    asked["messages"][0]["content"] = "weather? " * (600 * 1024)  # over 5 MiB

    answered = httpx.post(f"{stack.url}/v1/messages", json=asked, timeout=60)

    assert answered.status_code == 200
    assert json.loads(stack.provider.received[0].body) == asked


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_reply_larger_than_grpcs_default_message_passes_whole(
    stack: Stack, stream: bool
) -> None:
    stack.use(Handler(weather_rule(city="Oslo")))
    long = "Sunny. " * (800 * 1024)  # over 5 MiB
    if stream:
        events = [
            {
                "type": "message_start",
                "message": json.loads(message(stop_reason="end_turn"))
                | {"content": []},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": long},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        body = b"".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in events
        )
    else:
        body = message({"type": "text", "text": long}, stop_reason="end_turn")
    stack.provider.script = lambda _: Scripted(
        body=_split(body, 64 * 1024),
        content_type=STREAM if stream else JSON,
        chunked=stream,
    )

    answered = httpx.post(
        f"{stack.url}/v1/messages", json=_body("anthropic", stream), timeout=60
    )

    assert answered.status_code == 200
    assert answered.content == body


def test_reply_with_trailers_is_judged_at_its_trailers(stack: Stack) -> None:
    stack.use(Handler(rejects("Tokyo")))
    content_type, body = _recorded("anthropic", True)
    stack.provider.script = lambda _: Scripted(
        body=_split(body),
        content_type=content_type,
        trailers={"x-provider-checksum": "abc"},
    )

    assert _ask(stack, "anthropic", True) == ["Paris", "Lima"]


def test_reply_sent_slowly_is_held_until_its_end(stack: Stack) -> None:
    stack.use(Handler(rejects("Tokyo")))
    content_type, body = _recorded("anthropic", True)
    stack.provider.script = lambda _: Scripted(
        body=_split(body, 512), content_type=content_type, chunked=True, pause=0.02
    )

    assert _ask(stack, "anthropic", True) == ["Paris", "Lima"]


@pytest.mark.parametrize("status", [429, 500, 529])
def test_providers_error_reaches_the_client_as_it_was(
    stack: Stack, status: int
) -> None:
    stack.use(Handler(weather_rule()))
    error = json.dumps(
        {"type": "error", "error": {"type": "overloaded_error", "message": "Busy."}}
    ).encode()
    stack.provider.script = lambda _: Scripted(body=[error], status=status)

    answered = httpx.post(f"{stack.url}/v1/messages", json=_body("anthropic", False))

    assert answered.status_code == status
    assert answered.content == error


def test_providers_error_without_a_body_reaches_the_client(stack: Stack) -> None:
    stack.use(Handler(weather_rule()))
    stack.provider.script = lambda _: Scripted(body=[], status=503)

    answered = httpx.post(f"{stack.url}/v1/messages", json=_body("anthropic", False))

    assert answered.status_code == 503


@protocol
def broken_rule() -> Protocol:
    """Fail at every call."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        raise RuntimeError("the rule broke")

    return decide


def test_reply_the_sentinel_fails_on_is_refused(stack: Stack) -> None:
    stack.use(Handler(broken_rule()))
    _answer_with(stack, "chat", False)

    answered = httpx.post(f"{stack.url}/v1/chat/completions", json=_body("chat", False))

    assert answered.status_code == 500
    assert answered.json()["error"]["message"] == (
        "The sentinel failed, so the reply was not judged."
    )


@protocol
def slow_rule(seconds: float = 5.0) -> Protocol:
    """Take `seconds` to let each request go."""

    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        import anyio

        await anyio.sleep(seconds)
        return Decision.proceed()

    return decide


def test_request_the_sentinel_takes_too_long_on_is_refused(stack: Stack) -> None:
    stack.use(Handler(slow_rule(), time_limit=0.2))
    _answer_with(stack, "anthropic", False)

    answered = httpx.post(f"{stack.url}/v1/messages", json=_body("anthropic", False))

    assert answered.status_code == 504
    assert stack.provider.received == []


def test_unreadable_request_is_refused_before_the_provider_is_called(
    stack: Stack,
) -> None:
    stack.use(Handler(weather_rule()))

    answered = httpx.post(
        f"{stack.url}/v1/messages",
        content=b"not json",
        headers={"content-type": "application/json"},
    )

    assert answered.status_code == 400
    assert answered.json()["error"]["message"] == (
        "The request couldn't be read, so it was not judged."
    )
    assert stack.provider.received == []


@protocol
def waits_for(released: threading.Event, judged: list[str]) -> Protocol:
    """Let each reply's calls go once `released` is set, noting each judged."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        import anyio

        while not released.is_set():
            await anyio.sleep(0.01)
        judged.append(str(step.call.arguments.get("location")))
        return Decision.proceed()

    return decide


def test_judgment_goes_on_when_the_client_disconnects(stack: Stack) -> None:
    released = threading.Event()
    judged: list[str] = []
    stack.use(Handler(waits_for(released, judged)))
    _answer_with(stack, "anthropic", False)

    with pytest.raises(httpx.ReadTimeout):
        httpx.post(
            f"{stack.url}/v1/messages", json=_body("anthropic", False), timeout=0.2
        )
    released.set()

    stack.run(_until(lambda: len(judged) == 3))
    assert [_city(city) for city in judged] == ["Paris", "Tokyo", "Lima"]


async def _until(done: Any, seconds: float = 10.0) -> None:
    import anyio

    with anyio.fail_after(seconds):
        while not done():
            await anyio.sleep(0.01)


def test_path_the_sentinel_doesnt_judge_is_not_found(stack: Stack) -> None:
    stack.use(Handler(weather_rule()))

    answered = httpx.post(f"{stack.url}/v1/embeddings", json={"input": "x"})

    assert answered.status_code == 404
    assert stack.provider.received == []


def test_token_counting_goes_to_the_provider_unjudged(stack: Stack) -> None:
    stack.use(Handler(broken_rule()))
    stack.provider.script = lambda _: Scripted(body=[b'{"input_tokens": 3}'])

    answered = httpx.post(
        f"{stack.url}/v1/messages/count_tokens",
        json={"model": "claude-test", "messages": []},
        headers=inspect_headers(),
    )

    assert answered.json() == {"input_tokens": 3}
    assert [received.path for received in stack.provider.received] == [
        "/v1/messages/count_tokens"
    ]
    assert "x-inspect-context" not in stack.provider.received[0].headers


def test_endpoint_for_evals_is_reached_through_envoy(stack: Stack) -> None:
    stack.use(Handler(weather_rule()))

    answered = httpx.get(f"{stack.url}/sentinel/info")

    assert answered.status_code == 200
    assert stack.provider.received == []


@protocol
def notes_headers(seen: list[dict[str, str]]) -> Protocol:
    """Note the headers each request carried, as the sentinel is shown them."""

    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        assert context.proxy is not None
        seen.append(dict(context.proxy.headers))
        return Decision.proceed()

    return decide


def test_sentinel_is_shown_inspects_headers_and_never_a_credential(
    stack: Stack,
) -> None:
    seen: list[dict[str, str]] = []
    stack.use(Handler(notes_headers(seen)))
    _answer_with(stack, "anthropic", False)

    httpx.post(
        f"{stack.url}/v1/messages",
        json=_body("anthropic", False),
        headers={
            **inspect_headers(),
            "x-api-key": "secret",
            "authorization": "Bearer secret",
        },
    )

    assert set(seen[0]) == {"x-inspect-context", "x-irid"}
    # the provider still gets the credentials
    assert stack.provider.received[0].headers["x-api-key"] == "secret"


def test_reply_larger_than_the_shim_holds_is_refused() -> None:
    with envoy_stack(max_body_bytes=64 * 1024) as small:
        small.use(Handler(weather_rule()))
        body = message(
            {"type": "text", "text": "Sunny. " * 20_000}, stop_reason="end_turn"
        )
        small.provider.script = lambda _: Scripted(body=_split(body, 8 * 1024))

        answered = httpx.post(
            f"{small.url}/v1/messages", json=_body("anthropic", False)
        )

    assert answered.status_code == 502
    assert "larger than" in answered.json()["error"]["message"]


def test_request_is_refused_when_the_shim_cant_be_reached() -> None:
    with envoy_stack(shim=False) as alone:
        alone.provider.script = lambda _: Scripted(
            body=[message(stop_reason="end_turn")]
        )

        answered = httpx.post(
            f"{alone.url}/v1/messages", json=_body("anthropic", False)
        )

    assert answered.status_code == 500
    assert alone.provider.received == []


def test_configuration_for_the_real_providers_is_one_envoy_takes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "envoy.yaml"
    config.write_text(envoy_yaml())
    tmp_path.chmod(0o755)
    config.chmod(0o644)

    checked = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            f"--volume={tmp_path}:/etc/sentinel:ro",
            IMAGE,
            "--mode",
            "validate",
            "--config-path",
            "/etc/sentinel/envoy.yaml",
        ],
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0, checked.stderr
    assert "warning" not in checked.stderr
