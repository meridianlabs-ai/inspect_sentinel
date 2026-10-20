"""The Envoy shim and Envoy's configuration, without Envoy: what Envoy sends is written by hand, for what a real Envoy can't be made to send on purpose."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest

from inspect_sentinel.sidecar import Handler, Pass, Replace, Reply, Request
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._serve import (
    parse_arguments,
)
from inspect_sentinel.sidecar.envoy import (
    EnvoyShim,
    envoy_config,
    envoy_server,
    envoy_yaml,
)
from inspect_sentinel.sidecar.envoy._proto.envoy.config.core.v3 import base_pb2 as core
from inspect_sentinel.sidecar.envoy._proto.envoy.service.ext_proc.v3 import (
    external_processor_pb2 as pb,
)
from tests.sidecar._traffic import (
    every_call_rule,
    inspect_headers,
    message,
    weather,
    weather_rule,
)


@pytest.fixture
def anyio_backend() -> str:
    # gRPC's server runs on asyncio, and so does the shim
    return "asyncio"


_ANTHROPIC = (
    ("sentinel-format", "anthropic-messages"),
    ("sentinel-provider", "anthropic"),
)


class _Context:
    def __init__(self, metadata: Sequence[tuple[str, str]]) -> None:
        self._metadata = tuple(metadata)

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata


def _headers(headers: dict[str, str], *, end_of_stream: bool = False) -> pb.HttpHeaders:
    return pb.HttpHeaders(
        headers=core.HeaderMap(
            headers=[
                core.HeaderValue(key=name, raw_value=value.encode())
                for name, value in headers.items()
            ]
        ),
        end_of_stream=end_of_stream,
    )


async def _process(
    messages: Sequence[pb.ProcessingRequest],
    handler: Any = None,
    metadata: Sequence[tuple[str, str]] = _ANTHROPIC,
    max_body_bytes: int = 1024 * 1024,
) -> list[pb.ProcessingResponse]:
    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in messages:
            yield each

    shim = EnvoyShim(handler or Handler(weather_rule()), max_body_bytes=max_body_bytes)
    return [
        answer async for answer in shim.Process(sent(), cast(Any, _Context(metadata)))
    ]


_REQUEST = [
    pb.ProcessingRequest(
        request_headers=_headers({"content-type": "application/json"})
    ),
    pb.ProcessingRequest(
        request_body=pb.HttpBody(
            body=b'{"model": "claude-test", "max_tokens": 1, "messages": [{"role": "user", "content": "Weather?"}]}',
            end_of_stream=True,
        )
    ),
]


@pytest.mark.anyio
async def test_empty_chunk_that_is_not_the_end_doesnt_end_the_reply() -> None:
    body = message(weather("Paris"), weather("Lima", "toolu_2"))
    middle = len(body) // 2
    answers = await _process(
        [
            *_REQUEST,
            pb.ProcessingRequest(
                response_headers=_headers(
                    {":status": "200", "content-type": "application/json"}
                )
            ),
            pb.ProcessingRequest(response_body=pb.HttpBody(body=body[:middle])),
            # Envoy 1.39 sends an empty chunk now and then mid-body
            pb.ProcessingRequest(response_body=pb.HttpBody(body=b"")),
            pb.ProcessingRequest(
                response_body=pb.HttpBody(body=body[middle:], end_of_stream=True)
            ),
        ]
    )

    released = [
        answer.response_body.response.body_mutation.streamed_response
        for answer in answers
        if answer.WhichOneof("response") == "response_body"
    ]
    assert [(each.body, each.end_of_stream) for each in released] == [(body, True)]


@pytest.mark.anyio
async def test_route_that_doesnt_say_the_api_is_refused() -> None:
    answers = await _process(_REQUEST, metadata=[("sentinel-provider", "anthropic")])

    assert len(answers) == 1
    refused = answers[0].immediate_response
    assert refused.status.code == 500
    assert b"sentinel-format" in refused.body


@pytest.mark.anyio
async def test_inspects_headers_are_taken_out_of_the_request() -> None:
    answers = await _process(_REQUEST)

    assert list(answers[0].request_headers.response.header_mutation.remove_headers) == [
        "x-inspect-context",
        "x-irid",
    ]


def test_provider_at_a_path_is_refused() -> None:
    with pytest.raises(ValueError, match="has a path"):
        envoy_config(anthropic="https://gateway.example.com/anthropic")


def test_message_timeout_longer_than_envoy_allows_is_refused() -> None:
    with pytest.raises(ValueError, match="at most 3600"):
        envoy_config(message_timeout=7200)


def test_provider_left_out_has_no_routes() -> None:
    config = envoy_config(openai=None)
    manager = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][
        0
    ]["typed_config"]
    routes = manager["route_config"]["virtual_hosts"][0]["routes"]
    paths = [route["match"].get("path") for route in routes]

    assert "/v1/messages" in paths
    assert "/v1/chat/completions" not in paths
    assert "/v1/responses" not in paths
    assert "openai" not in [
        cluster["name"] for cluster in config["static_resources"]["clusters"]
    ]


_EXAMPLE = Path(__file__).parents[2] / "examples" / "always_sunny_envoy"


def test_examples_envoy_configuration_is_what_the_generator_writes() -> None:
    written = (_EXAMPLE / "envoy.yaml").read_text()
    # the file's own lines say how it was written, above the generator's
    generated = written[written.index("# Generated by inspect_sentinel") :]

    assert generated == envoy_yaml(sidecar_host="sidecar")


def test_example_sidecar_serves_where_its_envoy_looks() -> None:
    arguments = parse_arguments(["--sentinel", str(_EXAMPLE / "sentinel.yaml")])
    clusters = {
        cluster["name"]: cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
            "endpoint"
        ]["address"]["socket_address"]["port_value"]
        for cluster in envoy_config(sidecar_host="sidecar")["static_resources"][
            "clusters"
        ]
    }

    assert clusters["sentinel_sidecar"] == arguments.port
    assert clusters["sentinel_shim"] == arguments.envoy_port


_BODY = b'{"model": "claude-test", "max_tokens": 1, "messages": [{"role": "user", "content": "Weather?"}]}'


def _from_inspect(**headers: str) -> pb.ProcessingRequest:
    return pb.ProcessingRequest(
        request_headers=_headers(
            {"content-type": "application/json", **inspect_headers(), **headers}
        )
    )


def _reply(*parts: bytes, **headers: str) -> list[pb.ProcessingRequest]:
    return [
        pb.ProcessingRequest(
            response_headers=_headers(
                {":status": "200", "content-type": "application/json", **headers}
            )
        ),
        *(
            pb.ProcessingRequest(
                response_body=pb.HttpBody(
                    body=part, end_of_stream=index == len(parts) - 1
                )
            )
            for index, part in enumerate(parts)
        ),
    ]


async def _not_judged(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.reason


@pytest.mark.anyio
async def test_compressed_request_is_refused_and_the_eval_told_it_wasnt_judged() -> (
    None
):
    handler = Handler(weather_rule())
    answers = await _process(
        [
            _from_inspect(**{"content-encoding": "gzip"}),
            pb.ProcessingRequest(
                request_body=pb.HttpBody(
                    body=b"\x1f\x8b compressed", end_of_stream=True
                )
            ),
        ],
        handler,
    )

    assert answers[-1].immediate_response.status.code == 400
    assert await _not_judged(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_body_bytes", "headers"),
    [(64, {}), (1024 * 1024, {"content-encoding": "gzip"})],
    ids=["larger than the shim holds", "compressed"],
)
async def test_reply_refused_unread_tells_the_eval_it_wasnt_judged(
    max_body_bytes: int, headers: dict[str, str]
) -> None:
    handler = Handler(weather_rule())
    body = message(weather("Paris"))
    answers = await _process(
        [
            _from_inspect(),
            pb.ProcessingRequest(
                request_body=pb.HttpBody(body=_BODY, end_of_stream=True)
            ),
            *_reply(body[:100], body[100:], **headers),
        ],
        handler,
        max_body_bytes=max_body_bytes,
    )

    assert answers[-1].immediate_response.status.code == 502
    # not still awaiting the reply the client was refused
    assert await _not_judged(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_refusal_outside_the_error_statuses_is_sent_as_a_500() -> None:
    answers = await _process(
        [*_REQUEST, *_reply(message(weather("Paris")))],
        Handler(every_call_rule(), reject_status=700),
    )

    refused = answers[-1].immediate_response
    assert refused.status.code == 500
    # the body is written for the status sent
    assert json.loads(refused.body)["error"]["type"] == "api_error"


class _ReplacesEveryRequest:
    async def request(self, request: Request) -> Replace:
        return Replace(b"{}")


@pytest.mark.anyio
async def test_request_without_a_body_that_the_sidecar_replaces_is_refused() -> None:
    answers = await _process(
        [
            pb.ProcessingRequest(
                request_headers=_headers(
                    {"content-type": "application/json"}, end_of_stream=True
                )
            )
        ],
        _ReplacesEveryRequest(),
    )

    assert [answer.WhichOneof("response") for answer in answers] == [
        "immediate_response"
    ]
    assert answers[0].immediate_response.status.code == 500


def test_eval_endpoint_left_out_is_not_routed() -> None:
    config = envoy_config(eval_endpoint=False)
    manager = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][
        0
    ]["typed_config"]
    routes = manager["route_config"]["virtual_hosts"][0]["routes"]

    assert "/sentinel/" not in [route["match"].get("prefix") for route in routes]
    assert "sentinel_sidecar" in [
        cluster["name"] for cluster in config["static_resources"]["clusters"]
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_body_bytes", "headers"),
    [(64, {}), (1024 * 1024, {"content-encoding": "gzip"})],
    ids=["larger than the shim holds", "compressed"],
)
async def test_providers_error_passes_through_unjudged_whatever_its_size_or_encoding(
    max_body_bytes: int, headers: dict[str, str]
) -> None:
    handler = Handler(weather_rule())
    error = (
        b'{"type": "error", "error": {"type": "api_error", "message": "'
        + b"x" * 200
        + b'"}}'
    )
    replied = _reply(error[:100], error[100:], **headers)
    replied[0] = pb.ProcessingRequest(
        response_headers=_headers(
            {":status": "500", "content-type": "application/json", **headers}
        )
    )
    answers = await _process(
        [
            _from_inspect(),
            pb.ProcessingRequest(
                request_body=pb.HttpBody(body=_BODY, end_of_stream=True)
            ),
            *replied,
        ],
        handler,
        max_body_bytes=max_body_bytes,
    )

    passed = [
        answer.response_body.response.body_mutation.streamed_response
        for answer in answers
        if answer.WhichOneof("response") == "response_body"
    ]
    assert b"".join(each.body for each in passed) == error
    assert [each.end_of_stream for each in passed] == [False, True]
    assert await _not_judged(handler) == ("not_judged", "provider_error")


class _ReplacesEveryReply:
    async def request(self, request: Request) -> Pass:
        return Pass()

    async def reply(self, reply: Reply) -> Replace:
        return Replace(b"{}")


@pytest.mark.anyio
async def test_reply_without_a_body_that_the_sidecar_replaces_is_refused() -> None:
    answers = await _process(
        [
            *_REQUEST,
            pb.ProcessingRequest(
                response_headers=_headers(
                    {":status": "200", "content-type": "application/json"},
                    end_of_stream=True,
                )
            ),
        ],
        _ReplacesEveryReply(),
    )

    assert answers[-1].WhichOneof("response") == "immediate_response"
    assert answers[-1].immediate_response.status.code == 500


@pytest.mark.anyio
async def test_shim_listens_on_an_ipv6_address_as_the_endpoints_do() -> None:
    # uvicorn takes `::1` bare, and gRPC only in brackets
    server = envoy_server(
        Handler(weather_rule()), host="::1", port=0, max_body_bytes=1024
    )
    await server.stop(None)


class _FailsOnceEnvoyStopsWaiting:
    def __init__(self) -> None:
        self.judging = asyncio.Event()
        self.release = asyncio.Event()
        self.failed = asyncio.Event()

    async def request(self, request: Request) -> Pass:
        return Pass()

    async def reply(self, reply: Reply) -> Pass:
        self.judging.set()
        await self.release.wait()
        self.failed.set()
        raise RuntimeError("the rule broke")


@pytest.mark.anyio
async def test_judgment_that_fails_after_envoy_stopped_waiting_is_logged_by_the_shim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = _FailsOnceEnvoyStopsWaiting()
    shim = EnvoyShim(cast(Any, handler), max_body_bytes=1024 * 1024)
    reported: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: reported.append(context))

    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in [*_REQUEST, *_reply(message(stop_reason="end_turn"))]:
            yield each
        await asyncio.Event().wait()

    async def envoy() -> None:
        async for _ in shim.Process(sent(), cast(Any, _Context(_ANTHROPIC))):
            pass

    waiting = asyncio.ensure_future(envoy())
    await handler.judging.wait()
    waiting.cancel()
    with suppress(asyncio.CancelledError):
        await waiting
    with caplog.at_level(logging.ERROR):
        handler.release.set()
        await handler.failed.wait()
        for _ in range(3):
            await asyncio.sleep(0)
        gc.collect()
    loop.set_exception_handler(None)

    # not left to asyncio, which reports an exception nothing retrieved
    assert reported == []
    # logged as a failure Envoy waited for is
    [logged] = caplog.records
    assert re.fullmatch(
        r"The reply '[0-9a-f]{32}' was not judged\.", logged.getMessage()
    )
    assert logged.exc_info is not None
