"""The Envoy shim and Envoy's configuration, without Envoy: what Envoy sends is written by hand, for what a real Envoy can't be made to send on purpose."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from inspect_sentinel.sidecar import Handler
from inspect_sentinel.sidecar._serve import parse_arguments
from inspect_sentinel.sidecar.envoy import EnvoyShim, envoy_config, envoy_yaml
from inspect_sentinel.sidecar.envoy._proto.envoy.config.core.v3 import base_pb2 as core
from inspect_sentinel.sidecar.envoy._proto.envoy.service.ext_proc.v3 import (
    external_processor_pb2 as pb,
)
from tests.sidecar._traffic import message, weather, weather_rule


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


def _headers(headers: dict[str, str]) -> pb.HttpHeaders:
    return pb.HttpHeaders(
        headers=core.HeaderMap(
            headers=[
                core.HeaderValue(key=name, raw_value=value.encode())
                for name, value in headers.items()
            ]
        )
    )


async def _process(
    messages: Sequence[pb.ProcessingRequest],
    handler: Handler | None = None,
    metadata: Sequence[tuple[str, str]] = _ANTHROPIC,
) -> list[pb.ProcessingResponse]:
    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in messages:
            yield each

    shim = EnvoyShim(handler or Handler(weather_rule()), max_body_bytes=1024 * 1024)
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
async def test_compressed_request_is_refused() -> None:
    answers = await _process(
        [
            pb.ProcessingRequest(
                request_headers=_headers(
                    {"content-type": "application/json", "content-encoding": "gzip"}
                )
            )
        ]
    )

    assert answers[0].immediate_response.status.code == 415


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
