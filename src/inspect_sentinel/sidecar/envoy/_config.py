from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import yaml

from .._http import MAX_BODY_BYTES
from ._metadata import FORMAT_METADATA, INSPECT_HEADERS, PROVIDER_METADATA

ENVOY_VERSION = "1.39.4"
"""The Envoy release the configuration is written for and tested with. 1.39.1 is the earliest that works: it has the `ext_proc` fixes that keep a body whole when it is held."""

MESSAGE_TIMEOUT = 3600.0
"""How long Envoy waits for the shim to judge a request, in seconds, unless set otherwise: the longest Envoy allows."""

CA_FILE = "/etc/ssl/certs/ca-certificates.crt"
"""Where Envoy's image keeps the certificates it trusts."""

MAX_CONNECTIONS = 16384
"""The most connections from clients Envoy keeps open at once."""

# The routes that are judged: each is a path, the provider it goes to, and the
# API its body is written in.
_JUDGED: tuple[tuple[str, str, str], ...] = (
    ("/v1/messages", "anthropic", "anthropic-messages"),
    ("/v1/chat/completions", "openai", "openai-chat"),
    ("/v1/responses", "openai", "openai-responses"),
)

# Routes that go to a provider unjudged, because they don't call a model:
# Inspect counts tokens before compacting an Anthropic conversation.
_UNJUDGED: tuple[tuple[str, str], ...] = (("/v1/messages/count_tokens", "anthropic"),)

_EXT_PROC = "envoy.filters.http.ext_proc"
_SHIM_CLUSTER = "sentinel_shim"
_SIDECAR_CLUSTER = "sentinel_sidecar"


def envoy_config(
    *,
    port: int = 8080,
    sidecar_host: str = "127.0.0.1",
    sidecar_port: int = 8900,
    shim_port: int = 8901,
    anthropic: str | None = "https://api.anthropic.com",
    openai: str | None = "https://api.openai.com",
    max_body_bytes: int = MAX_BODY_BYTES,
    message_timeout: float = MESSAGE_TIMEOUT,
    ca_file: str = CA_FILE,
) -> dict[str, Any]:
    """Envoy's configuration for a model proxy with the sidecar behind it.

    Envoy listens on `port` and sends each model request on a route the shim judges through the `ext_proc` filter to the sidecar's Envoy shim, then on to the provider. The routes are Anthropic's `/v1/messages`, and OpenAI's `/v1/chat/completions` and `/v1/responses`, as each provider's SDK writes them, with a client's base URL of Envoy (with `/v1` for OpenAI's). Anthropic's `/v1/messages/count_tokens` goes to the provider unjudged, as it doesn't call a model, and `/sentinel/` goes to the sidecar's endpoint for evals. Every other path is answered with a 404.

    Each setting that is wrong for holding a reply by default is set:

    - The reply's body is sent to the shim in `FULL_DUPLEX_STREAMED` mode, the one mode in which the shim can hold a reply of any size and then refuse it with a real status. It needs the trailers sent too.
    - A request's body is sent whole, and Envoy waits `message_timeout` for the shim's answer, where it waits 200 ms by default.
    - A held reply is never timed out: neither the route nor an idle stream has a timeout.
    - Envoy holds a request of up to `max_body_bytes`, where it holds 1 MiB by default.
    - The shim is reached over HTTP/2, which gRPC needs, with room for many streams at once.
    - The provider is asked for an uncompressed reply, which the shim can read. A reply that comes compressed anyway is refused.
    - Inspect's own headers, which name the eval's sample, are taken out of every request to a provider.
    - A sidecar that can't be reached refuses the request, rather than letting it through unjudged.
    - A provider reached over TLS has its certificate checked, against `ca_file`, for the provider's host name. Envoy checks none by default.

    Nothing controls who can reach Envoy: put it where only the agents it serves can, or add Envoy's own authentication.

    Args:
        port: The port Envoy listens on, on every address.
        sidecar_host: Where Envoy reaches the sidecar.
        sidecar_port: The sidecar's port for its endpoints, its `--port`.
        shim_port: The sidecar's port for the Envoy shim, its `--envoy-port`.
        anthropic: Anthropic's API, or None to leave its routes out. An `http` URL is reached without TLS, as for a stand-in provider.
        openai: OpenAI's API, or None to leave its routes out.
        max_body_bytes: The largest request Envoy holds while the shim judges it, the sidecar's `--max-body-bytes`.
        message_timeout: How long Envoy waits for the shim to judge a request, in seconds, at most an hour. A request that takes longer is answered with a 504, and the sentinel still decides.
        ca_file: The certificates Envoy trusts a provider's to be signed by, in Envoy's file system. Envoy's image keeps the system's at the default.

    Raises:
        ValueError: If a provider's URL has a path, or no host, or `message_timeout` is more than an hour.
    """
    if not 0 < message_timeout <= MESSAGE_TIMEOUT:
        raise ValueError(
            f"Envoy waits at most {MESSAGE_TIMEOUT:g} seconds for a message, not {message_timeout:g}."
        )
    providers = {
        name: _Provider(name, url)
        for name, url in (("anthropic", anthropic), ("openai", openai))
        if url is not None
    }
    routes: list[dict[str, Any]] = []
    for path, provider, format in _JUDGED:
        if provider in providers:
            routes.append(
                _to_provider(
                    path,
                    providers[provider],
                    {
                        "overrides": {
                            "grpc_initial_metadata": [
                                {"key": FORMAT_METADATA, "value": format},
                                {"key": PROVIDER_METADATA, "value": provider},
                            ]
                        }
                    },
                )
            )
    for path, provider in _UNJUDGED:
        if provider in providers:
            # more specific than a judged route it starts with, so first
            routes.insert(
                0, _to_provider(path, providers[provider], {"disabled": True})
            )
    routes.append(
        {
            "match": {"prefix": "/sentinel/"},
            "route": {"cluster": _SIDECAR_CLUSTER, "timeout": "0s"},
            "typed_per_filter_config": _per_route({"disabled": True}),
        }
    )
    routes.append(
        {
            "match": {"prefix": "/"},
            "direct_response": {
                "status": 404,
                "body": {
                    "inline_string": "This proxy serves only the model routes the sentinel judges.\n"
                },
            },
            "typed_per_filter_config": _per_route({"disabled": True}),
        }
    )
    connection_manager = {
        "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
        "stat_prefix": "model_api",
        "codec_type": "AUTO",
        # a reply's trailers reach the client, as the provider sent them
        "http_protocol_options": {"enable_trailers": True},
        # a held reply sends the client nothing for as long as it is judged
        "stream_idle_timeout": "0s",
        "route_config": {
            "name": "model_api",
            "virtual_hosts": [
                {"name": "model_api", "domains": ["*"], "routes": routes}
            ],
        },
        "http_filters": [
            {
                "name": _EXT_PROC,
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor",
                    "grpc_service": {"envoy_grpc": {"cluster_name": _SHIM_CLUSTER}},
                    "failure_mode_allow": False,
                    "message_timeout": f"{message_timeout:g}s",
                    "processing_mode": {
                        "request_header_mode": "SEND",
                        "request_body_mode": "BUFFERED",
                        "request_trailer_mode": "SKIP",
                        "response_header_mode": "SEND",
                        "response_body_mode": "FULL_DUPLEX_STREAMED",
                        "response_trailer_mode": "SEND",
                    },
                },
            },
            {
                "name": "envoy.filters.http.router",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router"
                },
            },
        ],
    }
    clusters: list[dict[str, Any]] = [
        {
            "name": _SHIM_CLUSTER,
            **_endpoint(_SHIM_CLUSTER, sidecar_host, shim_port),
            "typed_extension_protocol_options": {
                "envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                    "@type": "type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions",
                    "explicit_http_config": {"http2_protocol_options": {}},
                }
            },
            # each request held is a stream to the shim, and Envoy's default
            # allows 1024 at once, made for short requests
            "circuit_breakers": {
                "thresholds": [{"max_requests": 65536, "max_pending_requests": 65536}]
            },
        },
        {
            "name": _SIDECAR_CLUSTER,
            **_endpoint(_SIDECAR_CLUSTER, sidecar_host, sidecar_port),
        },
        *(provider.cluster(max_body_bytes, ca_file) for provider in providers.values()),
    ]
    return {
        # Envoy warns at startup when nothing limits its connections
        "overload_manager": {
            "resource_monitors": [
                {
                    "name": "envoy.resource_monitors.global_downstream_max_connections",
                    "typed_config": {
                        "@type": "type.googleapis.com/envoy.extensions.resource_monitors.downstream_connections.v3.DownstreamConnectionsConfig",
                        "max_active_downstream_connections": MAX_CONNECTIONS,
                    },
                }
            ]
        },
        "static_resources": {
            "listeners": [
                {
                    "name": "model_api",
                    "address": {
                        "socket_address": {"address": "0.0.0.0", "port_value": port}
                    },
                    "per_connection_buffer_limit_bytes": max_body_bytes,
                    "filter_chains": [
                        {
                            "filters": [
                                {
                                    "name": "envoy.filters.network.http_connection_manager",
                                    "typed_config": connection_manager,
                                }
                            ]
                        }
                    ],
                }
            ],
            "clusters": clusters,
        },
    }


def envoy_yaml(**settings: Any) -> str:
    """`envoy_config()` as YAML, for Envoy's `-c`.

    Args:
        **settings: The arguments of `envoy_config()`.
    """
    header = f"# Generated by inspect_sentinel for Envoy {ENVOY_VERSION}.\n"
    return header + yaml.safe_dump(envoy_config(**settings), sort_keys=False)


class _Provider:
    def __init__(self, name: str, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"{name}: {url!r} is not an http or https URL.")
        if parts.path not in ("", "/"):
            raise ValueError(
                f"{name}: {url!r} has a path, and a provider is reached at its root."
            )
        self.name = name
        self.host = parts.hostname
        self.tls = parts.scheme == "https"
        self.port = parts.port or (443 if self.tls else 80)

    def cluster(self, max_body_bytes: int, ca_file: str) -> dict[str, Any]:
        cluster: dict[str, Any] = {
            "name": self.name,
            **_endpoint(self.name, self.host, self.port),
            "per_connection_buffer_limit_bytes": max_body_bytes,
            # Envoy drops a reply's trailers over HTTP/1.1 unless told
            # otherwise, and the shim judges a reply that has them at its
            # trailers
            "typed_extension_protocol_options": {
                "envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                    "@type": "type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions",
                    "explicit_http_config": {
                        "http_protocol_options": {"enable_trailers": True}
                    },
                }
            },
        }
        if self.tls:
            cluster["transport_socket"] = {
                "name": "envoy.transport_sockets.tls",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext",
                    "sni": self.host,
                    # Envoy checks a provider's certificate only when told
                    # what to trust and which name it must carry
                    "common_tls_context": {
                        "validation_context": {
                            "trusted_ca": {"filename": ca_file},
                            "match_typed_subject_alt_names": [
                                {"san_type": "DNS", "matcher": {"exact": self.host}}
                            ],
                        }
                    },
                },
            }
        return cluster


def _to_provider(
    path: str, provider: _Provider, ext_proc: dict[str, Any]
) -> dict[str, Any]:
    return {
        # an exact path, which a query string doesn't change
        "match": {"path": path},
        "route": {
            "cluster": provider.name,
            "host_rewrite_literal": provider.host,
            # a held reply is never cut off
            "timeout": "0s",
        },
        # Set, not removed: without the header a provider may choose any
        # encoding.
        "request_headers_to_add": [
            {
                "header": {"key": "accept-encoding", "value": "identity"},
                "append_action": "OVERWRITE_IF_EXISTS_OR_ADD",
            }
        ],
        "request_headers_to_remove": list(INSPECT_HEADERS),
        "typed_per_filter_config": _per_route(ext_proc),
    }


def _per_route(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        _EXT_PROC: {
            "@type": "type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExtProcPerRoute",
            **settings,
        }
    }


def _endpoint(name: str, host: str, port: int) -> dict[str, Any]:
    return {
        "type": "LOGICAL_DNS",
        "dns_lookup_family": "V4_PREFERRED",
        "connect_timeout": "5s",
        "load_assignment": {
            "cluster_name": name,
            "endpoints": [
                {
                    "lb_endpoints": [
                        {
                            "endpoint": {
                                "address": {
                                    "socket_address": {
                                        "address": host,
                                        "port_value": port,
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
        },
    }
