from __future__ import annotations

from .._handler import REQUEST_ID
from .._inspect_client import INSPECT_CONTEXT

FORMAT_METADATA = "sentinel-format"
"""The gRPC metadata in which Envoy's route tells the shim which API a request is written in, such as `anthropic-messages`."""

PROVIDER_METADATA = "sentinel-provider"
"""The gRPC metadata in which Envoy's route names the provider, such as `anthropic`."""

INSPECT_HEADERS = (INSPECT_CONTEXT, REQUEST_ID)
"""Inspect's own request headers, which the shim hands the sidecar and which never reach the provider."""
