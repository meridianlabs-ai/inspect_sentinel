"""The sidecar behind Envoy: Envoy's external processor, and Envoy's configuration for it.

Envoy runs no code of ours. Its `ext_proc` filter sends each model request and reply, over gRPC, to the Envoy shim in the sidecar's process, which puts them to the same handler as the endpoint for proxies. `python -m inspect_sentinel.sidecar --envoy-port PORT` serves the shim beside the sidecar's endpoints, and `python -m inspect_sentinel.sidecar.envoy` writes Envoy's configuration for it. Install the `envoy` extra for both.
"""

from ._config import ENVOY_VERSION, envoy_config, envoy_yaml
from ._shim import EnvoyShim, envoy_server

__all__ = ["ENVOY_VERSION", "EnvoyShim", "envoy_config", "envoy_server", "envoy_yaml"]
