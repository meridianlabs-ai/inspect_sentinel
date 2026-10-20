"""Write Envoy's configuration for the sidecar's Envoy shim."""

import argparse
import sys

from .._http import MAX_BODY_BYTES
from ._config import ENVOY_VERSION, MESSAGE_TIMEOUT, envoy_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m inspect_sentinel.sidecar.envoy",
        description=f"Write Envoy's configuration, for Envoy {ENVOY_VERSION}, to put the sidecar in front of model providers.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="The port Envoy listens on."
    )
    parser.add_argument(
        "--sidecar-host", default="127.0.0.1", help="Where Envoy reaches the sidecar."
    )
    parser.add_argument(
        "--sidecar-port", type=int, default=8900, help="The sidecar's --port."
    )
    parser.add_argument(
        "--shim-port", type=int, default=8901, help="The sidecar's --envoy-port."
    )
    parser.add_argument(
        "--anthropic",
        default="https://api.anthropic.com",
        metavar="URL",
        help="Anthropic's API. An http URL is reached without TLS.",
    )
    parser.add_argument(
        "--openai",
        default="https://api.openai.com",
        metavar="URL",
        help="OpenAI's API. An http URL is reached without TLS.",
    )
    parser.add_argument(
        "--no-anthropic", action="store_true", help="Leave Anthropic's routes out."
    )
    parser.add_argument(
        "--no-openai", action="store_true", help="Leave OpenAI's routes out."
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=MAX_BODY_BYTES,
        metavar="BYTES",
        help="The largest request Envoy holds while it is judged: the sidecar's --max-body-bytes.",
    )
    parser.add_argument(
        "--message-timeout",
        type=float,
        default=MESSAGE_TIMEOUT,
        metavar="SECONDS",
        help="How long Envoy waits for a request to be judged, at most an hour, the default.",
    )
    arguments = parser.parse_args()
    try:
        sys.stdout.write(
            envoy_yaml(
                port=arguments.port,
                sidecar_host=arguments.sidecar_host,
                sidecar_port=arguments.sidecar_port,
                shim_port=arguments.shim_port,
                anthropic=None if arguments.no_anthropic else arguments.anthropic,
                openai=None if arguments.no_openai else arguments.openai,
                max_body_bytes=arguments.max_body_bytes,
                message_timeout=arguments.message_timeout,
            )
        )
    except ValueError as ex:
        parser.error(str(ex))


main()
