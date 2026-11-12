from __future__ import annotations

from starlette.requests import Request as HttpRequest
from starlette.responses import PlainTextResponse
from starlette.routing import Route

HEALTH_PATH = "/health"
"""The path that answers a health check, on every port the sidecar listens on."""


async def read_within(http: HttpRequest, limit: int) -> bytes | None:
    """Read a request's body, giving up as soon as it is known to be larger than a limit.

    Args:
        http: The request.
        limit: The largest body to read, in bytes.

    Returns:
        The body, or None if it is larger than `limit`. No more than `limit` bytes are held either way.
    """
    declared = http.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in http.stream():
        if len(body) + len(chunk) > limit:
            return None
        body += chunk
    return bytes(body)


async def discard_within(http: HttpRequest, limit: int) -> bool:
    """Read a request's body and keep none of it, giving up as soon as it is known to be larger than a limit.

    For a request whose body nothing reads, sent by a client that sends one anyway. The body is read to its end before the request is answered, so a proxy still sending it doesn't see the connection close under it.

    Args:
        http: The request.
        limit: The largest body to read, in bytes.

    Returns:
        False if the body is larger than `limit`.
    """
    declared = http.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        return False
    read = 0
    async for chunk in http.stream():
        read += len(chunk)
        if read > limit:
            return False
    return True


def health_route() -> Route:
    """A route that answers `ok` to a GET of `HEALTH_PATH`, for whatever restarts the sidecar to probe.

    It asks for no credential and checks nothing: the sentinel is loaded before the sidecar listens, so an answer means the sidecar is ready, and a sentinel that blocks the process stops the answer too.
    """

    async def health(http: HttpRequest) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return Route(HEALTH_PATH, health, methods=["GET"])
