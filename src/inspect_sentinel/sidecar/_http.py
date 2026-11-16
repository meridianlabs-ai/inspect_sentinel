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
    """Read a request's body and discard it. Stop as soon as the body is known to be larger than a limit.

    Use it for a request whose body the sidecar doesn't need, when the client sends a body anyway. A body within the limit is read to its end before the request is answered, so that a proxy still sending the body doesn't see the connection close while it sends.

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
