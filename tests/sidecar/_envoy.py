"""A real Envoy in front of the sidecar's Envoy shim, for tests.

Envoy runs in Docker, with the configuration `envoy_config()` writes. The shim, the sidecar's endpoints and a stand-in provider run in this process, on an event loop of their own in a thread, reached from the container at `host.docker.internal`. The stand-in provider speaks HTTP/1.1 by hand, so that a test controls how a reply is split, when each part is sent, and its trailers.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Generator, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
import uvicorn
from starlette.applications import Starlette

from inspect_sentinel.sidecar import Answer, Handler, Reply, Request
from inspect_sentinel.sidecar._eval_endpoint import eval_routes
from inspect_sentinel.sidecar.envoy import ENVOY_VERSION, envoy_server, envoy_yaml

IMAGE = os.environ.get("SENTINEL_ENVOY_IMAGE", f"envoyproxy/envoy:v{ENVOY_VERSION}")
"""The Envoy the tests run, which `SENTINEL_ENVOY_IMAGE` changes, to test another release."""

T = TypeVar("T")


def docker_missing() -> str | None:
    """Why Envoy can't be run here, or None if it can."""
    if shutil.which("docker") is None:
        return "docker is not installed"
    found = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        return "docker is not running"
    return None


@dataclass
class Scripted:
    """What the stand-in provider answers a request with."""

    body: Sequence[bytes] = (b"{}",)
    """The reply's body, in the parts it is sent in."""

    status: int = 200
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict[str, str])
    """More headers, by name."""

    trailers: dict[str, str] | None = None
    """Trailers, which make the reply chunked."""

    chunked: bool = False
    """Whether the reply is chunked rather than of a declared length, as a stream is."""

    pause: float = 0.0
    """Seconds between one part and the next."""


@dataclass(frozen=True)
class Received:
    """A request as the stand-in provider received it."""

    method: str
    path: str
    headers: dict[str, str]
    """By lowercase name."""

    body: bytes


class _Provider:
    """A model provider that answers each request as a test scripted it."""

    def __init__(self) -> None:
        self.script: Callable[[Received], Scripted] = lambda _: Scripted()
        self.received: list[Received] = []

    async def serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                received = await _read_request(reader)
                if received is None:
                    return
                self.received.append(received)
                await _write_reply(writer, self.script(received))
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


async def _read_request(reader: asyncio.StreamReader) -> Received | None:
    head = await reader.readuntil(b"\r\n\r\n") if not reader.at_eof() else b""
    if not head:
        return None
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {
        name.strip().lower(): value.strip()
        for name, _, value in (line.partition(":") for line in lines[1:] if line)
    }
    if "content-length" in headers:
        body = await reader.readexactly(int(headers["content-length"]))
    elif headers.get("transfer-encoding", "").lower() == "chunked":
        body = b""
        while True:
            size = int((await reader.readuntil(b"\r\n")).strip().split(b";")[0], 16)
            body += await reader.readexactly(size + 2)
            body = body[:-2]
            if size == 0:
                # the trailers, if any, end with an empty line
                while (await reader.readuntil(b"\r\n")) != b"\r\n":
                    pass
                break
    else:
        body = b""
    return Received(method, path, headers, body)


async def _write_reply(writer: asyncio.StreamWriter, reply: Scripted) -> None:
    chunked = reply.chunked or reply.trailers is not None
    head = [
        f"HTTP/1.1 {reply.status} Status",
        f"content-type: {reply.content_type}",
        *(f"{name}: {value}" for name, value in reply.headers.items()),
    ]
    if chunked:
        head.append("transfer-encoding: chunked")
        if reply.trailers:
            head.append(f"trailer: {', '.join(reply.trailers)}")
    else:
        head.append(f"content-length: {sum(len(part) for part in reply.body)}")
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode())
    await writer.drain()
    for part in reply.body:
        if reply.pause:
            await asyncio.sleep(reply.pause)
        if chunked:
            if part:
                writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
        else:
            writer.write(part)
        await writer.drain()
    if chunked:
        trailers = "".join(
            f"{name}: {value}\r\n" for name, value in (reply.trailers or {}).items()
        )
        writer.write(f"0\r\n{trailers}\r\n".encode())
        await writer.drain()


class Switched:
    """A handler whose judgments go to whichever handler a test has set."""

    def __init__(self) -> None:
        self.handler: Handler | None = None

    async def request(self, request: Request) -> Answer:
        assert self.handler is not None
        return await self.handler.request(request)

    async def reply(self, reply: Reply) -> Answer:
        assert self.handler is not None
        return await self.handler.reply(reply)

    def __getattr__(self, name: str) -> Any:
        # the endpoint for evals asks the handler more than this
        assert self.handler is not None
        return getattr(self.handler, name)


@dataclass
class Stack:
    """Envoy, with the sidecar behind it and a stand-in provider behind that."""

    url: str
    """Envoy's listener, as the test process reaches it."""

    provider: _Provider
    switched: Switched
    loop: asyncio.AbstractEventLoop
    container: str

    def run(self, coroutine: Awaitable[T]) -> T:
        """Run a coroutine on the stack's event loop, where the handler runs, and wait for it."""
        future: Future[T] = asyncio.run_coroutine_threadsafe(
            _awaited(coroutine), self.loop
        )
        return future.result(timeout=60)

    def use(self, handler: Handler) -> None:
        """Judge with `handler` from now on, and forget what the provider received."""
        self.switched.handler = handler
        self.provider.received.clear()
        self.provider.script = lambda _: Scripted()

    def logs(self) -> str:
        """What Envoy has logged."""
        return subprocess.run(
            ["docker", "logs", self.container], capture_output=True, text=True
        ).stderr


async def _close(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.close()


async def _awaited(awaitable: Awaitable[T]) -> T:
    return await awaitable


@contextmanager
def envoy_stack(
    *,
    max_body_bytes: int = 64 * 1024 * 1024,
    message_timeout: float = 60.0,
    shim: bool = True,
) -> Generator[Stack]:
    """Start Envoy, the shim, the sidecar's endpoints and a stand-in provider, and stop them after.

    Args:
        max_body_bytes: What the shim holds, and Envoy holds of a request.
        message_timeout: How long Envoy waits for a request to be judged.
        shim: Whether to start the shim, or put in its place a listener that closes each connection Envoy makes.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    stop: asyncio.Event | None = None
    provider = _Provider()
    switched = Switched()
    ports: dict[str, int] = {}
    failed: list[BaseException] = []

    async def serve() -> None:
        nonlocal stop
        stop = asyncio.Event()
        # the container reaches these through the host's gateway, so they
        # listen on every address
        provider_server = await asyncio.start_server(provider.serve, "0.0.0.0", 0)
        ports["provider"] = provider_server.sockets[0].getsockname()[1]
        ports["shim"] = _free_port()
        stop_shim: Callable[[], Awaitable[None]]
        if shim:
            server = envoy_server(
                switched,  # type: ignore[arg-type] # judges as a Handler does
                host="0.0.0.0",
                port=ports["shim"],
                max_body_bytes=max_body_bytes,
            )
            await server.start()
            stop_shim = lambda: server.stop(0)  # noqa: E731
        else:
            # what Envoy reaches in place of the shim takes each connection
            # and closes it, so that Envoy fails at once, where a port
            # nothing listens on is held open by Docker Desktop
            refusing = await asyncio.start_server(_close, "0.0.0.0", ports["shim"])

            async def stop_refusing() -> None:
                refusing.close()
                await refusing.wait_closed()

            stop_shim = stop_refusing
        ports["sidecar"] = _free_port()
        sidecar = uvicorn.Server(
            uvicorn.Config(
                Starlette(routes=eval_routes(switched, max_body_bytes=max_body_bytes)),  # type: ignore[arg-type]
                host="0.0.0.0",
                port=ports["sidecar"],
                log_level="warning",
            )
        )
        serving = asyncio.ensure_future(sidecar.serve())
        while not sidecar.started:
            await asyncio.sleep(0.01)
        ready.set()
        await stop.wait()
        sidecar.should_exit = True
        await serving
        await stop_shim()
        provider_server.close()

    def run() -> None:
        try:
            loop.run_until_complete(serve())
        except BaseException as ex:  # noqa: BLE001 reported to the test
            failed.append(ex)
            ready.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(30)
    if failed:
        raise failed[0]
    folder = Path(tempfile.mkdtemp())
    container = ""
    try:
        config = folder / "envoy.yaml"
        config.write_text(
            envoy_yaml(
                port=8080,
                sidecar_host="host.docker.internal",
                sidecar_port=ports["sidecar"],
                shim_port=ports["shim"],
                anthropic=f"http://host.docker.internal:{ports['provider']}",
                openai=f"http://host.docker.internal:{ports['provider']}",
                max_body_bytes=max_body_bytes,
                message_timeout=message_timeout,
            )
        )
        # Envoy runs as a user of its own, which Docker Desktop hides
        folder.chmod(0o755)
        config.chmod(0o644)
        started = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--add-host=host.docker.internal:host-gateway",
                "--publish=127.0.0.1::8080",
                f"--volume={folder}:/etc/sentinel:ro",
                IMAGE,
                "--config-path",
                "/etc/sentinel/envoy.yaml",
                "--log-level",
                "warning",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        container = started.stdout.strip()
        published = subprocess.run(
            ["docker", "port", container, "8080/tcp"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]
        url = f"http://{published}"
        _wait_for(url, container)
        yield Stack(url, provider, switched, loop, container)
    finally:
        if container:
            subprocess.run(["docker", "rm", "--force", container], capture_output=True)
        shutil.rmtree(folder, ignore_errors=True)
        if stop is not None:
            loop.call_soon_threadsafe(stop.set)
        thread.join(30)
        loop.close()


def _wait_for(url: str, container: str) -> None:
    # Envoy answers a path it has no route for with a 404 once it listens
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/ready", timeout=1).status_code == 404:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    logs = subprocess.run(
        ["docker", "logs", container], capture_output=True, text=True
    ).stderr
    raise TimeoutError(f"Envoy didn't start:\n{logs}")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        return probe.getsockname()[1]


def headers_of(stack: Stack) -> list[dict[str, Any]]:
    """The headers of each request the provider received."""
    return [received.headers for received in stack.provider.received]
