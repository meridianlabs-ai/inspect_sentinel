from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import anyio

from ._exchange import Refuse, Replace, Reply, succeeded
from ._wire import Format

if TYPE_CHECKING:
    from ._handler import Handler

logger = logging.getLogger(__name__)

PENDING_BYTES = 1024 * 1024
"""How much of a reply that goes out as it arrives may wait for the proxy to collect it, by default, before feeding waits for it to drain."""


@dataclass(frozen=True, kw_only=True)
class ReplyStart:
    """A provider's reply to a `Request`, as its status and headers arrive, before any of its body."""

    id: str
    """The `id` of the request this answers."""

    provider: str
    """Whose API the body is written in."""

    model: str
    """The model the request was for, as the proxy names it."""

    user: str | None
    """Who the proxy says is calling, if it knows."""

    status: int
    """The provider's HTTP status."""

    content_type: str
    """The provider's `content-type`, which says whether the body is a stream."""

    format: Format | None = None
    """Which API the body is written in, if the proxy said. Without it, the API is worked out from `provider` and the body."""


@dataclass(frozen=True)
class Release:
    """Send these bytes on to the client, after what was sent before."""

    data: bytes


@dataclass(frozen=True)
class ErrorEvent:
    """Send this error event on, written in the provider's format, then end the reply.

    It refuses the turn once something has been sent, when the status can no longer change. The provider's SDK raises an exception on it.
    """

    data: bytes


@dataclass(frozen=True)
class Finished:
    """The reply has ended: end it for the client."""


Output: TypeAlias = Release | Refuse | ErrorEvent | Finished
"""What a reply stream gives a proxy to do, in order. A `Refuse` comes only while nothing has been sent, and the proxy sends it as an HTTP error. The output ends with `Finished`, a `Refuse` or an `ErrorEvent`."""


class ReplyStream:
    """A provider's reply, handed over as it arrives, and what the client is sent of it.

    Made by `Handler.reply_stream`. The proxy runs three things at once: `run`, which judges the reply; feeding the provider's bytes in with `feed` and then `end`; and reading what to send the client by iterating over the stream. Feeding a chunk means only that the stream has taken it, and ending the input only that no more of the provider's bytes are coming: judging, and what is sent, go on after that.

    The reply is held whole and judged as `Handler.reply` judges it, and what comes out is its answer: the provider's bytes, the replacement, or the refusal. A reply in a status outside 2xx is the provider's error, which the handler is told of and which goes out as it arrives, unread. A reply larger than `max_held_bytes` is refused with a 502 and not judged, as soon as it is fed past the limit.

    Feeding waits only for output that can drain: while more than `max_pending_bytes` of a reply that goes out as it arrives is waiting for the proxy to collect it. It never waits on what is held back to judge, which can only go out once more has been fed. Once the output has ended, bytes still fed are dropped.

    If the proxy stops before the end, it calls `abort`. A judgment already running goes on to the end, and its decision is kept for the run. A reply that ended before it was judged is recorded as cut off.

    Iterating raises what judging raised, as `Handler.reply` would, after the output that came before it. Output that stops without `Finished`, a `Refuse` or an `ErrorEvent` is never a complete reply.
    """

    def __init__(
        self,
        handler: Handler,
        start: ReplyStart,
        *,
        max_held_bytes: int,
        max_pending_bytes: int = PENDING_BYTES,
    ) -> None:
        self._handler = handler
        self._start = start
        self._max_held_bytes = max_held_bytes
        self._max_pending_bytes = max_pending_bytes
        self._held = bytearray()
        self._ended = False
        self._aborted = False
        self._over = False
        # the output has ended, so nothing more fed is wanted
        self._finished = False
        self._arrived = anyio.Event()
        self._drained = anyio.Event()
        # the delivery waiting for the proxy to collect an output, which an
        # abort cuts short, as no one is left to collect it
        self._delivering = anyio.CancelScope()
        # one output at a time, so that a slow proxy holds back the worker,
        # and through it the feeding
        self._send, self._receive = anyio.create_memory_object_stream[Output](1)
        self._raised: Exception | None = None

    @property
    def start(self) -> ReplyStart:
        """The reply's status and headers, as the proxy handed them over."""
        return self._start

    async def feed(self, data: bytes) -> None:
        """Take the next bytes of the provider's reply. Feed one chunk at a time.

        Waits while output that goes out as it arrives is backed up, until the proxy collects it. Bytes that take a reply held back to judge past its limit are not taken, and the reply is refused.

        Raises:
            RuntimeError: If the input has already been ended.
        """
        if self._ended:
            raise RuntimeError("The reply's input has already been ended.")
        if not succeeded(self._start.status):
            while True:
                self._drained = anyio.Event()
                if self._closed() or len(self._held) < self._max_pending_bytes:
                    break
                await self._drained.wait()
        elif len(self._held) + len(data) > self._max_held_bytes:
            self._over = True
            self._arrived.set()
            return
        if self._closed():
            return
        self._held += data
        self._arrived.set()

    async def end(self) -> None:
        """Say that no more of the provider's bytes are coming."""
        self._ended = True
        self._arrived.set()

    async def abort(self) -> None:
        """Say that the client has gone, or that the proxy has stopped waiting. Nothing more is sent."""
        self._aborted = True
        self._arrived.set()
        self._drained.set()
        self._delivering.cancel()

    async def run(self) -> None:
        """Judge the reply as it is handed over, until its output has ended."""
        try:
            await self._run()
        except Exception as ex:
            if self._aborted:
                # no one is reading the output to be told of it
                logger.error(
                    "The reply %r was not judged after its client had gone.",
                    self._start.id,
                    exc_info=ex,
                )
            else:
                self._raised = ex
        finally:
            self._finish()
            self._send.close()

    async def __aiter__(self) -> AsyncIterator[Output]:
        async with self._receive:
            async for output in self._receive:
                if self._aborted:
                    break
                yield output
        if self._raised is not None:
            raise self._raised

    async def _run(self) -> None:
        start = self._start
        if not succeeded(start.status):
            await self._provider_error()
            return
        await self._until(lambda: self._ended or self._aborted or self._over)
        if self._over:
            # the reply isn't read, as the endpoint for proxies doesn't read a
            # call larger than it takes
            await self._handler.reply_not_judged(start.id, "unreadable")
            await self._out(
                Refuse(
                    502,
                    f"The provider's reply is larger than the {self._max_held_bytes} bytes the sidecar holds, so it was not judged.",
                )
            )
            return
        if not self._ended:
            await self._handler.reply_not_judged(start.id, "cut_off")
            return
        answer = await self._handler.reply(self._reply(bytes(self._held)))
        if isinstance(answer, Refuse):
            await self._out(answer)
            return
        body = answer.body if isinstance(answer, Replace) else bytes(self._held)
        await self._out(Release(body))
        await self._out(Finished())

    async def _provider_error(self) -> None:
        # A provider's error goes on unjudged, and the handler reads nothing
        # of it: it is told at the start, and the body goes out as it comes,
        # whatever its size.
        answer = await self._handler.reply(self._reply(b""))
        if isinstance(answer, Refuse):
            await self._out(answer)
            return
        while True:
            await self._until(lambda: bool(self._held) or self._ended or self._aborted)
            if self._held:
                data = bytes(self._held)
                self._held.clear()
                self._drained.set()
                await self._out(Release(data))
            if self._aborted:
                return
            if self._ended and not self._held:
                await self._out(Finished())
                return

    def _closed(self) -> bool:
        return self._aborted or self._finished or self._over

    def _finish(self) -> None:
        self._finished = True
        self._held.clear()
        self._drained.set()

    async def _until(self, done: Callable[[], bool]) -> None:
        while True:
            # a new event before looking, so that what arrives after the look
            # sets the one waited on
            self._arrived = anyio.Event()
            if done():
                return
            await self._arrived.wait()

    async def _out(self, output: Output) -> None:
        if not isinstance(output, Release):
            # the last output, after which nothing fed is wanted
            self._finish()
        if self._aborted:
            return
        # only the delivery is cut short by an abort: a judgment never runs
        # inside this scope, so it goes on to the end
        with anyio.CancelScope() as self._delivering:
            if self._aborted:
                return
            try:
                await self._send.send(output)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                # the proxy stopped reading without saying so
                pass

    def _reply(self, body: bytes) -> Reply:
        start = self._start
        return Reply(
            id=start.id,
            provider=start.provider,
            format=start.format,
            model=start.model,
            user=start.user,
            status=start.status,
            content_type=start.content_type,
            body=body,
        )
