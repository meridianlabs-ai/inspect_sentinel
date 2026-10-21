from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import anyio

from ._errors import error_event, not_judged, refusal_for
from ._exchange import Refuse, Replace, Reply, succeeded
from ._release import (
    AnthropicTurn,
    ChatTurn,
    ClientCheck,
    EventReader,
    ResponsesTurn,
    Role,
    Turn,
)
from ._state import NotJudged
from ._wire import Format, StreamEvent, write_events

if TYPE_CHECKING:
    from ._handler import Handler

logger = logging.getLogger(__name__)

PENDING_BYTES = 1024 * 1024
"""How much of a reply that goes out as it arrives may wait for the proxy to collect it, by default, before feeding waits for it to drain."""

KEEPALIVE_SECONDS = 15.0
"""How long a client of a reply released early is sent nothing, by default, before it is sent a keepalive."""

KEEPALIVE = b": keepalive\n\n"
"""A keepalive: an SSE comment, which every provider SDK skips."""


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
    not_judged: NotJudged | None = None
    """Why the turn wasn't judged, when it wasn't, rather than being refused. A proxy that lets through what can't be judged may let the reply through instead, if it has sent its client nothing but keepalives."""


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

    A reply released early that has sent nothing for `keepalive` seconds sends a keepalive, `KEEPALIVE`, as a `Release`, which sends the status: a turn refused after it is refused with an `ErrorEvent`. A reply held whole sends none, so that a refusal keeps its status.

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
        keepalive: float | None = None,
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
        # under early release, how much of `_held` has been read into events,
        # and whether anything has gone out, after which the status is sent
        self._early = (
            succeeded(start.status)
            and handler.release == "early"
            and start.content_type.startswith("text/event-stream")
        )
        self._consumed = 0
        self._sent = False
        self._events = _Released(start)
        self._keepalive = keepalive if self._early else None
        # one output at a time, so that whether anything has gone out, which
        # a refusal is written by, is never changed by a keepalive meanwhile
        self._sending = anyio.Lock()
        self._last_out = 0.0

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
        if (
            succeeded(self._start.status)
            and len(self._held) + len(data) > self._max_held_bytes
        ):
            self._over = True
            self._arrived.set()
            return
        if self._early or not succeeded(self._start.status):
            # output that goes out as it arrives drains, so feeding waits for it
            while True:
                self._drained = anyio.Event()
                backlog = len(self._held) - self._consumed
                if self._closed() or backlog < self._max_pending_bytes:
                    break
                await self._drained.wait()
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
            await self._kept_alive()
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

    async def _kept_alive(self) -> None:
        # judging, with keepalives beside it until the output ends
        every = self._keepalive
        if every is None:
            await self._run()
            return
        raised: Exception | None = None
        self._last_out = anyio.current_time()
        async with anyio.create_task_group() as group:
            group.start_soon(self._keep_alive, every)
            try:
                await self._run()
            except Exception as ex:
                raised = ex
            finally:
                group.cancel_scope.cancel()
        if raised is not None:
            raise raised

    async def _keep_alive(self, every: float) -> None:
        while not self._closed():
            wait = self._last_out + every - anyio.current_time()
            if wait > 0:
                await anyio.sleep(wait)
                continue
            if self._events.format is None:
                # a refusal can be written in the stream only once its format
                # is known, so until then nothing may send the status
                await anyio.sleep(every)
                continue
            await self._out(Release(KEEPALIVE), keepalive=every)

    async def _run(self) -> None:
        start = self._start
        if not succeeded(start.status):
            await self._provider_error()
            return
        if self._early:
            await self._run_early()
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

    async def _run_early(self) -> None:
        # The text and reasoning before the turn's first call go out as they
        # arrive. The rest is held, and the whole reply is judged once it has
        # ended, as a reply held whole is, and only what follows what went
        # out is sent. Before anything has gone out a failure is raised, for
        # the proxy to refuse with a real status; after, it is an error event
        # in the stream.
        start, events = self._start, self._events
        try:
            ended = await self._release_early()
        except Exception as ex:
            # the reply couldn't be read as it arrived, so it isn't judged
            await self._handler.reply_not_judged(start.id, "unreadable")
            if not self._sent or self._aborted:
                raise
            await self._fail(
                refusal_for(ex, "reply", start.id, reply=True), "unreadable"
            )
            return
        if not ended:
            return
        released = bytes(events.released)
        try:
            answer = await self._handler.reply(
                self._reply(bytes(self._held)), released=released
            )
        except Exception as ex:
            if not self._sent or self._aborted:
                raise
            await self._fail(
                refusal_for(ex, "reply", start.id, reply=True), not_judged(ex)
            )
            return
        if isinstance(answer, Refuse):
            await self._fail(answer)
            return
        body = answer.body if isinstance(answer, Replace) else bytes(self._held)
        # the handler never changes what has gone out
        assert body.startswith(released)
        rest = body[len(released) :]
        if rest:
            self._sent = True
            await self._out(Release(rest))
        await self._out(Finished())

    async def _release_early(self) -> bool:
        # what goes out as it arrives, until the first event that is held;
        # whether the reply's input ended, rather than being aborted or
        # refused for its size
        events = self._events
        while True:
            await self._until(
                lambda: (
                    len(self._held) > self._consumed
                    or self._ended
                    or self._aborted
                    or self._over
                )
            )
            if self._aborted:
                await self._handler.reply_not_judged(self._start.id, "cut_off")
                return False
            if self._over:
                await self._handler.reply_not_judged(self._start.id, "unreadable")
                await self._fail(
                    Refuse(
                        502,
                        f"The provider's reply is larger than the {self._max_held_bytes} bytes the sidecar holds, so it was not judged.",
                    ),
                    "unreadable",
                )
                return False
            data = bytes(self._held[self._consumed :])
            # held bytes can only go out once the turn has ended, so feeding
            # never waits on them
            self._consumed = len(self._held)
            self._drained.set()
            ending = self._ended and self._consumed == len(self._held)
            if not events.holding:
                arrived = events.reader.push(data)
                if ending:
                    arrived += events.reader.finish()
                await self._release_ready(arrived)
            if self._aborted:
                # the client went while what was ready went out, so no
                # judgment starts
                await self._handler.reply_not_judged(self._start.id, "cut_off")
                return False
            if ending:
                return True

    async def _release_ready(self, arrived: list[StreamEvent]) -> None:
        events = self._events
        ready: list[tuple[StreamEvent, Role]] = []
        for event in arrived:
            events.originals.append(event)
            if events.holding:
                continue
            role = events.role(event)
            if role == "held":
                events.holding = True
                continue
            ready.append((event, role))
        if not ready:
            return
        if not self._sent and all(role in ("opening", "filler") for _, role in ready):
            # the opening waits for the first text or reasoning, so that a
            # turn can still be refused with a real status
            events.opening += [event for event, _ in ready]
            return
        out = [*events.opening, *(event for event, _ in ready)]
        events.opening = []
        if events.check is not None:
            events.check.check(out)
        written = write_events(out)
        events.released += written
        self._sent = True
        await self._out(Release(written))

    async def _fail(self, refused: Refuse, unjudged: NotJudged | None = None) -> None:
        # a refusal, or a turn that wasn't judged: an HTTP error while nothing
        # has gone out, and an error event in the stream once something has
        events = self._events
        if not self._sent or events.format is None:
            await self._out(refused)
            return
        await self._out(
            ErrorEvent(error_event(events.format, refused, events.originals), unjudged)
        )

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

    async def _out(self, output: Output, *, keepalive: float | None = None) -> None:
        if not isinstance(output, Release):
            # the last output, after which nothing fed is wanted
            self._finish()
        if self._aborted:
            return
        async with self._sending:
            if keepalive is not None:
                # wanted only while nothing else goes out, and the status goes
                # with it
                if self._closed() or anyio.current_time() - self._last_out < keepalive:
                    return
                self._sent = True
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
            self._last_out = anyio.current_time()

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


class _Released:
    """What a reply released early has read, holds and has sent."""

    def __init__(self, start: ReplyStart) -> None:
        self.start = start
        self.reader = EventReader()
        self.format: Format | None = start.format
        self.turn: Turn | None = None
        self.check: ClientCheck | None = None
        # the events read, as the provider sent them, which an error event is
        # written from
        self.originals: list[StreamEvent] = []
        # the opening, held for the first text or reasoning that goes out
        self.opening: list[StreamEvent] = []
        # once an event is held, so is everything after it
        self.holding = False
        self.released = bytearray()

    def role(self, event: StreamEvent) -> Role:
        turn = self._turn_for(event)
        return "filler" if turn is None else turn.place(event)

    def _turn_for(self, event: StreamEvent) -> Turn | None:
        # the format, from the proxy, or from the provider and the stream's
        # first event, once it has one
        if self.turn is not None:
            return self.turn
        if self.format is None:
            if self.start.provider == "anthropic":
                self.format = "anthropic-messages"
            elif event.data is None:
                return None
            else:
                chat = event.data.get("object") == "chat.completion.chunk"
                self.format = "openai-chat" if chat else "openai-responses"
        if self.format == "anthropic-messages":
            self.turn = AnthropicTurn()
        elif self.format == "openai-chat":
            self.turn = ChatTurn()
        else:
            self.turn = ResponsesTurn()
        self.check = ClientCheck(self.format)
        return self.turn
