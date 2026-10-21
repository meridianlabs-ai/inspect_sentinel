"""Reply streams through the endpoint for proxies: a reply a proxy hook hands over as it arrives, and what it is to send the caller of it."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypeAlias

import anyio
from anyio.abc import TaskGroup
from pydantic import BaseModel, Field

from ._exchange import Refuse
from ._handler import Handler
from ._inspect_client import SentinelField
from ._json_rpc import INVALID_PARAMS, INVALID_REQUEST, UNKNOWN_STREAM, RpcError
from ._reply_stream import (
    ErrorEvent,
    Finished,
    Output,
    Release,
    ReplyStart,
    ReplyStream,
)
from ._state import NotJudged
from ._wire import Format

POLL_SECONDS = 20.0
"""How long `next_release` waits for output before answering that none is ready yet."""

STREAM_EXPIRY = 120.0
"""How long a reply stream is kept once no call has named it, after which it is aborted."""

PENDING_BYTES = 1024 * 1024
"""How much output may wait for a proxy hook to collect it before the stream holds back, and with it the hook's `reply_chunk`."""

_PROVIDER = "The provider whose format the body is in, such as `anthropic` or `openai`. Where `format` is given, the body is read by `format`, and this can be any name the proxy has for the provider."
_FORMAT = "Which API the body is written in: `anthropic-messages`, `openai-chat` or `openai-responses`. A proxy that knows says, and the body is then read as that, whatever `provider` is. Without it, the API is worked out from `provider` and the body."


class BeginReplyParams(BaseModel):
    """Parameters of `begin_reply`: a provider's reply, as its status and headers arrive, to be handed over as it arrives."""

    id: str = Field(description="The `id` of the request this is the reply to.")
    provider: str = Field(description=_PROVIDER)
    format: Format | None = Field(default=None, description=_FORMAT)
    model: str = Field(description="The model, by the name the proxy shows for it.")
    user: str | None = Field(
        default=None, description="Who the proxy signed the caller in as."
    )
    status: int = Field(description="The HTTP status of the provider's reply.")
    content_type: str = Field(
        description="The reply's content type, which says whether it is one JSON object or a stream of events."
    )
    keepalive: bool = Field(
        default=True,
        description="Whether the proxy hook can send the caller a keepalive, an SSE comment, as a `release` while the reply is held. Where it can't, none is given, so that a refusal keeps its status for as long as nothing else has gone out.",
    )


class ReplyChunkParams(BaseModel):
    """Parameters of `reply_chunk`: the next bytes of a reply, in order."""

    id: str = Field(description="The `id` of the reply's request.")
    seq: int = Field(
        ge=0,
        description="The chunk's place, from 0. A chunk sent again with the same bytes is taken once.",
    )
    data: str = Field(description="The chunk's bytes, as base64.")


class StreamParams(BaseModel):
    """Parameters of `end_reply` and `abort_reply`: a reply, by its request's `id`."""

    id: str = Field(description="The `id` of the reply's request.")


class NextReleaseParams(BaseModel):
    """Parameters of `next_release`: what the caller is to be sent after what the proxy hook has."""

    id: str = Field(description="The `id` of the reply's request.")
    after: int = Field(
        ge=0,
        description="How many outputs the proxy hook already has. Asking again with the same `after` gives the same outputs.",
    )


class TakenResult(BaseModel):
    """The call was taken."""

    taken: Literal[True] = True


class ReleaseOutput(BaseModel):
    """Send these bytes to the caller, after what was sent before."""

    kind: Literal["release"] = "release"
    data: str = Field(description="The bytes, as base64.")


class RefuseOutput(BaseModel):
    """Send the caller this error, with this status. It comes only while nothing has been sent, and ends the reply."""

    kind: Literal["refuse"] = "refuse"
    status: int = Field(ge=400, le=599, description="The HTTP status of the error.")
    message: str = Field(description="What the caller is told.")
    sentinel: SentinelField | None = Field(
        default=None,
        description="For an Inspect client: what was decided, which the proxy hook puts into the error as a top-level `sentinel` field.",
    )


class ErrorEventOutput(BaseModel):
    """Send the caller this error event, written in the provider's format, and end the reply. It refuses the turn once something has been sent."""

    kind: Literal["error_event"] = "error_event"
    data: str = Field(description="The event, as base64.")
    not_judged: NotJudged | None = Field(
        default=None,
        description="Why the turn wasn't judged, when it wasn't, rather than being refused. A proxy hook that lets through what isn't judged may let the reply through instead, if it has sent its caller nothing but keepalives.",
    )


class FinishedOutput(BaseModel):
    """The reply has ended: end it for the caller."""

    kind: Literal["finished"] = "finished"


StreamOutput: TypeAlias = Annotated[
    ReleaseOutput | RefuseOutput | ErrorEventOutput | FinishedOutput,
    Field(discriminator="kind"),
]
"""One thing a proxy hook is to do for the caller."""


class NextReleaseResult(BaseModel):
    """What the caller is to be sent next. No outputs means none was ready within the wait, never that the reply has ended: it ends with `finished`, `refuse` or `error_event`."""

    position: int = Field(
        description="How many outputs the proxy hook has once it has these, for its next `next_release`."
    )
    outputs: list[StreamOutput]


@dataclass
class _Open:
    """A reply stream a proxy hook is feeding, and what it has given."""

    stream: ReplyStream
    begun: BeginReplyParams
    # each chunk taken, by its place, as a digest, so a chunk sent again can
    # be told from another, with whether the stream has it yet, which a chunk
    # sent again waits for
    chunks: list[bytes] = field(default_factory=list[bytes])
    fed: list[anyio.Event] = field(default_factory=list[anyio.Event])
    # one chunk is fed at a time, so that none overtakes another
    feeding: anyio.Lock = field(default_factory=anyio.Lock)
    # the outputs given, the first of them at `base`, as those before were
    # collected
    outputs: list[StreamOutput] = field(default_factory=list[StreamOutput])
    base: int = 0
    pending: int = 0
    error: RpcError | None = None
    done: bool = False
    changed: anyio.Event = field(default_factory=anyio.Event)
    collected: anyio.Event = field(default_factory=anyio.Event)
    touched: float = field(default_factory=anyio.current_time)


class ReplyStreams:
    """The reply streams that proxy hooks have open, run in the endpoint's task group."""

    def __init__(
        self,
        handler: Handler,
        *,
        max_body_bytes: int,
        not_judged: Callable[[Exception, str], RpcError],
        poll_seconds: float = POLL_SECONDS,
        expiry: float = STREAM_EXPIRY,
        max_pending_bytes: int = PENDING_BYTES,
    ) -> None:
        self._handler = handler
        self._max_body_bytes = max_body_bytes
        self._not_judged = not_judged
        self._poll_seconds = poll_seconds
        self._expiry = expiry
        self._max_pending_bytes = max_pending_bytes
        self._open: dict[str, _Open] = {}
        self._group: TaskGroup | None = None

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """Run the streams' work while the endpoint serves."""
        async with anyio.create_task_group() as group:
            self._group = group
            group.start_soon(self._sweep)
            try:
                yield
            finally:
                self._group = None
                for open in list(self._open.values()):
                    await open.stream.abort()
                group.cancel_scope.cancel()

    async def begin(self, given: BeginReplyParams) -> TakenResult:
        known = self._open.get(given.id)
        if known is not None:
            if known.begun != given:
                raise RpcError(
                    INVALID_PARAMS,
                    f"The reply {given.id!r} was begun with other details.",
                )
            known.touched = anyio.current_time()
            return TakenResult()
        if self._group is None:
            raise RpcError(
                INVALID_REQUEST, "This endpoint doesn't serve reply streams."
            )
        stream = self._handler.reply_stream(
            ReplyStart(
                id=given.id,
                provider=given.provider,
                format=given.format,
                model=given.model,
                user=given.user,
                status=given.status,
                content_type=given.content_type,
            ),
            max_held_bytes=self._max_body_bytes,
            max_pending_bytes=self._max_pending_bytes,
            keepalive=given.keepalive,
        )
        open = _Open(stream, given)
        self._open[given.id] = open
        self._group.start_soon(stream.run)
        self._group.start_soon(self._collect, open)
        return TakenResult()

    async def chunk(self, given: ReplyChunkParams) -> TakenResult:
        open = self._find(given.id)
        try:
            data = base64.b64decode(given.data, validate=True)
        except ValueError:
            raise RpcError(
                INVALID_PARAMS, "The parameters don't fit: data: A chunk is base64."
            ) from None
        digest = hashlib.sha256(data).digest()
        if given.seq < len(open.chunks):
            if open.chunks[given.seq] != digest:
                raise RpcError(
                    INVALID_PARAMS,
                    f"Chunk {given.seq} of the reply {given.id!r} was sent before with other bytes.",
                )
            # taken once the stream has it, as the first sending of it is
            await open.fed[given.seq].wait()
            return TakenResult()
        if given.seq > len(open.chunks):
            raise RpcError(
                INVALID_PARAMS,
                f"Chunk {given.seq} of the reply {given.id!r} comes before chunk {len(open.chunks)}.",
            )
        open.chunks.append(digest)
        fed = anyio.Event()
        open.fed.append(fed)
        try:
            async with open.feeding:
                await open.stream.feed(data)
        except RuntimeError:
            raise RpcError(
                INVALID_REQUEST, f"The reply {given.id!r} has already been ended."
            ) from None
        finally:
            fed.set()
        return TakenResult()

    async def end(self, given: StreamParams) -> TakenResult:
        open = self._find(given.id)
        # after the chunks being fed, so that none is left out
        async with open.feeding:
            await open.stream.end()
        return TakenResult()

    async def abort(self, given: StreamParams) -> TakenResult:
        open = self._open.pop(given.id, None)
        if open is not None:
            await open.stream.abort()
            open.done = True
            open.changed.set()
            # a collector waiting for room has none to wait for
            open.collected.set()
        return TakenResult()

    async def next(self, given: NextReleaseParams) -> NextReleaseResult:
        open = self._find(given.id)
        if given.after < open.base:
            raise RpcError(
                INVALID_PARAMS,
                f"The outputs of the reply {given.id!r} before {open.base} were collected.",
            )
        # what the proxy hook has is collected, which lets the stream go on
        taken = min(given.after, open.base + len(open.outputs)) - open.base
        if taken:
            open.pending -= sum(_size(output) for output in open.outputs[:taken])
            del open.outputs[:taken]
            open.base += taken
            open.collected.set()
        with anyio.move_on_after(self._poll_seconds):
            while not open.outputs and not open.done:
                open.changed = anyio.Event()
                await open.changed.wait()
        open.touched = anyio.current_time()
        if not open.outputs and open.error is not None:
            raise open.error
        if open.done and not open.outputs and open.error is None:
            # everything was collected, so the stream is let go of
            self._open.pop(given.id, None)
        return NextReleaseResult(
            position=open.base + len(open.outputs), outputs=list(open.outputs)
        )

    def _find(self, id: str) -> _Open:
        open = self._open.get(id)
        if open is None:
            raise RpcError(
                UNKNOWN_STREAM,
                f"There is no reply stream {id!r}: it was never begun, or has ended or expired.",
            )
        open.touched = anyio.current_time()
        return open

    async def _collect(self, open: _Open) -> None:
        # the stream's output, kept for the proxy hook to collect, and no more
        # of it than is allowed to wait, so that a slow hook holds the stream
        try:
            async for output in open.stream:
                while open.pending >= self._max_pending_bytes and not open.done:
                    open.collected = anyio.Event()
                    await open.collected.wait()
                kept = _output(output)
                open.outputs.append(kept)
                open.pending += _size(kept)
                open.changed.set()
        except Exception as ex:
            open.error = self._not_judged(ex, open.begun.id)
        finally:
            open.done = True
            open.changed.set()

    async def _sweep(self) -> None:
        # a stream no call has named for a while is one its proxy hook forgot,
        # as when the proxy restarted, and is aborted
        while True:
            await anyio.sleep(min(self._expiry / 4, 5.0))
            now = anyio.current_time()
            for id, open in list(self._open.items()):
                if now - open.touched > self._expiry:
                    await self.abort(StreamParams(id=id))


def _output(output: Output) -> StreamOutput:
    if isinstance(output, Release):
        return ReleaseOutput(data=base64.b64encode(output.data).decode("ascii"))
    if isinstance(output, ErrorEvent):
        return ErrorEventOutput(
            data=base64.b64encode(output.data).decode("ascii"),
            not_judged=output.not_judged,
        )
    if isinstance(output, Finished):
        return FinishedOutput()
    assert isinstance(output, Refuse)
    return RefuseOutput(
        status=output.status,
        message=output.message,
        sentinel=(
            SentinelField.model_validate(output.sentinel)
            if output.sentinel is not None
            else None
        ),
    )


def _size(output: StreamOutput) -> int:
    return (
        len(output.data) if isinstance(output, ReleaseOutput | ErrorEventOutput) else 0
    )
