from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from typing import Any, TypeVar, cast, get_args

import grpc

from .._errors import bounded, error_body, refusal_for
from .._exchange import Answer, Pass, Refuse, Replace, Reply, Request, succeeded
from .._handler import Handler
from .._reply_stream import ErrorEvent, Release, ReplyStart, ReplyStream
from .._wire import Format
from ._metadata import FORMAT_METADATA, INSPECT_HEADERS, PROVIDER_METADATA
from ._proto.envoy.config.core.v3 import base_pb2 as core
from ._proto.envoy.service.ext_proc.v3 import external_processor_pb2 as pb
from ._proto.envoy.service.ext_proc.v3 import external_processor_pb2_grpc as rpc
from ._proto.envoy.type.v3 import http_status_pb2 as http_status

logger = logging.getLogger(__name__)

_FORMATS: frozenset[str] = frozenset(get_args(Format))

# a provider's error type for a status, as each writes it; any other client
# error is an invalid request, and any other server error the provider's own
T = TypeVar("T")


class EnvoyShim(rpc.ExternalProcessorServicer):
    """Envoy's external processor: the sidecar's handler, behind Envoy's `ext_proc` filter.

    Envoy opens one stream for each HTTP request on a route that has the filter, and sends the request's headers and body, then the reply's headers, body and trailers. The shim reads the request whole, puts it to the handler, and sends it on as it was, as the handler replaced it, or not at all, answering the client in the provider's own error format. It hands the reply to the handler's reply stream as it arrives, and sends Envoy what the stream gives: the whole reply once it is judged, or, with the handler's `release` set to `early`, text as it arrives and each call once it is judged. While nothing has gone out, Envoy has sent the client nothing, the status included, so a refusal has a real status; after that, a refusal is an error event in the stream. Envoy's route says which API the request is written in and which provider it is for, in the gRPC metadata `sentinel-format` and `sentinel-provider`.

    A request or reply that can't be judged is refused: one that can't be read, one larger than the shim holds, one the sentinel fails on or doesn't finish within the handler's time limit, and one with a content encoding other than `identity`. A judgment goes on when Envoy stops waiting for it, as when the client disconnects, so that the sidecar's records and the run's state hold what was decided.

    Args:
        handler: Judges each request and each reply.
        max_body_bytes: The largest request or reply the shim holds.
        headers: The request headers, besides Inspect's own, that the sentinel is shown, by lowercase name. A credential is never shown, whatever is listed.
    """

    def __init__(
        self,
        handler: Handler,
        *,
        max_body_bytes: int,
        headers: Collection[str] = (),
    ) -> None:
        self._handler = handler
        self._max_body_bytes = max_body_bytes
        self._shown = (
            frozenset(INSPECT_HEADERS) | {name.lower() for name in headers}
        ) - _CREDENTIALS
        # judgments that outlive their streams, kept from being collected
        self._judging: set[asyncio.Task[Any]] = set()

    async def Process(  # noqa: N802 the name of the gRPC method
        self,
        request_iterator: AsyncIterator[pb.ProcessingRequest],
        context: grpc.aio.ServicerContext[pb.ProcessingRequest, pb.ProcessingResponse],
    ) -> AsyncIterator[pb.ProcessingResponse]:
        """Process one HTTP exchange through Envoy."""
        exchange = _Exchange(
            self._handler,
            judged=self._judged,
            spawn=self._spawn,
            max_body_bytes=self._max_body_bytes,
            shown=self._shown,
            metadata=context.invocation_metadata() or (),
        )
        # Envoy's messages are read beside the answers, which a reply released
        # as it arrives has ready whenever its sentinel decides
        pump = asyncio.ensure_future(exchange.read(request_iterator))
        try:
            while (response := await exchange.next()) is not None:
                yield response
        finally:
            pump.cancel()
            await asyncio.wait({pump})
            await exchange.close()

    def _spawn(self, work: Awaitable[None]) -> None:
        # work that outlives the stream that started it, as a judgment does
        task = asyncio.ensure_future(work)
        self._judging.add(task)
        task.add_done_callback(self._judging.discard)

    async def _judged(
        self, judging: Awaitable[T], orphaned: Callable[[Exception], object]
    ) -> T:
        # The judgment runs as a task of its own, which a stream that ends
        # doesn't cancel: the sentinel finishes and its decision is kept, as
        # when a proxy hook stops waiting for the endpoint for proxies.
        task = asyncio.ensure_future(judging)
        self._judging.add(task)
        task.add_done_callback(self._judging.discard)
        try:
            # Cancelling the wait leaves the task running, as `asyncio.shield`
            # would, without the report of its failure `shield` adds once its
            # caller is cancelled (Python 3.14).
            await asyncio.wait({task})
        except asyncio.CancelledError:
            # Envoy stopped waiting. What the judgment raises later goes to
            # `orphaned`, as it would have been handled here.
            def retrieved(done: asyncio.Future[T]) -> None:
                if not done.cancelled():
                    raised = done.exception()
                    if isinstance(raised, Exception):
                        orphaned(raised)

            task.add_done_callback(retrieved)
            raise
        return task.result()


# never shown to a sentinel, whatever the operator lists
_CREDENTIALS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "cookie",
    }
)


class _Exchange:
    """One HTTP request and its reply, as one stream from Envoy."""

    def __init__(
        self,
        handler: Handler,
        *,
        judged: Callable[
            [Awaitable[Answer], Callable[[Exception], object]], Awaitable[Answer]
        ],
        spawn: Callable[[Awaitable[None]], None],
        max_body_bytes: int,
        shown: Collection[str],
        metadata: Collection[tuple[str, str | bytes]],
    ) -> None:
        self._handler = handler
        self._judged = judged
        self._spawn = spawn
        self._max_body_bytes = max_body_bytes
        self._shown = shown
        given = {
            key.lower(): value.decode() if isinstance(value, bytes) else value
            for key, value in metadata
        }
        format = given.get(FORMAT_METADATA)
        self._format: Format | None = (
            cast("Format", format) if format in _FORMATS else None
        )
        self._provider = given.get(PROVIDER_METADATA, "")
        self._id = uuid.uuid4().hex
        self._headers: dict[str, str] = {}
        self._model = ""
        self._status = 0
        self._content_type = ""
        self._declared_length = False
        self._request_encoding = "identity"
        self._replied = False
        self._passing = False
        # the reply, judged as it arrives, and whether its body ends at trailers
        self._stream: ReplyStream | None = None
        self._body_ended = False
        self._trailers = False
        # the answers for Envoy, in order, which end with None
        self._answers: asyncio.Queue[pb.ProcessingResponse | None] = asyncio.Queue(1)
        self.finished = False

    async def read(self, messages: AsyncIterator[pb.ProcessingRequest]) -> None:
        """Read Envoy's messages, answering each, until the exchange has finished."""
        try:
            await self._read(messages)
        except Exception:
            # a stream that can't go on ends, rather than leave Envoy waiting
            logger.error("The exchange %r failed.", self._id, exc_info=True)
            await self.close()
            await self._answers.put(None)

    async def _read(self, messages: AsyncIterator[pb.ProcessingRequest]) -> None:
        async for message in messages:
            for answer in await self.on(message):
                await self._answers.put(answer)
            if self.finished:
                await self._answers.put(None)
                break
        if self.finished:
            return
        # Envoy has nothing more to send. A reply whose body has ended is
        # still being answered; otherwise there is nothing left to answer,
        # and a reply cut short is aborted, as when Envoy stops waiting.
        if self._stream is not None and self._body_ended:
            return
        await self.close()
        await self._answers.put(None)

    async def next(self) -> pb.ProcessingResponse | None:
        """The next answer for Envoy, or None once the exchange has finished."""
        return await self._answers.get()

    async def close(self) -> None:
        """Stop the reply's stream if it hasn't ended, as when Envoy has stopped waiting."""
        if self._stream is not None:
            await self._stream.abort()

    async def _answer(self, answer: pb.ProcessingResponse) -> None:
        await self._answers.put(answer)
        if self.finished:
            await self._answers.put(None)

    async def on(self, message: pb.ProcessingRequest) -> list[pb.ProcessingResponse]:
        """The answers to one message from Envoy, which are none while a reply is held."""
        if self.finished:
            # Envoy may send more after the end it was answered at
            return []
        kind = message.WhichOneof("request")
        if kind == "request_headers":
            return await self._request_headers(message.request_headers)
        if kind == "request_body":
            return [await self._request_body(message.request_body.body)]
        if kind == "response_headers":
            return await self._response_headers(message.response_headers)
        if kind == "response_body":
            return await self._response_body(message.response_body)
        if kind == "response_trailers":
            # with trailers, their arrival is the only sign the body has ended,
            # and they are answered after the body
            if self._stream is not None:
                self._trailers = self._body_ended = True
                await self._stream.end()
                return []
            self.finished = True
            return [pb.ProcessingResponse(response_trailers=pb.TrailersResponse())]
        # request trailers are not asked for, and nothing else needs an answer
        return []

    async def _request_headers(
        self, headers: pb.HttpHeaders
    ) -> list[pb.ProcessingResponse]:
        if self._format is None:
            return [
                self._refuse(
                    500,
                    f"Envoy's route doesn't say which API the request is written in, in the gRPC metadata {FORMAT_METADATA}.",
                )
            ]
        received = _read(headers.headers)
        self._headers = {
            name: value for name, value in received.items() if name in self._shown
        }
        self._request_encoding = received.get("content-encoding", "identity").lower()
        # Inspect's headers are the sidecar's, and never reach the provider
        answer = pb.ProcessingResponse(
            request_headers=pb.HeadersResponse(
                response=pb.CommonResponse(
                    header_mutation=pb.HeaderMutation(
                        remove_headers=list(INSPECT_HEADERS)
                    )
                )
            )
        )
        if not headers.end_of_stream:
            self._declared_length = "content-length" in received
            return [answer]
        # a request without a body is put to the handler as one with an empty
        # body, which no API reads
        judged = await self._request_body(b"")
        if judged.WhichOneof("response") == "immediate_response":
            return [judged]
        if judged.request_body.response.HasField("body_mutation"):
            # the handler replaced a request that has no body to replace,
            # which Envoy can't be told to send
            return [
                self._refuse(
                    500,
                    "The sidecar replaced a request without a body, which it can't send.",
                )
            ]
        return [answer]

    async def _request_body(self, body: bytes) -> pb.ProcessingResponse:
        if self._request_encoding != "identity":
            # Handed over as it is, the handler can't read it, and the eval is
            # told so; why is kept here.
            logger.warning(
                "The request %r is in content encoding %r, which isn't read.",
                self._id,
                self._request_encoding,
            )
        self._model = _model(body)
        request = Request(
            id=self._id,
            provider=self._provider,
            format=self._format,
            model=self._model,
            user=None,
            body=body,
            headers=self._headers,
        )
        answer = await self._judge(
            self._handler.request(request), "request", reply=False
        )
        if isinstance(answer, Refuse):
            return self._refusal(answer)
        if isinstance(answer, Pass):
            return pb.ProcessingResponse(
                request_body=pb.BodyResponse(response=pb.CommonResponse())
            )
        # Envoy keeps the request's content-length as it was, so a replaced
        # body that declared one declares its own
        mutation = pb.HeaderMutation(
            set_headers=[_header("content-length", str(len(answer.body)))]
            if self._declared_length
            else []
        )
        return pb.ProcessingResponse(
            request_body=pb.BodyResponse(
                response=pb.CommonResponse(
                    header_mutation=mutation,
                    body_mutation=pb.BodyMutation(body=answer.body),
                )
            )
        )

    async def _response_headers(
        self, headers: pb.HttpHeaders
    ) -> list[pb.ProcessingResponse]:
        received = _read(headers.headers)
        self._status = int(received.get(":status", "0") or 0)
        self._content_type = received.get("content-type", "")
        if not succeeded(self._status):
            return await self._provider_error(headers.end_of_stream)
        encoding = received.get("content-encoding", "identity").lower()
        if encoding != "identity":
            # Envoy's route asks the provider for an uncompressed reply, so a
            # compressed one is a provider that didn't listen
            await self._handler.reply_not_judged(self._id, "unreadable")
            return [
                self._refuse(
                    502,
                    f"The provider's reply came in content encoding {encoding!r}, which isn't judged.",
                )
            ]
        if not headers.end_of_stream:
            # Answering the headers sends the client nothing: the status goes
            # out with the first answer about the body.
            self._stream = self._handler.reply_stream(
                ReplyStart(
                    id=self._id,
                    provider=self._provider,
                    format=self._format,
                    model=self._model,
                    user=None,
                    status=self._status,
                    content_type=self._content_type,
                ),
                max_held_bytes=self._max_body_bytes,
            )
            # the stream's judging outlives Envoy's stream, as a judgment does
            self._spawn(self._stream.run())
            self._spawn(self._deliver(self._stream))
            return [
                pb.ProcessingResponse(
                    response_headers=pb.HeadersResponse(response=pb.CommonResponse())
                )
            ]
        # a reply without a body, which the handler is still shown, so that
        # it lets go of the request
        answer = await self._judge(
            self._handler.reply(self._reply_of(b"")), "reply", reply=True
        )
        self.finished = True
        if isinstance(answer, Refuse):
            return [self._refusal(answer)]
        if isinstance(answer, Replace):
            # a body for a reply that has none, which Envoy can't be told to
            # send on a stream whose headers have no body after them
            return [
                self._refuse(
                    500,
                    "The sidecar replaced a reply without a body, which it can't send.",
                )
            ]
        return [
            pb.ProcessingResponse(
                response_headers=pb.HeadersResponse(response=pb.CommonResponse())
            )
        ]

    async def _provider_error(self, end_of_stream: bool) -> list[pb.ProcessingResponse]:
        # A provider's error goes on unjudged, and the handler reads nothing
        # of it: it is told at the headers, and the body passes through as it
        # comes, whatever its size or encoding.
        self._replied = self._passing = True
        answer = await self._judge(
            self._handler.reply(self._reply_of(b"")), "reply", reply=True
        )
        if isinstance(answer, Refuse):
            return [self._refusal(answer)]
        self.finished = end_of_stream
        return [
            pb.ProcessingResponse(
                response_headers=pb.HeadersResponse(response=pb.CommonResponse())
            )
        ]

    async def _response_body(self, body: pb.HttpBody) -> list[pb.ProcessingResponse]:
        if self._passing:
            self.finished = body.end_of_stream
            return [_streamed(body.body, end_of_stream=body.end_of_stream)]
        if self._stream is None:
            return []
        await self._stream.feed(body.body)
        # Envoy 1.39 sends an empty chunk now and then that isn't the end: only
        # `end_of_stream`, or the trailers, says the body has ended
        if body.end_of_stream:
            self._body_ended = True
            await self._stream.end()
        return []

    async def _deliver(self, stream: ReplyStream) -> None:
        # what the reply stream gives, as Envoy's answers about the body
        try:
            async for output in stream:
                if isinstance(output, Release):
                    await self._answers.put(_streamed(output.data, end_of_stream=False))
                elif isinstance(output, Refuse):
                    await self._answer(self._refusal(output))
                    return
                else:
                    # the end of the body, or an error event in place of the
                    # rest of it; trailers are answered after the body
                    data = output.data if isinstance(output, ErrorEvent) else b""
                    answers = [_streamed(data, end_of_stream=not self._trailers)]
                    if self._trailers:
                        answers.append(
                            pb.ProcessingResponse(
                                response_trailers=pb.TrailersResponse()
                            )
                        )
                    self.finished = True
                    for answer in answers[:-1]:
                        await self._answers.put(answer)
                    await self._answer(answers[-1])
                    return
        except Exception as ex:
            await self._answer(self._refusal(self._not_judged(ex, "reply", reply=True)))

    def _reply_of(self, body: bytes) -> Reply:
        return Reply(
            id=self._id,
            provider=self._provider,
            format=self._format,
            model=self._model,
            user=None,
            status=self._status,
            content_type=self._content_type,
            body=body,
        )

    async def _judge(
        self, judging: Awaitable[Answer], what: str, *, reply: bool
    ) -> Answer:
        def not_judged(ex: Exception) -> Refuse:
            return self._not_judged(ex, what, reply=reply)

        try:
            return await self._judged(judging, not_judged)
        except Exception as ex:
            return not_judged(ex)

    def _not_judged(self, ex: Exception, what: str, *, reply: bool) -> Refuse:
        return refusal_for(ex, what, self._id, reply=reply)

    def _refuse(self, status: int, message: str) -> pb.ProcessingResponse:
        return self._refusal(Refuse(status, message))

    def _refusal(self, refused: Refuse) -> pb.ProcessingResponse:
        self.finished = True
        # a refusal is an error, as the endpoint for proxies bounds it
        status = bounded(refused.status)
        error = error_body(self._format, status, refused.message)
        if refused.sentinel is not None:
            error["sentinel"] = dict(refused.sentinel)
        return pb.ProcessingResponse(
            immediate_response=pb.ImmediateResponse(
                status=http_status.HttpStatus(
                    code=cast("http_status.StatusCode.ValueType", status)
                ),
                headers=pb.HeaderMutation(
                    set_headers=[_header("content-type", "application/json")]
                ),
                body=json.dumps(error).encode(),
                details="sentinel_refused",
            )
        )


def _streamed(body: bytes, *, end_of_stream: bool) -> pb.ProcessingResponse:
    return pb.ProcessingResponse(
        response_body=pb.BodyResponse(
            response=pb.CommonResponse(
                body_mutation=pb.BodyMutation(
                    streamed_response=pb.StreamedBodyResponse(
                        body=body, end_of_stream=end_of_stream
                    )
                )
            )
        )
    )


def _read(headers: core.HeaderMap) -> dict[str, str]:
    # Envoy sends a value as bytes in `raw_value`, or as text in `value`
    return {
        header.key.lower(): (
            header.raw_value.decode("latin-1") if header.raw_value else header.value
        )
        for header in headers.headers
    }


def _header(name: str, value: str) -> core.HeaderValueOption:
    return core.HeaderValueOption(
        header=core.HeaderValue(key=name, raw_value=value.encode()),
        append_action=core.HeaderValueOption.OVERWRITE_IF_EXISTS_OR_ADD,
    )


def _model(body: bytes) -> str:
    # the model the request names, as the proxy would name it; a body that
    # can't be read names none, and the handler says why it can't be read
    try:
        parsed: object = json.loads(body)
    except ValueError:
        return ""
    if isinstance(parsed, dict):
        model = cast("Mapping[str, object]", parsed).get("model")
        if isinstance(model, str):
            return model
    return ""


_MESSAGE_OVERHEAD = 1024 * 1024
"""Room in a gRPC message for what Envoy sends beside a body."""


def envoy_server(
    handler: Handler,
    *,
    host: str,
    port: int,
    max_body_bytes: int,
    headers: Collection[str] = (),
) -> grpc.aio.Server:
    """A gRPC server for Envoy's `ext_proc` filter, which puts what Envoy hands over to a handler.

    It listens without TLS, as Envoy's cluster for the sidecar is set up by `envoy_config()`: the sidecar is reached only from Envoy. Start it with `await server.start()`, in an asyncio event loop.

    Args:
        handler: Judges each request and each reply.
        host: The address to listen on.
        port: The port to listen on.
        max_body_bytes: The largest request or reply the shim holds. gRPC's own limit, 4 MiB, is raised to fit it, since Envoy sends a request's body in one message.
        headers: The request headers, besides Inspect's own, that the sentinel is shown, by lowercase name.
    """
    limit = max_body_bytes + _MESSAGE_OVERHEAD
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", limit),
            ("grpc.max_send_message_length", limit),
        ]
    )
    rpc.add_ExternalProcessorServicer_to_server(
        EnvoyShim(handler, max_body_bytes=max_body_bytes, headers=headers), server
    )
    # gRPC takes an IPv6 address in brackets, where uvicorn takes it bare
    bound = f"[{host}]" if ":" in host and not host.startswith("[") else host
    server.add_insecure_port(f"{bound}:{port}")
    return server
