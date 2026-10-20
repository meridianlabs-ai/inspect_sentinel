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

    Envoy opens one stream for each HTTP request on a route that has the filter, and sends the request's headers and body, then the reply's headers, body and trailers. The shim reads the request whole, puts it to the handler, and sends it on as it was, as the handler replaced it, or not at all, answering the client in the provider's own error format. It holds every chunk of the reply, saying nothing to Envoy until it has the whole reply, as Envoy then has sent the client nothing, the status included. It then puts the reply to the handler and sends it on, replaced, or refuses it with a real status. Envoy's route says which API the request is written in and which provider it is for, in the gRPC metadata `sentinel-format` and `sentinel-provider`.

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
            max_body_bytes=self._max_body_bytes,
            shown=self._shown,
            metadata=context.invocation_metadata() or (),
        )
        async for message in request_iterator:
            for response in await exchange.on(message):
                yield response
            if exchange.finished:
                return

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
        max_body_bytes: int,
        shown: Collection[str],
        metadata: Collection[tuple[str, str | bytes]],
    ) -> None:
        self._handler = handler
        self._judged = judged
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
        self._held = bytearray()
        self._replied = False
        self._passing = False
        self.finished = False

    async def on(self, message: pb.ProcessingRequest) -> list[pb.ProcessingResponse]:
        """The answers to one message from Envoy, which are none while a reply is held."""
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
            # with trailers, their arrival is the only sign the body has ended
            answers = [] if self._replied else await self._reply(end_of_stream=False)
            if not self.finished:
                answers.append(
                    pb.ProcessingResponse(response_trailers=pb.TrailersResponse())
                )
                self.finished = True
            return answers
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
            # out with the first answer about the body, which the shim holds.
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
        if self._replied:
            # Envoy may send more after the end it was answered at
            return []
        self._held += body.body
        if len(self._held) > self._max_body_bytes:
            self._replied = True
            self.finished = True
            # the reply isn't read, as the endpoint for proxies doesn't read a
            # call larger than it takes
            await self._handler.reply_not_judged(self._id, "unreadable")
            return [
                self._refuse(
                    502,
                    f"The provider's reply is larger than the {self._max_body_bytes} bytes the sidecar holds, so it was not judged.",
                )
            ]
        # Envoy 1.39 sends an empty chunk now and then that isn't the end: only
        # `end_of_stream`, or the trailers, says the body has ended
        if not body.end_of_stream:
            return []
        answers = await self._reply(end_of_stream=True)
        self.finished = True
        return answers

    async def _reply(self, *, end_of_stream: bool) -> list[pb.ProcessingResponse]:
        self._replied = True
        answer = await self._judge(
            self._handler.reply(self._reply_of(bytes(self._held))),
            "reply",
            reply=True,
        )
        if isinstance(answer, Refuse):
            self.finished = True
            return [self._refusal(answer)]
        body = answer.body if isinstance(answer, Replace) else bytes(self._held)
        return [_streamed(body, end_of_stream=end_of_stream)]

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
