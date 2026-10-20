"""A guardrail for the LiteLLM proxy that calls a remote Inspect Sentinel endpoint over HTTPS.

LiteLLM's `guardrails` setting names the class as `inspect_sentinel_litellm.LiteLLMGuardrail`. LiteLLM creates one at startup, then calls it with each model request before the request goes to a provider, and with the provider's reply before any of the reply goes to the caller. Each is put to the sidecar as a JSON-RPC 2.0 call, `judge_request` or `judge_reply`, whose result is carried out: pass, replace or refuse. A streamed reply is handed over as it arrives, as a reply stream, and the caller is sent what the sidecar releases of it. Inspect Sentinel does not run in LiteLLM: only this guardrail does, and the sentinels run in the sidecar it calls.

The class runs inside LiteLLM's process, so this package stands alone: it imports nothing from `inspect_sentinel`, and needs only LiteLLM and what LiteLLM's proxy has. The calls it makes and the results it reads are described by `proxy_endpoint.schema.json` in `inspect_sentinel`'s sidecar.

It is set in LiteLLM's config, beside `guardrail`, `mode` and `default_on`:

- `api_base`: the sidecar's address for proxies, as `https://<sentinel host>:<port>`. Required.
- `unreachable_fallback`: `fail_closed` refuses a call when the sidecar gives no usable answer, and `fail_open` lets it through. `fail_closed` unless set, which is LiteLLM's own default.
- `timeout`: how many seconds the sidecar may take to answer. No limit unless set, because a sentinel may ask a person.
- `sidecar_headers`: request headers the sidecar is shown besides Inspect's own, as a list of names, or prefixes ending in `*`. None unless set.
- `max_reply_bytes`: the largest reply that is held to be judged. 32 MiB unless set.

Which requests reach the class is LiteLLM's own choice: `default_on`, or a request or key that names the guardrail.

The guardrail also adds a route to LiteLLM's app, so that an Inspect eval reaches the sidecar's endpoint for evals at LiteLLM's own address: everything under `/sentinel/` is passed on to the sidecar, behind LiteLLM's own check of the caller's key.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import json
import logging
import re
import sys
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import aiohttp
from fastapi import Depends, HTTPException, Request, Response
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
    all_litellm_params,
)
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
        CallTypesLiteral,
    )

logger = logging.getLogger("inspect_sentinel_litellm")

_RESPONSES = "openai-responses"
_CHAT = "openai-chat"
# LiteLLM's name for the route a request came in on, and what a body on that
# route is written in whichever provider serves the model: whose API, and which
_APIS: dict[str, tuple[str, str]] = {
    "anthropic_messages": ("anthropic", "anthropic-messages"),
    "acompletion": ("openai", "openai-chat"),
    "completion": ("openai", "openai-chat"),
    "aresponses": ("openai", _RESPONSES),
    "responses": ("openai", _RESPONSES),
}
# the headers an Inspect eval puts on a model request: which sample it is for,
# and an id for the request. The sidecar is always shown them, unread, and the
# provider never is.
_INSPECT_HEADERS = frozenset({"x-inspect-context", "x-irid"})
# never shown to the sidecar, whatever is listed. LiteLLM leaves the caller's
# own key out of the headers it keeps, but not a provider's key that a caller
# sent along.
_CREDENTIALS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-litellm-api-key",
    }
)
# where what was told to the sidecar about a request is kept until its reply:
# in LiteLLM's own record of the request, which lasts as long as the request
# and is never sent to a provider
_KEPT = "inspect_sentinel"
_MAX_REPLY_BYTES = 32 * 1024 * 1024
# where the sidecar serves the methods this calls, and the version of them
_PATH = "/proxy/v1"
# a call to `next_release` waits up to 20 seconds for output, and is given as
# long again to answer
_POLL_TIMEOUT = aiohttp.ClientTimeout(total=40)
# what an eval reaches through LiteLLM: everything under it is the sidecar's
# endpoint for evals, and the paths an eval calls first
_EVAL_PREFIX = "/sentinel"
_EVAL_PATHS = ("/sentinel/info", "/sentinel/v1")
# one segment of a path under it that is passed on: never `.` or `..`
_SEGMENT = re.compile(r"(?!\.{1,2}$)[A-Za-z0-9._~-]+")
# a request's fields that carry a credential or say where a call goes, which
# the sidecar is never shown, besides every field LiteLLM keeps for itself
_NOT_SHOWN = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "organization",
        "headers",
        "extra_headers",
        "client_secret",
        "azure_ad_token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "vertex_credentials",
    }
)
_JSON = "application/json"
_LINE_END = re.compile(r"\r\n|\r|\n")
_STREAM = "text/event-stream"
# A request's own fields that LiteLLM rewrites the request from after this
# guardrail has run, so that the provider would be sent what the sidecar
# never judged. The provider's SDK merges `extra_body` over the request's
# other fields. LiteLLM's `completion()` inserts the continue messages, loads a
# stored prompt for the prompt fields, adds `litellm_system_prompt`, writes a
# custom prompt template from the template fields, and folds the system
# message into a user message when told the model has none. A client that
# calls through a provider's SDK sends none of them: the SDK merges its own
# `extra_body` before sending.
_REWRITTEN_FROM = frozenset(
    {
        "extra_body",
        "ensure_alternating_roles",
        "user_continue_message",
        "assistant_continue_message",
        "prompt_id",
        "prompt_variables",
        "prompt_label",
        "prompt_version",
        "prompt_environment",
        "langfuse_prompt_version",
        "litellm_system_prompt",
        "roles",
        "initial_prompt_value",
        "final_prompt_value",
        "bos_token",
        "eos_token",
        "custom_prompt_dict",
        "supports_system_message",
    }
)
# the code a failed Responses stream gives for each status of a refusal, as
# LiteLLM's own failed event gives them; any other client error is an
# invalid request, and only a server error is one a client tries again
_ERROR_CODES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "request_timeout",
    429: "rate_limit_exceeded",
}


_GATEWAY_TOOLS = (
    "A request with an MCP tool that LiteLLM's MCP gateway would run can't be "
    "judged, so it is refused."
)


def _gateway_runs(tools: object) -> bool:
    # whether LiteLLM's MCP gateway would take the request's MCP tools, by
    # the test LiteLLM itself makes before handing them to it, on every route
    if not isinstance(tools, list) or not tools:
        return False
    listed = cast("list[Any]", tools)
    try:
        module = importlib.import_module(
            "litellm.responses.mcp.litellm_proxy_mcp_handler"
        )
        claims: Any = module.LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway
    except (ImportError, AttributeError):
        # a LiteLLM that doesn't make the test might hand any MCP tool to it
        for tool in listed:
            if (
                isinstance(tool, dict)
                and cast("dict[str, Any]", tool).get("type") == "mcp"
            ):
                return True
        return False
    return bool(claims(tools=listed))


class _Written(BaseModel):
    """An event as the sidecar wrote it, every field kept.

    Every LiteLLM version writes an object for the caller as `data:` and its JSON, while bytes are passed on as they are only by later versions. LiteLLM's own classes are not used, since it writes some of them without a field the class doesn't have.
    """

    model_config = ConfigDict(extra="allow")


class SidecarError(Exception):
    """The sidecar could not be asked, or its answer could not be read or carried out."""


class LiteLLMGuardrail(CustomGuardrail):
    """Puts each of LiteLLM's model requests and replies to a sentinel sidecar.

    The requests it reads are those to LiteLLM's `/v1/messages`, `/chat/completions` and `/responses` routes. A request to another route goes on unread.

    Args:
        api_base: The sidecar's address for proxies. The path of the methods this calls is added to it.
        unreachable_fallback: `fail_closed` refuses a call when the sidecar gives no usable answer, and `fail_open` lets it through.
        timeout: How many seconds the sidecar may take to answer. No limit when None.
        sidecar_headers: Request headers the sidecar is shown besides Inspect's own, as names, or prefixes ending in `*`. A comma-separated string is read as a list.
        max_reply_bytes: The largest reply that is held to be judged.
        **kwargs: LiteLLM's own settings for a guardrail, which it passes to every one.

    Raises:
        ValueError: A setting makes no sense, which stops LiteLLM loading the guardrail.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        unreachable_fallback: str = "fail_closed",
        timeout: float | None = None,
        sidecar_headers: str | Iterable[str] | None = None,
        max_reply_bytes: int = _MAX_REPLY_BYTES,
        **kwargs: Any,
    ) -> None:
        if not api_base:
            raise ValueError("api_base must say where the sentinel sidecar listens.")
        if unreachable_fallback not in ("fail_closed", "fail_open"):
            raise ValueError(
                f"unreachable_fallback is 'fail_closed' or 'fail_open', not {unreachable_fallback!r}."
            )
        # a sentinel judges tool calls in a reply, so one that is shown only
        # requests would look installed and decide nothing
        mode = kwargs.get("event_hook")
        modes = (
            {mode} if isinstance(mode, str) else set(cast("Iterable[Any]", mode or ()))
        )
        if not {"pre_call", "post_call"} <= {
            str(getattr(m, "value", m)) for m in modes
        }:
            raise ValueError("mode must list both pre_call and post_call.")
        self._base = api_base.rstrip("/")
        self._url = self._base + _PATH
        self._let_through = unreachable_fallback == "fail_open"
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._shown_headers = _listed(sidecar_headers)
        self._max_reply_bytes = max_reply_bytes
        # made on first use, when LiteLLM's event loop is running
        self._session: aiohttp.ClientSession | None = None
        super().__init__(**kwargs)  # pyright: ignore[reportUnknownMemberType]
        # LiteLLM's proxy makes its guardrails once its app exists. Anywhere
        # else, such as a test that makes the class, there is no app to add to.
        if "litellm.proxy.proxy_server" in sys.modules:
            self.serve_eval_routes()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> dict[str, Any]:
        """Put a request to the sidecar. LiteLLM sends on `data`, which is changed in place when the sidecar replaces the request.

        Raises:
            HTTPException: The request is refused, by the sidecar, because it sets a field LiteLLM rewrites it from, such as `extra_body`, or because it has an MCP tool that LiteLLM's MCP gateway would run, or the sidecar gave no usable answer and `unreachable_fallback` is `fail_closed`. LiteLLM answers the caller with it in the error format of the route.
        """
        _withhold_inspect_headers(data)
        api = _APIS.get(call_type)
        if api is None:
            return data
        rewritten = sorted(
            # `False` counts, as `supports_system_message: false` is one
            name
            for name in _REWRITTEN_FROM
            if data.get(name) not in (None, "", {}, [])
        )
        if rewritten:
            # Refused whatever `unreachable_fallback` says, since nothing could
            # judge what the provider would be sent
            named = ", ".join(f"`{name}`" for name in rewritten)
            raise HTTPException(
                400, f"A request that sets {named} can't be judged, so it is refused."
            )
        if _gateway_runs(data.get("tools")):
            # LiteLLM's MCP gateway would run the model's calls to these tools
            # itself, and send the provider requests no guardrail is shown,
            # so the sidecar could judge neither. Refused whatever
            # `unreachable_fallback` says.
            raise HTTPException(400, _GATEWAY_TOOLS)
        try:
            record = _record(data)
            sent = _body(data, record)
            shown, as_sent = _readable_request(api[1], sent)
            headers = _headers(record)
            told: dict[str, Any] = {
                # LiteLLM's own id for the call is the caller's to set, so two
                # requests in flight could share it
                "id": str(uuid.uuid4()),
                "provider": api[0],
                "format": api[1],
                "model": str(data.get("model") or ""),
                "user": getattr(user_api_key_dict, "user_id", None),
            }
            answer = await self._ask(
                "judge_request",
                told
                | {
                    "headers": {
                        name: value
                        for name, value in headers.items()
                        if self._shows(name)
                    },
                    "body": shown,
                },
            )
            if answer["action"] == "replace":
                _replace_request(data, sent, _restored(answer.get("body"), as_sent))
        except SidecarError:
            self._unanswered()
            return data
        record[_KEPT] = told
        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Put a reply that was not streamed to the sidecar. LiteLLM sends the caller the reply returned, or the reply as it is for None.

        Raises:
            HTTPException: The reply is refused, or the sidecar gave no usable answer and `unreachable_fallback` is `fail_closed`.
        """
        told = _told(data)
        if told is None:
            return None
        try:
            answer = await self._ask_about_reply(told, _JSON, _whole(response))
            if answer["action"] == "replace":
                return _replaced_whole(response, _replacement(answer))
        except SidecarError:
            self._unanswered()
        return None

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """Hand a streamed reply to the sidecar as it arrives, and send on what the sidecar releases.

        The reply goes to the sidecar's reply stream as LiteLLM yields it, and what the sidecar releases is yielded as it is released: the whole reply once judged, or, where the sidecar releases replies early, text as it arrives and each call once judged. LiteLLM sends the caller nothing, the status included, until this yields.

        A reply refused while nothing has gone out is answered with an error event in place of the reply, which LiteLLM sends the caller as an error under the refusal's status. Once something has gone out, a refusal is an error event in the stream, which the provider's SDK raises on. Where the sidecar gives no usable answer, `unreachable_fallback` decides while nothing has gone out; after that, the reply ends with an error event, since what went out can't be matched to what LiteLLM yielded.
        """
        chunks = cast("AsyncIterator[Any]", response)
        told = _told(request_data)
        if told is None:
            async for chunk in chunks:
                yield chunk
            return
        relay = _Relay(self, told, chunks)
        try:
            async for chunk in relay.released():
                yield chunk
        finally:
            await relay.close()

    async def aclose(self) -> None:
        """Close the connection to the sidecar."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def serve_eval_routes(self) -> None:
        """Pass everything under `/sentinel/` in LiteLLM's app on to the sidecar, behind LiteLLM's check of the caller's key.

        An Inspect eval knows only the base URL of the model API it calls, which here is LiteLLM's, and reaches the sidecar's endpoint for evals there. LiteLLM's own routes would otherwise answer these paths first, or forward them to a provider, so the route is put ahead of them. A key that may call LiteLLM's OpenAI routes may call these too.

        LiteLLM's proxy calls this when it makes the guardrail. Calling it again adds nothing.
        """
        from litellm.proxy._types import LiteLLMRoutes
        from litellm.proxy.auth.user_api_key_auth import (  # pyright: ignore[reportMissingTypeStubs]
            user_api_key_auth,
        )
        from litellm.proxy.proxy_server import app

        path = _EVAL_PREFIX + "/{rest:path}"
        routes = cast("list[Any]", app.router.routes)
        if any(getattr(route, "path", None) == path for route in routes):
            return

        # FastAPI reads this function's annotations by name in this module, so
        # what they name is imported at its top
        async def forward(request: Request, rest: str) -> Response:
            # FastAPI has decoded the path, so `%2e%2e` is `..` here, and the
            # address it is passed on to would leave `/sentinel/`: only plain
            # segments are passed on
            if not all(_SEGMENT.fullmatch(part) for part in rest.split("/")):
                return Response(
                    content=json.dumps({"error": "Not found."}),
                    status_code=404,
                    media_type=_JSON,
                )
            session = self._session_to_sidecar()
            url = f"{self._base}{_EVAL_PREFIX}/{rest}"
            # only the body and what it is: the caller's key was LiteLLM's to
            # check, and is never the sidecar's to see
            headers = {"content-type": request.headers.get("content-type", _JSON)}
            try:
                async with session.request(
                    request.method,
                    url,
                    params=list(request.query_params.multi_items()),
                    data=await request.body(),
                    headers=headers,
                    timeout=self._timeout,
                ) as answer:
                    return Response(
                        content=await answer.read(),
                        status_code=answer.status,
                        media_type=answer.headers.get("content-type"),
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as ex:
                logger.warning(
                    "The sentinel sidecar could not be reached (%s).", type(ex).__name__
                )
                return Response(
                    content=json.dumps(
                        {"error": "The sentinel sidecar could not be reached."}
                    ),
                    status_code=502,
                    media_type=_JSON,
                )

        app.add_api_route(
            path,
            forward,
            methods=["GET", "POST"],
            dependencies=[Depends(user_api_key_auth)],
            include_in_schema=False,
        )
        # ahead of LiteLLM's own routes, some of which match any path
        routes.insert(0, routes.pop())
        allowed: list[str] = LiteLLMRoutes.openai_routes.value
        allowed.extend(p for p in _EVAL_PATHS if p not in allowed)

    def _shows(self, name: str) -> bool:
        # what the sidecar is shown of the caller's headers: Inspect's own, and
        # any that are listed
        return name not in _CREDENTIALS and (
            name in _INSPECT_HEADERS
            or any(_covers(pattern, name) for pattern in self._shown_headers)
        )

    def _session_to_sidecar(self) -> aiohttp.ClientSession:
        # one session for every call to the sidecar, made on first use, when
        # LiteLLM's event loop is running. A reply stream keeps a call to
        # `next_release` waiting beside its chunks, so its connections are not
        # limited to aiohttp's default of 100, which would leave chunks queued
        # behind waiting calls.
        if self._session is None:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0)
            )
        return self._session

    def _unanswered(self) -> None:
        # the sidecar gave no usable answer, and the reason is already logged
        if not self._let_through:
            raise HTTPException(503, "The sentinel sidecar gave no usable answer.")

    async def _ask_about_reply(
        self, told: dict[str, Any], content_type: str, body: bytes
    ) -> dict[str, Any]:
        return await self._ask(
            "judge_reply",
            told
            | {
                # LiteLLM calls a guardrail only with a reply the provider
                # gave, and never with a provider's error
                "status": 200,
                "content_type": content_type,
                "body": base64.b64encode(body).decode("ascii"),
            },
        )

    async def _ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # the result returned is to pass or to replace; a refusal is raised
        answer = await self._call(method, params)
        if answer.get("action") == "refuse":
            raise _refusal(answer)
        if answer.get("action") not in ("pass", "replace"):
            raise _failed("an answer with no action this class knows")
        return answer

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any]:
        # the result of a call; a call the sidecar didn't answer with one is
        # raised as `SidecarError`
        session = self._session_to_sidecar()
        # each call stands alone, so the call's own id tells nothing apart
        call = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        try:
            async with session.post(
                self._url, json=call, timeout=timeout or self._timeout
            ) as response:
                if response.status != 200:
                    raise _failed(f"status {response.status}")
                read: Any = await response.json()
        # before Python 3.11 asyncio's timeout is not the built-in one
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            TypeError,
            ValueError,
        ) as ex:
            raise _failed(type(ex).__name__) from None

        if not isinstance(read, dict):
            raise _failed("an answer that is not a JSON object")
        answered = cast("dict[str, Any]", read)
        error: Any = answered.get("error")
        if error is not None:
            # the sidecar did not judge it, and its code says why
            code: Any = (
                cast("dict[str, Any]", error).get("code")
                if isinstance(error, dict)
                else None
            )
            raise _failed(f"error {code}" if isinstance(code, int) else "an error")
        answer = answered.get("result")
        if not isinstance(answer, dict):
            raise _failed("an answer with no result")
        return cast("dict[str, Any]", answer)


class _Relay:
    """One streamed reply, handed to the sidecar's reply stream as LiteLLM yields it, and what the sidecar releases of it."""

    def __init__(
        self,
        guardrail: LiteLLMGuardrail,
        told: dict[str, Any],
        chunks: AsyncIterator[Any],
    ) -> None:
        self._guardrail = guardrail
        self._told = told
        self._chunks = chunks
        self._format: str = told["format"]
        # every chunk LiteLLM yielded, kept for the classes released events are
        # made in, and for a reply let through unjudged
        self.held: list[Any] = []
        self._pending = bytearray()
        self._size = 0
        self._read_all = False
        # LiteLLM's own error while reading, and the reason handing over
        # stopped, either of which ends the relay at once
        self._raised: Exception | None = None
        self._stopped: SidecarError | None = None
        self._failed = asyncio.Event()
        # what the reader wakes: the handing over, and a reply let through
        self._to_upload = asyncio.Event()
        self._to_yield = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._begun = False
        # once the sidecar has given the reply's end, how many outputs that
        # makes, which it is told the guardrail has, so it lets go of them
        self._ended_at: int | None = None

    async def released(self) -> AsyncIterator[Any]:
        """The chunks to yield to LiteLLM, as the sidecar releases them."""
        guardrail, told = self._guardrail, self._told
        sent = False
        # read from the start, so that a reply let through is read whatever
        # becomes of the stream
        self._tasks.append(asyncio.ensure_future(self._read()))
        try:
            await guardrail._call(  # pyright: ignore[reportPrivateUsage]
                "begin_reply", told | {"status": 200, "content_type": _STREAM}
            )
            self._begun = True
            self._tasks.append(asyncio.ensure_future(self._upload()))
            position = 0
            while True:
                answer = await self._next(position)
                outputs = cast("list[dict[str, Any]]", answer.get("outputs") or [])
                position_given = answer.get("position")
                if not isinstance(position_given, int):
                    raise _failed("a release with no position")
                position = position_given
                for output in outputs:
                    kind = output.get("kind")
                    if kind == "release":
                        for chunk in _released(
                            _decoded(output), self._format, self.held
                        ):
                            sent = True
                            yield chunk
                    elif kind == "error_event":
                        self._ended_at = position
                        for chunk in _released(
                            _decoded(output), self._format, self.held
                        ):
                            yield chunk
                        return
                    elif kind == "refuse":
                        self._ended_at = position
                        raise _refusal(output)
                    elif kind == "finished":
                        self._ended_at = position
                        return
                    else:
                        raise _failed("a release of a kind this class doesn't know")
        except HTTPException as refused:
            yield _refused_stream(self.held, refused, self._format)
        except SidecarError:
            if sent:
                # what went out can't be matched to what LiteLLM yielded, so
                # the reply can't go on unjudged from where it is
                yield _refused_mid_stream(
                    self.held,
                    HTTPException(503, "The sentinel sidecar gave no usable answer."),
                    self._format,
                )
                return
            try:
                guardrail._unanswered()  # pyright: ignore[reportPrivateUsage]
            except HTTPException as refused:
                yield _refused_stream(self.held, refused, self._format)
                return
            # let through unjudged, as LiteLLM yields it
            async for chunk in self._unjudged():
                yield chunk

    async def close(self) -> None:
        """Stop reading and handing over, and tell the sidecar the guardrail is done with the stream: that it has the reply's end, or that it stopped before it."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.wait(self._tasks)
        if not self._begun:
            return
        guardrail, id = self._guardrail, self._told["id"]
        try:
            if self._ended_at is not None:
                await guardrail._call(  # pyright: ignore[reportPrivateUsage]
                    "next_release", {"id": id, "after": self._ended_at}
                )
            else:
                await guardrail._call("abort_reply", {"id": id})  # pyright: ignore[reportPrivateUsage]
        except SidecarError:
            # the sidecar lets go of it once no call names it
            pass

    async def _next(self, position: int) -> dict[str, Any]:
        # the next release, or why the relay can't go on, whichever comes first
        polling = asyncio.ensure_future(
            self._guardrail._call(  # pyright: ignore[reportPrivateUsage]
                "next_release",
                {"id": self._told["id"], "after": position},
                _POLL_TIMEOUT,
            )
        )
        failing = asyncio.ensure_future(self._failed.wait())
        done, _ = await asyncio.wait(
            {polling, failing}, return_when=asyncio.FIRST_COMPLETED
        )
        failing.cancel()
        if polling not in done:
            polling.cancel()
            await asyncio.wait({polling, failing})
            if self._raised is not None:
                # LiteLLM's own error, which LiteLLM sees as it would have
                raise self._raised
            assert self._stopped is not None
            raise self._stopped
        return polling.result()

    async def _read(self) -> None:
        # LiteLLM's chunks, kept, and handed over as the caller would be sent
        # them while the reply is no larger than is held to be judged; the
        # rest is read all the same, for a reply let through unjudged
        limit = self._guardrail._max_reply_bytes  # pyright: ignore[reportPrivateUsage]
        try:
            async for chunk in self._chunks:
                self.held.append(chunk)
                if self._stopped is None:
                    written = _event(chunk)
                    self._size += len(written)
                    if self._size > limit:
                        self._stop(_failed("a reply larger than max_reply_bytes"))
                    else:
                        self._pending += written
                self._to_upload.set()
                self._to_yield.set()
        except Exception as ex:
            self._raised = ex
            self._failed.set()
        finally:
            self._read_all = True
            self._to_upload.set()
            self._to_yield.set()

    async def _upload(self) -> None:
        # what has arrived while the last chunk was being sent goes as one
        seq = 0
        try:
            while True:
                await self._to_upload.wait()
                self._to_upload.clear()
                if self._stopped is not None or self._raised is not None:
                    return
                if self._pending:
                    data, self._pending = bytes(self._pending), bytearray()
                    await self._guardrail._call(  # pyright: ignore[reportPrivateUsage]
                        "reply_chunk",
                        {
                            "id": self._told["id"],
                            "seq": seq,
                            "data": base64.b64encode(data).decode("ascii"),
                        },
                    )
                    seq += 1
                    self._to_upload.set()
                    continue
                if self._read_all:
                    await self._guardrail._call(  # pyright: ignore[reportPrivateUsage]
                        "end_reply", {"id": self._told["id"]}
                    )
                    return
        except SidecarError as ex:
            self._stop(ex)

    def _stop(self, why: SidecarError) -> None:
        # handing over can't go on, which ends the relay
        if self._stopped is None:
            self._stopped = why
        self._failed.set()

    async def _unjudged(self) -> AsyncIterator[Any]:
        # every chunk LiteLLM yields, as the reader keeps them
        given = 0
        while True:
            self._to_yield.clear()
            while given < len(self.held):
                yield self.held[given]
                given += 1
            if self._read_all:
                if self._raised is not None:
                    raise self._raised
                return
            await self._to_yield.wait()


def _listed(headers: str | Iterable[str] | None) -> tuple[str, ...]:
    names = headers.split(",") if isinstance(headers, str) else list(headers or ())
    patterns = tuple(str(name).strip().lower() for name in names if str(name).strip())
    for pattern in patterns:
        if not re.fullmatch(r"[a-z0-9_-]+\*?", pattern):
            raise ValueError(
                f"sidecar_headers takes header names, or prefixes ending in '*', not {pattern!r}."
            )
        if any(_covers(pattern, name) for name in _CREDENTIALS):
            raise ValueError(
                f"sidecar_headers may not list {pattern!r}, which would show the sidecar a credential."
            )
    return patterns


def _covers(pattern: str, name: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def _record(data: dict[str, Any]) -> dict[str, Any]:
    # LiteLLM's own record of the request as the caller sent it
    record = data.get("proxy_server_request")
    if not isinstance(record, dict):
        raise _failed("a request LiteLLM kept no record of")
    return cast("dict[str, Any]", record)


def _body(data: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    # `data` holds the request's own fields among LiteLLM's working state, some
    # of which is not JSON. The fields that are the request's are those in
    # LiteLLM's record of what the caller sent. Their values are taken from
    # `data`, which is what goes to the provider and which an earlier guardrail
    # may have changed.
    sent = record.get("body")
    if not isinstance(sent, dict):
        raise _failed("a request LiteLLM kept no record of")
    # LiteLLM writes what it knows of the caller, the key's hash and every
    # header among it, into `litellm_metadata`, or into `metadata` on a route
    # that has no such field of its own. A caller may also send LiteLLM a
    # provider's key, or where to call, among the request's fields.
    own = {"litellm_metadata"} if "litellm_metadata" in data else {"metadata"}
    hidden = own | _NOT_SHOWN | set(all_litellm_params)
    return {
        name: data[name]
        for name in cast("dict[str, Any]", sent)
        if name in data and name not in hidden
    }


def _withhold_inspect_headers(data: dict[str, Any]) -> None:
    # LiteLLM sends a caller's `x-` headers on to the provider only when its
    # operator says to, in `headers`. Inspect's own are never sent on.
    for field in ("headers", "extra_headers"):
        headers: Any = data.get(field)
        if isinstance(headers, dict):
            held = cast("dict[Any, Any]", headers)
            for name in [n for n in held if str(n).lower() in _INSPECT_HEADERS]:
                del held[name]


def _headers(record: dict[str, Any]) -> dict[str, str]:
    headers = record.get("headers")
    if not isinstance(headers, Mapping):
        return {}
    return {
        str(name).lower(): str(value)
        for name, value in cast("Mapping[Any, Any]", headers).items()
    }


def _told(data: dict[str, Any]) -> dict[str, Any] | None:
    # what the sidecar was told of the request this is the reply to, or None
    # when it was told nothing and so can't be asked about the reply
    record = data.get("proxy_server_request")
    told = (
        cast("dict[str, Any]", record).get(_KEPT) if isinstance(record, dict) else None
    )
    return cast("dict[str, Any]", told) if isinstance(told, dict) else None


def _replace_request(data: dict[str, Any], sent: dict[str, Any], body: object) -> None:
    if not isinstance(body, dict):
        raise _failed("a replaced request that is not a JSON object")
    replaced = cast("dict[str, Any]", body)
    # only a field of the request may be set, and never LiteLLM's own state
    if any(name in data and name not in sent for name in replaced):
        raise _failed("a replaced request that sets a field LiteLLM keeps")
    for name in sent.keys() - replaced.keys():
        del data[name]
    for name, value in replaced.items():
        if name not in sent or sent[name] != value:
            data[name] = value


def _whole(response: Any) -> bytes:
    # as LiteLLM is about to write it for the caller
    if isinstance(response, BaseModel):
        return response.model_dump_json().encode()
    if isinstance(response, dict):
        reply = cast("dict[str, Any]", response)
        # LiteLLM's own notes on the call, which it removes before it answers
        return json.dumps(
            {name: value for name, value in reply.items() if name != "_hidden_params"}
        ).encode()
    raise _failed("a reply of a kind this can't write")


def _event(chunk: Any) -> bytes:
    # as LiteLLM is about to write it for the caller. On the Messages route a
    # chunk is already the bytes of one or more events.
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        return chunk.encode()
    if isinstance(chunk, BaseModel):
        written = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        return f"data: {written}\n\n".encode()
    if isinstance(chunk, dict):
        return f"data: {json.dumps(chunk)}\n\n".encode()
    raise _failed("a reply stream of a kind this can't write")


def _readable(item: Any) -> Any:
    # LiteLLM writes a Claude model's thinking, when it was asked for through
    # the Responses API, as a `reasoning` item that OpenAI's own types reject:
    # it has no `summary`, and its parts are typed `output_text` where the API
    # has `reasoning_text`. The sidecar reads such an item in a reply, but
    # inspect_ai reads a request's items strictly, so a request from a client
    # that sends the item back as it got it can't be read. So the item is put
    # right in what the sidecar is sent. LiteLLM keeps it as the client sent it.
    if not isinstance(item, dict):
        return item
    entry = cast("dict[str, Any]", item)
    if entry.get("type") != "reasoning":
        return entry
    content: Any = entry.get("content")
    parts: Any = content
    if isinstance(content, list):
        parts = [
            {**cast("dict[str, Any]", part), "type": "reasoning_text"}
            if isinstance(part, dict)
            and cast("dict[str, Any]", part).get("type") == "output_text"
            else part
            for part in cast("list[Any]", content)
        ]
    if entry.get("summary") is not None and parts == content:
        return entry
    readable: dict[str, Any] = {**entry, "summary": entry.get("summary") or []}
    if "content" in entry:
        readable["content"] = parts
    return readable


def _readable_request(
    format: str, sent: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A request as the sidecar is shown it, and what to put back in a request the sidecar hands on.

    A client that sends LiteLLM's `reasoning` item back as it got it sends one the sidecar can't read (`_readable`). The second value gives the item as the client sent it for each one that was put right, by how the item is written once put right.
    """
    entries: Any = sent.get("input")
    if format != _RESPONSES or not isinstance(entries, list):
        return sent, {}
    given = cast("list[Any]", entries)
    readable = [_readable(entry) for entry in given]
    as_sent = {
        json.dumps(new, sort_keys=True): old
        for new, old in zip(readable, given, strict=True)
        if new is not old
    }
    return ({**sent, "input": readable} if as_sent else sent), as_sent


def _restored(body: object, as_sent: dict[str, Any]) -> object:
    # LiteLLM is sent each item as the client sent it, and not as it was put
    # right for the sidecar to read
    if not as_sent or not isinstance(body, dict):
        return body
    replaced = cast("dict[str, Any]", body)
    entries: Any = replaced.get("input")
    if not isinstance(entries, list):
        return replaced
    return {
        **replaced,
        "input": [
            as_sent.get(json.dumps(entry, sort_keys=True), entry)
            for entry in cast("list[Any]", entries)
        ],
    }


def _replacement(answer: dict[str, Any]) -> bytes:
    body = answer.get("body")
    try:
        if not isinstance(body, str):
            raise ValueError
        return base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        raise _failed("a replaced reply that is not base64") from None


def _replaced_whole(response: Any, body: bytes) -> Any:
    # LiteLLM goes on to read its own notes on the call from the reply it is
    # handed back, so the replacement is the same class and carries them
    try:
        parsed: Any = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError
        reply = cast("dict[str, Any]", parsed)
        if isinstance(response, dict):
            notes = cast("dict[str, Any]", response).get("_hidden_params")
            return reply if notes is None else reply | {"_hidden_params": notes}
        replaced = type(response)(**reply)
        if hasattr(response, "_hidden_params"):
            replaced._hidden_params = response._hidden_params
        return replaced
    except Exception:
        raise _failed("a replaced reply LiteLLM can't take") from None


def _decoded(output: dict[str, Any]) -> bytes:
    data = output.get("data")
    try:
        if not isinstance(data, str):
            raise ValueError
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise _failed("a release that is not base64") from None


def _released(data: bytes, format: str, held: list[Any]) -> list[Any]:
    # what the sidecar released, as chunks LiteLLM writes as they were sent
    classes = {
        getattr(chunk, "type", None): type(chunk)
        for chunk in held
        if isinstance(chunk, BaseModel)
    }
    if not classes:
        # the Messages route streams bytes
        return [data]
    try:
        events = _events(data)
        released = [
            # LiteLLM writes a chunk of its own class without a field the
            # class doesn't have, so a chunk with the sentinel field, and an
            # error the sidecar wrote, are ones that keep every field
            _Written(**event)
            if "error" in event
            or (format == _CHAT and "sentinel" in event)
            or event.get("type") not in classes
            else classes[event.get("type")](**event)
            for event in events
        ]
    except Exception:
        raise _failed("a release LiteLLM can't take") from None
    if data.strip() and not released and b"[DONE]" not in data:
        # the sidecar releases whole events, so bytes that hold none are not
        # a release
        raise _failed("a release that holds no event")
    return released


def _refused_mid_stream(held: list[Any], refused: HTTPException, format: str) -> Any:
    # A refusal once some of the stream has gone out: an error event the
    # provider's SDK raises on, in the provider's own form.
    if format == _RESPONSES:
        return _response_failed(held, refused)
    if format == _CHAT:
        return _refused_stream(held, refused, format)
    error = {
        "type": "error",
        "error": {"type": "api_error", "message": str(refused.detail)},
    }
    return f"event: error\ndata: {json.dumps(error)}\n\n".encode()


def _events(body: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    # the line endings an event stream may use. str.splitlines breaks at more,
    # some of which JSON allows unescaped inside a string
    for line in _LINE_END.split(body.decode()):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        # OpenAI's chat stream ends with an event that is not JSON, which
        # LiteLLM writes itself
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


def _refusal(answer: dict[str, Any]) -> HTTPException:
    status = answer.get("status", 400)
    message = answer.get("message")
    if (
        not isinstance(status, int)
        or not 400 <= status <= 599
        or not isinstance(message, str)
    ):
        # a refusal that can't be carried out as asked still refuses,
        # whatever `unreachable_fallback` says of a failure
        logger.warning("The sentinel sidecar's refusal could not be read.")
        return HTTPException(503, "The sentinel sidecar's refusal could not be read.")
    refused = HTTPException(status, message)
    field = answer.get("sentinel")
    if isinstance(field, dict):
        # LiteLLM writes these into the error it answers with, beside its own
        # fields, at `error.provider_specific_fields`
        setattr(refused, "provider_specific_fields", {"sentinel": field})  # noqa: B010
    return refused


def _refused_stream(held: list[Any], refused: HTTPException, format: str) -> Any:
    # A refusal raised here reaches the caller as LiteLLM writes an error in a
    # stream: a 500 without the sentinel field in some versions, and in others
    # an error whose body the provider's SDK keeps only as text. Every version
    # instead answers a stream whose first event is an error with that error
    # alone, as JSON, under the status the event names. LiteLLM then logs the
    # call as one that succeeded.
    if format == _RESPONSES:
        # a Responses client is told of a failed response by its own event
        return _response_failed(held, refused)
    error: dict[str, Any] = {
        "message": str(refused.detail),
        "type": _error_code(refused.status_code),
        "param": None,
        "code": refused.status_code,
    }
    extra = getattr(refused, "provider_specific_fields", None)
    if isinstance(extra, dict):
        error["provider_specific_fields"] = extra
    # written, since LiteLLM finds an error in a stream's first event only in
    # what is written
    return f"data: {json.dumps({'error': error})}\n\n"


def _response_failed(held: list[Any], refused: HTTPException) -> _Written:
    # the event that ends a Responses stream that failed, built on the
    # response the stream began, with the refusal and the sentinel field
    events = [
        chunk.model_dump(exclude_none=True)
        for chunk in held
        if isinstance(chunk, BaseModel)
    ]
    begun: dict[str, Any] = {}
    for event in events:
        if event.get("type") == "response.created":
            begun = cast("dict[str, Any]", event["response"])
            break
    response: dict[str, Any] = {
        **begun,
        "status": "failed",
        "error": {
            "code": _error_code(refused.status_code),
            "message": str(refused.detail),
        },
    }
    sequence = max((event.get("sequence_number", 0) for event in events), default=0)
    # The OpenAI SDK raises on an event with an error at its top, keeping
    # only that error, so the sentinel field goes in it; a client that reads
    # the events sees the response fail with the same error.
    error: dict[str, Any] = {
        **response["error"],
        "type": response["error"]["code"],
        "param": None,
    }
    extra = getattr(refused, "provider_specific_fields", None)
    if isinstance(extra, dict) and "sentinel" in extra:
        error["provider_specific_fields"] = {
            "sentinel": cast("dict[str, Any]", extra)["sentinel"]
        }
    event: dict[str, Any] = {
        "type": "response.failed",
        "sequence_number": sequence + 1,
        "response": response,
        "error": error,
    }
    return _Written(**event)


def _error_code(status: int) -> str:
    if 400 <= status < 500:
        return _ERROR_CODES.get(status, "invalid_request_error")
    return "server_error"


def _failed(reason: str) -> SidecarError:
    # the reason is one of this module's own short strings, never anything a
    # request, a reply or the sidecar's answer held
    logger.warning("The sentinel sidecar gave no usable answer (%s).", reason)
    return SidecarError(reason)
