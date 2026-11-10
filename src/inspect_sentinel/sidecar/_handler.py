from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallView
from inspect_ai.util import Store
from pydantic import JsonValue

from .._context import EvalContext, InspectSample, ProxyContext
from .._host import HostNative
from .._integration import HostContext, resolve_sentinel, run_sentinel
from .._report import Decision
from .._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from .._types import Sentinels
from ..config import watched_stages
from ._errors import TimeLimitError, UnauthorizedError
from ._errors import not_judged as _not_judged
from ._exchange import (
    MAX_BODY_BYTES,
    Answer,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    succeeded,
)
from ._history import record_sent, with_sent_ids
from ._host import Record, RunHost, SidecarHost, noting_models, stage
from ._inspect_ai import VIEWED_TOOLS, tool_call_view
from ._inspect_client import (
    INSPECT_CONTEXT,
    Exchange,
    Judged,
    SignedSample,
    Withheld,
    signed_sample,
)
from ._registration import (
    MAX_TASK_EXPIRY,
    MAX_TASK_REGISTRATIONS,
    TASK_EXPIRY,
    FileTask,
    RegisteredTask,
    RegistrationError,
    RegistrationsFullError,
    SampleFacts,
    Strictness,
    TaskRegistration,
    UnknownTaskError,
    UnregisteredSampleError,
)
from ._reply_stream import KEEPALIVE_SECONDS, PENDING_BYTES, ReplyStart, ReplyStream
from ._sign import TIMESTAMP_WINDOW, call_values, request_values, unproven
from ._state import (
    MADE_KEPT,
    Attempt,
    Conversation,
    Dropped,
    EndedOn,
    Held,
    Made,
    MemoryState,
    NotJudged,
    Rejected,
    Run,
    State,
)
from ._wire import (
    Edited,
    Edits,
    UnreadableError,
    amend_request,
    amendments,
    call_texts,
    edit_reply,
    entry_digests,
    grows,
    held_rejections,
    offered,
    read_opening,
    read_reply_noted,
    read_request,
    request_format,
    request_marks,
    unredacted,
)

# rejected turns in a row before the run is ended, as inspect_ai's bridge has
# it: a model that keeps proposing rejected calls would otherwise never stop
MAX_RESULT_WAIT = 30.0
"""The longest `request_result` waits for a request still being judged, in seconds."""

# how often a waiting `request_result` looks again for a new attempt
_RESULT_POLL = 0.05

_MAX_CONSECUTIVE_REJECTIONS = 3

_REPLACED = Refuse(400, "A later attempt at this sample replaced this one.")

# what inspect_ai tells the model of a rejected call that has no message
_REJECTED = "Tool call not approved."
_NOT_RUN = "This call was not run, because another call in the same turn was rejected."
_ENDED = "This call was not run, because a sentinel ended the run."


REQUEST_ID = "x-irid"
"""The header an Inspect eval names each of its model requests in."""


@dataclass(frozen=True)
class RequestResult:
    """What happened to a request an eval names."""

    latest: Attempt
    """The latest attempt at it."""
    records: list[Record]
    """What was recorded, every attempt's in order."""


@dataclass(frozen=True)
class SampleResult:
    """What an eval's sample came to."""

    outcome: Literal["proceed", "terminate"]
    """`terminate` if a sentinel ended the sample."""
    store: dict[str, Any]
    """The sentinel's store, by the keys the sentinel wrote."""


class ModifyError(Exception):
    """The sentinel decided to change a tool call that the sidecar can't change in the reply, such as one that is not a function call, so the reply was not judged."""


class Handler:
    """Judges what a proxy hands over with a sentinel, and answers what the proxy should do.

    It knows nothing of any proxy, nor of how it is reached. The endpoint for proxies turns each message it is sent into a `Request` or a `Reply`, and turns the `Answer` back into what it answers with.

    The four steps are found in the traffic. A request carries the results of the tool calls the client ran, judged as `AfterToolCall`, and is judged as `BeforeGenerate` before it goes to the provider. A reply is judged as `AfterGenerate`, and each tool call in it as `BeforeToolCall`, before the client is handed any of it.

    An Inspect eval names the sample of each request in its `x-inspect-context` header, and the sample is the run. It asks what was decided of each request with `request_result`, by the request's `x-irid`. Every client, the eval included, is handed the provider's reply as it was, or as the sentinel edited it.
    """

    def __init__(
        self,
        sentinel: Sentinels,
        *,
        host: HostNative | None = None,
        state: State | None = None,
        reject_status: int = 400,
        time_limit: float | None = None,
        eval_results: bool = True,
        tasks: Mapping[str, FileTask] | None = None,
        runtime_task_registration: bool = True,
        strictness: Strictness = "open",
        task_expiry: float = MAX_TASK_EXPIRY,
        max_task_registrations: int = MAX_TASK_REGISTRATIONS,
        max_registration_bytes: int = MAX_BODY_BYTES,
        release: Literal["whole", "early"] = "whole",
        keepalive: float | None = KEEPALIVE_SECONDS,
        timestamp_window: float = TIMESTAMP_WINDOW,
    ) -> None:
        """Create a handler.

        Args:
            sentinel: One protocol, or a sequence or mapping of monitors and protocols, as a task's `sentinel=` takes.
            host: What the sentinel may do to the outside world. Defaults to a `SidecarHost` with no model roles.
            state: Where runs are kept. Defaults to this process's memory.
            reject_status: The HTTP status a rejected turn is refused with. The default, 400, is one a client takes as final. A status a client retries by itself, such as 503, brings the client back with the same request, which then goes on with the rejection added.
            time_limit: The longest the sentinel may take to judge one request or one reply, in seconds, after which it is cancelled. None, the default, is no limit.
            eval_results: Whether the judgment `request_result` gives an Inspect eval holds the results: the explanations, references and metadata of the decisions, and what each monitor and protocol recorded. Without them, it holds only what Inspect acts on. A monitor's explanation tells an agent what the monitor looks for, so an eval that can't be trusted is not sent them.
            tasks: Tasks registered beforehand, by name, as the tasks folder's files hold them. Nothing an eval registers replaces one, and a request for a sample one doesn't hold is refused.
            runtime_task_registration: Whether an eval may register its tasks itself, at task init.
            strictness: How strict the handler is about a request it can't place. `open`, the default, judges every request, without the task's facts where it holds none. `known_tasks` refuses a request whose `x-inspect-context` names a task the handler doesn't hold, and judges one without the header. `registered_only` refuses any request that doesn't name a sample the handler holds, in an epoch the task's registration runs, including one without the header, and refuses a sample's registration that the task's registration doesn't name.
            task_expiry: The longest a task an eval registered is kept once no request has named it, in seconds, whatever the registration asks.
            max_task_registrations: How many tasks registered by evals are kept at once. A registration past it is refused.
            max_registration_bytes: The most a task registered by an eval holds, in bytes of JSON: its registration, the facts of each sample whose registration is complete, and each sample id a sample's registration added. A sample's registration past it fails. It defaults to the largest request body, which is what a task's whole registration could hold when it came in one call.
            release: How a streamed reply is passed on. `whole`, the default, holds every reply until the sentinel has judged it. `early` passes on the text and reasoning that come before the turn's first tool call as they arrive, holds the rest of the turn, and judges it once the turn has ended, as `whole` does. The sentinel is given the whole turn either way.
            keepalive: Under early release, the longest a client is sent nothing while a streamed reply is held, in seconds, after which it is sent a keepalive, an SSE comment that every provider SDK skips. The keepalive sends the status, so a turn refused after it is refused with an error event in the stream. None sends no keepalives.
            timestamp_window: How far, in seconds, the `timestamp` of an Inspect eval's model request or call to the endpoint for evals may be from this process's clock. One further is refused, so a `mac` copied from a log is of no use once the window has passed.

        Raises:
            ValueError: If `keepalive` or `timestamp_window` isn't a positive number of seconds.
        """
        if keepalive is not None and not keepalive > 0:
            raise ValueError(
                f"A keepalive is sent after some seconds, not {keepalive}."
            )
        if not timestamp_window > 0:
            raise ValueError(
                f"The timestamp window is some seconds, not {timestamp_window}."
            )
        self._timestamp_window = timestamp_window
        self._release: Literal["whole", "early"] = release
        self._keepalive = keepalive
        self._protocol = resolve_sentinel(sentinel)
        # a stage nothing watches is never built, as in Inspect's own process
        self._watched = watched_stages(self._protocol)
        self._host: HostNative = host if host is not None else SidecarHost()
        self._state: State = state if state is not None else MemoryState()
        self._reject_status = reject_status
        self._time_limit = time_limit
        self._eval_results = eval_results
        self._tasks = dict(tasks or {})
        self._runtime_task_registration = runtime_task_registration
        self._strictness: Strictness = strictness
        self._task_expiry = task_expiry
        self._max_task_registrations = max_task_registrations
        self._max_registration_bytes = max_registration_bytes
        # the tool results being judged now, by run, conversation and digest,
        # so that a request holding one waits for the request judging it
        self._judging: dict[tuple[str, str, str], anyio.Event] = {}

    @property
    def release(self) -> Literal["whole", "early"]:
        """How a streamed reply is passed on."""
        return self._release

    @property
    def eval_results(self) -> bool:
        """Whether an Inspect client is sent the results, or only what it acts on."""
        return self._eval_results

    @property
    def keepalive(self) -> float | None:
        """Under early release, how many seconds a client may be sent nothing before it is sent a keepalive, or None for no keepalives."""
        return self._keepalive

    @property
    def timestamp_window(self) -> float:
        """How far, in seconds, an Inspect eval's `timestamp` may be from this process's clock."""
        return self._timestamp_window

    async def request(self, request: Request) -> Answer:
        """Judge a request before the proxy sends it.

        Each tool result in the request that the sentinel hasn't been shown is put to it as an `AfterToolCall` step, in order. A result is shown once, however often the client sends it. The request is then put to the sentinel as a `BeforeGenerate` step. A stage nothing in the sentinel watches is left out. If either ends the run, the request is refused.

        If turns were rejected earlier in the conversation this request continues, they are added where they happened, so the model is still told.

        A request with `x-inspect-context` belongs to the sample it names, which is its run. A request whose header can't be read is refused, rather than judged as another client's would be: it says it is an eval's, and a proxy set to let through what can't be judged would let it through unjudged.

        The header's `mac` proves the values that name the run, made with the token the eval registered with. Before anything of the run is looked up or changed, a request is refused unjudged, whatever the strictness, if the handler holds no token for the eval, as after a restart, with the reason `unregistered`; or if its `timestamp` is outside the window or its `mac` doesn't match, with the reason `unauthenticated`. So a client without the eval's token that changes the sample, the attempt or its number, or names a sample of the eval, reaches no run.

        A conversation belongs to one attempt at the sample. A new attempt starts afresh, as Inspect starts the sample store afresh for it: its conversations, its count of rejected turns in a row, the sentinel's store, and the run's end if a sentinel ended the attempt before. A sample the eval has ended stays ended. Attempts are ordered by the number the header gives each. Once a later attempt has begun, a request of an earlier one is refused unjudged, as nobody waits for it, and so is one of another attempt with the same number: `request_result` says it was not judged, with the reason `replaced_attempt`, so an eval never takes it for a sentinel's terminate. A client that retries a request within the attempt keeps the attempt's conversation.

        A request of a run that a sentinel has ended is refused, and the sentinel is not asked again. For a run no eval names, that is a request which continues the conversation the run was ended in.

        Args:
            request: The request, as the proxy handed it over.

        Raises:
            UnreadableError: If the request can't be read, so neither it nor its reply could be judged.
            UnregisteredSampleError: If the request names a sample that a task registered beforehand doesn't hold, or the handler's strictness refuses a request it can't place.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        id = request.headers.get(REQUEST_ID)
        attempt = None
        if id:
            attempt = await self._state.attempt(id, request.id)
            # an attempt that is never judged has records too: none
            await self._state.recorder(request.id)
        try:
            signed = signed_sample(request.headers)
        except UnreadableError as ex:
            # refused rather than raised, but never judged, so not a terminate
            if attempt is not None:
                attempt.not_judged("unreadable")
            return Refuse(400, str(ex))
        sample = signed.sample if signed is not None else None
        if signed is not None:
            # proven before anything of its run is looked up or changed, and
            # refused rather than raised, so a proxy that lets through what
            # can't be judged doesn't let through a request that can't prove
            # its run
            refused = await self._unproven_request(signed)
            if refused is not None:
                if attempt is not None:
                    attempt.not_judged(refused[0])
                return Refuse(400, refused[1])
            if attempt is not None:
                attempt.eval_id = signed.sample.eval_id
        try:
            answer = await self._within_limit(
                self._request(request, sample, attempt), "request", request.id
            )
        except Exception as ex:
            if attempt is not None:
                attempt.not_judged(_not_judged(ex))
            raise
        if attempt is not None:
            if not isinstance(answer, Refuse):
                attempt.state = "awaiting_reply"
            elif attempt.judged is not None and attempt.judged.replaced:
                attempt.not_judged("replaced_attempt")
            else:
                # otherwise a request is refused only when its run has ended
                attempt.decide("terminate")
        return answer

    async def reply(self, reply: Reply, *, released: bytes = b"") -> Answer:
        """Judge a reply before the proxy passes any of it on.

        The reply is put to the sentinel as an `AfterGenerate` step, then each tool call in it as a `BeforeToolCall` step, in order, leaving out a stage nothing in the sentinel watches. Every call is judged, as in Inspect's own process, unless one ends the run. The rejected calls are taken out of the reply and the rest of the turn goes on, answered with `Replace`, as the other calls of a turn still run in Inspect's own process. Each later request of the conversation that holds the turn has the calls taken out put back, each with its rejection as its result. A turn whose every call is rejected is refused whole, as is one that can't be edited, and its rejections are added to the later requests of its conversation. The third turn refused whole in a row within an attempt at a sample ends the run; a turn in which a call ran starts the count again. A provider's error, a reply in a status outside 2xx, is passed on unjudged; a reply in any 2xx is the model's, and judged, as the provider's SDK reads it so.

        For an Inspect eval, what was decided is kept for `request_result`: every step judged during the exchange, the request's first. A turn handed over with calls taken out or changed, or refused, is kept with the calls a decision acted on redacted, as the eval is handed them.

        Args:
            reply: The reply, as the proxy handed it over.
            released: The start of the reply, as the client has already been sent it, for a reply released early. A reply edited in a way that changes it can't be sent, and is not judged.

        Raises:
            UnreadableError: If the reply can't be read, or answers a request this handler was never shown or no longer keeps.
            ModifyError: If the sentinel decides `modify` of a call the sidecar can't change, and rejects none of the turn's calls.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        held = await self._state.release(reply.id)
        attempt = held.attempt if held is not None else None
        if attempt is not None:
            attempt.state = "judging"
        try:
            answer = await self._within_limit(
                self._reply(reply, held, released), "reply", reply.id
            )
        except Exception as ex:
            if attempt is not None:
                attempt.not_judged(_not_judged(ex))
            raise
        if attempt is None or held is None:
            return answer
        if not succeeded(reply.status):
            attempt.not_judged("provider_error")
        elif not isinstance(answer, Refuse):
            attempt.decide("proceed")
        elif held.exchange.replaced:
            attempt.not_judged("replaced_attempt")
        elif held.exchange.ended:
            attempt.decide("terminate")
        else:
            attempt.decide("reject", held.exchange.message)
        return answer

    def reply_stream(
        self,
        start: ReplyStart,
        *,
        max_held_bytes: int = MAX_BODY_BYTES,
        max_pending_bytes: int = PENDING_BYTES,
        keepalive: bool = True,
    ) -> ReplyStream:
        """Judge a reply that the proxy hands over as it arrives, rather than whole.

        The proxy runs the stream's `run`, feeds it the provider's bytes, and sends the client what iterating over it gives, all at once. The reply is judged as `reply` judges it, once it has ended.

        Args:
            start: The reply's status and headers, before any of its body.
            max_held_bytes: The most of the reply the stream holds back to judge. A reply larger than it is refused with a 502, and not judged.
            max_pending_bytes: How much of what goes out as it arrives may wait for the proxy to collect it before feeding waits.
            keepalive: Whether the proxy can send its client a keepalive. Where it can't, none is given, so that a refusal keeps its status for as long as nothing else has gone out.
        """
        return ReplyStream(
            self,
            start,
            max_held_bytes=max_held_bytes,
            max_pending_bytes=max_pending_bytes,
            keepalive=self._keepalive if keepalive else None,
        )

    async def reply_not_judged(self, id: str, reason: NotJudged) -> None:
        """Let go of a request whose reply the proxy refused without handing it over, as one larger than it holds.

        The request is no longer kept, and an eval that names it is told its reply was not judged, for the reason given, rather than that it is still awaited.

        Args:
            id: The proxy's id for the request.
            reason: Why the reply was not judged.
        """
        held = await self._state.release(id)
        if held is not None and held.attempt is not None:
            held.attempt.not_judged(reason)

    async def eval_token(self, eval_id: str, presented: str) -> str | None:
        """The secret the sidecar keeps for an eval, which a registration that sends a secret must send.

        Args:
            eval_id: The eval's id.
            presented: The secret the caller sent for the eval.

        Returns:
            The secret kept, which the caller sent, as when a registration whose answer was lost is sent again; or None if none is kept, at the eval's first registration or once a restart or the expiry has lost it.

        Raises:
            UnauthorizedError: If the sidecar keeps another secret for the eval.
        """
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return None
        if not hmac.compare_digest(presented.encode(), kept.encode()):
            raise UnauthorizedError(
                f"The eval {eval_id!r} registered with another token."
            )
        return await self._state.eval_token(eval_id)

    async def eval_token_room(self) -> None:
        """Check there is room to keep one more eval's token, before a registration for a new eval is made.

        A token is kept for `--run-expiry` after the eval was last named, apart from the registration that made it, which may expire sooner or, for a task registered beforehand, take no room at all. So the tokens are bounded on their own, by the same number as the tasks registered by evals, and nothing kept is dropped to make room: a kept token guards the records of the eval's runs for as long as they are kept.

        Raises:
            RegistrationsFullError: If the handler already keeps as many tokens as it takes registrations.
        """
        if await self._state.eval_tokens() >= self._max_task_registrations:
            raise RegistrationsFullError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes, or the tokens of as many evals. One is taken again once a kept task or token expires."
            )

    async def keep_eval_token(self, eval_id: str, token: str) -> None:
        """Keep the secret an eval made and sent with a registration that succeeded, unless one is kept already.

        Kept only once a registration succeeds, so a registration that fails leaves no secret behind. The sidecar never makes a secret, and never sends one back.

        Raises:
            UnauthorizedError: If another registration for the eval kept another secret meanwhile.
        """
        kept = await self._state.keep_eval_token(eval_id, token)
        if not hmac.compare_digest(token.encode(), kept.encode()):
            raise UnauthorizedError(
                f"The eval {eval_id!r} registered with another token."
            )

    async def holds_eval(
        self, eval_id: str, method: str, timestamp: int, mac: str
    ) -> bool:
        """Whether the sidecar keeps a secret for the eval, and the caller proved it holds it.

        Args:
            eval_id: The eval's id.
            method: The method called.
            timestamp: When the caller signed, in seconds since the epoch.
            mac: The caller's HMAC of the method, the eval's id and the timestamp, made with the eval's secret.

        Returns:
            True if the caller proved it holds the eval's secret; False if the sidecar keeps none for the eval, so it holds nothing of the eval for anyone, as after a restart.

        Raises:
            UnauthorizedError: If the sidecar keeps a secret for the eval, and the `timestamp` is outside the window or the `mac` isn't made with it.
        """
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return False
        problem = unproven(
            kept,
            call_values(method, eval_id, timestamp),
            timestamp,
            mac,
            window=self._timestamp_window,
            what="The call's",
        )
        if problem is not None:
            raise UnauthorizedError(problem)
        # a call that proves it holds the secret keeps it
        await self._state.eval_token(eval_id)
        return True

    async def _unproven_request(
        self, signed: SignedSample
    ) -> tuple[NotJudged, str] | None:
        # why a request's header doesn't prove its run, or None if it does
        eval_id = signed.sample.eval_id
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return (
                "unregistered",
                f"This sidecar holds no registration of the eval {eval_id!r}, as after a restart, so it can't check the request's {INSPECT_CONTEXT} header. The eval registers its task again.",
            )
        problem = unproven(
            kept,
            request_values(signed.sample, signed.timestamp),
            signed.timestamp,
            signed.mac,
            window=self._timestamp_window,
            what=f"The request's {INSPECT_CONTEXT}",
        )
        if problem is not None:
            return "unauthenticated", f"{problem} The request was not judged."

        # A run belongs to the eval that started it. A client can register an
        # eval of its own and sign with that eval's token, so the token alone
        # doesn't stop it naming another eval's sample.
        held = await self._state.existing_run(
            _run_key(True, signed.sample.sample_uuid), renew=False
        )
        if held is not None and held.eval_id is not None and held.eval_id != eval_id:
            return (
                "unauthenticated",
                f"The request's {INSPECT_CONTEXT} header names a sample of another eval. The request was not judged.",
            )

        # a request that proves its run keeps the eval's token
        await self._state.eval_token(eval_id)
        return None

    async def request_result(
        self, id: str, eval_id: str, wait: float = 0
    ) -> RequestResult | Literal["dropped", "unknown"]:
        """What happened to a request an eval names, by the eval's id for it.

        Args:
            id: The eval's id for the request, as its `x-irid` header gave it. A client that retries a request sends it again under the same id.
            eval_id: The eval asking. A request whose `x-inspect-context` named another eval, or none, reads as unknown.
            wait: How long to wait, in seconds, while the latest attempt is still being judged, at most `MAX_RESULT_WAIT`.

        Returns:
            Where the latest attempt stands, and every attempt's records in order; `dropped` if they were kept and no longer are; or `unknown` if the request was never seen, or isn't the eval's.
        """
        deadline = time.monotonic() + min(max(wait, 0), MAX_RESULT_WAIT)
        while True:
            attempts = await self._state.attempts(id)
            if attempts is None:
                return "unknown"
            if isinstance(attempts, Dropped):
                # told only to the eval the request was of
                return "dropped" if attempts.eval_id == eval_id else "unknown"
            if any(attempt.eval_id != eval_id for attempt in attempts):
                return "unknown"
            latest = attempts[-1]
            left = deadline - time.monotonic()
            if latest.state != "judging" or left <= 0:
                break
            # woken at once by a decision, and looking again now and then,
            # as a retry of the request may have begun another attempt
            with anyio.move_on_after(min(left, _RESULT_POLL)):
                await latest.settled.wait()
        records: list[Record] = []
        for attempt in attempts:
            kept = await self._state.records(attempt.exchange)
            if kept is None:
                return "dropped"
            records.extend(kept)
        return RequestResult(attempts[-1], records)

    async def sample_result(
        self, sample_uuid: str, eval_id: str
    ) -> SampleResult | None:
        """What an eval's sample came to, and the sentinel's store, for the eval to copy into the sample's store before scoring. The handler keeps one run for each sample in each epoch, and its store and end are those of the sample's latest attempt.

        Args:
            sample_uuid: The sample's uuid, as the eval names it.
            eval_id: The eval asking. A sample of another eval reads as unknown.

        Returns:
            The sample's outcome and store, or None if nothing is kept for it, or it isn't the eval's.
        """
        # renewed only once it is found to be the eval's, so that another
        # eval asking keeps nothing alive
        key = _run_key(True, sample_uuid)
        kept = await self._state.existing_run(key, renew=False)
        if kept is None or kept.eval_id != eval_id:
            return None
        await self._state.existing_run(key)
        return SampleResult(
            "terminate" if kept.refusal is not None else "proceed",
            dict(kept.store.items()),
        )

    async def end_sample(
        self, sample_uuid: str, eval_id: str
    ) -> Literal["ended", "unknown"]:
        """Drop the state kept for an eval's sample, once the eval has scored it. A later request of the sample is refused, as for a sample a sentinel ended.

        Args:
            sample_uuid: The sample's uuid, as the eval names it.
            eval_id: The eval ending it. A sample of another eval is left as it is, and reads as unknown.

        Returns:
            `ended`, or `unknown` for a sample the handler never saw or that isn't the eval's. Ending a sample again has no effect.
        """
        ended = await self._state.end_run(_run_key(True, sample_uuid), eval_id)
        return "ended" if ended else "unknown"

    async def _within_limit(
        self, judging: Awaitable[Answer], what: str, id: str
    ) -> Answer:
        with anyio.move_on_after(self._time_limit):
            return await judging
        raise self._out_of_time(what, id)

    async def records(self, id: str) -> list[Record] | None:
        """What was recorded while a request and its reply were judged.

        Args:
            id: The request's id, as the proxy gave it.

        Returns:
            The records in order, the request's before the reply's, or None if the request was never seen or its records are no longer kept.
        """
        return await self._state.records(id)

    async def register_task(
        self, eval_id: str, registration: TaskRegistration, expiry: float | None = None
    ) -> float | None:
        """Keep a task an eval registered, so that its runs' sentinels are told which sample each is, once each sample's registration is complete.

        A registration replaces the eval's last one for the task, and keeps the samples whose registration was complete that it still names, so that registering again has no effect. A run that has already found its sample keeps what it found.

        Args:
            eval_id: The eval's id, as its requests name it.
            registration: The task, and the samples it runs.
            expiry: How long to keep it once no request has named the task, in seconds. A day if not given.

        A task registered beforehand is already registered, so its registration stores nothing and is not compared with the file, since the answer would tell a caller whether its guess at the file is right. This holds whether or not the handler takes registrations from evals.

        Returns:
            How long it is kept once no request has named the task: `expiry`, or less if that is longer than the handler keeps any. None for a task registered beforehand, which is never dropped.

        Raises:
            RegistrationError: If the handler takes no registrations from evals and the task was not registered beforehand.
            RegistrationsFullError: If the handler already keeps as many registrations as it takes.
        """
        if registration.task in self._tasks:
            return None
        if not self._runtime_task_registration:
            raise RegistrationError(
                "This sidecar takes no tasks registered by an eval."
            )
        granted = min(expiry if expiry is not None else TASK_EXPIRY, self._task_expiry)
        kept = RegisteredTask(registration)
        before = await self._state.registration(eval_id, registration.task)
        if before is not None:
            kept.keep_samples(
                {
                    id: sample
                    for id, sample in before.samples.items()
                    if id in kept.sample_ids
                }
            )
        if kept.held > self._max_registration_bytes:
            raise self._full(registration.task, kept.held)
        if not await self._state.register(
            eval_id, registration.task, kept, granted, self._max_task_registrations
        ):
            raise RegistrationsFullError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes. One is taken again once a kept task expires."
            )
        return granted

    async def complete_registration(
        self, eval_id: str, task: str, sample: SampleFacts
    ) -> bool:
        """Keep a sample's facts, before the sample's first epoch, so that its runs' sentinels are told them.

        Completing a sample's registration again replaces its facts, for the runs that start after. A sample the task's registration doesn't name, such as one a task adds while it runs, extends the registration, unless the handler is `registered_only`. A task registered beforehand already holds its samples, so nothing is stored, and the sample is not compared with the file, since the answer would tell a caller whether its guess at the file is right.

        Args:
            eval_id: The eval's id, as its requests name it.
            task: The task's name.
            sample: The sample.

        Returns:
            Whether the task was registered beforehand, so nothing was stored.

        Raises:
            UnknownTaskError: If the eval hasn't registered the task, or its registration is no longer kept.
            RegistrationError: If the handler is `registered_only` and the task's registration doesn't name the sample.
            RegistrationsFullError: If the task's registration would then hold more than the handler takes for a task.
        """
        if task in self._tasks:
            return True
        kept = await self._state.registration(eval_id, task)
        if kept is None:
            raise UnknownTaskError(
                f"This sidecar holds no registration of the task {task!r} by the eval {eval_id!r}. Register the task, then complete the sample's registration."
            )
        if sample.sample_id not in kept.sample_ids:
            if self._strictness == "registered_only":
                raise RegistrationError(
                    f"This sidecar takes only the samples a task's registration names, and the registration of {task!r} doesn't name the sample {sample.sample_id!r}."
                )
        held = kept.holding(sample)
        if held > self._max_registration_bytes:
            raise self._full(task, held)
        kept.complete(sample)
        return False

    def _full(self, task: str, held: int) -> RegistrationsFullError:
        return RegistrationsFullError(
            f"The registration of {task!r} would hold {held} bytes, and this sidecar keeps no more than {self._max_registration_bytes} for a task: its registration, and the facts of each sample whose registration is complete."
        )

    def _view(self, call: ToolCall, run: Run) -> ToolCallView:
        # A request names a tool only by the name the model sees. A task
        # registered beforehand can name the tool that is; otherwise the name
        # is taken for inspect_ai's own tool of that name, if it has a viewer.
        task = self._tasks.get(run.eval.task) if run.eval is not None else None
        tool = task.tools.get(call.function) if task is not None else None
        if tool is None and call.function in VIEWED_TOOLS:
            tool = f"inspect_ai/{call.function}"
        return tool_call_view(call, tool)

    async def _eval(self, sample: InspectSample) -> EvalContext | None:
        # the sample a request names, from the task's registration; asking
        # counts as the task being named, so its registration is kept
        fixed = self._tasks.get(sample.task)
        if fixed is None:
            kept = await self._state.registration(sample.eval_id, sample.task)
            if kept is None:
                if self._strictness != "open":
                    raise UnregisteredSampleError(
                        f"This sidecar judges only the requests of tasks it holds, and holds no task {sample.task!r} of the eval {sample.eval_id!r}."
                    )
                return None
            found = kept.eval_context(sample)
            if self._strictness == "registered_only":
                if found is None:
                    raise UnregisteredSampleError(
                        f"This sidecar judges only the samples it holds, and the registration of {sample.task!r} isn't complete for the sample {sample.sample_id!r}."
                    )
                if not 1 <= sample.epoch <= kept.registration.epochs:
                    raise UnregisteredSampleError(
                        f"The registration of {sample.task!r} runs epochs 1 to {kept.registration.epochs}, not epoch {sample.epoch}."
                    )
            return found
        found = fixed.eval_context(sample)
        if found is None:
            raise UnregisteredSampleError(
                f"The task {sample.task!r} is registered with this sidecar beforehand, without the sample {sample.sample_id!r}."
            )
        return found

    async def _request(
        self, request: Request, sample: InspectSample | None, attempt: Attempt | None
    ) -> Answer:
        if sample is None and self._strictness == "registered_only":
            raise UnregisteredSampleError(
                f"This sidecar judges only the requests of samples it holds, and the request has no {INSPECT_CONTEXT} header."
            )
        proxy = ProxyContext(
            provider=request.provider,
            model=request.model,
            user=request.user,
            headers=request.headers,
            inspect=sample,
        )
        opening = _opening(
            await read_opening(request.provider, request.body, request.format)
        )
        # an Inspect eval names the run: its sample, across retries
        named = sample is not None
        key = sample.sample_uuid if sample is not None else opening
        sample_attempt = sample.attempt if sample is not None else ""
        # a conversation is told apart within a named run by the attempt it
        # is part of, since a retry opens as the attempt it replaces did, and
        # then by how it opens
        conversation_key = f"{key}/{sample_attempt}/{opening}" if named else key
        run_key = _run_key(named, key)
        run = await self._state.run(run_key)
        restored = False
        if sample is not None:
            clash = _begin_attempt(run, sample)
            if clash is not None:
                return self._judged(clash, proxy, Exchange(replaced=True), attempt)
            if run.eval_id is None:
                run.eval_id = sample.eval_id
            # the eval holds the calls a decision acted on without their
            # arguments: the model's own go back before anything reads them
            written = request_format(request.provider, request.body, request.format)
            body = unredacted(
                request.provider,
                request.body,
                {
                    id: call
                    for id, (format, call) in run.redacted.items()
                    if format == written
                },
                request.format,
            )
            if body is not None:
                request = replace(request, body=body)
                restored = True
        marks = request_marks(request.provider, request.body, request.format)
        refusal = _ended(run, marks, sample_attempt, request)
        if refusal is not None:
            # the run has ended, and the sentinel is not asked again
            return self._judged(refusal, proxy, Exchange(), attempt)
        # the attempt's store, taken before anything below awaits, since a
        # retry may replace the run's store meanwhile
        store = run.store
        if sample is not None:
            found = await self._eval(sample)
            if run.eval is None:
                # a run keeps the facts it first found
                run.eval = found
        conversation = run.conversations.setdefault(conversation_key, Conversation())
        messages = await read_request(
            request.provider, request.body, request.format, start=conversation.noted
        )
        sends = offered(request.provider, request.body, request.format)

        answer: Answer = Replace(request.body) if restored else Pass()
        # only what the client sent holds results to judge: what the sidecar
        # adds below is its own writing
        client = messages
        # a conversation that holds two calls with one id can't be read, as
        # which of the two a result is of can't be told
        _calls(client)
        # A point this request doesn't continue from is kept for the requests
        # that do: a run's key is shared by conversations that open alike. A
        # client that was handed a rejected turn, as an Inspect eval is in the
        # judgment it asked for, may already have put it back itself.
        holds = {_digest(message) for message in messages}
        continued = [
            rejected
            for rejected in conversation.rejected
            if rejected.sent < len(marks) and marks[rejected.sent] == rejected.mark
        ]
        told = {
            rejected.sent: turns
            for rejected in continued
            if rejected.added
            and (turns := [t for t in rejected.turns if _digest(t[0]) not in holds])
        }
        # the rejections of refused turns the client put back itself
        refused = {
            id: amendment
            for rejected in continued
            for turn, written in zip(rejected.turns, rejected.written, strict=True)
            if _digest(turn[0]) in holds
            for id, amendment in written.items()
        }
        edited = [
            turn
            for turn in conversation.edited
            if turn.sent < len(marks) and marks[turn.sent] == turn.mark
        ]
        # A request that holds a call of an edited turn, which isn't held where
        # the turn was made, as when a client rewrote earlier entries of its
        # conversation, can't be matched with the edit: its results would be
        # judged as of a call that didn't run.
        if _edit_lost(run, edited, client):
            raise UnreadableError(
                "The request holds a call this sidecar changed or took out, after earlier entries of its conversation changed, so which call ran can't be told."
            )
        amended = await amend_request(
            request.provider,
            request.body,
            inserted=told,
            edited=edited,
            format=request.format,
        )
        if amended.body is not None:
            messages = await read_request(
                request.provider,
                amended.body,
                request.format,
                start=conversation.noted,
            )
            answer = Replace(amended.body)
        # what the model is sent, each message the conversation sent before
        # under the id it had then, as in Inspect's own process
        body = amended.body if amended.body is not None else request.body
        messages = with_sent_ids(conversation, messages, body)

        # the last entry in digests, for a run no eval names, so that a run
        # ended at this request remembers it without keeping what it holds
        digests = (
            []
            if named
            else entry_digests(request.provider, request.body, request.format)
        )
        last = digests[-1] if digests else None
        held = Held(
            run=run_key,
            conversation=conversation_key,
            named=named,
            sample_attempt=sample_attempt,
            answered=any(mark in run.answered for mark in marks[:-1]),
            model=_step_model(request.provider, request.model, sample),
            format=request_format(request.provider, request.body, request.format),
            recorder=await self._state.recorder(request.id),
            input=messages,
            offered=sends,
            marks=marks,
            proxy=proxy,
            store=store,
            attempt=attempt,
            last=last,
            request=None if named else request,
        )
        said = call_texts(request.provider, request.body, request.format)
        # a changed call's result is of the call that ran, whichever the client holds
        ran = {id: call for turn in edited for id, call in turn.modified.items()}
        restored = amended.restored | held_rejections(
            request.provider, request.body, refused, request.format
        )
        ended = await self._results(
            held, run, conversation, client, said, restored, ran
        )
        if ended is not None:
            return self._judged(ended, proxy, held.exchange, attempt)
        if BeforeGenerate in self._watched:
            before = BeforeGenerate(
                model=held.model,
                conversation=conversation_key,
                input=messages,
                # the agent's own message list is not known behind a proxy
                agent_messages=None,
                # what was sent before this request, which isn't sent yet
                history=list(conversation.sent),
                tools=sends.tools,
                tool_choice=sends.tool_choice,
                config=sends.config,
            )
            if _ends(await self._judge(held, run, before)):
                return self._judged(
                    self._end(held, run, "before the model was called"),
                    proxy,
                    held.exchange,
                    attempt,
                )
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, marks, sample_attempt, request)
        if refusal is not None:
            return self._judged(refusal, proxy, held.exchange, attempt)
        # the request goes on to the model, as the client sent it with what
        # this sidecar added
        record_sent(conversation, messages, body)
        await self._state.hold(request.id, held)
        # an Inspect client is told of the request's steps with the reply
        return answer

    async def _results(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        client: list[ChatMessage],
        said: Mapping[str, str],
        restored: frozenset[str] = frozenset(),
        ran: Mapping[str, ToolCall] | None = None,
    ) -> Refuse | None:
        if AfterToolCall not in self._watched:
            return None
        made: dict[str, tuple[ChatMessageAssistant, ToolCall, str]] = {}
        for entry in client:
            if isinstance(entry, ChatMessageAssistant):
                for call in entry.tool_calls or []:
                    made[call.id] = (entry, call, said.get(call.id, entry.text))
            if not isinstance(entry, ChatMessageTool):
                continue
            found = made.get(entry.tool_call_id or "")
            if found is None:
                raise UnreadableError(
                    f"The request holds a result for a call it doesn't hold: {entry.tool_call_id!r}."
                )
            assistant, call, words = found
            call = (ran or {}).get(call.id, call)
            digest = _result_digest(words, call, entry)
            key = (held.run, held.conversation, digest)
            if key in self._judging:
                while (judging := self._judging.get(key)) is not None:
                    await judging.wait()
                # the judgment waited for may have ended the run
                refusal = _ended(run, held.marks, held.sample_attempt, held.request)
                if refusal is not None:
                    return refusal
            # a rejection the client put back itself is the sidecar's own
            # writing, not the result of a call that ran
            if digest in conversation.judged or entry.tool_call_id in restored:
                continue
            sent = conversation.made.get(call.id)
            # a call this sidecar did not hand out has only the id its
            # reading of the request gave it
            message_id = sent.message_id if sent is not None else assistant.id
            assert message_id is not None
            step = AfterToolCall(
                conversation=held.conversation,
                message=assistant.text,
                message_id=message_id,
                call=call,
                result=entry,
                # not known: only what the model is sent crosses the wire, and
                # a monitor must not take that for everything the tool returned
                output=None,
                view=self._view(call, run),
                input=sent.input if sent is not None else None,
                agent_messages=None,
                # as before the next generate: without what this request sends
                history=list(conversation.sent),
                tools=sent.tools if sent is not None else None,
                tool_choice=sent.tool_choice if sent is not None else None,
            )
            judging = self._judging[key] = anyio.Event()
            try:
                if _ends(await self._judge(held, run, step)):
                    # not marked as judged, so a client that sends it again
                    # is refused again
                    return self._end(
                        held, run, f"at the result of the call to {call.function}"
                    )
                conversation.judged.add(digest)
            finally:
                del self._judging[key]
                judging.set()
        return None

    async def _reply(
        self, reply: Reply, held: Held | None, released: bytes = b""
    ) -> Answer:
        answer = await self._judge_reply(reply, held, released)
        if held is None or not succeeded(reply.status):
            return answer
        # an Inspect eval asks what was decided, by `request_result`
        if held.proxy.inspect is not None and held.attempt is not None:
            held.attempt.judged = held.exchange
        return answer

    async def _judge_reply(
        self, reply: Reply, held: Held | None, released: bytes = b""
    ) -> Answer:
        if not succeeded(reply.status):
            # nothing came from the model to judge
            return Pass()
        if held is None:
            raise UnreadableError(
                f"The reply {reply.id!r} answers a request this sidecar was never shown, or no longer keeps."
            )
        run = await self._state.run(held.run)
        conversation = run.conversations.setdefault(held.conversation, Conversation())
        # what the reply is read from, which an edit of it is read back from
        start = conversation.noted
        output, noted = await read_reply_noted(
            reply.provider,
            reply.content_type,
            reply.body,
            reply.format,
            start=conversation.noted,
        )
        # kept whatever is decided: the provider continues from what it sent
        conversation.noted = noted
        # billed whatever is decided, so a client refused the reply still counts it
        held.exchange.usage = output.usage
        refusal = _ended(run, held.marks, held.sample_attempt, held.request)
        if refusal is not None:
            # a sentinel ended the run while the model was answering, or a
            # later attempt replaced this one
            held.exchange.ended = True
            held.exchange.replaced = refusal is _REPLACED
            return refusal

        if AfterGenerate in self._watched:
            after = AfterGenerate(
                model=held.model,
                conversation=held.conversation,
                input=held.input,
                agent_messages=None,
                # as the core gives it: what the conversation has sent by
                # now, another request sent meanwhile included
                history=list(conversation.sent),
                tools=held.offered.tools,
                tool_choice=held.offered.tool_choice,
                config=held.offered.config,
                output=output,
            )
            if _ends(await self._judge(held, run, after)):
                if output.choices:
                    _withhold_ended(held, output.message)
                return self._end(held, run, "at the model's reply")

        if not output.choices:
            return self._ended_meanwhile(held, run) or self._passed(
                held, run, conversation, None
            )
        message = output.message
        # the reply is still read, and kept as passed, when no call is judged
        calls = (message.tool_calls or []) if BeforeToolCall in self._watched else []
        rejected: list[tuple[ToolCall, Decision]] = []
        modified: dict[str, ToolCall] = {}
        for call in calls:
            assert message.id is not None
            step = BeforeToolCall(
                conversation=held.conversation,
                message=message.text,
                message_id=message.id,
                call=call,
                view=self._view(call, run),
                input=held.input,
                agent_messages=None,
                history=list(conversation.sent),
                tools=held.offered.tools,
                tool_choice=held.offered.tool_choice,
            )
            decision = await self._judge(held, run, step)
            if decision is None or decision.action == "proceed":
                continue
            if decision.action == "terminate":
                _withhold_ended(held, message, call.id, rejected, modified)
                return self._end(held, run, f"at the call to {call.function}")
            if decision.action == "modify" and decision.modified is not None:
                modified[call.id] = decision.modified
            else:
                rejected.append((call, decision))
        refusal = self._ended_meanwhile(held, run)
        if refusal is not None:
            return refusal
        if not rejected and not modified:
            return self._passed(held, run, conversation, message)
        taken = {call.id for call, _ in rejected}
        if all(call.id in taken for call in message.tool_calls or []):
            return await self._reject(held, run, conversation, message, noted, rejected)
        try:
            return await self._edit(
                reply,
                held,
                run,
                conversation,
                output,
                start,
                noted,
                rejected,
                modified,
                released=released,
            )
        except UnreadableError as ex:
            if not rejected:
                raise ModifyError(
                    "The sentinel changed a call this sidecar can't change in the reply."
                ) from ex
            # the reply couldn't be edited as the sentinel decided, so the turn
            # is refused whole, as it was before calls could be taken out
            return await self._reject(held, run, conversation, message, noted, rejected)

    def _passed(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant | None,
    ) -> Pass:
        run.in_a_row.pop(held.sample_attempt, None)
        if not held.named:
            run.answered.add(held.mark)
        if message is not None:
            for call in message.tool_calls or []:
                assert message.id is not None
                conversation.made[call.id] = Made(
                    held.input,
                    held.offered.tools,
                    held.offered.tool_choice,
                    message.id,
                )
            while len(conversation.made) > MADE_KEPT:
                del conversation.made[next(iter(conversation.made))]
        return Pass()

    async def _edit(
        self,
        reply: Reply,
        held: Held,
        run: Run,
        conversation: Conversation,
        output: ModelOutput,
        start: JsonValue | None,
        noted: JsonValue | None,
        rejected: Sequence[tuple[ToolCall, Decision]],
        modified: Mapping[str, ToolCall],
        *,
        released: bytes = b"",
    ) -> Replace:
        # The rejected calls are taken out and the changed calls go out
        # changed, and the rest of the turn goes on, as in Inspect's own
        # process, where the other calls in a turn still run. The model's
        # message keeps its own calls there, so later requests have them put
        # back.
        message = output.message
        reasons = {
            call.id: decision.message or _REJECTED for call, decision in rejected
        }
        written = await amendments(held.format, message, reasons, modified, noted)
        if held.named:
            for id in reasons:
                run.redacted[id] = (held.format, written[id].call)
        body = await edit_reply(
            reply.provider,
            reply.content_type,
            reply.body,
            reply.format,
            Edits(frozenset(reasons), modified),
            output,
            start=start,
        )
        if not body.startswith(released):
            # what has gone out can't be taken back
            raise UnreadableError(
                "The reply as edited changes what the client was already sent."
            )
        calls = message.tool_calls or []
        conversation.edited.append(
            Edited(
                sent=held.sent,
                mark=held.mark,
                order=tuple(call.id for call in calls),
                taken_out=reasons,
                originals={call.id: call for call in calls if call.id in modified},
                modified=dict(modified),
                amendments=written,
            )
        )
        # an eval is handed neither a call taken out nor a changed call's
        # original arguments
        held.exchange.proposed = Withheld(
            message, reasons, redacted=frozenset(reasons), ran=dict(modified)
        )
        handed = message.model_copy(
            update={
                "tool_calls": [
                    modified.get(call.id, call)
                    for call in calls
                    if call.id not in reasons
                ]
            }
        )
        # some of the turn ran, which is progress, so it counts as a pass
        self._passed(held, run, conversation, handed)
        return Replace(body)

    async def _reject(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant,
        noted: JsonValue | None,
        rejected: Sequence[tuple[ToolCall, Decision]],
    ) -> Refuse:
        # only `message` is for the agent; a decision's explanation is for the log
        reasons = {
            call.id: decision.message or _REJECTED for call, decision in rejected
        }
        told = {
            other.id: reasons.get(other.id, _NOT_RUN)
            for other in message.tool_calls or []
        }
        # the first rejection is the one the client is refused with
        call, _ = rejected[0]
        why = reasons[call.id]
        count = run.in_a_row.get(held.sample_attempt, 0) + 1
        ends = count >= _MAX_CONSECUTIVE_REJECTIONS
        # made before anything is kept, since making it can fail, and a turn
        # kept without it would leave every later request of the conversation
        # unreadable
        written = (
            None if ends else await amendments(held.format, message, told, (), noted)
        )
        held.exchange.withheld = Withheld(message, told, redacted=frozenset(reasons))
        held.exchange.message = why
        run.in_a_row[held.sample_attempt] = count
        if written is not None and held.named:
            for id in reasons:
                run.redacted[id] = (held.format, written[id].call)
        if written is None:
            return self._end(
                held,
                run,
                f"at the call to {call.function} after {_MAX_CONSECUTIVE_REJECTIONS} rejected turns in a row",
            )
        # a client that comes back after a rejection sends the same request, so
        # its rejected turns in a row share a point
        point = next(
            (
                known
                for known in conversation.rejected
                if (known.sent, known.mark) == (held.sent, held.mark) and known.added
            ),
            None,
        )
        if point is None:
            point = Rejected(held.sent, held.mark, [])
            conversation.rejected.append(point)
        point.turns.append((message, told, noted))
        point.written.append(written)
        return Refuse(
            self._reject_status,
            f"A sentinel rejected the call to {call.function}: {why}",
        )

    async def _judge(self, held: Held, run: Run, step: Step) -> Decision | None:
        recorded = len(held.recorder.records)
        # which models judged the step, for the eval: its roles don't reach here
        with noting_models() as models:
            decision = await run_sentinel(
                self._protocol,
                HostContext(
                    host=RunHost(self._host, run.eval),
                    recorder=held.recorder,
                    store=held.store,
                    proxy=held.proxy,
                ),
                step,
            )
        call = (
            step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else None
        )
        held.exchange.judged.append(
            Judged(
                stage(step),
                call,
                decision,
                held.recorder.records[recorded:],
                tuple(models),
            )
        )
        return decision

    def _judged(
        self,
        refusal: Refuse,
        proxy: ProxyContext,
        exchange: Exchange,
        attempt: Attempt | None,
    ) -> Refuse:
        # an Inspect eval asks what was decided, by `request_result`; a
        # request of a replaced attempt was never judged
        exchange.replaced = exchange.replaced or refusal is _REPLACED
        if proxy.inspect is not None and attempt is not None:
            attempt.judged = exchange
        return refusal

    def _ended_meanwhile(self, held: Held, run: Run) -> Refuse | None:
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, held.marks, held.sample_attempt, held.request)
        if refusal is not None:
            held.exchange.ended = True
            held.exchange.replaced = refusal is _REPLACED
        return refusal

    def _end(self, held: Held, run: Run, where: str) -> Refuse:
        held.exchange.ended = True
        refusal = Refuse(400, f"A sentinel ended this run {where}.")
        run.in_a_row.pop(held.sample_attempt, None)
        if held.named:
            # a judgment that outlived its attempt ends only that attempt,
            # which nobody waits for
            if held.sample_attempt != run.attempt:
                return refusal
            run.refusal = refusal
            # nothing of the run goes on, so nothing is put back again
            for conversation in run.conversations.values():
                conversation.rejected.clear()
                conversation.edited.clear()
            return refusal
        if held.answered:
            run.ended_at[held.mark] = refusal
            if held.sent > 0 and held.last is not None:
                run.ended_on.append(
                    EndedOn(
                        held.sent - 1, held.marks[held.sent - 1], held.last, refusal
                    )
                )
        # A run no eval names, ended before any reply in its
        # conversation was passed on, is not remembered as ended. Its
        # conversation is still only how it opens, which is all another
        # conversation with the same key has too, and nothing tells them apart.
        # Other conversations with the same key go on, and still need their
        # turns put back, or told from results. So the rejected turns where
        # this one ended are only no longer added to a request sent there.
        conversation = run.conversations.get(held.conversation)
        for point in conversation.rejected if conversation is not None else []:
            if (point.sent, point.mark) == (held.sent, held.mark):
                point.added = False
        return refusal

    def _out_of_time(self, what: str, id: str) -> TimeLimitError:
        return TimeLimitError(
            f"The sentinel didn't finish judging the {what} {id!r} within {self._time_limit} seconds."
        )


def _withhold_ended(
    held: Held,
    message: ChatMessageAssistant,
    ended_at: str | None = None,
    rejected: Sequence[tuple[ToolCall, Decision]] = (),
    modified: Mapping[str, ToolCall] | None = None,
) -> None:
    # The client records the turn the run ended at, as an in-process sentinel
    # leaves it in the transcript. None of its calls runs, so a changed call
    # isn't shown as it ran. Every call a decision acted on is redacted: the
    # one the run ended at, and those rejected or changed before it. A run
    # ended at the reply had every call acted on. A call judged to proceed, or
    # never judged, keeps its arguments.
    calls = message.tool_calls or []
    acted_on: list[str]
    if ended_at is None:
        acted_on = [call.id for call in calls]
    else:
        acted_on = [ended_at, *(call.id for call, _ in rejected), *(modified or {})]
    redacted = frozenset(acted_on)
    held.exchange.withheld = Withheld(
        message, {call.id: _ENDED for call in calls}, redacted=redacted
    )


def _run_key(named: bool, key: str) -> str:
    # Runs an eval names and runs it doesn't are kept apart. A run no eval
    # names is keyed by how its conversation opens, which anyone who knows the
    # opening can work out, so the eval endpoint, which takes a sample's uuid,
    # must never reach one.
    return f"{'sample' if named else 'opening'}:{key}"


def _begin_attempt(run: Run, sample: InspectSample) -> Refuse | None:
    # A later attempt at the sample starts the run afresh: its model never saw
    # the last attempt's turns, and in Inspect's own process its sample store
    # starts empty. An earlier attempt's request is refused by `_ended`, and
    # another attempt with the same number is refused here, as which of them
    # is the sample's can't be told.
    if run.closed:
        return None
    if sample.attempt_number > run.attempt_number:
        run.attempt = sample.attempt
        run.attempt_number = sample.attempt_number
        run.store = Store()
        run.refusal = None
    elif sample.attempt_number == run.attempt_number and sample.attempt != run.attempt:
        return Refuse(
            400,
            f"Another attempt at this sample has the number {sample.attempt_number}.",
        )
    return None


def _ended(
    run: Run, marks: Sequence[str], attempt: str, request: Request | None = None
) -> Refuse | None:
    # what a request is refused with, if its run has ended: any request of a
    # run an eval names, or one that continues the conversation it was ended
    # in; and any request or reply of an attempt a later one replaced
    if not run.closed and attempt != run.attempt:
        return _REPLACED
    refusal = run.refusal or next(
        (run.ended_at[mark] for mark in marks if mark in run.ended_at), None
    )
    if refusal is not None or request is None:
        return refusal

    # A client may add the user's next message to the entry the ended
    # request ended with, as Claude Code adds it beside the last results,
    # rather than send it as an entry of its own. That request continues the
    # ended conversation too. The request is read once, whatever the run was
    # ended at.
    if not run.ended_on:
        return None
    digests = entry_digests(request.provider, request.body, request.format)
    for ended in run.ended_on:
        if ended.at >= len(marks) or marks[ended.at] != ended.before:
            continue
        if ended.at < len(digests) and (now := digests[ended.at]) is not None:
            if grows(now, ended.digest):
                return ended.refusal
    return None


def _ends(decision: Decision | None) -> bool:
    # an escalate nobody took has already been turned into a terminate
    return decision is not None and decision.action == "terminate"


def _model(provider: str, model: str) -> str:
    # a step names the model as `provider/name`
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


def _step_model(provider: str, model: str, sample: InspectSample | None) -> str:
    # Inspect's own name for the model, as in its process, where an Inspect
    # client gives one that ends in the model the request names: it can add
    # what precedes the name, such as `openai-api/together/`, and not name
    # another model
    claimed = sample.model if sample is not None else None
    if claimed is not None and model and claimed.endswith(f"/{model}"):
        return claimed
    return _model(provider, model)


def _opening(messages: Sequence[ChatMessage]) -> str:
    # How the conversation opens: what tells a run apart when the proxy
    # doesn't name it, and a conversation apart within a run. Conversations
    # that open alike share it.
    opening = [m for m in messages if m.role == "system"][:1] + [
        m for m in messages if m.role == "user"
    ][:1]
    digest = hashlib.sha256("\x00".join(m.text for m in opening).encode())
    return digest.hexdigest()[:16]


def _digest(message: ChatMessage) -> str:
    # What makes two messages the same turn. A reply the sidecar read, and the
    # same reply sent back in a later request, are read with different ids,
    # sources and provider details, so only what the turn says is compared.
    calls = [
        [call.id, call.function, call.arguments]
        for call in (
            message.tool_calls if isinstance(message, ChatMessageAssistant) else None
        )
        or []
    ]
    result = message.tool_call_id if isinstance(message, ChatMessageTool) else None
    return hashlib.sha256(
        json.dumps(
            [message.role, message.text, calls, result], sort_keys=True, default=str
        ).encode()
    ).hexdigest()


def _edit_lost(
    run: Run, edited: Sequence[Edited], messages: Sequence[ChatMessage]
) -> bool:
    # whether the messages hold a call id of an edited turn of the run that
    # isn't among those the request continues from
    ids = _calls(messages) | {
        message.tool_call_id
        for message in messages
        if isinstance(message, ChatMessageTool) and message.tool_call_id is not None
    }
    return any(
        turn not in edited and ids.intersection(turn.order)
        for conversation in run.conversations.values()
        for turn in conversation.edited
    )


def _calls(messages: Sequence[ChatMessage]) -> set[str]:
    # the ids of the tool calls the messages hold. Separate calls have
    # separate ids, so a conversation that holds two calls with one id can't
    # be read: which of them a result is of can't be told.
    calls: set[str] = set()
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            for call in message.tool_calls or []:
                if call.id in calls:
                    raise UnreadableError(
                        f"The request holds two tool calls with the id {call.id!r}."
                    )
                calls.add(call.id)
    return calls


def _result_digest(said: str, call: ToolCall, result: ChatMessageTool) -> str:
    # everything the step is given from the request, what the model said with
    # the call as well as the call and the result: the client wrote them all
    # into the request, so an id seen before doesn't vouch for what now
    # stands beside it
    digest = hashlib.sha256(
        json.dumps(
            [
                said,
                call.id,
                call.type,
                call.function,
                call.arguments,
                call.parse_error,
            ],
            sort_keys=True,
        ).encode()
    )
    digest.update(b"\x00")
    digest.update(result.model_dump_json(exclude={"id"}).encode())
    return digest.hexdigest()
