from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import anyio
import anyio.abc
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolCallView
from inspect_ai.util import Store
from pydantic import JsonValue

from .._context import EvalContext, InspectSample, ProxyContext
from .._host import HostNative, HumanAnswer
from .._integration import HostContext, resolve_sentinel, run_sentinel
from .._report import Decision, Escalations
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
from ._host import (
    AskHuman,
    ModelUsed,
    Record,
    RunHost,
    SidecarHost,
    add_usage,
    noting_models,
    stage,
)
from ._inspect_ai import VIEWED_TOOLS, tool_call_view
from ._inspect_client import (
    INSPECT_CONTEXT,
    REQUEST_ID,
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
from ._review import Question, Review, Reviews, escalated
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
    Pinned,
    Registered,
    Rejected,
    Run,
    State,
    Stopped,
)
from ._wire import (
    Edited,
    Edits,
    Format,
    UnreadableError,
    amend_request,
    amendments,
    bound_body,
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

logger = logging.getLogger(__name__)

# rejected turns in a row before the run is ended, as inspect_ai's bridge has
# it: a model that keeps proposing rejected calls would otherwise never stop
MAX_RESULT_WAIT = 30.0
"""The longest `request_result` waits for a request still being judged, in seconds."""

# how often a waiting `request_result` looks again for a new attempt
_RESULT_POLL = 0.05

_MAX_CONSECUTIVE_REJECTIONS = 3

_REPLACED = Refuse(400, "A later attempt at this sample replaced this one.")

PENDING = Refuse(
    400,
    "This model call is waiting for a person's review. Nothing was decided yet. The eval reads where the call stands with request_result.",
)
"""What a model call a sentinel asked a person about is answered with, and every try of it sent while it waits. No connection waits for the person."""

# what a request or reply is refused with when its judging is stopped, with
# nothing decided, or when its model call was closed before it came
_RUN_OVER = Refuse(400, "This run ended while the request waited for a person.")
_CANCELLED = Refuse(
    400, "The eval cancelled this model call while it waited for a person."
)
_CLOSED = Refuse(
    400,
    "This reply's model call was closed while the provider answered. The reply was not judged.",
)
_OTHER_REQUEST = Refuse(
    400,
    f"Another request with this {REQUEST_ID} waits, or waited, for a person's review, and this is not the same request. It was not judged.",
)
_OVER = Refuse(
    400,
    "This model call has ended. The eval reads what it came to with request_result.",
)
FAILED_TRY = "x-sentinel-failed-try"
"""The header the sidecar adds to a provider's error it serves from the held-reply path: the number of the failed try the error is of. No refusal of the sidecar's own has it, so an eval can tell a provider's error from one."""

_NO_CALL_NAMED = Refuse(
    400,
    f"The request has no {INSPECT_CONTEXT} header, so it names no model call whose held reply it asks for.",
)
_NO_HELD_REPLY = Refuse(
    400,
    f"This sidecar holds no reply for a model call of the eval with this {REQUEST_ID}: no person was asked about the call, or what was held for it is no longer kept. The eval reads where a call stands with request_result.",
)
_NOT_ENDED = Refuse(
    400,
    "This model call has not ended, so no reply is held for the eval yet. The eval reads where the call stands with request_result.",
)
_NOT_GOING_ON = Refuse(
    400,
    "This model call ended without its turn going on, so no reply is held for the eval. The eval reads what it came to with request_result.",
)

# how long what is left of a model call that waited for a person is kept once
# it has ended, its reply included, unless its sample is ended first
_SWEEP_EVERY = 60.0
"""How often the handler looks for the model calls a person was asked about that it has kept as long as it keeps anything, in seconds."""

# headers that say how a reply came, which no longer holds for the copy an
# eval reads
_NOT_KEPT = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        # the provider's cookies for the proxy's client, which an eval that
        # gets the held reply has no use for, and LiteLLM's guardrail drops
        "set-cookie",
        # the sidecar's own, which a provider can't set
        FAILED_TRY,
    }
)

_UNDER_REVIEW = Refuse(
    503,
    "A tool's result in this request is waiting for a person's review, which another model call of the sample asked for. The request was not judged. Sent again, it goes on once the review has ended.",
)
"""What a request is refused with where it holds a tool's result that another model call of its sample is paused on, for a person. The status is one a client sends the request again for by itself."""

_ANOTHER_EVALS = Refuse(
    400,
    f"The request's {INSPECT_CONTEXT} header names a sample of another eval. The request was not judged.",
)

# what inspect_ai tells the model of a rejected call that has no message
_REJECTED = "Tool call not approved."
_NOT_RUN = "This call was not run, because another call in the same turn was rejected."
_ENDED = "This call was not run, because a sentinel ended the run."


@dataclass(frozen=True)
class RequestResult:
    """What happened to a request an eval names."""

    latest: Attempt
    """The latest attempt at it."""
    records: list[Record]
    """What was recorded, every attempt's in order."""
    reviews: list[tuple[int, Review]]
    """What became of each question a sentinel asked a person, in order, each with the number of the try that asked, from 1."""
    judging_usage: dict[ModelUsed, ModelUsage]
    """What the provider billed for each model the sentinel called while judging every attempt, in the order first billed."""
    failure: Failure | None = None
    """The provider's last failed exchange, for a model call a person was asked about that may be sent on again because of it."""


@dataclass(frozen=True)
class HeldReply:
    """A provider's reply as the proxy passes it on, with the sentinel's edits in it, held for the eval whose model call waited for a person."""

    status: int
    """The provider's HTTP status."""
    headers: Mapping[str, str]
    """The provider's headers, by lowercase name, without those of one hop."""
    body: bytes
    """The body: a JSON reply whole, or a streamed reply's events from first to last."""
    format: Format
    """The API the request was written in."""
    stream: bool
    """Whether the body is a stream of events."""


@dataclass(frozen=True)
class Failure:
    """A failed exchange with the provider, for a model call a person was asked about."""

    try_number: int
    """The number of the try that was sent on, from 1."""
    reply: HeldReply | None
    """The provider's error, or None for a reply that was cut off."""
    received: float = 0.0
    """When the error came, on the monotonic clock, so that a wait the provider asked for is counted from then."""


@dataclass(eq=False)
class _Operation:
    """A model call a sentinel asked a person about: one operation, whatever tries of it are sent.

    Where it stands is the state of `attempt`, the try that first asked. A try sent later reattaches to it, and adds no attempt.
    """

    eval_id: str
    request_id: str
    """The eval's id for the model call, its `x-irid`."""
    sample: InspectSample
    binding: str
    """A digest of the request that began it. A request under the same `x-irid` with another is not a try of it."""
    attempt: Attempt
    pinned: Pinned
    """What is kept past its expiry while the operation goes on."""
    tries: int = 1
    """How many tries of the call have come."""
    sent: int = 1
    """The number of the try whose request or reply the operation holds: the one that asked, then each one sent on."""
    held: Held | None = None
    """The request, once its stage is finished, for as long as it waits to be sent on."""
    go: Pass | Replace | None = None
    """What a try that is sent on is answered with."""
    with_provider: tuple[str, Held] | None = None
    """The request of the try that was sent on, with the proxy's id for its exchange, from when it is sent on until its reply or its failure is handed over. The state keeps the request too, only as long as it keeps a request nothing names, and a call that is asked about is kept longer than that. So the call keeps the request itself, for a reply that comes after the state let it go."""
    reply: HeldReply | None = None
    """The provider's reply, once the call is decided with the turn going on, for the eval to get from the held-reply path."""
    failure: Failure | None = None
    """The provider's latest failed exchange, for as long as the call waits to be sent on again because of it."""
    ended: float | None = None
    """When it ended, on the monotonic clock. It is kept from then for as long as the state keeps a run nothing names, for the eval to read, and reading it doesn't keep it longer."""
    heard: float = 0.0
    """When the call was last heard of, on the monotonic clock: when it was opened, when it began to wait for a request or for a reply, and since then at each try of it, each time its eval asked what became of it, and each part of its reply that a proxy handed over. A call that waits, and isn't heard of for as long as the state keeps a run nothing names, is closed."""


@dataclass(eq=False)
class _Judging:
    """The judging, under way now, of a request or reply of an eval's attempt at a model call."""

    attempt: Attempt
    sample: InspectSample
    request_id: str
    binding: str
    scope: anyio.CancelScope
    """Cancelled to stop the judging with nothing decided."""
    turned: anyio.Event
    """Set once the exchange that began the judging can be answered: the judging is over, or it has asked a person."""
    limit: anyio.CancelScope | None = None
    """The time limit's scope, which a question open for a person doesn't count against."""
    asking: int = 0
    """How many questions the judging waits on now."""
    left: float | None = None
    """How much of the time limit was left when the judging began to wait on a person."""
    detached: bool = False
    """Whether the exchange was answered "pending review", so that the judging goes on with no connection open."""
    detached_at: float | None = None
    """When the exchange was answered "pending review", on the monotonic clock. Judging after it isn't time the client waited on its model call."""
    reattached: Refuse | None = None
    """What the exchange is refused with, where another try of the call asked a person first and holds the operation."""
    answer: Answer | None = None
    """What the exchange that began the judging is answered with, once the judging is over or has asked a person."""
    error: Exception | None = None
    """What the judging raised, which the exchange that began it raises in its place."""
    held: Held | None = None
    """For the judging of a reply, the request the reply answers. A model call a person was asked about goes back to waiting with it, should the judging be stopped because nobody waits for its exchange."""


# the judging this task is part of, where it is an eval's
_JUDGING: ContextVar[_Judging | None] = ContextVar("sidecar_judging", default=None)


@dataclass(eq=False)
class _ResultJudging:
    """The judging, under way now, of a tool's result that a request holds. Another request that holds the same result waits for it, and takes what it comes to."""

    woken: anyio.Event
    """Set once the judging is over, or has asked a person: a request waiting on it waits no longer for either."""
    judging: _Judging | None
    """The judging of the request that holds the result, where it is an eval's."""
    asked: bool = False
    """Whether the judging asked a person. No connection waits for a person, so a request that holds the same result is answered at once from then on."""


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
        review_timeout: float | None = None,
    ) -> None:
        """Create a handler.

        Args:
            sentinel: One protocol, or a sequence or mapping of monitors and protocols, as a task's `sentinel=` takes.
            host: What the sentinel may do to the outside world. Defaults to a `SidecarHost` with no model roles. A host with an `ask_human` of its own is asked every time; only `SidecarHost`'s, which asks nobody, gives way to asking through an eval, and only while the handler is `serving`.
            state: Where runs are kept. Defaults to this process's memory.
            reject_status: The HTTP status a rejected turn is refused with. The default, 400, is one a client takes as final. A status a client retries by itself, such as 503, brings the client back with the same request, which then goes on with the rejection added.
            time_limit: The longest the sentinel may take to judge one request or one reply, in seconds, after which it is cancelled. None, the default, is no limit.
            eval_results: Whether the judgment `request_result` gives an Inspect eval holds the results: the explanations, references and metadata of the decisions, and what each monitor and protocol recorded. Without them, it holds only what Inspect acts on. A monitor's explanation tells an agent what the monitor looks for, so an eval that can't be trusted is not sent them. Such an eval is not asked to answer for a person either: without results, a sentinel's question to a person is answered with caution, as for a run no eval names. Either way, `request_result` gives how long the judging took and what the judge models' calls cost, by model and role, for the eval's accounting.
            tasks: Tasks registered beforehand, by name, as the tasks folder's files hold them. Nothing an eval registers replaces one, and a request for a sample one doesn't hold is refused.
            runtime_task_registration: Whether an eval may register its tasks itself, at task init.
            strictness: How strict the handler is about a request it can't place. Under every strictness, a request with an `x-inspect-context` header must first pass the check of that header: it is refused, and not judged, if the handler holds no token for the eval it names, if its `timestamp` or `mac` fails the check, or if it names a run another eval started (see `request`). The strictness says what happens to a request that passes, and to one without the header. `open`, the default, judges each of those, without the task's facts where it holds none. `known_tasks` refuses a request whose `x-inspect-context` names a task the handler doesn't hold, and judges one without the header. `registered_only` refuses any request that doesn't name a sample the handler holds, in an epoch the task's registration runs, including one without the header, and refuses a sample's registration that the task's registration doesn't name.
            task_expiry: The longest a task an eval registered is kept once no request has named it, in seconds, whatever the registration asks.
            max_task_registrations: How many tasks registered by evals are kept at once, and how many evals' tokens. A registration past it is refused. It also bounds the evals whose tokens are kept, with runtime registration on or off: a token is kept until no request has named its eval for the run expiry, so an eval that has finished holds a place until then.
            max_registration_bytes: The most a task registered by an eval holds, in bytes of JSON: its registration, the facts of each sample whose registration is complete, and each sample id a sample's registration added. A sample's registration past it fails. It defaults to the largest request body, which is what a task's whole registration could hold when it came in one call.
            release: How a streamed reply is passed on. `whole`, the default, holds every reply until the sentinel has judged it. `early` passes on the text and reasoning that come before the turn's first tool call as they arrive, holds the rest of the turn, and judges it once the turn has ended, as `whole` does. The sentinel is given the whole turn either way.
            keepalive: Under early release, the longest a client is sent nothing while a streamed reply is held, in seconds, after which it is sent a keepalive, an SSE comment that every provider SDK skips. The keepalive sends the status, so a turn refused after it is refused with an error event in the stream. None sends no keepalives.
            timestamp_window: How far, in seconds, the `timestamp` of an Inspect eval's model request or call to the endpoint for evals may be from this process's clock. One further is refused, so a `mac` copied from a log is of no use once the window has passed.
            review_timeout: How long a question waits for a person's answer before it expires, in seconds, when a sentinel asks one during a run an Inspect eval names. The eval finds the question with `pending_questions` and answers it with `answer_question`. A question that expires is answered with caution, `reject` if offered and otherwise `terminate`, and the judging goes on from there. None, the default, is no expiry: a person may take days. The time a question is open is not counted against `time_limit`.

        Raises:
            ValueError: If `keepalive`, `timestamp_window` or `review_timeout` isn't a positive number of seconds.
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
        self._reviews = Reviews(review_timeout)
        self._trying: set[_Judging] = set()
        # the model calls a sentinel asked a person about, by eval and x-irid,
        # each kept until its sample ends or long after it ended
        self._operations: dict[tuple[str, str], _Operation] = {}
        # The same operations, by the attempt that asked. An attempt isn't
        # hashable by value, so it is keyed by `id()`; the operation holds the
        # attempt until it is dropped from here, so the id can't be reused
        # for another attempt meanwhile.
        self._by_attempt: dict[int, _Operation] = {}
        # the model call each open question was asked for, by question id
        self._asking: dict[str, _Operation] = {}
        # the provider's errors whose bodies are still passing, by the
        # proxy's id for the exchange, each with the try it is of
        self._failing: dict[str, tuple[_Operation, Failure]] = {}
        # where judging that outlives its exchange runs, while serving
        self._group: anyio.abc.TaskGroup | None = None
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
        # A person is asked through an eval only where the eval is sent
        # results, as an eval that isn't is one the deployment doesn't trust
        # to decide for itself. A host given with a way of its own to ask a
        # person keeps it.
        self._asks_eval = (
            eval_results and type(self._host).ask_human is SidecarHost.ask_human
        )
        self._tasks = dict(tasks or {})
        self._runtime_task_registration = runtime_task_registration
        self._strictness: Strictness = strictness
        self._task_expiry = task_expiry
        self._max_task_registrations = max_task_registrations
        self._max_registration_bytes = max_registration_bytes
        # the tool results being judged now, by run, conversation and digest,
        # so that a request holding one waits for the request judging it
        self._judging: dict[tuple[str, str, str], _ResultJudging] = {}
        # the model calls a person was asked about that have a try with the
        # provider, by the proxy's id for that try's exchange
        self._sent_on: dict[str, _Operation] = {}

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

    @property
    def eval_review(self) -> bool:
        """Whether a sentinel's question to a person is put to the Inspect eval whose run it is, for its approval UI. Not where evals aren't sent results, nor where the host asks a person its own way, nor while the handler isn't `serving`, as the judging that waits for a person then has nowhere to run."""
        return self._asks_eval and self._group is not None

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """Run the judging that outlives its exchange, for as long as the handler serves.

        A person may take days to answer a sentinel's question, so no connection waits for one. The model call is answered "pending review", and its judging goes on here, in a task of the handler's own, until it ends. Leaving stops whatever still waits. A handler that isn't serving asks nobody through an eval.
        """
        async with anyio.create_task_group() as group:
            self._group = group
            group.start_soon(self._sweeping)
            try:
                yield
            finally:
                self._group = None
                group.cancel_scope.cancel()

    @property
    def reviews(self) -> Reviews:
        """The questions a sentinel asked a person, held for Inspect evals to answer."""
        return self._reviews

    async def request(self, request: Request) -> Answer:
        """Judge a request before the proxy sends it.

        Each tool result in the request that the sentinel hasn't been shown is put to it as an `AfterToolCall` step, in order. A result is shown once, however often the client sends it. The request is then put to the sentinel as a `BeforeGenerate` step. A stage nothing in the sentinel watches is left out. If either ends the run, the request is refused.

        If turns were rejected earlier in the conversation this request continues, they are added where they happened, so the model is still told.

        A request with `x-inspect-context` belongs to the sample it names, which is its run. A request whose header can't be read is refused, rather than judged as another client's would be: it says it is an eval's, and a proxy set to let through what can't be judged would let it through unjudged.

        The header's `mac` proves the values that name the run, and the request's `x-irid`, made with the token the eval registered with. A request with the header and no `x-irid` can't be checked, and is refused as one whose header can't be read. Before anything of the run or of the `x-irid` is looked up or changed, a request is refused unjudged, whatever the strictness, if the handler holds no token for the eval, as after a restart; if its `timestamp` is outside the window or its `mac` doesn't match; or if it names a run another eval started. So a client without the eval's token that changes the sample, the attempt or its number, or the `x-irid`, or names a sample of the eval, reaches no run and changes no request's history.

        A request is kept under its `x-irid`, for `request_result`, only once its header has passed the check. One whose header can't be read, one refused unchecked, and one without the header add nothing to the history of any `x-irid`, so a request sent with another request's `x-irid` doesn't change what `request_result` answers for it.

        A conversation belongs to one attempt at the sample. A new attempt starts afresh, as Inspect starts the sample store afresh for it: its conversations, its count of rejected turns in a row, the sentinel's store, and the run's end if a sentinel ended the attempt before. A sample the eval has ended stays ended. Attempts are ordered by the number the header gives each. Once a later attempt has begun, a request of an earlier one is refused unjudged, as nobody waits for it, and so is one of another attempt with the same number: `request_result` says it was not judged, with the reason `replaced_attempt`, so an eval never takes it for a sentinel's terminate. A client that retries a request within the attempt keeps the attempt's conversation.

        A request of a run that a sentinel has ended is refused, and the sentinel is not asked again. For a run no eval names, that is a request which continues the conversation the run was ended in.

        Args:
            request: The request, as the proxy handed it over.

        Raises:
            UnreadableError: If the request can't be read, so neither it nor its reply could be judged.
            UnregisteredSampleError: If the request names a sample that a task registered beforehand doesn't hold, or the handler's strictness refuses a request it can't place.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        try:
            signed = signed_sample(request.headers)
        except UnreadableError as ex:
            # refused rather than raised, but never judged, so not a terminate
            return Refuse(400, str(ex))
        sample = signed.sample if signed is not None else None

        # Proven before anything of its run or its x-irid is looked up or
        # changed, and refused rather than raised, so a proxy that lets
        # through what can't be judged doesn't let through a request that
        # can't prove its run.
        attempt = None
        binding = ""
        if signed is not None:
            refused = await self._unproven_request(signed)
            if refused is not None:
                return Refuse(400, refused)
            binding = _binding(request, signed.sample)
            operation = self._operations.get((signed.sample.eval_id, signed.request_id))
            if operation is not None:
                # a try of a model call a person was asked about: it is not
                # judged again, whatever becomes of it
                return await self._reattach(operation, request, binding)
            attempt = await self._state.attempt(
                signed.request_id, request.id, signed.sample.eval_id
            )
            # an attempt that is never judged has records too: none
            await self._state.recorder(request.id)

        async def judge() -> Answer:
            try:
                answer = await self._within_limit(
                    self._request(request, sample, attempt, binding),
                    "request",
                    request.id,
                )
            except Exception as ex:
                if attempt is None:
                    raise
                if attempt.settled.is_set():
                    # closed while it was judged, as below: it stays as its
                    # closing left it, whatever the judging raised
                    logger.warning(
                        "Judging the request %r failed after its model call was closed: %s",
                        request.id,
                        ex,
                    )
                    return _stopped_answer(attempt)
                attempt.not_judged(_not_judged(ex))
                raise
            if attempt is None or signed is None:
                return answer
            if attempt.settled.is_set():
                # The model call was closed while its request was judged, as
                # when the eval cancelled it. Closing it stops the judging
                # only where the judging next waits, so judging with nothing
                # left to wait for runs to its end. A closed call stays
                # closed: what the judging came to is dropped, the request
                # is not kept to be sent on, and the call reads as its
                # closing left it.
                await self._state.release(request.id)
                return _stopped_answer(attempt)
            if answer is _ANOTHER_EVALS:
                # as when the check above refuses it: nothing of the request
                # is kept, so `request_result` reads it as unknown
                await self._state.forget_attempt(
                    signed.request_id, signed.sample.eval_id, attempt
                )
            elif not isinstance(answer, Refuse):
                waited = self._operation_of(attempt)
                if waited is None:
                    attempt.state = "awaiting_reply"
                else:
                    # The exchange was answered "pending review", so the
                    # request went nowhere. It is kept until the eval sends
                    # it again, and that try is sent on in its place.
                    waited.held = await self._state.release(request.id)
                    waited.go = answer
                    attempt.state = "awaiting_request"
                    # it waits from now, however long the person took
                    waited.heard = time.monotonic()
            elif answer is _UNDER_REVIEW:
                attempt.not_judged("result_under_review")
            elif attempt.judged is not None and attempt.judged.replaced:
                attempt.not_judged("replaced_attempt")
            else:
                # otherwise a request is refused only when its run has ended
                attempt.decide("terminate")
            return answer

        if attempt is None or signed is None:
            return await judge()
        return await self._exchange(
            judge, attempt, signed.sample, signed.request_id, binding
        )

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
        held = await self._released(reply.id)
        attempt = held.attempt if held is not None else None
        if attempt is not None:
            if attempt.settled.is_set():
                # Its model call was closed while the provider answered, as
                # when the eval cancelled it. A closed call stays closed: the
                # reply is not judged, not kept and not passed on.
                return _CLOSED
            attempt.state, attempt.reason = "judging", None

        async def judge() -> Answer:
            try:
                answer = await self._within_limit(
                    self._reply(reply, held, released), "reply", reply.id
                )
            except Exception as ex:
                if attempt is None:
                    raise
                if attempt.settled.is_set():
                    # closed while it was judged, as below: it stays as its
                    # closing left it, whatever the judging raised
                    logger.warning(
                        "Judging the reply %r failed after its model call was closed: %s",
                        reply.id,
                        ex,
                    )
                    return _CLOSED
                attempt.not_judged(_not_judged(ex))
                raise
            if attempt is None or held is None:
                return answer
            if attempt.settled.is_set():
                # The model call was closed while its reply was judged, as
                # when the eval cancelled it. Closing it stops the judging
                # only where the judging next waits, so judging with nothing
                # left to wait for runs to its end. A closed call stays
                # closed: the reply is not kept and not passed on, and the
                # call reads as its closing left it.
                return _CLOSED
            waited = self._operation_of(attempt)
            if not succeeded(reply.status):
                if waited is None:
                    attempt.not_judged("provider_error")
                else:
                    self._failed(waited, held, "provider_error", reply)
            elif not isinstance(answer, Refuse):
                if waited is not None:
                    # kept whichever stage the call waited at: the reply
                    # passed on now may never reach the eval
                    waited.reply = _held_reply(reply, held.format, answer)
                attempt.decide("proceed")
            elif held.exchange.replaced:
                attempt.not_judged("replaced_attempt")
            elif held.exchange.ended:
                attempt.decide("terminate")
            else:
                attempt.decide("reject", held.exchange.message)
            return answer

        sample = held.proxy.inspect if held is not None else None
        request_id = held.proxy.headers.get(REQUEST_ID) if held is not None else None
        if attempt is None or held is None or sample is None or request_id is None:
            return await judge()
        return await self._exchange(
            judge, attempt, sample, request_id, held.binding, held
        )

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
        failing = self._failing.pop(id, None)
        if failing is not None:
            # A provider's error whose body didn't all pass. The eval can't be
            # given the error whole, and what is missing may be what says
            # whether the call may be tried again, so it failed as a reply cut
            # off does: with no reply, for the eval to handle as a connection
            # that failed.
            operation, failure = failing
            attempt = operation.attempt
            if (
                operation.ended is None
                and not attempt.settled.is_set()
                and attempt.state == "awaiting_reply"
                and operation.sent == failure.try_number
            ):
                operation.failure = replace(failure, reply=None)
                attempt.state, attempt.reason = "awaiting_request", "cut_off"
            return
        held = await self._released(id)
        if held is None or held.attempt is None or held.attempt.settled.is_set():
            return
        waited = self._operation_of(held.attempt)
        if waited is not None and reason == "cut_off":
            # the exchange with the provider failed, so the call may be sent
            # on again
            self._failed(waited, held, "cut_off", None)
            return
        held.attempt.not_judged(reason)
        self._finish(held.attempt)

    async def error_body(self, id: str, body: bytes) -> None:
        """Take the body of a provider's error that the handler was shown without it, once all of it has arrived from the provider.

        A provider's error is not held: `reply` is shown it as its status and headers arrive, with `body_follows` set, and its body goes on as it comes. For a model call a person was asked about, the eval gets the error from the sidecar, so the call stays at `awaiting_reply` until the proxy says here that the error is whole. Only then is the eval told that the provider failed, and never of an error it can't yet be given.

        Args:
            id: The proxy's id for the request the error answers.
            body: The error's whole body. Empty where it had none, and where the proxy doesn't copy it, as one too large. A proxy whose error didn't all pass says so with `reply_not_judged`, and not here.
        """
        failing = self._failing.pop(id, None)
        if failing is None:
            return
        operation, failure = failing
        attempt = operation.attempt
        if (
            failure.reply is None
            or operation.ended is not None
            or attempt.settled.is_set()
            or attempt.state != "awaiting_reply"
            or operation.sent != failure.try_number
        ):
            # the call was closed, or went on, while the error passed
            return
        operation.failure = replace(failure, reply=replace(failure.reply, body=body))
        attempt.state, attempt.reason = "awaiting_request", "provider_error"

    def answer_question(
        self, eval_id: str, id: str, decision: str, reason: str | None
    ) -> None:
        """Take an eval's answer to a question a sentinel asked a person, as `Reviews.answer` does, and say at once that the model call no longer waits for it.

        Once the answer is taken, the call reads as `judging` until the sentinel has gone on, unless another of its questions is still open. So an eval that answers and then asks `request_result` is never told `awaiting_review` for a question it has already answered.

        Raises:
            UnknownQuestionError: If the eval has no open question with the id, and no receipt of an answer to it.
            AnsweredError: If the question was answered already, otherwise.
            ChoiceError: If `decision` wasn't offered, or is `modify`.
        """
        self._reviews.answer(eval_id, id, decision, reason)
        operation = self._asking.get(id)
        if operation is None:
            return
        attempt = operation.attempt
        # back to judging once no question of the call still waits for an
        # answer: each has ended, or has the answer just taken
        if attempt.state == "awaiting_review" and all(
            review.resolution is not None or review.decision is not None
            for review in attempt.reviews
        ):
            attempt.state = "judging"

    async def cancel_request(self, id: str, eval_id: str) -> bool:
        """Close a model call a sentinel asked a person about, which the eval no longer waits for. Nothing is decided for it.

        The call is closed wherever it stands before its end: its judging is stopped and its questions close, no try of it is sent on to the provider, and the reply to a request already with the provider is not judged, kept or passed on. It then reads as not judged, for the reason `cancelled`, and nothing opens it again.

        Args:
            id: The eval's id for the model call, its `x-irid`.
            eval_id: The eval cancelling. A call of another eval is left as it is.

        Returns:
            Whether this closed the call. False for a call that had already ended, one no person was asked about, and one the handler doesn't know, so False doesn't say the call is still open.
        """
        operation = self._operations.get((eval_id, id))
        if operation is None or operation.attempt.settled.is_set():
            return False
        self._stop("cancelled", lambda _, attempt: attempt is operation.attempt)
        return True

    async def _reattach(
        self, operation: _Operation, request: Request, binding: str
    ) -> Answer:
        # A request sent again under the `x-irid` of a model call a person was
        # asked about. It is a try of that call only if it is the same
        # request, and is never judged again. It reaches the provider only
        # where the call waits for exactly that, and then alone: nothing is
        # awaited between finding the call waiting and marking it sent, so two
        # tries sent at once can't both go on.
        if binding != operation.binding:
            return _OTHER_REQUEST
        operation.tries += 1
        operation.heard = time.monotonic()
        attempt = operation.attempt
        if attempt.settled.is_set():
            return _OVER
        held, go = operation.held, operation.go
        if attempt.state != "awaiting_request" or held is None or go is None:
            return PENDING
        operation.held = None
        operation.sent = operation.tries
        attempt.state, attempt.reason = "awaiting_reply", None
        self._sent_on[request.id] = operation
        operation.with_provider = (request.id, held)
        await self._state.hold(request.id, held)
        return go

    async def _released(self, id: str) -> Held | None:
        # The request an exchange's reply answers, which is no longer kept
        # for it. The state lets a request go once nothing has named it for
        # as long as it keeps one, while a model call a person was asked
        # about is kept for as long as its eval asks about it. So the request
        # of such a call's try with the provider is had from the call where
        # the state has let it go, and only for that try: an earlier try's
        # exchange finds nothing.
        held = await self._state.release(id)
        operation = self._sent_on.get(id)
        if operation is None or operation.with_provider is None:
            return held
        exchange, kept = operation.with_provider
        if exchange != id:
            return held
        operation.with_provider = None
        return held if held is not None else kept

    def _failed(
        self,
        operation: _Operation,
        held: Held,
        reason: Literal["provider_error", "cut_off"],
        reply: Reply | None,
    ) -> None:
        # The exchange with the provider failed, which is the only thing that
        # lets the provider be called again for a call a person was asked
        # about. The request stage stands, with the answers taken for it, and
        # the eval is told what failed apart from the error its own try got.
        operation.held = held
        # it waits from now, for a request or for the rest of an error
        operation.heard = time.monotonic()
        failure = Failure(
            operation.sent,
            _held_reply(reply, held.format, Pass()) if reply is not None else None,
            time.monotonic(),
        )
        if reply is not None and reply.body_follows:
            # Its body is still passing on to the client. The call stays at
            # `awaiting_reply` until the proxy says the error is whole
            # (`error_body`), so the eval is never told of an error that it
            # can't yet be given.
            self._failing[reply.id] = (operation, failure)
            operation.attempt.state, operation.attempt.reason = "awaiting_reply", None
            return
        operation.failure = failure
        operation.attempt.state, operation.attempt.reason = "awaiting_request", reason

    def _operation_of(self, attempt: Attempt) -> _Operation | None:
        # the model call a person was asked about that the attempt is of
        return self._by_attempt.get(id(attempt))

    def _finish(self, attempt: Attempt) -> None:
        # A model call a person was asked about has ended, however it did.
        # What it kept to go on is let go, and what remains is its record and
        # its reply, for the eval to read.
        operation = self._operation_of(attempt)
        if (
            operation is None
            or operation.ended is not None
            or not attempt.settled.is_set()
        ):
            return
        operation.ended = time.monotonic()
        operation.held = operation.go = operation.with_provider = None
        for exchange, (failing, _) in list(self._failing.items()):
            if failing is operation:
                del self._failing[exchange]
        for exchange, sent in list(self._sent_on.items()):
            if sent is operation:
                del self._sent_on[exchange]
        if self._group is not None:
            self._group.start_soon(self._state.unpin, operation.pinned)

    def _forget(self, key: tuple[str, str]) -> None:
        operation = self._operations.pop(key, None)
        if operation is not None:
            self._by_attempt.pop(id(operation.attempt), None)

    async def _sweeping(self) -> None:
        # for as long as the handler serves
        while True:
            await anyio.sleep(min(_SWEEP_EVERY, self._state.expiry))
            self.sweep()

    def sweep(self) -> tuple[int, int]:
        """Let go of the model calls a person was asked about that were kept as long as the state keeps a run nothing names (`--run-expiry`). A handler that is serving does this by itself, every minute.

        A call that waits for its request to be sent again, or for the provider's reply, and that nothing has named for that long, is closed. It reads as not judged, for the reason `expired`, and nothing opens it again. This is how long the sidecar keeps a call nothing names, and says nothing of whether its client has gone: a client that is still there is asking about the call, or sending it, which keeps it. A call with a question open is never closed here, however long the person takes: a question ends when its own time is up (`--review-timeout`), or never.

        A call that ended that long ago is forgotten. It was kept from its end for its eval to read, and reading it doesn't keep it longer.

        Returns:
            How many calls were closed, and how many forgotten.
        """
        now, expiry = time.monotonic(), self._state.expiry
        closed = forgotten = 0
        for key, operation in list(self._operations.items()):
            attempt = operation.attempt
            if operation.ended is not None:
                if now - operation.ended >= expiry:
                    self._forget(key)
                    forgotten += 1
            elif (
                attempt.state in ("awaiting_request", "awaiting_reply")
                and now - operation.heard >= expiry
            ):
                logger.info(
                    "The model call %s of the eval %s waited at %s with nothing naming it for %s seconds, and is closed.",
                    operation.request_id,
                    operation.eval_id,
                    attempt.state,
                    expiry,
                )
                self._expire(operation)
                closed += 1
        return closed, forgotten

    def _expire(self, operation: _Operation) -> None:
        # Close a model call that waited too long with nothing naming it, as
        # `cancel_request` closes one, for the reason `expired`. Such a call
        # has no question open, so there is none to close. Whatever of it
        # is still being judged is stopped.
        attempt = operation.attempt
        if attempt.stopped is None:
            attempt.stopped = "expired"
        for judging in list(self._trying):
            if judging.attempt is attempt:
                judging.scope.cancel()
        _settle_stopped(attempt)
        self._finish(attempt)

    def reply_progress(self, id: str) -> None:
        """Note that a proxy handed over part of the reply to a request, by the proxy's id for the exchange. A model call a person was asked about, whose try with that id is with the provider, is heard of, so it isn't closed for waiting while its reply is still coming."""
        operation = self._sent_on.get(id)
        if operation is not None:
            operation.heard = time.monotonic()

    async def _exchange(
        self,
        judge: Callable[[], Awaitable[Answer]],
        attempt: Attempt,
        sample: InspectSample,
        request_id: str,
        binding: str,
        held: Held | None = None,
    ) -> Answer:
        # Judge an eval's request or reply, and answer the exchange it came
        # in. Where a person may be asked through the eval, the judging runs
        # in a task of the handler's own, so that it can outlive the exchange:
        # at the first question the exchange is answered "pending review", and
        # the judging goes on when the answer comes. `held` is the request a
        # reply answers.
        judging = _Judging(
            attempt,
            sample,
            request_id,
            binding,
            anyio.CancelScope(),
            anyio.Event(),
            held=held,
        )
        group = self._group
        if group is None or not self._asks_eval:
            # nobody is asked through an eval, so nothing outlives the exchange
            current = _JUDGING.set(judging)
            self._trying.add(judging)
            try:
                return await judge()
            finally:
                self._trying.discard(judging)
                _JUDGING.reset(current)

        async def run() -> None:
            _JUDGING.set(judging)
            try:
                with judging.scope:
                    try:
                        judging.answer = await judge()
                    except Exception as ex:
                        judging.error = ex
                        if judging.detached:
                            # no exchange is left to raise it in, and the
                            # eval reads only that the call was not judged
                            logger.warning(
                                "Judging the model call %s of the eval %s failed after it waited for a person: %s",
                                request_id,
                                sample.eval_id,
                                ex,
                            )
                if judging.reattached is None:
                    # Another try of the call may have asked a person while
                    # this one was judged, and this one never asked: the
                    # call is one operation all the same, and its own result
                    # isn't given.
                    judging.reattached = self._reattached(judging)
                if judging.reattached is not None:
                    # another try of the call holds the operation, or came
                    # to an end after this one began, and this one adds
                    # nothing to what the eval reads of it but its billed
                    # judging
                    await self._forget_try(judging)
                    judging.answer = judging.reattached
                    if judging.error is not None:
                        # The try's own result isn't given, and what its
                        # judging raised is part of that result: the
                        # exchange is answered as the call is, so the error
                        # is said here and nowhere else.
                        logger.warning(
                            "Judging a try of the model call %s of the eval %s failed, and the try follows the call another try holds: %s",
                            request_id,
                            sample.eval_id,
                            judging.error,
                        )
                        judging.error = None
                elif judging.answer is None and judging.error is None:
                    # stopped, with nothing decided
                    if attempt.stopped is None:
                        self._abandoned(judging)
                    else:
                        _settle_stopped(attempt)
                    judging.answer = _stopped_answer(attempt)
                self._finish(attempt)
            finally:
                self._trying.discard(judging)
                judging.turned.set()

        self._trying.add(judging)
        group.start_soon(run)
        try:
            await judging.turned.wait()
        except anyio.get_cancelled_exc_class():
            # The exchange's caller went away before anyone was asked, so
            # nobody waits for this judging. It is stopped, and its task
            # then says that nothing came of the try (`_abandoned`).
            if not judging.detached:
                judging.scope.cancel()
            raise
        if judging.detached:
            return PENDING
        if judging.error is not None:
            raise judging.error
        assert judging.answer is not None
        return judging.answer

    def _abandoned(self, judging: _Judging) -> None:
        # The judging was stopped because its exchange's caller went away
        # before anyone was asked, and nothing was decided. Nothing came of
        # the try, so it reads as cut off, and not as still being judged.
        #
        # For a model call a person was asked about, this is the reply of
        # the try that was sent on, and it is lost with the caller. The call
        # goes back to waiting for a request, as when the proxy says a reply
        # was cut off, so that one more try may be sent on. Left at
        # `judging`, it would answer every later try "pending review" for
        # good.
        attempt = judging.attempt
        if attempt.settled.is_set():
            return
        operation = self._operation_of(attempt)
        if operation is not None and judging.held is not None:
            self._failed(operation, judging.held, "cut_off", None)
        else:
            attempt.not_judged("cut_off")

    async def _superseded(self, judging: _Judging) -> bool:
        # Whether a try of the model call that began after this one was
        # decided, so that its answer went to the client. An earlier try's
        # decision doesn't count: this try may be the client's own retry of
        # it, after its answer was lost.
        tries = await self._state.attempts(judging.request_id, judging.sample.eval_id)
        if not isinstance(tries, list):
            return False
        later = False
        for tried in tries:
            if tried is judging.attempt:
                later = True
            elif later and tried.state == "decided":
                return True
        return False

    async def _forget_try(self, judging: _Judging) -> None:
        # Drop a try that reattached from what `request_result` reads of the
        # call. Its judge models were billed, so their usage stays, with the
        # try the eval reads: the operation's, or the latest other one. The
        # client waited on the try while it was judged, so for a call a
        # person was asked about, whose judging time is added up over its
        # tries, that time stays too. A call nobody was asked about counts
        # its latest try's time alone.
        eval_id, request_id = judging.sample.eval_id, judging.request_id
        operation = self._operations.get((eval_id, request_id))
        into = operation.attempt if operation is not None else None
        if into is None:
            tries = await self._state.attempts(request_id, eval_id)
            if isinstance(tries, list):
                into = next(
                    (
                        tried
                        for tried in reversed(tries)
                        if tried is not judging.attempt
                    ),
                    None,
                )
        if into is not None and into is not judging.attempt:
            for used, billed in judging.attempt.judging_usage.items():
                add_usage(into.judging_usage, used, billed)
            if operation is not None:
                into.judging_time += judging.attempt.judging_time
        await self._state.forget_attempt(request_id, eval_id, judging.attempt)

    def _reattached(self, judging: _Judging) -> Refuse | None:
        # What a try being judged is answered with where another try of its
        # model call asked a person first and holds the call as one
        # operation, as a try sent later is: the call's own answer, or a
        # refusal of another request or of a call that has ended. None where
        # no other try holds it.
        first = self._operations.get((judging.sample.eval_id, judging.request_id))
        if first is None or first.attempt is judging.attempt:
            return None
        if first.binding != judging.binding:
            return _OTHER_REQUEST
        if first.attempt.settled.is_set():
            return _OVER
        return PENDING

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

    def _kept(self, eval_id: str, outcome: Registered) -> None:
        """Raise for a registration, or an eval's secret, that the state didn't keep.

        A token is kept for `--run-expiry` after the eval was last named, apart from the registration that made it, which may expire sooner or, for a task registered beforehand, take no room at all. So the tokens are bounded on their own, by the same number as the tasks registered by evals, and nothing kept is dropped to make room: a kept token guards the records of the eval's runs for as long as they are kept.

        Raises:
            UnauthorizedError: If the handler keeps another secret for the eval.
            RegistrationsFullError: If the handler already keeps as many registrations as it takes, or as many tokens.
        """
        if outcome == "another_token":
            raise UnauthorizedError(
                f"The eval {eval_id!r} registered with another token."
            )
        if outcome == "tokens_full":
            raise RegistrationsFullError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes, or the tokens of as many evals. One is taken again once a kept task or token expires."
            )
        if outcome == "full":
            raise RegistrationsFullError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes. One is taken again once a kept task expires."
            )

    async def keep_eval_token(self, eval_id: str, token: str) -> None:
        """Keep the secret an eval made and sent with a registration that stores no task, unless it is kept already.

        The sidecar never makes a secret, and never sends one back.

        Raises:
            UnauthorizedError: If the handler keeps another secret for the eval.
            RegistrationsFullError: If the handler already keeps as many tokens as it takes registrations.
        """
        self._kept(
            eval_id,
            await self._state.keep_eval_token(
                eval_id, token, self._max_task_registrations
            ),
        )

    async def holds_eval(
        self,
        eval_id: str,
        method: str,
        timestamp: int,
        mac: str,
        signed: Sequence[object] | None = None,
    ) -> bool:
        """Whether the sidecar keeps a secret for the eval, and the caller proved it holds it.

        Args:
            eval_id: The eval's id.
            method: The method called.
            timestamp: When the caller signed, in seconds since the epoch.
            mac: The caller's HMAC of the method, the eval's id and the timestamp, made with the eval's secret.
            signed: What the `mac` signs, where the method signs more than those three, as `answer_question` does.

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
            call_values(method, eval_id, timestamp) if signed is None else signed,
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

    async def _unproven_request(self, signed: SignedSample) -> str | None:
        # why a request's header doesn't prove its run, or None if it does
        eval_id = signed.sample.eval_id
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return f"This sidecar holds no registration of the eval {eval_id!r}, as after a restart, so it can't check the request's {INSPECT_CONTEXT} header. The eval registers its task again."
        problem = unproven(
            kept,
            request_values(signed.sample, signed.request_id, signed.timestamp),
            signed.timestamp,
            signed.mac,
            window=self._timestamp_window,
            what=f"The request's {INSPECT_CONTEXT}",
        )
        if problem is not None:
            return f"{problem} The request was not judged."

        # A run belongs to the eval that started it, whether still kept or
        # ended. A client can register an eval of its own and sign with that
        # eval's token, so the token alone doesn't stop it naming another
        # eval's sample.
        held = await self._state.peek_run(_run_key(True, signed.sample.sample_uuid))
        if held is not None and held.eval_id not in (None, eval_id):
            return _ANOTHER_EVALS.message

        # a request that proves its run keeps the eval's token
        await self._state.eval_token(eval_id)
        return None

    async def request_result(
        self, id: str, eval_id: str, wait: float = 0
    ) -> RequestResult | Literal["dropped", "unknown"]:
        """What happened to a request an eval names, by the eval's id for it.

        Args:
            id: The eval's id for the request, as its `x-irid` header gave it. A client that retries a request sends it again under the same id.
            eval_id: The eval asking. A request whose `x-inspect-context` named another eval reads as unknown, as does one that was never kept: one with no `x-inspect-context`, or one whose header couldn't be read or failed the check.
            wait: How long to wait, in seconds, while the latest attempt is still being judged, at most `MAX_RESULT_WAIT`.

        Returns:
            Where the request stands, and every try's records in order; `dropped` if they were kept and no longer are; or `unknown` if the request was never seen, or isn't the eval's. For a request no person was asked about, where it stands is where its latest try does. For one a person was asked about, it is where that one model call stands, whatever tries were sent since.
        """
        deadline = time.monotonic() + min(max(wait, 0), MAX_RESULT_WAIT)
        while True:
            attempts = await self._state.attempts(id, eval_id)
            if attempts is None:
                return "unknown"
            if isinstance(attempts, Dropped):
                # told only to the eval the request was of
                return "dropped" if attempts.eval_id == eval_id else "unknown"
            if any(attempt.eval_id != eval_id for attempt in attempts):
                return "unknown"
            operation = self._operations.get((eval_id, id))
            if operation is not None and not any(
                attempt is operation.attempt for attempt in attempts
            ):
                operation = None
            latest = operation.attempt if operation is not None else attempts[-1]
            if operation is not None:
                # its eval asks what became of it
                operation.heard = time.monotonic()
            left = deadline - time.monotonic()
            if latest.state != "judging" or left <= 0:
                break
            # woken at once by a decision, and looking again now and then,
            # as a retry of the request may have begun another attempt
            with anyio.move_on_after(min(left, _RESULT_POLL)):
                await latest.settled.wait()
        records: list[Record] = []
        usage: dict[ModelUsed, ModelUsage] = {}
        for attempt in attempts:
            kept = await self._state.records(attempt.exchange)
            if kept is None:
                return "dropped"
            records.extend(kept)
            # every attempt's judging was billed, a retried one's too
            for used, billed in attempt.judging_usage.items():
                add_usage(usage, used, billed)
        reviews = [
            (attempt.review_tries.get(review.id, number), review)
            for number, attempt in enumerate(attempts, 1)
            for review in attempt.reviews
        ]
        if operation is None:
            return RequestResult(latest, records, reviews, usage)
        failed = latest.state == "awaiting_request" and latest.reason is not None
        return RequestResult(
            latest,
            records,
            reviews,
            usage,
            failure=operation.failure if failed else None,
        )

    async def held_reply(self, headers: Mapping[str, str]) -> HeldReply | Refuse:
        """The reply held for a model call a person was asked about, for the eval that sends the call's request again to get it.

        The request is the model call's own, sent again under the same `x-irid`, so its `x-inspect-context` is checked as any model request's is: the `timestamp` and the `mac` prove the eval, the run and the `x-irid`. Nothing else of the request is read. The reply is given once the call is decided with the turn going on, as `request_result` says, and for as long as what was held for it is still kept. `request_result` doesn't carry it. While the call waits to be sent on again because the provider failed it (`awaiting_request`, with the reason `provider_error`), the provider's error is given in the reply's place, with the `FAILED_TRY` header naming the try it is of. Nothing of the call is changed, so either can be read again, and the provider is never called.

        Args:
            headers: The request's headers, by lowercase name.

        Returns:
            The held reply or the provider's error, or the refusal that says why there is none to give: the header is missing, unreadable or fails the check; the sidecar holds no such call for the eval; the call has not ended; or it ended without its turn going on.
        """
        try:
            signed = signed_sample(headers)
        except UnreadableError as ex:
            return Refuse(400, str(ex))
        if signed is None:
            return _NO_CALL_NAMED
        refused = await self._unproven_request(signed)
        if refused is not None:
            return Refuse(400, refused)

        # as `request_result` finds the call: the eval's own, by its
        # `x-irid`, for as long as its records are kept
        eval_id = signed.sample.eval_id
        operation = self._operations.get((eval_id, signed.request_id))
        if operation is None:
            return _NO_HELD_REPLY
        attempts = await self._state.attempts(signed.request_id, eval_id)
        if (
            attempts is None
            or isinstance(attempts, Dropped)
            or not any(attempt is operation.attempt for attempt in attempts)
        ):
            return _NO_HELD_REPLY

        # a request of the call, as a try of it is
        operation.heard = time.monotonic()
        attempt = operation.attempt
        failure = operation.failure
        if (
            attempt.state == "awaiting_request"
            and attempt.reason == "provider_error"
            and failure is not None
            and failure.reply is not None
        ):
            # the provider's latest error, which the call waits to be sent on
            # again because of
            return _served_error(failure)
        if not attempt.settled.is_set():
            return _NOT_ENDED
        reply = operation.reply
        if attempt.state != "decided" or attempt.outcome != "proceed" or reply is None:
            return _NOT_GOING_ON
        return reply

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
        kept = await self._state.peek_run(key)
        # an ended run keeps no store to copy
        if kept is None or kept.eval_id != eval_id or kept.closed:
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
        if ended:
            # nobody is left to answer the sample's questions, or to send an
            # answer to one of them again
            self._stop(
                "sample_ended",
                lambda sample, _: (
                    (sample.eval_id, sample.sample_uuid) == (eval_id, sample_uuid)
                ),
            )
            self._reviews.forget_sample(eval_id, sample_uuid)
            # the replies kept for the sample's calls go with it
            for key, operation in list(self._operations.items()):
                if (operation.eval_id, operation.sample.sample_uuid) == (
                    eval_id,
                    sample_uuid,
                ):
                    self._forget(key)
        return "ended" if ended else "unknown"

    async def _within_limit(
        self, judging: Awaitable[Answer], what: str, id: str
    ) -> Answer:
        # The time limit covers the sentinel's work on one request or reply.
        # The time a question is open for a person is not counted: the asker
        # holds the limit's deadline off while it waits.
        with anyio.move_on_after(self._time_limit) as limit:
            current = _JUDGING.get()
            if current is not None:
                current.limit = limit
            return await judging
        raise self._out_of_time(what, id)

    def _stop(
        self, cause: Stopped, stops: Callable[[InspectSample, Attempt], bool]
    ) -> None:
        # Nobody waits any more for the attempts that match. Each one being
        # judged is marked, whether or not it has asked a person yet, so that
        # judging which reaches `human()` later asks nobody. Judging that has
        # asked no one is left to finish, as before: what it decides is kept
        # or dropped by the checks that follow it.
        #
        # A model call a person was asked about is closed here, wherever it
        # stands, before anything else runs: its questions close, so none is
        # listed or answered from this moment, its judging is stopped with
        # nothing decided, no try of it is sent on, and it reads as ended at
        # once. It stays closed: a reply to a request already with the
        # provider is refused when it comes.
        for judging in list(self._trying):
            attempt = judging.attempt
            if not stops(judging.sample, attempt):
                continue
            if attempt.stopped is None:
                attempt.stopped = cause
            if self._operation_of(attempt) is not None:
                judging.scope.cancel()
        for operation in list(self._operations.values()):
            attempt = operation.attempt
            if attempt.settled.is_set() or not stops(operation.sample, attempt):
                continue
            if attempt.stopped is None:
                attempt.stopped = cause
            stopped = attempt.stopped
            if stopped != "expired":
                # a call closed for waiting too long had no question open
                self._reviews.close(
                    [r for r in attempt.reviews if r.resolution is None], stopped
                )
            _settle_stopped(attempt)
            self._finish(attempt)

    async def records(self, id: str) -> list[Record] | None:
        """What was recorded while a request and its reply were judged.

        Args:
            id: The request's id, as the proxy gave it.

        Returns:
            The records in order, the request's before the reply's, or None if the request was never seen or its records are no longer kept.
        """
        return await self._state.records(id)

    async def register_task(
        self,
        eval_id: str,
        registration: TaskRegistration,
        expiry: float | None = None,
        token: str | None = None,
    ) -> float | None:
        """Keep a task an eval registered, so that its runs' sentinels are told which sample each is, once each sample's registration is complete.

        A registration replaces the eval's last one for the task, and keeps the samples whose registration was complete that it still names, so that registering again has no effect. A run that has already found its sample keeps what it found.

        Args:
            eval_id: The eval's id, as its requests name it.
            registration: The task, and the samples it runs.
            expiry: How long to keep it once no request has named the task, in seconds. A day if not given.
            token: The secret the registration sent, which is kept if the eval has none. None for a registration that proved it holds the eval's secret.

        A task registered beforehand is already registered, so its registration stores nothing and is not compared with the file, since the answer would tell a caller whether its guess at the file is right. This holds whether or not the handler takes registrations from evals.

        The secret a registration sent is kept with its task in one step, or alone for a task registered beforehand. So a registration that fails leaves no secret behind, and of two first registrations of an eval that send different secrets, only one keeps anything.

        Returns:
            How long it is kept once no request has named the task: `expiry`, or less if that is longer than the handler keeps any. None for a task registered beforehand, which is never dropped.

        Raises:
            RegistrationError: If the handler takes no registrations from evals and the task was not registered beforehand.
            RegistrationsFullError: If the handler already keeps as many registrations as it takes, or as many tokens.
            UnauthorizedError: If the registration sent a secret and the handler keeps another for the eval.
        """
        if registration.task in self._tasks:
            if token is not None:
                await self.keep_eval_token(eval_id, token)
            return None
        if not self._runtime_task_registration:
            raise RegistrationError(
                "This sidecar takes no tasks registered by an eval."
            )
        granted = min(expiry if expiry is not None else TASK_EXPIRY, self._task_expiry)
        kept = RegisteredTask(registration)
        # Read without renewing: the registration's secret is only taken in
        # the step that keeps the task, so one refused there must not have
        # kept the task it would have replaced alive.
        before = await self._state.registration(eval_id, registration.task, renew=False)
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
        self._kept(
            eval_id,
            await self._state.register(
                eval_id,
                registration.task,
                kept,
                granted,
                self._max_task_registrations,
                token=token,
            ),
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
        self,
        request: Request,
        sample: InspectSample | None,
        attempt: Attempt | None,
        binding: str = "",
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
        if sample is not None:
            # Whose run it is was checked before the request was read, and
            # another eval may have begun the run while it was. So it is
            # checked again here, where nothing is awaited between the check
            # and the run becoming this eval's, and before anything below
            # can reset the run.
            held = await self._state.peek_run(run_key)
            if held is not None and held.eval_id not in (None, sample.eval_id):
                return _ANOTHER_EVALS
        run = await self._state.run(run_key)
        restored = False
        if sample is not None:
            if run.eval_id is None:
                run.eval_id = sample.eval_id
            was = run.attempt
            clash = _begin_attempt(run, sample)
            if run.attempt != was:
                eval_id, sample_uuid, now = (
                    sample.eval_id,
                    sample.sample_uuid,
                    run.attempt,
                )
                self._stop(
                    "replaced_attempt",
                    lambda other, _: (
                        (other.eval_id, other.sample_uuid) == (eval_id, sample_uuid)
                        and other.attempt != now
                    ),
                )
            if clash is not None:
                return self._judged(clash, proxy, Exchange(replaced=True), attempt)
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
            binding=binding,
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
                while (under_way := self._judging.get(key)) is not None:
                    if not under_way.asked:
                        await under_way.woken.wait()
                        continue
                    # The judging waited for asked a person, who may take
                    # days, and no connection waits for a person. A result is
                    # reviewed once, by the model call that asked.
                    current = _JUDGING.get()
                    reattached = (
                        self._reattached(current) if current is not None else None
                    )
                    if current is not None and reattached is not None:
                        # a try of that same model call: it follows the call
                        current.reattached = reattached
                        current.scope.cancel()
                        await anyio.lowlevel.checkpoint()
                    # Another model call of the sample: it is refused with a
                    # status its client sends it again for, and goes on once
                    # the review has ended and the result is judged.
                    return _UNDER_REVIEW
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
                assistant_message=sent.message if sent is not None else None,
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
            under_way = self._judging[key] = _ResultJudging(
                anyio.Event(), _JUDGING.get()
            )
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
                under_way.woken.set()
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
                assistant_message=message,
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
        produced: ChatMessageAssistant | None = None,
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
                    produced if produced is not None else message,
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
        self._passed(held, run, conversation, handed, produced=message)
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
        reviewed = len(held.attempt.reviews) if held.attempt is not None else 0
        # which models judged the step, for the eval: its roles don't reach here
        # each judge call's billed usage goes to the attempt as soon as the
        # call is done, so an eval asking while the step is judged sees it
        judging = _JUDGING.get()
        with noting_models(
            held.attempt.judging_usage if held.attempt is not None else None
        ) as models:
            started = time.monotonic()
            try:
                decision = await run_sentinel(
                    self._protocol,
                    HostContext(
                        host=RunHost(self._host, run.eval, self._asker(held)),
                        recorder=held.recorder,
                        store=held.store,
                        proxy=held.proxy,
                    ),
                    step,
                )
            finally:
                # The client waited on its model call while the step was
                # judged, failed or not, until the call was answered
                # "pending review". The time a person takes, and any judging
                # after it, isn't time the client waited on the call.
                if held.attempt is not None:
                    ended = time.monotonic()
                    if judging is not None and judging.detached_at is not None:
                        ended = min(ended, judging.detached_at)
                    held.attempt.judging_time += max(0.0, ended - started)
        call = (
            step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else None
        )
        held.exchange.judged.append(
            Judged(
                stage(step),
                call,
                decision,
                held.recorder.records[recorded:],
                tuple(models.called),
                tuple(
                    review.id
                    for review in (
                        held.attempt.reviews[reviewed:]
                        if held.attempt is not None
                        else []
                    )
                ),
            )
        )
        return decision

    def _asker(self, held: Held) -> AskHuman | None:
        # A run an Inspect eval names, whose request passed the check of its
        # header, has someone to ask: the eval shows the question in Inspect's
        # approval UI. Any other run has nobody, and is answered with caution,
        # as is every run where evals aren't sent results or the host asks a
        # person its own way.
        sample = held.proxy.inspect
        request_id = held.proxy.headers.get(REQUEST_ID)
        attempt = held.attempt
        if (
            not self.eval_review
            or not held.named
            or sample is None
            or request_id is None
            or attempt is None
        ):
            return None

        async def ask(
            step: BeforeToolCall | AfterToolCall,
            escalations: Escalations,
            choices: Sequence[str],
        ) -> HumanAnswer:
            judging = _JUDGING.get()
            assert judging is not None and judging.attempt is attempt
            # what the try has judged so far, for the eval, should its judging
            # be stopped here: the steps before this one stand
            attempt.judged = held.exchange
            if attempt.stopped is not None:
                # a try nobody waits for asks nobody: its judging stops
                # here, with nothing decided
                judging.scope.cancel()
                await anyio.lowlevel.checkpoint()
            operation = self._operation_of(attempt)
            if operation is None and await self._superseded(judging):
                # A later try of the call came to an end, and its answer was
                # given: this one asks nobody, and is answered as a call
                # that has ended.
                judging.reattached = _OVER
                judging.scope.cancel()
                await anyio.lowlevel.checkpoint()
            if operation is None:
                # nothing is awaited from here until the operation is kept
                key = (sample.eval_id, request_id)
                reattached = self._reattached(judging)
                if reattached is not None:
                    # Another try of the call asked a person while this one
                    # was judged. The call is one operation, and that try
                    # holds it: this one stops here and reattaches.
                    judging.reattached = reattached
                    judging.scope.cancel()
                    await anyio.lowlevel.checkpoint()
                # The call becomes one operation, bound to this request. It
                # is kept before anything is awaited, so that no other try
                # can begin a second one.
                operation = _Operation(
                    sample.eval_id,
                    request_id,
                    sample,
                    judging.binding,
                    attempt,
                    Pinned(
                        sample.eval_id,
                        sample.task,
                        held.run,
                        request_id,
                        (attempt.exchange,),
                    ),
                )
                operation.heard = time.monotonic()
                self._operations[key] = operation
                self._by_attempt[id(attempt)] = operation
                tries = await self._state.attempts(request_id, sample.eval_id)
                if isinstance(tries, list):
                    operation.tries = max(operation.tries, len(tries))
                    operation.sent = next(
                        (
                            number
                            for number, tried in enumerate(tries, 1)
                            if tried is attempt
                        ),
                        operation.tries,
                    )
                    # `request_result` reads the records of every try of the
                    # request, so those of tries before this one are kept too
                    operation.pinned = replace(
                        operation.pinned,
                        exchanges=tuple(
                            dict.fromkeys(
                                [*(tried.exchange for tried in tries), attempt.exchange]
                            )
                        ),
                    )
                # a person may take days, so what the call needs doesn't expire
                await self._state.pin(operation.pinned)
            asked = time.time()
            timeout = self._reviews.timeout
            question = Question(
                eval_id=sample.eval_id,
                sample_uuid=sample.sample_uuid,
                request_id=request_id,
                stage=stage(step),
                call=step.call,
                message=step.message,
                view=step.view,
                result=step.result if isinstance(step, AfterToolCall) else None,
                escalations=escalated(escalations),
                choices=tuple(choices),
                closes=asked + timeout if timeout is not None else None,
                asked=asked,
            )
            # kept with the try whether or not the step comes to a decision
            attempt.reviews.append(question.review)
            # Kept with the try, and not with the operation, which is
            # forgotten when its sample ends: the record of a question says
            # which try asked for as long as the record is kept.
            attempt.review_tries[question.id] = operation.sent
            self._asking[question.id] = operation
            attempt.state = "awaiting_review"
            # a request waiting on this judging of a tool's result is
            # answered now, as this exchange is
            for under_way in self._judging.values():
                if under_way.judging is judging:
                    under_way.asked = True
                    under_way.woken.set()
            if not judging.detached:
                # the exchange is answered "pending review" now, and no
                # connection waits for the person
                judging.detached = True
                judging.detached_at = time.monotonic()
                judging.turned.set()
            # the time a question is open isn't the sentinel's
            limit = judging.limit
            if judging.asking == 0 and limit is not None:
                if limit.deadline != math.inf:
                    judging.left = limit.deadline - anyio.current_time()
                    limit.deadline = math.inf
            judging.asking += 1
            try:
                return await self._reviews.ask(question)
            finally:
                self._asking.pop(question.id, None)
                judging.asking -= 1
                if judging.asking == 0:
                    if limit is not None and judging.left is not None:
                        limit.deadline = anyio.current_time() + judging.left
                        judging.left = None
                    if attempt.state == "awaiting_review":
                        attempt.state = "judging"

        return ask

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
            ended = held.proxy.inspect
            if ended is not None:
                self._stop(
                    "run_ended",
                    lambda other, attempt: (
                        attempt is not held.attempt
                        and (other.eval_id, other.sample_uuid)
                        == (ended.eval_id, ended.sample_uuid)
                    ),
                )
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


def _stopped_answer(attempt: Attempt) -> Refuse:
    # what an exchange is refused with when its judging was stopped
    if attempt.stopped == "replaced_attempt":
        return _REPLACED
    if attempt.stopped == "cancelled":
        return _CANCELLED
    return _RUN_OVER


def _settle_stopped(attempt: Attempt) -> None:
    # An attempt nobody waits for any more is settled, once, with nothing
    # decided for it. Its run ending reads as a request of an ended run does.
    if attempt.stopped is None or attempt.settled.is_set():
        return
    if attempt.stopped in ("replaced_attempt", "cancelled", "expired"):
        attempt.not_judged(attempt.stopped)
    else:
        # As a request of an ended run reads: the run was ended, by the eval
        # or by a sentinel at another request, and the judgment says so, with
        # the steps this try had judged before it was stopped. No decision is
        # made up for the step that was waiting.
        if attempt.judged is None:
            attempt.judged = Exchange()
        attempt.judged.ended = True
        attempt.decide("terminate")


_NOT_BOUND = frozenset({INSPECT_CONTEXT, "x-stainless-retry-count"})
"""The headers that are no part of what a request is, as each try has its own: Inspect's context, whose `timestamp` and `mac` are made afresh, and the count that a provider's SDK gives its own retries."""


def _binding(request: Request, sample: InspectSample) -> str:
    # What a request is, for telling a try of a model call from another
    # request sent under its `x-irid`: everything the proxy said of it that a
    # sentinel can read, or that says how it is read. A request sent again is
    # not judged, so what an approval was given for must be what goes on. The
    # header's `timestamp` and `mac` are made afresh for each try, and are
    # checked on each, so they are left out, as is the proxy's id for the
    # exchange, and so is a caching hint that changes with the time between
    # tries ("bound_body").
    try:
        format = request_format(request.provider, request.body, request.format)
    except UnreadableError:
        # such a request is never judged, so it begins no operation
        format = None
    headers = sorted(
        (name, value)
        for name, value in request.headers.items()
        if name not in _NOT_BOUND
    )
    return hashlib.sha256(
        json.dumps(
            [
                request.provider,
                format,
                request.model,
                request.user,
                headers,
                asdict(sample),
                hashlib.sha256(
                    bound_body(request.provider, request.body, request.format)
                ).hexdigest(),
            ],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _served_error(failure: Failure) -> HeldReply:
    # A provider's error as the held-reply path serves it, for the eval's code
    # for that provider to read as the provider's own. It is marked with the
    # try it is of, which no refusal of the sidecar's has. The eval's SDK is
    # told not to send the request again by itself, as it would only get this
    # answer again: whether the provider is called again is the eval's own
    # retry rules' to say. A wait the provider asked for is counted from when
    # the error came, not from when it is served.
    assert failure.reply is not None
    waited = time.monotonic() - failure.received
    headers = {
        name: _time_left(name, value, waited)
        for name, value in failure.reply.headers.items()
    }
    headers[FAILED_TRY] = str(failure.try_number)
    headers["x-should-retry"] = "false"
    return replace(failure.reply, headers=headers)


def _time_left(name: str, value: str, waited: float) -> str:
    # what is left of a wait given in seconds (`retry-after`) or milliseconds
    # (`retry-after-ms`); a date, or anything else, is as it was sent
    scale = {"retry-after": 1.0, "retry-after-ms": 1000.0}.get(name)
    if scale is None:
        return value
    try:
        asked = float(value)
    except ValueError:
        return value
    if not math.isfinite(asked) or asked < 0:
        return value
    return str(math.ceil(max(0.0, asked - waited * scale)))


def _held_reply(reply: Reply, format: Format, answer: Answer) -> HeldReply:
    # the reply as the proxy passes it on, for the eval to read as the
    # provider's own
    headers = {
        name: value for name, value in reply.headers.items() if name not in _NOT_KEPT
    }
    headers.setdefault("content-type", reply.content_type)
    return HeldReply(
        reply.status,
        headers,
        answer.body if isinstance(answer, Replace) else reply.body,
        format,
        "event-stream" in reply.content_type.lower(),
    )


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
