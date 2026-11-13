from __future__ import annotations

import hmac
import json
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

import anyio
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ModelUsage
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import Store
from pydantic import JsonValue

from .._context import EvalContext, ProxyContext
from ._exchange import Refuse, Request
from ._host import ListRecorder, ModelUsed, Record
from ._inspect_client import Exchange
from ._registration import RegisteredTask
from ._review import Review
from ._wire import Amendment, Edited, EntryDigest, Format, Offered, Turn

V = TypeVar("V")

MADE_KEPT = 64
"""How many tool calls a conversation keeps what the model was sent for. A call's result is judged without it once it is dropped."""

RUN_EXPIRY = 24 * 60 * 60
"""How long a run's state is kept once no request has named it, in seconds."""

DROPPED_IDS = 1_000_000
"""How many ids of requests whose records were dropped are kept, so that a request can be told apart from one never seen."""

Registered: TypeAlias = Literal["kept", "full", "tokens_full", "another_token"]
"""What became of a registration, or of an eval's secret alone. `kept`; `full` if it would be one registration more than are kept at once; `tokens_full` if it would be one eval's secret more than are kept at once; or `another_token` if the eval's secret is kept and isn't the one sent. Nothing is kept or changed unless it is `kept`."""

NotJudged: TypeAlias = Literal[
    "unreadable",
    "unregistered",
    "time_limit",
    "error",
    "provider_error",
    "cut_off",
    "replaced_attempt",
    "cancelled",
]
"""Why a request or its reply was not judged."""

Stopped: TypeAlias = Literal[
    "replaced_attempt", "sample_ended", "run_ended", "cancelled"
]
"""Why an attempt at a request has nobody waiting for it: a later attempt at its sample began, the eval ended its sample, a sentinel ended its run, or the eval cancelled the request while it waited for a person."""

Outcome: TypeAlias = Literal["proceed", "reject", "terminate"]
"""What an exchange came to."""


@dataclass(frozen=True)
class Dropped:
    """A request whose attempts were kept and no longer are."""

    eval_id: str | None
    """The eval its latest attempt's `x-inspect-context` named, the only one told it was dropped."""


@dataclass
class Attempt:
    """One attempt at a request an eval names, from the request to its reply."""

    exchange: str
    """The proxy's id for the exchange, under which its records are kept."""

    eval_id: str
    """The eval the request's `x-inspect-context` named, and proved with its `mac`: the only one `request_result` answers for it."""

    state: Literal[
        "judging",
        "awaiting_review",
        "awaiting_request",
        "awaiting_reply",
        "decided",
        "not_judged",
    ] = "judging"
    """Where the attempt stands. `awaiting_review` and `awaiting_request` are only for a model call a sentinel asked a person about: a question is open, or the request may be sent on to the provider."""

    reason: NotJudged | None = None
    """Why the attempt was not judged, when it wasn't. With `awaiting_request`, why the provider's last exchange failed, if one did: `provider_error` or `cut_off`."""

    outcome: Outcome | None = None
    """What the attempt came to, once decided."""

    message: str | None = None
    """What the model is told, for a rejection."""

    judged: Exchange | None = None
    """What was decided during the attempt, for an Inspect eval, from which `request_result` writes its judgment."""

    judging_time: float = 0.0
    """How long the sentinel spent judging the attempt's request and reply, in seconds, while the client waited on the model call."""

    judging_usage: dict[ModelUsed, ModelUsage] = field(
        default_factory=dict[ModelUsed, ModelUsage]
    )
    """What the provider billed for each model the sentinel called while judging the attempt, in the order first billed. A reply from inspect_ai's cache and a call that failed weren't billed, so a model with only those has no entry. Kept for an attempt whose reply never came too, since its request was judged and billed."""

    settled: anyio.Event = field(default_factory=anyio.Event)
    """Set once the attempt is decided or not judged, for a `request_result` that waits."""

    stopped: Stopped | None = None
    """Why nobody waits for the attempt, once that is so. Judging that reaches a question after this asks nobody, and an attempt that is waiting on a person when it is set is stopped, with no decision."""

    reviews: list[Review] = field(default_factory=list[Review])
    """What became of each question a sentinel asked a person during the attempt, in order."""

    review_tries: dict[str, int] = field(default_factory=dict[str, int])
    """The number of the try each of `reviews` was asked at, from 1, by the question's id. A try sent again for a model call a person was asked about adds no attempt, so the attempt's place among the request's attempts doesn't give it."""

    def decide(self, outcome: Outcome, message: str | None = None) -> None:
        """Mark the attempt decided. A decided attempt has no `reason`: one left from an earlier state, as from a provider's failure the attempt waited after, goes."""
        self.state, self.outcome, self.message = "decided", outcome, message
        self.reason = None
        self.settled.set()

    def not_judged(self, reason: NotJudged) -> None:
        """Mark the attempt as one that judging ended without a decision for."""
        self.state, self.reason = "not_judged", reason
        self.settled.set()


@dataclass
class Rejected:
    """The turns rejected in a row at one point of a client's conversation.

    The client's own conversation never holds them, so they are added to every later request that continues from that point. A model that stopped being told would see itself change course for no reason, and propose the rejected call again.
    """

    sent: int
    """How many entries the client had sent when the turns were rejected."""

    mark: str
    """The mark of those entries."""

    turns: list[Turn]
    """Each rejected turn."""

    written: list[Mapping[str, Amendment]] = field(
        default_factory=list[Mapping[str, Amendment]]
    )
    """For each rejected turn, its calls and their rejections as inspect_ai writes them in the conversation's format, by call id, so that a client that put the turn back itself can be told from one that ran its calls."""

    added: bool = True
    """Whether the turns are added to the requests that continue from the point. A run no eval names that was ended at the point, but isn't remembered as ended there, may next be sent the same request as a fresh start, so they stop being added. They are kept, so that a client that put a turn back itself is still told from one that ran its calls."""


@dataclass(frozen=True)
class Made:
    """What the model was sent for the generate that made a tool call."""

    input: list[ChatMessage]
    tools: list[ToolInfo]
    tool_choice: ToolChoice
    message_id: str
    """The id the reply holding the call was given, so the call's two steps share it."""

    message: ChatMessageAssistant
    """The reply holding the call as the model returned it, before any edit: the call's two steps share it."""


@dataclass(frozen=True)
class Sent:
    """What one request of a conversation sent the model, which a later request is matched against."""

    keys: list[str]
    """A digest of what the model was sent of each of the request's messages."""

    ids: list[str | None]
    """The id each of the request's messages was given, which a message of a later request matched to it keeps."""


@dataclass
class Conversation:
    """One conversation within a run."""

    rejected: list[Rejected] = field(default_factory=list[Rejected])
    """The points at which turns were rejected."""

    edited: list[Edited] = field(default_factory=list[Edited])
    """The turns the client was handed with calls taken out. Each is kept for as long as the conversation is: it says what the model is told in every later request that holds the turn."""

    noted: JsonValue | None = None
    """What inspect_ai's converters noted while reading the conversation's replies, which reading the next one needs."""

    judged: set[str] = field(default_factory=set[str])
    """Digests of the tool results in the conversation the sentinel has been shown and let go on."""

    made: dict[str, Made] = field(default_factory=dict[str, Made])
    """For the latest tool calls the client was handed, by call id, what the model was sent for the generate that made each. The oldest are dropped past `MADE_KEPT`, as each holds what the model was sent, which grows with the conversation."""

    sent: list[ChatMessage] = field(default_factory=list[ChatMessage])
    """Every message the conversation's requests sent the model, in the order first sent, each once: a step's `history`. Messages a compaction dropped from later requests are kept."""

    recent: list[Sent] = field(default_factory=list[Sent])
    """What each of the conversation's latest requests sent the model, latest last, which a later request is matched against."""


@dataclass
class Run:
    """What the sidecar keeps for one run."""

    store: Store = field(default_factory=Store)
    """The sentinel's store. For a run an eval names, it is the latest attempt's: a new attempt at the sample starts it afresh, as Inspect starts the sample store afresh."""

    attempt: str = ""
    """For a run an eval names, the attempt at the sample that the store and the run's end are of."""

    attempt_number: int = 0
    """That attempt's number. A request of an attempt with a lower one is refused unjudged, as nobody waits for it."""

    closed: bool = False
    """Whether the eval ended the sample, once it was scored. No later attempt starts the run again."""

    conversations: dict[str, Conversation] = field(
        default_factory=dict[str, Conversation]
    )
    """The run's conversations, by key."""

    in_a_row: dict[str, int] = field(default_factory=dict[str, int])
    """Rejected turns since a reply last passed, by the attempt an eval names, or `""`. A new attempt at a sample starts its own count, as its model never saw the earlier attempt's turns."""

    refusal: Refuse | None = None
    """What every later request is refused with, once a sentinel has ended a run an eval names, or the eval has ended its sample. A sentinel's end is the attempt's, as the store is."""

    answered: set[str] = field(default_factory=set[str])
    """For a run no eval names: the marks of the requests whose replies were passed on."""

    ended_at: dict[str, Refuse] = field(default_factory=dict[str, Refuse])
    """For a run no eval names: the marks of the requests the run was ended at, each with what a request that continues from it is refused with."""

    ended_on: list["EndedOn"] = field(default_factory=list["EndedOn"])
    """For a run no eval names: the last entry of each request the run was ended at, in digests. A request that continues from it with blocks added to that entry, as a client adds the user's next message to the one that holds the last results, is refused as well."""

    eval: EvalContext | None = None
    """The sample the run is, as its task was registered when the run first found it. A registration made later doesn't reach it."""

    eval_id: str | None = None
    """For a run an eval names, the eval, the only one the endpoint for evals answers for the run."""

    redacted: dict[str, tuple[Format, Any]] = field(
        default_factory=dict[str, tuple[Format, Any]]
    )
    """For a run an eval names, the model's own calls that a decision acted on, by call id, each in the format its request was written in. The eval holds each with its id and function only, and each later request has the model's own call put back."""


@dataclass(frozen=True)
class EndedOn:
    """For a run no eval names, the entry that a request the run was ended at ends with."""

    at: int
    """How many entries come before it."""
    before: str
    """The mark of those entries."""
    digest: EntryDigest
    """The entry, as its mark was taken of it, in digests."""
    refusal: Refuse
    """What a request that adds to it is refused with."""


@dataclass(frozen=True)
class Held:
    """A request that went on to the provider, kept until its reply is handed over."""

    run: str
    conversation: str
    named: bool
    """Whether an eval named the run."""
    sample_attempt: str
    """The attempt at the sample an eval names, or `""`."""
    answered: bool
    """Whether the request continues one whose reply was passed on, so that it holds a turn the model wrote."""
    model: str
    format: Format
    """The API the request is written in, which later requests of the conversation are too."""
    recorder: ListRecorder
    input: list[ChatMessage]
    """What the model was sent."""
    offered: Offered
    marks: list[str]
    """The request's marks, for each count of the entries the client sent, before anything was added."""
    proxy: ProxyContext
    """What the proxy said about the request."""
    store: Store
    """The sentinel's store as the request found it, its attempt's, so that a judgment that outlives its attempt writes nowhere a later attempt reads."""
    attempt: Attempt | None = None
    """The attempt at the request, for an eval that names its requests."""
    exchange: Exchange = field(default_factory=Exchange)
    """What was decided so far during the exchange, for an Inspect eval's judgment of it."""
    last: EntryDigest | None = None
    """For a run no eval names, the last entry the client sent, as its mark was taken of it, in digests, or None for a request with none, or whose last entry nothing can be added to."""
    request: Request | None = None
    """For a run no eval names, the request as the client sent it, so that a run ended meanwhile by another exchange refuses its reply when it continues the ended conversation with blocks added to its last entry."""
    binding: str = ""
    """For an eval's request, a digest of what the request is: what a request sent again under the same `x-irid` must match to be a try of the same model call."""

    @property
    def sent(self) -> int:
        """How many entries the client sent, before anything was added."""
        return len(self.marks) - 1

    @property
    def mark(self) -> str:
        """The mark of those entries."""
        return self.marks[-1]


@dataclass(frozen=True)
class Pinned:
    """What a model call that waits for a person needs kept, however long the person takes."""

    eval_id: str
    """The eval, whose secret is kept."""

    task: str
    """The eval's task, whose registration is kept."""

    run: str
    """The key of the call's run."""

    request: str
    """The eval's id for the request, whose attempts are kept."""

    exchanges: tuple[str, ...]
    """The proxy's ids for the exchanges of every attempt at the request, whose records are kept: `request_result` reads them all, those of tries before the one that asked included."""


class State(Protocol):
    """Where the sidecar keeps runs, the requests awaiting a reply, and what was recorded for each request.

    A sidecar that runs as several processes needs them to share this, and a run that outlasts a restart needs it kept. Only the in-memory implementation exists, which hands back the objects it keeps, so the handler changes them in place. A shared implementation would also need a way to write them back.
    """

    async def run(self, key: str) -> Run:
        """The run's state, made new if none is kept. A run its eval ended is not made again: an ended run is handed back, whose `refusal` refuses every request of it."""
        ...

    async def hold(self, id: str, held: Held) -> None:
        """Keep a request until its reply is handed over."""
        ...

    async def pin(self, pinned: Pinned) -> None:
        """Keep these past their expiry, until `unpin`. Each `pin` is undone by one `unpin`, so two calls that wait for a person can pin the same eval or run."""
        ...

    async def unpin(self, pinned: Pinned) -> None:
        """Undo one `pin`. What it kept expires again, counted from now."""
        ...

    async def release(self, id: str) -> Held | None:
        """The request a reply answers, no longer kept, or None if it isn't."""
        ...

    async def recorder(self, id: str) -> ListRecorder:
        """Where what is recorded for a request and its reply is kept, made new if none is kept."""
        ...

    async def records(self, id: str) -> list[Record] | None:
        """What was recorded for a request and its reply, or None if nothing is kept for it."""
        ...

    async def register(
        self,
        eval_id: str,
        task: str,
        facts: RegisteredTask,
        expiry: float,
        limit: int,
        *,
        token: str | None = None,
    ) -> Registered:
        """Keep a task an eval registered, in place of any it registered before, and with it the eval's secret if the registration sent one.

        The task and the secret are kept together or not at all, in one step that nothing else comes between. Were they two steps, two first registrations of an eval that sent different secrets could each keep something: one its secret, the other its task. An implementation that several processes share must keep them together too, as in one transaction.

        Args:
            eval_id: The eval's id.
            task: The task's name.
            facts: The task, the samples it runs, and those whose registration is complete.
            expiry: How long to keep it once no request has named it, in seconds.
            limit: How many registrations are kept at once, and how many evals' secrets.
            token: The secret the registration sent, which is kept if the eval has none. None for a registration that proved it holds the eval's secret.

        Returns:
            What became of it. Nothing is kept or changed unless it is `kept`.
        """
        ...

    async def registration(
        self, eval_id: str, task: str, *, renew: bool = True
    ) -> RegisteredTask | None:
        """A task an eval registered, or None if none is kept.

        Args:
            eval_id: The eval's id.
            task: The task's name.
            renew: Whether asking counts as the task being named, so it is kept for its expiry from now. False for a registration that reads the task it would replace before `register` has taken its secret, so that one refused there keeps nothing alive.
        """
        ...

    async def existing_run(self, key: str) -> Run | None:
        """The run's state, or None if none is kept. Asking counts as the run being named, so the run is kept for its expiry from now."""
        ...

    async def peek_run(self, key: str) -> Run | None:
        """The run's state, kept or ended, or None if neither is kept, renewing nothing.

        Every check of which eval a run belongs to reads the run this way first, and renews it only once the check passes. So a caller refused for naming another eval's run keeps nothing of it alive, whether the run is still kept or its eval ended it.
        """
        ...

    async def end_run(self, key: str, eval_id: str | None = None) -> bool:
        """Drop a run's state, and remember that it ended. A request or reply of the run still being judged finds it ended too.

        Args:
            key: The run's key.
            eval_id: The eval ending it, if an eval is: a run of another eval is left as it is.

        Returns:
            Whether the run was known, and the eval's if one was given: kept, or ended before.
        """
        ...

    async def attempt(self, request: str, exchange: str, eval_id: str) -> Attempt:
        """Begin a new attempt at a request an eval names, once its header has passed the check.

        Args:
            request: The eval's id for the request. A client that retries the request sends it again under the same id.
            exchange: The proxy's id for this attempt's exchange.
            eval_id: The eval the request's header named and proved.
        """
        ...

    async def forget_attempt(
        self, request: str, eval_id: str, attempt: Attempt
    ) -> None:
        """Keep an attempt at a request no longer, as if it had never begun, for a request refused before it was judged for naming another eval's run."""
        ...

    async def attempts(
        self, request: str, eval_id: str
    ) -> list[Attempt] | Dropped | None:
        """The attempts at a request of an eval in order, `Dropped` if they were kept and no longer are, or None if the eval's request was never seen.

        A request is kept under its eval and its id together. So another eval, which proves its own token and may send any id, never adds to an eval's request.
        """
        ...

    async def eval_token(self, eval_id: str, *, renew: bool = True) -> str | None:
        """The secret an eval proves its calls to the endpoint for evals and its model requests with, or None if none is kept.

        Args:
            eval_id: The eval's id.
            renew: Whether asking counts as the eval being named, so the secret is kept for its expiry from now. False to check a caller's proof first, so that one who can't prove it keeps nothing alive.
        """
        ...

    async def keep_eval_token(self, eval_id: str, token: str, limit: int) -> Registered:
        """Keep the secret an eval proves its calls and requests with, for as long as a run is kept once no call names the eval, for a registration that stores no task.

        One step, as `register` is: the check that the eval has no other secret, the check that there is room, and the keeping.

        Args:
            eval_id: The eval's id.
            token: The secret the registration sent.
            limit: How many evals' secrets are kept at once.

        Returns:
            `kept` if it is kept now or was already, which keeps it for its expiry from now; `tokens_full`; or `another_token`, which renews nothing.
        """
        ...


class MemoryState:
    """State kept in this process, and lost when it stops."""

    def __init__(self, expiry: float = RUN_EXPIRY) -> None:
        """Create the state.

        Args:
            expiry: How long a run is kept once no request has named it, in seconds. A request still awaiting its reply, and what was recorded for a request, are kept as long.
        """
        self._runs = _Expiring[Run](expiry)
        self._held = _Expiring[Held](expiry)
        self._recorders = _Expiring[ListRecorder](expiry)
        # what is left of a run its eval ended: only that it ended
        self._ended = _Expiring[Run](expiry)
        # a request's id outlives its attempts, so the eval can be told they
        # were dropped and not that the request was never seen
        self._dropped = _Expiring[Dropped](expiry, limit=DROPPED_IDS)
        self._attempts = _Expiring[list[Attempt]](
            expiry,
            dropped=lambda key, attempts: self._dropped.put(
                key, Dropped(attempts[-1].eval_id if attempts else None)
            ),
        )
        # each registration has an expiry of its own, and there are few, so
        # they are swept whole
        self._tasks: dict[tuple[str, str], tuple[float, float, RegisteredTask]] = {}
        self._pinned_tasks: Counter[tuple[str, str]] = Counter()
        self._tokens = _Expiring[str](expiry)

    async def run(self, key: str) -> Run:
        run = self._runs.get(key)
        if run is not None:
            return run
        ended = self._ended.get(key)
        if ended is not None:
            return ended
        run = Run()
        self._runs.put(key, run)
        return run

    async def hold(self, id: str, held: Held) -> None:
        self._held.put(id, held)

    async def release(self, id: str) -> Held | None:
        return self._held.pop(id)

    async def pin(self, pinned: Pinned) -> None:
        self._runs.pin(pinned.run)
        self._tokens.pin(pinned.eval_id)
        self._attempts.pin(_request_key(pinned.request, pinned.eval_id))
        for exchange in pinned.exchanges:
            self._recorders.pin(exchange)
        self._pinned_tasks[(pinned.eval_id, pinned.task)] += 1

    async def unpin(self, pinned: Pinned) -> None:
        self._runs.unpin(pinned.run)
        self._tokens.unpin(pinned.eval_id)
        self._attempts.unpin(_request_key(pinned.request, pinned.eval_id))
        for exchange in pinned.exchanges:
            self._recorders.unpin(exchange)
        task = (pinned.eval_id, pinned.task)
        self._pinned_tasks[task] -= 1
        if self._pinned_tasks[task] <= 0:
            del self._pinned_tasks[task]
            kept = self._tasks.get(task)
            if kept is not None:
                # it expires again, counted from now
                self._tasks[task] = (time.monotonic(), kept[1], kept[2])

    async def recorder(self, id: str) -> ListRecorder:
        recorder = self._recorders.get(id)
        if recorder is None:
            recorder = ListRecorder()
            self._recorders.put(id, recorder)
        return recorder

    async def records(self, id: str) -> list[Record] | None:
        recorder = self._recorders.get(id)
        return recorder.records if recorder is not None else None

    async def register(
        self,
        eval_id: str,
        task: str,
        facts: RegisteredTask,
        expiry: float,
        limit: int,
        *,
        token: str | None = None,
    ) -> Registered:
        # Nothing is awaited from the first check to the last write, so
        # nothing comes between them.
        self._expire_tasks()
        refused = self._token_refused(eval_id, token, limit)
        if refused is not None:
            return refused
        if (eval_id, task) not in self._tasks and len(self._tasks) >= limit:
            return "full"
        self._tasks[(eval_id, task)] = (time.monotonic(), expiry, facts)
        if token is not None:
            self._tokens.put(eval_id, token)
        return "kept"

    async def registration(
        self, eval_id: str, task: str, *, renew: bool = True
    ) -> RegisteredTask | None:
        self._expire_tasks()
        kept = self._tasks.get((eval_id, task))
        if kept is None:
            return None
        _, expiry, facts = kept
        if renew:
            self._tasks[(eval_id, task)] = (time.monotonic(), expiry, facts)
        return facts

    async def existing_run(self, key: str) -> Run | None:
        return self._runs.get(key)

    async def peek_run(self, key: str) -> Run | None:
        kept = self._runs.peek(key)
        return kept if kept is not None else self._ended.peek(key)

    async def end_run(self, key: str, eval_id: str | None = None) -> bool:
        # a run never seen is left as it is, rather than refused from then
        # on; and another eval ending the run keeps nothing alive
        kept = await self.peek_run(key)
        if kept is None or eval_id not in (None, kept.eval_id):
            return False
        run = self._runs.pop(key)
        if run is None:
            # ended before: ending it again renews what is left of it
            self._ended.get(key)
            return True
        run.closed = True
        if run.refusal is None:
            # held by any request or reply of the run still being judged
            run.refusal = Refuse(400, "This run has ended.")
        if self._ended.get(key) is None:
            self._ended.put(
                key, Run(refusal=run.refusal, closed=True, eval_id=run.eval_id)
            )
        return True

    async def attempt(self, request: str, exchange: str, eval_id: str) -> Attempt:
        key = _request_key(request, eval_id)
        attempts = self._attempts.get(key)
        if attempts is None:
            attempts = list[Attempt]()
            self._attempts.put(key, attempts)
        attempt = Attempt(exchange, eval_id)
        attempts.append(attempt)
        return attempt

    async def forget_attempt(
        self, request: str, eval_id: str, attempt: Attempt
    ) -> None:
        key = _request_key(request, eval_id)
        attempts = self._attempts.peek(key)
        if attempts is None:
            return
        # by identity, as two attempts may hold the same values
        attempts[:] = [kept for kept in attempts if kept is not attempt]
        if not attempts:
            self._attempts.pop(key)
        # an eval waiting on the attempt looks again, and finds it gone
        attempt.settled.set()

    async def attempts(
        self, request: str, eval_id: str
    ) -> list[Attempt] | Dropped | None:
        key = _request_key(request, eval_id)
        attempts = self._attempts.get(key)
        if attempts is not None:
            return attempts
        return self._dropped.get(key)

    async def eval_token(self, eval_id: str, *, renew: bool = True) -> str | None:
        return self._tokens.get(eval_id) if renew else self._tokens.peek(eval_id)

    async def keep_eval_token(self, eval_id: str, token: str, limit: int) -> Registered:
        refused = self._token_refused(eval_id, token, limit)
        if refused is not None:
            return refused
        self._tokens.put(eval_id, token)
        return "kept"

    def _token_refused(
        self, eval_id: str, token: str | None, limit: int
    ) -> Registered | None:
        # why a registration's secret can't be kept, or None if it can, or
        # the registration sent none; read without renewing, so a caller
        # with another secret keeps nothing alive
        if token is None:
            return None
        kept = self._tokens.peek(eval_id)
        if kept is None:
            return "tokens_full" if self._tokens.count() >= limit else None
        if not hmac.compare_digest(token.encode(), kept.encode()):
            return "another_token"
        return None

    def _expire_tasks(self) -> None:
        now = time.monotonic()
        for key, (used, expiry, _) in list(self._tasks.items()):
            if now - used >= expiry and key not in self._pinned_tasks:
                del self._tasks[key]


def _request_key(request: str, eval_id: str) -> str:
    # the eval and the request's id, in a form no two pairs share
    return json.dumps([eval_id, request])


class _Expiring(Generic[V]):
    # oldest use first, so what has expired is always at the front
    def __init__(
        self,
        expiry: float,
        *,
        limit: int | None = None,
        dropped: Callable[[str, V], None] | None = None,
    ) -> None:
        self._expiry = expiry
        self._limit = limit
        self._dropped = dropped
        self._kept: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._pinned: Counter[str] = Counter()

    def pin(self, key: str) -> None:
        # kept past its expiry until unpinned, as often as it was pinned
        self._pinned[key] += 1

    def unpin(self, key: str) -> None:
        self._pinned[key] -= 1
        if self._pinned[key] <= 0:
            del self._pinned[key]
            kept = self._kept.get(key)
            if kept is not None:
                # it expires again, counted from now
                self.put(key, kept[1])

    def get(self, key: str) -> V | None:
        self._expire()
        kept = self._kept.get(key)
        if kept is None:
            return None
        self.put(key, kept[1])
        return kept[1]

    def peek(self, key: str) -> V | None:
        # what `get` gives, without counting as a use
        self._expire()
        kept = self._kept.get(key)
        return kept[1] if kept is not None else None

    def put(self, key: str, value: V) -> None:
        # swept here too: what is only ever put, such as a request whose reply
        # never comes, would otherwise be kept for good
        self._expire()
        self._kept[key] = (time.monotonic(), value)
        self._kept.move_to_end(key)
        if self._limit is not None and len(self._kept) > self._limit:
            self._kept.popitem(last=False)

    def pop(self, key: str) -> V | None:
        self._expire()
        kept = self._kept.pop(key, None)
        return kept[1] if kept is not None else None

    def count(self) -> int:
        self._expire()
        return len(self._kept)

    def _expire(self) -> None:
        now = time.monotonic()
        # each entry is looked at once at most, as a pinned one goes to the back
        for _ in range(len(self._kept)):
            key, (used, value) = next(iter(self._kept.items()))
            if now - used < self._expiry:
                return
            if key in self._pinned:
                # kept, and counted as used now, so that what follows it is
                # still looked at
                self._kept[key] = (now, value)
                self._kept.move_to_end(key)
                continue
            self._kept.pop(key)
            if self._dropped is not None:
                self._dropped(key, value)
