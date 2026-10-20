from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallView
from pydantic import JsonValue

from .._context import EvalContext, InspectSample, ProxyContext
from .._host import HostNative
from .._integration import HostContext, resolve_sentinel, run_sentinel, watched_stages
from .._report import Decision
from .._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from .._types import Sentinels
from ._errors import TimeLimitError
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
from ._history import record_sent
from ._host import Record, RunHost, SidecarHost, stage
from ._inspect_ai import VIEWED_TOOLS, tool_call_view
from ._inspect_client import (
    Exchange,
    Judged,
    Withheld,
    inspect_sample,
    sentinel_field,
    with_sentinel,
)
from ._registration import (
    MAX_TASK_EXPIRY,
    MAX_TASK_REGISTRATIONS,
    TASK_EXPIRY,
    FileTask,
    RegistrationError,
    TaskFacts,
    UnregisteredSampleError,
)
from ._reply_stream import PENDING_BYTES, ReplyStart, ReplyStream
from ._state import (
    MADE_KEPT,
    Attempt,
    Conversation,
    Held,
    Made,
    MemoryState,
    NotJudged,
    Rejected,
    Run,
    State,
    occurrence,
)
from ._wire import (
    Edited,
    Edits,
    UnreadableError,
    amend_request,
    amendments,
    call_texts,
    check_edited_reply,
    edit_reply,
    held_rejections,
    offered,
    read_opening,
    read_reply,
    read_reply_noted,
    read_request,
    request_format,
    request_marks,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# rejected turns in a row before the run is ended, as inspect_ai's bridge has
# it: a model that keeps proposing rejected calls would otherwise never stop
_MAX_CONSECUTIVE_REJECTIONS = 3

# what inspect_ai tells the model of a rejected call that has no message
_REJECTED = "Tool call not approved."
_NOT_RUN = "This call was not run, because another call in the same turn was rejected."

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
class RunResult:
    """What a run came to."""

    outcome: Literal["proceed", "terminate"]
    """`terminate` if a sentinel ended the run."""
    store: dict[str, Any]
    """The sentinel's store, by the keys the sentinel wrote."""


def _not_judged(ex: Exception) -> NotJudged:
    if isinstance(ex, UnreadableError):
        return "unreadable"
    if isinstance(ex, UnregisteredSampleError):
        return "unregistered"
    if isinstance(ex, TimeLimitError):
        return "time_limit"
    return "error"


def _settle(held: Held, status: int, answer: Answer) -> None:
    # where the request's attempt stands, once its reply has been answered
    attempt = held.attempt
    if attempt is None:
        return
    if not succeeded(status):
        attempt.not_judged("provider_error")
    elif not isinstance(answer, Refuse):
        attempt.decide("proceed")
        attempt.proposed = held.exchange.proposed
    elif held.exchange.ended:
        attempt.decide("terminate")
        attempt.withheld = held.exchange.withheld
    else:
        attempt.decide("reject", held.exchange.message)
        attempt.withheld = held.exchange.withheld


class ModifyError(Exception):
    """The sentinel decided to change a tool call that the sidecar can't change in the reply, such as one that is not a function call, so the reply was not judged."""


@dataclass
class _Decided:
    """What was decided about a turn's calls, short of ending the run."""

    rejected: list[tuple[ToolCall, Decision]] = field(
        default_factory=list[tuple[ToolCall, Decision]]
    )
    modified: dict[str, ToolCall] = field(default_factory=dict[str, ToolCall])


class Handler:
    """Judges what a proxy hands over with a sentinel, and answers what the proxy should do.

    It knows nothing of any proxy, nor of how it is reached. The endpoint for proxies turns each message it is sent into a `Request` or a `Reply`, and turns the `Answer` back into what it answers with.

    The four steps are found in the traffic. A request carries the results of the tool calls the client ran, judged as `AfterToolCall`, and is judged as `BeforeGenerate` before it goes to the provider. A reply is judged as `AfterGenerate`, and each tool call in it as `BeforeToolCall`, before the client is handed any of it.

    An Inspect eval names the sample of each request in its `x-inspect-context` header, and the sample is the run. It is told what was decided in the `sentinel` field: on a refusal, and on every reply, which is then answered with `Replace`. Any other client is handed the provider's reply as it was.
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
        task_expiry: float = MAX_TASK_EXPIRY,
        max_task_registrations: int = MAX_TASK_REGISTRATIONS,
        release: Literal["whole", "early"] = "whole",
    ) -> None:
        """Create a handler.

        Args:
            sentinel: One protocol, or a sequence or mapping of monitors and protocols, as a task's `sentinel=` takes.
            host: What the sentinel may do to the outside world. Defaults to a `SidecarHost` with no model roles.
            state: Where runs are kept. Defaults to this process's memory.
            reject_status: The HTTP status a rejected turn is refused with. The default, 400, is one a client takes as final. A status a client retries by itself, such as 503, brings the client back with the same request, which then goes on with the rejection added.
            time_limit: The longest the sentinel may take to judge one request or one reply, in seconds, after which it is cancelled. None, the default, is no limit.
            eval_results: Whether an Inspect client's `sentinel` field holds the results: the explanations, references and metadata of the decisions, and what each monitor and protocol recorded. Without them, it holds only what Inspect acts on. A monitor's explanation tells an agent what the monitor looks for, so a client that can't be trusted is not sent them.
            tasks: Tasks registered beforehand, by name, as the tasks folder's files hold them. Nothing an eval registers replaces one, and a request for a sample one doesn't hold is refused.
            runtime_task_registration: Whether an eval may register its tasks itself, at task init.
            task_expiry: The longest a task an eval registered is kept once no request has named it, in seconds, whatever the registration asks.
            max_task_registrations: How many tasks registered by evals are kept at once. A registration past it is refused.
            release: How a streamed reply is passed on. `whole`, the default, holds every reply whole until the sentinel has judged it. `early` passes on text and reasoning as they arrive and holds each call until it is judged, which says the sentinel can judge a complete call with only the part of the turn that has arrived. A sentinel that watches `AfterGenerate` is given the finished turn, so its replies are held whole whatever this says.
        """
        self._protocol = resolve_sentinel(sentinel)
        # a stage nothing watches is never built, as in Inspect's own process
        self._watched = watched_stages(self._protocol)
        self._release: Literal["whole", "early"] = release
        if release == "early" and AfterGenerate in self._watched:
            self._release = "whole"
            logger.warning(
                "The sentinel watches AfterGenerate, which is given the finished turn, so replies are held whole although early release was asked for."
            )
        self._host: HostNative = host if host is not None else SidecarHost()
        self._state: State = state if state is not None else MemoryState()
        self._reject_status = reject_status
        self._time_limit = time_limit
        self._eval_results = eval_results
        self._tasks = dict(tasks or {})
        self._runtime_task_registration = runtime_task_registration
        self._task_expiry = task_expiry
        self._max_task_registrations = max_task_registrations
        # the tool results being judged now, by run, conversation and digest,
        # so that a request holding one waits for the request judging it
        self._judging: dict[tuple[str, str, str], anyio.Event] = {}

    @property
    def release(self) -> Literal["whole", "early"]:
        """How a streamed reply is passed on: `early` only when it was asked for and the sentinel doesn't watch `AfterGenerate`."""
        return self._release

    async def request(self, request: Request) -> Answer:
        """Judge a request before the proxy sends it.

        Each tool result in the request that the sentinel hasn't been shown is put to it as an `AfterToolCall` step, in order. A result is shown once, however often the client sends it. The request is then put to the sentinel as a `BeforeGenerate` step. A stage nothing in the sentinel watches is left out. If either ends the run, the request is refused.

        If turns were rejected earlier in the conversation this request continues, they are added where they happened, so the model is still told.

        A request with `x-inspect-context` belongs to the sample it names, which is its run. A refusal then carries the `sentinel` field. A request whose header can't be read is refused, rather than judged as another client's would be: it says it is an eval's, and a proxy set to let through what can't be judged would let it through unjudged.

        A conversation belongs to one attempt at the sample. A new attempt starts its conversations, and its count of rejected turns in a row, afresh, while the run's store, and the run's end if a sentinel ended it, carry over. A client that retries a request within the attempt keeps the attempt's conversation.

        A request of a run that a sentinel has ended is refused, and the sentinel is not asked again. For a run no eval names, that is a request which continues the conversation the run was ended in.

        Args:
            request: The request, as the proxy handed it over.

        Raises:
            UnreadableError: If the request can't be read, so neither it nor its reply could be judged.
            UnregisteredSampleError: If the request names a sample that a task registered beforehand doesn't hold.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        id = request.headers.get(REQUEST_ID)
        attempt = None
        if id:
            attempt = await self._state.attempt(id, request.id)
            # an attempt that is never judged has records too: none
            await self._state.recorder(request.id)
        try:
            sample = inspect_sample(request.headers)
        except UnreadableError as ex:
            # refused rather than raised, but never judged, so not a terminate
            if attempt is not None:
                attempt.not_judged("unreadable")
            return Refuse(400, str(ex))
        try:
            answer = await self._within_limit(
                self._request(request, sample, attempt), "request", request.id
            )
        except Exception as ex:
            if attempt is not None:
                attempt.not_judged(_not_judged(ex))
            raise
        if attempt is not None:
            if isinstance(answer, Refuse):
                # a request is refused only when its run has ended
                attempt.decide("terminate")
            else:
                attempt.state = "awaiting_reply"
        return answer

    async def reply(self, reply: Reply) -> Answer:
        """Judge a reply before the proxy passes any of it on.

        The reply is put to the sentinel as an `AfterGenerate` step, then each tool call in it as a `BeforeToolCall` step, in order, leaving out a stage nothing in the sentinel watches. Every call is judged, as in Inspect's own process, unless one ends the run. The rejected calls are taken out of the reply and the rest of the turn goes on, answered with `Replace`, as the other calls of a turn still run in Inspect's own process. Each later request of the conversation that holds the turn has the calls taken out put back, each with its rejection as its result. A turn whose every call is rejected is refused whole, as is one that can't be edited, and its rejections are added to the later requests of its conversation. The third turn refused whole in a row within an attempt at a sample ends the run; a turn in which a call ran starts the count again. A provider's error, a reply in a status outside 2xx, is passed on unjudged; a reply in any 2xx is the model's, and judged, as the provider's SDK reads it so.

        For an Inspect client, a reply that goes on is answered with `Replace`, the reply with the `sentinel` field added, and a refusal carries the field. The field holds every step judged during the exchange, the request's first. A reply that goes on but has no event the field can go in, such as a stream that never sends its last event, is handed over as judged, without the field.

        Args:
            reply: The reply, as the proxy handed it over.

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
                self._reply(reply, held), "reply", reply.id
            )
        except Exception as ex:
            if attempt is not None:
                attempt.not_judged(_not_judged(ex))
            raise
        if held is not None:
            _settle(held, reply.status, answer)
        return answer

    def reply_stream(
        self,
        start: ReplyStart,
        *,
        max_held_bytes: int = MAX_BODY_BYTES,
        max_pending_bytes: int = PENDING_BYTES,
    ) -> ReplyStream:
        """Judge a reply that the proxy hands over as it arrives, rather than whole.

        The proxy runs the stream's `run`, feeds it the provider's bytes, and sends the client what iterating over it gives, all at once. The reply is judged as `reply` judges it.

        Args:
            start: The reply's status and headers, before any of its body.
            max_held_bytes: The most of the reply the stream holds back to judge. A reply larger than it is refused with a 502, and not judged.
            max_pending_bytes: How much of what goes out as it arrives may wait for the proxy to collect it before feeding waits.
        """
        return ReplyStream(
            self,
            start,
            max_held_bytes=max_held_bytes,
            max_pending_bytes=max_pending_bytes,
        )

    async def early_turn(
        self, start: ReplyStart, *, stopped: Callable[[], bool] = lambda: False
    ) -> EarlyTurn:
        """Judge a streamed reply as it arrives, each call once it is complete, for a reply released early.

        Args:
            start: The reply's status and headers, before any of its body.
            stopped: Whether the proxy has stopped waiting for the reply, as when its client went. No call is put to the sentinel after that; one being judged then is decided.

        Raises:
            UnreadableError: If the reply answers a request this handler was never shown, or no longer keeps.
        """
        held = await self._state.release(start.id)
        if held is None:
            raise UnreadableError(
                f"The reply {start.id!r} answers a request this sidecar was never shown, or no longer keeps."
            )
        if held.attempt is not None:
            held.attempt.state = "judging"
        return EarlyTurn(self, start, held, stopped)

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

    async def request_result(
        self, id: str
    ) -> RequestResult | Literal["dropped", "unknown"]:
        """What happened to a request an eval names, by the eval's id for it.

        Args:
            id: The eval's id for the request, as its `x-irid` header gave it. A client that retries a request sends it again under the same id.

        Returns:
            Where the latest attempt stands, and every attempt's records in order; `dropped` if they were kept and no longer are; or `unknown` if the request was never seen.
        """
        attempts = await self._state.attempts(id)
        if attempts is None:
            return "unknown"
        if attempts == "dropped":
            return "dropped"
        records: list[Record] = []
        for attempt in attempts:
            kept = await self._state.records(attempt.exchange)
            if kept is None:
                return "dropped"
            records.extend(kept)
        return RequestResult(attempts[-1], records)

    async def run_result(self, run: str) -> RunResult | None:
        """What a run came to, and the sentinel's store, for an eval to copy into its sample's store before scoring.

        Args:
            run: The run, as the eval names it: its sample's uuid.

        Returns:
            The run's outcome and store, or None if nothing is kept for it.
        """
        kept = await self._state.existing_run(_run_key(True, run))
        if kept is None:
            return None
        return RunResult(
            "terminate" if kept.refusal is not None else "proceed",
            dict(kept.store.items()),
        )

    async def end_run(self, run: str) -> Literal["ended", "unknown"]:
        """Drop a run's state, once its eval has scored the sample. A later request of the run is refused, as for a run a sentinel ended.

        Args:
            run: The run, as the eval names it: its sample's uuid.

        Returns:
            `ended`, or `unknown` for a run the handler never saw. Ending a run again has no effect.
        """
        return "ended" if await self._state.end_run(_run_key(True, run)) else "unknown"

    async def _within_limit(self, judging: Awaitable[T], what: str, id: str) -> T:
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
        self, eval_id: str, facts: TaskFacts, expiry: float | None = None
    ) -> float | None:
        """Keep a task an eval registered, so that its runs' sentinels are told which sample each is.

        A registration replaces the eval's last one for the task. A run that has already found its sample keeps what it found.

        Args:
            eval_id: The eval's id, as its requests name it.
            facts: The task and its samples.
            expiry: How long to keep it once no request has named the task, in seconds. A day if not given.

        A task registered beforehand is already registered, so its registration stores nothing and is not compared with the file, since the answer would tell a caller whether its guess at the file is right. This holds whether or not the handler takes registrations from evals.

        Returns:
            How long it is kept once no request has named the task: `expiry`, or less if that is longer than the handler keeps any. None for a task registered beforehand, which is never dropped.

        Raises:
            RegistrationError: If the handler takes no registrations from evals and the task was not registered beforehand, or already keeps as many as it takes.
        """
        if facts.task in self._tasks:
            return None
        if not self._runtime_task_registration:
            raise RegistrationError(
                "This sidecar takes no tasks registered by an eval."
            )
        granted = min(expiry if expiry is not None else TASK_EXPIRY, self._task_expiry)
        if not await self._state.register(
            eval_id, facts.task, facts, granted, self._max_task_registrations
        ):
            raise RegistrationError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes. One is taken again once a kept task expires."
            )
        return granted

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
            facts = await self._state.registration(sample.eval_id, sample.task)
            return facts.eval_context(sample) if facts is not None else None
        found = fixed.eval_context(sample)
        if found is None:
            raise UnregisteredSampleError(
                f"The task {sample.task!r} is registered with this sidecar beforehand, without the sample {sample.sample_id!r}."
            )
        return found

    async def _request(
        self, request: Request, sample: InspectSample | None, attempt: Attempt | None
    ) -> Answer:
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
        marks = request_marks(request.provider, request.body, request.format)
        refusal = _ended(run, marks)
        if refusal is not None:
            # the run has ended, and the sentinel is not asked again
            return self._with_field(refusal, proxy, Exchange(), "terminate")
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

        answer: Answer = Pass()
        # only what the client sent holds results to judge: what the sidecar
        # adds below is its own writing
        client = messages
        # A point this request doesn't continue from is kept for the requests
        # that do: a run's key is shared by conversations that open alike. A
        # client that was handed a rejected turn, as an Inspect eval is in the
        # sentinel field, may already have put it back itself.
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

        held = Held(
            run=run_key,
            conversation=conversation_key,
            named=named,
            sample_attempt=sample_attempt,
            answered=any(mark in run.answered for mark in marks[:-1]),
            model=_model(request.provider, request.model),
            format=request_format(request.provider, request.body, request.format),
            recorder=await self._state.recorder(request.id),
            input=messages,
            calls=_calls(client),
            offered=sends,
            marks=marks,
            proxy=proxy,
            attempt=attempt,
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
            return self._with_field(ended, proxy, held.exchange, "terminate")
        if BeforeGenerate in self._watched:
            before = BeforeGenerate(
                model=held.model,
                conversation=conversation_key,
                input=messages,
                # the caller's own list is not known behind a proxy
                agent_messages=None,
                # what was sent before this request, which isn't sent yet
                history=list(conversation.sent),
                tools=sends.tools,
                tool_choice=sends.tool_choice,
                config=sends.config,
            )
            if _ends(await self._judge(held, run, before)):
                return self._with_field(
                    self._end(held, run, "before the model was called"),
                    proxy,
                    held.exchange,
                    "terminate",
                )
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, marks)
        if refusal is not None:
            return self._with_field(refusal, proxy, held.exchange, "terminate")
        # the request goes on to the model, as the client sent it with what
        # this sidecar added
        record_sent(
            conversation,
            messages,
            amended.body if amended.body is not None else request.body,
        )
        await self._state.hold(request.id, held)
        # an Inspect client is told of the request's steps with the reply
        return answer

    async def _results(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        client: list[ChatMessage],
        said: Mapping[str, Sequence[str]],
        restored: frozenset[str] = frozenset(),
        ran: Mapping[str, ToolCall] | None = None,
    ) -> Refuse | None:
        if AfterToolCall not in self._watched:
            return None
        made: dict[str, tuple[ChatMessageAssistant, ToolCall, str, int]] = {}
        held_calls = _calls(client)
        # the calls the client holds count too, those this sidecar never
        # handed out included
        for call_id, count in held_calls.items():
            issued = max(conversation.issued.get(call_id, 0), count)
            conversation.issued[call_id] = issued
        # how many calls with each id have come before, as a client can use
        # one id for calls in different turns, each with words of its own
        occurrences: dict[str, int] = {}
        for entry in client:
            if isinstance(entry, ChatMessageAssistant):
                for call in entry.tool_calls or []:
                    index = occurrences.get(call.id, 0)
                    occurrences[call.id] = index + 1
                    written = said.get(call.id, ())
                    words = written[index] if index < len(written) else entry.text
                    made[call.id] = (entry, call, words, index)
            if not isinstance(entry, ChatMessageTool):
                continue
            found = made.get(entry.tool_call_id or "")
            if found is None:
                raise UnreadableError(
                    f"The request holds a result for a call it doesn't hold: {entry.tool_call_id!r}."
                )
            assistant, call, words, index = found
            call = (ran or {}).get(call.id, call)
            # A call's place among those with its id tells it apart from the
            # others, as a client can use one id for calls in different
            # turns. A conversation that lost calls with its id, to
            # compaction, shifted that place, and a result there can't be
            # told from one judged before, so it is refused rather than
            # judged as a guess.
            if held_calls.get(call.id, 0) < conversation.issued.get(call.id, 0):
                raise UnreadableError(
                    f"The request holds fewer calls with the id {call.id!r} than its conversation did, so its results can't be told apart from those judged before."
                )
            # the call's occurrence too: a client that runs two calls with one
            # id, the same arguments and the same result ran them both
            digest = _result_digest(words, call, entry, index)
            key = (held.run, held.conversation, digest)
            if key in self._judging:
                while (judging := self._judging.get(key)) is not None:
                    await judging.wait()
                # the judgment waited for may have ended the run
                refusal = _ended(run, held.marks)
                if refusal is not None:
                    return refusal
            # a rejection the client put back itself is the sidecar's own
            # writing, not the result of a call that ran
            if digest in conversation.judged or entry.tool_call_id in restored:
                continue
            sent = conversation.made.get(occurrence(call.id, index))
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
                # left empty: only what the model is sent crosses the wire, and
                # a monitor must not take that for everything the tool returned
                output="",
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

    async def _reply(self, reply: Reply, held: Held | None) -> Answer:
        answer = await self._judge_reply(reply, held)
        if held is None or not succeeded(reply.status):
            return answer
        if held.proxy.inspect is None:
            return answer
        if isinstance(answer, Refuse):
            outcome = "terminate" if held.exchange.ended else "reject"
            return self._with_field(answer, held.proxy, held.exchange, outcome)
        body = answer.body if isinstance(answer, Replace) else reply.body
        field = self._field(held.exchange, "proceed")
        try:
            return Replace(with_sentinel(reply.content_type, body, field))
        except UnreadableError:
            # The reply was judged and goes on, and the run already counts it
            # as passed, so the client is handed it as judged, without the
            # field. Inspect then asks what was decided, by `request_result`.
            return answer

    async def _judge_reply(self, reply: Reply, held: Held | None) -> Answer:
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
        refusal = _ended(run, held.marks)
        if refusal is not None:
            # a sentinel ended the run while the model was answering
            held.exchange.ended = True
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
                return self._end(held, run, "at the model's reply")

        if not output.choices:
            return self._ended_meanwhile(held, run) or self._passed(
                held, run, conversation, None
            )
        message = output.message
        # the reply is still read, and kept as passed, when no call is judged
        calls = (message.tool_calls or []) if BeforeToolCall in self._watched else []
        decided = _Decided()
        for call in calls:
            ended = await self._judge_call(
                held, run, conversation, message, call, decided, turn_complete=True
            )
            if ended is not None:
                return ended
        return await self._conclude(
            reply, held, run, conversation, output, start, noted, decided
        )

    async def _judge_call(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant,
        call: ToolCall,
        decided: _Decided,
        *,
        turn_complete: bool,
    ) -> Refuse | None:
        # a call put to the sentinel, which ends the run or is noted in `decided`
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
            turn_complete=turn_complete,
        )
        decision = await self._judge(held, run, step)
        if decision is None or decision.action == "proceed":
            return None
        if decision.action == "terminate":
            return self._end(held, run, f"at the call to {call.function}")
        # the client is never handed the call as the model proposed it
        held.exchange.judged[-1].original = call
        if decision.action == "modify" and decision.modified is not None:
            decided.modified[call.id] = decision.modified
        else:
            decided.rejected.append((call, decision))
        return None

    async def _conclude(
        self,
        reply: Reply,
        held: Held,
        run: Run,
        conversation: Conversation,
        output: ModelOutput,
        start: JsonValue | None,
        noted: JsonValue | None,
        decided: _Decided,
        *,
        edited: bytes | None = None,
    ) -> Answer:
        # what the turn comes to, once each of its calls is decided. `edited`
        # is the reply as it already went out with the decisions made, for
        # one released as it arrived, which is checked rather than made.
        message = output.message
        rejected, modified = decided.rejected, decided.modified
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
                edited=edited,
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
            seen = dict(held.calls)
            for call in message.tool_calls or []:
                index = seen.get(call.id, 0)
                seen[call.id] = index + 1
                assert message.id is not None
                made = Made(
                    held.input,
                    held.offered.tools,
                    held.offered.tool_choice,
                    message.id,
                )
                conversation.made[occurrence(call.id, index)] = made
                # its place, not one more each time: a retry or another
                # reply to the same request hands out a call in the same
                # place, of which the client keeps one
                issued = max(conversation.issued.get(call.id, 0), index + 1)
                conversation.issued[call.id] = issued
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
        edited: bytes | None = None,
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
        edits = Edits(frozenset(reasons), modified)
        if edited is None:
            body = await edit_reply(
                reply.provider,
                reply.content_type,
                reply.body,
                reply.format,
                edits,
                output,
                start=start,
            )
        else:
            await check_edited_reply(
                reply.provider,
                reply.content_type,
                edited,
                reply.format,
                edits,
                output,
                start=start,
            )
            body = edited
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
        held.exchange.proposed = Withheld(message, reasons)
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
        held.exchange.withheld = Withheld(message, told)
        held.exchange.message = why
        run.in_a_row[held.sample_attempt] = count
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
        decision = await run_sentinel(
            self._protocol,
            HostContext(
                host=RunHost(self._host, run.eval),
                recorder=held.recorder,
                store=run.store,
                proxy=held.proxy,
            ),
            step,
        )
        call = (
            step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else None
        )
        held.exchange.judged.append(
            Judged(stage(step), call, decision, held.recorder.records[recorded:])
        )
        return decision

    def _with_field(
        self,
        refusal: Refuse,
        proxy: ProxyContext,
        exchange: Exchange,
        outcome: Literal["reject", "terminate"],
    ) -> Refuse:
        # an Inspect client is told what was decided, with the refusal
        if proxy.inspect is None:
            return refusal
        return replace(refusal, sentinel=self._field(exchange, outcome))

    def _field(
        self, exchange: Exchange, outcome: Literal["proceed", "reject", "terminate"]
    ) -> dict[str, Any]:
        return sentinel_field(exchange, outcome, trimmed=not self._eval_results)

    def _ended_meanwhile(self, held: Held, run: Run) -> Refuse | None:
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, held.marks)
        if refusal is not None:
            held.exchange.ended = True
        return refusal

    def _end(self, held: Held, run: Run, where: str) -> Refuse:
        held.exchange.ended = True
        refusal = Refuse(400, f"A sentinel ended this run {where}.")
        run.in_a_row.clear()
        if held.named:
            run.refusal = refusal
            # nothing of the run goes on, so nothing is put back again
            for conversation in run.conversations.values():
                conversation.rejected.clear()
                conversation.edited.clear()
            return refusal
        if held.answered:
            run.ended_at[held.mark] = refusal
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


class EarlyTurn:
    """A streamed reply judged as it arrives: each call once it is complete, then the turn once it has ended. Made by `Handler.early_turn`.

    Each call judged before the turn has ended is put to the sentinel with `turn_complete` false, and with the assistant text written before it. The whole reply is read again at its end, as `Handler.reply` reads it, and its calls must be the ones judged as it arrived. What is decided is kept as `Handler.reply` keeps it.
    """

    def __init__(
        self,
        handler: Handler,
        start: ReplyStart,
        held: Held,
        stopped: Callable[[], bool],
    ) -> None:
        self._handler = handler
        self._start = start
        self._held = held
        self._stopped = stopped
        self._decided = _Decided()
        self._judged: dict[str, ToolCall] = {}
        self._message_id: str | None = None
        self._ended: Refuse | None = None
        # the turn's answer, once it has one
        self._answer: Answer | None = None

    @property
    def edits(self) -> Edits:
        """The calls decided so far that are taken out, and those changed."""
        return Edits(
            frozenset(call.id for call, _ in self._decided.rejected),
            dict(self._decided.modified),
        )

    @property
    def field(self) -> Mapping[str, Any] | None:
        """For an Inspect client, the `sentinel` field for a turn that goes on, which goes in the stream's last event."""
        held = self._held
        if held.proxy.inspect is None:
            return None
        return self._handler._field(held.exchange, "proceed")  # pyright: ignore[reportPrivateUsage]

    async def judge(self, prefix: bytes) -> Refuse | None:
        """Judge the calls that have become complete, in order: each call in `prefix` not judged yet.

        Args:
            prefix: The reply so far, as the provider sent it, closed as if it ended there, without the calls that are not complete yet.

        Returns:
            The refusal, with the `sentinel` field for an Inspect client, if the run was ended. The turn's answer is then that refusal.

        Raises:
            UnreadableError: If the reply so far can't be read.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        if isinstance(self._answer, Refuse):
            return self._answer
        refusal = await self._guarded(self._judge(prefix))
        if refusal is None:
            return None
        answered = self._answered(refusal)
        assert isinstance(answered, Refuse)
        return answered

    async def finish(self, body: bytes, released: bytes) -> Answer | None:
        """Judge the turn once the reply has ended.

        Args:
            body: The whole reply, as the provider sent it.
            released: The whole reply, as the client was sent it, with the calls taken out and changed.

        Returns:
            What `Handler.reply` would answer, without the `sentinel` field on a turn that goes on, which for an Inspect client is `field`: `Pass`, `Replace` with `released` when calls were taken out or changed, or the refusal, with the field. None if the proxy stopped waiting before the turn could be answered.

        Raises:
            UnreadableError: If the reply can't be read, its calls are not those judged as it arrived, or what the client was sent doesn't read as the reply with the decisions made.
            ModifyError: If the sentinel changed a call the sidecar can't change, and rejected none of the turn's calls.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        if self._answer is not None:
            return self._answer
        answer = await self._guarded(self._finish(body, released))
        return None if answer is None else self._answered(answer)

    def failed(self, why: Exception | NotJudged) -> None:
        """Record a turn that couldn't be judged, because of an error, or for the reason given."""
        attempt = self._held.attempt
        if self._answer is None and attempt is not None and attempt.state == "judging":
            attempt.not_judged(why if isinstance(why, str) else _not_judged(why))

    def cut_off(self) -> None:
        """Record a turn whose stream ended before the turn was answered, as when its client disconnected."""
        attempt = self._held.attempt
        # a turn that failed to be judged keeps its reason
        if self._answer is None and attempt is not None and attempt.state == "judging":
            attempt.not_judged("cut_off")

    def _answered(self, answer: Answer) -> Answer:
        held = self._held
        if isinstance(answer, Refuse) and held.proxy.inspect is not None:
            outcome = "terminate" if held.exchange.ended else "reject"
            answer = self._handler._with_field(  # pyright: ignore[reportPrivateUsage]
                answer, held.proxy, held.exchange, outcome
            )
        self._answer = answer
        _settle(held, self._start.status, answer)
        return answer

    async def _guarded(self, judging: Awaitable[T]) -> T:
        try:
            return await self._handler._within_limit(  # pyright: ignore[reportPrivateUsage]
                judging, "reply", self._start.id
            )
        except Exception as ex:
            if self._held.attempt is not None:
                self._held.attempt.not_judged(_not_judged(ex))
            raise

    async def _judge(self, prefix: bytes) -> Refuse | None:
        handler, held, start = self._handler, self._held, self._start
        if self._ended is not None:
            return self._ended
        output = await read_reply(
            start.provider, start.content_type, prefix, start.format
        )
        run, conversation = await self._run()
        refusal = _ended(run, held.marks)
        if refusal is not None:
            # a sentinel ended the run while the model was answering
            held.exchange.ended = True
            self._ended = refusal
            return refusal
        if not output.choices:
            raise UnreadableError("The reply so far doesn't read as a message.")
        message = self._same_id(output.message)
        for call in message.tool_calls or []:
            if call.id in self._judged:
                continue
            if self._stopped():
                # a call being judged is decided, and no other is asked about
                return None
            self._judged[call.id] = call
            if BeforeToolCall not in handler._watched:  # pyright: ignore[reportPrivateUsage]
                continue
            ended = await handler._judge_call(  # pyright: ignore[reportPrivateUsage]
                held,
                run,
                conversation,
                message,
                call,
                self._decided,
                turn_complete=False,
            )
            if ended is not None:
                self._ended = ended
                return ended
        return None

    async def _finish(self, body: bytes, released: bytes) -> Answer | None:
        handler, held, start = self._handler, self._held, self._start
        run, conversation = await self._run()
        begun = conversation.noted
        output, noted = await read_reply_noted(
            start.provider, start.content_type, body, start.format, start=begun
        )
        # kept whatever is decided: the provider continues from what it sent
        conversation.noted = noted
        if self._ended is not None:
            return self._ended
        refusal = _ended(run, held.marks)
        if refusal is not None:
            held.exchange.ended = True
            return refusal
        reply = Reply(
            id=start.id,
            provider=start.provider,
            format=start.format,
            model=start.model,
            user=start.user,
            status=start.status,
            content_type=start.content_type,
            body=body,
        )
        if not output.choices:
            if self._judged:
                raise UnreadableError(
                    "The reply's calls are not those judged as it arrived."
                )
            return handler._ended_meanwhile(held, run) or handler._passed(  # pyright: ignore[reportPrivateUsage]
                held, run, conversation, None
            )
        message = self._same_id(output.message)
        output.choices[0].message = message
        calls = message.tool_calls or []
        given = {call.id: call for call in calls}
        if any(
            id not in given
            or (given[id].function, given[id].arguments)
            != (call.function, call.arguments)
            for id, call in self._judged.items()
        ):
            raise UnreadableError(
                "The reply's calls are not those judged as it arrived."
            )
        if BeforeToolCall in handler._watched:  # pyright: ignore[reportPrivateUsage]
            for call in calls:
                if call.id in self._judged:
                    continue
                if self._stopped():
                    return None
                ended = await handler._judge_call(  # pyright: ignore[reportPrivateUsage]
                    held,
                    run,
                    conversation,
                    message,
                    call,
                    self._decided,
                    turn_complete=True,
                )
                if ended is not None:
                    return ended
        return await handler._conclude(  # pyright: ignore[reportPrivateUsage]
            reply,
            held,
            run,
            conversation,
            output,
            begun,
            noted,
            self._decided,
            edited=released,
        )

    async def _run(self) -> tuple[Run, Conversation]:
        run = await self._handler._state.run(self._held.run)  # pyright: ignore[reportPrivateUsage]
        conversation = run.conversations.setdefault(
            self._held.conversation, Conversation()
        )
        return run, conversation

    def _same_id(self, message: ChatMessageAssistant) -> ChatMessageAssistant:
        # inspect_ai gives a message a new id each time it is read, and every
        # step of one turn names the same message
        if self._message_id is None:
            self._message_id = message.id
        return message.model_copy(update={"id": self._message_id})


def _run_key(named: bool, key: str) -> str:
    # Runs an eval names and runs it doesn't are kept apart. A run no eval
    # names is keyed by how its conversation opens, which anyone who knows the
    # opening can work out, so the eval endpoint, which takes a sample's uuid,
    # must never reach one.
    return f"{'sample' if named else 'opening'}:{key}"


def _ended(run: Run, marks: Sequence[str]) -> Refuse | None:
    # what a request is refused with, if its run has ended: any request of a
    # run an eval names, or one that continues the conversation it was ended
    # in
    return run.refusal or next(
        (run.ended_at[mark] for mark in marks if mark in run.ended_at), None
    )


def _ends(decision: Decision | None) -> bool:
    # an escalate nobody took has already been turned into a terminate
    return decision is not None and decision.action == "terminate"


def _model(provider: str, model: str) -> str:
    # a step names the model as `provider/name`
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


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
    ids = set(_calls(messages)) | {
        message.tool_call_id
        for message in messages
        if isinstance(message, ChatMessageTool) and message.tool_call_id is not None
    }
    return any(
        turn not in edited and ids.intersection(turn.order)
        for conversation in run.conversations.values()
        for turn in conversation.edited
    )


def _calls(messages: Sequence[ChatMessage]) -> dict[str, int]:
    # how many tool calls with each id the messages hold
    calls: dict[str, int] = {}
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            for call in message.tool_calls or []:
                calls[call.id] = calls.get(call.id, 0) + 1
    return calls


def _result_digest(
    said: str, call: ToolCall, result: ChatMessageTool, index: int
) -> str:
    # everything the step is given from the request, what the model said with
    # the call as well as the call and the result: the client wrote them all
    # into the request, so an id seen before doesn't vouch for what now
    # stands beside it
    digest = hashlib.sha256(
        json.dumps(
            [
                said,
                call.id,
                index,
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
