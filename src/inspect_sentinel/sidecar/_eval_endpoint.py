from __future__ import annotations

import base64
import hmac
import json
from pathlib import Path
from typing import Any, Literal

from inspect_ai.tool import ToolCall
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ._errors import UnauthorizedError
from ._exchange import Refuse
from ._handler import MAX_RESULT_WAIT, Handler, HeldReply, RequestResult
from ._http import discard_within, read_within
from ._inspect_client import (
    JudgingUsage,
    Judgment,
    judging_usage,
    judgment,
    record_json,
)
from ._json_rpc import (
    ANSWER_REFUSED,
    METHOD_NOT_FOUND,
    QUESTION_ANSWERED,
    REGISTRATION_REFUSED,
    REGISTRATIONS_FULL,
    UNAUTHORIZED,
    UNKNOWN_QUESTION,
    UNKNOWN_SAMPLE,
    UNKNOWN_TASK,
    RpcError,
    answered,
    failed,
    invalid_params,
    read_call,
)
from ._registration import (
    RegistrationError,
    RegistrationsFullError,
    SampleFacts,
    TaskRegistration,
    UnknownTaskError,
)
from ._review import (
    ANSWERABLE,
    MAX_QUESTIONS_WAIT,
    AnsweredError,
    ChoiceError,
    Question,
    UnknownQuestionError,
)
from ._sign import MIN_TOKEN_LENGTH, answer_values
from ._state import NotJudged, Outcome

PATH = "/sentinel/v1"
"""Where the methods are served. The `v1` is the version of the methods."""

HELD_REPLY_PATH = f"{PATH}/held-reply"
"""Where an eval gets the reply held for a model call a person was asked about. Not a method: the eval sends the model call's own request here, with the code that sends it to the provider, and is answered as the provider would answer."""

INFO_PATH = "/sentinel/info"
"""The probe, which has no version, since it is how an eval learns which versions there are."""

VERSIONS = [1]
"""The versions of the methods this sidecar serves."""

TOKEN_VARIABLE = "INSPECT_SENTINEL_EVAL_TOKEN"
"""The environment variable the endpoint's own token is read from, when it has a port of its own."""


_TIMESTAMP = "When the caller signed, in whole seconds since the epoch. A call whose timestamp is further from the sidecar's clock than its window, five minutes by default, is refused with `unauthorized`."

_MAC = 'An HMAC-SHA256, in hex, made with the eval\'s token over `json.dumps([str(v) for v in [method, eval_id, timestamp]], separators=(",", ":"))`, such as `["sample_result","eval_789","1794182400"]`. It proves only that the caller holds the token, not the call\'s other parameters.'


class EvalCall(BaseModel):
    """What every call for an eval presents, but a registration that sends the eval's token: the eval, and proof that the caller holds its token."""

    eval_id: str = Field(description="The eval's id, as its requests name it.")
    timestamp: int = Field(description=_TIMESTAMP)
    mac: str = Field(
        description=f"{_MAC} A call whose `mac` isn't made with the token the sidecar keeps for the eval is refused with `unauthorized`. A sidecar that keeps no token for the eval, as after a restart, holds nothing of it for anyone."
    )


class RegisterTaskParams(TaskRegistration):
    """Parameters of `register_task`: a task an eval runs, and the ids of its samples, sent at task init. It carries either the eval's `token`, or a `timestamp` and a `mac`, and not both."""

    eval_id: str = Field(description="The eval's id, as its requests name it.")
    token: str | None = Field(
        default=None,
        min_length=MIN_TOKEN_LENGTH,
        description=f"The token the eval made for this sidecar, at least {MIN_TOKEN_LENGTH} characters, sent only to a sidecar that holds no token for the eval: at the eval's first registration, when that registration is sent again because its answer was lost, or after the sidecar lost the token in a restart or once it expired. The sidecar keeps it once the registration succeeds, and never sends it back. A sidecar that keeps another token for the eval answers `unauthorized`.",
    )
    timestamp: int | None = Field(default=None, description=_TIMESTAMP)
    mac: str | None = Field(
        default=None,
        description=f"{_MAC} Sent in place of the token once the sidecar holds it. A sidecar that holds no token for the eval answers `unknown_task`, and the eval registers again with its token.",
    )
    expiry: float | None = Field(
        default=None,
        gt=0,
        description="How long to keep the registration once no request has named the task, in seconds. A day if not given.",
    )

    @model_validator(mode="after")
    def _token_or_mac(self) -> RegisterTaskParams:
        partly_signed = (self.timestamp is None) != (self.mac is None)
        if partly_signed or (self.token is None) == (self.mac is None):
            raise ValueError(
                "A registration carries the eval's token, or a timestamp and a mac, and not both."
            )
        return self


class RegisterTaskResult(BaseModel):
    """What the sidecar stored, or that the task was registered beforehand and nothing was stored."""

    eval_id: str
    task: str
    samples: int = Field(description="How many samples the registration named.")
    fixed: bool = Field(
        description="Whether the task was registered beforehand, in the tasks folder. Its sentinel is then given the folder's facts, the registration is not compared with them, and nothing is stored."
    )
    expiry: float | None = Field(
        default=None,
        description="How long the registration is kept once no request has named the task, in seconds: what was asked, or less if the sidecar keeps none that long. Absent when `fixed`, as a task registered beforehand is never dropped.",
    )


class CompleteRegistrationParams(SampleFacts, EvalCall):
    """Parameters of `complete_registration`: one sample of a registered task, sent before the sample's first epoch."""

    task: str = Field(description="The task's name, as `register_task` gave it.")


class CompleteRegistrationResult(BaseModel):
    """That the sample's facts were stored, or that the task was registered beforehand and nothing was stored."""

    eval_id: str
    task: str
    sample_id: str | int
    fixed: bool = Field(
        description="Whether the task was registered beforehand, in the tasks folder. Its sentinel is then given the folder's facts, the sample is not compared with them, and nothing is stored."
    )


class RequestResultParams(EvalCall):
    """Parameters of `request_result`: one model request, by the eval's id for it."""

    id: str = Field(description="The request's `x-irid`.")
    wait: float = Field(
        default=0,
        ge=0,
        description=f"How long to wait, in seconds, while the request is still being judged, at most {MAX_RESULT_WAIT:g}. The answer comes as soon as the state is no longer `judging`. An eval asks again while it still is.",
    )


class ReviewRecord(BaseModel):
    """What became of one question a sentinel asked a person while the request or its reply was judged."""

    id: str = Field(description="The question's id, as `pending_questions` gave it.")
    try_number: int = Field(
        description="Which try of the request the question was asked at, from 1, in the order the sidecar was sent them: the try whose request or reply was being judged. A client that tries a request again sends it under the same `x-irid`."
    )
    stage: Literal["tool_call", "tool_result"] = Field(
        description="Where the step was judged."
    )
    call: str = Field(description="The id of the call the question was about.")
    asked: float = Field(description="When it was asked, in seconds since the epoch.")
    decision: Literal["approve", "reject", "terminate"] | None = Field(
        default=None,
        description="What the eval's answer picked, when the sidecar took one. The answer's `mac` shows that the eval sent it, not that a person chose it.",
    )
    reason: str | None = Field(
        default=None, description="The reason the eval's answer gave, if any."
    )
    answered: float | None = Field(
        default=None, description="When the answer was taken."
    )
    resolution: Literal["answered", "timed_out", "cancelled"] | None = Field(
        default=None,
        description="How the wait ended. `answered`: on the eval's answer, which the sentinel was given. `timed_out`: the question expired, where the operator set a review timeout, and the sentinel was given the cautious answer. `cancelled`: the wait was stopped and the step came to no decision, so an answer taken just before took no effect. Left out while the step still waits.",
    )
    cause: (
        Literal[
            "replaced_attempt",
            "sample_ended",
            "run_ended",
            "cancelled",
            "stopped",
        ]
        | None
    ) = Field(
        default=None,
        description="Why the wait was cancelled: `replaced_attempt` when a later attempt at the sample began, `sample_ended` when the eval ended the sample, `run_ended` when a sentinel ended the run at another request, `cancelled` when the eval cancelled the request with `cancel_request`, and `stopped` otherwise, as when the sidecar shuts down.",
    )
    resolved: float | None = Field(default=None, description="When the wait ended.")


class ReplyRecord(BaseModel):
    """A provider's reply, as the proxy passes it on, for the eval to read with the code that reads that provider's replies from the network."""

    status: int = Field(description="The provider's HTTP status.")
    headers: dict[str, str] = Field(
        description="The provider's headers, by lowercase name, without the headers of one hop, such as `content-length`. Empty but for `content-type` where the proxy hands the sidecar no reply headers."
    )
    body: str = Field(
        description="The body, as base64: a JSON reply whole, or a streamed reply's events from first to last. A sentinel's edits are in it. Empty for a provider's error too large to keep."
    )
    format: Literal["anthropic-messages", "openai-chat", "openai-responses"] = Field(
        description="The API the request was written in, and so the reply."
    )
    stream: bool = Field(description="Whether the body is a stream of events.")


class FailureRecord(BaseModel):
    """The provider's last failed exchange, for a request a person was asked about. The eval acts on this, and not on the error its own try got, which may be the sidecar's `pending review`."""

    try_number: int = Field(
        description="Which try of the request was sent on to the provider and failed, from 1. It tells a failure the eval has already acted on from a new one."
    )
    reply: ReplyRecord | None = Field(
        default=None,
        description="The provider's error. Left out for a reply that was cut off, which the eval handles as a connection that failed.",
    )


class RequestResultResult(BaseModel):
    """What happened to the request. With `dropped` or `unknown` records, nothing else is said."""

    id: str
    records: Literal["held", "dropped", "unknown"] = Field(
        description="`held` if the sidecar has the request's records, `dropped` if it had them and no longer keeps them, `unknown` if it never kept the request: one it never saw, one without an `x-inspect-context` it could read and check, or one whose `x-inspect-context` named another eval."
    )
    state: (
        Literal[
            "judging",
            "awaiting_review",
            "awaiting_request",
            "awaiting_reply",
            "decided",
            "not_judged",
        ]
        | None
    ) = Field(
        default=None,
        description="Where the request stands. For a request no person was asked about, that is where its latest try stands. For one a person was asked about, it is where that one model call stands, whatever tries of it were sent since. `awaiting_review`: a question is open, and the eval answers it. `awaiting_request`: the request may be sent on to the provider, so the eval sends the same request again under the same `x-irid`.",
    )
    reason: NotJudged | None = Field(
        default=None,
        description="With `not_judged`, why the request was not judged. With `awaiting_request`, `provider_error` or `cut_off` where the request is there because the provider's last exchange failed, which `failure` describes. Left out where the request only waits to be sent.",
    )
    outcome: Outcome | None = Field(
        default=None, description="What the request came to, once decided."
    )
    message: str | None = Field(
        default=None, description="What the model is told, for a rejection."
    )
    recorded: list[dict[str, Any]] | None = Field(
        default=None,
        description="What each monitor and protocol recorded, every attempt's in order. Left out where the sidecar doesn't send results.",
    )
    judgment: Judgment | None = Field(
        default=None,
        description="What was decided of the request, once decided, which the eval acts on and logs. Trimmed where the sidecar doesn't send results.",
    )
    reviews: list[ReviewRecord] | None = Field(
        default=None,
        description="What became of each question a sentinel asked a person, in order, including one whose step was stopped before it came to a decision. A judged step names its questions in its `reviews`. Left out where nobody was asked.",
    )
    failure: FailureRecord | None = Field(
        default=None,
        description="The provider's last failed exchange, with `awaiting_request` and a `reason`.",
    )
    judging_time: float | None = Field(
        default=None,
        ge=0,
        description="How long the sentinel spent judging the latest attempt's request and reply so far, in seconds. The client waited on its model call meanwhile, so an eval counts this time as waiting, not as the agent's working time. For a call a person was asked about, it is added up over the call's tries, each counted only while it was open. 0 where nothing was judged. Sent with `judging_usage` on every answer about records the sidecar holds, whether or not the judgment is trimmed.",
    )
    judging_usage: list[JudgingUsage] | None = Field(
        default=None,
        description="What the provider billed for each model the sentinel called while judging every attempt at the request, retried attempts included, since each was billed. One entry per role and model that was billed, in the order first billed: a reply from inspect_ai's cache and a call that failed weren't billed. Empty where nothing was billed. Sent with `judging_time` on every answer about records the sidecar holds, whether or not the judgment is trimmed.",
    )


class SampleResultParams(EvalCall):
    """Parameters of `sample_result`: one sample in one epoch, before it is scored."""

    sample_uuid: str = Field(
        description="The sample's uuid, as `x-inspect-context` gives it: one sample in one epoch, the same across the sample's retries."
    )


class SampleResultResult(BaseModel):
    """What the sample came to, and the sentinel's store."""

    sample_uuid: str
    outcome: Literal["proceed", "terminate"] = Field(
        description="`terminate` if a sentinel ended the sample's latest attempt."
    )
    store: dict[str, Any] = Field(
        description="The sentinel's store of the sample's latest attempt, by the keys the sentinel wrote, to copy into the sample's store. Each attempt starts with an empty store, as the sample store does."
    )


class EndSampleParams(EvalCall):
    """Parameters of `end_sample`: one sample in one epoch, once it has been scored."""

    sample_uuid: str = Field(
        description="The sample's uuid, as `x-inspect-context` gives it."
    )


class EndSampleResult(BaseModel):
    """Whether the sample's state was dropped."""

    sample_uuid: str
    state: Literal["ended", "unknown"] = Field(
        description="`ended`, or `unknown` for a sample the sidecar never saw."
    )


class PendingQuestionsParams(EvalCall):
    """Parameters of `pending_questions`: the eval's questions for a person that nobody has answered yet."""

    sample_uuid: str | None = Field(
        default=None,
        description="Only this sample's questions, if given, by its uuid as `x-inspect-context` gives it. Inspect asks for the sample whose model call is waiting.",
    )
    wait: float = Field(
        default=0,
        ge=0,
        description=f"How long to wait, in seconds, while there are none, at most {MAX_QUESTIONS_WAIT:g}. The answer comes as soon as there is one.",
    )


class Escalation(BaseModel):
    """An escalation the asking protocol was handed."""

    name: str = Field(description="The escalating monitor's or protocol's name.")
    explanation: str | None = Field(
        default=None,
        description="Why it escalated, if it said.",
    )


class QuestionAsked(BaseModel):
    """A question a sentinel asked a person, about a tool call in one of the eval's model requests, which the step waits on."""

    id: str = Field(description="The question's id, which `answer_question` names.")
    sample_uuid: str = Field(description="The sample whose run asked.")
    request_id: str = Field(
        description="The eval's id for the model request the call was found in, its `x-irid`. That request waits on the answer."
    )
    stage: Literal["tool_call", "tool_result"] = Field(
        description="`tool_call` before the call runs, `tool_result` once it has run and its result is about to reach the model."
    )
    call: dict[str, Any] = Field(
        description="The call, as Inspect writes a `ToolCall`, with its arguments. Show it to the person; it is not to be put into the agent's conversation."
    )
    message: str = Field(description="The assistant's text alongside the call.")
    view: dict[str, Any] = Field(
        description="How the call is shown, as Inspect writes a `ToolCallView`."
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="The call's result, as Inspect writes a `ChatMessageTool`, at `tool_result`.",
    )
    escalations: list[Escalation] = Field(
        description="The escalations the asking protocol was handed, in order: who escalated, and why."
    )
    choices: list[str] = Field(
        description=f"What the person may pick. An answer is one of these, and one of {', '.join(ANSWERABLE)}."
    )
    asked: float = Field(
        description="When the question was asked, in seconds since the epoch, for showing how long it has waited."
    )
    closes: float | None = Field(
        default=None,
        description="When the question expires, in seconds since the epoch, where the operator set a review timeout. The step is then answered with caution, `reject` if offered and otherwise `terminate`, and the judging goes on. Left out where the question has no expiry.",
    )


class PendingQuestionsResult(BaseModel):
    """The eval's unanswered questions, oldest first."""

    questions: list[QuestionAsked]


class AnswerQuestionParams(EvalCall):
    """Parameters of `answer_question`: what the person picked.

    For now the answer comes on the eval's token, and the sidecar takes it only for a question from one of the eval's own runs. Reviewers are to answer on credentials of their own, which the sidecar checks: until then, a client that holds the eval's token can answer that eval's questions.
    """

    id: str = Field(description="The question's id.")
    decision: Literal["approve", "reject", "terminate"] = Field(
        description="What the person picked, one of the question's `choices`."
    )
    reason: str | None = Field(
        default=None,
        description="The person's reason, if they gave one. At `reject` it is what the model is told.",
    )
    mac: str = Field(
        description='An HMAC-SHA256, in hex, made with the eval\'s token over `json.dumps([str(v) for v in [method, eval_id, timestamp, id, decision, reason or ""]], separators=(",", ":"))`. Unlike other calls\' `mac`, it covers the answer, so one copied from an answer can\'t send another.'
    )


class AnswerQuestionResult(BaseModel):
    """The answer the step was given. Sending the same answer again gives the same result."""

    id: str
    decision: Literal["approve", "reject", "terminate"]


class CancelRequestParams(EvalCall):
    """Parameters of `cancel_request`: a model request a sentinel asked a person about, which the eval no longer waits for.

    The request is closed wherever it stands before its end, with nothing decided: its judging is stopped and its questions close, no try of it is sent on to the provider, and the reply to a try already with the provider is not judged, kept or passed on. It then reads as `not_judged`, with the reason `cancelled`, and nothing opens it again.
    """

    id: str = Field(description="The request's `x-irid`.")


class CancelRequestResult(BaseModel):
    """Whether this call closed the request."""

    id: str
    cancelled: bool = Field(
        description="Whether this call was what closed the request. False for a request that had already ended, one no person was asked about, and one the sidecar doesn't know. A second `cancel_request` for a request is answered false too, so false doesn't say the request is still open: `request_result` says where it stands."
    )


_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "register_task": (RegisterTaskParams, RegisterTaskResult),
    "complete_registration": (
        CompleteRegistrationParams,
        CompleteRegistrationResult,
    ),
    "request_result": (RequestResultParams, RequestResultResult),
    "sample_result": (SampleResultParams, SampleResultResult),
    "end_sample": (EndSampleParams, EndSampleResult),
    "pending_questions": (PendingQuestionsParams, PendingQuestionsResult),
    "answer_question": (AnswerQuestionParams, AnswerQuestionResult),
    "cancel_request": (CancelRequestParams, CancelRequestResult),
}

_RESULTS = frozenset({"sample_result"})

# asking a person through an eval, which a sidecar that sends an eval no
# results doesn't do: it doesn't trust that eval to decide for itself
_REVIEW = frozenset({"pending_questions", "answer_question", "cancel_request"})

_ERRORS: dict[int, tuple[str, str]] = {
    REGISTRATION_REFUSED: (
        "registration_refused",
        "The sidecar takes no tasks registered by an eval, and this task was not registered beforehand; or, for `complete_registration`, the sidecar takes only the samples a task's registration names, and this sample isn't one.",
    ),
    REGISTRATIONS_FULL: (
        "registrations_full",
        "The sidecar keeps as many task registrations as it takes, and takes another once a kept one expires; or, for `complete_registration`, the task's registration holds as much as the sidecar takes for a task. A failure, not a refusal: the deployment has not said that the task may not run.",
    ),
    UNKNOWN_SAMPLE: (
        "unknown_sample",
        "The sidecar holds nothing for the sample, so an eval never mistakes a forgotten sample for a clean one.",
    ),
    UNKNOWN_TASK: (
        "unknown_task",
        "The sidecar holds no registration of the task by the eval: it was never registered, or is no longer kept. The eval registers the task again, then completes the sample's registration. Answered to a `register_task` with a `mac` when the sidecar holds no token for the eval, as after a restart or once the token expired: the eval registers again with its token.",
    ),
    UNKNOWN_QUESTION: (
        "unknown_question",
        "The sidecar holds no such question for the eval. It was never asked, or it was another eval's. Or it closed unanswered: nobody answered in time, and the step was answered with caution, or its model call was closed first, as when the eval cancelled the call or ended the sample. Or it was answered, and the receipt of its answer is no longer kept, as once the eval has ended the sample.",
    ),
    QUESTION_ANSWERED: (
        "question_answered",
        "The question was already answered, with another choice or reason. The first answer stands.",
    ),
    ANSWER_REFUSED: (
        "answer_refused",
        "The answer picks a choice the question doesn't offer.",
    ),
    UNAUTHORIZED: (
        "unauthorized",
        "The call's `mac` isn't made with the token the sidecar keeps for the eval, or its `timestamp` is further from the sidecar's clock than the window allows; or a `register_task` sent another token than the one the sidecar keeps for the eval.",
    ),
}


def eval_routes(
    handler: Handler,
    *,
    max_body_bytes: int,
    token: str | None = None,
    results: bool = True,
) -> list[Route]:
    """The routes of the endpoint for evals: the probe at `/sentinel/info`, the methods at `/sentinel/v1`, and a held reply at `/sentinel/v1/held-reply`.

    An eval registers its tasks and completes each sample's registration before the sample runs, asks what became of each request and of a sample, and ends a sample once it is scored. The eval makes a token and sends it in a registration to a sidecar that holds none for the eval. Every other call proves the caller holds that token with a `timestamp` and a `mac`, so only the eval reads what was decided of its requests and samples. Every call is a JSON-RPC 2.0 request. No method clears a tally or changes the sentinel, and one allows a call: `answer_question`, which answers a question the sentinel itself asked a person, for now on the eval's token, and only for the eval's own runs (`design/sidecar-plan.md`, step 13). A call larger than `max_body_bytes` is not read, and is answered with an HTTP 413.

    `/sentinel/v1/held-reply` is not a method. Once a model call that waited for a person is decided with the turn going on, the eval POSTs the call's own request there, under the same `x-irid` and with an `x-inspect-context` signed as any model request's is, and is answered with the reply the sidecar held: the provider's status, headers and body, a stream's events from first to last. The eval's code for that provider reads it as the provider's own. Only the headers that prove the call are read, never the body. Any other request is refused with a 400 and a JSON `error`, and the eval reads where the call stands with `request_result`. Getting the reply changes nothing, so it can be got again.

    Args:
        handler: The handler the proxy's requests and replies are put to.
        max_body_bytes: The largest call that is read.
        token: A credential every call to the methods must present as a bearer token, when the endpoint has a port of its own. None where it shares the proxy's port and access control.
        results: Whether the results are sent: `sample_result`, and what each monitor and protocol recorded in `request_result`. A record holds each monitor's explanation, which tells an agent what the monitor looks for, so where an eval can't be trusted they are turned off. `request_result` is still served without them, since it is how an eval learns that a turn was refused or edited. Without results, `pending_questions`, `answer_question`, `cancel_request` and `/sentinel/v1/held-reply` aren't served either: an eval that isn't trusted with a monitor's explanation isn't trusted to answer for a person. Build the handler with `eval_results` off as well, so that it asks no question nobody can answer.
    """

    async def info(http: HttpRequest) -> Response:
        return JSONResponse(
            {
                "service": "inspect_sentinel",
                "versions": VERSIONS,
                "release": handler.release,
                # whether an eval is to ask for its questions for a person:
                # an eval asks a sidecar that doesn't say so nothing
                "review": handler.eval_review and results,
            }
        )

    def unauthorized(http: HttpRequest) -> Response | None:
        # the endpoint's own credential, where it has a port of its own
        if token is None:
            return None
        presented = http.headers.get("authorization", "")
        if hmac.compare_digest(presented.encode(), f"Bearer {token}".encode()):
            return None
        return JSONResponse(
            {"error": "A bearer token is required."},
            status_code=401,
            headers={"www-authenticate": "Bearer"},
        )

    async def call(http: HttpRequest) -> Response:
        refused = unauthorized(http)
        if refused is not None:
            return refused
        body = await read_within(http, max_body_bytes)
        if body is None:
            return JSONResponse(
                {"error": f"A call is at most {max_body_bytes} bytes."},
                status_code=413,
            )
        id: int | str | None = None
        try:
            asked = read_call(body)
            id = asked.id
            if not results and asked.method in _RESULTS:
                raise RpcError(
                    METHOD_NOT_FOUND,
                    f"This sidecar doesn't send results, so there is no method {asked.method!r}.",
                )
            if not results and asked.method in _REVIEW:
                raise RpcError(
                    METHOD_NOT_FOUND,
                    f"This sidecar doesn't send results, so it asks no person through an eval, and there is no method {asked.method!r}.",
                )
            result = await _answer(handler, asked.method, asked.params, results)
        except RpcError as ex:
            return failed(id, ex)
        return answered(id, result)

    async def held_reply(http: HttpRequest) -> Response:
        # Every refusal is a status that a provider's SDK doesn't send again
        # by itself, so an eval's request for a reply that isn't there fails
        # once, and the eval asks `request_result` what became of the call.
        refused = unauthorized(http)
        if refused is not None:
            return refused
        if not results:
            return JSONResponse(
                {
                    "error": "This sidecar doesn't send results, so it asks no person through an eval, and holds no reply for one."
                },
                status_code=404,
            )
        # the body is the model call's request, which nothing here reads
        if not await discard_within(http, max_body_bytes):
            return JSONResponse(
                {"error": f"A request is at most {max_body_bytes} bytes."},
                status_code=413,
            )
        held = await handler.held_reply(
            {name.lower(): value for name, value in http.headers.items()}
        )
        if isinstance(held, Refuse):
            return JSONResponse({"error": held.message}, status_code=held.status)
        return _held_reply_response(held)

    return [
        Route(INFO_PATH, info, methods=["GET"]),
        Route(PATH, call, methods=["POST"]),
        Route(HELD_REPLY_PATH, held_reply, methods=["POST"]),
    ]


async def _answer(
    handler: Handler, method: str, params: dict[str, Any], results: bool = True
) -> BaseModel:
    if method not in _METHODS:
        raise RpcError(METHOD_NOT_FOUND, f"There is no method {method!r}.")
    try:
        given = _METHODS[method][0].model_validate(params)
    except ValidationError as ex:
        raise invalid_params(ex) from None
    try:
        return await _answered(handler, method, given, results)
    except UnauthorizedError as ex:
        raise RpcError(UNAUTHORIZED, str(ex)) from None


async def _answered(
    handler: Handler, method: str, given: BaseModel, results: bool
) -> BaseModel:
    if isinstance(given, RegisterTaskParams):
        # the token or the proof first: a caller that has neither registers
        # nothing, and reads nothing of what the eval registered before
        if given.token is not None:
            await handler.eval_token(given.eval_id, given.token)
        else:
            assert given.timestamp is not None and given.mac is not None
            if not await handler.holds_eval(
                given.eval_id, method, given.timestamp, given.mac
            ):
                raise RpcError(
                    UNKNOWN_TASK,
                    f"The sidecar holds no token for the eval {given.eval_id!r}, as after a restart or once it expired, so the eval registers the task again with its token.",
                )
        registration = TaskRegistration.model_validate(
            given.model_dump(include=set(TaskRegistration.model_fields))
        )
        try:
            # The token is kept with the registration, in one step. So a
            # registration that fails leaves no token behind, and one that
            # finds another token kept meanwhile leaves no task.
            granted = await handler.register_task(
                given.eval_id, registration, given.expiry, given.token
            )
        except RegistrationsFullError as ex:
            raise RpcError(REGISTRATIONS_FULL, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        return RegisterTaskResult(
            eval_id=given.eval_id,
            task=given.task,
            samples=len(given.sample_ids),
            fixed=granted is None,
            expiry=granted,
        )
    if isinstance(given, AnswerQuestionParams):
        return await _answer_question(handler, given)
    assert isinstance(given, EvalCall)
    # a sidecar that keeps no token for the eval holds nothing of it
    holds = await handler.holds_eval(given.eval_id, method, given.timestamp, given.mac)
    if isinstance(given, CompleteRegistrationParams):
        if not holds:
            raise RpcError(
                UNKNOWN_TASK,
                f"The sidecar holds nothing of the eval {given.eval_id!r}, which registers the task again.",
            )
        sample = SampleFacts.model_validate(
            given.model_dump(include=set(SampleFacts.model_fields))
        )
        try:
            fixed = await handler.complete_registration(
                given.eval_id, given.task, sample
            )
        except UnknownTaskError as ex:
            raise RpcError(UNKNOWN_TASK, str(ex)) from None
        except RegistrationsFullError as ex:
            raise RpcError(REGISTRATIONS_FULL, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        return CompleteRegistrationResult(
            eval_id=given.eval_id,
            task=given.task,
            sample_id=given.sample_id,
            fixed=fixed,
        )
    if isinstance(given, RequestResultParams):
        found = (
            await handler.request_result(given.id, given.eval_id, given.wait)
            if holds
            else "unknown"
        )
        if not isinstance(found, RequestResult):
            return RequestResultResult(id=given.id, records=found)
        latest = found.latest
        return RequestResultResult(
            id=given.id,
            records="held",
            state=latest.state,
            reason=latest.reason,
            outcome=latest.outcome,
            message=latest.message,
            recorded=(
                [record_json(record) for record in found.records] if results else None
            ),
            judgment=(
                judgment(
                    latest.judged,
                    latest.outcome,
                    trimmed=not (results and handler.eval_results),
                )
                if latest.state == "decided"
                and latest.judged is not None
                and latest.outcome is not None
                else None
            ),
            judging_time=latest.judging_time,
            judging_usage=judging_usage(found.judging_usage),
            reviews=[
                ReviewRecord(
                    id=review.id,
                    try_number=number,
                    stage="tool_result"
                    if review.stage == "tool_result"
                    else "tool_call",
                    call=review.call,
                    asked=review.asked,
                    decision=_DECISION.validate_python(review.decision),
                    reason=review.reason,
                    answered=review.answered,
                    resolution=review.resolution,
                    cause=review.cause,
                    resolved=review.resolved,
                )
                for number, review in found.reviews
            ]
            or None,
            failure=(
                FailureRecord(
                    try_number=found.failure.try_number,
                    reply=(
                        _reply_record(found.failure.reply)
                        if found.failure.reply is not None
                        else None
                    ),
                )
                if found.failure is not None
                else None
            ),
        )
    if isinstance(given, CancelRequestParams):
        return CancelRequestResult(
            id=given.id,
            cancelled=(
                await handler.cancel_request(given.id, given.eval_id)
                if holds
                else False
            ),
        )
    if isinstance(given, SampleResultParams):
        run = (
            await handler.sample_result(given.sample_uuid, given.eval_id)
            if holds
            else None
        )
        if run is None:
            raise RpcError(
                UNKNOWN_SAMPLE,
                f"The sidecar holds nothing for the sample {given.sample_uuid!r}.",
            )
        return SampleResultResult(
            sample_uuid=given.sample_uuid, outcome=run.outcome, store=run.store
        )
    if isinstance(given, EndSampleParams):
        return EndSampleResult(
            sample_uuid=given.sample_uuid,
            state=(
                await handler.end_sample(given.sample_uuid, given.eval_id)
                if holds
                else "unknown"
            ),
        )
    if isinstance(given, PendingQuestionsParams):
        # a sidecar that holds nothing of the eval holds no question of it
        found = (
            await handler.reviews.pending(given.eval_id, given.sample_uuid, given.wait)
            if holds
            else []
        )
        return PendingQuestionsResult(
            questions=[_question(question) for question in found]
        )
    raise TypeError(f"No method takes {type(given).__name__}.")


async def _answer_question(
    handler: Handler, given: AnswerQuestionParams
) -> AnswerQuestionResult:
    signed = answer_values(
        given.eval_id, given.timestamp, given.id, given.decision, given.reason
    )
    if not await handler.holds_eval(
        given.eval_id, "answer_question", given.timestamp, given.mac, signed
    ):
        raise RpcError(
            UNKNOWN_QUESTION,
            f"The sidecar holds nothing of the eval {given.eval_id!r}, so no question of it.",
        )
    try:
        handler.answer_question(given.eval_id, given.id, given.decision, given.reason)
    except UnknownQuestionError as ex:
        raise RpcError(UNKNOWN_QUESTION, str(ex)) from None
    except AnsweredError as ex:
        raise RpcError(QUESTION_ANSWERED, str(ex)) from None
    except ChoiceError as ex:
        raise RpcError(ANSWER_REFUSED, str(ex)) from None
    return AnswerQuestionResult(id=given.id, decision=given.decision)


def _held_reply_response(held: HeldReply) -> Response:
    # The reply as the provider sent it and the proxy passed it on, for the
    # provider's own client code to read. Its body is sent whole, a stream's
    # events from first to last, with a length of its own: the headers of
    # one hop were left out when it was held.
    headers = dict(held.headers)
    if held.stream and "event-stream" not in headers.get("content-type", "").lower():
        headers["content-type"] = "text/event-stream"
    return Response(content=held.body, status_code=held.status, headers=headers)


def _reply_record(reply: HeldReply) -> ReplyRecord:
    return ReplyRecord(
        status=reply.status,
        headers=dict(reply.headers),
        body=base64.b64encode(reply.body).decode(),
        format=reply.format,
        stream=reply.stream,
    )


def _question(question: Question) -> QuestionAsked:
    return QuestionAsked(
        id=question.id,
        sample_uuid=question.sample_uuid,
        request_id=question.request_id,
        stage="tool_result" if question.stage == "tool_result" else "tool_call",
        call=_CALL.dump_python(question.call, mode="json", exclude_none=True),
        message=question.message,
        view=question.view.model_dump(mode="json", exclude_none=True),
        result=(
            question.result.model_dump(mode="json", exclude_none=True)
            if question.result is not None
            else None
        ),
        escalations=[
            Escalation(name=escalation.name, explanation=escalation.explanation)
            for escalation in question.escalations
        ],
        choices=list(question.choices),
        asked=question.asked,
        closes=question.closes,
    )


_CALL = TypeAdapter(ToolCall)
_DECISION = TypeAdapter(Literal["approve", "reject", "terminate"] | None)


def schema() -> dict[str, Any]:
    """The JSON Schema of each method's parameters and result, and the errors the methods may answer with.

    A call to method `m` has `params` that fit the definition named in `methods[m].params`, and is answered with a `result` that fits the definition named in `methods[m].result`, or with an error whose code is a key of `errors` or one of JSON-RPC's own.
    """
    models = {model: None for pair in _METHODS.values() for model in pair}
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for evals",
        description=f"JSON-RPC 2.0 methods, POSTed to {PATH}, with a probe at GET {INFO_PATH}. A model call's request POSTed to {HELD_REPLY_PATH} is not a method: it is answered with the reply held for that call, as the provider sent it. `methods` names the definitions that a method's `params` and `result` fit. `errors` lists the codes of the method errors of the endpoint's own.",
    )
    return {
        "$schema": GenerateJsonSchema.schema_dialect,
        **definitions,
        "methods": {
            name: {
                "params": {"$ref": f"#/$defs/{params.__name__}"},
                "result": {"$ref": f"#/$defs/{result.__name__}"},
            }
            for name, (params, result) in _METHODS.items()
        },
        "errors": {
            str(code): {"name": name, "description": description}
            for code, (name, description) in _ERRORS.items()
        },
    }


SCHEMA_FILE = Path(__file__).with_name("eval_endpoint.schema.json")
"""Where `schema()` is kept, for an eval's client to be written and tested against without running this package."""


if __name__ == "__main__":
    # rewrites the schema file; run after changing a model above
    SCHEMA_FILE.write_text(json.dumps(schema(), indent=2) + "\n")
