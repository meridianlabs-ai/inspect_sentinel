from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ._errors import UnauthorizedError
from ._handler import MAX_RESULT_WAIT, Handler, RequestResult
from ._http import read_within
from ._inspect_client import Judgment, judgment, record_json
from ._json_rpc import (
    METHOD_NOT_FOUND,
    REGISTRATION_REFUSED,
    REGISTRATIONS_FULL,
    UNAUTHORIZED,
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
from ._sign import MIN_TOKEN_LENGTH
from ._state import NotJudged, Outcome

PATH = "/sentinel/v1"
"""Where the methods are served. The `v1` is the version of the methods."""

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
        description=f"How long to wait, in seconds, while the latest attempt is still being judged, at most {MAX_RESULT_WAIT:g}. The answer comes as soon as the attempt is decided or not judged. An eval asks again while the state is still `judging`.",
    )


class RequestResultResult(BaseModel):
    """What happened to the request. With `dropped` or `unknown` records, nothing else is said."""

    id: str
    records: Literal["held", "dropped", "unknown"] = Field(
        description="`held` if the sidecar has the request's records, `dropped` if it had them and no longer keeps them, `unknown` if it never kept the request: one it never saw, one without an `x-inspect-context` it could read and check, or one whose `x-inspect-context` named another eval."
    )
    state: Literal["judging", "awaiting_reply", "decided", "not_judged"] | None = Field(
        default=None, description="Where the latest attempt stands."
    )
    reason: NotJudged | None = Field(
        default=None, description="Why the latest attempt was not judged."
    )
    outcome: Outcome | None = Field(
        default=None, description="What the latest attempt came to, once decided."
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
        description="What was decided during the latest attempt, once decided, which the eval acts on and logs. Trimmed where the sidecar doesn't send results.",
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


_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "register_task": (RegisterTaskParams, RegisterTaskResult),
    "complete_registration": (
        CompleteRegistrationParams,
        CompleteRegistrationResult,
    ),
    "request_result": (RequestResultParams, RequestResultResult),
    "sample_result": (SampleResultParams, SampleResultResult),
    "end_sample": (EndSampleParams, EndSampleResult),
}

_RESULTS = frozenset({"sample_result"})

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
    """The routes of the endpoint for evals: the probe at `/sentinel/info`, and the methods at `/sentinel/v1`.

    An eval registers its tasks and completes each sample's registration before the sample runs, asks what became of each request and of a sample, and ends a sample once it is scored. The eval makes a token and sends it in a registration to a sidecar that holds none for the eval. Every other call proves the caller holds that token with a `timestamp` and a `mac`, so only the eval reads what was decided of its requests and samples. Every call is a JSON-RPC 2.0 request. No method allows a call, clears a tally or changes the sentinel. A call larger than `max_body_bytes` is not read, and is answered with an HTTP 413.

    Args:
        handler: The handler the proxy's requests and replies are put to.
        max_body_bytes: The largest call that is read.
        token: A credential every call to the methods must present as a bearer token, when the endpoint has a port of its own. None where it shares the proxy's port and access control.
        results: Whether the results are sent: `sample_result`, and what each monitor and protocol recorded in `request_result`. A record holds each monitor's explanation, which tells an agent what the monitor looks for, so where an eval can't be trusted they are turned off. `request_result` is still served without them, since it is how an eval learns that a turn was refused or edited.
    """

    async def info(http: HttpRequest) -> Response:
        return JSONResponse(
            {
                "service": "inspect_sentinel",
                "versions": VERSIONS,
                "release": handler.release,
            }
        )

    async def call(http: HttpRequest) -> Response:
        if token is not None:
            presented = http.headers.get("authorization", "")
            if not hmac.compare_digest(presented.encode(), f"Bearer {token}".encode()):
                return JSONResponse(
                    {"error": "A bearer token is required."},
                    status_code=401,
                    headers={"www-authenticate": "Bearer"},
                )
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
            result = await _answer(handler, asked.method, asked.params, results)
        except RpcError as ex:
            return failed(id, ex)
        return answered(id, result)

    return [
        Route(INFO_PATH, info, methods=["GET"]),
        Route(PATH, call, methods=["POST"]),
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
        # nothing
        if given.token is not None:
            held = await handler.eval_token(given.eval_id, given.token) is not None
        else:
            assert given.timestamp is not None and given.mac is not None
            held = await handler.holds_eval(
                given.eval_id, method, given.timestamp, given.mac
            )
            if not held:
                raise RpcError(
                    UNKNOWN_TASK,
                    f"The sidecar holds no token for the eval {given.eval_id!r}, as after a restart or once it expired, so the eval registers the task again with its token.",
                )
        registration = TaskRegistration.model_validate(
            given.model_dump(include=set(TaskRegistration.model_fields))
        )
        try:
            # before the registration, so one whose token can't be kept
            # isn't kept either
            if not held:
                await handler.eval_token_room()
            granted = await handler.register_task(
                given.eval_id, registration, given.expiry
            )
        except RegistrationsFullError as ex:
            raise RpcError(REGISTRATIONS_FULL, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        # kept only now, so a registration that failed leaves none behind
        if not held:
            assert given.token is not None
            await handler.keep_eval_token(given.eval_id, given.token)
        return RegisterTaskResult(
            eval_id=given.eval_id,
            task=given.task,
            samples=len(given.sample_ids),
            fixed=granted is None,
            expiry=granted,
        )
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
    raise TypeError(f"No method takes {type(given).__name__}.")


def schema() -> dict[str, Any]:
    """The JSON Schema of each method's parameters and result, and the errors the methods may answer with.

    A call to method `m` has `params` that fit the definition named in `methods[m].params`, and is answered with a `result` that fits the definition named in `methods[m].result`, or with an error whose code is a key of `errors` or one of JSON-RPC's own.
    """
    models = {model: None for pair in _METHODS.values() for model in pair}
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for evals",
        description=f"JSON-RPC 2.0 methods, POSTed to {PATH}, with a probe at GET {INFO_PATH}. `methods` names the definitions that a method's `params` and `result` fit. `errors` lists the codes of the method errors of the endpoint's own.",
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
