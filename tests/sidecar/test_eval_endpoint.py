import argparse
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import anyio
import httpx
import jsonschema
import pytest
from inspect_ai.util import StoreModel
from starlette.applications import Starlette

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Request,
    UnreadableError,
    _handler,
    _state,
)
from inspect_sentinel.sidecar._errors import TimeLimitError, UnauthorizedError
from inspect_sentinel.sidecar._eval_endpoint import (
    HELD_REPLY_PATH,
    INFO_PATH,
    PATH,
    SCHEMA_FILE,
    TOKEN_VARIABLE,
    eval_routes,
    schema,
)
from inspect_sentinel.sidecar._host import SidecarHost
from inspect_sentinel.sidecar._json_rpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    REGISTRATION_REFUSED,
    REGISTRATIONS_FULL,
    UNAUTHORIZED,
    UNKNOWN_SAMPLE,
    UNKNOWN_TASK,
)
from inspect_sentinel.sidecar._registration import (
    FileTask,
    RegisteredTask,
    RegistrationsFullError,
    Strictness,
    TaskRegistration,
    UnregisteredSampleError,
)
from inspect_sentinel.sidecar._serve import add_arguments, apps
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import (
    TOKEN,
    eval_call,
    holding,
    inspect_headers,
    message,
    reply,
    request,
    weather,
    weather_rule,
)

_SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())

_REGISTRATION: dict[str, Any] = {
    "eval_id": "eval_789",
    "token": TOKEN,
    "task": "philadelphia",
    "task_description": None,
    "metadata": {},
    "expiry": 604800,
    "sample_ids": [1],
    "epochs": 2,
}

_COMPLETION: dict[str, Any] = {
    "eval_id": "eval_789",
    "task": "philadelphia",
    "sample_id": 1,
    "sample_description": None,
    "sample_input": "File today's weather report for Philadelphia.",
    "metadata": {"desk": "weather"},
}


def _client(handler: Handler, **options: Any) -> httpx.AsyncClient:
    app = Starlette(routes=eval_routes(handler, max_body_bytes=1024 * 1024, **options))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
    )


async def _call(
    client: httpx.AsyncClient, method: str, params: dict[str, Any], **headers: str
) -> dict[str, Any]:
    answered = await client.post(
        PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers=headers,
    )
    assert answered.status_code == 200
    body: dict[str, Any] = answered.json()
    if "result" in body:
        assert _fits(_SCHEMA["methods"][method]["result"], body["result"])
    return body


def _fits(part: dict[str, Any], value: Any) -> bool:
    try:
        jsonschema.validate(
            value, {**_SCHEMA, **part}, cls=jsonschema.Draft202012Validator
        )
    except jsonschema.ValidationError:
        return False
    return True


def _asked(id: str, irid: str = "irid-1", **sample: Any) -> Request:
    # a request an Inspect eval sends, with its id for the request
    return replace(request(id), headers=inspect_headers(irid=irid, **sample))


def _token(eval_id: str) -> str:
    # each eval's own, as Inspect makes one for each
    return TOKEN if eval_id == "eval_789" else f"{eval_id}-token-made-for-these-tests"


async def _eval(
    handler: Handler, method: str, eval_id: str = "eval_789"
) -> dict[str, Any]:
    # what a call of `method` for the eval presents, once the eval's first
    # registration has left the sidecar its token
    await holding(handler, eval_id, _token(eval_id))
    return eval_call(method, eval_id, token=_token(eval_id))


async def _registered(client: httpx.AsyncClient, **changes: Any) -> None:
    # registers the task with the eval's token, which later calls prove
    answered = await _call(client, "register_task", _REGISTRATION | changes)
    assert "result" in answered


@pytest.mark.anyio
async def test_the_probe_says_which_versions_are_served() -> None:
    handler = Handler(weather_rule())
    async with handler.serving(), _client(handler) as client:
        answered = await client.get(INFO_PATH)

    assert answered.json() == {
        "service": "inspect_sentinel",
        "versions": [1],
        "release": "whole",
        "review": True,
    }


@pytest.mark.anyio
async def test_the_probe_of_a_handler_that_isnt_serving_says_no_eval_is_asked() -> None:
    # the judging that waits for a person runs while the handler serves, so
    # outside that nobody is asked through an eval
    async with _client(Handler(weather_rule())) as client:
        answered = await client.get(INFO_PATH)

    assert answered.json()["review"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("handler_options", "route_options"),
    [({"eval_results": False}, {}), ({}, {"results": False})],
)
async def test_the_probe_says_when_an_eval_isnt_asked_for_a_person(
    handler_options: dict[str, Any], route_options: dict[str, Any]
) -> None:
    # an eval that isn't sent results isn't asked to answer for a person, so
    # it needn't look for questions
    handler = Handler(weather_rule(), **handler_options)
    async with handler.serving(), _client(handler, **route_options) as client:
        answered = await client.get(INFO_PATH)

    assert answered.json()["review"] is False


@pytest.mark.anyio
async def test_the_probe_says_how_replies_are_released() -> None:
    async with _client(Handler(weather_rule(), release="early")) as client:
        answered = await client.get(INFO_PATH)

    assert answered.json()["release"] == "early"


@pytest.mark.anyio
async def test_a_task_is_registered() -> None:
    async with _client(Handler(weather_rule(), task_expiry=86400)) as client:
        answered = await _call(client, "register_task", _REGISTRATION)

    # the sidecar never sends a token back
    assert answered["result"] == {
        "eval_id": "eval_789",
        "task": "philadelphia",
        "samples": 1,
        "fixed": False,
        # no longer than the sidecar keeps any
        "expiry": 86400,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("runtime_task_registration", [True, False])
async def test_a_task_registered_beforehand_is_already_registered(
    runtime_task_registration: bool,
) -> None:
    task = FileTask.model_validate({"task": "philadelphia", "samples": []})
    handler = Handler(
        weather_rule(),
        tasks={"philadelphia": task},
        runtime_task_registration=runtime_task_registration,
    )

    async with _client(handler) as client:
        answered = await _call(client, "register_task", _REGISTRATION)

    # never dropped, so no expiry
    assert answered["result"] == {
        "eval_id": "eval_789",
        "task": "philadelphia",
        "samples": 1,
        "fixed": True,
    }


@pytest.mark.anyio
async def test_a_sidecar_that_takes_no_registrations_refuses_another_task() -> None:
    async with _client(
        Handler(weather_rule(), runtime_task_registration=False)
    ) as client:
        answered = await _call(client, "register_task", _REGISTRATION)

    assert answered["error"]["code"] == REGISTRATION_REFUSED


@pytest.mark.anyio
async def test_a_registration_past_the_limit_fails_with_its_own_error() -> None:
    # anything that can reach the endpoint can register, so what is kept is
    # bounded, and nothing kept is dropped to make room
    async with _client(Handler(weather_rule(), max_task_registrations=2)) as client:
        kept = [
            await _call(client, "register_task", _REGISTRATION | {"eval_id": id})
            for id in ["eval_1", "eval_2"]
        ]
        refused = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_3"}
        )
        again = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_1"}
        )

    assert all("result" in answered for answered in kept)
    # a failure, which Inspect's `optional` covers, and not the refusal of a
    # sidecar that takes no registrations
    assert refused["error"]["code"] == REGISTRATIONS_FULL
    assert "already keeps the 2 tasks" in refused["error"]["message"]
    # a registration of a task kept already replaces it
    assert "result" in again


@pytest.mark.anyio
async def test_a_sample_past_what_a_registration_holds_fails_with_the_same_error() -> (
    None
):
    handler = Handler(weather_rule(), max_registration_bytes=1_000)
    async with _client(handler) as client:
        await _registered(client)
        answered = await _call(
            client,
            "complete_registration",
            _COMPLETION
            | eval_call("complete_registration")
            | {"sample_input": "x" * 2_000},
        )

    # a failure, not a refusal: the task may run once its registration is
    # made again
    assert answered["error"]["code"] == REGISTRATIONS_FULL
    assert "no more than 1000" in answered["error"]["message"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "params",
    [
        {k: v for k, v in _REGISTRATION.items() if k != "eval_id"},
        _REGISTRATION | {"expiry": 0},
        _REGISTRATION | {"epochs": 0},
    ],
    ids=["no eval id", "no time to keep it", "no epochs"],
)
async def test_a_registration_that_doesnt_fit_is_an_error(
    params: dict[str, Any],
) -> None:
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(client, "register_task", params)

    assert answered["error"]["code"] == INVALID_PARAMS


@pytest.mark.anyio
async def test_a_samples_registration_is_completed() -> None:
    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await _call(
            client,
            "complete_registration",
            _COMPLETION | eval_call("complete_registration"),
        )

    assert answered["result"] == {
        "eval_id": "eval_789",
        "task": "philadelphia",
        "sample_id": 1,
        "fixed": False,
    }


@pytest.mark.anyio
async def test_a_samples_registration_for_a_task_not_registered_is_its_own_error() -> (
    None
):
    handler = Handler(weather_rule())
    async with _client(handler) as client:
        answered = await _call(
            client,
            "complete_registration",
            _COMPLETION | await _eval(handler, "complete_registration"),
        )
        # a sidecar that keeps no token for the eval, as after a restart,
        # holds nothing of it either
        lost = await _call(
            client,
            "complete_registration",
            _COMPLETION | eval_call("complete_registration", "eval_lost"),
        )

    # so an eval whose registration expired knows to register the task again
    assert answered["error"]["code"] == UNKNOWN_TASK
    assert "Register the task" in answered["error"]["message"]
    assert lost["error"]["code"] == UNKNOWN_TASK


@pytest.mark.anyio
async def test_a_samples_registration_for_a_task_registered_beforehand_stores_nothing() -> (
    None
):
    task = FileTask.model_validate({"task": "philadelphia", "samples": []})
    handler = Handler(weather_rule(), tasks={"philadelphia": task})

    async with _client(handler) as client:
        await _registered(client)
        answered = await _call(
            client,
            "complete_registration",
            _COMPLETION | eval_call("complete_registration"),
        )

    assert answered["result"]["fixed"] is True


@pytest.mark.anyio
async def test_a_sidecar_that_takes_only_registered_samples_refuses_another() -> None:
    async with _client(Handler(weather_rule(), strictness="registered_only")) as client:
        await _registered(client)
        answered = await _call(
            client,
            "complete_registration",
            _COMPLETION | eval_call("complete_registration") | {"sample_id": 2},
        )

    assert answered["error"]["code"] == REGISTRATION_REFUSED
    assert "doesn't name the sample 2" in answered["error"]["message"]


@pytest.mark.anyio
async def test_a_samples_registration_that_doesnt_fit_is_an_error() -> None:
    without_input = {k: v for k, v in _COMPLETION.items() if k != "sample_input"}

    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await _call(
            client,
            "complete_registration",
            without_input | eval_call("complete_registration"),
        )

    assert answered["error"]["code"] == INVALID_PARAMS


async def _request_result(
    handler: Handler, irid: str = "irid-1", **params: Any
) -> dict[str, Any]:
    eval = await _eval(handler, "request_result")
    async with _client(handler) as client:
        answered = await _call(client, "request_result", {"id": irid} | eval | params)
    result: dict[str, Any] = answered["result"]
    return result


@pytest.mark.anyio
async def test_a_request_awaiting_its_reply() -> None:
    handler = await holding(Handler(weather_rule()))
    await handler.request(_asked("x"))

    # weather_rule watches only tool calls, so nothing was judged or billed yet
    assert await _request_result(handler) == {
        "id": "irid-1",
        "records": "held",
        "state": "awaiting_reply",
        "recorded": [],
        "judging_time": 0.0,
        "judging_usage": [],
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "city", "outcome", "message"),
    [
        ("reject", "Cairo", "proceed", None),
        ("reject", "Tokyo", "reject", "Tokyo is not allowed."),
        ("terminate", "Tokyo", "terminate", None),
    ],
)
async def test_a_request_whose_reply_was_judged_is_decided(
    action: str, city: str, outcome: str, message: str | None
) -> None:
    handler = await holding(Handler(weather_rule(action, "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message(city)))

    result = await _request_result(handler)

    assert (result["state"], result["outcome"], result.get("message")) == (
        "decided",
        outcome,
        message,
    )
    assert [record["call"] for record in result["recorded"]] == ["toolu_1"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "outcome"), [("reject", "reject"), ("terminate", "terminate")]
)
async def test_a_request_no_eval_names_is_unknown_to_every_eval(
    action: str, outcome: str
) -> None:
    # judged as any client's, and not an eval's to read
    handler = Handler(weather_rule(action, "Tokyo"))
    await handler.request(replace(request("x"), headers={"x-irid": "irid-1"}))
    answer = await handler.reply(reply(_message("Tokyo")))

    result = await _request_result(handler)

    assert isinstance(answer, Refuse)
    assert result == {"id": "irid-1", "records": "unknown"}


def _message(city: str) -> bytes:
    return message(weather(city))


@pytest.mark.anyio
async def test_a_request_the_provider_failed_was_not_judged() -> None:
    handler = await holding(Handler(weather_rule()))
    await handler.request(_asked("x"))
    await handler.reply(reply(b"overloaded", status=529))

    result = await _request_result(handler)

    assert (result["state"], result["reason"]) == ("not_judged", "provider_error")
    assert "judgment" not in result


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "city"),
    [("reject", "Oslo"), ("reject", "Tokyo"), ("terminate", "Tokyo")],
    ids=["proceed", "reject", "terminate"],
)
async def test_request_result_gives_what_was_decided_and_the_reply_carries_none(
    action: str, city: str
) -> None:
    handler = await holding(Handler(weather_rule(action, city)))
    await handler.request(_asked("x"))

    answer = await handler.reply(reply(_message("Tokyo")))

    # the eval is handed what any client is: the reply, or the refusal
    assert isinstance(answer, Pass | Refuse)
    result = await _request_result(handler)
    outcome = "proceed" if isinstance(answer, Pass) else action
    assert result["judgment"]["outcome"] == outcome
    assert result["judgment"]["steps"][-1]["recorded"]


@pytest.mark.anyio
async def test_request_result_gives_the_field_of_a_request_refused() -> None:
    handler = await holding(Handler(weather_rule("terminate", "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    # the run has ended, so the next request is refused before the model
    answer = await handler.request(_asked("y", irid="irid-2"))

    assert isinstance(answer, Refuse)
    result = await _request_result(handler, "irid-2")
    assert result["judgment"] == {"version": 1, "outcome": "terminate", "steps": []}


@pytest.mark.anyio
async def test_request_result_trims_the_field_where_results_are_not_sent() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    eval = await _eval(handler, "request_result")
    async with _client(handler, results=False) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"} | eval)

    field = answered["result"]["judgment"]
    assert field["trimmed"] is True
    assert all("recorded" not in step for step in field["steps"])
    assert set(field["steps"][-1]["decision"]) <= {"action", "message", "modified"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "options",
    [
        {
            "tasks": {
                "philadelphia": FileTask.model_validate(
                    {"task": "philadelphia", "samples": []}
                )
            }
        },
        {"strictness": "known_tasks"},
    ],
    ids=["a sample the tasks folder doesn't hold", "a task the sidecar doesn't hold"],
)
async def test_a_request_the_sidecar_cant_place_was_not_judged(
    options: dict[str, Any],
) -> None:
    handler = await holding(Handler(weather_rule(), **options))

    with pytest.raises(UnregisteredSampleError):
        await handler.request(_asked("x"))

    # how an eval learns that the sample can't run there, and not to retry it
    result = await _request_result(handler)
    assert (result["state"], result["reason"]) == ("not_judged", "unregistered")


@pytest.mark.anyio
async def test_a_request_that_cant_be_read_was_not_judged() -> None:
    handler = await holding(Handler(weather_rule()))
    unreadable = replace(_asked("x"), body=b"not json")

    with pytest.raises(UnreadableError):
        await handler.request(unreadable)

    result = await _request_result(handler)
    assert (result["state"], result["reason"]) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_a_request_whose_inspect_header_cant_be_read_is_refused() -> None:
    handler = Handler(weather_rule())
    headers = {"x-inspect-context": "{not json", "x-irid": "irid-1"}

    # refused, so a proxy that lets through what can't be judged doesn't
    answer = await handler.request(replace(request("x"), headers=headers))

    assert isinstance(answer, Refuse)
    # it names no eval that could read it
    result = await _request_result(handler)
    assert result == {"id": "irid-1", "records": "unknown"}


@pytest.mark.anyio
async def test_a_retried_request_holds_every_attempts_records() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo"), reject_status=503))
    await handler.request(_asked("x1"))
    await handler.reply(reply(_message("Tokyo"), id="x1"))
    # the client retries the request under the same id
    await handler.request(_asked("x2"))
    await handler.reply(reply(_message("Oslo"), id="x2"))

    result = await _request_result(handler)

    assert (result["state"], result["outcome"]) == ("decided", "proceed")
    assert [record["report"]["action"] for record in result["recorded"]] == [
        "reject",
        "proceed",
    ]


@protocol
def asks_models() -> Protocol:
    # rejects Tokyo, after asking its role's model and a model it names
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision:
        await context.host.generate("Is this call suspicious?")
        await context.host.generate("Is it?", model="mockllm/named")
        if "Tokyo" in json.dumps(step.call.arguments):
            return Decision.reject("Tokyo is not allowed.")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("results", [True, False], ids=["sent", "trimmed"])
async def test_request_result_gives_the_judging_usage_of_every_attempt(
    results: bool,
) -> None:
    handler = await holding(
        Handler(
            asks_models(),
            host=SidecarHost({"monitor": "mockllm/monitor"}),
            reject_status=503,
        )
    )

    async def judging_usage() -> list[dict[str, Any]]:
        eval = await _eval(handler, "request_result")
        async with _client(handler, results=results) as client:
            answered = await _call(client, "request_result", {"id": "irid-1"} | eval)
        used: list[dict[str, Any]] = answered["result"]["judging_usage"]
        return used

    await handler.request(_asked("x1"))
    await handler.reply(reply(_message("Tokyo"), id="x1"))
    once = await judging_usage()
    # the client retries the request under the same id, and the sentinel's
    # models are billed again
    await handler.request(_asked("x2"))
    await handler.reply(reply(_message("Oslo"), id="x2"))
    twice = await judging_usage()

    # by the sidecar's role, and with no role for a model the monitor named
    assert [(used.get("role"), used["model"]) for used in once] == [
        ("monitor", "mockllm/monitor"),
        (None, "mockllm/named"),
    ]
    for first, both in zip(once, twice, strict=True):
        assert first["usage"]["total_tokens"] > 0
        assert both["usage"]["total_tokens"] == 2 * first["usage"]["total_tokens"]


@protocol
def takes_its_time() -> ProtocolGroup:
    # on the test's clock: 2 s to judge each request, and 3 s to reject a call
    # for Tokyo or 1 s to pass any other
    async def request(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision:
        _Clock.now += 2.0
        return Decision.proceed()

    async def call(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision:
        if "Tokyo" in json.dumps(step.call.arguments):
            _Clock.now += 3.0
            return Decision.reject("Tokyo is not allowed.")
        _Clock.now += 1.0
        return Decision.proceed()

    return ProtocolGroup(request, call)


@pytest.mark.anyio
async def test_request_result_gives_how_long_the_latest_attempt_was_judged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_handler, "time", _Clock)
    _Clock.now = 0.0
    handler = await holding(Handler(takes_its_time(), reject_status=503))
    await handler.request(_asked("x1"))
    await handler.reply(reply(_message("Tokyo"), id="x1"))
    first = (await _request_result(handler))["judging_time"]
    # the client retries the request under the same id
    await handler.request(_asked("x2"))
    await handler.reply(reply(_message("Oslo"), id="x2"))

    result = await _request_result(handler)

    # the request's judging and its reply's, together
    assert first == 2.0 + 3.0
    # only the latest attempt: an earlier one's time isn't the eval's to count
    # again
    assert result["judging_time"] == 2.0 + 1.0
    # no model judged, so nothing was billed, and the answer says so
    assert result["judging_usage"] == []


@protocol
def bills_then_fails() -> Protocol:
    # asks its model, which is billed, takes 4 s on the test's clock, then
    # fails
    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision:
        await context.host.generate("Is this request suspicious?")
        _Clock.now += 4.0
        raise RuntimeError("the monitor broke")

    return decide


@protocol
def judges_the_request() -> Protocol:
    # asks its model about each request, which is billed, takes 2 s on the
    # test's clock, and passes it
    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision:
        await context.host.generate("Is this request suspicious?")
        _Clock.now += 2.0
        return Decision.proceed()

    return decide


@protocol
def bills_then_hangs() -> Protocol:
    # asks its model, which is billed, then never decides
    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision:
        await context.host.generate("Is this request suspicious?")
        await anyio.sleep_forever()
        return Decision.proceed()

    return decide


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_judging_cut_off_by_the_time_limit_is_still_counted() -> None:
    # the sentinel is cancelled at the time limit, so its time and its billed
    # judge call are kept only by what runs as it is cancelled
    handler = await holding(
        Handler(
            bills_then_hangs(),
            host=SidecarHost({"monitor": "mockllm/monitor"}),
            time_limit=0.1,
        )
    )
    with pytest.raises(TimeLimitError):
        await handler.request(_asked("x"))

    result = await _request_result(handler)

    assert (result["state"], result["reason"]) == ("not_judged", "time_limit")
    assert result["judging_time"] > 0
    [used] = result["judging_usage"]
    assert (used["role"], used["model"]) == ("monitor", "mockllm/monitor")
    assert used["usage"]["total_tokens"] > 0


async def _fails_its_sentinel(handler: Handler) -> None:
    # the sentinel's failure is the request's
    with pytest.raises(RuntimeError, match="the monitor broke"):
        await handler.request(_asked("x"))


async def _gets_a_provider_error(handler: Handler) -> None:
    await handler.request(_asked("x"))
    await handler.reply(reply(b"overloaded", status=529))


async def _never_gets_a_reply(handler: Handler) -> None:
    await handler.request(_asked("x"))


async def _has_its_reply_cut_off(handler: Handler) -> None:
    # the proxy refused the reply without handing it over
    await handler.request(_asked("x"))
    await handler.reply_not_judged("x", "cut_off")


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize(
    ("sentinel", "ends", "judged_for", "state"),
    [
        (bills_then_fails, _fails_its_sentinel, 4.0, None),
        (judges_the_request, _gets_a_provider_error, 2.0, "not_judged"),
        (judges_the_request, _never_gets_a_reply, 2.0, "awaiting_reply"),
        (judges_the_request, _has_its_reply_cut_off, 2.0, "not_judged"),
    ],
    ids=[
        "a sentinel that failed",
        "a provider error",
        "a reply that never came",
        "a reply cut off",
    ],
)
async def test_judging_that_ended_without_a_decision_is_still_counted(
    sentinel: Callable[[], Protocol],
    ends: Callable[[Handler], Awaitable[None]],
    judged_for: float,
    state: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the request is judged, and the judge model billed, and the call never
    # comes to a decision
    monkeypatch.setattr(_handler, "time", _Clock)
    _Clock.now = 0.0
    handler = await holding(
        Handler(sentinel(), host=SidecarHost({"monitor": "mockllm/monitor"}))
    )
    await ends(handler)

    result = await _request_result(handler)

    assert result["state"] != "decided"
    if state is not None:
        assert result["state"] == state
    assert result["judging_time"] == judged_for
    [used] = result["judging_usage"]
    assert (used["role"], used["model"]) == ("monitor", "mockllm/monitor")
    assert used["usage"]["total_tokens"] > 0


@pytest.mark.anyio
async def test_a_request_never_seen_is_unknown() -> None:
    handler = Handler(weather_rule())
    # a request with no id of the eval's is not one it can ask about
    await handler.request(request("x"))

    assert await _request_result(handler) == {"id": "irid-1", "records": "unknown"}


class _Clock:
    now = 0.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.now


@pytest.mark.anyio
async def test_a_request_whose_records_were_dropped_says_so_for_a_while(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = await holding(Handler(weather_rule(), state=MemoryState(expiry=100)))
    await handler.request(_asked("x"))

    _Clock.now = 150.0
    dropped = await _request_result(handler)
    _Clock.now = 300.0
    forgotten = await _request_result(handler)

    assert dropped == {"id": "irid-1", "records": "dropped"}
    assert forgotten == {"id": "irid-1", "records": "unknown"}


_WRONG_TOKEN = "a-token-the-eval-never-made-for-these-tests"

_UNPROVEN = [
    "a model request with a wrong mac",
    "a model request with a stale timestamp",
    "an eval call with a wrong mac",
    "an eval call with a stale timestamp",
    "a registration with a wrong token",
    "a registration with a wrong mac",
]

_PROVEN = ["a model request", "an eval call", "a registration with the token"]


async def _send(handler: Handler, client: httpx.AsyncClient, case: str) -> bool:
    # sends one request or call that names `eval_789`, and says whether the
    # sidecar took it; nothing here keeps the token for the eval, so only the
    # request or call itself can renew it
    stale = int(time.time()) - 1000
    result = _OTHER_METHODS["sample_result"]
    unsigned = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    model_requests: dict[str, dict[str, Any]] = {
        "a model request with a wrong mac": {"token": _WRONG_TOKEN},
        "a model request with a stale timestamp": {"timestamp": stale},
        "a model request": {},
    }
    calls = {
        "an eval call with a wrong mac": (
            "sample_result",
            result | eval_call("sample_result", token=_WRONG_TOKEN),
        ),
        "an eval call with a stale timestamp": (
            "sample_result",
            result | eval_call("sample_result", timestamp=stale),
        ),
        "an eval call": ("sample_result", result | eval_call("sample_result")),
        "a registration with a wrong token": (
            "register_task",
            _REGISTRATION | {"token": _WRONG_TOKEN},
        ),
        "a registration with a wrong mac": (
            "register_task",
            unsigned | eval_call("register_task", token=_WRONG_TOKEN),
        ),
        "a registration with the token": ("register_task", _REGISTRATION),
    }
    if case in model_requests:
        headers = inspect_headers(irid="irid-1", **model_requests[case])
        answer = await handler.request(replace(request("x"), headers=headers))
        return not isinstance(answer, Refuse)
    method, params = calls[case]
    answered = await _call(client, method, params)
    return "result" in answered or answered["error"]["code"] == UNKNOWN_SAMPLE


@pytest.mark.anyio
@pytest.mark.parametrize("case", _UNPROVEN)
async def test_a_request_or_call_that_fails_the_check_doesnt_keep_the_token(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    # The token is kept for 100 seconds after the eval was last named. A
    # sender that can't prove the token is refused, and must not keep it
    # alive, or anyone could keep an eval's token forever.
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    state = MemoryState(expiry=100)
    handler = await holding(Handler(weather_rule(), state=state))

    _Clock.now = 60.0
    async with _client(handler) as client:
        taken = await _send(handler, client, case)
    _Clock.now = 120.0

    assert not taken
    assert await state.eval_token("eval_789", renew=False) is None


@pytest.mark.anyio
@pytest.mark.parametrize("case", _PROVEN)
async def test_a_request_or_call_that_passes_the_check_keeps_the_token(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    state = MemoryState(expiry=100)
    handler = await holding(Handler(weather_rule(), state=state))

    _Clock.now = 60.0
    async with _client(handler) as client:
        taken = await _send(handler, client, case)
    _Clock.now = 120.0

    assert taken
    assert await state.eval_token("eval_789", renew=False) == TOKEN


@pytest.mark.anyio
async def test_a_request_dropped_is_unknown_to_another_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = await holding(Handler(weather_rule(), state=MemoryState(expiry=100)))
    await handler.request(_asked("x"))

    _Clock.now = 150.0
    other = await _eval(handler, "request_result", "eval_other")
    async with _client(handler) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"} | other)

    # nothing says that another eval's request ever existed
    assert answered["result"] == {"id": "irid-1", "records": "unknown"}


class _Tally(StoreModel):
    lookups: int = 0


@protocol
def counts_lookups() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        context.store_as(_Tally).lookups += 1
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_run_says_what_it_came_to_and_its_store() -> None:
    handler = await holding(Handler(counts_lookups()))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Oslo")))

    eval = await _eval(handler, "sample_result")
    async with _client(handler) as client:
        answered = await _call(
            client, "sample_result", {"sample_uuid": "sample_123"} | eval
        )
        other = await _call(
            client,
            "sample_result",
            {"sample_uuid": "sample_123"}
            | await _eval(handler, "sample_result", "eval_other"),
        )

    # another eval can't read it
    assert other["error"]["code"] == UNKNOWN_SAMPLE
    result = answered["result"]
    assert result["sample_uuid"] == "sample_123"
    assert result["outcome"] == "proceed"
    # keyed as the sentinel wrote it, by the store model and its field
    assert [
        value for key, value in result["store"].items() if key.endswith("lookups")
    ] == [1]


@pytest.mark.anyio
async def test_a_sample_a_sentinel_ended_says_so() -> None:
    handler = await holding(Handler(weather_rule("terminate", "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    eval = await _eval(handler, "sample_result")
    async with _client(handler) as client:
        answered = await _call(
            client, "sample_result", {"sample_uuid": "sample_123"} | eval
        )

    assert answered["result"]["outcome"] == "terminate"


@pytest.mark.anyio
async def test_a_run_no_eval_names_cant_be_reached_by_its_opening() -> None:
    # Such a run is keyed by a digest of how its conversation opens, which
    # anyone who knows the opening can work out.
    handler = Handler(counts_lookups())
    asked = request("x")
    await handler.request(asked)
    await handler.reply(reply(_message("Tokyo")))
    opening = _handler._opening(
        await _handler.read_opening(asked.provider, asked.body, asked.format)
    )

    async with _client(handler) as client:
        result = await _call(
            client,
            "sample_result",
            {"sample_uuid": opening} | await _eval(handler, "sample_result"),
        )
        ended = await _call(
            client,
            "end_sample",
            {"sample_uuid": opening} | await _eval(handler, "end_sample"),
        )
    later = await handler.request(request("y"))

    assert result["error"]["code"] == UNKNOWN_SAMPLE
    assert ended["result"]["state"] == "unknown"
    assert not isinstance(later, Refuse)


@pytest.mark.anyio
async def test_a_sample_never_seen_has_no_result() -> None:
    handler = Handler(weather_rule())
    eval = await _eval(handler, "sample_result")
    async with _client(handler) as client:
        answered = await _call(
            client, "sample_result", {"sample_uuid": "sample_123"} | eval
        )

    assert answered["error"]["code"] == UNKNOWN_SAMPLE


@pytest.mark.anyio
async def test_a_sample_its_eval_ended_is_dropped_and_refused_from_then_on() -> None:
    handler = await holding(Handler(counts_lookups()))
    await handler.request(_asked("x"))

    eval = await _eval(handler, "end_sample")
    other = await _eval(handler, "end_sample", "eval_other")
    async with _client(handler) as client:
        # another eval can't end it
        not_its = await _call(
            client, "end_sample", {"sample_uuid": "sample_123"} | other
        )
        ended = await _call(client, "end_sample", {"sample_uuid": "sample_123"} | eval)
        again = await _call(client, "end_sample", {"sample_uuid": "sample_123"} | eval)
        never = await _call(client, "end_sample", {"sample_uuid": "sample_999"} | eval)
        never_again = await _call(
            client, "end_sample", {"sample_uuid": "sample_999"} | eval
        )
        result = await _call(
            client,
            "sample_result",
            {"sample_uuid": "sample_123"} | eval_call("sample_result"),
        )
    later = await handler.request(_asked("y", irid="irid-2"))
    unseen = await handler.request(_asked("z", irid="irid-3", sample_uuid="sample_999"))

    assert [
        answered["result"]["state"]
        for answered in (not_its, ended, again, never, never_again)
    ] == ["unknown", "ended", "ended", "unknown", "unknown"]
    assert result["error"]["code"] == UNKNOWN_SAMPLE
    assert isinstance(later, Refuse)
    assert (await _request_result(handler, "irid-2"))["outcome"] == "terminate"
    # a run the sidecar never saw is left as it was, not refused from then on
    assert not isinstance(unseen, Refuse)


@pytest.mark.anyio
async def test_results_can_be_withheld() -> None:
    async with _client(Handler(weather_rule()), results=False) as client:
        withheld = await _call(client, "sample_result", {"sample_uuid": "sample_123"})
        registered = await _call(client, "register_task", _REGISTRATION)

    assert withheld["error"]["code"] == METHOD_NOT_FOUND
    assert "doesn't send results" in withheld["error"]["message"]
    assert "result" in registered


@pytest.mark.anyio
async def test_a_refused_turn_is_told_without_the_results() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    eval = await _eval(handler, "request_result")
    async with _client(handler, results=False) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"} | eval)

    result = answered["result"]
    assert (result["outcome"], result["message"]) == ("reject", "Tokyo is not allowed.")
    assert result["judgment"]["withheld"] is not None
    assert "recorded" not in result


@pytest.mark.anyio
async def test_a_port_of_its_own_needs_its_own_token() -> None:
    handler = Handler(weather_rule())
    eval = await _eval(handler, "end_sample")
    async with _client(handler, token="s3cret") as client:
        without = await client.post(PATH, json={})
        wrong = await client.post(
            PATH, json={}, headers={"authorization": "Bearer nope"}
        )
        right = await _call(
            client,
            "end_sample",
            {"sample_uuid": "sample_123"} | eval,
            authorization="Bearer s3cret",
        )
        probe = await client.get(INFO_PATH)

    assert (without.status_code, wrong.status_code) == (401, 401)
    assert right["result"]["state"] == "unknown"
    # the probe asks for no token
    assert probe.status_code == 200


@pytest.mark.anyio
async def test_a_call_larger_than_the_limit_is_not_read() -> None:
    app = Starlette(routes=eval_routes(Handler(weather_rule()), max_body_bytes=10))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
    ) as client:
        answered = await client.post(PATH, json=_REGISTRATION)

    assert answered.status_code == 413


def _arguments(*given: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(["--sentinel", "x", *given])


def _paths(app: Starlette) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_both_endpoints_share_a_port_by_default() -> None:
    served = apps(Handler(weather_rule()), _arguments())

    assert [(port, _paths(app)) for app, port in served] == [
        (8900, {"/proxy/v1", "/health", INFO_PATH, PATH, HELD_REPLY_PATH})
    ]


def test_the_endpoint_for_evals_can_have_a_port_and_a_token_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_VARIABLE, "s3cret")

    served = apps(Handler(weather_rule()), _arguments("--eval-port", "8901"))

    assert [(port, _paths(app)) for app, port in served] == [
        (8900, {"/proxy/v1", "/health"}),
        (8901, {"/health", INFO_PATH, PATH, HELD_REPLY_PATH}),
    ]


def test_a_port_of_its_own_without_a_token_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_VARIABLE, raising=False)

    with pytest.raises(ValueError, match=TOKEN_VARIABLE):
        apps(Handler(weather_rule()), _arguments("--eval-port", "8901"))


def test_schema_file_is_what_the_models_say() -> None:
    # rewrite it with: python -m inspect_sentinel.sidecar._eval_endpoint
    assert _SCHEMA == schema()


def test_calls_in_these_tests_fit_the_schema_file() -> None:
    methods = _SCHEMA["methods"]
    assert _fits(methods["register_task"]["params"], _REGISTRATION)
    signed = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    assert _fits(
        methods["register_task"]["params"], signed | eval_call("register_task")
    )
    eval = eval_call("request_result")
    assert _fits(
        methods["complete_registration"]["params"],
        _COMPLETION | eval_call("complete_registration"),
    )
    assert _fits(methods["request_result"]["params"], {"id": "irid-1"} | eval)
    assert _fits(
        methods["request_result"]["params"], {"id": "irid-1", "wait": 30} | eval
    )
    # every call for an eval proves it holds the eval's token
    assert not _fits(methods["complete_registration"]["params"], _COMPLETION)
    assert not _fits(methods["request_result"]["params"], {"id": "irid-1"})
    # a token is at least 32 characters
    assert not _fits(
        methods["register_task"]["params"], _REGISTRATION | {"token": "short"}
    )


@pytest.mark.anyio
async def test_a_reply_in_flight_when_its_eval_ended_the_run_is_refused() -> None:
    handler = await holding(Handler(counts_lookups()))
    await handler.request(_asked("x"))
    await handler.end_sample("sample_123", "eval_789")

    answer = await handler.reply(reply(_message("Oslo")))

    assert isinstance(answer, Refuse)
    # the run is not made again
    assert await handler.sample_result("sample_123", "eval_789") is None
    result = await _request_result(handler)
    assert (result["state"], result["outcome"]) == ("decided", "terminate")


# made in each test, as an event belongs to the event loop it is made in:
# one that lets the request be judged, one that lets the reply be, and one
# set once the reply's judging has begun
_gates: list[anyio.Event] = []


@protocol
def waits_to_be_told() -> ProtocolGroup:
    async def before(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        await _gates[0].wait()
        return Decision.proceed()

    async def call(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        _gates[2].set()
        await _gates[1].wait()
        return Decision.proceed()

    return ProtocolGroup(before, call)


@protocol
def bills_then_waits() -> Protocol:
    # asks its model, which is billed, then waits to be told to decide
    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision:
        await context.host.generate("Is this request suspicious?")
        _gates[0].set()
        await _gates[1].wait()
        return Decision.proceed()

    return decide


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_a_judge_call_counts_while_its_step_is_still_judged() -> None:
    # an eval asking about an attempt still being judged, as for one it
    # retried, is told of the calls already billed
    _gates[:] = [anyio.Event(), anyio.Event()]
    handler = await holding(
        Handler(bills_then_waits(), host=SidecarHost({"monitor": "mockllm/monitor"}))
    )

    result: dict[str, Any] = {}
    async with anyio.create_task_group() as group:
        group.start_soon(handler.request, _asked("x"))
        await _gates[0].wait()
        result = await _request_result(handler)
        _gates[1].set()

    assert result["state"] == "judging"
    [used] = result["judging_usage"]
    assert used["usage"]["total_tokens"] > 0


@pytest.mark.anyio
async def test_a_request_judged_while_its_eval_ended_the_run_is_refused() -> None:
    _gates[:] = [anyio.Event(), anyio.Event(), anyio.Event()]
    handler = await holding(Handler(waits_to_be_told()))
    answers: list[Any] = []

    async def ask() -> None:
        answers.append(await handler.request(_asked("x")))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        await anyio.wait_all_tasks_blocked()
        await handler.end_sample("sample_123", "eval_789")
        _gates[0].set()

    (answer,) = answers
    assert isinstance(answer, Refuse)
    assert await handler.sample_result("sample_123", "eval_789") is None


@pytest.mark.anyio
async def test_a_reply_being_judged_is_judging() -> None:
    _gates[:] = [anyio.Event(), anyio.Event(), anyio.Event()]
    handler = await holding(Handler(waits_to_be_told()))
    states: list[str] = []

    async def ask() -> None:
        await handler.request(_asked("x"))
        await handler.reply(reply(_message("Oslo")))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        await anyio.wait_all_tasks_blocked()
        states.append((await _request_result(handler))["state"])
        _gates[0].set()
        await _gates[2].wait()
        states.append((await _request_result(handler))["state"])
        _gates[1].set()

    states.append((await _request_result(handler))["state"])
    # judging the request, then judging the reply, then decided
    assert states == ["judging", "judging", "decided"]


@pytest.mark.anyio
async def test_a_rejected_turn_is_given_back_by_request_result() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    result = await _request_result(handler)

    withheld = result["judgment"]["withheld"]
    assert withheld["message"]["role"] == "assistant"
    assert withheld["results"] == {"toolu_1": "Tokyo is not allowed."}
    # the rejected call's arguments never reach the eval
    (call,) = withheld["message"]["tool_calls"]
    assert (call["id"], call["function"], call["arguments"]) == (
        "toolu_1",
        "get_weather",
        {},
    )
    assert "view" not in call
    assert withheld["redacted"] == ["toolu_1"]


@pytest.mark.anyio
async def test_a_turn_with_a_call_taken_out_is_given_back_by_request_result() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_asked("x"))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")))
    )

    result = await _request_result(handler)

    # the turn went on without Tokyo
    assert result["outcome"] == "proceed"
    assert "withheld" not in result["judgment"]
    proposed = result["judgment"]["proposed"]
    assert [
        (call["id"], call["arguments"]) for call in proposed["message"]["tool_calls"]
    ] == [("toolu_1", {}), ("toolu_2", {"location": "Paris"})]
    assert proposed["results"] == {"toolu_1": "Tokyo is not allowed."}
    assert proposed["redacted"] == ["toolu_1"]


_GUESSED = "a-token-guessed-by-someone-other-than-the-eval"


def _wrong(method: str) -> dict[str, Any]:
    # a call for the eval signed with a token that isn't the eval's
    return eval_call(method, token=_GUESSED)


@pytest.mark.anyio
async def test_only_the_evals_own_token_reaches_its_calls() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    async with _client(handler) as client:
        await _registered(client)
        await handler.request(_asked("x"))
        await handler.reply(reply(_message("Tokyo")))
        without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
        refused = [
            await _call(
                client,
                "complete_registration",
                _COMPLETION | _wrong("complete_registration"),
            ),
            await _call(
                client, "request_result", {"id": "irid-1"} | _wrong("request_result")
            ),
            await _call(
                client,
                "sample_result",
                {"sample_uuid": "sample_123"} | _wrong("sample_result"),
            ),
            await _call(
                client,
                "end_sample",
                {"sample_uuid": "sample_123"} | _wrong("end_sample"),
            ),
            # nobody but the eval registers for it again
            await _call(client, "register_task", _REGISTRATION | {"token": _GUESSED}),
            await _call(
                client, "register_task", without_token | _wrong("register_task")
            ),
        ]
        again = await _call(
            client, "register_task", without_token | eval_call("register_task")
        )
        read = await _call(
            client, "request_result", {"id": "irid-1"} | eval_call("request_result")
        )

    assert [answered["error"]["code"] for answered in refused] == [UNAUTHORIZED] * 6
    assert "result" in again
    assert read["result"]["outcome"] == "reject"


@pytest.mark.anyio
async def test_a_request_of_another_eval_is_unknown() -> None:
    handler = await holding(Handler(weather_rule()))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Oslo")))

    other = await _eval(handler, "request_result", "eval_other")
    async with _client(handler) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"} | other)

    # nothing says that the request exists
    assert answered["result"] == {"id": "irid-1", "records": "unknown"}


@pytest.mark.anyio
async def test_a_registration_that_fails_leaves_no_token_behind() -> None:
    # had the failed registration kept a token it never gave out, the eval
    # could never register again, once there was room
    handler = Handler(weather_rule(), max_task_registrations=1)
    async with _client(handler) as client:
        await _call(client, "register_task", _REGISTRATION | {"eval_id": "eval_a"})
        full = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_b"}
        )
        again = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_b"}
        )

    assert full["error"]["code"] == REGISTRATIONS_FULL
    assert again["error"]["code"] == REGISTRATIONS_FULL


@pytest.mark.anyio
async def test_a_token_past_the_registration_limit_fails_with_its_own_error() -> None:
    # a token outlives the registration that made it, and a task registered
    # beforehand takes no registration at all, so the tokens are bounded on
    # their own: by the same number, and nothing kept is dropped to make room
    task = FileTask.model_validate({"task": "philadelphia", "samples": []})
    handler = Handler(
        weather_rule(), tasks={"philadelphia": task}, max_task_registrations=2
    )
    async with _client(handler) as client:
        kept = [
            await _call(
                client,
                "register_task",
                _REGISTRATION | {"eval_id": id, "token": _token(id)},
            )
            for id in ["eval_1", "eval_2"]
        ]
        refused = await _call(
            client,
            "register_task",
            _REGISTRATION | {"eval_id": "eval_3", "token": _token("eval_3")},
        )
        again = await _call(
            client,
            "register_task",
            _REGISTRATION | {"eval_id": "eval_1", "token": _token("eval_1")},
        )

    assert all("result" in k for k in kept)
    assert refused["error"]["code"] == REGISTRATIONS_FULL
    assert "or the tokens of as many evals" in refused["error"]["message"]
    # a kept eval registers again as before
    assert "result" in again


@pytest.mark.anyio
async def test_a_registration_whose_eval_gets_no_token_is_not_kept() -> None:
    # the first eval's registration expires at once and frees its slot, but
    # its token stays; the next eval is refused before anything is kept of it
    handler = Handler(weather_rule(), max_task_registrations=1)
    async with _client(handler) as client:
        await _call(
            client,
            "register_task",
            _REGISTRATION | {"eval_id": "eval_1", "expiry": 0.001},
        )
        await anyio.sleep(0.01)
        refused = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_2"}
        )

    assert refused["error"]["code"] == REGISTRATIONS_FULL
    assert await handler._state.registration("eval_2", "philadelphia") is None  # pyright: ignore[reportPrivateUsage]
    assert await handler._state.eval_token("eval_2") is None  # pyright: ignore[reportPrivateUsage]


class _GatedState(MemoryState):
    """State whose lookups of a task's last registration wait in turn, as a shared backend's may. A registration held there has passed the check of its token, and has kept nothing yet."""

    def __init__(self, gates: list[anyio.Event]) -> None:
        super().__init__()
        self.gates = gates
        self._arriving = iter(gates)

    async def registration(
        self, eval_id: str, task: str, *, renew: bool = True
    ) -> RegisteredTask | None:
        gate = next(self._arriving, None)
        if gate is not None:
            await gate.wait()
        return await super().registration(eval_id, task, renew=renew)


async def _registered_in_turn(
    handler: Handler,
    state: _GatedState,
    first: dict[str, Any],
    second: dict[str, Any],
    between: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Two registrations that both pass the check of the eval's token before
    # either keeps anything, then keep, or fail to, in the order they came,
    # with `between` called once the first has.
    answers: dict[str, dict[str, Any]] = {}
    async with _client(handler) as client:

        async def register(name: str, changes: dict[str, Any]) -> None:
            answers[name] = await _call(
                client, "register_task", _REGISTRATION | changes
            )

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(register, "first", first)
            await anyio.wait_all_tasks_blocked()
            tasks.start_soon(register, "second", second)
            await anyio.wait_all_tasks_blocked()
            state.gates[0].set()
            await anyio.wait_all_tasks_blocked()
            if between is not None:
                between()
            state.gates[1].set()
    return answers["first"], answers["second"]


@pytest.mark.anyio
async def test_of_two_first_registrations_with_other_tokens_only_one_keeps_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The eval's token and its task are kept in one step. So the registration
    # that comes second finds the first's token kept, and keeps neither its
    # own token nor its task in place of the first's. Nor does it keep the
    # first's task alive: that is kept for 100 seconds from when the first
    # registered it, at 0, though the second is refused at 60.
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    state = _GatedState([anyio.Event(), anyio.Event()])
    handler = Handler(weather_rule(), state=state)

    def a_minute_on() -> None:
        _Clock.now = 60.0

    first, second = await _registered_in_turn(
        handler,
        state,
        {"sample_ids": [1], "expiry": 100},
        {"token": _token("another"), "sample_ids": [2], "expiry": 100},
        between=a_minute_on,
    )

    assert "result" in first
    assert second["error"]["code"] == UNAUTHORIZED
    kept = await state.registration("eval_789", "philadelphia", renew=False)
    assert kept is not None and kept.sample_ids == {1}
    assert await state.eval_token("eval_789", renew=False) == TOKEN
    _Clock.now = 120.0
    assert await state.registration("eval_789", "philadelphia", renew=False) is None


@pytest.mark.anyio
async def test_a_first_registration_sent_twice_at_once_is_kept_both_times() -> None:
    # the same token, as when a registration whose answer was slow is sent
    # again: each is kept, and the eval has the one token
    state = _GatedState([anyio.Event(), anyio.Event()])
    handler = Handler(weather_rule(), state=state)

    first, second = await _registered_in_turn(
        handler, state, {"sample_ids": [1]}, {"sample_ids": [1, 2]}
    )

    assert "result" in first and "result" in second
    kept = await state.registration("eval_789", "philadelphia")
    assert kept is not None and kept.sample_ids == {1, 2}
    assert await state.eval_token("eval_789", renew=False) == TOKEN


@pytest.mark.anyio
async def test_a_registration_cancelled_before_it_is_kept_keeps_nothing() -> None:
    # Cancelled at the last point it waits before the one step that keeps
    # its token and its task: neither is kept, and the eval registers
    # afterwards as for the first time.
    state = _GatedState([anyio.Event()])
    handler = Handler(weather_rule(), state=state)
    async with _client(handler) as client:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_call, client, "register_task", _REGISTRATION)
            await anyio.wait_all_tasks_blocked()
            tasks.cancel_scope.cancel()

        assert await state.eval_token("eval_789", renew=False) is None
        assert await state.registration("eval_789", "philadelphia") is None
        await _registered(client, token=_token("another"))

    assert await state.eval_token("eval_789", renew=False) == _token("another")


@pytest.mark.anyio
async def test_a_registration_refused_for_its_token_changes_no_task() -> None:
    # refused in the state's one step too, not only by the check before it:
    # the task the eval registered is as it was, and its token isn't renewed
    handler = Handler(weather_rule())
    registration = TaskRegistration.model_validate(
        {k: v for k, v in _REGISTRATION.items() if k not in ("eval_id", "token")}
    )
    await handler.register_task("eval_789", registration, token=TOKEN)
    changed = registration.model_copy(update={"sample_ids": [2]})

    with pytest.raises(UnauthorizedError):
        await handler.register_task("eval_789", changed, token=_token("another"))

    kept = await handler._state.registration("eval_789", "philadelphia")  # pyright: ignore[reportPrivateUsage]
    assert kept is not None and kept.sample_ids == {1}
    assert await handler._state.eval_token("eval_789", renew=False) == TOKEN  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_a_folder_tasks_registration_keeps_the_token_or_nothing() -> None:
    # a task registered beforehand stores no task, so its registration keeps
    # the eval's token alone, in one step with the checks of it
    task = FileTask.model_validate({"task": "philadelphia", "samples": []})
    handler = Handler(
        weather_rule(), tasks={"philadelphia": task}, max_task_registrations=1
    )
    registration = TaskRegistration.model_validate(
        {k: v for k, v in _REGISTRATION.items() if k not in ("eval_id", "token")}
    )

    assert await handler.register_task("eval_1", registration, token=TOKEN) is None
    with pytest.raises(UnauthorizedError):
        await handler.register_task("eval_1", registration, token=_token("another"))
    with pytest.raises(RegistrationsFullError):
        await handler.register_task("eval_2", registration, token=_token("eval_2"))

    assert await handler._state.eval_token("eval_1", renew=False) == TOKEN  # pyright: ignore[reportPrivateUsage]
    assert await handler._state.eval_token("eval_2", renew=False) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_a_first_registration_sent_again_with_its_token_is_accepted() -> None:
    # the eval makes its token and sends it with its first registration, so
    # when the answer is lost the registration can be sent again with it
    async with _client(Handler(weather_rule())) as client:
        first = await _call(client, "register_task", _REGISTRATION)
        again = await _call(client, "register_task", _REGISTRATION)

    assert "token" not in first["result"]
    assert "token" not in again["result"]
    assert again["result"]["task"] == "philadelphia"


@pytest.mark.anyio
async def test_a_registration_with_another_token_is_refused() -> None:
    # once the sidecar keeps the eval's token, nobody registers for the eval
    # with another
    handler = Handler(weather_rule())
    async with _client(handler) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(
            client, "register_task", _REGISTRATION | {"token": _GUESSED}
        )

    assert answered["error"]["code"] == UNAUTHORIZED
    assert "another token" in answered["error"]["message"]


@pytest.mark.anyio
async def test_a_registration_with_a_mac_once_the_token_is_held_is_accepted() -> None:
    without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(
            client, "register_task", without_token | eval_call("register_task")
        )

    assert answered["result"]["task"] == "philadelphia"


@pytest.mark.anyio
async def test_a_registration_with_neither_token_nor_mac_is_invalid_params() -> None:
    without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(client, "register_task", without_token)

    assert answered["error"]["code"] == INVALID_PARAMS


@pytest.mark.anyio
async def test_a_registration_with_both_token_and_mac_is_invalid_params() -> None:
    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(
            client, "register_task", _REGISTRATION | eval_call("register_task")
        )

    assert answered["error"]["code"] == INVALID_PARAMS


@pytest.mark.anyio
@pytest.mark.parametrize("left_out", ["timestamp", "mac"])
async def test_a_registration_with_half_a_signature_is_invalid_params(
    left_out: str,
) -> None:
    without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    half = {k: v for k, v in eval_call("register_task").items() if k != left_out}
    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(client, "register_task", without_token | half)

    assert answered["error"]["code"] == INVALID_PARAMS


@pytest.mark.anyio
async def test_a_token_shorter_than_32_characters_is_invalid_params() -> None:
    handler = Handler(weather_rule())
    async with _client(handler) as client:
        answered = await _call(
            client, "register_task", _REGISTRATION | {"token": "x" * 31}
        )

    assert answered["error"]["code"] == INVALID_PARAMS
    assert await handler._state.eval_token("eval_789") is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_a_registration_with_a_mac_and_no_token_held_is_unknown_task() -> None:
    # as after a restart: the eval registers again, with its token
    without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(
            client, "register_task", without_token | eval_call("register_task")
        )

    assert answered["error"]["code"] == UNKNOWN_TASK
    assert "holds no token for the eval" in answered["error"]["message"]


@pytest.mark.anyio
async def test_a_registration_with_a_wrong_mac_is_unauthorized() -> None:
    without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(
            client, "register_task", without_token | _wrong("register_task")
        )

    assert answered["error"]["code"] == UNAUTHORIZED
    assert "mac isn't one made with the eval's token" in answered["error"]["message"]


@pytest.mark.anyio
async def test_a_registration_with_a_stale_timestamp_is_unauthorized() -> None:
    without_token = {k: v for k, v in _REGISTRATION.items() if k != "token"}
    stale = int(time.time()) - 400
    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(
            client,
            "register_task",
            without_token | eval_call("register_task", timestamp=stale),
        )

    assert answered["error"]["code"] == UNAUTHORIZED
    # how far off, so an operator sees a clock problem
    assert "40" in answered["error"]["message"]
    assert "seconds behind the sidecar's clock" in answered["error"]["message"]


# each method but `register_task`, with its params besides the eval's proof
_OTHER_METHODS: dict[str, dict[str, Any]] = {
    "complete_registration": _COMPLETION,
    "request_result": {"id": "irid-1"},
    "sample_result": {"sample_uuid": "sample_123"},
    "end_sample": {"sample_uuid": "sample_123"},
}


@pytest.mark.anyio
@pytest.mark.parametrize("method", list(_OTHER_METHODS))
async def test_a_call_with_a_wrong_mac_is_unauthorized(method: str) -> None:
    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await _call(client, method, _OTHER_METHODS[method] | _wrong(method))

    assert answered["error"]["code"] == UNAUTHORIZED


@pytest.mark.anyio
@pytest.mark.parametrize("method", list(_OTHER_METHODS))
async def test_a_call_with_another_methods_mac_is_unauthorized(method: str) -> None:
    # a mac proves one method's call, not another's
    other = "end_sample" if method != "end_sample" else "sample_result"
    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await _call(
            client, method, _OTHER_METHODS[method] | eval_call(other)
        )

    assert answered["error"]["code"] == UNAUTHORIZED


@pytest.mark.anyio
@pytest.mark.parametrize("method", list(_OTHER_METHODS))
@pytest.mark.parametrize("off", [-400, 400], ids=["behind", "ahead"])
async def test_a_call_with_a_stale_timestamp_is_unauthorized(
    method: str, off: int
) -> None:
    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await _call(
            client,
            method,
            _OTHER_METHODS[method]
            | eval_call(method, timestamp=int(time.time()) + off),
        )

    assert answered["error"]["code"] == UNAUTHORIZED
    assert "timestamp is" in answered["error"]["message"]


@pytest.mark.anyio
@pytest.mark.parametrize("method", list(_OTHER_METHODS))
async def test_a_call_with_a_timestamp_too_large_for_a_float_is_unauthorized(
    method: str,
) -> None:
    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await _call(
            client,
            method,
            _OTHER_METHODS[method] | eval_call(method, timestamp=10**400),
        )

    assert answered["error"]["code"] == UNAUTHORIZED
    assert "timestamp is far" in answered["error"]["message"]


@pytest.mark.anyio
@pytest.mark.parametrize("mac", ["", "*", "\ud800", "é" * 64])
async def test_a_call_whose_mac_isnt_hex_is_unauthorized(mac: str) -> None:
    params = _OTHER_METHODS["sample_result"] | eval_call("sample_result") | {"mac": mac}
    # sent as JSON escapes, as any client may, since a lone surrogate can't be
    # encoded as UTF-8
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "sample_result", "params": params}
    )
    async with _client(Handler(weather_rule())) as client:
        await _registered(client)
        answered = await client.post(
            PATH, content=body, headers={"content-type": "application/json"}
        )

    assert answered.status_code == 200
    assert answered.json()["error"]["code"] == UNAUTHORIZED


@pytest.mark.anyio
@pytest.mark.parametrize("method", list(_OTHER_METHODS))
async def test_a_call_with_no_token_held_gets_the_holds_nothing_answer(
    method: str,
) -> None:
    # as after a restart, whoever signed the call
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(
            client, method, _OTHER_METHODS[method] | eval_call(method)
        )

    expected = {
        "complete_registration": ("error", UNKNOWN_TASK),
        "request_result": ("records", "unknown"),
        "sample_result": ("error", UNKNOWN_SAMPLE),
        "end_sample": ("state", "unknown"),
    }[method]
    if expected[0] == "error":
        assert answered["error"]["code"] == expected[1]
    else:
        assert answered["result"][expected[0]] == expected[1]


@pytest.mark.anyio
async def test_request_result_waits_for_a_reply_being_judged() -> None:
    _gates[:] = [anyio.Event(), anyio.Event(), anyio.Event()]
    handler = await holding(Handler(waits_to_be_told()))
    results: list[dict[str, Any]] = []

    async def ask() -> None:
        await handler.request(_asked("x"))
        await handler.reply(reply(_message("Oslo")))

    async def wait_for_it() -> None:
        results.append(await _request_result(handler, wait=10))

    _gates[0].set()
    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        await _gates[2].wait()
        group.start_soon(wait_for_it)
        await anyio.wait_all_tasks_blocked()
        # still waiting while the reply is judged
        assert not results
        _gates[1].set()

    (result,) = results
    assert (result["state"], result["outcome"]) == ("decided", "proceed")


@pytest.mark.anyio
async def test_request_result_waits_no_longer_than_asked() -> None:
    _gates[:] = [anyio.Event(), anyio.Event(), anyio.Event()]
    handler = await holding(Handler(waits_to_be_told()))
    result: dict[str, Any] = {}

    async def ask() -> None:
        await handler.request(_asked("x"))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        await anyio.wait_all_tasks_blocked()
        with anyio.fail_after(5):
            result = await _request_result(handler, wait=0.1)
        _gates[0].set()

    assert result["state"] == "judging"


@pytest.mark.anyio
@pytest.mark.parametrize("strictness", ["open", "known_tasks", "registered_only"])
async def test_after_a_restart_a_signed_request_is_unregistered_until_the_eval_registers_again(
    strictness: Strictness,
) -> None:
    # a fresh state, as after a restart: the sidecar holds no token for the
    # eval, so it can't check the request, whatever its strictness
    handler = Handler(weather_rule(), state=MemoryState(), strictness=strictness)
    refused = await handler.request(_asked("x"))
    # Inspect's retry registers the task again, with the same token
    async with _client(handler) as client:
        await _registered(client)
        await _call(
            client,
            "complete_registration",
            _COMPLETION | eval_call("complete_registration"),
        )
    judged = await handler.request(_asked("y", irid="irid-2"))

    assert isinstance(refused, Refuse) and refused.status == 400
    assert "holds no registration of the eval" in refused.message
    # the request that couldn't be checked was never kept
    assert await handler._state.attempts("irid-1", "eval_789") is None  # pyright: ignore[reportPrivateUsage]
    assert not isinstance(judged, Refuse)
    result = await _request_result(handler, "irid-2")
    assert result["state"] == "awaiting_reply"
