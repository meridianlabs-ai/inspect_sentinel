import base64
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import anyio
import httpx
import jsonschema
import pytest
from starlette.applications import Starlette

from inspect_sentinel import human, sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._host import HumanAnswer
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    SidecarHost,
    _state,
)
from inspect_sentinel.sidecar._eval_endpoint import (
    HELD_REPLY_PATH,
    PATH,
    SCHEMA_FILE,
    eval_routes,
)
from inspect_sentinel.sidecar._handler import FAILED_TRY, PENDING, _time_left
from inspect_sentinel.sidecar._json_rpc import (
    ANSWER_REFUSED,
    METHOD_NOT_FOUND,
    QUESTION_ANSWERED,
    UNAUTHORIZED,
    UNKNOWN_QUESTION,
)
from inspect_sentinel.sidecar._proxy_endpoint import PATH as PROXY_PATH
from inspect_sentinel.sidecar._proxy_endpoint import proxy_app
from inspect_sentinel.sidecar._review import UnknownQuestionError
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import (
    JSON,
    STREAM,
    TOKEN,
    anthropic_recorded,
    eval_call,
    holding,
    inspect_headers,
    message,
    ran,
    reply,
    request,
    signed,
    weather,
)

_SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())

_OTHER_TOKEN = "eval-other-token-that-inspect-made-for-these-tests"


@protocol
def escalates() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.escalate("every weather lookup is checked by a person")

    return decide


def _reviewed(choices: list[str] | None = None, **options: Any) -> Handler:
    # every call escalates, and human() asks a person about it
    return Handler(
        sequential([escalates(), human(stages=["tool_call"], choices=choices)]),
        **options,
    )


def _client(handler: Handler, **options: Any) -> httpx.AsyncClient:
    app = Starlette(routes=eval_routes(handler, max_body_bytes=1024 * 1024, **options))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
    )


@asynccontextmanager
async def _serving(
    handler: Handler, **options: Any
) -> AsyncGenerator[httpx.AsyncClient]:
    # the handler serving, as a running sidecar's does, and its endpoint for
    # evals; nothing in the block may hang
    await holding(handler)
    with anyio.fail_after(20):
        async with handler.serving(), _client(handler, **options) as client:
            yield client


async def _call(
    client: httpx.AsyncClient, method: str, params: dict[str, Any]
) -> dict[str, Any]:
    answered = await client.post(
        PATH, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    )
    assert answered.status_code == 200
    body: dict[str, Any] = answered.json()
    if "result" in body:
        jsonschema.validate(
            body["result"],
            {**_SCHEMA, **_SCHEMA["methods"][method]["result"]},
            cls=jsonschema.Draft202012Validator,
        )
    return body


def _from_eval(
    id: str = "x",
    irid: str = "irid-1",
    then: list[dict[str, Any]] | None = None,
    **sample: Any,
) -> Request:
    # a request an Inspect eval sends, signed, with its id for the request
    return replace(
        request(id, then=then or ()), headers=inspect_headers(irid=irid, **sample)
    )


async def _pending(
    client: httpx.AsyncClient,
    eval_id: str = "eval_789",
    token: str = TOKEN,
    **params: Any,
) -> list[dict[str, Any]]:
    answered = await _call(
        client,
        "pending_questions",
        eval_call("pending_questions", eval_id, token=token) | params,
    )
    questions: list[dict[str, Any]] = answered["result"]["questions"]
    return questions


def _answer_params(
    id: str,
    decision: str,
    reason: str | None = None,
    *,
    eval_id: str = "eval_789",
    token: str = TOKEN,
    signed_decision: str | None = None,
) -> dict[str, Any]:
    # an answer as Inspect signs one: the mac covers the answer itself
    at = int(time.time())
    mac = signed(
        token,
        [
            "answer_question",
            eval_id,
            at,
            id,
            signed_decision or decision,
            reason if reason is not None else "",
        ],
    )
    params: dict[str, Any] = {
        "eval_id": eval_id,
        "timestamp": at,
        "mac": mac,
        "id": id,
        "decision": decision,
    }
    if reason is not None:
        params["reason"] = reason
    return params


async def _answer(
    client: httpx.AsyncClient, id: str, decision: str, reason: str | None = None
) -> dict[str, Any]:
    return await _call(client, "answer_question", _answer_params(id, decision, reason))


async def _result(
    client: httpx.AsyncClient, irid: str = "irid-1", *, until: str | None = None
) -> dict[str, Any]:
    # what `request_result` says of the request, once its state is `until`
    while True:
        answered = await _call(
            client,
            "request_result",
            eval_call("request_result") | {"id": irid, "wait": 1},
        )
        result: dict[str, Any] = answered["result"]
        if until is None or result.get("state") == until:
            return result
        await anyio.sleep(0.01)


async def _cancel(
    client: httpx.AsyncClient,
    irid: str = "irid-1",
    eval_id: str = "eval_789",
    token: str = TOKEN,
) -> bool:
    answered = await _call(
        client,
        "cancel_request",
        eval_call("cancel_request", eval_id, token=token) | {"id": irid},
    )
    cancelled: bool = answered["result"]["cancelled"]
    return cancelled


async def _asked(
    handler: Handler,
    client: httpx.AsyncClient,
    *content: dict[str, Any],
    irid: str = "irid-1",
    id: str = "x",
    **sample: Any,
) -> dict[str, Any]:
    # the eval's request goes on, and a person is asked about the first call
    # in its reply: the question, once the reply has been answered "pending
    # review"
    assert isinstance(await handler.request(_from_eval(id, irid, **sample)), Pass)
    answer = await handler.reply(
        reply(message(*(content or (weather("Oslo"),))), id=id)
    )
    assert answer is PENDING
    (question,) = [q for q in await _pending(client) if q["request_id"] == irid]
    return question


def _result_request(id: str = "y") -> Request:
    # the eval's next request, which holds the result of the call it ran
    return _from_eval(id, "irid-2", ran("Oslo", "Rain."))


async def _asked_at_result(
    handler: Handler, client: httpx.AsyncClient
) -> dict[str, Any]:
    # a call ran, and a person is asked about its result, which the eval's
    # next request holds: the question, once that request has been answered
    # "pending review"
    assert isinstance(await handler.request(_from_eval("x")), Pass)
    first = await handler.reply(reply(message(weather("Oslo"))))
    if first is PENDING:
        # the handler asks about the call too
        (asked,) = await _pending(client)
        await _answer(client, asked["id"], "approve")
        await _result(client, until="decided")
    assert await handler.request(_result_request()) is PENDING
    (question,) = await _pending(client)
    return question


def _at_results(**options: Any) -> Handler:
    return Handler(human(stages=["tool_result"]), **options)


def _error(
    status: int = 529, body: bytes = b'{"type":"error"}', id: str = "y2"
) -> Reply:
    return replace(
        reply(body, id=id, status=status),
        headers={"retry-after": "7", "content-length": str(len(body))},
    )


# --- a question at a tool call: the provider has already replied ---


@pytest.mark.anyio
async def test_a_question_answers_the_model_call_pending_review_at_once() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        assert isinstance(await handler.request(_from_eval()), Pass)

        answer = await handler.reply(reply(message(weather("Oslo"))))

        # no connection waits for the person: the exchange is over, with a
        # status no provider SDK sends again by itself
        assert isinstance(answer, Refuse)
        assert answer.status == 400
        assert "waiting for a person's review" in answer.message
        result = await _result(client)
        assert result["state"] == "awaiting_review"
        assert "outcome" not in result
        (question,) = await _pending(client)
        assert question["request_id"] == "irid-1"


@pytest.mark.anyio
async def test_an_approved_call_is_decided_and_its_reply_is_kept_for_the_eval() -> None:
    handler = _reviewed()
    body = message(weather("Oslo"))
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        answered = await _answer(client, question["id"], "approve", "Looks fine.")

        assert answered["result"] == {"id": question["id"], "decision": "approve"}
        result = await _result(client, until="decided")
        assert result["outcome"] == "proceed"
        # The provider's reply is held for the eval, which gets it by sending
        # the call's request again. `request_result` doesn't carry it.
        assert "reply" not in result
        held = await _held_reply(client)
        assert (held.status_code, held.headers["content-type"]) == (200, JSON)
        assert held.content == body
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_neither_a_held_reply_nor_a_served_error_carries_the_providers_cookies() -> (
    None
):
    cookies = {"content-type": JSON, "set-cookie": "session=abc; Path=/"}
    handler = _reviewed()
    async with _serving(handler) as client:
        await _approved(
            handler, client, replace(reply(message(weather("Oslo"))), headers=cookies)
        )
        held = await _held_reply(client)
    errors = _at_results()
    async with _serving(errors) as client:
        await _failed_at_the_provider(
            errors, client, replace(_error(status=403), headers=cookies)
        )
        served = await _held_reply(client, "irid-2")

    assert held.status_code == 200 and "set-cookie" not in held.headers
    assert served.status_code == 403 and "set-cookie" not in served.headers


@pytest.mark.anyio
async def test_a_held_reply_has_the_providers_headers_without_those_of_one_hop() -> (
    None
):
    handler = _reviewed()
    async with _serving(handler) as client:
        assert isinstance(await handler.request(_from_eval()), Pass)
        held = replace(
            reply(message(weather("Oslo"))),
            headers={
                "content-type": JSON,
                "content-length": "311",
                "transfer-encoding": "chunked",
                "request-id": "req_011",
                "anthropic-ratelimit-requests-remaining": "49",
            },
        )
        assert await handler.reply(held) is PENDING
        (question,) = await _pending(client)
        await _answer(client, question["id"], "approve")

        await _result(client, until="decided")
        got = await _held_reply(client)

    assert got.headers["content-type"] == JSON
    assert got.headers["request-id"] == "req_011"
    assert got.headers["anthropic-ratelimit-requests-remaining"] == "49"
    # the headers of one hop are the sidecar's own for this answer
    assert "transfer-encoding" not in got.headers
    assert got.headers["content-length"] == str(len(got.content))


@pytest.mark.anyio
async def test_a_question_says_what_the_person_decides_about() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client, irid="irid-42")

    assert question["sample_uuid"] == "sample_123"
    assert question["request_id"] == "irid-42"
    assert question["stage"] == "tool_call"
    assert question["call"]["function"] == "get_weather"
    assert question["call"]["arguments"] == {"location": "Oslo"}
    assert question["choices"] == ["approve", "reject", "terminate"]
    assert question["escalations"] == [
        {
            "name": "escalates",
            "explanation": "every weather lookup is checked by a person",
        }
    ]
    # a person may take days: a question has no expiry unless the operator
    # sets one
    assert "closes" not in question


@pytest.mark.anyio
async def test_a_person_who_rejects_has_the_model_told_their_reason() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        await _answer(client, question["id"], "reject", "Not Oslo today.")

        result = await _result(client, until="decided")
    assert result["outcome"] == "reject"
    assert result["message"] == "Not Oslo today."


@pytest.mark.anyio
async def test_a_person_who_terminates_ends_the_run() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        await _answer(client, question["id"], "terminate", "Stop here.")

        result = await _result(client, until="decided")
        assert result["outcome"] == "terminate"
        # every later request of the run is refused
        later = await handler.request(_from_eval(id="y", irid="irid-2"))
        assert isinstance(later, Refuse)


@pytest.mark.anyio
async def test_an_accepted_answer_is_not_the_end_while_another_call_is_judged() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        first = await _asked(
            handler, client, weather("Oslo"), weather("Bergen", "toolu_2")
        )
        assert first["call"]["id"] == "toolu_1"

        await _answer(client, first["id"], "approve")

        # the turn's next call raises a question of its own
        result = await _result(client, until="awaiting_review")
        assert "outcome" not in result
        (second,) = await _pending(client)
        assert second["call"]["id"] == "toolu_2"
        await _answer(client, second["id"], "approve")
        result = await _result(client, until="decided")

    assert result["outcome"] == "proceed"
    assert [review["call"] for review in result["reviews"]] == ["toolu_1", "toolu_2"]


@pytest.mark.anyio
async def test_a_held_reply_has_the_sentinels_edits_in_it() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        first = await _asked(
            handler, client, weather("Oslo"), weather("Bergen", "toolu_2")
        )
        await _answer(client, first["id"], "approve")
        await _result(client, until="awaiting_review")
        (second,) = await _pending(client)

        await _answer(client, second["id"], "reject", "Not Bergen.")

        result = await _result(client, until="decided")
        got = await _held_reply(client)

    # the turn goes on without the rejected call, as the proxy would pass it
    assert result["outcome"] == "proceed"
    assert [block["id"] for block in got.json()["content"]] == ["toolu_1"]


# --- a request sent again reattaches, and is never judged again ---


@pytest.mark.anyio
async def test_a_request_sent_again_while_a_person_decides_is_pending_too() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        again = await handler.request(_from_eval("x2"))

        # the same question, and no second one
        assert again is PENDING
        assert [q["id"] for q in await _pending(client)] == [question["id"]]
        assert (await _result(client))["state"] == "awaiting_review"


@pytest.mark.anyio
async def test_a_request_sent_again_once_its_call_is_decided_is_refused() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve")
        decided = await _result(client, until="decided")

        again = await handler.request(_from_eval("x2"))

        # the provider is not called a second time: the result is read
        assert isinstance(again, Refuse)
        assert "has ended" in again.message and "request_result" in again.message
        assert await _result(client) == decided
        assert await _pending(client) == []


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["body", "header", "model"])
async def test_another_request_under_the_same_irid_is_refused_and_inherits_nothing(
    change: str,
) -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        before = await _result(client)
        other = _from_eval("x2")
        if change == "body":
            other = replace(other, body=other.body.replace(b"weather", b"forecast"))
        elif change == "header":
            other = replace(other, headers={**other.headers, "x-trace": "abc"})
        else:
            other = replace(other, model="claude-other")

        refused = await handler.request(other)

        # an approval is for the request the person was asked about
        assert isinstance(refused, Refuse)
        assert refused is not PENDING
        assert "not the same request" in refused.message
        assert await _result(client) == before
        assert [q["id"] for q in await _pending(client)] == [question["id"]]


@pytest.mark.anyio
async def test_a_try_signed_afresh_is_the_same_request() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)
        later = replace(
            request("x2"),
            headers=inspect_headers(irid="irid-1", timestamp=int(time.time()) + 30),
        )

        # the timestamp and mac change with each try, and aren't in the binding
        assert await handler.request(later) is PENDING


@pytest.mark.anyio
async def test_a_try_that_was_already_being_judged_reattaches_when_it_would_ask() -> (
    None
):
    handler = _reviewed()
    async with _serving(handler) as client:
        # two tries of one request reach the provider before either is asked about
        assert isinstance(await handler.request(_from_eval("x")), Pass)
        assert isinstance(await handler.request(_from_eval("x2")), Pass)
        assert await handler.reply(reply(message(weather("Oslo")))) is PENDING

        second = await handler.reply(reply(message(weather("Oslo")), id="x2"))

        # one model call, one question
        assert second is PENDING
        (question,) = await _pending(client)
        await _answer(client, question["id"], "approve")
        result = await _result(client, until="decided")
    assert [review["id"] for review in result["reviews"]] == [question["id"]]


# --- a question at a tool's result: the provider has not been called ---


@pytest.mark.anyio
async def test_a_question_at_a_tools_result_is_asked_before_the_provider_is_called() -> (
    None
):
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)

        assert question["stage"] == "tool_result"
        assert question["request_id"] == "irid-2"
        assert question["choices"] == ["approve", "terminate"]
        assert question["result"]["content"] == "Rain."
        assert (await _result(client, "irid-2"))["state"] == "awaiting_review"


@pytest.mark.anyio
async def test_an_approved_result_waits_for_the_request_to_be_sent_again() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)

        await _answer(client, question["id"], "approve")

        waiting = await _result(client, "irid-2", until="awaiting_request")
        # nothing failed: the request only waits to be sent
        assert "reason" not in waiting and "failure" not in waiting
        # the same request, under the same x-irid, is sent on, and not judged again
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        assert (await _result(client, "irid-2"))["state"] == "awaiting_reply"
        assert await _pending(client) == []
        body = message({"type": "text", "text": "It rains."}, stop_reason="end_turn")
        assert isinstance(await handler.reply(reply(body, id="y2")), Pass)
        result = await _result(client, "irid-2")
        got = await _held_reply(client, "irid-2")

    assert (result["state"], result["outcome"]) == ("decided", "proceed")
    # kept though the eval's own exchange brought the reply: it may be lost on the way
    assert got.content == body
    (review,) = result["reviews"]
    assert (review["stage"], review["resolution"]) == ("tool_result", "answered")


@pytest.mark.anyio
async def test_only_one_try_is_sent_on_to_the_provider() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        answers: list[Any] = []

        async def send(id: str) -> None:
            answers.append(await handler.request(_result_request(id)))

        async with anyio.create_task_group() as group:
            group.start_soon(send, "y2")
            group.start_soon(send, "y3")

        assert sorted(type(answer).__name__ for answer in answers) == ["Pass", "Refuse"]
        assert PENDING in answers
        # and a try sent while the provider answers is pending as well
        assert await handler.request(_result_request("y4")) is PENDING


@pytest.mark.anyio
async def test_a_reply_lost_on_its_way_to_the_eval_is_read_from_the_sidecar() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        body = message({"type": "text", "text": "It rains."}, stop_reason="end_turn")
        assert isinstance(await handler.reply(reply(body, id="y2")), Pass)

        # the eval never got the response, and sends the request again
        again = await handler.request(_result_request("y3"))

        assert isinstance(again, Refuse) and "has ended" in again.message
        got = await _held_reply(client, "irid-2")
    assert (got.status_code, got.content) == (200, body)


@pytest.mark.anyio
async def test_a_reply_the_sidecar_never_hears_of_lets_no_other_try_through() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)

        # the proxy stopped: whether the provider answered can't be told
        assert await handler.request(_result_request("y3")) is PENDING
        assert (await _result(client, "irid-2"))["state"] == "awaiting_reply"


# --- the provider is called again only after a failed exchange ---


@pytest.mark.anyio
async def test_a_providers_error_sends_the_call_back_to_wait_for_a_request() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)

        # the provider's error goes on to the try's client, as from any call
        assert isinstance(await handler.reply(_error()), Pass)

        failed = await _result(client, "irid-2")
        assert (failed["state"], failed["reason"]) == (
            "awaiting_request",
            "provider_error",
        )
        # what failed, apart from any error the eval holds: the second try
        # the sidecar saw, with what says whether it may be tried again
        assert failed["failure"] == {
            "try_number": 2,
            "reply": {
                "status": 529,
                "headers": {"retry-after": "7", "content-type": JSON},
                "body": base64.b64encode(b'{"type":"error"}').decode(),
                "format": "anthropic-messages",
                "stream": False,
            },
        }
        # one more try is sent on, and nobody is asked about the result again
        assert isinstance(await handler.request(_result_request("y3")), Pass)
        assert await _pending(client) == []
        body = message({"type": "text", "text": "It rains."}, stop_reason="end_turn")
        assert isinstance(await handler.reply(reply(body, id="y3")), Pass)
        result = await _result(client, "irid-2")

    assert (result["state"], result["outcome"]) == ("decided", "proceed")
    assert "failure" not in result and "reason" not in result
    assert len(result["reviews"]) == 1


@pytest.mark.anyio
async def test_a_failure_is_the_providers_whatever_error_the_eval_holds() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        # a try sent while the provider answers gets the sidecar's own 400
        assert await handler.request(_result_request("y3")) is PENDING

        # and then the provider's exchange fails, with an error that may be retried
        await handler.reply(_error(status=529))

        failed = await _result(client, "irid-2")
    # the failure is of the try that was sent on, and holds the provider's status
    assert failed["failure"]["try_number"] == 2
    assert failed["failure"]["reply"]["status"] == 529


@pytest.mark.anyio
async def test_after_a_failed_exchange_the_new_reply_is_judged_and_may_ask() -> None:
    handler = Handler(human(stages=["tool_call", "tool_result"]))
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        await handler.reply(_error())
        assert isinstance(await handler.request(_result_request("y3")), Pass)

        # only the request stage is kept: the reply's own call is asked about
        again = await handler.reply(
            reply(message(weather("Bergen", "toolu_2")), id="y3")
        )

        assert again is PENDING
        (asked,) = await _pending(client)
        assert (asked["stage"], asked["call"]["id"]) == ("tool_call", "toolu_2")
        await _answer(client, asked["id"], "approve")
        result = await _result(client, "irid-2", until="decided")

    assert result["outcome"] == "proceed"
    # each question with the try whose request or reply was being judged
    assert [(r["stage"], r["try_number"]) for r in result["reviews"]] == [
        ("tool_result", 1),
        ("tool_call", 3),
    ]


@pytest.mark.anyio
async def test_a_reply_cut_off_is_a_failure_without_a_reply() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)

        await handler.reply_not_judged("y2", "cut_off")

        failed = await _result(client, "irid-2")
        assert (failed["state"], failed["reason"]) == ("awaiting_request", "cut_off")
        assert failed["failure"] == {"try_number": 2}
        assert isinstance(await handler.request(_result_request("y3")), Pass)


@pytest.mark.anyio
async def test_an_errors_body_that_passed_in_parts_is_handed_over_afterwards() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        # a proxy that passes an error on as it comes shows it without its body
        await handler.reply(replace(_error(body=b""), body_follows=True))

        # the eval isn't told the provider failed while the error still passes
        passing = await _result(client, "irid-2")
        assert passing["state"] == "awaiting_reply" and "failure" not in passing
        _refused(await _held_reply(client, "irid-2"))

        await handler.error_body("y2", b'{"type":"overloaded"}')

        failed = await _result(client, "irid-2")
    assert (failed["state"], failed["reason"]) == ("awaiting_request", "provider_error")
    assert base64.b64decode(failed["failure"]["reply"]["body"]) == (
        b'{"type":"overloaded"}'
    )


@pytest.mark.anyio
@pytest.mark.parametrize("how", ["no copy", "cut off"])
async def test_an_error_whose_body_isnt_handed_over_is_kept_without_it(
    how: str,
) -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        await handler.reply(replace(_error(body=b""), body_follows=True))

        if how == "no copy":
            # one too large or compressed to copy: the proxy says it passed
            await handler.error_body("y2", b"")
        else:
            # the client went before the error's body had all passed
            await handler.reply_not_judged("y2", "cut_off")

        failed = await _result(client, "irid-2")
        got = await _held_reply(client, "irid-2")
    assert failed["state"] == "awaiting_request"
    if how == "no copy":
        # the provider's error, whole as far as the proxy says, without a copy
        assert failed["reason"] == "provider_error"
        assert failed["failure"]["reply"]["status"] == 529
        assert failed["failure"]["reply"]["body"] == ""
        assert got.status_code == 529
    else:
        # An error that didn't all pass isn't one to give the eval: what is
        # missing may say whether the call may be tried again. It failed as a
        # connection does, and there is no error to get.
        assert failed["reason"] == "cut_off"
        assert "reply" not in failed["failure"]
        assert got.status_code == 400 and FAILED_TRY not in got.headers


@pytest.mark.anyio
async def test_an_error_that_passes_after_its_call_was_cancelled_reopens_nothing() -> (
    None
):
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        await handler.reply(replace(_error(body=b""), body_follows=True))
        assert await _cancel(client, "irid-2") is True

        await handler.error_body("y2", b'{"type":"overloaded"}')

        result = await _result(client, "irid-2")
    assert (result["state"], result["reason"]) == ("not_judged", "cancelled")
    assert "failure" not in result


@pytest.mark.anyio
async def test_an_sdks_own_retry_is_the_same_request_whatever_its_retry_count() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        first = _result_request("y2")

        # shown to the sidecar where an operator lists it, and counted up by
        # the SDK on each retry of its own
        retry = replace(
            first, headers={**first.headers, "x-stainless-retry-count": "1"}
        )
        sent_on = await handler.request(retry)
        # a header a sentinel may decide on is still part of what was approved
        other = replace(
            _result_request("y3"),
            headers={**first.headers, "x-environment": "production"},
        )
        refused = await handler.request(other)

    assert isinstance(sent_on, Pass)
    assert isinstance(refused, Refuse) and "not the same request" in refused.message


# --- cancel_request closes the call wherever it stands ---


@pytest.mark.anyio
async def test_cancelling_a_call_a_person_is_asked_about_closes_its_question() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        assert await _cancel(client) is True

        result = await _result(client)
        assert (result["state"], result["reason"]) == ("not_judged", "cancelled")
        assert "outcome" not in result
        (review,) = result["reviews"]
        assert (review["resolution"], review["cause"]) == ("cancelled", "cancelled")
        assert await _pending(client) == []
        late = await _answer(client, question["id"], "approve")
        assert late["error"]["code"] == UNKNOWN_QUESTION
        # this call closed nothing, which doesn't say the call is still open
        assert await _cancel(client) is False
        assert (await _result(client))["state"] == "not_judged"
        # and nothing opens it again
        again = await handler.request(_from_eval("x2"))
        assert isinstance(again, Refuse) and "has ended" in again.message


@protocol(portable=False)
def waits_at(call: str, reached: anyio.Event, go: anyio.Event) -> Protocol:
    """A slow protocol before `human()`: at one call, it waits until it is let go on."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.id == call:
            reached.set()
            await go.wait()
        return Decision.escalate("checked by a person, after a while")

    return decide


@pytest.mark.anyio
async def test_cancelling_between_questions_stops_the_next_from_being_asked() -> None:
    reached, go = anyio.Event(), anyio.Event()
    handler = Handler(
        sequential([waits_at("toolu_2", reached, go), human(stages=["tool_call"])])
    )
    async with _serving(handler) as client:
        first = await _asked(
            handler, client, weather("Oslo"), weather("Bergen", "toolu_2")
        )
        await _answer(client, first["id"], "approve")
        # the sentinel is at work on the turn's next call
        await reached.wait()
        assert (await _result(client))["state"] == "judging"

        assert await _cancel(client) is True
        go.set()
        await anyio.sleep(0.1)

        assert await _pending(client) == []
        result = await _result(client)
    assert (result["state"], result["reason"]) == ("not_judged", "cancelled")
    assert [review["call"] for review in result["reviews"]] == ["toolu_1"]


@pytest.mark.anyio
async def test_cancelling_a_call_that_waits_for_a_request_sends_nothing_on() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")

        assert await _cancel(client, "irid-2") is True

        again = await handler.request(_result_request("y2"))
        assert isinstance(again, Refuse) and "has ended" in again.message
        result = await _result(client, "irid-2")
    assert (result["state"], result["reason"]) == ("not_judged", "cancelled")


@pytest.mark.anyio
async def test_a_reply_that_comes_after_its_call_was_cancelled_doesnt_revive_it() -> (
    None
):
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)

        # the request already with the provider can't be called back
        assert await _cancel(client, "irid-2") is True
        late = await handler.reply(
            reply(message(weather("Bergen", "toolu_2")), id="y2")
        )

        # its reply is not judged, not kept and not passed on
        assert isinstance(late, Refuse) and "was closed" in late.message
        result = await _result(client, "irid-2")
        assert (result["state"], result["reason"]) == ("not_judged", "cancelled")
        _refused(await _held_reply(client, "irid-2"))
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_cancelling_closes_only_a_call_a_person_was_asked_about() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await holding(handler, "eval_other", _OTHER_TOKEN)
        # a request no person is asked about, still with the provider
        assert isinstance(await handler.request(_from_eval("p", "irid-7")), Pass)
        await _asked(handler, client)

        assert await _cancel(client, "irid-7") is False
        assert await _cancel(client, "irid-never-sent") is False
        # another eval's call reads as unknown to it
        assert await _cancel(client, "irid-1", "eval_other", _OTHER_TOKEN) is False

        assert (await _result(client, "irid-7"))["state"] == "awaiting_reply"
        assert (await _result(client))["state"] == "awaiting_review"


# --- what ends a call before it is decided ---


@pytest.mark.anyio
async def test_ending_the_sample_closes_its_calls_and_their_questions() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        ended = await _call(
            client,
            "end_sample",
            eval_call("end_sample") | {"sample_uuid": "sample_123"},
        )

        assert ended["result"]["state"] == "ended"
        assert await _pending(client) == []
        late = await _answer(client, question["id"], "approve")
        assert late["error"]["code"] == UNKNOWN_QUESTION
        # the call reads as a request of an ended run does, with nothing
        # decided for the step that waited
        result = await _result(client)
        assert (result["state"], result["outcome"]) == ("decided", "terminate")
        (review,) = result["reviews"]
        assert (review["resolution"], review["cause"]) == ("cancelled", "sample_ended")


@pytest.mark.anyio
async def test_an_answer_sent_again_after_its_sample_ended_is_for_no_question() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, until="decided")

        await handler.end_sample("sample_123", "eval_789")

        again = await _answer(client, question["id"], "approve")
    assert again["error"]["code"] == UNKNOWN_QUESTION


@pytest.mark.anyio
async def test_a_later_attempt_at_the_sample_closes_the_earlier_ones_calls() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)

        # Inspect retries the sample: a new attempt, with a higher number
        retried = await handler.request(
            _from_eval("r", "irid-5", attempt="attempt_2", attempt_number=2)
        )

        assert isinstance(retried, Pass)
        assert await _pending(client) == []
        result = await _result(client)
    assert (result["state"], result["reason"]) == ("not_judged", "replaced_attempt")
    (review,) = result["reviews"]
    assert (review["resolution"], review["cause"]) == ("cancelled", "replaced_attempt")


@protocol
def ends_at_bergen() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Bergen":
            return Decision.terminate("No lookups of Bergen.")
        return Decision.escalate("checked by a person")

    return decide


@pytest.mark.anyio
async def test_a_run_a_sentinel_ends_closes_its_other_calls() -> None:
    handler = Handler(sequential([ends_at_bergen(), human(stages=["tool_call"])]))
    async with _serving(handler) as client:
        await _asked(handler, client)
        assert isinstance(await handler.request(_from_eval("z", "irid-9")), Pass)

        # another call of the same sample ends the run
        ended = await handler.reply(
            reply(message(weather("Bergen", "toolu_9")), id="z")
        )

        assert isinstance(ended, Refuse) and "ended this run" in ended.message
        assert await _pending(client) == []
        result = await _result(client)
    assert (result["state"], result["outcome"]) == ("decided", "terminate")
    (review,) = result["reviews"]
    assert (review["resolution"], review["cause"]) == ("cancelled", "run_ended")


@pytest.mark.anyio
async def test_a_sample_ended_before_its_first_question_is_never_asked_it() -> None:
    reached, go = anyio.Event(), anyio.Event()
    handler = Handler(
        sequential([waits_at("toolu_1", reached, go), human(stages=["tool_call"])])
    )
    answers: list[Any] = []
    async with _serving(handler) as client:
        assert isinstance(await handler.request(_from_eval("x")), Pass)

        async def judge() -> None:
            answers.append(await handler.reply(reply(message(weather("Oslo")))))

        async with anyio.create_task_group() as group:
            group.start_soon(judge)
            # the sample ends while the protocol before human() is still judging
            await reached.wait()
            assert await handler.end_sample("sample_123", "eval_789") == "ended"
            go.set()

        assert await _pending(client) == []
        found = await handler.request_result("irid-1", "eval_789")

    (answer,) = answers
    assert isinstance(answer, Refuse) and answer is not PENDING
    assert not isinstance(found, str)
    # no question was ever kept for the ended sample
    assert found.reviews == []


@pytest.mark.anyio
async def test_a_closed_question_takes_no_answer_from_the_moment_it_closes() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        await handler.end_sample("sample_123", "eval_789")

        # at once, before the judging that was waiting has run again
        with pytest.raises(UnknownQuestionError):
            handler.reviews.answer("eval_789", question["id"], "approve", None)
        assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
async def test_stopping_the_sidecar_stops_what_waits() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)
    # a restart loses a paused call: nothing waits once the handler stops serving
    assert handler.reviews._open == {}  # pyright: ignore[reportPrivateUsage]
    assert handler.eval_review is False


# --- a question's expiry, and the limits that don't bound a review ---


@pytest.mark.anyio
async def test_a_question_that_expires_is_answered_with_caution_and_judging_goes_on() -> (
    None
):
    handler = _reviewed(review_timeout=0.2)
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        assert question["closes"] - question["asked"] == pytest.approx(0.2)

        result = await _result(client, until="decided")

        assert await _pending(client) == []
    # expiry is not cancellation: the call is decided, by the cautious answer
    assert result["outcome"] == "reject"
    (review,) = result["reviews"]
    assert review["resolution"] == "timed_out"
    assert "decision" not in review


@pytest.mark.anyio
async def test_an_expired_question_where_reject_isnt_offered_ends_the_run() -> None:
    handler = _reviewed(choices=["approve", "terminate"], review_timeout=0.2)
    async with _serving(handler) as client:
        await _asked(handler, client)

        result = await _result(client, until="decided")

    assert result["outcome"] == "terminate"


@pytest.mark.anyio
async def test_the_time_a_question_is_open_isnt_counted_against_the_time_limit() -> (
    None
):
    handler = _reviewed(time_limit=0.3)
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        # the person takes longer than the sentinel may
        await anyio.sleep(0.6)

        await _answer(client, question["id"], "approve")

        result = await _result(client, until="decided")
    assert result["outcome"] == "proceed"


@pytest.mark.anyio
def _state_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # the state's own clock, which the test moves on: the event loop keeps
    # the real one
    now = [0.0]

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return now[0]

    monkeypatch.setattr(_state, "time", Clock)
    return now


async def _decided_or_dropped(client: httpx.AsyncClient) -> dict[str, Any]:
    # what `request_result` says once the call is decided, or once its
    # records are no longer kept
    with anyio.fail_after(10):
        while True:
            result = await _result(client)
            if result["records"] != "held" or result.get("state") == "decided":
                return result
            await anyio.sleep(0.01)


@pytest.mark.anyio
async def test_a_call_that_waits_for_a_person_keeps_what_it_needs_past_its_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _state_clock(monkeypatch)
    handler = _reviewed(state=MemoryState(expiry=10))
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        # longer than the run, the eval's token and the request's records are kept
        now[0] = 100.0

        # the eval's token still proves its calls, and the run is still there
        answered = await _answer(client, question["id"], "approve")
        assert answered["result"]["decision"] == "approve"
        result = await _decided_or_dropped(client)

    assert (result["records"], result["outcome"]) == ("held", "proceed")


@pytest.mark.anyio
async def test_a_paused_call_keeps_the_records_of_its_earlier_tries_past_their_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _state_clock(monkeypatch)
    handler = _reviewed(state=MemoryState(expiry=10))
    async with _serving(handler) as client:
        # a first try of the call, whose reply never came
        assert isinstance(await handler.request(_from_eval("x1")), Pass)
        await handler.reply_not_judged("x1", "cut_off")
        # the eval tries again under the same x-irid, and a person is asked
        question = await _asked(handler, client, id="x2")
        # the person takes longer than records are kept
        now[0] = 100.0

        await _answer(client, question["id"], "approve")
        result = await _decided_or_dropped(client)

    # `request_result` reads every try's records, the first's included
    assert (result["records"], result["outcome"]) == ("held", "proceed")


@pytest.mark.anyio
async def test_the_time_a_person_takes_isnt_judging_time() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        # the person takes a while, with no connection waiting on the call
        await anyio.sleep(0.5)
        await _answer(client, question["id"], "approve")
        result = await _result(client, until="decided")

    # the judging counts only until the call was answered "pending review"
    assert 0 <= result["judging_time"] < 0.5


# --- who is asked, and who may answer ---


@pytest.mark.anyio
async def test_another_eval_neither_sees_nor_answers_the_question() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await holding(handler, "eval_other", _OTHER_TOKEN)
        question = await _asked(handler, client)

        assert await _pending(client, "eval_other", _OTHER_TOKEN) == []
        intruding = await _call(
            client,
            "answer_question",
            _answer_params(
                question["id"], "approve", eval_id="eval_other", token=_OTHER_TOKEN
            ),
        )

        assert intruding["error"]["code"] == UNKNOWN_QUESTION
        # the question is still the eval's to answer
        await _answer(client, question["id"], "reject", "No.")
        assert (await _result(client, until="decided"))["outcome"] == "reject"


@pytest.mark.anyio
async def test_an_answer_is_refused_unless_its_mac_covers_the_choice_sent() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        # signed as a rejection, sent as an approval
        tampered = await _call(
            client,
            "answer_question",
            _answer_params(question["id"], "approve", signed_decision="reject"),
        )

        assert tampered["error"]["code"] == UNAUTHORIZED
        assert (await _result(client))["state"] == "awaiting_review"


@pytest.mark.anyio
async def test_the_same_answer_sent_again_is_taken_and_another_is_refused() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, until="decided")

        again = await _answer(client, question["id"], "approve")
        other = await _answer(client, question["id"], "reject", "Changed my mind.")

    assert again["result"] == {"id": question["id"], "decision": "approve"}
    assert other["error"]["code"] == QUESTION_ANSWERED


@pytest.mark.anyio
async def test_a_choice_the_question_doesnt_offer_is_refused() -> None:
    handler = _reviewed(choices=["approve", "terminate"])
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        refused = await _answer(client, question["id"], "reject")

        assert refused["error"]["code"] == ANSWER_REFUSED
        await _answer(client, question["id"], "approve")
        assert (await _result(client, until="decided"))["outcome"] == "proceed"


@pytest.mark.anyio
async def test_a_run_no_eval_names_is_answered_with_caution_and_asks_nobody() -> None:
    handler = _reviewed()
    async with _serving(handler):
        await handler.request(request())

        answer = await handler.reply(reply(message(weather("Oslo"))))

        assert isinstance(answer, Refuse)
        assert "No person can be asked from the sidecar yet." in answer.message
        assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
async def test_an_eval_that_isnt_sent_results_is_asked_nothing() -> None:
    # an eval the deployment doesn't trust with results doesn't answer for a
    # person either: the step fails closed, as for a run no eval names
    handler = _reviewed(eval_results=False)
    async with _serving(handler):
        await handler.request(_from_eval())

        answer = await handler.reply(reply(message(weather("Oslo"))))

        assert isinstance(answer, Refuse)
        assert "No person can be asked from the sidecar yet." in answer.message
        assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
async def test_a_handler_that_isnt_serving_asks_nobody_through_an_eval() -> None:
    # the judging that waits for a person has nowhere to run
    handler = await holding(_reviewed())
    assert handler.eval_review is False
    await handler.request(_from_eval())

    with anyio.fail_after(5):
        answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    assert "No person can be asked from the sidecar yet." in answer.message


@pytest.mark.anyio
@pytest.mark.parametrize(
    "method", ["pending_questions", "answer_question", "cancel_request"]
)
async def test_a_sidecar_that_sends_no_results_serves_no_review_method(
    method: str,
) -> None:
    handler = _reviewed(eval_results=False)
    params = (
        _answer_params("q1", "approve")
        if method == "answer_question"
        else eval_call(method) | {"id": "irid-1"}
    )
    async with _serving(handler, results=False) as client:
        refused = await _call(client, method, params)

    assert refused["error"]["code"] == METHOD_NOT_FOUND


@pytest.mark.anyio
async def test_pending_questions_can_be_asked_for_one_sample() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        assert await _pending(client, sample_uuid="sample_123") == [question]
        assert await _pending(client, sample_uuid="sample_456") == []


@pytest.mark.anyio
async def test_a_sidecar_holding_nothing_of_the_eval_holds_no_question() -> None:
    handler = _reviewed()
    async with handler.serving(), _client(handler) as client:
        assert await _pending(client) == []
        unknown = await _answer(client, "q1", "approve")
        assert await _cancel(client) is False

    assert unknown["error"]["code"] == UNKNOWN_QUESTION


class _Approves(SidecarHost):
    """A host with a way of its own to ask a person."""

    async def ask_human(self, step: Any, escalations: Any, choices: Any) -> HumanAnswer:
        return HumanAnswer(decision="approve", reason="The deployment's own reviewer.")


@pytest.mark.anyio
async def test_a_host_with_its_own_way_to_ask_a_person_is_still_asked() -> None:
    handler = _reviewed(host=_Approves())
    async with _serving(handler):
        await handler.request(_from_eval())

        answer = await handler.reply(reply(message(weather("Oslo"))))

        assert isinstance(answer, Pass)
        assert await handler.reviews.pending("eval_789") == []


# --- what is kept of a question ---


@pytest.mark.anyio
async def test_request_result_says_what_became_of_each_question() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve", "Looks fine.")

        result = await _result(client, until="decided")

    (review,) = result["reviews"]
    assert review["id"] == question["id"]
    assert (review["try_number"], review["stage"], review["call"]) == (
        1,
        "tool_call",
        "toolu_1",
    )
    assert (review["decision"], review["reason"], review["resolution"]) == (
        "approve",
        "Looks fine.",
        "answered",
    )
    assert review["asked"] <= review["answered"] <= review["resolved"]
    # the record holds the call's id, and nothing of what it was called with
    assert "Oslo" not in json.dumps(review)
    # the step that asked names its question
    (step,) = [s for s in result["judgment"]["steps"] if s["stage"] == "tool_call"]
    assert step["reviews"] == [question["id"]]


@pytest.mark.anyio
async def test_only_a_receipt_is_kept_of_an_answered_question() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve", "Looks fine.")
        await _result(client, until="decided")

        # no question is held once its step stops waiting, and what is kept
        # of the answer holds nothing of the call
        assert handler.reviews._open == {}  # pyright: ignore[reportPrivateUsage]
        (receipt,) = handler.reviews._receipts.values()  # pyright: ignore[reportPrivateUsage]
    assert vars(receipt) == {
        "eval_id": "eval_789",
        "sample_uuid": "sample_123",
        "decision": "approve",
        "reason": "Looks fine.",
    }


@pytest.mark.anyio
async def test_questions_are_listed_oldest_first() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        first = await _asked(handler, client)
        second = await _asked(
            handler, client, irid="irid-9", id="z", sample_uuid="sample_456"
        )

        listed = await _pending(client)

    assert [q["id"] for q in listed] == [first["id"], second["id"]]
    assert [q["sample_uuid"] for q in listed] == ["sample_123", "sample_456"]


@pytest.mark.anyio
async def test_an_answer_taken_isnt_replaced_once_its_question_closes() -> None:
    reached, go = anyio.Event(), anyio.Event()
    handler = Handler(
        sequential([waits_at("toolu_2", reached, go), human(stages=["tool_call"])])
    )
    async with _serving(handler) as client:
        question = await _asked(
            handler, client, weather("Oslo"), weather("Bergen", "toolu_2")
        )
        await _answer(client, question["id"], "approve", "Fine.")
        await reached.wait()

        # the sample ends while the turn's next call is judged, which drops
        # the answer's receipt
        await handler.end_sample("sample_123", "eval_789")

        with pytest.raises(UnknownQuestionError):
            handler.reviews.answer("eval_789", question["id"], "reject", "No.")
        go.set()
        found = await handler.request_result("irid-1", "eval_789")

    # the answer that was acknowledged is the one on the record
    assert not isinstance(found, str)
    ((_, review),) = found.reviews
    assert (review.decision, review.reason) == ("approve", "Fine.")


def test_a_replaced_reply_is_what_is_held() -> None:
    # the body the proxy passes on in the reply's place
    from inspect_sentinel.sidecar._handler import (
        _held_reply,  # pyright: ignore[reportPrivateUsage]
    )

    held = _held_reply(
        reply(b"as the provider sent it"), "anthropic-messages", Replace(b"edited")
    )

    assert held.body == b"edited"


@pytest.mark.anyio
async def test_judging_whose_proxy_goes_away_before_anyone_is_asked_asks_nobody() -> (
    None
):
    reached, go = anyio.Event(), anyio.Event()
    handler = Handler(
        sequential([waits_at("toolu_1", reached, go), human(stages=["tool_call"])])
    )
    async with _serving(handler) as client:
        assert isinstance(await handler.request(_from_eval()), Pass)

        async with anyio.create_task_group() as group:
            group.start_soon(handler.reply, reply(message(weather("Oslo"))))
            await reached.wait()
            # the proxy's own caller goes away, and its call with it
            group.cancel_scope.cancel()
        go.set()
        await anyio.sleep(0.1)

        # nobody waits for the judging, so it didn't go on to ask a person
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_the_sentinels_own_time_still_counts_after_a_person_answered() -> None:
    reached, go = anyio.Event(), anyio.Event()
    handler = Handler(
        sequential([waits_at("toolu_2", reached, go), human(stages=["tool_call"])]),
        time_limit=0.3,
    )
    async with _serving(handler) as client:
        first = await _asked(
            handler, client, weather("Oslo"), weather("Bergen", "toolu_2")
        )
        await _answer(client, first["id"], "approve")

        # the sentinel never finishes the turn's next call
        result = await _result(client, until="not_judged")

    assert result["reason"] == "time_limit"


@pytest.mark.anyio
async def test_a_proxy_is_told_to_refuse_a_call_that_waits_for_a_person() -> None:
    handler = _reviewed()
    asked = _from_eval()
    async with _serving(handler) as client:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_app(handler)),
            base_url="http://sidecar",
        ) as proxy:

            async def judge(method: str, params: dict[str, Any]) -> dict[str, Any]:
                answered = await proxy.post(
                    PROXY_PATH,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params,
                    },
                )
                result: dict[str, Any] = answered.json()["result"]
                return result

            common = {"id": "x", "provider": "anthropic", "model": "claude-test"}
            passed = await judge(
                "judge_request",
                common
                | {
                    "user": "user-1",
                    "headers": dict(asked.headers),
                    "body": json.loads(asked.body),
                },
            )
            refused = await judge(
                "judge_reply",
                common
                | {
                    "status": 200,
                    "content_type": JSON,
                    "headers": {"Request-Id": "req_011"},
                    "body": base64.b64encode(message(weather("Oslo"))).decode(),
                },
            )

        # a refusal in a status no provider SDK sends again by itself
        assert passed == {"action": "pass"}
        assert refused["action"] == "refuse" and refused["status"] == 400
        assert "waiting for a person's review" in refused["message"]
        (question,) = await _pending(client)
        await _answer(client, question["id"], "approve")
        await _result(client, until="decided")
        got = await _held_reply(client)

    # and the eval gets the reply the proxy handed over, headers and all
    assert (got.headers["request-id"], got.headers["content-type"]) == ("req_011", JSON)


# --- the held reply, which the eval gets by sending the call's request again ---

# what a provider's SDK doesn't send again by itself
_NOT_SENT_AGAIN = {400, 401, 404, 413}


async def _held_reply(
    client: httpx.AsyncClient,
    irid: str = "irid-1",
    *,
    body: bytes = b'{"model": "claude-test"}',
    **signing: Any,
) -> httpx.Response:
    # the model call's own request, sent again under its `x-irid`, as the
    # eval's code for the provider sends it
    return await client.post(
        HELD_REPLY_PATH, headers=inspect_headers(irid=irid, **signing), content=body
    )


def _refused(answered: httpx.Response, *said: str) -> None:
    assert answered.status_code == 400
    message: str = answered.json()["error"]
    assert all(part in message for part in said), message


async def _approved(handler: Handler, client: httpx.AsyncClient, held: Reply) -> None:
    # a person approves the call in `held`, and the call is decided
    assert isinstance(await handler.request(_from_eval()), Pass)
    assert await handler.reply(held) is PENDING
    (question,) = await _pending(client)
    await _answer(client, question["id"], "approve")
    await _result(client, until="decided")


async def _failed_at_the_provider(
    handler: Handler, client: httpx.AsyncClient, error: Reply
) -> None:
    # a person approved the call, its request was sent on, and the provider
    # answered with `error`
    question = await _asked_at_result(handler, client)
    await _answer(client, question["id"], "approve")
    await _result(client, "irid-2", until="awaiting_request")
    assert isinstance(await handler.request(_result_request("y2")), Pass)
    await handler.reply(error)


@pytest.mark.anyio
async def test_a_providers_error_is_served_where_the_held_reply_is() -> None:
    handler = _at_results()
    body = b'{"type":"error","error":{"type":"permission_error","message":"No."}}'
    async with _serving(handler) as client:
        await _failed_at_the_provider(
            handler,
            client,
            replace(
                _error(status=403, body=body),
                headers={
                    "content-type": JSON,
                    "request-id": "req_403",
                    # a provider can't set the sidecar's own header
                    FAILED_TRY: "99",
                    "x-should-retry": "true",
                },
            ),
        )

        got = await _held_reply(client, "irid-2")
        again = await _held_reply(client, "irid-2")
        after = await _result(client, "irid-2")

    # the provider's own status, headers and body, for the eval's code for
    # that provider to raise as the provider's error
    assert (got.status_code, got.content) == (403, body)
    assert got.headers["request-id"] == "req_403"
    # marked with the try it is of, as `request_result` numbers it
    assert got.headers[FAILED_TRY] == str(after["failure"]["try_number"]) == "2"
    # the eval's SDK isn't to send the request again by itself
    assert got.headers["x-should-retry"] == "false"
    # getting it changes nothing, so it can be got again
    assert (again.status_code, again.content) == (403, body)
    assert (after["state"], after["reason"]) == ("awaiting_request", "provider_error")


@pytest.mark.anyio
async def test_no_refusal_of_the_held_reply_path_is_marked_as_a_providers_error() -> (
    None
):
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)

        # a person still decides: the sidecar's own 400
        refused = await _held_reply(client)
        unknown = await _held_reply(client, "irid-none")

    assert (refused.status_code, unknown.status_code) == (400, 400)
    assert FAILED_TRY not in refused.headers and FAILED_TRY not in unknown.headers


@pytest.mark.anyio
async def test_a_held_reply_never_carries_the_failed_try_header() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        held = replace(
            reply(message(weather("Oslo"))),
            headers={"content-type": JSON, FAILED_TRY: "1"},
        )
        await _approved(handler, client, held)

        got = await _held_reply(client)

    assert got.status_code == 200 and FAILED_TRY not in got.headers


@pytest.mark.anyio
async def test_a_reply_cut_off_has_no_error_to_serve() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")
        assert isinstance(await handler.request(_result_request("y2")), Pass)
        await handler.reply_not_judged("y2", "cut_off")

        got = await _held_reply(client, "irid-2")

    # nothing came from the provider, so the eval sends the request again
    assert got.status_code == 400 and FAILED_TRY not in got.headers


@pytest.mark.anyio
async def test_a_wait_the_provider_asked_for_is_counted_from_when_the_error_came() -> (
    None
):
    handler = _at_results()
    async with _serving(handler) as client:
        await _failed_at_the_provider(
            handler,
            client,
            replace(
                _error(status=429),
                headers={"content-type": JSON, "retry-after": "30"},
            ),
        )

        got = await _held_reply(client, "irid-2")

    # never longer than the provider asked, and less by the time since
    assert 0 <= int(got.headers["retry-after"]) <= 30


def test_what_is_left_of_a_wait_is_what_the_provider_asked_less_the_time_since() -> (
    None
):
    assert _time_left("retry-after", "30", 12.4) == "18"
    assert _time_left("retry-after-ms", "1500", 1.0) == "500"
    # a wait that has passed is none
    assert _time_left("retry-after", "5", 60.0) == "0"
    # a date says when, whatever the time since, and no other header is a wait
    date = "Wed, 21 Oct 2026 07:28:00 GMT"
    assert _time_left("retry-after", date, 12.4) == date
    assert _time_left("x-ratelimit-reset-requests", "30", 12.4) == "30"


@pytest.mark.anyio
async def test_the_held_reply_is_answered_as_the_provider_answered() -> None:
    handler = _reviewed()
    body = message(weather("Oslo"))
    async with _serving(handler) as client:
        await _approved(
            handler,
            client,
            replace(
                reply(body),
                headers={
                    "content-type": JSON,
                    "content-length": "7",
                    "transfer-encoding": "chunked",
                    "request-id": "req_011",
                },
            ),
        )

        answered = await _held_reply(client)

    assert answered.status_code == 200
    assert answered.content == body
    # the provider's headers, and a length that is the body's own
    assert dict(answered.headers) == {
        "content-type": JSON,
        "request-id": "req_011",
        "content-length": str(len(body)),
    }


@pytest.mark.anyio
async def test_a_held_stream_is_answered_with_its_events_from_first_to_last() -> None:
    handler = _reviewed()
    events = anthropic_recorded("single-tool-call")
    async with _serving(handler) as client:
        await _approved(handler, client, reply(events, content_type=STREAM))

        answered = await _held_reply(client)

    assert answered.status_code == 200
    assert answered.headers["content-type"] == STREAM
    assert answered.content == events


@pytest.mark.anyio
async def test_the_held_reply_can_be_got_again_and_getting_it_changes_nothing() -> None:
    handler = _reviewed()
    body = message(weather("Oslo"))
    async with _serving(handler) as client:
        await _approved(handler, client, reply(body))
        decided = await _result(client)

        first = await _held_reply(client)
        second = await _held_reply(client)

        assert first.content == second.content == body
        assert await _result(client) == decided
        # and a try sent to the provider's route is still refused
        again = await handler.request(_from_eval("x2"))
        assert isinstance(again, Refuse) and "has ended" in again.message


@pytest.mark.anyio
async def test_nothing_of_the_request_but_its_proof_is_read_for_the_held_reply() -> (
    None
):
    handler = _reviewed()
    body = message(weather("Oslo"))
    async with _serving(handler) as client:
        await _approved(handler, client, reply(body))

        # another body than the call's request had, and no body at all
        other = await _held_reply(client, body=b'{"model": "another"}')
        none = await _held_reply(client, body=b"")

    assert other.content == none.content == body


@pytest.mark.anyio
async def test_a_call_a_person_still_decides_has_no_held_reply_yet() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        answered = await _held_reply(client)

        _refused(answered, "has not ended", "request_result")
        # the question is still open, and still the eval's to answer
        assert [q["id"] for q in await _pending(client)] == [question["id"]]
        assert (await _result(client))["state"] == "awaiting_review"


@pytest.mark.anyio
async def test_asking_for_a_held_reply_never_sends_a_waiting_request_on() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")

        answered = await _held_reply(client, "irid-2")

        _refused(answered, "has not ended")
        # the call still waits for its request, which is sent on as before
        assert (await _result(client, "irid-2"))["state"] == "awaiting_request"
        assert isinstance(await handler.request(_result_request("y2")), Pass)


@pytest.mark.anyio
async def test_a_rejected_call_has_no_held_reply() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "reject", "Not Oslo.")
        await _result(client, until="decided")

        answered = await _held_reply(client)

    _refused(answered, "ended without its turn going on", "request_result")


@pytest.mark.anyio
async def test_a_cancelled_call_has_no_held_reply() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)
        assert await _cancel(client) is True

        answered = await _held_reply(client)

    _refused(answered, "ended without its turn going on")


@pytest.mark.anyio
async def test_a_call_nobody_was_asked_about_has_no_held_reply() -> None:
    # its reply went to the eval with its own exchange, and none was held
    handler = Handler(human(stages=["tool_result"]))
    async with _serving(handler) as client:
        assert isinstance(await handler.request(_from_eval()), Pass)
        assert isinstance(await handler.reply(reply(message(weather("Oslo")))), Pass)

        judged = await _held_reply(client)
        never_seen = await _held_reply(client, "irid-9")

    _refused(judged, "holds no reply", "request_result")
    _refused(never_seen, "holds no reply")


@pytest.mark.anyio
async def test_a_held_reply_is_not_kept_past_its_samples_end() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _approved(handler, client, reply(message(weather("Oslo"))))
        assert (await _held_reply(client)).status_code == 200
        await _call(
            client,
            "end_sample",
            eval_call("end_sample") | {"sample_uuid": "sample_123"},
        )

        answered = await _held_reply(client)

    _refused(answered, "holds no reply")


@pytest.mark.anyio
async def test_another_eval_cant_get_the_held_reply() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await holding(handler, "eval_other", _OTHER_TOKEN)
        await _approved(handler, client, reply(message(weather("Oslo"))))

        # the call's `x-irid`, signed by another eval for a sample of its own,
        # and for the sample the call is of
        own_sample = await _held_reply(
            client,
            eval_id="eval_other",
            token=_OTHER_TOKEN,
            sample_uuid="sample_of_the_other",
        )
        this_sample = await _held_reply(
            client, eval_id="eval_other", token=_OTHER_TOKEN
        )

    _refused(own_sample, "holds no reply")
    _refused(this_sample, "names a sample of another eval")


@pytest.mark.anyio
async def test_a_held_reply_is_given_only_to_a_request_that_proves_its_call() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _approved(handler, client, reply(message(weather("Oslo"))))
        signed_for_it = inspect_headers(irid="irid-1")

        no_header = await client.post(HELD_REPLY_PATH, headers={"x-irid": "irid-1"})
        unreadable = await client.post(
            HELD_REPLY_PATH,
            headers={"x-irid": "irid-1", "x-inspect-context": "not json"},
        )
        no_irid = await client.post(
            HELD_REPLY_PATH,
            headers={"x-inspect-context": signed_for_it["x-inspect-context"]},
        )
        # a mac made for another request of the sample, and one made with
        # another key, and one made too long ago
        another_irid = await client.post(
            HELD_REPLY_PATH,
            headers=inspect_headers(irid="irid-2") | {"x-irid": "irid-1"},
        )
        another_key = await _held_reply(client, token=_OTHER_TOKEN)
        stale = await _held_reply(client, timestamp=int(time.time()) - 3600)

    _refused(no_header, "no x-inspect-context header")
    _refused(unreadable, "isn't one this sidecar can read")
    _refused(no_irid, "no x-irid header")
    _refused(another_irid, "mac isn't one made with the eval's token")
    _refused(another_key, "mac isn't one made with the eval's token")
    _refused(stale, "timestamp")


@pytest.mark.anyio
async def test_a_sidecar_that_sends_no_results_serves_no_held_reply() -> None:
    handler = _reviewed(eval_results=False)
    async with _serving(handler, results=False) as client:
        answered = await _held_reply(client)

    assert answered.status_code == 404
    assert "doesn't send results" in answered.json()["error"]


@pytest.mark.anyio
async def test_the_held_reply_needs_the_endpoints_own_credential_where_it_has_one() -> (
    None
):
    handler = _reviewed()
    credential = "the-endpoint-credential-for-these-tests"
    async with _serving(handler, token=credential) as client:
        client.headers["authorization"] = f"Bearer {credential}"
        await _approved(handler, client, reply(message(weather("Oslo"))))

        with_it = await _held_reply(client)
        del client.headers["authorization"]
        without = await _held_reply(client)

    assert with_it.status_code == 200
    assert without.status_code == 401
    assert without.headers["www-authenticate"] == "Bearer"


@pytest.mark.anyio
async def test_a_request_for_a_held_reply_larger_than_the_limit_is_not_read() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _approved(handler, client, reply(message(weather("Oslo"))))

        # the endpoint in these tests reads at most 1 MiB
        answered = await _held_reply(client, body=b" " * (1024 * 1024 + 1))

    assert answered.status_code == 413


@pytest.mark.anyio
async def test_no_refusal_of_a_held_reply_is_a_status_an_sdk_sends_again() -> None:
    # 408, 409, 429 and 5xx are what the providers' SDKs retry by themselves
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)

        statuses = {
            (await _held_reply(client)).status_code,
            (await _held_reply(client, "irid-9")).status_code,
            (await _held_reply(client, token=_OTHER_TOKEN)).status_code,
            (await client.post(HELD_REPLY_PATH)).status_code,
        }

    assert statuses <= _NOT_SENT_AGAIN
