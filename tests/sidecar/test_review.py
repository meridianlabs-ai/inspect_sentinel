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
    Respond,
    SidecarHost,
)
from inspect_sentinel.sidecar._eval_endpoint import PATH, SCHEMA_FILE, eval_routes
from inspect_sentinel.sidecar._handler import PENDING
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
    TOKEN,
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


def _held_for(irid: str, id: str = "x3") -> Request:
    # the eval's request sent again under an id of its own, for the reply the
    # sidecar held for the call `irid`
    return replace(
        request(id), headers=inspect_headers(irid=f"{irid}-again", held_reply_for=irid)
    )


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
        assert "outcome" not in result and "reply" not in result
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
        # the provider's reply, for the eval to read as the provider's own
        assert result["reply"] == {
            "status": 200,
            "headers": {"content-type": JSON},
            "body": base64.b64encode(body).decode(),
            "format": "anthropic-messages",
            "stream": False,
        }
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_a_kept_reply_has_the_providers_headers_without_those_of_one_hop() -> (
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

        result = await _result(client, until="decided")

    assert result["reply"]["headers"] == {
        "content-type": JSON,
        "request-id": "req_011",
        "anthropic-ratelimit-requests-remaining": "49",
    }


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
    # a refused turn has no reply for the eval to read
    assert "reply" not in result


@pytest.mark.anyio
async def test_a_person_who_terminates_ends_the_run() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        await _answer(client, question["id"], "terminate", "Stop here.")

        result = await _result(client, until="decided")
        assert result["outcome"] == "terminate"
        assert "reply" not in result
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
async def test_a_kept_reply_has_the_sentinels_edits_in_it() -> None:
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

    # the turn goes on without the rejected call, as the proxy would pass it
    assert result["outcome"] == "proceed"
    passed = json.loads(base64.b64decode(result["reply"]["body"]))
    assert [block["id"] for block in passed["content"]] == ["toolu_1"]


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
async def test_a_request_sent_again_once_its_call_is_approved_gets_the_held_reply() -> (
    None
):
    handler = _reviewed()
    body = message(weather("Oslo"))
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve")
        decided = await _result(client, until="decided")

        again = await handler.request(_from_eval("x2"))

        # the reply the provider gave the first time: it is not called again
        assert again == Respond(200, {"content-type": JSON}, body)
        assert await _result(client) == decided
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_a_request_for_a_held_reply_under_a_new_irid_gets_it() -> None:
    handler = _reviewed()
    body = message(weather("Oslo"))
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, until="decided")

        # Inspect gives each request it sends an id of its own, and names the
        # call whose held reply it is for in its header
        held = await handler.request(_held_for("irid-1"))

        assert held == Respond(200, {"content-type": JSON}, body)


@pytest.mark.anyio
async def test_a_request_sent_again_once_its_call_is_rejected_is_refused() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "reject", "Not Oslo.")
        await _result(client, until="decided")

        again = await handler.request(_held_for("irid-1"))

    # a refused turn has no reply to hand back: the eval reads the judgment
    assert isinstance(again, Refuse)
    assert "has ended" in again.message and "request_result" in again.message


@pytest.mark.anyio
async def test_a_request_for_a_held_reply_of_an_unknown_call_is_refused() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        await _asked(handler, client)

        refused = await handler.request(_held_for("irid-never-sent"))

    assert isinstance(refused, Refuse)
    assert "holds nothing for" in refused.message


@pytest.mark.anyio
async def test_held_reply_for_is_signed_with_the_request() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, until="decided")
        signed = _held_for("irid-1")
        context = json.loads(signed.headers["x-inspect-context"])
        # signed for another call's held reply, and sent for this one
        context["held_reply_for"] = "irid-other"
        forged = replace(
            signed,
            headers={**signed.headers, "x-inspect-context": json.dumps(context)},
        )

        refused = await handler.request(forged)

    assert isinstance(refused, Refuse) and "mac" in refused.message


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

    assert (result["state"], result["outcome"]) == ("decided", "proceed")
    # kept though the eval's own exchange brought the reply: it may be lost on the way
    assert base64.b64decode(result["reply"]["body"]) == body
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

        # and is handed the reply the provider gave, which isn't asked again
        assert isinstance(again, Respond) and again.body == body
        result = await _result(client, "irid-2")
    assert base64.b64decode(result["reply"]["body"]) == body


@pytest.mark.anyio
async def test_a_request_sent_for_a_held_call_at_a_tools_result_goes_on() -> None:
    handler = _at_results()
    async with _serving(handler) as client:
        question = await _asked_at_result(handler, client)
        await _answer(client, question["id"], "approve")
        await _result(client, "irid-2", until="awaiting_request")

        # sent under an id of its own, naming the call it is for
        sent = await handler.request(
            replace(
                request("y2", then=ran("Oslo", "Rain.")),
                headers=inspect_headers(irid="irid-2b", held_reply_for="irid-2"),
            )
        )

        assert isinstance(sent, Pass)
        assert (await _result(client, "irid-2"))["state"] == "awaiting_reply"


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
        await handler.reply(_error(body=b""))

        await handler.error_body("y2", b'{"type":"overloaded"}')

        failed = await _result(client, "irid-2")
    assert base64.b64decode(failed["failure"]["reply"]["body"]) == (
        b'{"type":"overloaded"}'
    )


# --- cancel_request closes the call wherever it stands ---


@pytest.mark.anyio
async def test_cancelling_a_call_a_person_is_asked_about_closes_its_question() -> None:
    handler = _reviewed()
    async with _serving(handler) as client:
        question = await _asked(handler, client)

        assert await _cancel(client) is True

        result = await _result(client)
        assert (result["state"], result["reason"]) == ("not_judged", "cancelled")
        assert "outcome" not in result and "reply" not in result
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
        assert "reply" not in result
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
async def test_a_call_that_waits_for_a_person_keeps_what_it_needs_past_its_expiry() -> (
    None
):
    # long enough for the question to be asked before anything expires
    handler = _reviewed(state=MemoryState(expiry=1))
    async with _serving(handler) as client:
        question = await _asked(handler, client)
        # longer than the run, the eval's token and the request's records are kept
        await anyio.sleep(1.3)

        # the eval's token still proves its calls, and the run is still there
        answered = await _answer(client, question["id"], "approve")
        assert answered["result"]["decision"] == "approve"
        result = await _result(client, until="decided")

    assert (result["records"], result["outcome"]) == ("held", "proceed")


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


def test_a_replaced_reply_is_what_is_kept() -> None:
    # the body the proxy passes on in the reply's place
    from inspect_sentinel.sidecar._handler import (
        _kept_reply,  # pyright: ignore[reportPrivateUsage]
    )

    kept = _kept_reply(
        reply(b"as the provider sent it"), "anthropic-messages", Replace(b"edited")
    )

    assert kept.body == b"edited"


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
    assert "reply" not in result


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
        result = await _result(client, until="decided")

    # and the eval reads the reply the proxy handed over, headers and all
    assert result["reply"]["headers"] == {
        "request-id": "req_011",
        "content-type": JSON,
    }
