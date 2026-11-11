import json
import time
from collections.abc import Awaitable, Callable
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
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Answer, Handler, Pass, Refuse, Request
from inspect_sentinel.sidecar._eval_endpoint import PATH, SCHEMA_FILE, eval_routes
from inspect_sentinel.sidecar._json_rpc import (
    ANSWER_REFUSED,
    QUESTION_ANSWERED,
    UNAUTHORIZED,
    UNKNOWN_QUESTION,
)
from tests.sidecar._traffic import (
    TOKEN,
    eval_call,
    holding,
    inspect_headers,
    message,
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


def _from_eval(id: str = "x", irid: str = "irid-1", **sample: Any) -> Request:
    # a request an Inspect eval sends, signed, with its id for the request
    return replace(request(id), headers=inspect_headers(irid=irid, **sample))


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


async def _replied_while(
    handler: Handler, act: Callable[[], Awaitable[None]], *, irid: str = "irid-1"
) -> Answer:
    # the eval's request goes on, and its reply is judged, which waits on a
    # person, while `act` does what the eval would
    await handler.request(_from_eval(irid=irid))
    answers: list[Answer] = []

    async def judge() -> None:
        answers.append(await handler.reply(reply(message(weather("Oslo")))))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(judge)
            await act()
    return answers[0]


@pytest.mark.anyio
async def test_a_person_who_approves_lets_the_call_through() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:

        async def approve() -> None:
            (question,) = await _pending(client, wait=5)
            answered = await _answer(client, question["id"], "approve", "Looks fine.")
            assert answered["result"] == {"id": question["id"], "decision": "approve"}

        answer = await _replied_while(handler, approve)

        assert isinstance(answer, Pass)
        # an answered question is no longer pending
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_a_question_says_what_the_person_decides_about() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        seen: list[dict[str, Any]] = []

        async def look() -> None:
            seen.extend(await _pending(client, wait=5))
            await _answer(client, seen[0]["id"], "approve")

        await _replied_while(handler, look, irid="irid-42")

    (question,) = seen
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
    assert question["closes"] - question["asked"] == pytest.approx(600)


@pytest.mark.anyio
async def test_a_person_who_rejects_has_the_model_told_their_reason() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:

        async def reject() -> None:
            (question,) = await _pending(client, wait=5)
            await _answer(client, question["id"], "reject", "Not Oslo today.")

        answer = await _replied_while(handler, reject)

    assert isinstance(answer, Refuse)
    assert "Not Oslo today." in answer.message


@pytest.mark.anyio
async def test_a_person_who_terminates_ends_the_run() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:

        async def terminate() -> None:
            (question,) = await _pending(client, wait=5)
            await _answer(client, question["id"], "terminate", "Stop here.")

        answer = await _replied_while(handler, terminate)

    assert isinstance(answer, Refuse)
    assert "ended this run" in answer.message
    # every later request of the run is refused
    later = await handler.request(_from_eval(id="y", irid="irid-2"))
    assert isinstance(later, Refuse)


@pytest.mark.anyio
async def test_nobody_answering_in_time_fails_closed_and_closes_the_question() -> None:
    handler = await holding(_reviewed(review_timeout=0.2))
    async with _client(handler) as client:
        seen: list[dict[str, Any]] = []

        async def wait_it_out() -> None:
            seen.extend(await _pending(client, wait=5))

        answer = await _replied_while(handler, wait_it_out)

        assert isinstance(answer, Refuse)
        assert "Nobody answered within 0.2 seconds" in answer.message
        # an answer after the step stopped waiting is for no question
        late = await _answer(client, seen[0]["id"], "approve")
        assert late["error"]["code"] == UNKNOWN_QUESTION
        assert await _pending(client) == []


@pytest.mark.anyio
async def test_nobody_answering_where_reject_isnt_offered_ends_the_run() -> None:
    handler = await holding(
        _reviewed(choices=["approve", "terminate"], review_timeout=0.2)
    )
    async with _client(handler) as client:

        async def wait_it_out() -> None:
            await _pending(client, wait=5)

        answer = await _replied_while(handler, wait_it_out)

    assert isinstance(answer, Refuse)
    assert "ended this run" in answer.message


@pytest.mark.anyio
async def test_another_eval_neither_sees_nor_answers_the_question() -> None:
    handler = await holding(_reviewed())
    await holding(handler, "eval_other", _OTHER_TOKEN)
    async with _client(handler) as client:

        async def intrude_then_approve() -> None:
            (question,) = await _pending(client, wait=5)
            assert await _pending(client, "eval_other", _OTHER_TOKEN) == []
            intruding = await _call(
                client,
                "answer_question",
                _answer_params(
                    question["id"],
                    "approve",
                    eval_id="eval_other",
                    token=_OTHER_TOKEN,
                ),
            )
            assert intruding["error"]["code"] == UNKNOWN_QUESTION
            # the question is still the eval's to answer
            await _answer(client, question["id"], "reject", "No.")

        answer = await _replied_while(handler, intrude_then_approve)

    assert isinstance(answer, Refuse)


@pytest.mark.anyio
async def test_an_answer_is_refused_unless_its_mac_covers_the_choice_sent() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:

        async def tamper_then_reject() -> None:
            (question,) = await _pending(client, wait=5)
            # signed as a rejection, sent as an approval
            tampered = await _call(
                client,
                "answer_question",
                _answer_params(question["id"], "approve", signed_decision="reject"),
            )
            assert tampered["error"]["code"] == UNAUTHORIZED
            await _answer(client, question["id"], "reject", "No.")

        answer = await _replied_while(handler, tamper_then_reject)

    assert isinstance(answer, Refuse)


@pytest.mark.anyio
async def test_the_same_answer_sent_again_is_taken_and_another_is_refused() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        asked: list[str] = []

        async def approve() -> None:
            (question,) = await _pending(client, wait=5)
            asked.append(question["id"])
            await _answer(client, question["id"], "approve")

        await _replied_while(handler, approve)

        again = await _answer(client, asked[0], "approve")
        assert again["result"] == {"id": asked[0], "decision": "approve"}
        other = await _answer(client, asked[0], "reject", "Changed my mind.")
        assert other["error"]["code"] == QUESTION_ANSWERED


@pytest.mark.anyio
async def test_a_choice_the_question_doesnt_offer_is_refused() -> None:
    handler = await holding(_reviewed(choices=["approve", "terminate"]))
    async with _client(handler) as client:

        async def try_reject() -> None:
            (question,) = await _pending(client, wait=5)
            refused = await _answer(client, question["id"], "reject")
            assert refused["error"]["code"] == ANSWER_REFUSED
            await _answer(client, question["id"], "approve")

        answer = await _replied_while(handler, try_reject)

    assert isinstance(answer, Pass)


@pytest.mark.anyio
async def test_a_run_no_eval_names_is_answered_with_caution_and_asks_nobody() -> None:
    handler = await holding(_reviewed())
    await handler.request(request())

    with anyio.fail_after(5):
        answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    assert "No person can be asked from the sidecar yet." in answer.message
    assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
async def test_a_question_holds_no_explanation_where_results_arent_sent() -> None:
    handler = await holding(_reviewed(eval_results=False))
    async with _client(handler) as client:
        seen: list[dict[str, Any]] = []

        async def look() -> None:
            seen.extend(await _pending(client, wait=5))
            await _answer(client, seen[0]["id"], "approve")

        await _replied_while(handler, look)

    assert seen[0]["escalations"] == [{"name": "escalates"}]


@pytest.mark.anyio
async def test_pending_questions_can_be_asked_for_one_sample() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:

        async def look_then_approve() -> None:
            (question,) = await _pending(client, wait=5, sample_uuid="sample_123")
            assert await _pending(client, sample_uuid="sample_456") == []
            await _answer(client, question["id"], "approve")

        answer = await _replied_while(handler, look_then_approve)

    assert isinstance(answer, Pass)


@pytest.mark.anyio
async def test_a_sidecar_holding_nothing_of_the_eval_holds_no_question() -> None:
    handler = _reviewed()
    async with _client(handler) as client:
        assert await _pending(client) == []
        unknown = await _answer(client, "q1", "approve")

    assert unknown["error"]["code"] == UNKNOWN_QUESTION
