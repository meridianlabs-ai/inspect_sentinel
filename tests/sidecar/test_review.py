import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
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
    Answer,
    Handler,
    Pass,
    Refuse,
    Request,
    SidecarHost,
)
from inspect_sentinel.sidecar._errors import TimeLimitError
from inspect_sentinel.sidecar._eval_endpoint import PATH, SCHEMA_FILE, eval_routes
from inspect_sentinel.sidecar._json_rpc import (
    ANSWER_REFUSED,
    METHOD_NOT_FOUND,
    QUESTION_ANSWERED,
    UNAUTHORIZED,
    UNKNOWN_QUESTION,
)
from inspect_sentinel.sidecar._review import Question, UnknownQuestionError
from tests.sidecar._traffic import (
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
async def test_an_eval_that_isnt_sent_results_is_asked_nothing() -> None:
    # an eval the deployment doesn't trust with results doesn't answer for a
    # person either: the step fails closed, as for a run no eval names
    handler = await holding(_reviewed(eval_results=False))
    await handler.request(_from_eval())

    with anyio.fail_after(5):
        answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    assert "No person can be asked from the sidecar yet." in answer.message
    assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["pending_questions", "answer_question"])
async def test_a_sidecar_that_sends_no_results_serves_no_review_method(
    method: str,
) -> None:
    handler = await holding(_reviewed(eval_results=False))
    params = (
        eval_call(method)
        if method == "pending_questions"
        else _answer_params("q1", "approve")
    )
    async with _client(handler, results=False) as client:
        refused = await _call(client, method, params)

    assert refused["error"]["code"] == METHOD_NOT_FOUND


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


class _Approves(SidecarHost):
    """A host with a way of its own to ask a person."""

    async def ask_human(self, step: Any, escalations: Any, choices: Any) -> HumanAnswer:
        return HumanAnswer(decision="approve", reason="The deployment's own reviewer.")


@pytest.mark.anyio
async def test_a_host_with_its_own_way_to_ask_a_person_is_still_asked() -> None:
    handler = await holding(_reviewed(host=_Approves()))
    await handler.request(_from_eval())

    with anyio.fail_after(5):
        answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Pass)
    assert await handler.reviews.pending("eval_789") == []


async def _open(
    handler: Handler, *, besides: tuple[str, ...] = (), count: int = 1
) -> list[Question]:
    # the eval's open questions other than `besides`, once there are `count`
    with anyio.fail_after(5):
        while True:
            found = [
                question
                for question in await handler.reviews.pending("eval_789")
                if question.id not in besides
            ]
            if len(found) >= count:
                return found
            await anyio.sleep(0.01)


async def _result(client: httpx.AsyncClient, irid: str = "irid-1") -> dict[str, Any]:
    answered = await _call(
        client, "request_result", eval_call("request_result") | {"id": irid}
    )
    result: dict[str, Any] = answered["result"]
    return result


class _Replies:
    """Replies judged at once, each in a task of its own, as a proxy hands them over."""

    def __init__(self, handler: Handler, group: Any) -> None:
        self._handler, self._group = handler, group
        self.answers: dict[str, Answer | BaseException] = {}

    def judge(self, id: str, *content: dict[str, Any]) -> None:
        async def judging() -> None:
            try:
                self.answers[id] = await self._handler.reply(
                    reply(message(*(content or (weather("Oslo"),))), id=id)
                )
            except Exception as ex:
                self.answers[id] = ex

        self._group.start_soon(judging)

    def request(self, held: Request) -> None:
        async def judging() -> None:
            self.answers[held.id] = await self._handler.request(held)

        self._group.start_soon(judging)

    def stop(self) -> None:
        """Stop judging, as when the proxy's own caller goes away."""
        self._group.cancel_scope.cancel()


@asynccontextmanager
async def _judging(handler: Handler) -> AsyncGenerator[_Replies]:
    # the block ends once every reply and request it began has been answered
    async with anyio.create_task_group() as group:
        yield _Replies(handler, group)


def _refused(answer: Answer | BaseException | None, why: str) -> bool:
    return isinstance(answer, Refuse) and why in answer.message


@pytest.mark.anyio
async def test_a_question_at_a_tools_result_is_asked_and_answered() -> None:
    handler = await holding(Handler(human(stages=["tool_result"])))
    await handler.request(_from_eval("x"))
    assert isinstance(await handler.reply(reply(message(weather("Oslo")))), Pass)

    async with _client(handler) as client:
        with anyio.fail_after(10):
            async with _judging(handler) as replies:
                replies.request(
                    replace(
                        request("y", then=ran("Oslo", "Rain.")),
                        headers=inspect_headers(irid="irid-2"),
                    )
                )
                (question,) = await _pending(client, wait=5)
                await _answer(client, question["id"], "approve")

    assert question["stage"] == "tool_result"
    assert question["request_id"] == "irid-2"
    assert question["choices"] == ["approve", "terminate"]
    assert question["result"]["content"] == "Rain."
    assert isinstance(replies.answers["y"], Pass)


@pytest.mark.anyio
async def test_a_later_try_stops_the_earlier_one_with_nothing_decided() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        with anyio.fail_after(10):
            async with _judging(handler) as replies:
                await handler.request(_from_eval("x"))
                replies.judge("x")
                (first,) = await _open(handler)
                # the client gives up on its first try, and sends the request again
                await handler.request(_from_eval("y"))
                replies.judge("y")
                (second,) = await _open(handler, besides=(first.id,))
                # the first try's question is gone, and can't be answered
                assert [q["id"] for q in await _pending(client)] == [second.id]
                late = await _answer(client, first.id, "approve")
                assert late["error"]["code"] == UNKNOWN_QUESTION
                await _answer(client, second.id, "approve", "Fine.")
        result = await _result(client)

    # only the first exchange is refused: no rejection, and the run goes on
    assert _refused(replies.answers["x"], "A later try of this request replaced")
    assert isinstance(replies.answers["y"], Pass)
    assert (result["state"], result["outcome"]) == ("decided", "proceed")
    assert [
        (r["try_number"], r["resolution"], r.get("cause"), r.get("decision"))
        for r in result["reviews"]
    ] == [(1, "cancelled", "superseded", None), (2, "answered", None, "approve")]


@pytest.mark.anyio
async def test_a_later_try_is_asked_about_the_reply_it_got() -> None:
    handler = await holding(_reviewed())
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (first,) = await _open(handler)
            await handler.request(_from_eval("y"))
            # the provider answers the later try with another call
            replies.judge("y", weather("Bergen", "toolu_2"))
            (second,) = await _open(handler, besides=(first.id,))
            handler.reviews.answer("eval_789", second.id, "approve", None)

    assert (second.call.id, second.call.arguments) == (
        "toolu_2",
        {"location": "Bergen"},
    )
    assert isinstance(replies.answers["y"], Pass)


@pytest.mark.anyio
async def test_a_later_try_at_a_tools_result_doesnt_end_the_run() -> None:
    # the cautious answer at a result is terminate, which would end the run
    # for the later try too: the earlier try is stopped with nothing decided
    handler = await holding(Handler(human(stages=["tool_result"])))
    await handler.request(_from_eval("x"))
    await handler.reply(reply(message(weather("Oslo"))))

    def later(id: str) -> Request:
        return replace(
            request(id, then=ran("Oslo", "Rain.")),
            headers=inspect_headers(irid="irid-2"),
        )

    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            replies.request(later("y1"))
            (first,) = await _open(handler)
            replies.request(later("y2"))
            (second,) = await _open(handler, besides=(first.id,))
            handler.reviews.answer("eval_789", second.id, "approve", None)

    assert _refused(replies.answers["y1"], "A later try of this request replaced")
    assert isinstance(replies.answers["y2"], Pass)


@pytest.mark.anyio
async def test_a_reply_that_arrives_after_a_later_try_began_asks_nobody() -> None:
    handler = await holding(_reviewed())
    await handler.request(_from_eval("x"))
    # the later try begins before the first try's reply arrives
    await handler.request(_from_eval("y"))

    with anyio.fail_after(5):
        late = await handler.reply(reply(message(weather("Oslo")), id="x"))

    assert _refused(late, "A later try of this request replaced")
    assert await handler.reviews.pending("eval_789") == []
    found = await handler.request_result("irid-1", "eval_789")
    assert not isinstance(found, str)
    assert found.reviews == []
    # the later try is still asked, and judged
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            replies.judge("y")
            (question,) = await _open(handler)
            handler.reviews.answer("eval_789", question.id, "approve", None)
    assert isinstance(replies.answers["y"], Pass)


@pytest.mark.anyio
async def test_an_answer_taken_before_its_try_is_stopped_stays_on_the_record() -> None:
    handler = await holding(_reviewed())
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)
            handler.reviews.answer("eval_789", question.id, "approve", "Fine.")
            # a later try begins before the step is given the answer
            await handler.request(_from_eval("y"))
            found = await handler.request_result("irid-1", "eval_789")
            replies.stop()

    assert not isinstance(found, str)
    (number, review), *_ = found.reviews
    assert (number, review.decision, review.reason) == (1, "approve", "Fine.")
    assert review.answered is not None


@pytest.mark.anyio
async def test_ending_the_sample_closes_its_questions() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        with anyio.fail_after(10):
            async with _judging(handler) as replies:
                await handler.request(_from_eval("x"))
                replies.judge("x")
                (question,) = await _open(handler)
                assert await handler.end_sample("sample_123", "eval_789") == "ended"
        late = await _answer(client, question.id, "approve")

    assert _refused(replies.answers["x"], "This run ended while the request waited")
    assert late["error"]["code"] == UNKNOWN_QUESTION
    assert await handler.reviews.pending("eval_789") == []
    assert (question.review.resolution, question.review.cause) == (
        "cancelled",
        "sample_ended",
    )


@pytest.mark.anyio
async def test_an_answer_sent_again_after_its_sample_ended_is_for_no_question() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        asked: list[str] = []

        async def approve() -> None:
            (question,) = await _pending(client, wait=5)
            asked.append(question["id"])
            await _answer(client, question["id"], "approve")

        await _replied_while(handler, approve)
        assert "result" in await _answer(client, asked[0], "approve")

        await handler.end_sample("sample_123", "eval_789")
        again = await _answer(client, asked[0], "approve")

    assert again["error"]["code"] == UNKNOWN_QUESTION


@pytest.mark.anyio
async def test_a_later_attempt_at_the_sample_closes_the_earlier_ones_questions() -> (
    None
):
    handler = await holding(_reviewed())
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)
            await handler.request(
                _from_eval("y", irid="irid-2", attempt="attempt_2", attempt_number=2)
            )

    assert _refused(replies.answers["x"], "A later attempt at this sample replaced")
    assert (question.review.resolution, question.review.cause) == (
        "cancelled",
        "replaced_attempt",
    )
    found = await handler.request_result("irid-1", "eval_789")
    assert not isinstance(found, str)
    assert (found.latest.state, found.latest.reason) == (
        "not_judged",
        "replaced_attempt",
    )


@protocol
def storms_end_the_run() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Storm":
            return Decision.terminate("a storm")
        return Decision.escalate("every other lookup is checked by a person")

    return decide


@pytest.mark.anyio
async def test_a_run_a_sentinel_ends_closes_its_other_questions() -> None:
    handler = await holding(
        Handler(sequential([storms_end_the_run(), human(stages=["tool_call"])]))
    )
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)
            # another request of the run, whose reply ends it
            await handler.request(_from_eval("z", irid="irid-2"))
            replies.judge("z", weather("Storm", "toolu_9"))

    assert _refused(replies.answers["z"], "A sentinel ended this run")
    assert _refused(replies.answers["x"], "This run ended while the request waited")
    assert (question.review.resolution, question.review.cause) == (
        "cancelled",
        "run_ended",
    )
    # the eval reads the stopped request as one of an ended run, with a
    # judgment it can act on: without one it would take the sidecar's answer
    # as unavailable
    async with _client(handler) as client:
        result = await _result(client)
    assert (result["state"], result["outcome"]) == ("decided", "terminate")
    assert result["judgment"]["outcome"] == "terminate"
    # nothing was decided at the call that was waiting
    assert [s for s in result["judgment"]["steps"] if s["stage"] == "tool_call"] == []
    assert [(r["resolution"], r["cause"]) for r in result["reviews"]] == [
        ("cancelled", "run_ended")
    ]


@pytest.mark.anyio
async def test_a_time_limit_shorter_than_the_review_timeout_closes_sooner() -> None:
    handler = await holding(_reviewed(time_limit=0.3))
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)

    # the question said when it would really close, not when the review
    # timeout would have passed
    assert 0 < question.closes - question.asked <= 0.3
    assert isinstance(replies.answers["x"], TimeLimitError)
    assert (question.review.resolution, question.review.cause) == (
        "cancelled",
        "time_limit",
    )
    assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
async def test_a_reply_the_proxy_stops_judging_closes_its_question() -> None:
    handler = await holding(_reviewed())
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)
            replies.stop()

    assert (question.review.resolution, question.review.cause) == (
        "cancelled",
        "stopped",
    )
    assert await handler.reviews.pending("eval_789") == []


@pytest.mark.anyio
async def test_request_result_says_what_became_of_each_question() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        asked: list[str] = []

        async def approve() -> None:
            (question,) = await _pending(client, wait=5)
            asked.append(question["id"])
            await _answer(client, question["id"], "approve", "Looks fine.")

        await _replied_while(handler, approve)
        result = await _result(client)

    (review,) = result["reviews"]
    assert review["id"] == asked[0]
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
    assert step["reviews"] == asked


@pytest.mark.anyio
async def test_request_result_tells_a_timeout_from_an_answer() -> None:
    handler = await holding(_reviewed(review_timeout=0.2))
    async with _client(handler) as client:

        async def wait_it_out() -> None:
            await _pending(client, wait=5)

        await _replied_while(handler, wait_it_out)
        result = await _result(client)

    (review,) = result["reviews"]
    assert review["resolution"] == "timed_out"
    assert "decision" not in review or review["decision"] is None
    assert result["outcome"] == "reject"


@pytest.mark.anyio
async def test_only_a_receipt_is_kept_of_an_answered_question() -> None:
    handler = await holding(_reviewed())
    async with _client(handler) as client:

        async def approve() -> None:
            (question,) = await _pending(client, wait=5)
            await _answer(client, question["id"], "approve", "Looks fine.")

        await _replied_while(handler, approve)

    # no question is held once its step stops waiting, and what is kept of
    # the answer holds nothing of the call
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
    handler = await holding(_reviewed())
    async with _client(handler) as client:
        with anyio.fail_after(10):
            async with _judging(handler) as replies:
                await handler.request(_from_eval("x"))
                replies.judge("x")
                (first,) = await _open(handler)
                await handler.request(
                    _from_eval("z", irid="irid-9", sample_uuid="sample_456")
                )
                replies.judge("z")
                await _open(handler, count=2)
                listed = await _pending(client)
                for question in listed:
                    await _answer(client, question["id"], "approve")

    assert [q["sample_uuid"] for q in listed] == ["sample_123", "sample_456"]
    assert listed[0]["id"] == first.id


@protocol(portable=False)
def waits_then_escalates(reached: anyio.Event, go: anyio.Event) -> Protocol:
    """A slow protocol before `human()`: it escalates once it is let go on."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        reached.set()
        await go.wait()
        return Decision.escalate("checked by a person, after a while")

    return decide


@pytest.mark.anyio
async def test_a_sample_ended_before_its_first_question_is_never_asked_it() -> None:
    reached, go = anyio.Event(), anyio.Event()
    handler = await holding(
        Handler(
            sequential([waits_then_escalates(reached, go), human(stages=["tool_call"])])
        )
    )
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            # the sample ends while the protocol before human() is still judging
            await reached.wait()
            assert await handler.end_sample("sample_123", "eval_789") == "ended"
            go.set()

    assert _refused(replies.answers["x"], "This run ended while the request waited")
    assert await handler.reviews.pending("eval_789") == []
    found = await handler.request_result("irid-1", "eval_789")
    assert not isinstance(found, str)
    # no question was ever kept for the ended sample
    assert found.reviews == []


@pytest.mark.anyio
async def test_a_closed_question_takes_no_answer_from_the_moment_it_closes() -> None:
    handler = await holding(_reviewed())
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)
            await handler.end_sample("sample_123", "eval_789")
            # at once, before the step that was waiting has run again
            with pytest.raises(UnknownQuestionError):
                handler.reviews.answer("eval_789", question.id, "approve", None)
            assert await handler.reviews.pending("eval_789") == []

    assert question.review.decision is None
    assert _refused(replies.answers["x"], "This run ended while the request waited")


@pytest.mark.anyio
async def test_an_answer_taken_isnt_replaced_once_its_question_closes() -> None:
    handler = await holding(_reviewed())
    with anyio.fail_after(10):
        async with _judging(handler) as replies:
            await handler.request(_from_eval("x"))
            replies.judge("x")
            (question,) = await _open(handler)
            handler.reviews.answer("eval_789", question.id, "approve", "Fine.")
            # the sample ends before the step is given the answer, which
            # drops the answer's receipt
            await handler.end_sample("sample_123", "eval_789")
            with pytest.raises(UnknownQuestionError):
                handler.reviews.answer("eval_789", question.id, "reject", "No.")
            replies.stop()

    # the answer that was acknowledged is the one on the record
    assert (question.review.decision, question.review.reason) == ("approve", "Fine.")
