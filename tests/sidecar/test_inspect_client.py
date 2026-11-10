import json
import time
from dataclasses import replace
from typing import Any

import anyio
import pytest
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context, InspectSample, ProxyContext
from inspect_sentinel._decorators import protocol
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    UnreadableError,
    _sign,
    _state,
    read_reply,
)
from inspect_sentinel.sidecar._handler import _ENDED, RequestResult, SampleResult
from inspect_sentinel.sidecar._host import ListRecorder, SidecarHost
from inspect_sentinel.sidecar._inspect_client import judgment, withheld_turn
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import (
    JSON,
    SAMPLE,
    STREAM,
    anthropic_recorded,
    every_call_rule,
    holding,
    inspect_headers,
    message,
    openai_recorded,
    openai_request,
    ran,
    reply,
    request,
    storm_rule,
    weather,
    weather_rule,
)


def _from_inspect(asked: Request, irid: str | None = None, **changes: Any) -> Request:
    headers = inspect_headers(**({"irid": irid} if irid else {}), **changes)
    return replace(asked, headers=headers)


async def _not_judged(handler: Handler, irid: str) -> tuple[str, str | None, bool]:
    # what `request_result` says of a request: its state, why it wasn't
    # judged, and whether it has a judgment
    found = await handler.request_result(irid, "eval_789")
    assert not isinstance(found, str)
    latest = found.latest
    return (
        latest.state,
        latest.reason,
        latest.judged is not None and latest.outcome is not None,
    )


# each calls get_weather for Paris, then Tokyo, then Lima
_EXCHANGES = [
    "anthropic",
    "anthropic stream",
    *(
        f"openai {name}"
        for name in (
            "chat",
            "chat stream",
            "responses",
            "responses stream",
        )
    ),
]


def _exchange(name: str, id: str = "x") -> tuple[Request, Reply]:
    if name.startswith("anthropic"):
        if name.endswith("stream"):
            content_type, body = STREAM, anthropic_recorded("parallel-tool-calls")
        else:
            content_type, body = (
                JSON,
                message(
                    weather("Paris", "toolu_1"),
                    weather("Tokyo", "toolu_2"),
                    weather("Lima", "toolu_3"),
                ),
            )
        return _from_inspect(request(id)), reply(body, id=id, content_type=content_type)
    recorded = name.removeprefix("openai ")
    content_type, body = openai_recorded(recorded)
    asked = Request(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        body=json.dumps(openai_request(recorded.split()[0])).encode(),
        headers=inspect_headers(),
    )
    answered = Reply(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        status=200,
        content_type=content_type,
        body=body,
    )
    return asked, answered


async def _judgments(
    handler: Handler, irid: str = "request_456"
) -> list[dict[str, Any]]:
    # what an eval is told of each attempt at the request, as `request_result`
    # writes the latest's
    attempts = await handler._state.attempts(irid, "eval_789")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(attempts, list)
    return [
        judgment(
            attempt.judged, attempt.outcome, trimmed=not handler.eval_results
        ).model_dump(mode="json", exclude_none=True)
        for attempt in attempts
        if attempt.judged is not None and attempt.outcome is not None
    ]


async def _judged(handler: Handler) -> dict[str, Any]:
    # what an eval is told of the latest attempt
    return (await _judgments(handler))[-1]


def _stages(field: dict[str, Any]) -> list[tuple[str, str]]:
    return [(step["stage"], step["decision"]["action"]) for step in field["steps"]]


@pytest.mark.anyio
@pytest.mark.parametrize("name", _EXCHANGES)
async def test_a_reply_that_goes_on_is_judged_at_every_step(
    name: str,
) -> None:
    handler = await holding(Handler(weather_rule("reject", "Cairo")))
    asked, answered = _exchange(name)
    assert await handler.request(asked) == Pass()

    answer = await handler.reply(answered)

    # handed on as the provider sent it, as to any client
    assert answer == Pass()
    field = await _judged(handler)
    assert (field["version"], field["outcome"]) == (1, "proceed")
    assert "withheld" not in field
    assert _stages(field) == [
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
    ]
    calls = (
        await read_reply(answered.provider, answered.content_type, answered.body)
    ).message.tool_calls or []
    assert [step.get("call") for step in field["steps"]] == [
        *(call.id for call in calls),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("name", _EXCHANGES)
async def test_a_refused_turn_is_told_redacted_with_what_was_recorded(
    name: str,
) -> None:
    asked, answered = _exchange(name)
    output = await read_reply(answered.provider, answered.content_type, answered.body)
    calls = output.message.tool_calls or []
    # the recorded streams name the cities in more than one way
    cities = [call.arguments["location"] for call in calls]
    handler = await holding(Handler(every_call_rule()))
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    field = await _judged(handler)
    assert field["outcome"] == "reject"
    assert _stages(field) == [("tool_call", "reject")] * 3
    # the whole turn is withheld, every call in it, with what the model is told
    # of each when the sidecar adds it to the next request; each call was
    # rejected, so none of their arguments reaches the eval
    withheld = field["withheld"]
    assert [
        (call["id"], call["function"], call["arguments"])
        for call in withheld["message"]["tool_calls"]
    ] == [(call.id, "get_weather", {}) for call in calls]
    assert withheld["redacted"] == [call.id for call in calls]
    assert withheld["message"]["role"] == "assistant"
    assert withheld["results"] == {
        call.id: f"{city} is not allowed."
        for call, city in zip(calls, cities, strict=True)
    }
    rejected = field["steps"][1]
    assert rejected["call"] == calls[1].id
    assert rejected["decision"] == {
        "action": "reject",
        "explanation": f"lookup of {cities[1]}",
        "message": f"{cities[1]} is not allowed.",
    }
    assert [
        (record["kind"], record["path"], record["name"], record["call"])
        for record in rejected["recorded"]
    ] == [("record", "", "every_call_rule", rejected["call"])]
    assert rejected["recorded"][0]["report"] == rejected["decision"]
    # no step carries a call as the model proposed it
    assert all("original" not in step for step in field["steps"])
    # what the provider billed for the reply, which the refusal can't carry
    assert output.usage is not None
    assert field["usage"] == output.usage.model_dump(mode="json", exclude_none=True)


@pytest.mark.anyio
@pytest.mark.parametrize("name", _EXCHANGES)
async def test_a_turn_with_a_call_taken_out_is_told_with_the_call_redacted(
    name: str,
) -> None:
    asked, answered = _exchange(name)
    output = await read_reply(answered.provider, answered.content_type, answered.body)
    calls = output.message.tool_calls or []
    city = calls[1].arguments["location"]
    handler = await holding(Handler(weather_rule("reject", city)))
    await handler.request(asked)

    answer = await handler.reply(answered)

    # the turn goes on without the rejected call
    assert isinstance(answer, Replace)
    handed = await read_reply(answered.provider, answered.content_type, answer.body)
    assert [call.id for call in handed.message.tool_calls or []] == [
        calls[0].id,
        calls[2].id,
    ]
    field = await _judged(handler)
    assert field["outcome"] == "proceed"
    assert _stages(field) == [
        ("tool_call", "proceed"),
        ("tool_call", "reject"),
        ("tool_call", "proceed"),
    ]
    assert "original" not in field["steps"][1]
    # the turn for the eval to put back what was taken out, which it holds
    # without the call's arguments
    proposed = field["proposed"]
    assert [
        (call["id"], call["arguments"]) for call in proposed["message"]["tool_calls"]
    ] == [
        (calls[0].id, calls[0].arguments),
        (calls[1].id, {}),
        (calls[2].id, calls[2].arguments),
    ]
    assert proposed["redacted"] == [calls[1].id]
    assert proposed["results"] == {calls[1].id: f"{city} is not allowed."}
    assert "withheld" not in field
    # the reply that goes on carries its own usage
    assert "usage" not in field


@pytest.mark.anyio
async def test_a_run_ended_at_a_call_is_the_last_step() -> None:
    handler = await holding(Handler(weather_rule("terminate", "Tokyo")))
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    field = await _judged(handler)
    assert field["outcome"] == "terminate"
    assert _stages(field)[-1] == ("tool_call", "terminate")
    # the turn the run ended at, which the client records; none of it ran,
    # and the call it ended at is redacted
    withheld = field["withheld"]
    assert [
        (call["id"], call["arguments"]) for call in withheld["message"]["tool_calls"]
    ] == [
        ("toolu_1", {"location": "Paris"}),
        ("toolu_2", {}),
        ("toolu_3", {"location": "Lima"}),
    ]
    assert withheld["redacted"] == ["toolu_2"]
    assert set(withheld["results"].values()) == {_ENDED}
    output = await read_reply(answered.provider, answered.content_type, answered.body)
    assert output.usage is not None
    assert field["usage"]["input_tokens"] == output.usage.input_tokens


@protocol
def acts_on_paris_and_ends_at_tokyo(action: str = "reject") -> Protocol:
    """Reject or change a lookup of Paris, end the run at one of Tokyo, and let the rest go."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        city = step.call.arguments.get("location")
        if city == "Tokyo":
            return Decision.terminate("a lookup of Tokyo")
        if city != "Paris":
            return Decision.proceed()
        if action == "reject":
            return Decision.reject("a lookup of Paris", message="Paris is not allowed.")
        changed = replace(step.call, arguments={"location": "Oslo"})
        return Decision(action="modify", modified=changed)

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["reject", "modify"])
async def test_a_run_ended_at_a_call_redacts_the_calls_decided_before_it(
    action: str,
) -> None:
    handler = await holding(Handler(acts_on_paris_and_ends_at_tokyo(action)))
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    field = await _judged(handler)
    assert field["outcome"] == "terminate"
    # judging stopped at the call the run ended at: the third was never judged
    assert _stages(field) == [("tool_call", action), ("tool_call", "terminate")]
    # none of the turn ran, so the call decided before the end is redacted
    # too, whether rejected or changed, and the call never judged keeps its
    # arguments
    withheld = field["withheld"]
    assert [
        (call["id"], call["arguments"]) for call in withheld["message"]["tool_calls"]
    ] == [
        ("toolu_1", {}),
        ("toolu_2", {}),
        ("toolu_3", {"location": "Lima"}),
    ]
    assert withheld["redacted"] == ["toolu_1", "toolu_2"]
    assert set(withheld["results"].values()) == {_ENDED}


@protocol
def ends_at_the_reply() -> Protocol:
    """End the run at the model's reply."""

    async def decide(
        context: Context, step: AfterGenerate, escalations: Escalations
    ) -> Decision | None:
        return Decision.terminate("the reply")

    return decide


@pytest.mark.anyio
async def test_a_run_ended_at_the_reply_carries_the_turn() -> None:
    handler = await holding(Handler(ends_at_the_reply()))
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    field = await _judged(handler)
    assert field["outcome"] == "terminate"
    assert _stages(field) == [("model_output", "terminate")]
    # the whole turn was judged, so every call in it is redacted
    withheld = field["withheld"]
    assert [call["arguments"] for call in withheld["message"]["tool_calls"]] == [{}] * 3
    assert len(withheld["redacted"]) == 3
    assert set(withheld["results"].values()) == {_ENDED}


@pytest.mark.anyio
async def test_the_rejection_that_ends_the_run_says_the_run_ended() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    answers: list[Any] = []
    for id in ("a", "b", "c"):
        await handler.request(_from_inspect(request(id)))
        answers.append(await handler.reply(reply(message(weather("Tokyo")), id=id)))

    judgments = await _judgments(handler)
    assert all(isinstance(answer, Refuse) for answer in answers)
    assert [field["outcome"] for field in judgments] == [
        "reject",
        "reject",
        "terminate",
    ]
    # the step itself was a rejection, the third in a row
    assert _stages(judgments[-1])[-1] == ("tool_call", "reject")


@pytest.mark.anyio
async def test_a_run_ended_at_a_result_is_refused_with_the_result_step() -> None:
    handler = await holding(Handler(storm_rule()))
    asked = _from_inspect(request(then=ran("Oslo", "A storm.")))

    answer = await handler.request(asked)

    assert isinstance(answer, Refuse)
    assert await _judged(handler) == {
        "version": 1,
        "outcome": "terminate",
        "steps": [
            {
                "stage": "tool_result",
                "call": "toolu_1",
                "decision": {"action": "terminate", "explanation": "a storm"},
                "recorded": [
                    {
                        "kind": "record",
                        "path": "",
                        "factory": "storm_rule",
                        "name": "storm_rule",
                        "function": "decide",
                        "report": {"action": "terminate", "explanation": "a storm"},
                        "call": "toolu_1",
                    }
                ],
            }
        ],
    }


@protocol
def ends_before_the_model() -> ProtocolGroup:
    async def result(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed()

    async def before(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        return Decision.terminate("never")

    return ProtocolGroup(result, before)


@protocol
def watches_every_stage() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_reply_is_told_with_the_request_steps_before_its_own() -> None:
    handler = await holding(Handler(watches_every_stage()))
    asked, answered = _exchange("anthropic")
    await handler.request(replace(asked, body=request(then=ran("Oslo", "Sunny.")).body))

    answer = await handler.reply(answered)

    assert answer == Pass()
    assert _stages(await _judged(handler)) == [
        ("tool_result", "proceed"),
        ("model_input", "proceed"),
        ("model_output", "proceed"),
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
    ]


@pytest.mark.anyio
async def test_a_run_ended_before_the_model_is_refused_with_the_request_steps() -> None:
    handler = await holding(Handler(ends_before_the_model()))

    answer = await handler.request(_from_inspect(request(then=ran("Oslo", "Sunny."))))

    assert isinstance(answer, Refuse)
    assert _stages(await _judged(handler)) == [
        ("tool_result", "proceed"),
        ("model_input", "terminate"),
    ]


@pytest.mark.anyio
async def test_a_request_of_an_ended_run_is_refused_with_no_steps() -> None:
    handler = await holding(Handler(storm_rule()))
    await handler.request(_from_inspect(request("a", then=ran("Oslo", "A storm."))))

    answer = await handler.request(_from_inspect(request("b", "Something else")))

    assert isinstance(answer, Refuse)
    # the sample is the run, however its conversations open
    assert (
        answer.message
        == "A sentinel ended this run at the result of the call to get_weather."
    )
    assert await _judged(handler) == {"version": 1, "outcome": "terminate", "steps": []}


@pytest.mark.anyio
async def test_each_sample_is_a_run_of_its_own() -> None:
    handler = await holding(Handler(storm_rule()))
    await handler.request(_from_inspect(request("a", then=ran("Oslo", "A storm."))))

    # another epoch of the same sample opens the same way
    answer = await handler.request(
        _from_inspect(request("b"), epoch=3, sample_uuid="sample_124")
    )

    assert answer == Pass()


def _turn(location: str, id: str, told: str) -> list[dict[str, Any]]:
    # a rejected Anthropic turn put back in a conversation, as its client may
    return [
        {"role": "assistant", "content": [weather(location, id)]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": told,
                    "is_error": True,
                }
            ],
        },
    ]


@pytest.mark.anyio
async def test_a_client_that_put_a_withheld_turn_back_is_not_told_of_it_again() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_9")), id="a"))

    # the eval put the turn back itself, from the field
    answer = await handler.request(
        _from_inspect(
            request("b", then=_turn("Tokyo", "toolu_9", "Tokyo is not allowed."))
        )
    )

    assert answer == Pass()


@pytest.mark.anyio
async def test_a_client_that_didnt_put_a_withheld_turn_back_is_told_of_it() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_9")), id="a"))

    answer = await handler.request(_from_inspect(request("b")))

    assert isinstance(answer, Replace)
    roles = [entry["role"] for entry in json.loads(answer.body)["messages"]]
    assert roles == ["user", "assistant", "user"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "value",
    [
        "not json",
        json.dumps([SAMPLE]),
        json.dumps(SAMPLE | {"version": 2}),
        json.dumps({k: v for k, v in SAMPLE.items() if k != "sample_uuid"}),
        json.dumps(SAMPLE | {"epoch": "2"}),
        json.dumps({k: v for k, v in SAMPLE.items() if k != "attempt_number"}),
    ],
)
async def test_a_header_that_cant_be_read_is_refused(value: str) -> None:
    # refused, not left unjudged, which a proxy set to fail open would let go
    answer = await Handler(weather_rule()).request(
        replace(request(), headers={"x-inspect-context": value})
    )

    assert answer == Refuse(
        400, "The request's x-inspect-context header isn't one this sidecar can read."
    )


@pytest.mark.anyio
async def test_a_new_attempt_is_not_told_of_a_turn_rejected_in_the_last() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_9")), id="a"))

    # a retry of the request within the attempt is told of it
    retried = await handler.request(_from_inspect(request("b")))
    # Inspect's retry of the sample opens as the attempt it replaces did
    afresh = await handler.request(
        _from_inspect(request("c"), attempt="attempt_2", attempt_number=2)
    )

    assert isinstance(retried, Replace)
    assert afresh == Pass()


@pytest.mark.anyio
async def test_a_new_attempt_counts_its_rejected_turns_in_a_row_afresh() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))

    async def rejected(id: str, attempt: int) -> Refuse:
        await handler.request(
            _from_inspect(
                request(id), attempt=f"attempt_{attempt}", attempt_number=attempt
            )
        )
        answer = await handler.reply(
            reply(message(weather("Tokyo", f"toolu_{id}")), id=id)
        )
        assert isinstance(answer, Refuse)
        return answer

    await rejected("a", 1)
    await rejected("b", 1)
    third = await rejected("c", 2)

    # the third rejected turn of the sample, but the first of its attempt
    assert third.message == (
        "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )


class _Lookups(StoreModel):
    count: int = 0


@protocol
def counts_lookups_and_ends_at_storms() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        if isinstance(step, AfterToolCall) and "storm" in step.result.text:
            return Decision.terminate("a storm")
        if isinstance(step, BeforeToolCall):
            lookups = context.store_as(_Lookups)
            lookups.count += 1
            return Decision.reject("counted", message=f"Lookup {lookups.count}.")
        return None

    return decide


@pytest.mark.anyio
async def test_a_new_attempt_starts_the_runs_store_and_its_end_afresh() -> None:
    handler = await holding(Handler(counts_lookups_and_ends_at_storms()))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Oslo", "toolu_1")), id="a"))

    # each attempt has a store of its own, as the sample store is
    await handler.request(
        _from_inspect(request("b"), attempt="attempt_2", attempt_number=2)
    )
    counted = await handler.reply(reply(message(weather("Oslo", "toolu_2")), id="b"))
    # and a run a sentinel ended is ended for that attempt only
    await handler.request(
        _from_inspect(
            request("c", then=ran("Oslo", "A storm.")),
            attempt="attempt_2",
            attempt_number=2,
        )
    )
    ended = await handler.sample_result("sample_123", "eval_789")
    afresh = await handler.request(
        _from_inspect(request("d"), attempt="attempt_3", attempt_number=3)
    )
    latest = await handler.sample_result("sample_123", "eval_789")

    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."
    assert ended is not None
    assert ended.outcome == "terminate"
    assert afresh == Pass()
    assert latest == SampleResult("proceed", {})


@pytest.mark.anyio
async def test_a_replaced_attempts_requests_and_replies_are_refused_unjudged() -> None:
    handler = await holding(Handler(counts_lookups_and_ends_at_storms()))
    await handler.request(_from_inspect(request("a"), "irid-a"))
    await handler.request(
        _from_inspect(request("b"), "irid-b", attempt="attempt_2", attempt_number=2)
    )

    # the first attempt's reply arrives after its retry began
    late = await handler.reply(reply(message(weather("Oslo", "toolu_1")), id="a"))
    # and a request of the first attempt doesn't start it again
    again = await handler.request(_from_inspect(request("c"), "irid-c"))
    counted = await handler.reply(reply(message(weather("Lima", "toolu_2")), id="b"))

    replaced = "A later attempt at this sample replaced this one."
    assert isinstance(late, Refuse)
    assert late.message == replaced
    assert isinstance(again, Refuse)
    assert again.message == replaced
    # the late reply was never judged, so it counted nothing
    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."
    # the eval is told neither was judged, never that a sentinel ended them
    unjudged = ("not_judged", "replaced_attempt", False)
    assert await _not_judged(handler, "irid-a") == unjudged
    assert await _not_judged(handler, "irid-c") == unjudged


# set by the test that waits in a judgment
_GATE: dict[str, anyio.Event] = {}


@protocol
def waits_at_oslo_and_ends_at_paris() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        location = step.call.arguments.get("location")
        if location == "Oslo":
            _GATE["started"].set()
            await _GATE["release"].wait()
            return None
        lookups = context.store_as(_Lookups)
        lookups.count += 1
        if location == "Paris":
            return Decision.terminate("a storm in Paris")
        return Decision.reject("counted", message=f"Lookup {lookups.count}.")

    return decide


@pytest.mark.anyio
async def test_a_judgment_that_outlives_its_attempt_leaves_the_next_alone() -> None:
    _GATE.update(started=anyio.Event(), release=anyio.Event())
    handler = await holding(Handler(waits_at_oslo_and_ends_at_paris()))
    await handler.request(_from_inspect(request("a")))
    answers: dict[str, object] = {}

    async def judged_slowly() -> None:
        # Paris is judged once the next attempt has begun
        turn = message(weather("Oslo", "toolu_1"), weather("Paris", "toolu_3"))
        answers["a"] = await handler.reply(reply(turn, id="a"))

    # the retry begins while the first attempt's reply is still being judged
    async with anyio.create_task_group() as group:
        group.start_soon(judged_slowly)
        await _GATE["started"].wait()
        await handler.request(
            _from_inspect(request("b"), attempt="attempt_2", attempt_number=2)
        )
        _GATE["release"].set()
    counted = await handler.reply(reply(message(weather("Lima", "toolu_2")), id="b"))
    result = await handler.sample_result("sample_123", "eval_789")

    # its count and its end stay with the attempt it was part of
    assert isinstance(answers["a"], Refuse)
    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."
    assert result is not None
    assert result.outcome == "proceed"


class _SlowRecorders(MemoryState):
    """State whose second lookup of the first request's recorder waits, as a shared backend's may: the request has begun its attempt, and not yet been held."""

    def __init__(self) -> None:
        super().__init__()
        self._looked_up = 0

    async def recorder(self, id: str) -> ListRecorder:
        if id == "a":
            self._looked_up += 1
            if self._looked_up == 2:
                _GATE["started"].set()
                await _GATE["release"].wait()
        return await super().recorder(id)


@protocol
def counts_requests() -> Protocol:
    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        context.store_as(_Lookups).count += 1
        return None

    return decide


@pytest.mark.anyio
async def test_a_request_held_up_while_a_retry_begins_keeps_its_attempts_store() -> (
    None
):
    _GATE.update(started=anyio.Event(), release=anyio.Event())
    handler = await holding(Handler(counts_requests(), state=_SlowRecorders()))

    # the retry begins while the first attempt's request waits on the state
    async with anyio.create_task_group() as group:
        group.start_soon(handler.request, _from_inspect(request("a")))
        await _GATE["started"].wait()
        await handler.request(
            _from_inspect(request("b"), attempt="attempt_2", attempt_number=2)
        )
        _GATE["release"].set()
    result = await handler.sample_result("sample_123", "eval_789")

    # only the retry's own request is counted in its store
    assert result is not None
    assert result.store["_Lookups::count"] == 1


@pytest.mark.anyio
async def test_an_earlier_attempt_first_seen_after_a_later_one_is_refused() -> None:
    handler = await holding(Handler(counts_lookups_and_ends_at_storms()))
    await handler.request(
        _from_inspect(request("b"), attempt="attempt_2", attempt_number=2)
    )

    # the first attempt's only request reaches the sidecar after its retry's
    late = await handler.request(_from_inspect(request("a")))
    counted = await handler.reply(reply(message(weather("Lima", "toolu_2")), id="b"))

    assert isinstance(late, Refuse)
    assert late.message == "A later attempt at this sample replaced this one."
    # the retry goes on, its store its own
    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."


@pytest.mark.anyio
async def test_two_attempts_with_the_same_number_are_told_apart_by_refusal() -> None:
    handler = await holding(Handler(counts_lookups_and_ends_at_storms()))
    await handler.request(_from_inspect(request("a"), "irid-a"))

    other = await handler.request(
        _from_inspect(request("b"), "irid-b", attempt="attempt_other", attempt_number=1)
    )
    counted = await handler.reply(reply(message(weather("Lima", "toolu_1")), id="a"))

    assert isinstance(other, Refuse)
    assert other.message == "Another attempt at this sample has the number 1."
    assert await _not_judged(handler, "irid-b") == (
        "not_judged",
        "replaced_attempt",
        False,
    )
    # the attempt first seen with the number goes on
    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."


@pytest.mark.anyio
async def test_a_sample_the_eval_ended_stays_ended_for_a_new_attempt() -> None:
    handler = await holding(Handler(counts_lookups_and_ends_at_storms()))
    await handler.request(_from_inspect(request("a")))
    await handler.end_sample("sample_123", "eval_789")

    answer = await handler.request(
        _from_inspect(request("b"), attempt="attempt_2", attempt_number=2)
    )

    assert isinstance(answer, Refuse)
    assert answer.message == "This run has ended."


@protocol
def reports_the_proxy() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        proxy = context.proxy
        assert proxy is not None
        return Decision.reject("proxy", message=repr(proxy))

    return decide


@pytest.mark.anyio
async def test_sentinel_is_given_what_the_proxy_and_the_header_say() -> None:
    handler = await holding(Handler(concurrent({"reports": reports_the_proxy()})))
    # a sample id outside ASCII arrives escaped, as a header value must be ASCII
    asked = _from_inspect(request(), sample_id="Zürich")
    await handler.request(asked)

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    expected = ProxyContext(
        provider="anthropic",
        model="claude-test",
        user="user-1",
        headers=asked.headers,
        inspect=InspectSample(
            eval_id="eval_789",
            task="philadelphia",
            sample_id="Zürich",
            epoch=2,
            sample_uuid="sample_123",
            attempt="attempt_1",
            attempt_number=1,
        ),
    )
    # given to a monitor or protocol at any depth
    assert answer.message.endswith(f": {expected!r}")


@pytest.mark.anyio
async def test_a_client_without_the_header_is_handed_the_reply_as_it_was() -> None:
    handler = await holding(Handler(weather_rule("reject", "Cairo")))
    asked, answered = _exchange("anthropic stream")
    await handler.request(replace(asked, headers={}))

    assert await handler.reply(answered) == Pass()


@pytest.mark.anyio
async def test_a_client_without_the_header_is_refused_as_an_evals_would_be() -> None:
    handler = Handler(weather_rule())
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Tokyo"))))

    assert answer == Refuse(
        400, "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )


@pytest.mark.anyio
async def test_a_provider_error_goes_on_as_it_was() -> None:
    handler = await holding(Handler(weather_rule()))
    await handler.request(_from_inspect(request()))

    assert await handler.reply(reply(b"overloaded", status=529)) == Pass()


@pytest.mark.anyio
async def test_without_eval_results_only_what_inspect_acts_on_is_sent() -> None:
    handler = await holding(Handler(every_call_rule(), eval_results=False))
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    field = await _judged(handler)
    assert field["trimmed"] is True
    assert all("recorded" not in step for step in field["steps"])
    rejected = field["steps"][1]
    assert rejected["decision"] == {
        "action": "reject",
        "message": "Tokyo is not allowed.",
    }
    assert "original" not in rejected


@protocol
def asks_models() -> Protocol:
    # a protocol that asks its role's model, then a model it names, at each call
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        if not isinstance(step, BeforeToolCall):
            return None
        await context.host.generate("Is this call suspicious?")
        await context.host.generate("Is it?", model="mockllm/named")
        await context.host.generate("Again?")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("eval_results", [True, False], ids=["sent", "trimmed"])
async def test_each_step_says_which_models_judged_it(eval_results: bool) -> None:
    # the sidecar's roles, not the eval's: so the eval can tell which judged
    handler = await holding(
        Handler(
            asks_models(),
            host=SidecarHost({"monitor": "mockllm/monitor"}),
            eval_results=eval_results,
        )
    )
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert answer == Pass()
    steps = (await _judged(handler))["steps"]
    assert [step["stage"] for step in steps].count("tool_call") == 3
    for step in steps:
        if eval_results and step["stage"] == "tool_call":
            assert step["models"] == [
                {"role": "monitor", "model": "mockllm/monitor"},
                {"model": "mockllm/named"},
            ]
        else:
            assert "models" not in step


@pytest.mark.anyio
async def test_a_stream_without_its_last_event_is_judged_and_goes_on() -> None:
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    sent = anthropic_recorded("single-tool-call")
    # a stream whose last event never came, which is read all the same
    cut = b"".join(
        event + b"\n\n"
        for event in sent.split(b"\n\n")
        if event and b"message_delta" not in event
    )
    await handler.request(_from_inspect(request("a")))

    answer = await handler.reply(reply(cut, id="a", content_type=STREAM))
    # the turn was passed, so the conversation goes on from it
    after = await handler.request(
        _from_inspect(request("b", then=ran("Paris", "Sunny.")))
    )

    assert answer == Pass()
    assert after == Pass()
    assert await handler.records("a") is not None


@protocol
def rejects_tokyo_and_ends_at_its_result() -> ProtocolGroup:
    """Reject a lookup of Tokyo, and end the run if a result for one is ever judged, as a call that never ran has none."""

    async def before(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.terminate("a result for Tokyo")
        return Decision.proceed()

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_rejection_the_client_put_back_is_not_judged_as_a_result() -> None:
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")), id="a")
    )

    # the eval put the turn back as the model proposed it, from the field,
    # with the rejection as inspect_ai writes it
    restored = [
        {
            "role": "assistant",
            "content": [weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "Tokyo is not allowed.",
                    "is_error": True,
                },
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Sunny."},
            ],
        },
    ]
    answer = await handler.request(_from_inspect(request("b", then=restored)))

    # nothing to add, and no result for Tokyo was judged
    assert answer == Pass()


@pytest.mark.anyio
async def test_a_result_for_a_call_taken_out_is_refused() -> None:
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")), id="a")
    )

    # a client that claims to have run the call it was never handed
    ran_anyway = [
        {
            "role": "assistant",
            "content": [weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Rain."},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Sunny."},
            ],
        },
    ]

    with pytest.raises(UnreadableError, match="never ran"):
        await handler.request(_from_inspect(request("b", then=ran_anyway)))


@protocol
def changes_tokyo_and_ends_at_its_result() -> ProtocolGroup:
    """Change a lookup of Tokyo to Oslo, and end the run if a result is ever judged as Tokyo's, as only the lookup of Oslo ran."""

    async def before(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            oslo = replace(step.call, arguments={"location": "Oslo"})
            return Decision(action="modify", modified=oslo)
        return Decision.proceed()

    async def after(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.terminate("a result judged as Tokyo's")
        return Decision.proceed()

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_changed_calls_result_is_judged_as_the_call_that_ran() -> None:
    handler = await holding(Handler(changes_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    answer = await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    assert isinstance(answer, Replace)
    field = await _judged(handler)
    assert "original" not in field["steps"][0]
    assert field["steps"][0]["decision"]["modified"]["arguments"] == {
        "location": "Oslo"
    }
    # the eval holds the call as it ran, never the model's arguments
    (held,) = field["proposed"]["message"]["tool_calls"]
    assert held["arguments"] == {"location": "Oslo"}

    # the eval keeps the model's call in its conversation, as in its own
    # process, with the result of the call that ran
    answer = await handler.request(
        _from_inspect(request("b", then=ran("Tokyo", "Snow.", "toolu_1")))
    )

    # nothing to put back, and the result was judged as Oslo's
    assert answer == Pass()


def _refused_turn_put_back(told: str) -> list[dict[str, Any]]:
    # a refused turn as an eval puts it back from the field
    return [
        {"role": "assistant", "content": [weather("Tokyo", "toolu_1")]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": told,
                    "is_error": True,
                }
            ],
        },
    ]


@pytest.mark.anyio
async def test_a_refused_turns_rejection_the_client_put_back_is_not_judged() -> None:
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    refused = await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    assert isinstance(refused, Refuse)

    answer = await handler.request(
        _from_inspect(
            request("b", then=_refused_turn_put_back("Tokyo is not allowed."))
        )
    )

    # nothing to add, and no result for Tokyo was judged
    assert answer == Pass()


@pytest.mark.anyio
async def test_another_result_for_a_refused_call_is_unreadable() -> None:
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))

    with pytest.raises(UnreadableError, match="refused turn"):
        await handler.request(
            _from_inspect(request("b", then=_refused_turn_put_back("Rain.")))
        )


@pytest.mark.anyio
async def test_a_changed_call_the_eval_holds_as_it_ran_goes_to_the_model_as_made() -> (
    None
):
    handler = await holding(Handler(changes_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))

    answer = await handler.request(
        _from_inspect(request("b", then=ran("Oslo", "Snow.", "toolu_1")))
    )

    # the model is sent its own call, and the result was judged as Oslo's
    assert isinstance(answer, Replace)
    assert _calls_sent(answer.body) == [("toolu_1", {"location": "Tokyo"})]


def _redacted(id: str) -> dict[str, Any]:
    # a call as an eval holds one a decision acted on
    return {"type": "tool_use", "id": id, "name": "get_weather", "input": {}}


def _calls_sent(body: bytes) -> list[tuple[str, Any]]:
    # the calls in the request the model is sent
    return [
        (block["id"], block["input"])
        for entry in json.loads(body)["messages"]
        if entry["role"] == "assistant" and isinstance(entry["content"], list)
        for block in entry["content"]
        if block.get("type") == "tool_use"
    ]


@pytest.mark.anyio
async def test_a_call_taken_out_that_the_eval_holds_redacted_goes_back_as_made() -> (
    None
):
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")), id="a")
    )

    held = [
        {
            "role": "assistant",
            "content": [_redacted("toolu_1"), weather("Paris", "toolu_2")],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "Tokyo is not allowed.",
                    "is_error": True,
                },
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Sunny."},
            ],
        },
    ]
    answer = await handler.request(_from_inspect(request("b", then=held)))

    # the model is sent the call it made, and no result for Tokyo was judged
    assert isinstance(answer, Replace)
    assert _calls_sent(answer.body) == [
        ("toolu_1", {"location": "Tokyo"}),
        ("toolu_2", {"location": "Paris"}),
    ]


def _openai_from_inspect(
    api: str, id: str, then: list[dict[str, Any]] | None = None
) -> Request:
    # an OpenAI request from the Inspect sample, its conversation followed by `then`
    return replace(
        request(id),
        provider="openai",
        model="gpt-test",
        body=json.dumps(openai_request(api, then or [])).encode(),
        headers=inspect_headers(),
    )


def _openai_held(
    api: str, calls: list[dict[str, Any]], written: dict[str, tuple[Any, ...]]
) -> list[dict[str, Any]]:
    # the turn as the eval holds it in the request's format: each call as
    # Inspect was handed it, then its result, the sidecar's own rejection for
    # a call taken out
    def result(call: dict[str, Any]) -> list[dict[str, Any]]:
        if call["id"] in written:
            return list(written[call["id"]])
        if api == "chat":
            return [{"role": "tool", "tool_call_id": call["id"], "content": "Sunny."}]
        return [
            {"type": "function_call_output", "call_id": call["id"], "output": "Sunny."}
        ]

    if api == "chat":
        turn: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in calls
                ],
            }
        ]
    else:
        turn = [
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": call["function"],
                "arguments": json.dumps(call["arguments"]),
            }
            for call in calls
        ]
    return turn + [entry for call in calls for entry in result(call)]


def _openai_calls_sent(api: str, body: bytes) -> list[tuple[str, Any]]:
    # the calls in the request the model is sent, each with its arguments read
    sent = json.loads(body)
    if api == "chat":
        return [
            (call["id"], json.loads(call["function"]["arguments"]))
            for entry in sent["messages"]
            if entry.get("role") == "assistant"
            for call in entry.get("tool_calls", [])
        ]
    return [
        (item["call_id"], json.loads(item["arguments"]))
        for item in sent["input"]
        if item.get("type") == "function_call"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_a_call_taken_out_that_the_eval_holds_redacted_goes_back_as_made_on_openai(
    api: str,
) -> None:
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_openai_from_inspect(api, "a"))
    content_type, body = openai_recorded(api)
    await handler.reply(
        replace(
            reply(body, id="a", content_type=content_type),
            provider="openai",
            model="gpt-test",
        )
    )

    # the turn as the eval was handed it, Tokyo's call with its id and
    # function only, and the rejection as the sidecar wrote it
    found = await handler.request_result("request_456", "eval_789")
    assert not isinstance(found, str) and found.latest.judged is not None
    proposed = found.latest.judged.proposed
    assert proposed is not None
    (redacted,) = proposed.redacted
    calls = withheld_turn(proposed).message["tool_calls"]
    run = await handler._state.existing_run("sample:sample_123")  # pyright: ignore[reportPrivateUsage]
    assert run is not None
    (conversation,) = run.conversations.values()
    written = {
        id: amendment.result
        for id, amendment in conversation.edited[0].amendments.items()
    }
    held = _openai_held(api, calls, written)
    answer = await handler.request(_openai_from_inspect(api, "b", held))

    # the model is sent the call it made, and no result for Tokyo was judged
    assert isinstance(answer, Replace)
    assert _openai_calls_sent(api, answer.body) == [
        (
            call["id"],
            {"location": "Tokyo"} if call["id"] == redacted else call["arguments"],
        )
        for call in calls
    ]
    assert redacted not in [record.call for record in await handler.records("b") or []]


@pytest.mark.anyio
async def test_a_refused_turn_that_the_eval_holds_redacted_goes_back_as_made() -> None:
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    await handler.request(_from_inspect(request("a")))
    refused = await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    assert isinstance(refused, Refuse)

    held = _refused_turn_put_back("Tokyo is not allowed.")
    held[0]["content"] = [_redacted("toolu_1")]
    answer = await handler.request(_from_inspect(request("b", then=held)))

    # not added again, sent as the model made it, and not judged as a result
    assert isinstance(answer, Replace)
    assert _calls_sent(answer.body) == [("toolu_1", {"location": "Tokyo"})]


@pytest.mark.anyio
async def test_a_call_no_decision_acted_on_is_never_put_back() -> None:
    # an empty call of the eval's own is left as it is
    handler = await holding(Handler(rejects_tokyo_and_ends_at_its_result()))
    held = [
        {"role": "assistant", "content": [_redacted("toolu_9")]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_9", "content": "Sunny."}
            ],
        },
    ]

    answer = await handler.request(_from_inspect(request("a", then=held)))

    assert answer == Pass()


def _signed_header(irid: str, **changes: Any) -> dict[str, Any]:
    # the header Inspect sends with the request `irid`, signed, as a dict to
    # change after signing
    signed = inspect_headers(irid=irid, **changes)
    header: dict[str, Any] = json.loads(signed["x-inspect-context"])
    return header


def _sent(header: dict[str, Any], id: str, irid: str) -> Request:
    return replace(
        request(id), headers={"x-inspect-context": json.dumps(header), "x-irid": irid}
    )


async def _reason(handler: Handler, irid: str) -> str | None:
    attempts = await handler._state.attempts(irid, "eval_789")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(attempts, list)
    return attempts[-1].reason


async def _kept(handler: Handler, irid: str, eval_id: str = "eval_789") -> bool:
    # whether anything was kept under the eval and the request id, for
    # `request_result`
    attempts = await handler._state.attempts(irid, eval_id)  # pyright: ignore[reportPrivateUsage]
    return attempts is not None


async def _terminated() -> Handler:
    # a handler whose run of `SAMPLE` a sentinel has ended
    handler = await holding(Handler(weather_rule("terminate", "Tokyo")))
    await handler.request(_from_inspect(request("a"), "irid-a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))
    ended = await handler.sample_result("sample_123", "eval_789")
    assert ended is not None and ended.outcome == "terminate"
    return handler


_NAMES_THE_RUN = (
    "task",
    "sample_id",
    "epoch",
    "sample_uuid",
    "attempt",
    "attempt_number",
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "another_task"),
        ("sample_id", 2),
        ("epoch", 1),
        ("sample_uuid", "sample_999"),
        ("attempt", "attempt_2"),
        ("attempt_number", 2),
        ("timestamp", None),
    ],
)
async def test_a_header_changed_after_it_was_signed_is_refused(
    field: str, value: Any
) -> None:
    handler = await _terminated()
    header = _signed_header("irid-b")
    header[field] = header["timestamp"] + 1 if field == "timestamp" else value

    answer = await handler.request(_sent(header, "b", "irid-b"))

    assert isinstance(answer, Refuse) and answer.status == 400
    assert "mac isn't one made with the eval's token" in answer.message
    # never told the mac that would have matched
    matching = _signed_header(
        "irid-b",
        **{name: header[name] for name in _NAMES_THE_RUN},
        timestamp=header["timestamp"],
    )
    assert matching["mac"] not in answer.message
    assert not await _kept(handler, "irid-b")
    # the run is as it was: still ended, and no other run was begun
    assert (await handler.sample_result("sample_123", "eval_789")) == SampleResult(
        "terminate", {}
    )
    assert await handler.sample_result("sample_999", "eval_789") is None
    still = await handler.request(_from_inspect(request("c"), "irid-c"))
    assert isinstance(still, Refuse)
    assert await _reason(handler, "irid-c") is None


@pytest.mark.anyio
async def test_a_raised_attempt_number_doesnt_start_a_terminated_run_afresh() -> None:
    # signed as the attempt it was, then raised, as a retry would be
    handler = await _terminated()
    header = _signed_header("irid-b")
    header |= {"attempt": "attempt_2", "attempt_number": 2}

    answer = await handler.request(_sent(header, "b", "irid-b"))
    retried = await handler.request(_from_inspect(request("c"), "irid-c"))

    assert isinstance(answer, Refuse)
    assert not await _kept(handler, "irid-b")
    assert isinstance(retried, Refuse)
    ended = await handler.sample_result("sample_123", "eval_789")
    assert ended is not None and ended.outcome == "terminate"


@pytest.mark.anyio
async def test_a_mac_made_with_another_evals_token_is_refused() -> None:
    handler = await holding(Handler(weather_rule()))
    await holding(handler, "eval_other", "eval_other-token-made-for-these-tests")
    signed_by_other = inspect_headers(token="eval_other-token-made-for-these-tests")

    answer = await handler.request(replace(request("x"), headers=signed_by_other))

    assert isinstance(answer, Refuse) and answer.status == 400
    assert not await _kept(handler, "request_456")
    assert await handler.sample_result("sample_123", "eval_789") is None


@pytest.mark.anyio
async def test_a_request_naming_another_evals_sample_is_refused() -> None:
    # the eval's run, and a request that names it from another eval, signed
    # with the token of the eval it names, which the sender doesn't hold
    handler = await holding(Handler(weather_rule("terminate", "Tokyo")))
    await handler.request(_from_inspect(request("a"), "irid-a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))
    header = _signed_header("irid-b", eval_id="eval_other")

    answer = await handler.request(_sent(header, "b", "irid-b"))

    # the sidecar holds no token for that eval, so it can't check the request
    assert isinstance(answer, Refuse) and answer.status == 400
    assert "holds no registration of the eval" in answer.message
    assert not await _kept(handler, "irid-b", "eval_other")
    # and nothing of the other eval's run changed or became readable
    assert await handler.sample_result("sample_123", "eval_other") is None


@pytest.mark.anyio
async def test_an_eval_of_the_clients_own_cant_reach_another_evals_run() -> None:
    # a client that registered an eval of its own holds that eval's token, so
    # it can sign a header that names the terminated sample of another eval,
    # with a later attempt that would otherwise start the run afresh
    handler = await _terminated()
    own_token = "eval-own-token-the-client-registered-with"
    await holding(handler, "eval_own", own_token)

    answer = await handler.request(
        _from_inspect(
            request("b"),
            "irid-b",
            token=own_token,
            eval_id="eval_own",
            attempt="attempt_2",
            attempt_number=2,
        )
    )

    assert isinstance(answer, Refuse) and answer.status == 400
    assert not await _kept(handler, "irid-b", "eval_own")
    ended = await handler.sample_result("sample_123", "eval_789")
    assert ended is not None and ended.outcome == "terminate"


@pytest.mark.anyio
@pytest.mark.parametrize("ended", [False, True], ids=["kept run", "ended run"])
async def test_a_request_refused_for_naming_another_evals_run_keeps_it_no_longer(
    monkeypatch: pytest.MonkeyPatch, ended: bool
) -> None:
    # The run, or what is left of it once its eval ended it, is kept for 100
    # seconds after its eval last named it. A request from an eval of the
    # client's own that names the run is refused as another eval's, keeps
    # nothing of its own, and must not keep the run past its expiry.
    monkeypatch.setattr(_state, "time", _ExpiryClock)
    _ExpiryClock.now = 0.0
    handler = await holding(
        Handler(weather_rule("terminate", "Tokyo"), state=MemoryState(expiry=100))
    )
    await handler.request(_from_inspect(request("a"), "irid-a"))
    if ended:
        assert await handler.end_sample("sample_123", "eval_789") == "ended"
    own_token = "eval-own-token-the-client-registered-with"
    await holding(handler, "eval_own", own_token)

    _ExpiryClock.now = 60.0
    answer = await handler.request(
        _from_inspect(request("b"), "irid-b", token=own_token, eval_id="eval_own")
    )
    _ExpiryClock.now = 120.0

    assert isinstance(answer, Refuse) and answer.status == 400
    assert "names a sample of another eval" in answer.message
    assert not await _kept(handler, "irid-b", "eval_own")
    # `end_sample` reads an ended run, which `sample_result` doesn't
    assert await handler.end_sample("sample_123", "eval_789") == "unknown"
    assert await handler.sample_result("sample_123", "eval_789") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "asked",
    [
        "sample_result",
        "end_sample",
        "sample_result of an ended run",
        "end_sample of an ended run",
    ],
)
async def test_another_eval_asking_of_a_run_keeps_it_no_longer(
    monkeypatch: pytest.MonkeyPatch, asked: str
) -> None:
    # The run, or what is left of it once its eval ended it, is kept for 100
    # seconds after its eval last named it. Another eval asking of it, or
    # ending it, is told nothing and must not keep it past that.
    monkeypatch.setattr(_state, "time", _ExpiryClock)
    _ExpiryClock.now = 0.0
    handler = await holding(Handler(weather_rule(), state=MemoryState(expiry=100)))
    await handler.request(_from_inspect(request("a"), "irid-a"))
    if asked.endswith("of an ended run"):
        assert await handler.end_sample("sample_123", "eval_789") == "ended"

    _ExpiryClock.now = 60.0
    if asked.startswith("sample_result"):
        assert await handler.sample_result("sample_123", "eval_own") is None
    else:
        assert await handler.end_sample("sample_123", "eval_own") == "unknown"
    _ExpiryClock.now = 120.0

    assert await handler.sample_result("sample_123", "eval_789") is None
    assert await handler.end_sample("sample_123", "eval_789") == "unknown"


@pytest.mark.anyio
async def test_the_eval_asking_of_its_own_run_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_state, "time", _ExpiryClock)
    _ExpiryClock.now = 0.0
    handler = await holding(Handler(weather_rule(), state=MemoryState(expiry=100)))
    await handler.request(_from_inspect(request("a"), "irid-a"))

    _ExpiryClock.now = 60.0
    assert await handler.sample_result("sample_123", "eval_789") is not None
    _ExpiryClock.now = 120.0

    assert await handler.sample_result("sample_123", "eval_789") is not None


@pytest.mark.anyio
async def test_another_evals_request_with_the_same_id_leaves_the_result_as_it_was() -> (
    None
):
    # A client that registered an eval of its own signs a header that passes
    # the check, with the id of another eval's request. A request is kept
    # under its eval and its id together, so the other eval's is untouched.
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(replace(request("a"), headers=inspect_headers(irid="irid-a")))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))
    before = await handler.request_result("irid-a", "eval_789")
    own_token = "eval-own-token-the-client-registered-with"
    await holding(handler, "eval_own", own_token)

    await handler.request(
        replace(
            request("b"),
            headers=inspect_headers(
                irid="irid-a",
                token=own_token,
                eval_id="eval_own",
                sample_uuid="sample_own",
            ),
        )
    )
    after = await handler.request_result("irid-a", "eval_789")

    assert not isinstance(before, str) and not isinstance(after, str)
    assert after.latest is before.latest
    assert after.latest.outcome == "reject"


class _ExpiryClock:
    # what the state's expiry reads
    now = 0.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.now


@pytest.mark.anyio
@pytest.mark.parametrize("left_out", ["mac", "timestamp", "an unknown version"])
async def test_a_header_without_its_credentials_is_unreadable(left_out: str) -> None:
    handler = await holding(Handler(weather_rule()))
    header = _signed_header("irid-x")
    if left_out == "an unknown version":
        header["version"] = 2
    else:
        del header[left_out]

    answer = await handler.request(_sent(header, "x", "irid-x"))

    assert answer == Refuse(
        400, "The request's x-inspect-context header isn't one this sidecar can read."
    )
    assert not await _kept(handler, "irid-x")


@pytest.mark.anyio
@pytest.mark.parametrize("irid", [None, ""], ids=["left out", "empty"])
async def test_a_header_without_a_request_id_is_unreadable(irid: str | None) -> None:
    # the mac signs the request's id, so without one it can't be checked
    handler = await holding(Handler(weather_rule()))
    header = inspect_headers()["x-inspect-context"]
    headers = {"x-inspect-context": header} | (
        {"x-irid": irid} if irid is not None else {}
    )

    answer = await handler.request(replace(request("x"), headers=headers))

    assert answer == Refuse(
        400,
        "The request has an x-inspect-context header and no x-irid header, which Inspect sends with it.",
    )
    assert await handler.sample_result("sample_123", "eval_789") is None


async def _with_a_judged_request() -> tuple[Handler, RequestResult]:
    # a handler that judged the eval's request `irid-a`, and what
    # `request_result` answers for it
    handler = await holding(Handler(weather_rule("reject", "Tokyo")))
    await handler.request(_from_inspect(request("a"), "irid-a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))
    before = await handler.request_result("irid-a", "eval_789")
    assert not isinstance(before, str) and before.latest.outcome == "reject"
    return handler, before


@pytest.mark.anyio
@pytest.mark.parametrize("sample", ["the same sample", "another sample"])
async def test_a_mac_made_for_one_request_id_is_refused_with_another(
    sample: str,
) -> None:
    # signed for `irid-b`, and sent as `irid-a`, so that what is decided of
    # it would be filed under `irid-a`
    handler, before = await _with_a_judged_request()
    other = {"sample_id": 2, "sample_uuid": "sample_999"}
    header = _signed_header("irid-b", **(other if sample == "another sample" else {}))

    answer = await handler.request(_sent(header, "b", "irid-a"))

    assert isinstance(answer, Refuse) and answer.status == 400
    assert "mac isn't one made with the eval's token" in answer.message
    assert await handler.request_result("irid-a", "eval_789") == before
    assert not await _kept(handler, "irid-b")
    assert await handler.sample_result("sample_999", "eval_789") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "junk", ["a wrong mac", "a stale timestamp", "an unreadable header", "no header"]
)
async def test_a_junk_request_with_another_requests_id_leaves_its_result_as_it_was(
    junk: str,
) -> None:
    handler, before = await _with_a_judged_request()
    headers = {
        "a wrong mac": inspect_headers(
            irid="irid-a", token="a-token-the-eval-never-made-for-these-tests"
        ),
        "a stale timestamp": inspect_headers(
            irid="irid-a", timestamp=int(time.time()) - 1000
        ),
        "an unreadable header": {"x-inspect-context": "{not json", "x-irid": "irid-a"},
        "no header": {"x-irid": "irid-a"},
    }[junk]

    await handler.request(replace(request("b"), headers=headers))

    assert await handler.request_result("irid-a", "eval_789") == before


class _Clock:
    now = 1_794_182_400.0

    @classmethod
    def time(cls) -> float:
        return cls.now


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("off", "accepted"),
    [(-300, True), (300, True), (-301, False), (301, False)],
    ids=["just inside, behind", "just inside, ahead", "behind", "ahead"],
)
async def test_a_header_is_accepted_only_within_the_window(
    monkeypatch: pytest.MonkeyPatch, off: int, accepted: bool
) -> None:
    # the signature's own clock only
    monkeypatch.setattr(_sign, "time", _Clock)
    handler = await holding(Handler(weather_rule()))
    header = _signed_header("irid-x", timestamp=int(_Clock.now) + off)

    answer = await handler.request(_sent(header, "x", "irid-x"))

    if accepted:
        assert answer == Pass()
    else:
        assert isinstance(answer, Refuse) and answer.status == 400
        way = "behind" if off < 0 else "ahead of"
        assert f"301 seconds {way} the sidecar's clock" in answer.message
        assert not await _kept(handler, "irid-x")


@pytest.mark.anyio
@pytest.mark.parametrize("timestamp", [10**400, -(10**400)], ids=["ahead", "behind"])
async def test_a_timestamp_too_large_for_a_float_is_refused(timestamp: int) -> None:
    handler = await holding(Handler(weather_rule()))
    header = _signed_header("irid-x", timestamp=timestamp)

    answer = await handler.request(_sent(header, "x", "irid-x"))

    assert isinstance(answer, Refuse) and answer.status == 400
    assert "far" in answer.message
    assert not await _kept(handler, "irid-x")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("epoch", "1" + "0" * 4400),
        ("attempt_number", "1" + "0" * 4400),
        ("sample_id", "1" + "0" * 4400),
        ("timestamp", "1" + "0" * 4400),
        ("epoch", "1" + "0" * 400),
        ("mac", '""'),
        ("mac", '"*"'),
        ("mac", '"\\ud800"'),
        ("task", '"\\ud800"'),
    ],
)
async def test_a_header_with_an_odd_value_is_refused_not_failed(
    field: str, raw: str
) -> None:
    # a value too large to turn into text or a float, or text that can't be
    # encoded, is refused like any wrong header, and never raises: a proxy
    # that lets through what the sentinel failed on would let it through
    handler = await holding(Handler(weather_rule()))
    header = _signed_header("irid-x")
    header[field] = 0
    text = json.dumps(header).replace(f'"{field}": 0', f'"{field}": {raw}')

    answer = await handler.request(
        replace(request("x"), headers={"x-inspect-context": text, "x-irid": "irid-x"})
    )

    assert isinstance(answer, Refuse) and answer.status == 400
    assert not await _kept(handler, "irid-x")


@pytest.mark.anyio
async def test_the_window_is_the_handlers_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sign, "time", _Clock)
    handler = await holding(Handler(weather_rule(), timestamp_window=10))

    inside = await handler.request(
        _sent(_signed_header("irid-a", timestamp=int(_Clock.now) - 10), "a", "irid-a")
    )
    outside = await handler.request(
        _sent(_signed_header("irid-b", timestamp=int(_Clock.now) - 11), "b", "irid-b")
    )

    assert inside == Pass()
    assert isinstance(outside, Refuse)
