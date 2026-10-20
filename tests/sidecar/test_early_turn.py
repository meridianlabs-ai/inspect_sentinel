import copy
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import (
    Answer,
    Handler,
    Pass,
    Refuse,
    Replace,
    Reply,
    ReplyStart,
    Request,
    _anthropic,
    _openai,
)
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._inspect_client import with_sentinel
from inspect_sentinel.sidecar._release import (
    AnthropicTurn,
    ChatTurn,
    ResponsesTurn,
    Turn,
    closed_prefix,
)
from inspect_sentinel.sidecar._wire import Format, write_events, written_events
from tests.sidecar._traffic import (
    STREAM,
    anthropic_recorded,
    inspect_headers,
    openai_recorded,
    openai_request,
    request,
)

_FORMATS: dict[str, Format] = {
    "anthropic": "anthropic-messages",
    "chat": "openai-chat",
    "responses": "openai-responses",
}


def _exchange(name: str, *, inspect: bool) -> tuple[Request, Reply]:
    # a recorded reply calling get_weather for Paris, Tokyo and Lima at once
    if name == "anthropic":
        asked = request(sample="sample_123") if inspect else request()
        provider, body = "anthropic", anthropic_recorded("parallel-tool-calls")
    else:
        provider, (_, body) = "openai", openai_recorded(f"{name} stream")
        asked = Request(
            id="x",
            provider="openai",
            model="gpt-test",
            user="user-1",
            body=json.dumps(openai_request(name)).encode(),
            headers=inspect_headers() if inspect else {},
        )
    return asked, Reply(
        id="x",
        provider=provider,
        model=asked.model,
        user="user-1",
        status=200,
        content_type=STREAM,
        body=body,
        format=_FORMATS[name],
    )


@protocol
def noting(seen: list[tuple[str, bool, str]], action: str, city: str) -> Protocol:
    """Decide `action` for a weather lookup of `city`, or of every city, noting what each call was given."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        seen.append((step.call.id, step.turn_complete, step.message))
        # the Responses recording names a city with its country, "Tokyo, Japan"
        location = str(step.call.arguments.get("location", ""))
        if city != "every" and not location.startswith(city):
            return Decision.proceed()
        if action == "modify":
            changed = replace(step.call, arguments={"location": "Oslo"})
            return Decision(action="modify", modified=changed)
        if action == "terminate":
            return Decision.terminate("ended")
        return Decision.reject(f"lookup of {city}", message=f"{city} is not allowed.")

    return decide


def _turn(name: str) -> Turn:
    return {"anthropic": AnthropicTurn, "chat": ChatTurn, "responses": ResponsesTurn}[
        name
    ]()


def _editor(name: str) -> Any:
    if name == "anthropic":
        return _anthropic.StreamEditor()
    return _openai.ChunkEditor() if name == "chat" else _openai.ResponsesEditor()


async def _early(
    handler: Handler, reply: Reply, name: str
) -> tuple[Answer, bytes | None, Mapping[str, Any] | None]:
    # what the reply stream does: each call judged once it is complete, the
    # stream edited as it goes out, and the turn judged at its end
    start = ReplyStart(
        id=reply.id,
        provider=reply.provider,
        model=reply.model,
        user=reply.user,
        status=reply.status,
        content_type=reply.content_type,
        format=reply.format,
    )
    early = await handler.early_turn(start)
    turn, editor = _turn(name), _editor(name)
    assert reply.format is not None
    so_far: list[Any] = []
    for event in written_events(reply.body):
        placement = turn.place(event)
        if placement.completed_before:
            refusal = await early.judge(closed_prefix(reply.format, so_far, turn.open))
            if refusal is not None:
                return refusal, None, None
        so_far.append(event)
        if placement.completes:
            refusal = await early.judge(closed_prefix(reply.format, so_far, turn.open))
            if refusal is not None:
                return refusal, None, None
    released = write_events(
        editor.edit(copy.deepcopy(written_events(reply.body)), early.edits)
    )
    editor.finish(early.edits)
    answer = await early.finish(reply.body, released)
    return answer, released, early.field


def _comparable(body: bytes) -> bytes:
    # inspect_ai gives a message a new id each time it is read
    return re.sub(rb'("id": )"[A-Za-z0-9]{22}"', rb'\1"id"', body)


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(_FORMATS))
@pytest.mark.parametrize(
    ("action", "city"),
    [
        ("reject", "Tokyo"),
        ("reject", "Cairo"),
        ("modify", "Tokyo"),
        ("terminate", "Tokyo"),
        ("reject", "every"),
    ],
    ids=["one rejected", "none rejected", "one changed", "run ended", "all rejected"],
)
@pytest.mark.parametrize("inspect", [False, True], ids=["any client", "inspect"])
async def test_judging_calls_as_they_arrive_decides_what_judging_the_whole_reply_does(
    name: str, action: str, city: str, inspect: bool
) -> None:
    asked, reply = _exchange(name, inspect=inspect)
    whole_seen: list[tuple[str, bool, str]] = []
    early_seen: list[tuple[str, bool, str]] = []
    whole = Handler(noting(whole_seen, action, city))
    await whole.request(asked)
    expected = await whole.reply(reply)
    early = Handler(noting(early_seen, action, city))
    await early.request(asked)

    answer, released, field = await _early(early, reply, name)

    if isinstance(expected, Refuse):
        assert isinstance(answer, Refuse)
        assert (answer.status, answer.message) == (expected.status, expected.message)
    else:
        assert isinstance(answer, Pass | Replace) and released is not None
        # an edited turn is answered with what the client was sent
        assert isinstance(answer, Pass) or answer.body == released
        sent = with_sentinel(STREAM, released, field) if field else released
        given = expected.body if isinstance(expected, Replace) else reply.body
        assert _comparable(sent) == _comparable(given)
    # the same calls were judged, each before its turn ended, except under a
    # run that ended, where the whole reply judges none after the end
    assert [id for id, _, _ in early_seen] == [id for id, _, _ in whole_seen]
    assert all(not complete for _, complete, _ in early_seen)
    assert all(complete for _, complete, _ in whole_seen)
    if inspect:
        assert await _latest(early) == await _latest(whole)


async def _latest(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.outcome


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(_FORMATS))
async def test_a_call_judged_early_is_given_only_the_text_before_it(name: str) -> None:
    asked, reply = _exchange(name, inspect=False)
    whole_seen: list[tuple[str, bool, str]] = []
    whole = Handler(noting(whole_seen, "reject", "Cairo"))
    await whole.request(asked)
    await whole.reply(reply)
    early_seen: list[tuple[str, bool, str]] = []
    handler = Handler(noting(early_seen, "reject", "Cairo"))
    await handler.request(asked)

    await _early(handler, reply, name)

    turn_text = whole_seen[0][2]
    assert all(turn_text.startswith(text) for _, _, text in early_seen)


@pytest.mark.anyio
async def test_a_reply_whose_calls_differ_from_those_judged_is_unreadable() -> None:
    asked, reply = _exchange("anthropic", inspect=False)
    handler = Handler(noting([], "reject", "Cairo"))
    await handler.request(asked)
    early = await handler.early_turn(
        ReplyStart(
            id="x",
            provider="anthropic",
            model=reply.model,
            user=None,
            status=200,
            content_type=STREAM,
            format="anthropic-messages",
        )
    )
    events = written_events(reply.body)
    turn = AnthropicTurn()
    so_far: list[Any] = []
    for event in events:
        placement = turn.place(event)
        so_far.append(event)
        if placement.completes:
            await early.judge(closed_prefix("anthropic-messages", so_far, turn.open))
    changed = reply.body.replace(b"Tokyo", b"Osaka")

    from inspect_sentinel.sidecar import UnreadableError

    with pytest.raises(UnreadableError, match="not those judged"):
        await early.finish(changed, changed)
