import json
from dataclasses import replace
from typing import Any, cast

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.tool import ToolCall

from inspect_sentinel.sidecar._openai import _check_responses_stream
from inspect_sentinel.sidecar._wire import (
    Edits,
    UnreadableError,
    edit_reply,
    read_reply,
    written_events,
)
from tests.sidecar._traffic import (
    JSON,
    STREAM,
    anthropic_recorded,
    message,
    openai_recorded,
    weather,
)

# Every recorded reply that holds three calls, Paris, Tokyo and Lima, in each
# API, streamed and not: (provider, content type, body).
_REPLIES: dict[str, tuple[str, str, bytes]] = {
    "anthropic": (
        "anthropic",
        JSON,
        message(
            {"type": "text", "text": "Looking up all three."},
            weather("Paris", "toolu_1"),
            weather("Tokyo", "toolu_2"),
            weather("Lima", "toolu_3"),
        ),
    ),
    "anthropic stream": (
        "anthropic",
        STREAM,
        anthropic_recorded("parallel-tool-calls"),
    ),
    **{
        f"openai {name}": ("openai", *openai_recorded(name))
        for name in ("chat", "chat stream", "responses", "responses stream")
    },
}


async def _read(name: str) -> tuple[str, str, bytes, ModelOutput]:
    provider, content_type, body = _REPLIES[name]
    return provider, content_type, body, await read_reply(provider, content_type, body)


def _modified(output: ModelOutput, index: int, location: str) -> ToolCall:
    call = (output.message.tool_calls or [])[index]
    return replace(call, arguments={"location": location})


# which calls to take out and which to change, by their place in the reply
_CASES: dict[str, tuple[list[int], dict[int, str]]] = {
    "first taken out": ([0], {}),
    "middle taken out": ([1], {}),
    "last taken out": ([2], {}),
    "two taken out": ([0, 2], {}),
    "one changed": ([], {1: "Oslo"}),
    "one changed, one taken out": ([0], {2: "Oslo"}),
}


@pytest.mark.anyio
@pytest.mark.parametrize("name", _REPLIES)
@pytest.mark.parametrize("case", _CASES)
async def test_an_edited_reply_reads_back_with_only_the_edits(
    name: str, case: str
) -> None:
    provider, content_type, body, output = await _read(name)
    calls = output.message.tool_calls or []
    removed, changed = _CASES[case]
    edits = Edits(
        frozenset(calls[index].id for index in removed),
        {
            calls[index].id: _modified(output, index, location)
            for index, location in changed.items()
        },
    )

    edited = await edit_reply(provider, content_type, body, None, edits, output)

    read = await read_reply(provider, content_type, edited)
    assert [(call.id, call.arguments) for call in read.message.tool_calls or []] == [
        (call.id, edits.modified.get(call.id, call).arguments)
        for call in calls
        if call.id not in edits.removed
    ]
    assert read.message.text == output.message.text
    assert read.stop_reason == output.stop_reason


@pytest.mark.anyio
@pytest.mark.parametrize(
    "name", [name for name, (_, kind, _) in _REPLIES.items() if kind == STREAM]
)
async def test_a_stream_keeps_the_events_an_edit_doesnt_touch(name: str) -> None:
    provider, content_type, body, output = await _read(name)
    last = (output.message.tool_calls or [])[-1]

    edited = await edit_reply(
        provider, content_type, body, None, Edits(frozenset({last.id})), output
    )

    # every event before the last call's first goes out as it was written
    before = [event.lines for event in written_events(body)]
    after = [event.lines for event in written_events(edited)]
    first = next(
        index
        for index, lines in enumerate(before)
        if any(last.id in line for line, _ in lines)
    )
    assert after[:first] == before[:first]


@pytest.mark.anyio
@pytest.mark.parametrize("name", _REPLIES)
async def test_an_edit_of_a_call_the_reply_doesnt_hold_is_refused(name: str) -> None:
    provider, content_type, body, output = await _read(name)

    with pytest.raises(UnreadableError, match="no call"):
        await edit_reply(
            provider,
            content_type,
            body,
            None,
            Edits(frozenset({"call_absent"})),
            output,
        )


@pytest.mark.anyio
async def test_a_responses_call_that_is_not_a_function_call_cant_be_changed() -> None:
    reply: dict[str, Any] = json.loads(openai_recorded("responses")[1])
    reply["output"].append(
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_custom",
            "name": "shell",
            "input": "ls",
            "status": "completed",
        }
    )
    body = json.dumps(reply).encode()
    output = await read_reply("openai", JSON, body)
    custom = ToolCall(id="call_custom", function="shell", arguments={"input": "pwd"})

    with pytest.raises(UnreadableError, match="not a function call"):
        await edit_reply(
            "openai", JSON, body, None, Edits(modified={"call_custom": custom}), output
        )


def _responses_events() -> list[dict[str, Any]]:
    _, body = openai_recorded("responses stream")
    return [event.data for event in written_events(body) if event.data is not None]


def _check(events: list[dict[str, Any]]) -> None:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    _check_responses_stream(written_events(body))


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["taken out", "changed"])
async def test_a_responses_call_streamed_only_as_done_is_edited_in_every_event(
    change: str,
) -> None:
    events = _responses_events()
    first = await read_reply("openai", STREAM, _body(events))
    call = (first.message.tool_calls or [])[1]
    index = next(
        event["output_index"]
        for event in events
        if event["type"] == "response.output_item.added"
        and event["item"].get("call_id") == call.id
    )
    # the call's done event alone, without the events it began with, which a
    # client may still act on
    body = _body(
        [
            event
            for event in events
            if event.get("output_index") != index
            or event["type"] == "response.output_item.done"
        ]
    )
    output = await read_reply("openai", STREAM, body)
    edits = (
        Edits(frozenset({call.id}))
        if change == "taken out"
        else Edits(
            frozenset(), {call.id: replace(call, arguments={"location": "Oslo"})}
        )
    )

    edited = await edit_reply("openai", STREAM, body, None, edits, output)

    items: list[dict[str, Any]] = []
    for event in written_events(edited):
        data: dict[str, Any] = event.data or {}
        response: dict[str, Any] = data.get("response") or {}
        candidates: list[Any] = [data.get("item"), *(response.get("output") or [])]
        for item in candidates:
            if isinstance(item, dict):
                entry = cast(dict[str, Any], item)
                if entry.get("call_id") == call.id:
                    items.append(entry)
    if change == "taken out":
        assert items == []
    else:
        assert items
        assert all(
            json.loads(item["arguments"]) == {"location": "Oslo"} for item in items
        )


def _body(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def test_a_responses_stream_that_holds_together_passes() -> None:
    _check(_responses_events())


def test_a_responses_stream_whose_deltas_disagree_with_its_output_is_refused() -> None:
    events = _responses_events()
    delta = next(
        event
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    )
    delta["delta"] = '{"location": "Atlantis"}'

    with pytest.raises(UnreadableError, match="different arguments"):
        _check(events)


def _done_only(events: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    # an item's done event alone, without the events it began with
    return [
        event
        for event in events
        if event.get("output_index") != index
        or event["type"] == "response.output_item.done"
    ]


def _done_item(events: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return next(
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["output_index"] == index
    )


def _call_index(events: list[dict[str, Any]]) -> int:
    return next(
        event["output_index"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    )


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"arguments": '{"location": "Atlantis"}'}, "different arguments"),
        ({"call_id": "call_other"}, "a place it doesn't have"),
    ],
    ids=["its arguments", "its call"],
)
def test_a_responses_call_streamed_only_as_done_that_disagrees_with_its_output_is_refused(
    change: dict[str, str], says: str
) -> None:
    events = _responses_events()
    index = _call_index(events)
    events = _done_only(events, index)
    _done_item(events, index).update(change)

    with pytest.raises(UnreadableError, match=says):
        _check(events)


@pytest.mark.parametrize(
    "event", ["response.output_item.added", "response.output_item.done"]
)
def test_a_responses_call_named_otherwise_in_an_event_than_in_its_output_is_refused(
    event: str,
) -> None:
    events = _responses_events()
    index = _call_index(events)
    next(
        found
        for found in events
        if found["type"] == event and found["output_index"] == index
    )["item"]["name"] = "delete_files"

    with pytest.raises(UnreadableError, match="other details"):
        _check(events)


def _items_at(event: dict[str, Any], index: int) -> list[dict[str, Any]]:
    # the items an event holds at an output index, as an item event or in the
    # response it ends with
    response: dict[str, Any] = event.get("response") or {}
    output: list[Any] = response.get("output") or []
    found: list[Any] = [
        event.get("item") if event.get("output_index") == index else None,
        *output[index : index + 1],
    ]
    return [cast(dict[str, Any], item) for item in found if isinstance(item, dict)]


def _as_custom_tool_call(events: list[dict[str, Any]], index: int) -> None:
    # the call made into a call of a custom tool, whose input is streamed as a
    # function call's arguments are
    for event in events:
        for item in _items_at(event, index):
            if item.get("type") == "function_call":
                item["type"] = "custom_tool_call"
                item["input"] = item.pop("arguments")
        if event.get("output_index") == index:
            if event["type"] == "response.function_call_arguments.delta":
                event["type"] = "response.custom_tool_call_input.delta"
            elif event["type"] == "response.function_call_arguments.done":
                event["type"] = "response.custom_tool_call_input.done"
                event["input"] = event.pop("arguments")


def _as_computer_call(events: list[dict[str, Any]], index: int) -> None:
    # the call made into a computer call, which has no deltas
    events[:] = [
        event
        for event in events
        if event.get("output_index") != index or event["type"] in _ITEM_EVENTS
    ]
    for event in events:
        for item in _items_at(event, index):
            if item.get("type") == "function_call":
                item["type"] = "computer_call"
                del item["name"], item["arguments"]
                item["action"] = {"type": "click", "button": "left", "x": 10, "y": 20}
                item["pending_safety_checks"] = []


_ITEM_EVENTS = ("response.output_item.added", "response.output_item.done")


@pytest.mark.parametrize("kind", ["custom tool", "computer"])
def test_a_responses_call_of_another_kind_that_holds_together_passes(
    kind: str,
) -> None:
    events = _responses_events()
    index = _call_index(events)
    (_as_custom_tool_call if kind == "custom tool" else _as_computer_call)(
        events, index
    )

    _check(events)


def test_a_custom_tool_call_whose_deltas_disagree_with_its_output_is_refused() -> None:
    events = _responses_events()
    index = _call_index(events)
    _as_custom_tool_call(events, index)
    next(
        event
        for event in events
        if event["type"] == "response.custom_tool_call_input.delta"
        and event["output_index"] == index
    )["delta"] = "rm -rf /"

    with pytest.raises(UnreadableError, match="different arguments"):
        _check(events)


def test_a_computer_call_whose_done_action_disagrees_with_its_output_is_refused() -> (
    None
):
    events = _responses_events()
    index = _call_index(events)
    _as_computer_call(events, index)
    _done_item(events, index)["action"] = {"type": "keypress", "keys": ["ENTER"]}

    with pytest.raises(UnreadableError, match="other details"):
        _check(events)


def test_a_call_streamed_in_events_that_cant_be_checked_is_refused() -> None:
    events = _responses_events()
    index = _call_index(events)
    events.insert(
        -1,
        {
            "type": "response.shell_call_command.delta",
            "output_index": index,
            "command_index": 0,
            "delta": "rm -rf /",
            "sequence_number": 0,
        },
    )

    with pytest.raises(UnreadableError, match="can't check"):
        _check(events)


@pytest.mark.anyio
async def test_a_kept_call_streamed_only_as_done_must_agree_once_another_is_taken_out() -> (
    None
):
    events = _responses_events()
    output = await read_reply("openai", STREAM, _body(events))
    kept, taken = (output.message.tool_calls or [])[:2]
    index = next(
        event["output_index"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"].get("call_id") == kept.id
    )
    events = _done_only(events, index)
    # the arguments a client acting on the done event would run, which are not
    # the ones the sentinel judged
    _done_item(events, index)["arguments"] = '{"location": "Atlantis"}'
    body = _body(events)

    with pytest.raises(UnreadableError, match="different arguments"):
        await edit_reply(
            "openai", STREAM, body, None, Edits(frozenset({taken.id})), output
        )


def test_a_responses_stream_with_an_item_out_of_place_is_refused() -> None:
    events = _responses_events()
    for event in events:
        if event.get("type") == "response.function_call_arguments.delta":
            event["output_index"] = 9999

    with pytest.raises(UnreadableError):
        _check(events)


@pytest.mark.anyio
async def test_a_chat_stream_call_without_an_index_is_refused() -> None:
    _, body = openai_recorded("chat stream")
    output = await read_reply("openai", STREAM, body)
    first = (output.message.tool_calls or [])[0]
    broken = body.replace(b'"index":0,"id"', b'"id"', 1)

    with pytest.raises(UnreadableError, match="no index"):
        await edit_reply(
            "openai", STREAM, broken, None, Edits(frozenset({first.id})), output
        )
