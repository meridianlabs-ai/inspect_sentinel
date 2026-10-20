import json
import time

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from inspect_sentinel.sidecar._history import EXACT_PAIRS, record_sent
from inspect_sentinel.sidecar._state import Conversation


def _read(*texts: str) -> list[ChatMessage]:
    # a request as the sidecar reads it: every message with an id of its own,
    # as each read gives
    roles = {"s": ChatMessageSystem, "u": ChatMessageUser, "a": ChatMessageAssistant}
    return [roles[text[0]](content=text[2:]) for text in texts]


def _sent(*requests: list[ChatMessage], body: bytes = b"") -> list[str]:
    conversation = Conversation()
    for input in requests:
        record_sent(conversation, input, body)
    return [f"{message.role[0]}:{message.text}" for message in conversation.sent]


def test_each_message_counts_once_however_many_requests_send_it() -> None:
    assert _sent(
        _read("s:Be brief.", "u:Weather?"),
        _read("s:Be brief.", "u:Weather?", "a:Sunny.", "u:Tomorrow?"),
    ) == ["s:Be brief.", "u:Weather?", "a:Sunny.", "u:Tomorrow?"]


def test_a_request_sent_again_adds_nothing() -> None:
    # a client that retries sends the same request
    assert _sent(_read("u:Weather?"), _read("u:Weather?")) == ["u:Weather?"]


def test_what_a_compaction_drops_is_kept_and_only_its_summary_is_new() -> None:
    assert _sent(
        _read("s:Be brief.", "u:Weather?", "a:Sunny.", "u:Tomorrow?", "a:Rain."),
        _read(
            "s:Be brief.", "u:Weather?", "u:Summary: sunny.", "u:Tomorrow?", "a:Rain."
        ),
    ) == [
        "s:Be brief.",
        "u:Weather?",
        "a:Sunny.",
        "u:Tomorrow?",
        "a:Rain.",
        "u:Summary: sunny.",
    ]


def test_a_message_said_again_counts_again() -> None:
    assert _sent(
        _read("u:Go on."),
        _read("u:Go on.", "a:Done.", "u:Go on."),
    ) == ["u:Go on.", "a:Done.", "u:Go on."]


def test_a_changed_message_is_a_new_version() -> None:
    # as Inspect counts a message sent again with other content
    assert _sent(
        _read("u:Weather?", "a:Sunny."),
        _read("u:Weather?", "a:Sunny, 20C."),
    ) == ["u:Weather?", "a:Sunny.", "a:Sunny, 20C."]


def test_a_call_whose_arguments_are_written_in_another_order_is_the_same() -> None:
    # Inspect compares messages by value, and a client may write a call's
    # arguments in another order when it sends the call again
    def calling(arguments: dict[str, str]) -> list[ChatMessage]:
        call = ToolCall(id="call_1", function="get_weather", arguments=arguments)
        return [
            *_read("u:Weather?"),
            ChatMessageAssistant(content="", tool_calls=[call]),
        ]

    assert _sent(
        calling({"city": "Oslo", "units": "metric"}),
        calling({"units": "metric", "city": "Oslo"}),
    ) == ["u:Weather?", "a:"]


def test_a_call_is_told_apart_by_an_id_the_request_carries() -> None:
    # two calls alike but for their ids, which the request carries, are two
    def calling(id: str) -> list[ChatMessage]:
        call = ToolCall(id=id, function="get_weather", arguments={"city": "Oslo"})
        return [
            *_read("u:Weather?"),
            ChatMessageAssistant(content="", tool_calls=[call]),
        ]

    assert _sent(
        calling("call_1"), calling("call_2"), body=b'["call_1", "call_2"]'
    ) == ["u:Weather?", "a:", "a:"]


def test_an_argument_named_id_is_kept_whole() -> None:
    # A call's arguments are what the model asked for, whatever their names.
    # OpenAI's requests carry them as a JSON string, in which an argument's
    # value is not written as the body writes an id.
    def calling(record: str) -> list[ChatMessage]:
        call = ToolCall(id="call_1", function="read_record", arguments={"id": record})
        return [
            *_read("u:Read it."),
            ChatMessageAssistant(content="", tool_calls=[call]),
        ]

    assert _sent(calling("record_A"), calling("record_B"), body=b'["call_1"]') == [
        "u:Read it.",
        "a:",
        "a:",
    ]


def test_a_request_that_opens_alike_between_two_of_an_agent_does_not_count_its_turns_again() -> (
    None
):
    # a monitor's call, which opens as the agent's conversation does
    assert _sent(
        _read("s:Be brief.", "u:Weather?", "a:Sunny.", "u:Tomorrow?"),
        _read("s:Be brief.", "u:Weather?", "u:Rate this."),
        _read(
            "s:Be brief.", "u:Weather?", "a:Sunny.", "u:Tomorrow?", "a:Rain.", "u:Ok."
        ),
    ) == [
        "s:Be brief.",
        "u:Weather?",
        "a:Sunny.",
        "u:Tomorrow?",
        "u:Rate this.",
        "a:Rain.",
        "u:Ok.",
    ]


def test_two_agents_that_open_alike_and_take_turns_each_count_their_turns_once() -> (
    None
):
    turns = [
        _read("u:Task.", "a:Plan A.", "u:Go."),
        _read("u:Task.", "a:Plan B.", "u:Go."),
        _read("u:Task.", "a:Plan A.", "u:Go.", "a:Done A."),
        _read("u:Task.", "a:Plan B.", "u:Go.", "a:Done B."),
    ]
    # Each agent's request is matched against its own last one. What the two
    # say alike, as both are told to go, is one message: without an id, the
    # second agent's can't be told from the first's.
    once = ["u:Task.", "a:Plan A.", "u:Go.", "a:Plan B.", "a:Done A.", "a:Done B."]
    assert _sent(*turns) == once
    # taking turns again adds nothing more
    assert _sent(*turns, *turns) == once


@pytest.mark.parametrize(
    "body",
    [
        '["call_é1", "call_é2"]'.encode(),
        b'["call_\\u00E91", "call_\\u00E92"]',
        b'["call\\/1", "call\\/2"]',
    ],
    ids=["raw UTF-8", "escaped in capitals", "an escaped slash"],
)
def test_an_id_the_request_carries_is_known_however_the_body_writes_it(
    body: bytes,
) -> None:
    ids = [value for value in json.loads(body)]

    def calling(id: str) -> list[ChatMessage]:
        call = ToolCall(id=id, function="get_weather", arguments={"city": "Oslo"})
        return [
            *_read("u:Weather?"),
            ChatMessageAssistant(content="", tool_calls=[call]),
        ]

    assert _sent(calling(ids[0]), calling(ids[1]), body=body) == [
        "u:Weather?",
        "a:",
        "a:",
    ]


def test_turns_said_again_after_a_compaction_are_new() -> None:
    # The request after the compaction continues the compacted one, which it
    # begins as, though an earlier one shares more with it. What the agent
    # says again after the summary is new.
    assert _sent(
        _read("s:Be brief.", "u:Weather?", "a:Done.", "u:Continue."),
        _read("s:Be brief.", "u:Weather?", "u:Summary."),
        _read("s:Be brief.", "u:Weather?", "u:Summary.", "a:Done.", "u:Continue."),
    ) == [
        "s:Be brief.",
        "u:Weather?",
        "a:Done.",
        "u:Continue.",
        "u:Summary.",
        "a:Done.",
        "u:Continue.",
    ]


def test_a_compaction_that_keeps_the_latest_turns_continues_the_agents_request() -> (
    None
):
    # a monitor's call that opens alike begins as the compacted request does
    # for as long as the agent's does, and the agent's shares more
    assert _sent(
        _read("u:Task.", "a:One.", "u:Go.", "a:Two.", "u:Go on."),
        _read("u:Task.", "u:Rate this."),
        _read("u:Task.", "u:Summary.", "a:Two.", "u:Go on.", "a:Three."),
    ) == [
        "u:Task.",
        "a:One.",
        "u:Go.",
        "a:Two.",
        "u:Go on.",
        "u:Rate this.",
        "u:Summary.",
        "a:Three.",
    ]


def test_many_messages_alike_are_matched_in_time_proportional_to_their_number() -> None:
    # A request of many messages alike, which an exact match compares with
    # every other, as a client might send, must not stall the sidecar.
    messages = [ChatMessageUser(content="ok") for _ in range(4_000)]
    requests = [messages[: 3_984 + count] for count in range(17)]
    compacted = [messages[0], ChatMessageUser(content="Summary."), *messages[2:]]
    conversation = Conversation()

    started = time.perf_counter()
    for input in [*requests, compacted]:
        record_sent(conversation, input, b"")
    elapsed = time.perf_counter() - started

    # each counted once, and the summary new
    assert [message.text for message in conversation.sent] == ["ok"] * 4_000 + [
        "Summary."
    ]
    # an exact match took over ten seconds a request at this size
    assert elapsed < 5


def test_a_long_request_after_a_compaction_is_matched_in_order() -> None:
    # past what is compared exactly, what follows the shared beginning is
    # matched in order: the turns the compaction kept are matched, and only
    # the summary and the new turn are new
    turns = [f"a:Turn {number}." for number in range(300)]
    assert _sent(
        _read("u:Task.", *turns),
        _read("u:Task.", "u:Summary.", *turns[50:], "a:New."),
    ) == ["u:Task.", *turns, "u:Summary.", "a:New."]


def test_messages_repeated_past_the_exact_bound_can_be_taken_for_earlier_ones() -> None:
    # A limit, kept here so that a change to it is seen. The earlier request
    # holds B, C and D scattered, then A, then B, C and D together; this one
    # sends B, C and D after a summary, then A again, which is new. Past what
    # is compared exactly, they are matched in order, with the scattered
    # ones, so the new A is taken for the earlier one and left out. Compared
    # exactly, the run would be matched and the new A counted.
    asides = [f"u:Aside {number}." for number in range(200)]
    earlier = [
        "u:B.",
        "u:X.",
        "u:C.",
        "u:Y.",
        "u:D.",
        "u:A.",
        *asides,
        "u:B.",
        "u:C.",
        "u:D.",
    ]
    later = [
        "u:Summary.",
        "u:B.",
        "u:C.",
        "u:D.",
        "u:A.",
        *[f"a:New {n}." for n in range(200)],
    ]
    assert len(earlier) * len(later) > EXACT_PAIRS

    sent = _sent(_read("u:Task.", *earlier), _read("u:Task.", *later))

    assert sent.count("u:A.") == 1
