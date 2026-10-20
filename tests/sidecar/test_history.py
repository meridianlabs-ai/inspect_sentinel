from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from inspect_sentinel.sidecar._history import record_sent
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
