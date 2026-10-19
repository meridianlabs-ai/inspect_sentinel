from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

from inspect_sentinel.sidecar._history import record_sent
from inspect_sentinel.sidecar._state import Conversation


def _read(*texts: str) -> list[ChatMessage]:
    # a request as the sidecar reads it: every message with an id of its own,
    # as each read gives
    roles = {"s": ChatMessageSystem, "u": ChatMessageUser, "a": ChatMessageAssistant}
    return [roles[text[0]](content=text[2:]) for text in texts]


def _sent(*requests: list[ChatMessage]) -> list[str]:
    conversation = Conversation()
    for input in requests:
        record_sent(conversation, input)
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
