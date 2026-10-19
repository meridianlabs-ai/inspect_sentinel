"""What a conversation has sent the model, as the core's `step.history` holds it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from difflib import SequenceMatcher

from inspect_ai.model import ChatMessage

from ._state import Conversation

# the id, source and metadata are not sent to the model
_NOT_SENT = {"id": None, "source": None, "metadata": None}


def record_sent(conversation: Conversation, input: Sequence[ChatMessage]) -> None:
    """Add what one request of the conversation sent the model to what it has sent.

    A message counts once, as in Inspect's own process. Inspect tells a message it sent before by its id, which a request doesn't carry: each read gives a message a new one. So a request is matched against the one before it by what each message says, in order. A message matched there was sent before; any other is new, even one that says what an earlier message said, as a user may say the same thing twice. A compaction that keeps the latest turns behind a summary leaves them matched, and only the summary is new.
    """
    keys = [_key(message) for message in input]
    matched: dict[int, ChatMessage] = {}
    found = SequenceMatcher(None, conversation.last_sent_keys, keys, autojunk=False)
    for block in found.get_matching_blocks():
        for offset in range(block.size):
            matched[block.b + offset] = conversation.last_sent[block.a + offset]
    last: list[ChatMessage] = []
    for position, message in enumerate(input):
        original = matched.get(position)
        if original is None:
            conversation.sent.append(message)
            original = message
        last.append(original)
    conversation.last_sent = last
    conversation.last_sent_keys = keys


def _key(message: ChatMessage) -> str:
    # what the model is sent of the message, with each object's keys in one
    # order, as Inspect compares messages by value and a client may write a
    # call's arguments in another
    dumped = message.model_copy(update=_NOT_SENT).model_dump(mode="json")
    return json.dumps(dumped, sort_keys=True)
