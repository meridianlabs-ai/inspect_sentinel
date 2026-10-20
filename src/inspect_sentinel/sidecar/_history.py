"""What a conversation has sent the model, as the core's `step.history` holds it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any, cast

from inspect_ai.model import ChatMessage

from ._state import Conversation

# Not sent to the model: the id, source and metadata, and the model an
# assistant message names, which a reader fills from the request it reads, so
# that it changes when a conversation goes on with another model.
_NOT_SENT = {"id": None, "source": None, "metadata": None, "model": None}


def record_sent(
    conversation: Conversation, input: Sequence[ChatMessage], body: bytes
) -> None:
    """Add what one request of the conversation sent the model to what it has sent.

    A message counts once, as in Inspect's own process. Inspect tells a message it sent before by its id, which a request doesn't carry: each read gives a message a new one. So a request is matched against the one before it by what each message says, in order. A message matched there was sent before; any other is new, even one that says what an earlier message said, as a user may say the same thing twice. A compaction that keeps the latest turns behind a summary leaves them matched, and only the summary is new.

    Args:
        conversation: The conversation.
        input: The messages the request sends the model, as read from `body`.
        body: The request body they were read from, which tells an id the request carries from one a reader made up.
    """
    said = body.decode("utf-8", errors="replace")
    keys = [_key(message, said) for message in input]
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


def _key(message: ChatMessage, said: str) -> str:
    # What the model is sent of the message, with each object's keys in one
    # order, as Inspect compares messages by value and a client may write a
    # call's arguments in another. A reader makes up an id for a provider's
    # tool item or call that came without one, a new one at each read, so an
    # id the request doesn't carry is left out where a reader puts one: a
    # call's, a content item's, and the call a tool message answers. A call's
    # arguments and all else are kept whole.
    dumped: dict[str, Any] = message.model_copy(update=_NOT_SENT).model_dump(
        mode="json"
    )
    if dumped.get("tool_call_id") is not None and not _carried(
        dumped["tool_call_id"], said
    ):
        dumped["tool_call_id"] = None
    for field in ("tool_calls", "content"):
        items: object = dumped.get(field)
        if isinstance(items, list):
            dumped[field] = [
                _without_made_up_id(item, said) for item in cast("list[object]", items)
            ]
    return json.dumps(dumped, sort_keys=True)


def _without_made_up_id(item: object, said: str) -> object:
    if not isinstance(item, dict):
        return item
    entry = cast("dict[str, Any]", item)
    if "id" in entry and not _carried(entry["id"], said):
        return {key: value for key, value in entry.items() if key != "id"}
    return entry


def _carried(id: object, said: str) -> bool:
    # whether the request carries the id, as JSON writes a string
    return isinstance(id, str) and json.dumps(id) in said
