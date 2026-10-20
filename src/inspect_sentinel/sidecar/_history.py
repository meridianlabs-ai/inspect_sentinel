"""What a conversation has sent the model, as the core's `step.history` holds it."""

from __future__ import annotations

import hashlib
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


RECENT_REQUESTS = 16
"""How many of a conversation's latest requests a request is matched against. Requests that open alike share a conversation, such as agents given the same task, and each request is matched against the one it continues; this many such agents taking turns are told apart."""


def record_sent(
    conversation: Conversation, input: Sequence[ChatMessage], body: bytes
) -> None:
    """Add what one request of the conversation sent the model to what it has sent.

    A message counts once, as in Inspect's own process. Inspect tells a message it sent before by its id, which a request doesn't carry: each read gives a message a new one. So a request is matched by what each message says, in order, against the earlier request it continues: of the conversation's latest requests, the one it shares most with. Requests that open alike share a conversation, such as an agent's, a monitor's call that opens as the agent's does, and another agent given the same task, so the request just before may not be the one this continues. A message matched there was sent before; any other is new, even one that says what an earlier message said, as a user may say the same thing twice. A compaction that keeps the latest turns behind a summary leaves them matched, and only the summary is new.

    Args:
        conversation: The conversation.
        input: The messages the request sends the model, as read from `body`.
        body: The request body they were read from, which tells an id the request carries from one a reader made up.
    """
    carried = _strings(body)
    keys = [_digest(_key(message, carried)) for message in input]
    matched: set[int] = set()
    for earlier in conversation.recent:
        found = SequenceMatcher(None, earlier, keys, autojunk=False)
        shared = {
            block.b + offset
            for block in found.get_matching_blocks()
            for offset in range(block.size)
        }
        # the latest of those that share most, as they come latest last
        if len(shared) >= len(matched):
            matched = shared
    for position, message in enumerate(input):
        if position not in matched:
            conversation.sent.append(message)
    conversation.recent.append(keys)
    del conversation.recent[:-RECENT_REQUESTS]


def _strings(body: bytes) -> frozenset[str]:
    # every string the request holds, read once, as an id is told carried by
    # being one of them, however the body escapes it
    found: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            for item in cast("dict[str, object]", value).values():
                collect(item)
        elif isinstance(value, list):
            for item in cast("list[object]", value):
                collect(item)

    try:
        collect(json.loads(body))
    except ValueError:
        pass
    return frozenset(found)


def _digest(key: str) -> str:
    # what the conversation keeps of each message's key, which is as long as
    # the message
    return hashlib.sha256(key.encode()).hexdigest()


def _key(message: ChatMessage, carried: frozenset[str]) -> str:
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
        dumped["tool_call_id"], carried
    ):
        dumped["tool_call_id"] = None
    for field in ("tool_calls", "content"):
        items: object = dumped.get(field)
        if isinstance(items, list):
            dumped[field] = [
                _without_made_up_id(item, carried)
                for item in cast("list[object]", items)
            ]
    return json.dumps(dumped, sort_keys=True)


def _without_made_up_id(item: object, carried: frozenset[str]) -> object:
    if not isinstance(item, dict):
        return item
    entry = cast("dict[str, Any]", item)
    if "id" in entry and not _carried(entry["id"], carried):
        return {key: value for key, value in entry.items() if key != "id"}
    return entry


def _carried(id: object, carried: frozenset[str]) -> bool:
    return isinstance(id, str) and id in carried
