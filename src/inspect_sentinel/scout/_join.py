from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from pandas import DataFrame  # pyright: ignore[reportMissingTypeStubs]

# a row's key: transcript, conversation (None where not matched by it), stage
# and step id
Key: TypeAlias = tuple[str, str | None, str, str]


# a results row of either scanner, with what identifies its step
@dataclass(frozen=True)
class Located:
    transcript_id: str
    conversation: str
    stage: str
    step_id: str
    turns: str | None
    record: dict[str, Any]


def located(frame: DataFrame, side: str) -> list[Located]:
    # each row of a scanner's step results, with its step: a read row's is
    # its replay_step_id, which an agent bridge's check names apart from its
    # own key, else its own step id
    found: list[Located] = []
    records: Any = cast(Any, frame).to_dict("records")
    for row in cast(list[dict[str, Any]], records):
        transcript_id = row.get("transcript_id")
        error = row.get("scan_error")
        if isinstance(error, str):
            raise ValueError(
                f"The {side} results for transcript {transcript_id!r} are an error, so they cannot be compared: {error}"
            )
        stage = row.get("metadata.stage")
        if not isinstance(stage, str):
            raise ValueError(
                f"The {side} results row for transcript {transcript_id!r} is not a step's; pass the step results of as_scanner() and as_reader(), not results='transcript'."
            )
        turns = row.get("metadata.turns")
        joined = row.get("metadata.replay_step_id")
        found.append(
            Located(
                transcript_id=str(transcript_id),
                conversation=str(row["metadata.conversation"]),
                stage=stage,
                step_id=joined
                if isinstance(joined, str)
                else str(row["metadata.step_id"]),
                turns=turns if isinstance(turns, str) else None,
                record=row,
            )
        )
    return found


def without_conversations(replayed: list[Located]) -> set[str]:
    # the transcripts whose replayed conversations are not read mode's, as the
    # replay did not take them from the log's span tree
    return {r.transcript_id for r in replayed if r.turns != "recorded"}


def grouped(rows: list[Located], unmatched: set[str]) -> dict[Key, list[Located]]:
    # the rows by key, keeping every row of a key, so a caller sees two that
    # could not be told apart rather than one replacing the other
    keyed: dict[Key, list[Located]] = {}
    for row in rows:
        conversation = row.conversation if row.transcript_id not in unmatched else None
        key = (row.transcript_id, conversation, row.stage, row.step_id)
        keyed.setdefault(key, []).append(row)
    return keyed


def shared_key(key: Key, side: str) -> str:
    # why two rows of one side share a key
    transcript_id, conversation, stage, step_id = key
    where = f"transcript {transcript_id!r}, stage {stage!r} and step id {step_id!r}"
    if conversation is not None:
        return f"Two {side} rows have conversation {conversation!r}, {where}, so they cannot be matched."
    return f"Two {side} rows have {where}, from different conversations, and they cannot be told apart: the replay took the transcript's conversations from a stored timeline or its messages, not the log's span tree that read mode's come from. With a stored timeline, replay without timeline= to compare them."


def order(key: Key) -> tuple[str, str, str, str]:
    transcript_id, conversation, stage, step_id = key
    return transcript_id, stage, step_id, conversation or ""


def parsed(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value
