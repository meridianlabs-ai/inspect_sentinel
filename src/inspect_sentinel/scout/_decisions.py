from __future__ import annotations

import importlib
import json
import math
from typing import TYPE_CHECKING, Any, cast

from ._join import Key, Located, grouped, located, order, without_conversations

if TYPE_CHECKING:
    from pandas import DataFrame  # pyright: ignore[reportMissingTypeStubs]


def compare_decisions(replayed: DataFrame, recorded: DataFrame) -> DataFrame:
    """What a monitor or protocol replayed over transcripts decided at each step, beside what the one in the runs recorded there.

    Pass the step results of `as_scanner(...)` and of `as_reader(...)`, each a scanner's dataframe from `scan_results_df()`, from one scan or two. Rows are matched as `replay_check_differences()` matches them: by transcript, conversation, stage and step id where the replay took its conversations from the log's span tree, as read mode does (`turns` `"recorded"`), and otherwise by transcript, stage and step id alone. A read row's step id is its `replay_step_id`, so an agent bridge's check is matched with the replayed step of the call it checked, and one with no such step is only recorded. A decision recorded `superseded`, which did not take effect, is left out.

    A row is never paired by a guess. Where two rows of one side have the same key (two agents' steps that share a tool call id, in a transcript matched without conversations), each row with that key, on either side, is listed unmatched, as read mode lists two bridge checks that name one step.

    Args:
        replayed: The replay scanner's step results.
        recorded: The read scanner's step results.

    Returns:
        A pandas `DataFrame` with a row per check and the columns `transcript_id`, `conversation` (`None` where the rows were matched without it), `step_id` (the replayed step, or for a row only read mode has, its `replay_step_id`, else its own `step_id`), `stage`, `replayed` and `recorded` (the decision: a protocol's action, or a monitor's suspicion, a number or a dict of dimensions; `None` on the side that lacks the row, or where the replay abstained), `agree` (`True` where both sides have the row and the same decision), `replayed_explanation` and `recorded_explanation`, and `status`: `"matched"`, `"only replayed"` or `"only recorded"`. Sorted by transcript, stage, step and conversation.

    Raises:
        ValueError: If a results row is a scan error, as for a transcript whose monitor failed, or is not a step's, as with `as_scanner(results="transcript")`.
    """
    pandas: Any = importlib.import_module("pandas")
    rows_replayed = located(replayed, "replayed")
    unmatched = without_conversations(rows_replayed)
    left = grouped(rows_replayed, unmatched)
    right = grouped(
        [
            r
            for r in located(recorded, "recorded")
            if r.record.get("metadata.status") != "superseded"
        ],
        unmatched,
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(left.keys() | right.keys(), key=order):
        mine, theirs = left.get(key, []), right.get(key, [])
        if len(mine) == 1 and len(theirs) == 1:
            rows.append(_row(key, mine[0], theirs[0]))
            continue
        rows.extend(_row(key, row, None) for row in mine)
        rows.extend(_row(key, None, row) for row in theirs)
    # object columns, which keep a missing value None rather than NaN, but for
    # the one that is a flag
    frame = pandas.DataFrame(rows, columns=_COLUMNS, dtype=object)
    frame["agree"] = frame["agree"].astype(bool)
    return cast("DataFrame", frame)


_COLUMNS = [
    "transcript_id",
    "conversation",
    "step_id",
    "stage",
    "replayed",
    "recorded",
    "agree",
    "replayed_explanation",
    "recorded_explanation",
    "status",
]


def _row(key: Key, mine: Located | None, theirs: Located | None) -> dict[str, Any]:
    transcript_id, conversation, stage, step_id = key
    replayed = _decision(mine) if mine is not None else None
    recorded = _decision(theirs) if theirs is not None else None
    matched = mine is not None and theirs is not None
    return {
        "transcript_id": transcript_id,
        "conversation": conversation,
        "step_id": step_id,
        "stage": stage,
        "replayed": replayed,
        "recorded": recorded,
        "agree": matched and replayed is not None and replayed == recorded,
        "replayed_explanation": _explanation(mine),
        "recorded_explanation": _explanation(theirs),
        "status": "matched"
        if matched
        else "only recorded"
        if mine is None
        else "only replayed",
    }


def _decision(row: Located) -> Any:
    # a protocol's action, else a monitor's suspicion
    answer = row.record.get("answer")
    if isinstance(answer, str):
        return answer
    return _value(row.record.get("value"), row.record.get("value_type"))


def _value(value: Any, value_type: Any) -> Any:
    # the value as its result held it: a frame holds a missing one as NaN,
    # and holds values as text where its value types differ
    if value_type == "null" or (isinstance(value, float) and math.isnan(value)):
        return None
    if not isinstance(value, str) or value_type == "string":
        return value
    if value_type == "boolean":
        return value.lower() == "true"
    if value_type in ("number", "object", "array"):
        return json.loads(value)
    return value


def _explanation(row: Located | None) -> str | None:
    explanation = row.record.get("explanation") if row is not None else None
    return explanation if isinstance(explanation, str) else None
