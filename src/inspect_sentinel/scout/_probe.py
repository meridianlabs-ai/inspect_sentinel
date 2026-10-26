from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, Any, cast

from inspect_ai.util import StoreModel
from pydantic_core import to_jsonable_python

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Context,
    EvalContext,
    MonitorGroup,
    Observation,
    Step,
    monitor,
)

if TYPE_CHECKING:
    from pandas import DataFrame  # pyright: ignore[reportMissingTypeStubs]

# the fields replay cannot hold: the untruncated tool output is not in a log,
# a viewer's context is not recorded, and the agent's own list is not either
_ABSENT = {(AfterToolCall, "output")} | {
    (stage, "agent_messages")
    for stage in (BeforeGenerate, AfterGenerate, BeforeToolCall, AfterToolCall)
}
# the conversation lists, digested by their length and last element, which
# keeps a step's digest from reading the whole conversation
_TIPS = {"history", "human_decisions"}
_ROLES = {"system", "user", "assistant", "tool"}


@monitor(name="replay_check")
def replay_check() -> MonitorGroup:
    """A monitor that reports a digest of everything it is handed, to check that replay hands a monitor what it was handed live.

    It watches all four stages and never abstains: each observation has suspicion `0.0` and metadata holding a digest of each field of the step (`step`) and of `context.eval` (`eval`). Attach it to an eval observe-only (`observe_only(replay_check())`), then scan the logs with `as_scanner(replay_check())` and with `as_reader("replay_check")`, and hand the two scanners' results to `replay_check_differences()`: it lists each step, stage and field whose digest differs, and the steps only one scanner reported, so an empty table means replay handed the monitor what it saw live.

    A field's digest is a hash of its JSON form, with message ids left out, since a log stores a message once by its content. `history` and `human_decisions` are digested by their length and their last element, and `context.eval` in two parts. The parts that cannot change within a sample (the task, ids, epoch, descriptions and the sample's input, usually the large part) are digested once per store, which is the sample's live and a unit's in replay, once the context is there, and kept in the probe's store, so the sample's store in the log holds that digest. The metadata, which a solver can change, is digested at every step. So the probe's work per step beyond the step's own input is in proportion to the metadata, and every digest is of the context the step was handed. It cannot see an edit inside those lists, such as a message the agent changed after sending it. `agent_messages` and `AfterToolCall.output`, which replay does not hold, and the view's `context` are left out.
    """

    def digest(context: Context, step: Step) -> Observation:
        # A host may build a new EvalContext for every step, as inspect_ai
        # does, so the digest of the parts that cannot change within a sample,
        # its input above all, is kept in the store, which is the sample's
        # live and the unit's in replay. The metadata, which a solver can
        # change, is digested at every step.
        # An absent context is digested where it is found, never kept, since
        # a host can learn the sample later in the same run, as a sidecar does
        # when its task is registered late.
        current = context.eval
        if current is None:
            fixed = _hash(None)
        else:
            state = context.store_as(_ProbeState)
            if state.fixed_digest is None:
                state.fixed_digest = _hash(_eval_form(current))
            fixed = state.fixed_digest
        return Observation(
            suspicion=0.0,
            metadata={"step": _step_digest(step), "eval": _combined(fixed, current)},
        )

    async def model_input(context: Context, step: BeforeGenerate) -> Observation:
        return digest(context, step)

    async def model_output(context: Context, step: AfterGenerate) -> Observation:
        return digest(context, step)

    async def tool_call(context: Context, step: BeforeToolCall) -> Observation:
        return digest(context, step)

    async def tool_result(context: Context, step: AfterToolCall) -> Observation:
        return digest(context, step)

    return MonitorGroup(model_input, model_output, tool_call, tool_result)


class _ProbeState(StoreModel):
    fixed_digest: str | None = None


def _step_digest(step: Step) -> dict[str, str]:
    digests: dict[str, str] = {}
    for field in fields(step):
        if (type(step), field.name) in _ABSENT:
            continue
        value = getattr(step, field.name)
        if field.name == "view":
            value = value.call
        elif field.name in _TIPS and value is not None:
            items: Sequence[Any] = value
            value = {"length": len(items), "last": items[-1] if items else None}
        elif isinstance(value, Sequence) and not isinstance(value, str | list):
            value = list(cast(Sequence[Any], value))
        digests[field.name] = _hash(value)
    return digests


def context_digest(context: EvalContext | None) -> str:
    # the probe's digest of a context, computed whole
    return _combined(_hash(_eval_form(context)), context)


def _combined(fixed: str, context: EvalContext | None) -> str:
    return _hash([fixed, context.metadata if context is not None else None])


def _eval_form(context: EvalContext | None) -> Any:
    # the parts of a context that cannot change within a sample
    if context is None:
        return None
    return {
        "task": context.task,
        "task_description": context.task_description,
        # the id's type too: 7 and "7" are different samples
        "sample_id": [type(context.sample_id).__name__, context.sample_id],
        "epoch": context.epoch,
        "sample_description": context.sample_description,
        "sample_input": context.sample_input,
    }


def _hash(value: Any) -> str:
    form = _without_ids(to_jsonable_python(value, fallback=repr))
    text = json.dumps(form, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _without_ids(value: Any) -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        message = mapping.get("role") in _ROLES
        return {
            k: _without_ids(v)
            for k, v in mapping.items()
            if not (message and k == "id")
        }
    if isinstance(value, list):
        return [_without_ids(v) for v in value]  # type: ignore[misc]
    return value


def replay_check_differences(replayed: DataFrame, recorded: DataFrame) -> DataFrame:
    """The differences between `replay_check()` replayed and read over the same transcripts: empty where replay handed the monitor what it saw live.

    Pass the results of `as_scanner(replay_check())` and of `as_reader("replay_check")`, each a scanner's dataframe from `scan_results_df()`, from one scan or two. Rows are matched by transcript, step id and stage.

    Args:
        replayed: The replay scanner's results.
        recorded: The read scanner's results.

    Returns:
        A pandas `DataFrame` with a row per difference and the columns `transcript_id`, `step_id`, `stage`, `field` (`step.<name>` for a field of the step, `eval` for `context.eval`, `None` for a step one side lacks), `replayed` and `recorded` (the two digests, `None` on the side that lacks the step), and `reason`: `"differs"`, `"only replayed"` or `"only recorded"`. Sorted by transcript, stage and step.

    Raises:
        ValueError: If a results row is not one of `replay_check()`'s, so it has no digests to compare.
    """
    pandas: Any = importlib.import_module("pandas")
    left, right = _digests(replayed, "replayed"), _digests(recorded, "recorded")
    rows: list[dict[str, Any]] = []
    for key in sorted(left.keys() | right.keys()):
        transcript_id, stage, step_id = key
        where = {"transcript_id": transcript_id, "step_id": step_id, "stage": stage}
        mine, theirs = left.get(key), right.get(key)
        if mine is None or theirs is None:
            rows.append(
                {
                    **where,
                    "field": None,
                    "replayed": None,
                    "recorded": None,
                    "reason": "only recorded" if mine is None else "only replayed",
                }
            )
            continue
        for field in sorted(mine.keys() | theirs.keys()):
            if mine.get(field) != theirs.get(field):
                rows.append(
                    {
                        **where,
                        "field": field,
                        "replayed": mine.get(field),
                        "recorded": theirs.get(field),
                        "reason": "differs",
                    }
                )
    return cast("DataFrame", pandas.DataFrame(rows, columns=_COLUMNS))


_COLUMNS = [
    "transcript_id",
    "step_id",
    "stage",
    "field",
    "replayed",
    "recorded",
    "reason",
]


def _digests(frame: DataFrame, side: str) -> dict[tuple[str, str, str], dict[str, str]]:
    # each row's digests by field, keyed by transcript, stage and step id
    found: dict[tuple[str, str, str], dict[str, str]] = {}
    records: Any = cast(Any, frame).to_dict("records")
    for row in cast(list[dict[str, Any]], records):
        metadata = _parsed(row.get("metadata.metadata"))
        if not isinstance(metadata, dict) or "step" not in metadata:
            raise ValueError(
                f"The {side} results row for transcript {row.get('transcript_id')!r} holds no replay_check() digests; pass the results of as_scanner(replay_check()) and as_reader('replay_check')."
            )
        digests = cast(dict[str, Any], metadata)
        fields = {
            f"step.{k}": v for k, v in cast(dict[str, str], digests["step"]).items()
        }
        fields["eval"] = digests["eval"]
        key = (
            str(row["transcript_id"]),
            str(row["metadata.stage"]),
            str(row["metadata.step_id"]),
        )
        found[key] = fields
    return found


def _parsed(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value
