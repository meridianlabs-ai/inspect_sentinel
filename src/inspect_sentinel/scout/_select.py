from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

from inspect_sentinel import HumanStandIn, ReplayStep

# sentinel's own replay runner, internal since 2026-10-26: the adapter is its
# only caller
from inspect_sentinel._replay import HumanFallback

from ._sources import import_source, source_file

if TYPE_CHECKING:
    from pandas import DataFrame  # pyright: ignore[reportMissingTypeStubs]

StepSelector: TypeAlias = Callable[[str, ReplayStep], bool]
"""Which steps of a transcript a scan runs the sentinel at: handed the transcript's id and a step, true to run it."""

StepTable: TypeAlias = (
    "DataFrame | Iterable[tuple[str, str]] | Iterable[Mapping[str, str | None]]"
)
"""A label table of the steps a scan runs the sentinel at: a pandas `DataFrame` with `transcript_id` and `step_id` columns, or a results dataframe from `scan_results_df()`, whose step ids are in `metadata.step_id`; or rows of `(transcript_id, step_id)` pairs, or of mappings with those keys. A `stage` column or key (`metadata.stage` in a results dataframe) limits each row to the step at that stage; without one, or where a row's stage is missing, a row selects its step at every stage, a tool call's `tool_call` and `tool_result` steps alike, since they share the call's id. A `conversation` column or key (`metadata.conversation`) likewise limits a row to the step in that conversation, since two agents' tool calls can share an id; it is matched against the replay's conversations, which with `as_scanner(timeline=)` are the stored timeline's."""

F = TypeVar("F", bound=Callable[..., Any])

_NAME = "__sentinel_replay_name__"
_KINDS = {"step_selector": "a step selector", "human_stand_in": "a human stand-in"}
_registered: dict[str, dict[str, Callable[..., Any]]] = {kind: {} for kind in _KINDS}


class Table:
    # a label table as a selector, with the transcripts it lists, so a scan
    # skips a transcript it does not list before rebuilding its steps; a row
    # with no conversation or stage selects its step in every one
    def __init__(self, rows: Iterable[tuple[str, str | None, str | None, str]]) -> None:
        self.rows = frozenset(rows)
        self.transcripts = frozenset(row[0] for row in self.rows)

    def __call__(self, transcript_id: str, step: ReplayStep) -> bool:
        return any(
            (transcript_id, conversation, stage, step.id) in self.rows
            for conversation in (step.step.conversation, None)
            for stage in (step.stage, None)
        )


def table_rows(labels: Any) -> list[list[str | None]]:
    # a label table's (transcript_id, conversation, stage, step_id) rows, as a
    # scan records them, the conversation or stage None where the table gives
    # none
    columns = getattr(labels, "columns", None)
    if columns is not None:
        names = set(cast(Iterable[str], columns))
        step = _column(names, "step_id")
        if "transcript_id" not in names or step is None:
            raise ValueError(
                f"A label table needs a transcript_id column and a step_id (or a results dataframe's metadata.step_id) column; it has {sorted(names)}."
            )
        rows = zip(
            cast(Iterable[object], labels["transcript_id"]),
            _optional(labels, _column(names, "conversation")),
            _optional(labels, _column(names, "stage")),
            cast(Iterable[object], labels[step]),
            strict=True,
        )
    else:
        rows = (_row(row) for row in cast(Iterable[object], labels))
    found = {(str(t), _text(c), _text(g), str(s)) for t, c, g, s in rows}
    return [
        list(row)
        for row in sorted(found, key=lambda r: (r[0], r[1] or "", r[2] or "", r[3]))
    ]


def _column(names: set[str], name: str) -> str | None:
    # a label table's column, or a results dataframe's, which keeps it in the
    # result's metadata
    return next((c for c in (name, f"metadata.{name}") if c in names), None)


def _optional(labels: Any, column: str | None) -> Iterable[object]:
    # a column's values with a missing one as None, or None for every row
    if column is None:
        return [None] * len(labels)
    return labels[column].astype(object).where(labels[column].notna(), None)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _row(row: object) -> tuple[object, object, object, object]:
    if isinstance(row, Mapping):
        mapping = cast(Mapping[str, object], row)
        return (
            mapping["transcript_id"],
            mapping.get("conversation"),
            mapping.get("stage"),
            mapping["step_id"],
        )
    transcript_id, step_id = cast(tuple[object, object], row)
    return transcript_id, None, None, step_id


def step_selector(selector: F) -> F:
    """Register a step selector, so a scan that uses it can run in Scout's worker processes and be resumed in another process.

    A selector is a function `(transcript_id, step) -> bool` defined at module level, in a module or in the script that runs the scan. A scan records its module, name and file, and finds it again by importing the module, or loading the file. One defined in a notebook, REPL or `python -c` has no file to load, so `as_scanner()` refuses it: put it in a `.py` file next to the notebook and import it from there.

    Args:
        selector: The selector.
    """
    return _register("step_selector", selector)


def human_stand_in(stand_in: F) -> F:
    """Register a stand-in for the person `human()` asks, so a scan that uses it can run in Scout's worker processes and be resumed in another process.

    A stand-in is an `async` function `(step, escalations, choices) -> HumanAnswer`, as `HumanStandIn` describes, defined at module level, in a module or in the script that runs the scan. A scan records it as `@step_selector` records a selector.

    Args:
        stand_in: The stand-in.
    """
    return _register("human_stand_in", stand_in)


def _register(kind: str, function: F) -> F:
    qualname = getattr(function, "__qualname__", "")
    module = getattr(function, "__module__", None)
    if not module or not qualname or "<" in qualname:
        raise TypeError(
            f"{_KINDS[kind].capitalize()} must be a function defined at module level, so another process can find it; {function!r} is not."
        )
    setattr(function, _NAME, kind)
    _registered[kind][f"{module}:{qualname}"] = function
    return function


def callback_ref(kind: str, callback: object, parameter: str) -> dict[str, str | None]:
    # what a scan records of a registered callback: its module, name and file
    if isinstance(callback, str):
        found = _registered[kind].get(callback)
        if found is None:
            raise ValueError(f"No {_KINDS[kind]} is registered as {callback!r}.")
        callback = found
    decorator = "@step_selector" if kind == "step_selector" else "@human_stand_in"
    if getattr(callback, _NAME, None) != kind:
        raise TypeError(
            f"as_scanner({parameter}=...) takes {_KINDS[kind]} registered with {decorator}, so the scan can record it and find it again in another process; {callback!r} is not registered."
        )
    module: str = getattr(callback, "__module__", "")
    qualname: str = getattr(callback, "__qualname__", "")
    file = source_file(module)
    if module == "__main__" and file is None:
        raise TypeError(
            f"as_scanner({parameter}=...) cannot record {qualname!r}: it was defined in a notebook, REPL or `python -c`, which another process cannot load, and a scan runs in Scout's worker processes and resumes in new ones. Define it in a module or a script; from a notebook, put it in a .py file next to the notebook and import it."
        )
    return {"module": module, "qualname": qualname, "file": file}


def callback_from(kind: str, ref: Mapping[str, str | None]) -> Callable[..., Any]:
    # the registered callback a scan recorded, from this process's registry,
    # else from its module or file, by name: a script's functions are
    # registered under the name it ran as, here or in a worker
    module, qualname = ref.get("module"), ref.get("qualname")
    if module is None or qualname is None:
        raise ValueError(
            f"A recorded {_KINDS[kind]} needs a module and a name: {ref!r}."
        )
    found = _registered[kind].get(f"{module}:{qualname}")
    if found is not None:
        return found
    loaded = import_source(module, ref.get("file"))
    callback: object = loaded
    for part in qualname.split("."):
        callback = getattr(callback, part, None)
    if loaded is None or getattr(callback, _NAME, None) != kind:
        raise ValueError(
            f"Cannot find the {_KINDS[kind]} {module}:{qualname}: importing its module, or loading its file {ref.get('file')!r}, did not define it."
        )
    return cast(Callable[..., Any], callback)


# the fixed answers as_scanner(human=) takes
ANSWERS = ("approve", "reject", "terminate")


def selector_from(steps: object) -> StepSelector | None:
    if steps is None:
        return None
    if isinstance(steps, Mapping):
        recorded = cast(Mapping[str, Any], steps)
        if "table" in recorded:
            rows = cast(list[list[str | None]], recorded["table"])
            return Table((str(t), c, g, str(s)) for t, c, g, s in rows)
        return cast(StepSelector, callback_from("step_selector", recorded))
    raise TypeError(
        f"steps must be a recorded label table or a registered selector; got {steps!r}."
    )


def stand_in_from(human: object) -> HumanFallback:
    if human in ANSWERS:
        return cast(HumanFallback, human)
    if isinstance(human, Mapping):
        return cast(
            HumanStandIn,
            callback_from("human_stand_in", cast(Mapping[str, str | None], human)),
        )
    raise TypeError(
        f"human must be one of {', '.join(repr(a) for a in ANSWERS)} or a registered stand-in; got {human!r}."
    )
