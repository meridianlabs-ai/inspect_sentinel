from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias, TypeVar, cast

from pydantic import BaseModel, PrivateAttr

from inspect_sentinel import HumanPolicy, HumanStandIn, ReplayStep
from inspect_sentinel._integration import import_source, source_file

StepSelector: TypeAlias = Callable[[str, ReplayStep], bool]
"""Which steps of a transcript a scan runs the sentinel at: handed the transcript's id and a step, true to run it."""

F = TypeVar("F", bound=Callable[..., Any])

_NAME = "__sentinel_replay_name__"
_KINDS = {"step_selector": "a step selector", "human_stand_in": "a human stand-in"}
_registered: dict[str, dict[str, Callable[..., Any]]] = {kind: {} for kind in _KINDS}


class StepsIn(BaseModel):
    """A step selector from a label table: true for the steps whose transcript id and step id it lists. `steps_in()` builds one."""

    keys: list[tuple[str, str]]
    """The `(transcript_id, step_id)` pairs selected."""

    _selected: frozenset[tuple[str, str]] = PrivateAttr(
        default_factory=frozenset[tuple[str, str]]
    )

    def model_post_init(self, context: Any) -> None:
        self._selected = frozenset(self.keys)

    def __call__(self, transcript_id: str, step: ReplayStep) -> bool:
        """Whether the table lists the step."""
        return (transcript_id, step.id) in self._selected


def steps_in(labels: Any) -> StepsIn:
    """A step selector for `as_scanner(steps=...)` that runs the sentinel at the steps a label table lists.

    Args:
        labels: The table: a pandas `DataFrame` with `transcript_id` and `step_id` columns, as a results dataframe has, or rows of `(transcript_id, step_id)` pairs or of mappings with those keys.
    """
    columns = getattr(labels, "columns", None)
    if columns is not None:
        missing = {"transcript_id", "step_id"} - set(cast(Iterable[str], columns))
        if missing:
            raise ValueError(
                f"A label table needs transcript_id and step_id columns; it has no {', '.join(sorted(missing))}."
            )
        pairs = zip(
            cast(Iterable[object], labels["transcript_id"]),
            cast(Iterable[object], labels["step_id"]),
            strict=True,
        )
    else:
        pairs = (_pair(row) for row in cast(Iterable[object], labels))
    return StepsIn(keys=sorted({(str(t), str(s)) for t, s in pairs}))


def _pair(row: object) -> tuple[object, object]:
    if isinstance(row, Mapping):
        mapping = cast(Mapping[str, object], row)
        return mapping["transcript_id"], mapping["step_id"]
    transcript_id, step_id = cast(tuple[object, object], row)
    return transcript_id, step_id


def step_selector(selector: F) -> F:
    """Register a step selector, so a scan that uses it can run in Scout's worker processes and be resumed in another process.

    A selector is a function `(transcript_id, step) -> bool` defined at module level, in a module or in the script that runs the scan. A scan records its module, name and file, and finds it again by importing the module, or loading the file. One defined in a notebook, REPL or `python -c` has no file to load, so `as_scanner()` refuses it.

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
            f"as_scanner({parameter}=...) takes {_KINDS[kind]} registered with {decorator}, so the scan can record it and find it again in another process; {callback!r} is not registered. replay_monitor() takes any callable."
        )
    module: str = getattr(callback, "__module__", "")
    qualname: str = getattr(callback, "__qualname__", "")
    file = source_file(module)
    if module == "__main__" and file is None:
        raise TypeError(
            f"as_scanner({parameter}=...) cannot record {qualname!r}: it was defined in a notebook, REPL or `python -c`, which another process cannot load, and a scan runs in Scout's worker processes and resumes in new ones. Define it in a module or a script."
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


def selector_from(steps: object) -> StepSelector | None:
    if steps is None:
        return None
    if isinstance(steps, StepsIn):
        return steps
    if isinstance(steps, Mapping):
        recorded = cast(Mapping[str, Any], steps)
        if "keys" in recorded:
            return StepsIn.model_validate(recorded)
        return cast(StepSelector, callback_from("step_selector", recorded))
    raise TypeError(
        f"steps must be a label selector or a registered selector; got {steps!r}."
    )


def stand_in_from(human: object) -> HumanPolicy:
    if human == "recorded" or human == "fail_closed":
        return cast(HumanPolicy, human)
    if isinstance(human, Mapping):
        return cast(
            HumanStandIn,
            callback_from("human_stand_in", cast(Mapping[str, str | None], human)),
        )
    raise TypeError(
        f"human must be 'recorded', 'fail_closed' or a registered stand-in; got {human!r}."
    )
