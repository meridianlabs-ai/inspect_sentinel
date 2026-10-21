from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias, TypeVar, cast

from pydantic import BaseModel, PrivateAttr

from inspect_sentinel import HumanPolicy, HumanStandIn, ReplayStep

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
    """Register a step selector, so a scan that uses it can be resumed and rebuilt in another process.

    A selector is a module-level function `(transcript_id, step) -> bool`. A scan records it by its import path and finds it again by importing its module.

    Args:
        selector: The selector.
    """
    return _register("step_selector", selector)


def human_stand_in(stand_in: F) -> F:
    """Register a stand-in for the person `human()` asks, so a scan that uses it can be resumed and rebuilt in another process.

    A stand-in is a module-level `async` function `(step, escalations, choices) -> HumanAnswer`, as `HumanStandIn` describes. A scan records it by its import path and finds it again by importing its module.

    Args:
        stand_in: The stand-in.
    """
    return _register("human_stand_in", stand_in)


def _register(kind: str, function: F) -> F:
    qualname = getattr(function, "__qualname__", "")
    module = getattr(function, "__module__", None)
    if not module or not qualname or "<" in qualname:
        raise TypeError(
            f"{_KINDS[kind].capitalize()} must be a function defined at module level, so a resumed scan can import it; {function!r} is not."
        )
    name = f"{module}:{qualname}"
    setattr(function, _NAME, name)
    _registered[kind][name] = function
    return function


def registered_name(kind: str, function: object, parameter: str) -> str:
    name = getattr(function, _NAME, None)
    if not isinstance(name, str) or _registered[kind].get(name) is not function:
        decorator = "@step_selector" if kind == "step_selector" else "@human_stand_in"
        raise TypeError(
            f"as_scanner({parameter}=...) takes {_KINDS[kind]} registered with {decorator}, so the scan can record it and find it again when it resumes or runs in another process; {function!r} is not registered. replay_monitor() takes any callable."
        )
    return name


def lookup(kind: str, name: str) -> Callable[..., Any]:
    found = _registered[kind].get(name)
    if found is None:
        module = name.partition(":")[0]
        try:
            importlib.import_module(module)
        except ImportError as ex:
            raise ValueError(
                f"Cannot find {_KINDS[kind]} {name!r}: its module does not import ({ex})."
            ) from ex
        found = _registered[kind].get(name)
    if found is None:
        raise ValueError(
            f"Cannot find {_KINDS[kind]} {name!r}: importing its module did not register it."
        )
    return found


def selector_from(steps: object) -> StepSelector | None:
    if steps is None:
        return None
    if isinstance(steps, StepsIn):
        return steps
    if isinstance(steps, Mapping):
        return StepsIn.model_validate(steps)
    if isinstance(steps, str):
        return cast(StepSelector, lookup("step_selector", steps))
    raise TypeError(
        f"steps must be a label selector or a registered name; got {steps!r}."
    )


def stand_in_from(human: str) -> HumanPolicy:
    if human == "recorded" or human == "fail_closed":
        return human
    return cast(HumanStandIn, lookup("human_stand_in", human))
