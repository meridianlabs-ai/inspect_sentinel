from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from inspect_ai._util.registry import (
    RegistryInfo,
    registry_info,
    registry_unqualified_name,
)

from ._context import validate_instance_name
from ._decorators import Declared, Outcome, declared, step_types
from ._report import DECISION_CLASSES, Decision
from ._step import AfterToolCall, Step
from ._types import Group, Sentinel


def validate_decision_shape(decision: Decision, step: Step) -> None:
    """Check a decision for the deterministic protocol bugs no type can express, raising `ValueError` on one.

    The decision's class must be its action's (`model_copy(update=...)` can change the action without the class), the action must be legal for the stage (`reject` and `modify` are not legal after a tool call), `modified` must be set exactly when the action is `modify`, a `modify` may rewrite only the call's arguments (its `id` and `function` must match the step's call), and `message` may be set only on a `reject`.

    Args:
        decision: The decision a layer returned.
        step: The step it decided about.
    """
    if isinstance(step, AfterToolCall) and decision.action in ("reject", "modify"):
        raise ValueError(
            f"A decision after a tool call cannot {decision.action}; the call has already run."
        )
    if not isinstance(decision, DECISION_CLASSES[decision.action]):
        raise ValueError(
            f"A {type(decision).__name__} cannot have action {decision.action!r}; build a {DECISION_CLASSES[decision.action].__name__} instead. "
            "`model_copy(update=...)` does not change a decision's class."
        )
    if (decision.action == "modify") != (decision.modified is not None):
        raise ValueError(
            "`modified` must be set exactly when the action is `modify`; "
            f"got action {decision.action!r} with modified={decision.modified!r}."
        )
    modified = decision.modified
    if modified is not None and (
        modified.id != step.call.id or modified.function != step.call.function
    ):
        raise ValueError(
            "A `modify` may change only the call's arguments; "
            f"got id {modified.id!r} and function {modified.function!r} "
            f"for call {step.call.id!r} to {step.call.function!r}."
        )
    if decision.message is not None and decision.action != "reject":
        raise ValueError(
            f"`message` may be set only on a `reject`; got action {decision.action!r}. "
            "Telling the agent something while the step proceeds needs a way to deliver it, which does not exist yet."
        )


def check_declared(function: object, decision: Decision | None, label: str) -> None:
    allowed = declared(function)
    got: Outcome = None if decision is None else decision.action
    if got not in allowed.outcomes:
        returned = (
            "None, abstaining"
            if got is None
            else f"a {DECISION_CLASSES[got].__name__} ({got!r})"
        )
        raise TypeError(
            f"protocol {label} returned {returned}, which its return annotation does not allow; it is annotated to return {describe_outcomes(allowed)}."
        )


def describe_outcomes(allowed: Declared) -> str:
    if allowed.broad:
        names = ["Decision"]
    else:
        names = [
            cls.__name__
            for action, cls in DECISION_CLASSES.items()
            if action in allowed.outcomes
        ]
    if None in allowed.outcomes:
        names.append("None")
    return " | ".join(names)


def validate_shape(decision: Decision, step: Step, label: str) -> None:
    try:
        validate_decision_shape(decision, step)
    except ValueError as ex:
        raise ValueError(f"protocol {label}: {ex}") from ex


def check_child(
    child: object, expected: Literal["monitor", "protocol"] | None
) -> tuple[RegistryInfo, frozenset[type[Any]]]:
    accepted = step_types(
        cast(Any, child)
    )  # rejects factories and undecorated functions
    info = registry_info(child)
    if expected is not None and info.type != expected:
        raise TypeError(f"Expected a {expected}, got the {info.type} {info.name!r}.")
    return info, accepted


def named_children(
    children: Sentinel | Mapping[str, Sentinel] | Iterable[Sentinel],
    expected: Literal["monitor", "protocol"] | None,
) -> list[tuple[str, Sentinel]]:
    pairs: list[tuple[object, Sentinel]]
    if callable(children) or isinstance(children, Group):
        pairs = [(None, children)]
        keyed = False
    elif isinstance(children, Mapping):
        mapping = cast(Mapping[str, Sentinel], children)
        pairs = [(key, child) for key, child in mapping.items()]
        keyed = True
    elif isinstance(children, Sequence) and not isinstance(children, str):
        pairs = [(None, child) for child in children]
        keyed = False
    else:
        raise TypeError(
            "children must be a monitor, protocol or group, a Mapping or a Sequence; a set or an iterator has no configuration order"
        )
    named: list[tuple[str, Sentinel]] = []
    seen: set[str] = set()
    for given, child in pairs:
        info, _ = check_child(child, expected)
        name = validate_instance_name(
            given if keyed else registry_unqualified_name(info)
        )
        if name in seen:
            raise ValueError(
                f"Duplicate instance name {name!r} in one layer. Give the children distinct names with a mapping."
            )
        seen.add(name)
        named.append((name, child))
    return named
