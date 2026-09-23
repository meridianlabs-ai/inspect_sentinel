from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeAlias, TypeVar, cast, get_args, get_type_hints

from inspect_ai._util.registry import (
    RegistryInfo,
    RegistryType,
    registry_add,
    registry_name,
    registry_tag,
)

from ._context import Context
from ._report import Decision, Observation, Report
from ._step import AfterToolCall, BeforeToolCall, Step

Monitor: TypeAlias = (
    Callable[[Context, BeforeToolCall], Awaitable[Observation | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Observation | None]]
)
"""A monitor: observes one kind of step and reports a suspicion score, or abstains."""

ControlProtocol: TypeAlias = (
    Callable[[Context, Step], Awaitable[Decision | None]]
    | Callable[[Context, BeforeToolCall], Awaitable[Decision | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Decision | None]]
)
"""A protocol: decides what happens at a step, or abstains. Annotating `Step` runs it at every stage."""

Monitors: TypeAlias = Mapping[str, Monitor] | Sequence[Monitor]
"""Monitors handed to a protocol, named by mapping key or by registry name."""

Protocols: TypeAlias = Mapping[str, ControlProtocol] | Sequence[ControlProtocol]
"""Protocols handed to a protocol, named the same way."""

Children: TypeAlias = (
    Mapping[str, Monitor | ControlProtocol] | Sequence[Monitor | ControlProtocol]
)
"""Monitors and protocols together, for the compositions that record either."""

P = ParamSpec("P")
SentinelT = TypeVar("SentinelT")

_STEP_TYPES = frozenset(get_args(Step))


def monitor(factory: Callable[P, Monitor]) -> Callable[P, Monitor]:
    """Register a monitor factory.

    The factory's return annotation must be `Monitor` and the function it returns must be `async`, take `(context, step)`, annotate `step` with exactly one stage payload, and be annotated to return `Observation | None`. These are checked when the factory is called. A monitor watches one stage; a concern spanning two stages is two monitors. The factory must return a fresh function on each call; a shared function would make two configured instances indistinguishable in the log and the store.

    Args:
        factory: A function returning a monitor.
    """
    return _register("monitor", factory, Observation)


def protocol(factory: Callable[P, ControlProtocol]) -> Callable[P, ControlProtocol]:
    """Register a protocol factory.

    Same contract as `@monitor`, with the returned function annotated to return `Decision | None`, and `step` may also be annotated `Step` for a protocol that runs at every stage, such as a composition that only forwards the step to its children. The factory must return a fresh function on each call; a shared function would make two configured instances indistinguishable in the log and the store.

    Args:
        factory: A function returning a protocol.
    """
    return _register("protocol", factory, Decision)


def _register(
    kind: RegistryType,
    factory: Callable[P, SentinelT],
    report_type: type[Report],
) -> Callable[P, SentinelT]:
    name = registry_name(factory, factory.__name__)
    params = list(inspect.signature(factory).parameters.keys())
    info = RegistryInfo(type=kind, name=name, metadata=dict(params=params))

    @wraps(factory)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> SentinelT:
        instance = factory(*args, **kwargs)
        _validate(cast(Callable[..., Any], instance), kind, report_type)
        registry_tag(factory, instance, info.model_copy(deep=True), *args, **kwargs)
        return instance

    registry_add(wrapper, info)
    return wrapper


def _validate(
    instance: Callable[..., Any],
    kind: RegistryType,
    report_type: type[Report],
) -> None:
    name = getattr(instance, "__name__", repr(instance))
    if not inspect.iscoroutinefunction(instance):
        raise TypeError(f"A {kind} must be an async function; {name} is not.")
    signature = inspect.signature(instance)
    parameters = list(signature.parameters)
    if len(parameters) != 2:
        raise TypeError(
            f"A {kind} takes exactly (context, step); {name} takes {parameters}."
        )
    if any(
        p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        for p in signature.parameters.values()
    ):
        raise TypeError(
            f"A {kind} takes (context, step) as positional parameters; {name} does not."
        )
    try:
        hints = get_type_hints(instance)
    except NameError as ex:
        raise TypeError(
            f"{name}: could not resolve the annotation `{ex.name}`. Import it at module level and at runtime, not under TYPE_CHECKING or inside the factory."
        ) from ex
    except TypeError as ex:
        raise TypeError(
            f"A {kind} must be a plain async function; {name} could not be introspected."
        ) from ex
    step_hint = hints.get(parameters[1])
    if step_hint is None:
        raise TypeError(
            f"The step parameter of {kind} {name} must be annotated with a stage payload."
        )
    members = set(get_args(step_hint) or (step_hint,))
    if not members or not members <= _STEP_TYPES:
        raise TypeError(
            f"The step parameter of {kind} {name} is annotated {step_hint!r}, which is not a stage payload."
        )
    if kind == "monitor" and len(members) != 1:
        raise TypeError(
            f"A monitor watches one stage; {name} annotates step as {step_hint!r}. Write one monitor per stage."
        )
    returned = hints.get("return")
    if returned is None:
        raise TypeError(
            f"A {kind} must annotate its return as {report_type.__name__} | None; {name} has no return annotation."
        )
    members = set(get_args(returned) or (returned,))
    if report_type not in members or not members <= {report_type, type(None)}:
        raise TypeError(
            f"A {kind} must be annotated to return {report_type.__name__} | None; {name} returns {returned!r}."
        )
