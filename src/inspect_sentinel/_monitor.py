from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
from typing import (
    Any,
    NamedTuple,
    ParamSpec,
    TypeAlias,
    cast,
    get_args,
    get_type_hints,
)

from inspect_ai._util.registry import (
    RegistryInfo,
    RegistryType,
    is_registry_object,
    registry_add,
    registry_info,
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
"""Monitors handed to a protocol, named by mapping key or by registry name without its package prefix."""

Protocols: TypeAlias = Mapping[str, ControlProtocol] | Sequence[ControlProtocol]
"""Protocols handed to a protocol, named the same way."""

Children: TypeAlias = (
    Mapping[str, Monitor | ControlProtocol] | Sequence[Monitor | ControlProtocol]
)
"""Monitors and protocols together, for the compositions that record either."""

P = ParamSpec("P")

_STEP_TYPES = frozenset(get_args(Step))
STEP_TYPES_ATTR = "__sentinel_step_types__"


class Member(NamedTuple):
    function: Callable[[Context, Step], Awaitable[Report | None]]
    accepted: frozenset[type[Any]]


class Group:
    def __init__(self, members: Sequence[Member]) -> None:
        self.members = tuple(members)


class MonitorGroup(Group):
    async def __call__(self, context: Context, step: Step) -> Observation | None:
        raise TypeError("a group of functions runs only through the runner")


class ProtocolGroup(Group):
    async def __call__(self, context: Context, step: Step) -> Decision | None:
        raise TypeError("a group of functions runs only through the runner")


def members(sentinel: Monitor | ControlProtocol) -> tuple[Member, ...]:
    if isinstance(sentinel, Group):
        return sentinel.members
    function = cast(Callable[[Context, Step], Awaitable[Report | None]], sentinel)
    return (Member(function, step_types(sentinel)),)


def monitor(
    factory: Callable[P, Monitor | Sequence[Monitor]],
) -> Callable[P, Monitor]:
    """Register a monitor factory.

    The factory returns a function that must be `async`, take `(context, step)`, annotate `step` with exactly one stage payload, and be annotated to return `Observation | None`. These are checked when the factory is called. Each function watches one stage. A factory may instead return a non-empty sequence of such functions with distinct `__name__`s; together they are one configured instance, sharing its name, path and `store_as` namespace, and each runs, in the order given, at the stage it watches, sequentially since they share one store. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

    Args:
        factory: A function returning a monitor, or a sequence of functions that form one.
    """
    return cast(Callable[P, Monitor], _register("monitor", factory, Observation))


def protocol(
    factory: Callable[P, ControlProtocol | Sequence[ControlProtocol]],
) -> Callable[P, ControlProtocol]:
    """Register a protocol factory.

    Same contract as `@monitor`, with each returned function annotated to return `Decision | None`, and `step` may also be annotated `Step` for a protocol that runs at every stage, such as a composition that only forwards the step to its children. As with `@monitor`, the factory may return a non-empty sequence of functions with distinct `__name__`s that form one instance; a function among them that returns `terminate` or calls `final()` ends the instance's run, and those after it do not run and are not recorded. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

    Args:
        factory: A function returning a protocol, or a sequence of functions that form one.
    """
    return cast(Callable[P, ControlProtocol], _register("protocol", factory, Decision))


def step_types(sentinel: Monitor | ControlProtocol) -> frozenset[type[Any]]:
    """The step payload types a configured monitor or protocol accepts.

    Args:
        sentinel: An instance returned by a `@monitor` or `@protocol` factory.
    """
    found = getattr(sentinel, STEP_TYPES_ATTR, None)
    if found is None:
        if is_registry_object(sentinel):
            info = registry_info(sentinel)
            if info.type in ("monitor", "protocol"):
                raise TypeError(
                    f"{info.name!r} is the factory, not a configured instance; call it to configure one."
                )
            raise TypeError(
                f"{info.name!r} is a {info.type}, not a monitor or protocol."
            )
        raise TypeError(
            f"{getattr(sentinel, '__name__', sentinel)!r} has no step types recorded. Was its factory decorated with @monitor or @protocol?"
        )
    return frozenset(found)


def _register(
    kind: RegistryType,
    factory: Callable[P, object],
    report_type: type[Report],
) -> Callable[P, object]:
    name = registry_name(factory, factory.__name__)
    params = list(inspect.signature(factory).parameters.keys())
    info = RegistryInfo(type=kind, name=name, metadata=dict(params=params))

    @wraps(factory)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        instance = factory(*args, **kwargs)
        if isinstance(instance, Sequence) and not isinstance(instance, str):
            instance = _group(cast(Sequence[object], instance), kind, report_type)
            accepted = frozenset[type[Any]]().union(
                *(member.accepted for member in instance.members)
            )
        else:
            accepted = _validate(cast(Callable[..., Any], instance), kind, report_type)
        setattr(instance, STEP_TYPES_ATTR, accepted)
        registry_tag(factory, instance, info.model_copy(deep=True), *args, **kwargs)
        return instance

    registry_add(wrapper, info)
    return wrapper


def _group(
    functions: Sequence[object], kind: RegistryType, report_type: type[Report]
) -> Group:
    if not functions:
        raise ValueError(
            f"A {kind} factory that returns a sequence must return at least one function."
        )
    found: list[Member] = []
    seen: set[str] = set()
    for function in functions:
        accepted = _validate(cast(Callable[..., Any], function), kind, report_type)
        name: str = cast(Any, function).__name__
        if name in seen:
            raise ValueError(
                f"Duplicate function name {name!r} in one {kind}; the functions of an instance are told apart by name, so give each a distinct one."
            )
        seen.add(name)
        found.append(
            Member(
                cast(Callable[[Context, Step], Awaitable[Report | None]], function),
                accepted,
            )
        )
    return MonitorGroup(found) if kind == "monitor" else ProtocolGroup(found)


def _validate(
    instance: Callable[..., Any],
    kind: RegistryType,
    report_type: type[Report],
) -> frozenset[type[Any]]:
    name = getattr(instance, "__name__", repr(instance))
    if not inspect.isfunction(instance) or not inspect.iscoroutinefunction(instance):
        raise TypeError(
            f"A {kind} must be a plain async function, not a method, partial or class; {name} is not."
        )
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
    return_members = set(get_args(returned) or (returned,))
    if report_type not in return_members or not return_members <= {
        report_type,
        type(None),
    }:
        raise TypeError(
            f"A {kind} must be annotated to return {report_type.__name__} | None; {name} returns {returned!r}."
        )
    return frozenset(members)
