from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import (
    Any,
    NamedTuple,
    ParamSpec,
    cast,
    get_args,
    get_type_hints,
    overload,
)
from typing import Protocol as TypingProtocol

from inspect_ai._util.registry import (
    RegistryInfo,
    RegistryType,
    is_registry_object,
    registry_add,
    registry_has,
    registry_info,
    registry_name,
    registry_tag,
)

from ._context import Context
from ._portable import check_portable
from ._report import Decision, Observation, Report
from ._step import Step
from ._types import (
    Group,
    Monitor,
    MonitorGroup,
    Protocol,
    ProtocolGroup,
    Sentinel,
    SentinelFunction,
)

P = ParamSpec("P")

_STEP_TYPES = frozenset(get_args(Step))
# an attribute rather than registry metadata: per-member stages live in groups,
# and registry metadata would need a name<->class mapping
STEP_TYPES_ATTR = "__sentinel_step_types__"
VERSION = "version"
PORTABLE = "portable"
ENTRY_FIELDS = ("name", "params", VERSION, "meta")


class _Outside:
    pass


_OUTSIDE = _Outside()

# the function the runner is invoking; None inside its body, _OUTSIDE when no
# runner is invoking anything, so a direct call outside a run still works
_invoking: ContextVar[object] = ContextVar("sentinel_invoking", default=_OUTSIDE)


async def invoke(
    function: SentinelFunction, context: Context, step: Step
) -> Report | None:
    token = _invoking.set(function)
    try:
        return await function(context, step)
    finally:
        _invoking.reset(token)


class Member(NamedTuple):
    function: SentinelFunction
    accepted: frozenset[type[Any]]


def members(sentinel: Sentinel) -> tuple[Member, ...]:
    functions: tuple[object, ...] = (
        sentinel.functions if isinstance(sentinel, Group) else (sentinel,)
    )
    return tuple(
        Member(cast(SentinelFunction, function), step_types(cast(Any, function)))
        for function in functions
    )


class _MonitorDecorator(TypingProtocol):
    @overload
    def __call__(self, factory: Callable[P, Monitor]) -> Callable[P, Monitor]: ...
    @overload
    def __call__(
        self, factory: Callable[P, MonitorGroup]
    ) -> Callable[P, MonitorGroup]: ...


class _ProtocolDecorator(TypingProtocol):
    @overload
    def __call__(self, factory: Callable[P, Protocol]) -> Callable[P, Protocol]: ...
    @overload
    def __call__(
        self, factory: Callable[P, ProtocolGroup]
    ) -> Callable[P, ProtocolGroup]: ...
    @overload
    def __call__(
        self, factory: Callable[P, Protocol | ProtocolGroup]
    ) -> Callable[P, Protocol | ProtocolGroup]: ...


@overload
def monitor(factory: Callable[P, Monitor], /) -> Callable[P, Monitor]: ...
@overload
def monitor(factory: Callable[P, MonitorGroup], /) -> Callable[P, MonitorGroup]: ...
@overload
def monitor(
    *, name: str | None = None, version: int = 0, portable: bool = True
) -> _MonitorDecorator: ...
def monitor(
    factory: Callable[P, Monitor | MonitorGroup] | None = None,
    /,
    *,
    name: str | None = None,
    version: int = 0,
    portable: bool = True,
) -> Callable[P, Monitor | MonitorGroup] | _MonitorDecorator:
    """Register a monitor factory.

    Use as `@monitor`, or as `@monitor(name=..., version=..., portable=...)`. The factory returns a function that must be `async`, take `(context, step)`, annotate `step` with exactly one stage payload, and be annotated `-> Observation`, or `-> Observation | None` if it can abstain. These are checked when the factory is called. Each function watches one stage, so `Step` or any other union is rejected; to watch several, return a `MonitorGroup` of functions, which together are one configured instance. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

    A configured function runs only through the runners while a sentinel is running: a protocol that calls one directly from its body raises `RuntimeError`, and must call it through `run_monitors` or `run_children` instead. Outside a run, as in a unit test, a direct call works.

    Args:
        factory: A function returning a monitor, or a `MonitorGroup`.
        name: The registered name, in place of the factory's `__name__`.
        version: The monitor's version, recorded in its registry metadata so a calibration can say which version it measured; bump it when a change alters the scores. Defaults to 0.
        portable: Whether the monitor can run outside an eval, in a proxy, recorded in its registry metadata. When True, the default, calling the factory first checks what its code references: a list of standard-library modules that only compute, `inspect_ai.core` and its dependencies, `anyio`, `inspect_sentinel`'s public API, and `StoreModel` and `Reference` until they move into `inspect_ai.core`, with effects only through `context`. Functions and classes of the author's own modules that it references are checked the same way, and any of them that cannot be checked is reported. When the factory itself has no source, as in the plain REPL, the check is skipped; Jupyter notebooks are checked. False opts out, for a monitor that only runs in an eval.

    Raises:
        TypeError: If `version` is not an integer, `portable` is not a bool, or a factory parameter is named `name`, `params`, `version` or `meta`, the keys of a configuration entry.
        ValueError: If `version` is negative.
        PortabilityError: When the factory is called, if the monitor is portable and its code references what a portable monitor cannot use.
    """
    if factory is not None:
        return cast(
            Callable[P, Monitor | MonitorGroup],
            _register("monitor", factory, Observation, None, 0, True),
        )

    def decorate(factory: Callable[P, Any]) -> Callable[P, Any]:
        return _register("monitor", factory, Observation, name, version, portable)

    return cast(_MonitorDecorator, decorate)


@overload
def protocol(factory: Callable[P, Protocol], /) -> Callable[P, Protocol]: ...
@overload
def protocol(factory: Callable[P, ProtocolGroup], /) -> Callable[P, ProtocolGroup]: ...
@overload
def protocol(
    factory: Callable[P, Protocol | ProtocolGroup], /
) -> Callable[P, Protocol | ProtocolGroup]: ...
@overload
def protocol(
    *, name: str | None = None, version: int = 0, portable: bool = True
) -> _ProtocolDecorator: ...
def protocol(
    factory: Callable[P, Protocol | ProtocolGroup] | None = None,
    /,
    *,
    name: str | None = None,
    version: int = 0,
    portable: bool = True,
) -> Callable[P, Protocol | ProtocolGroup] | _ProtocolDecorator:
    """Register a protocol factory.

    Same contract as `@monitor`, with each returned function annotated `-> Decision`, or `-> Decision | None` if it can abstain, and `step` may also be annotated `Step` for a protocol that runs at every stage, such as a composition that only forwards the step to its children. A function never dispatches on the payload type: a union of some stages but not all is rejected, so write a `ProtocolGroup` with one function per stage instead. A union of every stage written out is `Step`. To configure several functions as one instance, return a `ProtocolGroup`; a function among them that returns `terminate` or calls `decide_final()` ends the instance's run, and those after it do not run and are not recorded. The factory must return fresh functions on each call; a shared function would make two configured instances indistinguishable in the log and the store. A function must let a cancellation exception propagate; one that swallows it can report after a sibling has already decided `terminate`.

    As with `@monitor`, a configured function called directly from another protocol's body during a run raises `RuntimeError`; call it through `run_protocols` or `run_children`.

    Args:
        factory: A function returning a protocol, or a `ProtocolGroup`.
        name: The registered name, in place of the factory's `__name__`.
        version: The protocol's version, recorded in its registry metadata; bump it when a change alters its decisions. Defaults to 0.
        portable: Whether the protocol can run outside an eval, checked when the factory is called as for `@monitor`, against `inspect_sentinel`'s public API and the same list. Children, passed in as arguments or configured inside the factory, are checked by their own factories, not as part of this one. Defaults to True; False opts out.

    Raises:
        TypeError: If `version` is not an integer, `portable` is not a bool, or a factory parameter is named `name`, `params`, `version` or `meta`, the keys of a configuration entry.
        ValueError: If `version` is negative.
        PortabilityError: When the factory is called, if the protocol is portable and its code references what a portable protocol cannot use.
    """
    if factory is not None:
        return cast(
            Callable[P, Protocol | ProtocolGroup],
            _register("protocol", factory, Decision, None, 0, True),
        )

    def decorate(factory: Callable[P, Any]) -> Callable[P, Any]:
        return _register("protocol", factory, Decision, name, version, portable)

    return cast(_ProtocolDecorator, decorate)


def step_types(sentinel: Sentinel) -> frozenset[type[Any]]:
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
    name: str | None,
    version: object,
    portable: object,
) -> Callable[P, Any]:
    registered_name = registry_name(factory, name or factory.__name__)
    if not isinstance(version, int) or isinstance(version, bool):
        raise TypeError(
            f"{registered_name}: version must be an integer, not {version!r}."
        )
    if version < 0:
        raise ValueError(
            f"{registered_name}: version must be 0 or more, not {version}."
        )
    if not isinstance(portable, bool):
        raise TypeError(
            f"{registered_name}: portable must be True or False, not {portable!r}."
        )
    signature = inspect.signature(factory).parameters
    for param in signature.values():
        if param.name in ENTRY_FIELDS and param.kind not in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{registered_name} cannot have a parameter named {param.name!r}, since a configuration entry uses {', '.join(repr(f) for f in ENTRY_FIELDS)} as its own keys; rename it."
            )
    params = list(signature.keys())
    info = RegistryInfo(
        type=kind,
        name=registered_name,
        metadata={"params": params, VERSION: version, PORTABLE: portable},
    )
    # configuration finds a factory by name alone, so the name must be one kind
    other: RegistryType = "protocol" if kind == "monitor" else "monitor"
    if registry_has(other, registered_name):
        raise ValueError(
            f"{registered_name!r} is already registered as a {other}; a monitor and a protocol cannot share a name."
        )
    group_type = MonitorGroup if kind == "monitor" else ProtocolGroup

    @wraps(factory)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        if portable:
            check_portable(factory, kind, registered_name)
        returned = factory(*args, **kwargs)
        instance: object
        if isinstance(returned, Group):
            if not isinstance(returned, group_type):
                raise TypeError(
                    f"A {kind} factory must return a {group_type.__name__}, not a {type(returned).__name__}; {registered_name} did."
                )
            functions = tuple(
                _configure(function, kind, report_type)
                for function in returned.functions
            )
            instance = group_type(*functions)
            accepted = frozenset[type[Any]]().union(
                *(step_types(function) for function in functions)
            )
        elif isinstance(returned, (list, tuple)):
            raise TypeError(
                f"{registered_name} returned a {type(cast(object, returned)).__name__}; to configure several functions as one {kind}, return {group_type.__name__}(...)."
            )
        else:
            instance = _configure(returned, kind, report_type)
            accepted = step_types(cast(Sentinel, instance))
        setattr(instance, STEP_TYPES_ATTR, accepted)
        registry_tag(factory, instance, info.model_copy(deep=True), *args, **kwargs)
        return instance

    registry_add(wrapper, info)
    return wrapper


def _configure(function: object, kind: RegistryType, report_type: type[Report]) -> Any:
    accepted = _validate_signature(
        cast(Callable[..., Any], function), kind, report_type
    )
    guarded = _guard(cast(SentinelFunction, function))
    setattr(guarded, STEP_TYPES_ATTR, accepted)
    return guarded


def _guard(function: SentinelFunction) -> SentinelFunction:
    @wraps(function)
    async def guarded(context: Context, step: Step) -> Report | None:
        current = _invoking.get()
        if current is _OUTSIDE:
            return await function(context, step)
        if current is not guarded:
            raise RuntimeError(
                f"{function.__name__} was called directly during a sentinel run; call children through run_monitors/run_protocols/run_children."
            )
        token = _invoking.set(None)
        try:
            return await function(context, step)
        finally:
            _invoking.reset(token)

    return guarded


def check_stages(
    kind: RegistryType,
    name: str,
    declared: frozenset[type[Any]],
    stages: frozenset[type[Any]],
) -> None:
    # a union of every stage is `Step` by value, so it cannot be told apart from
    # `Step` written out; only a proper subset with several members is partial
    if len(declared) == 1:
        return
    annotated = " | ".join(sorted(stage.__name__ for stage in declared))
    if kind == "monitor":
        raise TypeError(
            f"A monitor watches one stage; {name} annotates step as {annotated}. Return a MonitorGroup with one function per stage."
        )
    if declared != stages:
        raise TypeError(
            f"A protocol function handles one stage, or every stage; {name} annotates step as {annotated}, which covers some stages but not all. Return a ProtocolGroup with one function per stage, or annotate step as `Step` if the protocol is stage-agnostic."
        )


def _validate_signature(
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
    check_stages(kind, name, frozenset(members), _STEP_TYPES)
    returned = hints.get("return")
    if returned is None:
        raise TypeError(
            f"A {kind} must annotate its return as {report_type.__name__}, or {report_type.__name__} | None if it can abstain; {name} has no return annotation."
        )
    return_members = set(get_args(returned) or (returned,))
    if report_type not in return_members or not return_members <= {
        report_type,
        type(None),
    }:
        raise TypeError(
            f"A {kind} must be annotated to return {report_type.__name__}, or {report_type.__name__} | None if it can abstain; {name} returns {returned!r}."
        )
    return frozenset(members)
