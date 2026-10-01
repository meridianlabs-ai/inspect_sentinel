from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeAlias, cast, final

from ._context import Context
from ._report import Decision, Observation, Report
from ._step import AfterToolCall, BeforeToolCall, Step

Monitor: TypeAlias = (
    Callable[[Context, BeforeToolCall], Awaitable[Observation | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Observation | None]]
)
"""A monitor: observes a step at one stage and reports a suspicion score, or abstains."""

Protocol: TypeAlias = (
    Callable[[Context, Step], Awaitable[Decision | None]]
    | Callable[[Context, BeforeToolCall], Awaitable[Decision | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Decision | None]]
)
"""A control protocol, not `typing.Protocol`: decides what happens at a step, or abstains. Annotating `Step` runs it at every stage."""

SentinelFunction: TypeAlias = Callable[[Context, Step], Awaitable[Report | None]]


class Group:
    def __init__(
        self, functions: tuple[Callable[..., Awaitable[Report | None]], ...]
    ) -> None:
        if not functions:
            raise ValueError(f"{type(self).__name__} needs at least one function.")
        seen: set[str] = set()
        for function in functions:
            name = getattr(function, "__name__", repr(function))
            if name in seen:
                raise ValueError(
                    f"Duplicate function name {name!r} in one {type(self).__name__}; the functions of an instance are told apart by name, so give each a distinct one."
                )
            seen.add(name)
        self._functions = functions

    @property
    def functions(self) -> tuple[Callable[..., Awaitable[Report | None]], ...]:
        return self._functions


@final
class MonitorGroup(Group):
    """Several monitor functions that form one monitor instance.

    Return one from a `@monitor` factory in place of a single function. Together the functions are one configured instance, sharing its name, path and `store_as` namespace; each runs, in the order given, at the stage it watches, sequentially since they share one store, and each report records which `function` made it. A group is not callable: hand it to a protocol, which runs it through `run_monitors` or `run_children`.
    """

    def __init__(self, *functions: Monitor) -> None:
        """Group monitor functions into one instance.

        Args:
            *functions: At least one monitor function, each with a distinct `__name__`.

        Raises:
            ValueError: If there are no functions or two share a `__name__`.
        """
        super().__init__(functions)

    @property
    def functions(self) -> tuple[Monitor, ...]:
        """The member functions, in the order they run."""
        return cast(tuple[Monitor, ...], self._functions)


@final
class ProtocolGroup(Group):
    """Several protocol functions that form one protocol instance.

    Return one from a `@protocol` factory in place of a single function. As with `MonitorGroup`, the functions share one name, path and `store_as` namespace and run in the order given at the stages they accept; a function that returns `terminate` or calls `decide_final()` ends the instance's run, and those after it do not run and are not recorded. A group is not callable: hand it to a protocol, which runs it through `run_protocols` or `run_children`.
    """

    def __init__(self, *functions: Protocol) -> None:
        """Group protocol functions into one instance.

        Args:
            *functions: At least one protocol function, each with a distinct `__name__`.

        Raises:
            ValueError: If there are no functions or two share a `__name__`.
        """
        super().__init__(functions)

    @property
    def functions(self) -> tuple[Protocol, ...]:
        """The member functions, in the order they run."""
        return cast(tuple[Protocol, ...], self._functions)


Monitors: TypeAlias = (
    Mapping[str, Monitor | MonitorGroup] | Sequence[Monitor | MonitorGroup]
)
"""Monitors handed to a protocol, named by mapping key or by registry name without its package prefix."""

Protocols: TypeAlias = (
    Mapping[str, Protocol | ProtocolGroup] | Sequence[Protocol | ProtocolGroup]
)
"""Protocols handed to a protocol, named the same way."""

Sentinel: TypeAlias = Monitor | MonitorGroup | Protocol | ProtocolGroup
"""A monitor or protocol, or a group of either."""

Sentinels: TypeAlias = Sentinel | Mapping[str, Sentinel] | Sequence[Sentinel]
"""What `Task(sentinel=)` and the compositions accept: one sentinel, or a sequence or mapping of them, named by mapping key or by registry name without its package prefix."""
