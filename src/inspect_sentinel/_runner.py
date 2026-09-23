from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from ._report import Action, Decision, Observation, Reported

_PRECEDENCE: dict[Action, int] = {
    "terminate": 4,
    "reject": 3,
    "modify": 2,
    "continue": 1,
}


class Observations(Sequence[Reported[Observation]]):
    """What a layer's monitors reported, in configuration order."""

    def __init__(self, items: Iterable[Reported[Observation]] = ()) -> None:
        self._items = tuple(items)

    @overload
    def __getitem__(self, index: int) -> Reported[Observation]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Reported[Observation]]: ...
    def __getitem__(
        self, index: int | slice
    ) -> Reported[Observation] | Sequence[Reported[Observation]]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Reported[Observation]]:
        return iter(self._items)

    def max_suspicion(self) -> float | None:
        """The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed."""
        scores = [
            max(o.report.suspicion.values())
            if isinstance(o.report.suspicion, dict)
            else o.report.suspicion
            for o in self._items
        ]
        return max(scores) if scores else None


class Decisions(Sequence[Reported[Decision]]):
    """What a layer's protocols decided, in configuration order."""

    def __init__(self, items: Iterable[Reported[Decision]] = ()) -> None:
        self._items = tuple(items)

    @overload
    def __getitem__(self, index: int) -> Reported[Decision]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Reported[Decision]]: ...
    def __getitem__(
        self, index: int | slice
    ) -> Reported[Decision] | Sequence[Reported[Decision]]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Reported[Decision]]:
        return iter(self._items)

    def strongest(self) -> Reported[Decision] | None:
        """The strongest decision by `terminate > reject > modify > continue`; `escalate` does not count; `None` if nobody decided."""
        ranked = [d for d in self._items if d.report.action in _PRECEDENCE]
        if not ranked:
            return None
        return max(ranked, key=lambda d: _PRECEDENCE[d.report.action])


@dataclass(frozen=True)
class Reports:
    """Both families of report from one layer."""

    observations: Observations
    """From the layer's monitors."""

    decisions: Decisions
    """From the layer's protocols."""
