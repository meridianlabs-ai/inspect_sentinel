from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Generic, NamedTuple, cast, overload

from ._report import PRECEDENCE, Decision, Observation, R_co, Reported


class _ReportSequence(Sequence[Reported[R_co]], Generic[R_co]):
    def __init__(self, items: Iterable[Reported[R_co]] = ()) -> None:
        self._items = tuple(items)

    @overload
    def __getitem__(self, index: int) -> Reported[R_co]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Reported[R_co]]: ...
    def __getitem__(
        self, index: int | slice
    ) -> Reported[R_co] | Sequence[Reported[R_co]]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Reported[R_co]]:
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._items == cast("_ReportSequence[R_co]", other)._items

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._items)!r})"


def peak_suspicion(observation: Observation) -> float:
    suspicion = observation.suspicion
    return max(suspicion.values()) if isinstance(suspicion, dict) else suspicion


class Observations(_ReportSequence[Observation]):
    """What a layer's monitors reported, in configuration order."""

    def max_suspicion(self) -> float | None:
        """The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed."""
        scores = [peak_suspicion(o.report) for o in self._items]
        return max(scores) if scores else None


class Decisions(_ReportSequence[Decision]):
    """What a layer's protocols decided, in configuration order."""

    def strongest(self) -> Reported[Decision] | None:
        """The strongest decision by `terminate > reject > modify > escalate > continue`, the first in configuration order on a tie; `None` if nobody decided."""
        if not self._items:
            return None
        return max(self._items, key=lambda d: PRECEDENCE[d.report.action])


class Reports(NamedTuple):
    """Both families of report from one layer, unpackable as `observations, decisions = await run_children(...)`."""

    observations: Observations
    """From the layer's monitors."""

    decisions: Decisions
    """From the layer's protocols."""
