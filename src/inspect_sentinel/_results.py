from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from logging import getLogger
from typing import Generic, NamedTuple, cast, overload

from ._report import PRECEDENCE, Decision, Failed, Observation, R_co, Reported

logger = getLogger(__name__)


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
    """What a layer's monitors reported, in configuration order.

    When a monitor failed, reading the observations raises, so a protocol that reads them fails closed rather than deciding without the failed monitor. A protocol that decides anyway checks `failed` and reads `succeeded`.
    """

    def __init__(
        self, items: Iterable[Reported[Observation]] = (), failed: Iterable[Failed] = ()
    ) -> None:
        super().__init__(items)
        self._failed = tuple(failed)

    @property
    def failed(self) -> tuple[Failed, ...]:
        """The monitor functions that raised instead of reporting, in configuration order."""
        return self._failed

    @property
    def succeeded(self) -> Observations:
        """The observations of the monitors that did not fail, readable whether or not any failed."""
        return Observations(self._items)

    @overload
    def __getitem__(self, index: int) -> Reported[Observation]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Reported[Observation]]: ...
    def __getitem__(
        self, index: int | slice
    ) -> Reported[Observation] | Sequence[Reported[Observation]]:
        self._check()
        return super().__getitem__(index)

    def __len__(self) -> int:
        self._check()
        return super().__len__()

    def __iter__(self) -> Iterator[Reported[Observation]]:
        self._check()
        return super().__iter__()

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (
            super().__eq__(other)
            and self._failed == cast("Observations", other)._failed
        )

    def __repr__(self) -> str:
        if not self._failed:
            return super().__repr__()
        return f"{type(self).__name__}({list(self._items)!r}, failed={list(self._failed)!r})"

    def max_suspicion(self) -> float | None:
        """The highest suspicion reported, taking the highest dimension of a structured score; `None` if nobody observed."""
        self._check()
        scores = [peak_suspicion(o.report) for o in self._items]
        return max(scores) if scores else None

    def _check(self) -> None:
        if self._failed:
            raise MonitorFailedError(self._failed)


class MonitorFailedError(RuntimeError):
    """Raised on reading the observations of a layer in which a monitor failed. Check `Observations.failed` first to decide without the failed monitors."""

    def __init__(self, failed: Sequence[Failed]) -> None:
        listed = "; ".join(
            f"{f.path!r} (function {f.function!r}): {type(f.error).__name__}: {f.error}"
            for f in failed
        )
        super().__init__(
            f"Cannot read the observations, since {len(failed)} monitor(s) failed: {listed}. A protocol that decides without failed monitors checks `observations.failed` and reads `observations.succeeded`."
        )
        self.failed = tuple(failed)
        self.__cause__ = failed[0].error


_warned: set[tuple[str, type[Exception]]] = set()


def warn_failed(failed: Iterable[Failed]) -> None:
    # for compositions that only record their monitors: the failure is already
    # in the transcript, and this is the one place it is surfaced live
    for f in failed:
        key = (f.path, type(f.error))
        if key not in _warned:
            _warned.add(key)
            logger.warning(
                f"Monitor {f.path!r} (function {f.function!r}) failed with {type(f.error).__name__}: {f.error}. Nothing reads its observations here, so the step continued; later {type(f.error).__name__} failures from it are recorded but not logged."
            )


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
