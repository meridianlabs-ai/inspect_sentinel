from __future__ import annotations

import math
from collections.abc import Mapping, Set
from typing import NamedTuple, cast

from .._context import Context
from .._decorators import members, protocol
from .._report import Decision, Observation, Reported
from .._results import Observations
from .._runner import describe, run_monitors
from .._step import BeforeToolCall
from .._types import Group, Monitor, MonitorGroup, Monitors, Protocol
from .._validate import named_children


@protocol
def threshold(
    monitors: Monitor | MonitorGroup | Monitors,
    reject_at: float | Mapping[str, float],
    terminate_at: float | Mapping[str, float] | None = None,
) -> Protocol:
    """Act on suspicion scores: `reject` when any score reaches its reject threshold, `terminate` when any reaches its terminate threshold.

    A threshold is one number for every score, or a mapping that sets one per dimension. In a mapping, `"<monitor>.<dimension>"` applies to one dimension of one monitor (the monitor's instance name, as in `Reported.name`), a bare `"<dimension>"` applies to that dimension from every monitor, a monitor reporting a single number is keyed by its instance name, and `"*"` applies to anything unlisted. The most specific key wins: qualified, then bare, then `"*"`. A score no key covers never triggers. A number is shorthand for `{"*": number}`. A dimension name containing `.` must be qualified.

    Terminate is checked first. A reject or terminate is explained for the log: it names the highest-scoring score that crossed (the first in configuration order on a tie) and carries its monitor's explanation and references, e.g. `suspicion 0.95 from audit: posts credentials to an external host` when the threshold came from `"*"` or the score is a single number, and `sabotage 0.72 ≥ 0.60 from audit: edits the tests` when it came from the dimension's own key. Neither sets `message`, so the agent gets the host's default "not approved" text and learns nothing about what the monitors saw.

    The smallest protocol, and the one calibration feeds: the constants come from a benign score distribution, which is why they live here and not on a monitor. Every monitor, and every function of a `MonitorGroup`, must watch `BeforeToolCall`, since that is the only stage a `reject` is legal at.

    Args:
        monitors: The monitor, or monitors, whose scores are thresholded.
        reject_at: Suspicion at or above which the call is rejected: one number, or a mapping by dimension.
        terminate_at: Suspicion at or above which the sample is ended, in the same form; `None` never terminates. Where it and `reject_at` both cover a score, it must be above `reject_at`.
    """
    named = named_children(monitors, "monitor")
    if not named:
        raise ValueError("threshold needs at least one child.")
    names = {name for name, _ in named}
    rejects = _thresholds("reject_at", reject_at, names)
    terminates = (
        None
        if terminate_at is None
        else _thresholds("terminate_at", terminate_at, names)
    )
    if terminates is not None:
        for key in {**rejects, **terminates}:
            name, dimension = _split(key, names)
            low = _covering(rejects, name, dimension)
            high = _covering(terminates, name, dimension)
            if low is not None and high is not None and high.value <= low.value:
                raise ValueError(
                    f"threshold's terminate_at must be above reject_at wherever both apply, or reject would never fire; for {key!r} got reject_at={low.value}, terminate_at={high.value}."
                )
    for name, child in named:
        grouped = isinstance(child, Group)
        for member in members(child):
            if BeforeToolCall not in member.accepted:
                label = describe(name, member.function.__name__, grouped)
                raise TypeError(
                    f"threshold acts before tool calls; monitor {label} never watches that stage"
                )

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        observations = await run_monitors(monitors, context, step)
        if not observations:
            return None
        if terminates is not None:
            crossing = _crossing(observations, terminates)
            if crossing is not None:
                return Decision.terminate(
                    crossing.explain(), references=crossing.reported.report.references
                )
        crossing = _crossing(observations, rejects)
        if crossing is not None:
            return Decision.reject(
                crossing.explain(), references=crossing.reported.report.references
            )
        return Decision.proceed()

    return decide


_DEFAULT = "*"


def _thresholds(
    parameter: str, value: float | Mapping[str, float], names: Set[str]
) -> dict[str, float]:
    thresholds = (
        dict(cast(Mapping[object, object], value))
        if isinstance(value, Mapping)
        else {_DEFAULT: value}
    )
    if not thresholds:
        raise ValueError(f"threshold's {parameter} must set at least one threshold.")
    for key, limit in thresholds.items():
        if not isinstance(key, str) or not _well_formed(key):
            raise ValueError(
                f"threshold's {parameter} has a malformed key {key!r}; a key is '*', a dimension, a monitor's instance name, or '<monitor>.<dimension>'."
            )
        if "." in key and _split(key, names).dimension is None:
            raise ValueError(
                f"threshold's {parameter} key {key!r} names no monitor of this threshold; its monitors are {sorted(names)}, and a dimension containing '.' must be qualified by one."
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
        ):
            raise ValueError(
                f"threshold's {parameter} thresholds must be finite numbers; got {key!r}: {limit!r}."
            )
    return cast(dict[str, float], thresholds)


def _well_formed(key: str) -> bool:
    if key == _DEFAULT:
        return True
    return _DEFAULT not in key and all(key.split("."))


class _Scored(NamedTuple):
    name: str
    dimension: str | None


def _split(key: str, names: Set[str]) -> _Scored:
    qualifiers = [name for name in names if key.startswith(f"{name}.")]
    if not qualifiers:
        return _Scored(key, None)
    name = max(qualifiers, key=len)
    return _Scored(name, key[len(name) + 1 :])


class _Limit(NamedTuple):
    key: str
    value: float


def _covering(
    thresholds: Mapping[str, float], name: str, dimension: str | None
) -> _Limit | None:
    candidates = (
        (name, _DEFAULT)
        if dimension is None
        else (f"{name}.{dimension}", dimension, _DEFAULT)
    )
    for candidate in candidates:
        if candidate in thresholds:
            return _Limit(candidate, thresholds[candidate])
    return None


class _Crossing(NamedTuple):
    reported: Reported[Observation]
    dimension: str | None
    score: float
    limit: _Limit

    def explain(self) -> str:
        if self.dimension is None or self.limit.key == _DEFAULT:
            reason = f"suspicion {self.score:.2f} from {self.reported.name}"
        else:
            reason = f"{self.dimension} {self.score:.2f} ≥ {self.limit.value:.2f} from {self.reported.name}"
        explanation = self.reported.report.explanation
        return f"{reason}: {explanation}" if explanation else reason


def _crossing(
    observations: Observations, thresholds: Mapping[str, float]
) -> _Crossing | None:
    highest: _Crossing | None = None
    for reported in observations:
        suspicion = reported.report.suspicion
        scores = (
            [(None, suspicion)]
            if not isinstance(suspicion, dict)
            else list(suspicion.items())
        )
        for dimension, score in scores:
            limit = _covering(thresholds, reported.name, dimension)
            if limit is None or score < limit.value:
                continue
            if highest is None or score > highest.score:
                highest = _Crossing(reported, dimension, score, limit)
    return highest
