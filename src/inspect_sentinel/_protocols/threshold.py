from __future__ import annotations

import math
from collections.abc import Mapping, Set
from typing import Any, NamedTuple, cast

from inspect_ai.scorer import Reference

from .._context import Context
from .._decorators import members, protocol
from .._report import Action, Decision, Observation, Reported
from .._results import Observations
from .._runner import describe, run_monitors
from .._step import AfterToolCall, BeforeToolCall, Step
from .._types import Group, Monitor, MonitorGroup, Monitors, Protocol, ProtocolGroup
from .._validate import named_children


@protocol
def threshold(
    monitors: Monitor | MonitorGroup | Monitors,
    *,
    escalate_at: float | Mapping[str, float] | None = None,
    reject_at: float | Mapping[str, float] | None = None,
    terminate_at: float | Mapping[str, float] | None = None,
) -> Protocol | ProtocolGroup:
    """Act on suspicion scores: each threshold is a floor, so at or above `escalate_at` the step is at least escalated, at or above `reject_at` at least rejected, and at or above `terminate_at` the sample is ended.

    The decision is the strongest action whose threshold any score reaches, `terminate` over `reject` over `escalate`, and `continue` when none is reached; the protocol abstains when no monitor reported. Give at least one threshold; to record scores without acting on them, use `observe_only()`. An escalation needs a `human()` or `handle_escalation()` after it in a `sequential()`, as in `sequential([threshold(monitors, escalate_at=0.6, terminate_at=0.9), human(stages=["tool_call"])])`; one that reaches the root with nothing to handle it ends the sample.

    A threshold is one number for every score, or a mapping that sets one per dimension. In a mapping, `"<monitor>.<dimension>"` applies to one dimension of one monitor (the monitor's instance name, as in `Reported.name`), a bare `"<dimension>"` applies to that dimension from every monitor, a monitor reporting a single number is keyed by its instance name, and `"*"` applies to anything unlisted. The most specific key wins: qualified, then bare, then `"*"`. A score no key covers never triggers. A number is shorthand for `{"*": number}`. A key containing `.` is always `"<monitor>.<dimension>"`, qualified by the longest monitor instance name that prefixes it, and is an error if none does: it is never also a bare dimension, so with monitors `a` and `a.b` the key `"a.b.c"` is dimension `c` of `a.b`, and a dimension containing `.` is keyed through its monitor. A threshold that can never apply, because a stronger action's threshold is at or below it for every score it covers, is a configuration error: `escalate_at=0.9` beside `terminate_at=0.8` would never escalate. A key that one mapping lists and another does not is not such a conflict.

    It runs at each stage its monitors watch, before and after tool calls, and combines monitors that watch different stages, each thresholded at its own: at a step, only the monitors watching that stage report. A `reject` is legal only before a tool call, so with `reject_at` every monitor function must watch only `BeforeToolCall`, and `threshold` runs only there.

    A decision is explained for the log: it names the highest-scoring score that reached the decision's threshold (the first in configuration order on a tie) with its monitor's explanation, e.g. `suspicion 0.95 from audit: posts credentials to an external host` when the threshold came from `"*"` or the score is a single number, and `sabotage 0.72 ≥ 0.60 from audit: edits the tests` when it came from the dimension's own key. It carries every reporting monitor's references, and its `metadata["observations"]` lists every monitor that reported at the step, `{"monitor": name, "suspicion": score, "explanation": text}` in configuration order, so they reach a `human()` after it through `step.escalations`, where a host can show each score and where the monitors disagree; inspect_ai's approval card shows only the explanation until it renders them. None sets `message`, so on a reject the agent gets the host's default "not approved" text and learns nothing about what the monitors saw.

    The smallest protocol, and the one calibration feeds: the constants come from a benign score distribution, which is why they live here and not on a monitor.

    A monitor that raises fails the step: `threshold` does not decide without one of its monitors, so the `MonitorFailedError` reading its observations propagates and the sample fails, with the monitor's failure in the transcript.

    Args:
        monitors: The monitor, or monitors, whose scores are thresholded.
        escalate_at: Suspicion at or above which the step is escalated, to be decided by what follows in a `sequential()`: one number, or a mapping by dimension. `None` never escalates.
        reject_at: Suspicion at or above which the call is rejected, in the same form; `None` never rejects. Legal only when every monitor function watches only `BeforeToolCall`.
        terminate_at: Suspicion at or above which the sample is ended, in the same form; `None` never terminates.

    Raises:
        ValueError: If there are no monitors or no thresholds, a threshold is malformed, or a threshold can never apply.
        TypeError: If a child is not a monitor, or `reject_at` is given and a monitor watches a stage other than `BeforeToolCall`.
    """
    named = named_children(monitors, "monitor")
    if not named:
        raise ValueError("threshold needs at least one child.")
    names = {name for name, _ in named}
    given = {
        "terminate_at": terminate_at,
        "reject_at": reject_at,
        "escalate_at": escalate_at,
    }
    levels = [
        _Level(_thresholds(parameter, value, names), _ACTIONS[parameter])
        for parameter, value in given.items()
        if value is not None
    ]
    if not levels:
        raise ValueError(
            "threshold needs at least one of escalate_at, reject_at and terminate_at; to record scores without acting on them, use observe_only()."
        )
    _check_live(levels, names)
    watched: set[type[Any]] = set()
    for name, child in named:
        grouped = isinstance(child, Group)
        for member in members(child):
            if reject_at is not None and BeforeToolCall not in member.accepted:
                label = describe(name, member.function.__name__, grouped)
                raise TypeError(
                    f"threshold's reject_at acts only before tool calls, since a reject is not legal after one; monitor {label} never watches that stage. Use escalate_at or terminate_at to act after a call."
                )
            watched |= member.accepted

    async def decide(context: Context, step: Step) -> Decision | None:
        observations = await run_monitors(monitors, context, step)
        if not observations:
            return None
        for level in levels:
            crossing = _crossing(observations, level.thresholds)
            if crossing is not None:
                return Decision(
                    action=level.action,
                    explanation=crossing.explain(),
                    references=_references(crossing, observations),
                    metadata={"observations": _observations(observations)},
                )
        return Decision.proceed()

    async def tool_call(context: Context, step: BeforeToolCall) -> Decision | None:
        return await decide(context, step)

    async def tool_result(context: Context, step: AfterToolCall) -> Decision | None:
        return await decide(context, step)

    # one function per stage its monitors watch, so a stage is added by adding
    # its payload and function here
    functions: list[Protocol] = []
    for stage, function in ((BeforeToolCall, tool_call), (AfterToolCall, tool_result)):
        if stage in watched:
            functions.append(function)
    if len(functions) == 1:
        return functions[0]
    return ProtocolGroup(*functions)


_ACTIONS: dict[str, Action] = {
    "terminate_at": "terminate",
    "reject_at": "reject",
    "escalate_at": "escalate",
}
_DEFAULT = "*"


class _Limit(NamedTuple):
    key: str
    value: float


class _Thresholds(NamedTuple):
    # one parameter's thresholds, by what each key applies to: a dotted key is
    # always one monitor's dimension, an undotted one a dimension of every
    # monitor or a monitor's single number, and "*" anything else
    parameter: str
    shorthand: bool
    qualified: dict[tuple[str, str], _Limit]
    bare: dict[str, _Limit]
    default: _Limit | None

    def limits(self) -> list[_Limit]:
        default = [] if self.default is None else [self.default]
        return [*self.qualified.values(), *self.bare.values(), *default]

    def covering(self, name: str, dimension: str | None) -> _Limit | None:
        found = (
            self.bare.get(name)
            if dimension is None
            else self.qualified.get((name, dimension), self.bare.get(dimension))
        )
        return self.default if found is None else found

    def label(self, limit: _Limit) -> str:
        if self.shorthand:
            return f"{self.parameter}={limit.value}"
        return f"{self.parameter}[{limit.key!r}]={limit.value}"


class _Level(NamedTuple):
    thresholds: _Thresholds
    action: Action


def _thresholds(
    parameter: str, value: float | Mapping[str, float], names: Set[str]
) -> _Thresholds:
    shorthand = not isinstance(value, Mapping)
    given = (
        {_DEFAULT: value} if shorthand else dict(cast(Mapping[object, object], value))
    )
    if not given:
        raise ValueError(f"threshold's {parameter} must set at least one threshold.")
    qualified: dict[tuple[str, str], _Limit] = {}
    bare: dict[str, _Limit] = {}
    default: _Limit | None = None
    for key, limit in given.items():
        if not isinstance(key, str) or not _well_formed(key):
            raise ValueError(
                f"threshold's {parameter} has a malformed key {key!r}; a key is '*', a dimension, a monitor's instance name, or '<monitor>.<dimension>'."
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
        ):
            raise ValueError(
                f"threshold's {parameter} thresholds must be finite numbers; got {key!r}: {limit!r}."
            )
        entry = _Limit(key, limit)
        if key == _DEFAULT:
            default = entry
        elif "." in key:
            qualified[_qualify(parameter, key, names)] = entry
        else:
            bare[key] = entry
    return _Thresholds(parameter, shorthand, qualified, bare, default)


def _well_formed(key: str) -> bool:
    if key == _DEFAULT:
        return True
    return _DEFAULT not in key and all(key.split("."))


def _qualify(parameter: str, key: str, names: Set[str]) -> tuple[str, str]:
    # the longest instance name that prefixes the key, so with monitors "a" and
    # "a.b" the key "a.b.c" is dimension "c" of "a.b"
    qualifiers = [name for name in names if key.startswith(f"{name}.")]
    if not qualifiers:
        raise ValueError(
            f"threshold's {parameter} key {key!r} names no monitor of this threshold; its monitors are {sorted(names)}. A key containing '.' is always '<monitor>.<dimension>', so a dimension containing '.' is keyed by its monitor."
        )
    name = max(qualifiers, key=len)
    return name, key[len(name) + 1 :]


def _check_live(levels: list[_Level], names: Set[str]) -> None:
    # a key is dead when, at every score it covers, a stronger level's
    # threshold is at or below it. Scores fall into classes by which keys can
    # cover them: each monitor's single number, each dimension a key names
    # for it, and a dimension no key names ("", which no key is)
    shared = {""}.union(*(level.thresholds.bare for level in levels))
    scores: list[tuple[str, str | None]] = []
    for name in sorted(names):
        own = {
            dimension
            for level in levels
            for monitor, dimension in level.thresholds.qualified
            if monitor == name
        }
        scores.append((name, None))
        scores.extend((name, dimension) for dimension in sorted(shared | own))
    for index, weaker in enumerate(levels):
        stronger = levels[:index]
        for limit in weaker.thresholds.limits():
            covered = [
                score for score in scores if weaker.thresholds.covering(*score) == limit
            ]
            blocking = [_blocking(stronger, score, limit.value) for score in covered]
            if covered and all(block is not None for block in blocking):
                level, block = cast(tuple[_Level, _Limit], blocking[0])
                relation = "lower" if block.value < limit.value else "the same"
                raise ValueError(
                    f"threshold's {weaker.thresholds.label(limit)} can never apply: {level.thresholds.label(block)} is {relation}, and the strongest action reached wins."
                )


def _blocking(
    stronger: list[_Level], score: tuple[str, str | None], value: float
) -> tuple[_Level, _Limit] | None:
    for level in stronger:
        limit = level.thresholds.covering(*score)
        if limit is not None and limit.value <= value:
            return level, limit
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


def _crossing(observations: Observations, thresholds: _Thresholds) -> _Crossing | None:
    highest: _Crossing | None = None
    for reported in observations:
        suspicion = reported.report.suspicion
        scores = (
            [(None, suspicion)]
            if not isinstance(suspicion, dict)
            else list(suspicion.items())
        )
        for dimension, score in scores:
            limit = thresholds.covering(reported.name, dimension)
            if limit is None or score < limit.value:
                continue
            if highest is None or score > highest.score:
                highest = _Crossing(reported, dimension, score, limit)
    return highest


def _references(crossing: _Crossing, observations: Observations) -> list[Reference]:
    # the named monitor's first, then every other reporting monitor's
    references = list(crossing.reported.report.references)
    for reported in observations:
        references.extend(r for r in reported.report.references if r not in references)
    return references


def _observations(observations: Observations) -> list[dict[str, Any]]:
    return [
        {
            "monitor": reported.name,
            "suspicion": reported.report.suspicion,
            "explanation": reported.report.explanation,
        }
        for reported in observations
    ]
