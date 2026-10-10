from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, cast, get_args

from inspect_ai._util.registry import registry_info, registry_unqualified_name

from ._decorators import Outcome, members, outcomes
from ._host import Recorder
from ._report import DECISION_CLASSES, Decision, Failed, Report, Reported
from ._step import AfterToolCall, BeforeToolCall, Step
from ._types import Protocol, Sentinel

if TYPE_CHECKING:
    from ._context import Context


class EscalationRecorder:
    # forwards every record, keeping the escalates of the step, so an
    # unhandled one at the root can say where it came from
    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder
        self.escalates: list[Reported[Decision]] = []

    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        self._recorder.record(context, factory, step, reported)
        if (
            isinstance(reported.report, Decision)
            and reported.report.action == "escalate"
        ):
            self.escalates.append(cast(Reported[Decision], reported))

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        self._recorder.failed(context, factory, step, failed)

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        self._recorder.cancelled(context, factory, step, name)

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        self._recorder.bypassed(context, factory, step, name)

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        self._recorder.superseded(context, factory, step, reported)


def sources(escalates: Sequence[Reported[Decision]]) -> list[Reported[Decision]]:
    # an escalate reached the root when every layer above it escalated too;
    # of those, the innermost are where the escalation came from
    paths = {e.path for e in escalates}
    reached = [e for e in escalates if all(a in paths for a in _ancestors(e.path))]
    return [
        e
        for e in reached
        if not any(r.path != e.path and e.path in _ancestors(r.path) for r in reached)
    ]


def _ancestors(path: str) -> list[str]:
    if not path:
        return []
    parts = path.split("/")
    return ["", *("/".join(parts[:i]) for i in range(1, len(parts)))]


def unhandled(escalations: Sequence[Reported[Decision]]) -> Decision:
    # what run_sentinel returns for an escalate at the root: a terminate, since
    # nothing was configured to answer it
    named = "; ".join(
        f"{e.path or e.name}: {e.report.explanation}"
        if e.report.explanation
        else e.path or e.name
        for e in escalations
    )
    return Decision.terminate(
        f"unhandled escalation from {named}. To say what an escalation means, end the configuration with human() or handle_escalation('terminate' | 'continue') in a sequential().",
        references=[r for e in escalations for r in e.report.references],
    )


# what sequential and concurrent combine, so the build check follows their
# children; a protocol without it is judged by its return annotation alone
COMPOSITION_ATTR = "__sentinel_composition__"
Mode = Literal["in_order", "at_once"]
F = TypeVar("F")


def composed(run: F, mode: Mode, children: Sequence[tuple[str, Sentinel]]) -> F:
    setattr(run, COMPOSITION_ATTR, (mode, tuple(children)))
    return run


class _Flow(NamedTuple):
    # (path below the instance, whether its annotation names Escalate) for
    # each escalate that can leave the instance
    escalates: frozenset[tuple[str, bool]]
    # always returns a decision other than escalate
    decides: bool
    # always returns a decision that outranks escalate
    outranks: bool


_NOTHING = _Flow(frozenset(), False, False)
_OUTRANK: frozenset[Outcome] = frozenset({"terminate", "reject", "modify"})
_STAGE_NAMES: dict[type[Any], str] = {
    BeforeToolCall: "tool_call",
    AfterToolCall: "tool_result",
}


def check_handled(root: Protocol) -> None:
    sources: dict[str, list[str]] = {}
    for stage in get_args(Step):
        for path, named in _flow(root, stage).escalates:
            if named:
                label = path or registry_unqualified_name(registry_info(root))
                sources.setdefault(label, []).append(_STAGE_NAMES[stage])
    if sources:
        listed = ", ".join(
            f"{path} ({', '.join(stages)})" for path, stages in sorted(sources.items())
        )
        raise ValueError(
            f"This sentinel can escalate with nothing to handle the escalation, which would end the sample: {listed}. "
            "Each is annotated to return Escalate, and nothing after it always decides. "
            "End the configuration with human() or handle_escalation('terminate' | 'continue') in a sequential(), "
            "or take Escalate out of the annotation of a protocol that never escalates."
        )


def _flow(sentinel: Sentinel, stage: type[Any]) -> _Flow:
    composition = getattr(sentinel, COMPOSITION_ATTR, None)
    if composition is not None:
        mode, children = cast(
            tuple[Mode, tuple[tuple[str, Sentinel], ...]], composition
        )
        return (
            _in_order(children, stage)
            if mode == "in_order"
            else _at_once(children, stage)
        )
    if registry_info(sentinel).type == "monitor":
        return _NOTHING
    running = [m for m in members(sentinel) if stage in m.accepted]
    # a group's functions combine as concurrent's children do, at one path
    return _together([("", _declared(outcomes(m.function))) for m in running])


def _declared(declared: frozenset[Outcome]) -> _Flow:
    escalates = (
        frozenset({("", not set(DECISION_CLASSES) <= declared)})
        if "escalate" in declared
        else frozenset[tuple[str, bool]]()
    )
    decides = None not in declared and "escalate" not in declared
    return _Flow(escalates, decides, decides and declared <= _OUTRANK)


def _in_order(children: Sequence[tuple[str, Sentinel]], stage: type[Any]) -> _Flow:
    # the first link that always decides ends the chain, so the links after it
    # never run and the escalates before it are answered
    escalates: set[tuple[str, bool]] = set()
    for name, child in children:
        flow = _flow(child, stage)
        if flow.decides:
            return _Flow(frozenset(), True, False)
        escalates |= _below(name, flow)
    return _Flow(frozenset(escalates), False, False)


def _at_once(children: Sequence[tuple[str, Sentinel]], stage: type[Any]) -> _Flow:
    return _together([(name, _flow(child, stage)) for name, child in children])


def _together(flows: Sequence[tuple[str, _Flow]]) -> _Flow:
    # an escalate outranks continue, so it leaves unless a peer always decides
    # something stronger
    if any(flow.outranks for _, flow in flows):
        return _Flow(frozenset(), True, True)
    escalates = frozenset[tuple[str, bool]]().union(
        *(_below(name, flow) for name, flow in flows)
    )
    decides = any(flow.decides for _, flow in flows) and not escalates
    return _Flow(escalates, decides, False)


def _below(name: str, flow: _Flow) -> set[tuple[str, bool]]:
    # a group's functions are at the group's own path, named ""
    return {
        ("/".join(part for part in (name, path) if part), named)
        for path, named in flow.escalates
    }
