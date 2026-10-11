from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeVar, cast, get_args

from inspect_ai._util.registry import registry_info, registry_unqualified_name

from ._decorators import Outcome, declared, members
from ._report import Decision, Reported, Terminate
from ._step import AfterToolCall, BeforeToolCall, Step
from ._types import Protocol, Sentinel


@dataclass(eq=False)
class Invocation:
    # one call of one protocol function during a step, under the call that ran
    # it; the functions of a group, and repeat calls at one path, are told apart
    parent: Invocation | None
    decided: Reported[Decision] | None = None


# the invocation whose function is running, which the runner's children take
# as their parent; task groups copy it to the tasks they start
_current: ContextVar[Invocation | None] = ContextVar(
    "sentinel_invocation", default=None
)
# the invocations of the step that decided, in the order they decided
_decided: ContextVar[list[Invocation] | None] = ContextVar(
    "sentinel_decided", default=None
)


def invocation() -> Invocation:
    return Invocation(parent=_current.get())


@contextmanager
def running(node: Invocation) -> Generator[None]:
    token = _current.set(node)
    try:
        yield
    finally:
        _current.reset(token)


def decided(node: Invocation, reported: Reported[Decision]) -> None:
    node.decided = reported
    found = _decided.get()
    if found is not None:
        found.append(node)


@contextmanager
def collecting() -> Generator[list[Invocation]]:
    # a step starts with no invocation above it, so one run from inside
    # another step's protocol is not taken for part of that step
    found: list[Invocation] = []
    decided_token = _decided.set(found)
    current_token = _current.set(None)
    try:
        yield found
    finally:
        _current.reset(current_token)
        _decided.reset(decided_token)


def sources(found: Sequence[Invocation]) -> list[Reported[Decision]]:
    # an escalate reached the root when every invocation above it escalated
    # too; of those, the innermost are where the escalation came from
    reached = [node for node in found if _escalated(node) and _reaches(node)]
    inner = [
        node
        for node in reached
        if not any(node in _ancestors(other) for other in reached)
    ]
    return [node.decided for node in inner if node.decided is not None]


def _escalated(node: Invocation) -> bool:
    return node.decided is not None and node.decided.report.action == "escalate"


def _reaches(node: Invocation) -> bool:
    return all(_escalated(ancestor) for ancestor in _ancestors(node))


def _ancestors(node: Invocation) -> list[Invocation]:
    ancestors: list[Invocation] = []
    parent = node.parent
    while parent is not None:
        ancestors.append(parent)
        parent = parent.parent
    return ancestors


def unhandled(escalations: Sequence[Reported[Decision]]) -> Terminate:
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
    # what the instance can return at the stage: actions, and None when it
    # can abstain; it includes escalate exactly when escalates is not empty
    outcomes: frozenset[Outcome]

    @property
    def decides(self) -> bool:
        # always returns a decision other than escalate
        return None not in self.outcomes and "escalate" not in self.outcomes

    @property
    def outranks(self) -> bool:
        # always returns a decision that outranks escalate
        return self.decides and self.outcomes <= _OUTRANK


_OUTRANK: frozenset[Outcome] = frozenset({"terminate", "reject", "modify"})
_ABSTAINS = _Flow(frozenset(), frozenset({None}))
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


def undecided_stages(sentinel: Sentinel, stages: Iterable[type[Any]]) -> list[str]:
    # the stages, by name, at which the sentinel may abstain or escalate
    return [
        _STAGE_NAMES[stage] for stage in stages if not _flow(sentinel, stage).decides
    ]


def _flow(sentinel: Sentinel, stage: type[Any]) -> _Flow:
    if registry_info(sentinel).type == "monitor":
        return _ABSTAINS
    running = [m for m in members(sentinel) if stage in m.accepted]
    flows = [("", _function(m.function, stage)) for m in running]
    # a group's functions combine by the strongest, at the group's own path
    return flows[0][1] if len(flows) == 1 else _together(flows)


def _function(function: object, stage: type[Any]) -> _Flow:
    composition = getattr(function, COMPOSITION_ATTR, None)
    if composition is not None:
        mode, children = cast(
            tuple[Mode, tuple[tuple[str, Sentinel], ...]], composition
        )
        if mode == "in_order":
            return _in_order(children, stage)
        return _together([(name, _flow(child, stage)) for name, child in children])
    found = declared(function)
    escalates = (
        frozenset({("", not found.broad)})
        if "escalate" in found.outcomes
        else frozenset[tuple[str, bool]]()
    )
    return _Flow(escalates, found.outcomes)


def _in_order(children: Sequence[tuple[str, Sentinel]], stage: type[Any]) -> _Flow:
    # a link's decision other than escalate ends the chain, so the first link
    # that always decides answers the escalates before it and the links after
    # it never run
    escalates: set[tuple[str, bool]] = set()
    outcomes: set[Outcome] = set()
    for name, child in children:
        flow = _flow(child, stage)
        outcomes |= flow.outcomes - {"escalate", None}
        if flow.decides:
            return _Flow(frozenset(), frozenset(outcomes))
        escalates |= _below(name, flow)
    # reaching the end, the chain returns the last escalate, continue when only
    # monitors took part, or abstains
    outcomes |= {"continue", None}
    if escalates:
        outcomes.add("escalate")
    return _Flow(frozenset(escalates), frozenset(outcomes))


def _together(flows: Sequence[tuple[str, _Flow]]) -> _Flow:
    # the strongest decision wins, so an escalate leaves unless a peer always
    # decides something stronger; a contested modify becomes a reject
    if not flows:
        return _ABSTAINS
    decided = set[Outcome]().union(*(flow.outcomes for _, flow in flows)) - {None}
    if "modify" in decided and len(flows) > 1:
        decided.add("reject")
    if any(flow.outranks for _, flow in flows):
        return _Flow(frozenset(), frozenset(decided & _OUTRANK))
    escalates = set[tuple[str, bool]]().union(
        *(_below(name, flow) for name, flow in flows)
    )
    if all(None in flow.outcomes for _, flow in flows):
        decided.add(None)
    return _Flow(frozenset(escalates), frozenset(decided))


def _below(name: str, flow: _Flow) -> set[tuple[str, bool]]:
    # a group's functions are at the group's own path, named ""
    return {
        ("/".join(part for part in (name, path) if part), named)
        for path, named in flow.escalates
    }
