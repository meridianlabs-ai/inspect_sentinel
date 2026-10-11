from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from inspect_ai.core import Reference

from ._report import Decision, Reported


@dataclass(eq=False)
class Invocation:
    # one call of one protocol function during a step, under the call that ran
    # it; the functions of a group, and repeat calls at one path, are told apart
    parent: Invocation | None
    decided: Reported[Decision] | None = None
    # a decision the function marked as passing up its children's, rewritten
    forwarded: Decision | None = None


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


def forwarding(decision: Decision) -> Decision:
    # for a protocol that passes up a child's decision rewritten, as concurrent
    # adds its vote summary, so the rewrite is not taken for a reason of its own
    node = _current.get()
    if node is not None:
        node.forwarded = decision
    return decision


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
    # too; of those, each that did not pass up one from below it is where an
    # escalation came from
    reached = [node for node in found if _escalated(node) and _reaches(node)]
    return [
        node.decided
        for node in reached
        if node.decided is not None
        and not _forwards(node, node.decided.report, reached)
    ]


def _forwards(
    node: Invocation, report: Decision, reached: Sequence[Invocation]
) -> bool:
    # a protocol that consulted an escalating child and escalated with a reason
    # of its own is a source too; one that returned the child's escalation, or
    # an equal one, is not
    if node.forwarded is report:
        return True
    return any(
        other.decided is not None
        and other.decided.report == report
        and node in _ancestors(other)
        for other in reached
    )


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


def unhandled(escalations: Sequence[Reported[Decision]]) -> Decision:
    # what run_sentinel returns for an escalate at the root: a terminate, since
    # nothing was configured to answer it
    named = "; ".join(
        f"{e.path or e.name}: {e.report.explanation}"
        if e.report.explanation
        else e.path or e.name
        for e in escalations
    )
    # a parent that escalates on its own may carry its child's references up
    references: list[Reference] = []
    for e in escalations:
        references.extend(r for r in e.report.references if r not in references)
    return Decision.terminate(
        f"unhandled escalation from {named}. To say what an escalation means, end the configuration with human() or handle_escalation('terminate' | 'continue') in a sequential().",
        references=references,
    )
