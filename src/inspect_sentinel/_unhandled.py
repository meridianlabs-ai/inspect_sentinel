from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ._report import Decision, Reported


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
    found: list[Invocation] = []
    token = _decided.set(found)
    try:
        yield found
    finally:
        _decided.reset(token)


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
