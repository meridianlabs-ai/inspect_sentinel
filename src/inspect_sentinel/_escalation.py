from __future__ import annotations

from collections.abc import Iterable
from logging import getLogger
from typing import Any, NamedTuple, TypeVar

from inspect_ai._util.registry import registry_info, registry_unqualified_name

from ._decorators import ESCALATION_ATTR, step_types
from ._step import AfterToolCall, BeforeToolCall
from ._types import Protocol, Sentinel

logger = getLogger(__name__)

# by the names `SentinelEvent.stage` records
STAGES: dict[str, type[Any]] = {
    "tool_call": BeforeToolCall,
    "tool_result": AfterToolCall,
}
_STAGE_NAMES = {stage: name for name, stage in STAGES.items()}


class Escalation(NamedTuple):
    # (path below the instance, stage) for each place an escalate can come
    # from, the path empty for the instance itself
    escalates: frozenset[tuple[str, type[Any]]]
    # stages at which, handed escalations, the instance always decides
    # something other than escalate
    handles: frozenset[type[Any]]


NEVER = Escalation(frozenset(), frozenset())

S = TypeVar("S")


def declare(sentinel: S, escalation: Escalation) -> S:
    # what a shipped protocol's factory returns; the decorator carries it to the
    # configured instance
    setattr(sentinel, ESCALATION_ATTR, escalation)
    return sentinel


def escalation(sentinel: Sentinel) -> Escalation:
    declared: Escalation | None = getattr(sentinel, ESCALATION_ATTR, None)
    if declared is not None:
        return declared
    if registry_info(sentinel).type == "monitor":
        return NEVER
    # a protocol that is not shipped is assumed to escalate wherever it runs
    return Escalation(
        frozenset(("", stage) for stage in step_types(sentinel)), frozenset()
    )


def handles(stages: Iterable[type[Any]]) -> Escalation:
    return Escalation(frozenset(), frozenset(stages))


def in_order(children: Iterable[tuple[str, Sentinel]]) -> Escalation:
    # a later link that handles a stage decides there for every earlier
    # escalate; the chain starts empty, so it handles nothing for the layer above
    escalates: set[tuple[str, type[Any]]] = set()
    for name, child in children:
        found = escalation(child)
        escalates = {
            (path, stage) for path, stage in escalates if stage not in found.handles
        }
        escalates |= _below(name, found)
    return Escalation(frozenset(escalates), frozenset())


def at_once(children: Iterable[tuple[str, Sentinel]]) -> Escalation:
    # an escalate outranks continue, so a peer's escalate is not handled by
    # another peer, and a handler is one only where no peer can escalate
    escalates: set[tuple[str, type[Any]]] = set()
    handled: set[type[Any]] = set()
    for name, child in children:
        found = escalation(child)
        escalates |= _below(name, found)
        handled |= found.handles
    return Escalation(
        frozenset(escalates),
        frozenset(handled - {stage for _, stage in escalates}),
    )


def _below(name: str, found: Escalation) -> set[tuple[str, type[Any]]]:
    return {
        (f"{name}/{path}" if path else name, stage) for path, stage in found.escalates
    }


_warned: set[str] = set()


def warn_unhandled(root: Protocol) -> None:
    escalates = escalation(root).escalates
    if not escalates:
        return
    root_name = registry_unqualified_name(registry_info(root))
    stages: dict[str, list[str]] = {}
    for path, stage in escalates:
        stages.setdefault(path or root_name, []).append(_STAGE_NAMES[stage])
    sources = ", ".join(
        f"{path} ({', '.join(sorted(names))})" for path, names in sorted(stages.items())
    )
    message = (
        f"This sentinel may escalate with nothing to handle the escalation, which ends the sample as an unhandled escalation: {sources}. "
        "A protocol not shipped with inspect_sentinel is assumed to be able to escalate at every stage it watches. "
        "To state what an escalation means, end the configuration with human() or handle_escalation('terminate' | 'continue') in a sequential(), "
        "as in sequential([..., handle_escalation('terminate')])."
    )
    if message not in _warned:
        _warned.add(message)
        logger.warning(message)
