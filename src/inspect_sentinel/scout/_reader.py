from __future__ import annotations

from typing import Any

# inspect_ai's designated boundary for sentinel, not inspect_ai's public API
from inspect_ai._sentinel._integration import MalformedLogError
from inspect_ai.event import (
    SentinelEvent,
)
from inspect_scout import Result, Scanner, Transcript, scanner

from inspect_sentinel import Decision, Failed, MonitorFailedError, Observation, Report

from ._positions import Positions
from ._results import Cites, step_result
from ._scanner import EVENTS, Stage, within


# the replay scanner's events, so a cite label numbers the same event in both
@scanner(name="read", messages="all", events=EVENTS)
def as_reader(path: str = "", *, stage: Stage | None = None) -> Scanner[Transcript]:
    """Read what a monitor or protocol reported in the runs it was attached to, as a transcript scanner: read mode.

    No inference runs. Each `SentinelEvent` the instance at `path` recorded with status `reported`, and each of its decisions recorded `superseded`, becomes one `Result` in the transcript's resultset, mapped as `as_scanner()` maps a replayed report and citing the step's model event or message the same way, so the two modes' rows join on transcript, `step_id` and `stage`. A superseded decision's metadata says so (`status`). A step the instance abstained at recorded nothing, so it has no result; a step that left no model event or tool result cites nothing. A failure recorded at or below `path`, at any stage, makes the transcript's result an error, as in replay, even where a protocol went on without the failed monitor.

    Args:
        path: The instance's path, as `SentinelEvent.path` records it: empty for the root protocol (the default), else such as `attempt/suspicion`, or a lone monitor's name.
        stage: Only the steps at this stage; all stages when None, each row carrying its stage, since a tool call's two steps share its id.
    """

    async def scan(transcript: Transcript) -> list[Result]:
        positions = Positions(transcript.events, transcript.transcript_id)
        cites = Cites(transcript)
        # each step's children's reports, by conversation as well as stage and
        # id, since two agents' tool calls can share an id
        children: dict[tuple[str, str, str], list[SentinelEvent]] = {}
        for event in transcript.events:
            if isinstance(event, SentinelEvent) and _below(event.path, path):
                key = (event.conversation, event.stage, event.step_id)
                children.setdefault(key, []).append(event)
        # a failure at or below the instance, at any stage, fails the read, as
        # in replay, including one a protocol went on without
        failures = [
            Failed(
                name=event.path.rsplit("/", 1)[-1] or _name(event),
                path=event.path,
                function=event.function or "",
                error=RuntimeError(event.error),
            )
            for event in transcript.events
            if isinstance(event, SentinelEvent)
            and event.status == "error"
            and within(event.path, path)
        ]
        if failures:
            raise MonitorFailedError(failures)
        results: list[Result] = []
        for index, event in enumerate(transcript.events):
            if not isinstance(event, SentinelEvent) or event.path != path:
                continue
            if stage is not None and event.stage != stage:
                continue
            superseded = event.status == "superseded" and event.kind == "decision"
            if event.status != "reported" and not superseded:
                continue
            report = _report(event)
            shown = (
                [
                    _child(c)
                    for c in children.get(
                        (event.conversation, event.stage, event.step_id), []
                    )
                    if c.status == "reported"
                ]
                if isinstance(report, Decision)
                else []
            )
            results.append(
                step_result(
                    step_id=event.step_id,
                    stage=event.stage,
                    label=path or _name(event),
                    report=report,
                    citation=positions.cite(event, index),
                    cites=cites,
                    children=shown,
                    metadata={"status": "superseded"} if superseded else {},
                )
            )
        return results

    return scan


def _below(child: str, parent: str) -> bool:
    if child == parent:
        return False
    return parent == "" or child.startswith(parent + "/")


def _name(event: SentinelEvent) -> str:
    return event.factory.rsplit("/", 1)[-1]


def _report(event: SentinelEvent) -> Report:
    if event.kind == "observation":
        if event.suspicion is None:
            raise MalformedLogError(
                f"The observation recorded at {event.path!r}, {event.stage} step {event.step_id!r}, has no suspicion."
            )
        return Observation(
            suspicion=event.suspicion,
            explanation=event.explanation,
            references=list(event.references),
            metadata=event.metadata,
        )
    if event.action is None:
        raise MalformedLogError(
            f"The decision recorded at {event.path!r}, {event.stage} step {event.step_id!r}, has no action."
        )
    return Decision(
        action=event.action,
        audit=event.audit,
        modified=event.modified,
        explanation=event.explanation,
        references=list(event.references),
        message=event.message,
        metadata=event.metadata,
    )


def _child(event: SentinelEvent) -> dict[str, Any]:
    shown: dict[str, Any] = {"path": event.path, "function": event.function}
    if event.kind == "observation":
        shown["suspicion"] = event.suspicion
    else:
        shown["action"] = event.action
    shown["explanation"] = event.explanation
    return shown
