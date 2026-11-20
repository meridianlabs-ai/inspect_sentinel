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

    No inference runs. Each `SentinelEvent` the instance at `path` recorded with status `reported`, and each of its decisions recorded `superseded`, becomes one `Result` in the transcript's resultset, mapped as `as_scanner()` maps a replayed report and citing the step's model event or message the same way, so the two modes' rows join on transcript, `conversation`, `step_id` and `stage`. A row's `conversation` is the one the log recorded, from its span tree. A replay takes its conversations from the span tree too where the log records each model call's origin (`turns` `"recorded"`). Where it doesn't, join on transcript, `step_id` and `stage` instead, which is ambiguous only in a log that records no execution ids, where two agents' tool calls can share an id: a replay with `timeline=` takes its conversations from the stored timeline (`"stored"`), and one of a log without origins infers its turns (`"inferred"`), from the log's events, or from its messages alone where the transcript has no model events or is from a log written by inspect_ai before 0.3.100 (including a sample a retried eval carried over from one when its span tree shows the older layout). A superseded decision's metadata says so (`status`). A step the instance abstained at recorded nothing, so it has no result; a step that left no model event or tool result cites nothing. A failure recorded at or below `path`, at any stage, makes the transcript's result an error, as in replay, even where a protocol went on without the failed monitor.

    A row's `step_id` is the id replay gives the step the check judged, which is not always the `step_id` the event records. A check of a later native run of a call id that an earlier run's steps took (a retry after a tool error, a provider that reuses ids, two agents) is keyed by its `execution_id`, as replay names that run. Which runs replay built is read from the events: each run counts as built unless it ended in a parsing or approval error that no check names. Where replay skipped an earlier run for a reason the events don't show (a call no model call proposed, or one in a region that checks no tool calls), a later run's checks are keyed by their execution and join no replayed row. Where replay skipped a checked earlier run with a warning (a call its model call's proposals don't settle, or one a sentinel modified whose checked call the log doesn't show), that run's checks keep the call id, which replay gives the later run, so they join that run's replayed row. A check an agent bridge recorded (in a `sentinel_bridge` span) is keyed by its `execution_id`, else, for a result checked under an id the bridge handed out more than once, by the id of its outermost sentinel span, which its children's reports share; replay builds no step for a bridged call, so these rows join no replayed row, deliberately, and never take the row of a native call under the same id. A check that names an execution cites that run.

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
                key = (event.conversation, event.stage, positions.step_id(event))
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
                        (event.conversation, event.stage, positions.step_id(event)), []
                    )
                    if c.status == "reported"
                ]
                if isinstance(report, Decision)
                else []
            )
            results.append(
                step_result(
                    step_id=positions.step_id(event),
                    stage=event.stage,
                    conversation=event.conversation,
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
