from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ._host import Recorder
from ._report import Decision, Failed, Report, Reported
from ._step import Step

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
