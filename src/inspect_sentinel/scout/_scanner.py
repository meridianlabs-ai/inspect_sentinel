from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

from inspect_ai.log import SentinelConfig
from inspect_ai.util import registry_info
from inspect_scout import Result, Scanner, Transcript, scanner

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Decision,
    HumanPolicy,
    HumanStandIn,
    MonitorFailedError,
    ReplayReport,
    ReplayStep,
    Sentinels,
    Split,
    replay_monitor,
)
from inspect_sentinel.config import (
    sentinel_config,
    sentinel_from_config,
    watched_stages,
)

from ._context import Contexts
from ._results import Citations, child_report, step_result, transcript_result
from ._select import (
    StepSelector,
    StepsIn,
    callback_ref,
    selector_from,
    stand_in_from,
)
from ._sources import factory_sources, import_factory_sources
from ._steps import transcript_steps, within_depth

Stage: TypeAlias = Literal["model_input", "model_output", "tool_call", "tool_result"]

STAGES: dict[type[Any], Stage] = {
    BeforeGenerate: "model_input",
    AfterGenerate: "model_output",
    BeforeToolCall: "tool_call",
    AfterToolCall: "tool_result",
}

# Scout loads timelines for a scanner whose function carries this attribute,
# as `llm_scanner` asks for them only when it needs them; a copy of Scout's
# private `SCANNER_TIMELINE_NEEDED_ATTR`, pinned to it by a test
TIMELINE_NEEDED_ATTR = "___scanner_timeline_needed___"

# what the log driver reads: model and tool events for the steps, compactions
# for their segments, approvals and sentinel events for earlier decisions and
# recorded answers, and spans for conversations and each step's cut
_EVENTS: list[Any] = [
    "model",
    "tool",
    "compaction",
    "approval",
    "sentinel",
    "span_begin",
    "span_end",
]


def as_scanner(
    sentinel: Sentinels | SentinelConfig,
    *,
    unit: Literal["step", "transcript"] = "step",
    stage: Stage | None = None,
    split: Split = "compaction",
    max_unit_steps: int | None = None,
    max_units: int = 10,
    steps: StepsIn | StepSelector | str | None = None,
    human: Literal["recorded", "fail_closed"] | HumanStandIn | str = "recorded",
    timeline: str | None = None,
    depth: int | None = None,
    compaction: Literal["all", "last"] | int = "all",
) -> Scanner[Transcript]:
    """Replay a monitor or protocol over each transcript a Scout scan reads, as a transcript scanner.

    Each transcript's steps are rebuilt as the sentinel would have been handed them live, the sentinel runs over them with `replay_monitor()`, and each step at `stage` becomes one `Result` in the transcript's resultset, so `scan_results_df(..., rows="results")` gives one row per step. A result's metadata holds the step's `step_id` and `stage`, how the transcript's agent turns were established (`turns`: `"recorded"` from the log, `"inferred"`, or `"stored"` from a stored timeline) and, when the step reached `human()`, where the answer came from (`human`). Its references cite the step's model event or message first, then what the report cites.

    For a monitor, a result's `value` is its suspicion, a number or a dict by dimension, `None` where it abstained, and the observation's own metadata is under `metadata`. For a protocol, it is the root's decision: `value` is whether the step was flagged (`reject`, `modify` or `terminate`; `escalate` declines to decide, so it is not), `answer` the action, and the metadata holds `audit`, `modified` and the children's reports. A monitor that raises, anywhere in the sentinel, makes the transcript's result an error, even where a protocol went on without it.

    Every stage the sentinel watches runs, in order within a unit, as live, so a monitor whose stages share state sees the same state; `stage` only selects which results are returned. The sentinel is recorded in the scan as the configuration an eval log records, with the modules that define its monitors and protocols (by name, and by file for a script), so a scan can be resumed and run in other processes.

    Args:
        sentinel: A monitor or protocol, or monitors and protocols that resolve to one root, as `Task(sentinel=)` takes them, or their configuration. Several monitors with no protocol are refused: scan them one at a time.
        unit: `"step"` for one result per step, `"transcript"` for one per transcript: a monitor's highest suspicion, per dimension when structured, or a protocol's strongest decision, citing the steps that set it.
        stage: The stage whose steps give results. Required when the sentinel watches more than one stage; defaults to its only one otherwise.
        split: How each transcript's steps are grouped into units, as `replay_monitor()` takes it. Recorded in the scan.
        max_unit_steps: Split units further into runs of at most this many steps, each with a fresh store.
        max_units: How many units of one transcript run at once.
        steps: Which steps to run the sentinel at: a `steps_in()` label table, or a selector `(transcript_id, step) -> bool` registered with `@step_selector`, or the `module:name` it is registered as. Every step is still rebuilt; only the chosen ones run, so a monitor's state does not see the others.
        human: How `human()` is answered where no answer was recorded at the step: `"recorded"`, the fail-closed answer; `"fail_closed"`, the fail-closed answer even where one was recorded; or a stand-in registered with `@human_stand_in`, or the `module:name` it is registered as. A selector or stand-in defined in a notebook, REPL or `python -c` is refused, since a scan's worker processes cannot load it.
        timeline: The stored timeline whose agent spans and branches are the conversations, such as Petri's `"target"`, in place of the log's span tree. Raises for a transcript with no timeline of that name.
        depth: Run only conversations at most this deep in the timeline, counting as `llm_scanner` does: only spans that make model calls count, so 1 is the outermost such span on each branch. All conversations when None.
        compaction: Which compaction segments of each conversation to run: `"all"`, `"last"`, or that many of the latest, at least 1.

    Raises:
        ValueError: If `stage` is missing or not a stage the sentinel watches, the sentinel is several monitors and no protocol, or `depth` or `compaction` is less than 1.
        TypeError: If `steps` or `human` is a callable not registered for a scan.
    """
    if depth is not None and depth < 1:
        raise ValueError(f"depth must be at least 1; got {depth}.")
    if not isinstance(compaction, str) and compaction < 1:
        raise ValueError(
            f"compaction must be 'all', 'last' or at least 1 segment; got {compaction}."
        )
    config = (
        sentinel if isinstance(sentinel, SentinelConfig) else sentinel_config(sentinel)
    )
    chosen = _stage(sentinel_from_config(config), stage)
    selected: object = (
        steps
        if steps is None or isinstance(steps, StepsIn)
        else callback_ref("step_selector", steps, "steps")
    )
    policy: object = (
        human
        if human == "recorded" or human == "fail_closed"
        else callback_ref("human_stand_in", human, "human")
    )
    return sentinel_replay(
        # as JSON text: Scout builds any registry entry it finds in a
        # scanner's parameters before calling its factory, which would rebuild
        # the sentinel before its factories' modules are imported
        sentinel=config.model_dump_json(),
        sources=factory_sources(config),
        unit=unit,
        stage=chosen,
        split=split,
        max_unit_steps=max_unit_steps,
        max_units=max_units,
        steps=selected,
        human=policy,
        timeline=timeline,
        depth=depth,
        compaction=compaction,
    )


@scanner(name="replay", messages="all", events=_EVENTS, condensed=True)
def sentinel_replay(
    *,
    sentinel: str,
    sources: list[dict[str, str | None]],
    unit: Literal["step", "transcript"],
    stage: Stage,
    split: Split,
    max_unit_steps: int | None,
    max_units: int,
    steps: Any,
    human: Any,
    timeline: str | None,
    depth: int | None,
    compaction: Literal["all", "last"] | int,
) -> Scanner[Transcript]:
    rebuilt = _Rebuilt(sentinel, sources, steps, human)

    async def scan(transcript: Transcript) -> Result | list[Result]:
        root, selector, policy, contexts = rebuilt.get()
        eval_info = await contexts.eval_info(transcript)
        built, turns = transcript_steps(transcript, eval_info, timeline)
        segments = _segments(built, compaction)
        shallow = (
            within_depth(transcript, timeline, depth) if depth is not None else None
        )
        transcript_id = transcript.transcript_id

        def select(step: ReplayStep) -> bool:
            if step.segment not in segments:
                return False
            if shallow is not None and step.step.conversation not in shallow:
                return False
            return selector is None or selector(transcript_id, step)

        reports = await replay_monitor(
            root,
            built,
            context=await contexts.eval_context(transcript),
            select=select,
            human=policy,
            split=split,
            max_unit_steps=max_unit_steps,
            max_units=max_units,
        )
        citations = Citations(transcript, timeline)
        results: list[Result] = []
        label = ""
        for report in reports:
            # any failure at or below the scanned instance fails the transcript,
            # including one a protocol went on without
            failed = [f for f in report.failed if within(f.path, report.path)]
            if failed:
                raise MonitorFailedError(failed)
            label = report.path or report.name
            if report.step.stage == stage:
                results.append(_result(report, citations, turns))
        if unit == "transcript":
            return transcript_result(results, label)
        return results

    if timeline is not None or depth is not None:
        # Scout reads timelines only for a scanner that asks for them
        setattr(scan, TIMELINE_NEEDED_ATTR, True)
    return scan


class _Rebuilt:
    # The sentinel, selector and stand-in, rebuilt from the scan's recorded
    # parameters in each process the scan runs in, on its first transcript.
    # Scout sends the scanner to its worker processes by pickling it, and a
    # sentinel's functions cannot be pickled, so only the parameters travel.
    def __init__(
        self,
        sentinel: str,
        sources: list[dict[str, str | None]],
        steps: Any,
        human: Any,
    ) -> None:
        self._sentinel = sentinel
        self._sources = sources
        self._steps = steps
        self._human = human
        self._built: (
            tuple[Sentinels, StepSelector | None, HumanPolicy, Contexts] | None
        ) = None

    def get(self) -> tuple[Sentinels, StepSelector | None, HumanPolicy, Contexts]:
        if self._built is None:
            config = SentinelConfig.model_validate_json(self._sentinel)
            # a process that has not imported the modules registering the
            # sentinel's factories, such as a worker or a resumed scan
            import_factory_sources(config, self._sources)
            self._built = (
                sentinel_from_config(config),
                selector_from(self._steps),
                stand_in_from(self._human),
                Contexts(),
            )
        return self._built

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_built": None}


def within(path: str, scanned: str) -> bool:
    return scanned == "" or path == scanned or path.startswith(scanned + "/")


def _result(report: ReplayReport, citations: Citations, turns: str) -> Result:
    reported = report.report
    metadata: dict[str, Any] = {"turns": turns}
    if report.human is not None:
        metadata["human"] = report.human
    children = (
        [child_report(r) for r in report.reports if r is not reported]
        if reported is not None and isinstance(reported.report, Decision)
        else []
    )
    return step_result(
        step_id=report.step.id,
        stage=report.step.stage,
        label=report.path or report.name,
        report=reported.report if reported is not None else None,
        citation=citations.cite(report.step),
        children=children,
        metadata=metadata,
    )


def _stage(sentinel: Sentinels, stage: Stage | None) -> Stage:
    instances = _instances(sentinel)
    watched: set[Stage] = set()
    for instance in instances:
        watched |= {STAGES[t] for t in watched_stages(instance)}
    if stage is not None:
        if stage not in watched:
            raise ValueError(
                f"The sentinel does not watch the {stage!r} stage; it watches {sorted(watched)}."
            )
        return stage
    if len(watched) != 1:
        raise ValueError(
            f"The sentinel watches {sorted(watched)}; pass stage= to choose which steps give results, since a tool call's two steps share its id."
        )
    return next(iter(watched))


def _instances(sentinel: Sentinels) -> list[Any]:
    if isinstance(sentinel, Mapping):
        instances = list(cast(Mapping[str, Any], sentinel).values())
    elif isinstance(sentinel, Sequence):
        instances = list(cast(Sequence[Any], sentinel))
    else:
        return [sentinel]
    if len(instances) > 1 and all(
        registry_info(i).type == "monitor" for i in instances
    ):
        raise ValueError(
            "as_scanner() scans one monitor at a time, so each step has one result; pass each monitor to its own scanner, or wrap them in a protocol."
        )
    return instances


def _segments(
    steps: list[ReplayStep], compaction: Literal["all", "last"] | int
) -> set[tuple[str, int]]:
    # the compaction segments of each conversation to run
    latest: dict[str, int] = {}
    for step in steps:
        conversation, segment = step.segment
        latest[conversation] = max(latest.get(conversation, 0), segment)
    if compaction == "all":
        keep = max(latest.values(), default=0) + 1
    else:
        keep = 1 if compaction == "last" else compaction
    return {
        (conversation, segment)
        for conversation, last in latest.items()
        for segment in range(max(0, last - keep + 1), last + 1)
    }
