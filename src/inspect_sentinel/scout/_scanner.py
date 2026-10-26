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
    HumanStandIn,
    MonitorFailedError,
    ReplayStep,
    Sentinels,
)

# sentinel's own replay runner, internal since 2026-10-26: the adapter is its
# only caller
from inspect_sentinel._replay import HumanFallback, ReplayReport, replay_monitor
from inspect_sentinel.config import (
    sentinel_config,
    sentinel_from_config,
    watched_stages,
)

from ._context import Contexts
from ._results import Citations, child_report, step_result, transcript_result
from ._select import (
    ANSWERS,
    StepSelector,
    StepTable,
    Table,
    callback_ref,
    selector_from,
    stand_in_from,
    table_pairs,
)
from ._sources import factory_sources, import_factory_sources
from ._steps import transcript_steps

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
    results: Literal["step", "transcript"] = "step",
    stage: Stage | None = None,
    steps: StepTable | StepSelector | str | None = None,
    human: Literal["approve", "reject", "terminate"] | HumanStandIn | str = "reject",
    timeline: str | None = None,
    depth: int | None = None,
) -> Scanner[Transcript]:
    """Replay a monitor or protocol over each transcript a Scout scan reads, as a transcript scanner.

    Each transcript's steps are rebuilt as the sentinel would have been handed them live, the sentinel runs over them through the same runner as an eval, and each step at a stage it watches becomes one `Result` in the transcript's resultset, so `scan_results_df(..., rows="results")` gives one row per step and stage. A result's metadata holds the step's `step_id`, `stage` and `depth`, how the transcript's agent turns were established (`turns`: `"recorded"` from the log, `"inferred"`, or `"stored"` from a stored timeline) and, when the step reached `human()`, where the answer came from (`human`) and, where it was `human=`'s fixed answer, that answer (`human_answer`). Its references cite the step's model event or message first, then what the report cites.

    For a monitor, a result's `value` is its suspicion, a number or a dict by dimension, `None` where it abstained, and the observation's own metadata is under `metadata`. For a protocol, it is the root's decision: `value` is whether the step was flagged (`reject`, `modify` or `terminate`; `escalate` declines to decide, so it is not), `answer` the action, and the metadata holds `audit`, `modified` and the children's reports. A monitor that raises, anywhere in the sentinel, makes the transcript's result an error, even where a protocol went on without it.

    Each transcript replays in one pass, its steps in order with one store for the whole sample, as live, so a monitor's state carries across compactions and conversations; transcripts run concurrently as for any Scout scanner. The sentinel is recorded in the scan as the configuration an eval log records, with the modules that define its monitors and protocols (by name, and by file for a script), so a scan can be resumed and run in other processes.

    Args:
        sentinel: A monitor or protocol, or monitors and protocols that resolve to one root, as `Task(sentinel=)` takes them, or their configuration. Several monitors with no protocol are refused: scan them one at a time.
        results: `"step"` for one result per step, `"transcript"` for one per transcript: a monitor's highest suspicion, per dimension when structured, or a protocol's strongest decision, citing the steps that set it.
        stage: The stage whose steps give results; every stage the sentinel watches when None, each row carrying its stage. Every watched stage still runs either way. Scout's summary of a scan averages a scanner's values over all its rows, so pass `stage` for a summary of one stage.
        steps: Which steps to run the sentinel at: a label table (`StepTable`), such as a filtered results dataframe, or a selector `(transcript_id, step) -> bool` registered with `@step_selector`, or the `module:name` it is registered as. The others are not run, so a monitor's state does not see them. A transcript a table does not list gives no results and its steps are not rebuilt; narrow the scan's transcripts to the table's too, so they are not read.
        human: How `human()` is answered where no answer was recorded at the step; a recorded answer is always given first. `"approve"`, `"reject"` or `"terminate"` gives that answer, or `terminate` where the asking `human()` does not offer it, as `reject` is not offered after a tool runs or a model replies. Or a stand-in registered with `@human_stand_in`, or the `module:name` it is registered as. A selector or stand-in defined in a notebook, REPL or `python -c` is refused, since a scan's worker processes cannot load it; put it in a `.py` file next to the notebook.
        timeline: The stored timeline whose agent spans and branches are the conversations, such as Petri's `"target"`, in place of the log's span tree. Raises for a transcript with no timeline of that name.
        depth: Run only steps whose conversation is at most this deep in the timeline (`ReplayStep.depth`), counting as `llm_scanner(depth=)` does. All steps when None.

    Raises:
        ValueError: If `stage` is not a stage the sentinel watches, the sentinel is several monitors and no protocol, `steps` is a table without the columns it needs, or `depth` is less than 1.
        TypeError: If `steps` or `human` is a callable not registered for a scan.
    """
    if depth is not None and depth < 1:
        raise ValueError(f"depth must be at least 1; got {depth}.")
    config = (
        sentinel if isinstance(sentinel, SentinelConfig) else sentinel_config(sentinel)
    )
    _stages(sentinel_from_config(config), stage)
    selected: object = (
        None
        if steps is None
        else callback_ref("step_selector", steps, "steps")
        if isinstance(steps, str) or callable(steps)
        else {"table": table_pairs(steps)}
    )
    policy: object = (
        human if human in ANSWERS else callback_ref("human_stand_in", human, "human")
    )
    return sentinel_replay(
        # as JSON text: Scout builds any registry entry it finds in a
        # scanner's parameters before calling its factory, which would rebuild
        # the sentinel before its factories' modules are imported
        sentinel=config.model_dump_json(),
        sources=factory_sources(config),
        results=results,
        stage=stage,
        steps=selected,
        human=policy,
        timeline=timeline,
        depth=depth,
    )


@scanner(name="replay", messages="all", events=_EVENTS)
def sentinel_replay(
    *,
    sentinel: str,
    sources: list[dict[str, str | None]],
    results: Literal["step", "transcript"],
    stage: Stage | None,
    steps: Any,
    human: Any,
    timeline: str | None,
    depth: int | None,
) -> Scanner[Transcript]:
    rebuilt = _Rebuilt(sentinel, sources, steps, human)

    async def scan(transcript: Transcript) -> Result | list[Result]:
        root, selector, policy, contexts = rebuilt.get()
        stages = _stages(root, stage)
        transcript_id = transcript.transcript_id
        if isinstance(selector, Table) and transcript_id not in selector.transcripts:
            return []
        eval_info = await contexts.eval_info(transcript)
        built, turns = transcript_steps(transcript, eval_info, timeline)

        def select(step: ReplayStep) -> bool:
            if depth is not None and (step.depth is None or step.depth > depth):
                return False
            return selector is None or selector(transcript_id, step)

        reports = await replay_monitor(
            root,
            built,
            context=await contexts.eval_context(transcript),
            select=select,
            human=policy,
        )
        citations = Citations(transcript, timeline)
        found: list[Result] = []
        label = ""
        for report in reports:
            # any failure at or below the scanned instance fails the transcript,
            # including one a protocol went on without
            failed = [f for f in report.failed if within(f.path, report.path)]
            if failed:
                raise MonitorFailedError(failed)
            label = report.path or report.name
            if report.step.stage in stages:
                found.append(_result(report, citations, turns))
        if results == "transcript":
            return transcript_result(found, label)
        return found

    if timeline is not None:
        # Scout reads stored timelines only for a scanner that asks for them;
        # without one, each step's depth is counted in a timeline built from
        # the span tree, so Scout is not asked to build another
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
            tuple[Sentinels, StepSelector | None, HumanFallback, Contexts] | None
        ) = None

    def get(
        self,
    ) -> tuple[Sentinels, StepSelector | None, HumanFallback, Contexts]:
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
    metadata: dict[str, Any] = {"depth": report.step.depth, "turns": turns}
    if report.human is not None:
        metadata["human"] = report.human
    if report.human_answer is not None:
        metadata["human_answer"] = report.human_answer
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


def _stages(sentinel: Sentinels, stage: Stage | None) -> set[Stage]:
    # the stages whose steps give results: the one named, else every one the
    # sentinel watches
    watched: set[Stage] = set()
    for instance in _instances(sentinel):
        watched |= {STAGES[t] for t in watched_stages(instance)}
    if stage is None:
        return watched
    if stage not in watched:
        raise ValueError(
            f"The sentinel does not watch the {stage!r} stage; it watches {sorted(watched)}."
        )
    return {stage}


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
