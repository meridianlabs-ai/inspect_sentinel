"""Run a monitor live twice and replay it twice through Scout, and gather what it reported and asked at each step.

The live runs are evals of the haystacks with the monitor attached observe-only; read mode (`as_reader`) reads what it reported there. The replays are `as_scanner` scans of the first live run's log. Steps are matched across runs by a key that does not depend on the run: at a tool stage, the tool call's id (the step id); at a generate stage, the first tool call of the turn's output, since a generate's step id is minted by the run.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from inspect_ai import eval
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from inspect_scout import scan, scan_results_df, transcripts_from

from inspect_sentinel import Monitor, MonitorGroup, observe_only
from inspect_sentinel.scout import Stage, as_reader, as_scanner

from .haystack import Haystack, haystack_task
from .prompts import (
    Request,
    StepId,
    every_step_marked,
    live_requests,
    replay_requests,
    sample_key,
)

RunName = Literal["live_a", "live_b", "replay_1", "replay_2"]
"""The four runs: two live, and two replays of `live_a`'s log."""

RUNS: tuple[RunName, ...] = ("live_a", "live_b", "replay_1", "replay_2")

Key = tuple[str, str, str]
"""A step across runs: the sample, the stage, and the run-independent step key."""


@dataclass
class StepRecord:
    """One step, across the runs."""

    sample: str
    """The sample's id, with `#epoch` after a first epoch."""

    stage: str
    """The step's stage."""

    key: str
    """The step's run-independent key: at a tool stage its tool call id; at a generate stage the turn's first tool call id."""

    calls: tuple[str, ...]
    """The tool call ids the step covers: its call at a tool stage, the turn's calls at a generate stage."""

    scores: dict[RunName, float | None] = field(default_factory=lambda: {})
    """The monitor's suspicion at the step in each run it ran in; `None` where it abstained. A structured suspicion is its highest dimension."""

    requests: dict[RunName, list[Request]] = field(default_factory=lambda: {})
    """The monitor model's requests at the step, by run."""


@dataclass(frozen=True)
class MonitorRun:
    """A monitor's steps across the four runs, at the stage it is measured at."""

    name: str
    """The monitor's name, as it reports live."""

    stage: Stage
    """The stage measured."""

    steps: list[StepRecord]
    """The steps at `stage`, in `live_a`'s order."""

    logs: tuple[str, ...]
    """The live logs: `live_a`'s, and `live_b`'s where there was a second live run."""

    scans: dict[str, str]
    """Each scan's location, by what it read or replayed."""


def run_live(
    haystacks: Sequence[Haystack],
    monitor: Monitor | MonitorGroup,
    log_dir: Path,
    model_roles: Mapping[str, str],
    *,
    compact_at: int = 3000,
    max_connections: int | None = None,
) -> str:
    """Run the haystacks once with `monitor` attached observe-only, and return the log's location."""
    [log] = eval(
        haystack_task(haystacks, observe_only(monitor), compact_at=compact_at),
        model_roles=dict(model_roles),
        log_dir=str(log_dir),
        display="none",
        max_connections=max_connections,
        fail_on_error=True,
    )
    if log.status != "success":
        raise RuntimeError(f"The live run in {log_dir} did not succeed: {log.error}")
    return log.location


def collect(
    name: str,
    monitor: Callable[[], Monitor | MonitorGroup],
    stage: Stage,
    logs: Sequence[str],
    scans: Path,
    model_roles: Mapping[str, str],
    *,
    path: str | None = None,
) -> MonitorRun:
    """Read `monitor`'s live reports from `logs`, replay it twice over the first, and gather each step at `stage` across the runs.

    Args:
        name: A name for the monitor, for the scans.
        monitor: Makes the monitor to replay, as it ran live.
        stage: The stage to measure.
        logs: One or two live logs of the same samples with the monitor attached, the second for the live-against-live noise floor.
        scans: Where to write the scans.
        model_roles: The model roles for the replays, as the live runs had them.
        path: The monitor's instance path in the live sentinel, for read mode; `name` when None, as for a lone monitor.
    """
    if not 1 <= len(logs) <= 2:
        raise ValueError(f"collect() takes one or two live logs, got {len(logs)}.")
    read = [read_eval_log(log, resolve_attachments=True) for log in logs]
    records: dict[Key, StepRecord] = {}
    order: list[Key] = []
    locations: dict[str, str] = {}

    def record(
        sample: str, step: StepId, keys: Mapping[str, tuple[str, tuple[str, ...]]]
    ) -> StepRecord | None:
        found_stage, step_id = step
        if found_stage != stage:
            return None
        key, calls = keys.get(step_id, (step_id, ()))
        if found_stage in ("tool_call", "tool_result"):
            calls = (step_id,)
        k = (sample, found_stage, key)
        if k not in records:
            records[k] = StepRecord(
                sample=sample, stage=found_stage, key=key, calls=calls
            )
            order.append(k)
        return records[k]

    for run, log, location in zip(("live_a", "live_b"), read, logs, strict=False):
        run = cast(RunName, run)
        samples = {s.uuid: s for s in log.samples or []}
        keys = {s.uuid: generate_keys(s.events) for s in log.samples or []}
        scanned = _scan(
            {run: as_reader(path or name, stage=stage)},
            location,
            scans / f"{name}-{run}",
            model_roles,
        )
        locations[run] = scanned
        for transcript_id, results, _ in _transcripts(scanned, run):
            sample = samples[transcript_id]
            for result in results:
                step = (result["metadata"]["stage"], result["metadata"]["step_id"])
                found = record(sample_key(sample), step, keys[sample.uuid])
                if found is not None:
                    found.scores[run] = _score(result.get("value"))
        for sample in log.samples or []:
            for step, requests in live_requests(sample.events).items():
                found = record(sample_key(sample), step, keys[sample.uuid])
                if found is not None:
                    found.requests[run] = requests

    log_a = read[0]
    samples = {s.uuid: s for s in log_a.samples or []}
    keys = {s.uuid: generate_keys(s.events) for s in log_a.samples or []}
    for run in ("replay_1", "replay_2"):
        run = cast(RunName, run)
        scanned = _scan(
            {run: as_scanner(monitor(), stage=stage, steps=every_step_marked)},
            logs[0],
            scans / f"{name}-{run}",
            model_roles,
        )
        locations[run] = scanned
        for transcript_id, results, events in _transcripts(scanned, run):
            sample = samples[transcript_id]
            for result in results:
                step = (result["metadata"]["stage"], result["metadata"]["step_id"])
                found = record(sample_key(sample), step, keys[sample.uuid])
                if found is not None:
                    found.scores[run] = _score(result.get("value"))
            for step, requests in replay_requests(events).items():
                found = record(sample_key(sample), step, keys[sample.uuid])
                if found is not None:
                    found.requests[run] = requests

    return MonitorRun(
        name=name,
        stage=stage,
        steps=[records[k] for k in order],
        logs=tuple(logs),
        scans=locations,
    )


def generate_keys(events: Sequence[Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Each generate step id in a sample's events, as the run-independent key and the tool calls of the turn it produced.

    A generate's step id is the `uuid` of its first `ModelEvent`. An attempt that failed and was retried has no output, so its turn is the next attempt's in the same span.
    """
    keys: dict[str, tuple[str, tuple[str, ...]]] = {}
    pending: dict[str | None, list[str]] = {}
    for event in events:
        if not isinstance(event, ModelEvent) or event.origin != "agent":
            continue
        calls = tuple(
            call.id
            for call in (
                event.output.message.tool_calls if event.output.choices else None
            )
            or []
        )
        waiting = pending.setdefault(event.span_id, [])
        if event.uuid is not None:
            waiting.append(event.uuid)
        if event.error is None and event.output.choices:
            key = calls[0] if calls else (event.uuid or "")
            for uuid in waiting:
                keys[uuid] = (key, calls)
            waiting.clear()
    return keys


def _scan(
    scanners: Mapping[str, Any], log: str, scans: Path, model_roles: Mapping[str, str]
) -> str:
    status = scan(
        scanners=dict(scanners),
        transcripts=transcripts_from(log),
        scans=str(scans),
        model_roles=dict(model_roles),
        display="none",
        max_processes=1,
        fail_on_error=True,
    )
    if not status.complete:
        raise RuntimeError(f"The scan in {scans} did not complete.")
    return status.location


def _transcripts(
    location: str, scanner: str
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    # each transcript's results, in order, and the events the scanner recorded
    frame: Any = scan_results_df(
        location,
        scanner=scanner,
        rows="transcripts",
        exclude_columns=["input", "input_data"],
    ).scanners[scanner]
    found: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for row in cast(list[dict[str, Any]], frame.to_dict("records")):
        if isinstance(row.get("scan_error"), str):
            raise RuntimeError(
                f"{scanner} failed on transcript {row['transcript_id']}: {row['scan_error']}"
            )
        found.append(
            (
                str(row["transcript_id"]),
                _json_list(row.get("value")),
                _json_list(row.get("scan_events")),
            )
        )
    return found


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    return cast(list[dict[str, Any]], value) if isinstance(value, list) else []


def _score(value: Any) -> float | None:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return None
    if isinstance(value, dict):
        scores = [float(v) for v in cast(dict[str, Any], value).values()]
        return max(scores) if scores else None
    return float(value)
