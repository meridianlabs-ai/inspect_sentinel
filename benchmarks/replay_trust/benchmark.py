"""The benchmark end to end: haystacks, live runs, replays, measurements, spend, and the results written out."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from inspect_sentinel.scout import Stage

from .haystack import OBSERVED, Haystack, haystack, review
from .measure import THRESHOLD, Measurement, Outcome, combined, measure
from .monitors import CALLS_A_MODEL, MONITORS, STAGE
from .report import summary
from .run import RUNS, MonitorRun, collect, run_live
from .spend import PRICES, pricing, spend, used

JUDGE = "replay_trust_judge/judge"
"""The deterministic stand-in for the judges' model."""

SIZES = (20, 100, 300)
"""The haystack sizes run by default, in agent turns."""


@dataclass(frozen=True)
class Settings:
    """What to run."""

    out: Path
    """Where the logs, scans and results go."""

    sizes: tuple[int, ...] = SIZES
    """The haystack sizes, in agent turns."""

    samples: int = 1
    """Haystacks of each size."""

    monitors: tuple[str, ...] = tuple(MONITORS)
    """The monitors to measure, by name."""

    monitor_model: str = JUDGE
    """The judges' model, as the `monitor` role; the deterministic judge by default."""

    compact_at: int = 3000
    """The agent's input size, in tokens, at which its conversation is summarised."""

    max_connections: int | None = None
    """Concurrent connections to the monitor model."""


@dataclass
class MonitorResult:
    """One monitor's measurements, one cell per haystack."""

    name: str
    stage: str
    calls_a_model: bool
    cells: dict[str, Measurement] = field(default_factory=lambda: {})
    """The measurements by haystack id. Each is decided on its own; they are never pooled."""

    run: MonitorRun | None = None

    @property
    def outcome(self) -> Outcome:
        """The cells' outcomes `combined()`: trusted only where every cell is, and inconclusive with none."""
        return combined(m.outcome for m in self.cells.values())


def run_benchmark(settings: Settings) -> dict[str, Any]:
    """Run every monitor live twice and replay it twice over the haystacks, measure, and write `results.json`, `steps.json` and `summary.md` under `settings.out`."""
    settings.out.mkdir(parents=True, exist_ok=True)
    haystacks = [
        haystack(n, i) for n in settings.sizes for i in range(settings.samples)
    ]
    (settings.out / "labels.json").write_text(
        json.dumps({h.id: [asdict(n) for n in h.needles] for h in haystacks}, indent=2)
    )
    roles = {"monitor": settings.monitor_model}
    results: list[MonitorResult] = []
    for name in settings.monitors:
        factory = MONITORS[name]
        logs = [
            run_live(
                haystacks,
                factory(),
                settings.out / "logs" / name / run,
                roles,
                compact_at=settings.compact_at,
                max_connections=settings.max_connections,
            )
            for run in ("a", "b")
        ]
        run = collect(
            name,
            factory,
            cast(Stage, STAGE[name]),
            logs,
            settings.out / "scans",
            roles,
            path=f"{OBSERVED}/{name}",
            review=review,
        )
        results.append(measured(run, haystacks, name in CALLS_A_MODEL))
    return write(settings.out, results, haystacks, _settings_json(settings))


def measured(
    run: MonitorRun, haystacks: Sequence[Haystack], calls_a_model: bool
) -> MonitorResult:
    """`run`'s measurements per haystack, against its needles and the call its person was asked about."""
    result = MonitorResult(
        name=run.name, stage=run.stage, calls_a_model=calls_a_model, run=run
    )
    for h in haystacks:
        result.cells[h.id] = measure(
            run,
            [n.call_id for n in h.needles],
            samples=[h.id],
            calls_a_model=calls_a_model,
            asked=[h.reviewed],
        )
    return result


def write(
    out: Path,
    results: Sequence[MonitorResult],
    haystacks: Sequence[Haystack],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Write `results.json`, `steps.json` and `summary.md` to `out`, and return the results."""
    turns = {h.id: len(h.turns) for h in haystacks}
    spends = [
        asdict(
            spend(
                r.name,
                size,
                [
                    t
                    for hid, m in r.cells.items()
                    if turns[hid] == size
                    for t in m.input_tokens
                ],
            )
        )
        for r in results
        if r.calls_a_model
        for size in sorted(set(turns.values()))
    ]
    document: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "settings": settings,
        "threshold": THRESHOLD,
        "haystacks": {
            h.id: {
                "turns": len(h.turns),
                "needles": [asdict(n) for n in h.needles],
                "retried_turn": h.retried_turn,
                "reviewed": h.reviewed,
            }
            for h in haystacks
        },
        "monitors": {
            r.name: {
                "stage": r.stage,
                "calls_a_model": r.calls_a_model,
                "logs": list(r.run.logs) if r.run else [],
                "scans": dict(r.run.scans) if r.run else {},
                "cells": {
                    cell: _without_tokens(m.as_json()) for cell, m in r.cells.items()
                },
                "usage": used(
                    {
                        run: [
                            q
                            for step in r.run.steps
                            for q in step.requests.get(run, [])
                        ]
                        for run in RUNS
                    },
                    str(settings["monitor_model"]),
                )
                if r.run and r.calls_a_model
                else None,
                "outcome": r.outcome,
            }
            for r in results
        },
        "spend": {
            "prices": [_price(p.model) for p in PRICES],
            "by_monitor_and_size": spends,
            "total": {
                p.model: sum(s["dollars"][p.model] for s in spends) for p in PRICES
            },
            "calls": sum(s["calls"] for s in spends),
        },
    }
    # nothing measured is inconclusive
    document["outcome"] = combined(m["outcome"] for m in document["monitors"].values())
    (out / "results.json").write_text(json.dumps(document, indent=2, default=str))
    (out / "steps.json").write_text(
        json.dumps(
            {
                r.name: [
                    {
                        "sample": s.sample,
                        "stage": s.stage,
                        "key": s.key,
                        "calls": list(s.calls),
                        "scores": s.scores,
                        "requests": {
                            run: len(found) for run, found in s.requests.items()
                        },
                    }
                    for s in r.run.steps
                ]
                for r in results
                if r.run is not None
            },
            indent=1,
        )
    )
    (out / "summary.md").write_text(summary(document))
    return document


def _price(model: str) -> dict[str, Any]:
    found = pricing(model)
    assert found is not None
    return {
        "model": model,
        **found.cost.model_dump(),
        "estimated": found.estimated,
        "source": found.source,
    }


def _without_tokens(measurement: dict[str, Any]) -> dict[str, Any]:
    tokens = measurement.pop("input_tokens")
    measurement["mean_input_tokens"] = sum(tokens) / len(tokens) if tokens else None
    return measurement


def _settings_json(settings: Settings) -> dict[str, Any]:
    found = asdict(settings)
    found["out"] = str(settings.out)
    return found


def _environment() -> dict[str, Any]:
    def head() -> str | None:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "inspect_ai": version("inspect_ai"),
        "inspect_scout": version("inspect_scout"),
        "inspect_sentinel": version("inspect_sentinel"),
        "inspect_sentinel_commit": head(),
    }
