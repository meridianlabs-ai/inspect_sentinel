"""The benchmark end to end: haystacks, live runs, replays, measurements, spend, and the results written out."""

from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from inspect_sentinel import Monitor, MonitorGroup
from inspect_sentinel.scout import Stage

from .haystack import OBSERVED, Haystack, haystack, review
from .measure import THRESHOLD, Measurement, measure
from .monitors import CALLS_A_MODEL, MONITORS, STAGE
from .report import summary
from .run import RUNS, MonitorRun, collect, run_live
from .spend import PRICES, pricing, spend, used

JUDGE = "replay_trust_judge/judge"
"""The deterministic stand-in for the judges' model."""


@dataclass(frozen=True)
class Settings:
    """What to run."""

    out: Path
    """Where the logs, scans and results go."""

    sizes: tuple[int, ...] = (20, 100, 300)
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
    """One monitor's measurements: per haystack, and over all of them."""

    name: str
    stage: str
    calls_a_model: bool
    haystacks: dict[str, Measurement] = field(default_factory=lambda: {})
    pooled: Measurement | None = None
    run: MonitorRun | None = None


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


def run_on_logs(
    out: Path,
    monitor: str,
    stage: Stage,
    logs: Sequence[str],
    model_roles: Mapping[str, str],
    *,
    needles: Sequence[str] = (),
    path: str | None = None,
    calls_a_model: bool = True,
) -> dict[str, Any]:
    """Measure your own monitor on your own logs: replay it twice over the first log and compare with what it reported live.

    Args:
        out: Where the scans and results go.
        monitor: The monitor's factory, as `module:name`, called with no arguments.
        stage: The stage to measure.
        logs: One or two eval logs of the same samples with the monitor attached observe-only. With two, the live-against-live noise floor is measured; with one, the replay-against-replay floor stands in.
        model_roles: The model roles the live runs had, such as `{"monitor": "openai/gpt-4o-mini"}`.
        needles: The tool call ids labelled as breaking the policy, for measurement 3.
        path: The monitor's instance path in the live sentinel; its name when None.
        calls_a_model: Whether to compare the monitor's model requests.
    """
    factory = load_factory(monitor)
    name = monitor.rsplit(":", 1)[-1]
    out.mkdir(parents=True, exist_ok=True)
    run = collect(name, factory, stage, logs, out / "scans", model_roles, path=path)
    result = MonitorResult(name=name, stage=stage, calls_a_model=calls_a_model, run=run)
    result.pooled = measure(run, needles, calls_a_model=calls_a_model)
    settings = {
        "monitor": monitor,
        "stage": stage,
        "logs": list(logs),
        "model_roles": dict(model_roles),
        "needles": list(needles),
        "threshold": THRESHOLD,
    }
    return write(out, [result], [], settings)


def measured(
    run: MonitorRun, haystacks: Sequence[Haystack], calls_a_model: bool
) -> MonitorResult:
    """`run`'s measurements per haystack and pooled, against the haystacks' needles and the calls their person was asked about."""
    result = MonitorResult(
        name=run.name, stage=run.stage, calls_a_model=calls_a_model, run=run
    )
    for h in haystacks:
        result.haystacks[h.id] = measure(
            run,
            [n.call_id for n in h.needles],
            samples=[h.id],
            calls_a_model=calls_a_model,
            asked=[h.reviewed],
        )
    result.pooled = measure(
        run,
        [n.call_id for h in haystacks for n in h.needles],
        calls_a_model=calls_a_model,
        asked=[h.reviewed for h in haystacks],
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
                    for hid, m in r.haystacks.items()
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
                "haystacks": {
                    hid: _without_tokens(m.as_json()) for hid, m in r.haystacks.items()
                },
                "pooled": _without_tokens(r.pooled.as_json()) if r.pooled else None,
                "usage": used(
                    {
                        run: [
                            q
                            for step in r.run.steps
                            for q in step.requests.get(run, [])
                        ]
                        for run in RUNS
                    },
                    str(
                        settings.get("monitor_model")
                        or settings.get("model_roles", {}).get("monitor", "")
                    ),
                )
                if r.run and r.calls_a_model
                else None,
                "accepted": r.pooled.accepted if r.pooled else None,
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
    # nothing measured is not accepted
    accepted = [
        m["accepted"]
        for m in document["monitors"].values()
        if m["accepted"] is not None
    ]
    document["accepted"] = bool(accepted) and all(accepted)
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


def load_factory(reference: str) -> Callable[[], Monitor | MonitorGroup]:
    """The monitor factory `module:name` names."""
    module, _, name = reference.partition(":")
    if not name:
        raise ValueError(f"Name a monitor factory as module:name, got {reference!r}.")
    return cast(
        Callable[[], Monitor | MonitorGroup],
        getattr(importlib.import_module(module), name),
    )


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
