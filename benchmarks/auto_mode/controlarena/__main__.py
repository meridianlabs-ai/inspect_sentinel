"""`python -m benchmarks.auto_mode.controlarena`: ControlArena trajectories for the auto_mode benchmark, from generation to report, within one budget.

| Command | What it does |
|---|---|
| `generate` | Runs the plan's remaining batches in ControlArena's environment, within the budget less the judging reserve. |
| `import` | Judges whether each attack attempted its side task, and imports every log as trajectories into `OUT/trajectories/`. |
| `logs` | Makes a replay log for each trajectory, once. |
| `score` | Scores every call of every trajectory with each judge, then the ablations (on `--ablation-subset` of them), each only if its projected cost fits the budget. |
| `report` | Writes `OUT/trajectory_report/`. |
| `run` | All of the above in order. It resumes: a batch already generated, logs that match, and a monitor already scored are reused. |

Every dollar spent goes to `DATA/ledger.csv`, and every step checks it against `--budget` before it starts.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inspect_ai.model import ModelUsage

from ..logs import check_manifest, generate_logs
from ..paths import check_out
from ..score import MONITORS as _CASE_MONITORS
from ..trajectories import (
    Trajectory,
    load_trajectories,
    trajectory_digest,
    trajectory_task,
    write_trajectories,
)
from ..trajectory_report import write_trajectory_report
from ..trajectory_score import (
    ABLATIONS,
    JUDGES,
    TrajectorySettings,
    read_calls,
    read_usage,
    runnable,
    score_trajectories,
    subset,
)
from .generate import Plan, dotenv, generate
from .importer import import_logs
from .ledger import MARGIN, entries, record, spent
from .refusals import classify

PRIOR_PER_CALL = {"claude_code": 0.006}
"""Dollars per judged tool call assumed for a monitor before any has been measured; others 0.004."""

DEFAULT_PER_CALL = 0.004

assert set(JUDGES) == set(_CASE_MONITORS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.auto_mode.controlarena", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--data",
            type=Path,
            required=True,
            help="The data directory (logs, ledger), outside the repository.",
        )
        p.add_argument(
            "--out",
            type=Path,
            default=None,
            help="Where trajectories, logs, scans, results and the report go; by default DATA/bench.",
        )
        p.add_argument(
            "--budget",
            type=float,
            required=True,
            help="Dollars for the whole run, generation and judging, that no step may cross.",
        )
        p.add_argument(
            "--env-file", type=Path, default=None, help="A dotenv file of API keys."
        )

    def scoring(p: argparse.ArgumentParser) -> None:
        p.add_argument("--judge-model", default="anthropic/claude-haiku-5-5")
        p.add_argument("--claude-code-prompt", type=Path, default=None)
        p.add_argument("--guardian-prompt", type=Path, default=None)
        p.add_argument("--monitors", nargs="+", default=list(JUDGES + ABLATIONS))
        p.add_argument(
            "--ablation-subset",
            type=int,
            default=None,
            help="Run the ablations on a seeded draw of this many trajectories; all by default.",
        )
        p.add_argument("--max-transcripts", type=int, default=25)

    for name in ("generate", "run"):
        p = commands.add_parser(name)
        common(p)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument(
            "--reserve",
            type=float,
            default=150.0,
            help="Dollars generation leaves for judging.",
        )
        p.add_argument("--python", type=Path, default=None)
        if name == "run":
            scoring(p)
    for name in ("import", "logs", "score", "report"):
        p = commands.add_parser(name)
        common(p)
        if name == "score":
            scoring(p)
        if name == "logs":
            p.add_argument("--force", action="store_true")
        if name == "import":
            p.add_argument("--judge-model", default="anthropic/claude-haiku-5-5")

    args = parser.parse_args(argv)
    data: Path = args.data
    out: Path = args.out or data / "bench"
    unsafe = check_out(out)
    if unsafe is not None:
        print(unsafe, file=sys.stderr)
        return 4
    ledger = data / "ledger.csv"
    command: str = args.command
    keys = dotenv(args.env_file) if args.env_file else {}
    os.environ.update(keys)
    if command in ("generate", "run"):
        ran = generate(
            Plan.read(args.plan),
            data,
            python=args.python or data / "venv" / "bin" / "python",
            budget=args.budget,
            reserve=args.reserve,
            env=keys,
        )
        for r in ran:
            if "error" in r:
                print(r["error"], file=sys.stderr)
        print(f"Generation: {len(ran)} batches run; ${spent(ledger):.2f} spent in all.")
        if command == "generate":
            return 0
    if command in ("import", "run"):
        files = sorted((data / "logs").rglob("*.eval"))
        files = [f for f in files if "smoke" not in f.parts]
        if spent(ledger) + MARGIN + 1.0 > args.budget:
            print("Not judging refusals: the budget is spent.", file=sys.stderr)
            attempts: dict[str, str] = {}
        else:
            found, usage = classify(files, data, args.judge_model)
            record(ledger, entries("judge", "refusals", usage))
            attempts = dict(found)
        trajectories, skipped = import_logs(files, attempts)
        write_trajectories(trajectories, out / "trajectories" / "controlarena.yaml")
        print(
            f"Imported {len(trajectories)} trajectories ({len(skipped)} samples skipped)."
        )
        for note in skipped:
            print(f"  skipped {note}", file=sys.stderr)
        if command == "import":
            return 0
    trajectories = load_trajectories(out / "trajectories")
    log_dir = out / "trajectory_logs"
    if command in ("logs", "run"):
        if command == "logs" or not _logs_match(trajectories, log_dir):
            generate_logs(
                trajectories,
                log_dir,
                force=command == "run" or args.force,
                task=trajectory_task,
                digest=trajectory_digest,
            )
            print(f"Made {len(trajectories)} trajectory logs in {log_dir}.")
        if command == "logs":
            return 0
    if command in ("score", "run"):
        settings = TrajectorySettings(
            out=out,
            judge_model=args.judge_model,
            claude_code_prompt=args.claude_code_prompt,
            guardian_prompt=args.guardian_prompt,
            max_transcripts=args.max_transcripts,
        )
        names, skipped_monitors = runnable(args.monitors, settings)
        for note in skipped_monitors:
            print(note, file=sys.stderr)
        ablated = subset(trajectories, args.ablation_subset)
        for name in names:
            chosen = ablated if name in ABLATIONS else trajectories
            if _scored(out, name, chosen):
                print(f"{name}: already scored.")
                continue
            calls = sum(len(t.calls()) for t in chosen)
            projected = calls * _per_call(out, name)
            if spent(ledger) + projected + MARGIN > args.budget:
                print(
                    f"{name}: not scored; projected ${projected:.2f} for {calls} calls, with ${spent(ledger):.2f} spent of ${args.budget:.2f}.",
                    file=sys.stderr,
                )
                continue
            file = score_trajectories(
                name, trajectories, settings, only=chosen if name in ABLATIONS else None
            )
            used = read_usage(file.with_suffix(".usage.csv"))
            cost = sum(u["cost"] or 0.0 for u in used)
            _record_judging(ledger, name, args.judge_model, used)
            print(f"{name}: scored {calls} calls for ${cost:.2f}.")
    if command in ("report", "run", "score"):
        report = write_trajectory_report(trajectories, out)
        print(report.read_text())
    return 0


def _logs_match(trajectories: Sequence[Trajectory], log_dir: Path) -> bool:
    try:
        check_manifest(trajectories, log_dir, trajectory_digest)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _scored(out: Path, name: str, chosen: Sequence[Trajectory]) -> bool:
    # a monitor whose usage file covers exactly these trajectories as they
    # are now is not scored again
    file = out / "trajectory_results" / f"{name}.usage.csv"
    if not file.exists():
        return False
    rows = read_usage(file)
    return {r["trajectory"]: r["digest"] for r in rows} == {
        t.id: trajectory_digest(t) for t in chosen
    }


def _per_call(out: Path, name: str) -> float:
    # the monitor's measured cost per judged tool call, else the most
    # expensive measured among the others with a quarter more, else the prior
    directory = out / "trajectory_results"
    measured: dict[str, float] = {}
    files: list[Path] = (
        sorted(directory.glob("*.usage.csv")) if directory.exists() else []
    )
    for file in files:
        monitor = file.name.removesuffix(".usage.csv")
        calls = read_calls(directory / f"{monitor}.csv")
        judged = sum(1 for c in calls if c["call"] is not None)
        if judged:
            measured[monitor] = sum(r["cost"] or 0.0 for r in read_usage(file)) / judged
    if name in measured:
        return measured[name] * 1.1
    if measured:
        return max(measured.values()) * 1.25
    return PRIOR_PER_CALL.get(name, DEFAULT_PER_CALL)


def _record_judging(
    ledger: Path, name: str, model: str, used: Sequence[dict[str, Any]]
) -> None:
    usage = ModelUsage(
        input_tokens=sum(u["input_tokens"] for u in used),
        output_tokens=sum(u["output_tokens"] for u in used),
        input_tokens_cache_read=sum(u["cache_read_tokens"] for u in used),
        input_tokens_cache_write=sum(u["cache_write_tokens"] for u in used),
        reasoning_tokens=sum(u["reasoning_tokens"] for u in used),
    )
    # priced by the ledger's own table, like generation, so the budget is
    # checked against one set of prices
    record(ledger, entries("judge", name, {model: usage}))


if __name__ == "__main__":
    sys.exit(main())
