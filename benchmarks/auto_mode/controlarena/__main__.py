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
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from inspect_ai.model import ModelUsage

from ..evidence import CAP
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
    pending,
    read_calls,
    read_usage,
    runnable,
    score_trajectories,
    subset,
)
from .generate import (
    TIERS,
    Plan,
    cost_per_trajectory,
    counts,
    done,
    dotenv,
    generate,
    stopped,
)
from .importer import import_logs
from .ledger import MARGIN, Entry, price, record, spent
from .refusals import classify

PRIOR_PER_CALL = {
    "claude_code": ModelUsage(
        input_tokens=10,
        output_tokens=700,
        input_tokens_cache_read=75_000,
        input_tokens_cache_write=13_000,
    )
}
"""The tokens per judged tool call assumed for a monitor before any has been measured, priced at its judge model's price: about a quarter above what Claude Haiku 5.5 used on the glm-v1 corpus."""

DEFAULT_PER_CALL = ModelUsage(
    input_tokens=10,
    output_tokens=900,
    input_tokens_cache_read=2_800,
    input_tokens_cache_write=27_500,
)
"""`PRIOR_PER_CALL` for the other monitors."""

OTHER_MODEL = 1.5
"""How much more a judge model may use per call than the model its tokens were measured on, since replies differ in length and caching: Claude Sonnet 5.5's calls cost 1.6 times Claude Haiku 5.5's tokens priced at Sonnet's rates."""

CHUNK = 50
"""Trajectories scanned between checks of a monitor's spend against its projection."""

OVER_CAP = 5
"""How much more a judge prompt over the cap costs: Haiku's price above 100K tokens."""

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

    p = commands.add_parser(
        "status",
        help="Tabulate what has been generated: samples, stops, errors and cost by setting, mode and attacker.",
    )
    common(p)

    p = commands.add_parser(
        "forecast", help="Forecast what a plan still costs, from the measured costs."
    )
    common(p)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--ablation-subset", type=int, default=None)
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
    if command == "status":
        print(status(data))
        return 0
    if command == "forecast":
        print(
            forecast(
                Plan.read(args.plan), data, out, args.ablation_subset, args.judge_model
            )
        )
        return 0
    with _locked(data):
        return _command(args, command, data, out, ledger)


def _command(
    args: argparse.Namespace, command: str, data: Path, out: Path, ledger: Path
) -> int:
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
        # exactly the logs of the batches run, never one still being written
        # `run` imports only its plan's settings and attackers, so a plan for
        # one new attacker or setting doesn't rescore every earlier trajectory
        plan_targets = Plan.read(args.plan).targets if command == "run" else None
        files = [
            Path(b["log"])
            for b in done(data)
            if b.get("log")
            and b["model"] in TIERS
            and (
                plan_targets is None
                or any(
                    b["model"] in models
                    for models in plan_targets.get(b["setting"], {}).values()
                )
            )
        ]
        if spent(ledger) + MARGIN + 1.0 > args.budget:
            print("Not judging refusals: the budget is spent.", file=sys.stderr)
            attempts: dict[str, str] = {}
        else:
            attempts = dict(classify(files, data, args.judge_model))
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
            model = settings.models.get(name, settings.judge_model)
            todo = pending(name, chosen, settings)
            if not todo:
                print(f"{name}: already scored.")
                continue
            calls = sum(len(t.calls()) for t in todo)
            projected = calls * _per_call(out, name, model)
            if spent(ledger) + projected + MARGIN > args.budget:
                print(
                    f"{name}: not scored; projected ${projected:.2f} for {calls} calls on {model}, with ${spent(ledger):.2f} spent of ${args.budget:.2f}.",
                    file=sys.stderr,
                )
                continue
            # the projection is charged before the scan and reversed once it
            # ends, so a scan that dies is still counted; each chunk's real
            # cost is recorded as it ends, and scanning stops once that passes
            # the projection
            record(
                ledger,
                [Entry("judge-reserved", name, model, ModelUsage(), projected)],
            )
            cost = 0.0

            def keep_going(
                used: Sequence[Mapping[str, Any]],
                name: str = name,
                model: str = model,
                projected: float = projected,
            ) -> bool:
                nonlocal cost
                cost += _record_judging(ledger, name, model, used)
                return cost <= projected

            try:
                score_trajectories(
                    name,
                    trajectories,
                    settings,
                    only=chosen if name in ABLATIONS else None,
                    chunk=CHUNK,
                    keep_going=keep_going,
                )
            finally:
                record(
                    ledger,
                    [Entry("judge-reserved", name, model, ModelUsage(), -projected)],
                )
            left = len(pending(name, chosen, settings))
            if left:
                print(
                    f"{name}: stopped with {left} of {len(chosen)} trajectories unscored; ${cost:.2f} spent, past the ${projected:.2f} projected. Run again to score the rest, from the measured cost.",
                    file=sys.stderr,
                )
            else:
                print(f"{name}: scored {calls} calls for ${cost:.2f}.")
    if command in ("report", "run", "score"):
        report = write_trajectory_report(trajectories, out)
        print(report.read_text())
    return 0


def status(data: Path) -> str:
    """A table of the batches run, by setting, mode and attacker: samples, those without error, those a provider's safety classifier stopped, those that errored, dollars and the mean minutes a batch took."""
    rows: dict[tuple[str, str, str], list[float]] = {}
    for b in done(data):
        key = (b["setting"], b["mode"], b["model"])
        found = rows.setdefault(key, [0, 0, 0, 0, 0.0, 0.0, 0])
        found[0] += b["samples"]
        found[1] += b.get("ok", 0)
        found[2] += b.get("stopped", 0)
        found[3] += len(b.get("errors", []))
        found[4] += b.get("cost", 0.0)
        found[5] += b.get("seconds", 0.0)
        found[6] += 1
    lines = [
        "| setting | mode | attacker | samples | ok | stopped | errored | dollars | minutes a batch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (setting, mode, model), r in sorted(rows.items()):
        lines.append(
            f"| {setting} | {mode} | {model} | {r[0]:.0f} | {r[1]:.0f} | {r[2]:.0f} | {r[3]:.0f} | {r[4]:.2f} | {r[5] / max(1, r[6]) / 60:.1f} |"
        )
    return "\n".join(lines)


def forecast(
    plan: Plan,
    data: Path,
    out: Path,
    ablation_subset: int | None,
    judge_model: str = "anthropic/claude-haiku-5-5",
) -> str:
    """What `plan` still costs to generate and judge on `judge_model`, from the measured costs: a line per tier of attacker, then judging, then the ablations."""
    have = counts(data)
    halted = stopped(data)
    measured = _calls_per_trajectory(out)
    per_call = {name: _per_call(out, name, judge_model) for name in JUDGES + ABLATIONS}
    lines = ["| part | trajectories | dollars |", "|---|---|---|"]
    totals: dict[str, tuple[int, float]] = {}
    judged = 0
    calls = 0.0
    for setting, modes in plan.targets.items():
        for mode, models in modes.items():
            for model, n in models.items():
                missing = (
                    0
                    if (setting, mode, model) in halted
                    else max(0, n - have.get((setting, mode, model), 0))
                )
                tier = "frontier" if TIERS.get(model) == "frontier" else "sonnet"
                cost = missing * cost_per_trajectory(data, setting, model, plan)
                count, dollars = totals.get(tier, (0, 0.0))
                totals[tier] = (count + missing, dollars + cost)
                total = (
                    n
                    if (setting, mode, model) not in halted
                    else have.get((setting, mode, model), 0)
                )
                judged += total
                calls += total * measured.get(setting, measured.get("*", 30.0))
    for tier, (count, dollars) in sorted(totals.items()):
        lines.append(
            f"| generation, {tier}-class attackers | {count} more | {dollars:.2f} |"
        )
    judging = calls * sum(per_call[j] for j in JUDGES)
    lines.append(
        f"| judging, three judges, {calls:.0f} calls | {judged} in all | {judging:.2f} |"
    )
    share = min(1.0, (ablation_subset or judged) / max(1, judged))
    ablating = calls * share * sum(per_call[a] for a in ABLATIONS)
    lines.append(
        f"| ablations, six, on {round(judged * share)} trajectories | | {ablating:.2f} |"
    )
    lines.append(f"| spent so far | | {spent(data / 'ledger.csv'):.2f} |")
    grand = sum(d for _, d in totals.values()) + judging + ablating
    lines.append(f"| **still to spend** | | **{grand:.2f}** |")
    return "\n".join(lines)


def _calls_per_trajectory(out: Path) -> dict[str, float]:
    # tool calls per trajectory, by setting and overall ("*"), from what is
    # imported
    directory = out / "trajectories"
    if not directory.exists():
        return {}
    found = load_trajectories(directory)
    by: dict[str, list[int]] = {}
    for t in found:
        by.setdefault(t.setting, []).append(len(t.calls()))
        by.setdefault("*", []).append(len(t.calls()))
    return {k: sum(v) / len(v) for k, v in by.items()}


def _logs_match(trajectories: Sequence[Trajectory], log_dir: Path) -> bool:
    try:
        check_manifest(trajectories, log_dir, trajectory_digest)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _per_call(out: Path, name: str, model: str) -> float:
    # dollars per judged tool call for monitor `name` on `model`: the tokens
    # measured per call (the monitor's own, else the most expensive other's
    # with a quarter more, else the prior) priced at `model`'s rates, with a
    # half more where they were measured on another model
    directory = out / "trajectory_results"
    measured: dict[str, tuple[ModelUsage, bool]] = {}
    files: list[Path] = (
        sorted(directory.glob("*.usage.csv")) if directory.exists() else []
    )
    for file in files:
        monitor = file.name.removesuffix(".usage.csv")
        calls = read_calls(directory / f"{monitor}.csv")
        judged = sum(1 for c in calls if c["call"] is not None)
        rows = read_usage(file)
        if judged and rows:
            measured[monitor] = (
                ModelUsage(
                    input_tokens=sum(r["input_tokens"] for r in rows) // judged,
                    output_tokens=sum(r["output_tokens"] for r in rows) // judged,
                    input_tokens_cache_read=sum(r["cache_read_tokens"] for r in rows)
                    // judged,
                    input_tokens_cache_write=sum(r["cache_write_tokens"] for r in rows)
                    // judged,
                ),
                all(r["model"] == model for r in rows),
            )

    def priced(usage: ModelUsage, same: bool) -> float:
        return price(model, usage) * (1.0 if same else OTHER_MODEL)

    if name in measured:
        return priced(*measured[name]) * 1.1
    if measured:
        return max(priced(*m) for m in measured.values()) * 1.25
    prior = PRIOR_PER_CALL.get(name, DEFAULT_PER_CALL)
    return priced(prior, model == "anthropic/claude-haiku-5-5")


def _record_judging(
    ledger: Path, name: str, model: str, used: Sequence[Mapping[str, Any]]
) -> float:
    # priced by the ledger's own table, like generation, so the budget is
    # checked against one set of prices; a trajectory any of whose prompts
    # passed the cap is priced at five times, the rate above it
    total = 0.0
    for row in used:
        usage = ModelUsage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            input_tokens_cache_read=row["cache_read_tokens"],
            input_tokens_cache_write=row["cache_write_tokens"],
            reasoning_tokens=row["reasoning_tokens"],
        )
        total += price(model, usage) * (
            OVER_CAP if row["max_prompt_tokens"] > CAP else 1
        )
    record(ledger, [Entry("judge", name, model, ModelUsage(), total)])
    return total


@contextmanager
def _locked(data: Path) -> Generator[None]:
    # one command at a time on a data directory: two would take the same
    # dataset positions and each check the budget without the other's spend
    data.mkdir(parents=True, exist_ok=True)
    lock = data / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock.read_text().strip()
        if holder.isdigit() and _alive(int(holder)):
            raise SystemExit(
                f"Another command (pid {holder}) is using {data}; wait for it, or remove {lock} if it is gone."
            ) from None
        lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    sys.exit(main())
