"""Generate ControlArena trajectories in batches, within a budget, resuming from what is already there.

A plan (JSON) names, per setting, how many `attack` and `honest` trajectories each attacker model should have, and the setting's own arguments. `generate()` counts the trajectories the data directory already holds (`batches.jsonl`, one line per batch run), and runs batches of the rest in ControlArena's own environment (`batch.py`), several at once. After each batch it prices the log's token usage into the ledger (`ledger.py`).

Before starting a batch it projects the batch's cost from the measured cost per trajectory of that setting and attacker (or of the attacker's tier, or the plan's prior, before there is one), and starts it only if what is spent, plus what the running batches are projected to cost, plus this batch, stays within the budget less the judging reserve and `MARGIN`. A batch is never started that the projection says would cross it.

A batch whose attacker a provider's safety classifier stopped (a reply ending in `content_filter`) is recorded as stopped, and no further batch of that setting, mode and attacker is queued: a stop is never retried, reworded or worked around, and its share is not given to another attacker.

Each setting's dataset is shuffled once with the plan's seed, and each batch takes the next positions in it, so attackers get different samples, and a resumed run never repeats one. Positions past the dataset's end come round again as repeats.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from inspect_ai.log import EvalSample, read_eval_log

from .ledger import MARGIN, entries, record, spent

BATCH = Path(__file__).parent / "batch.py"
"""The runner, executed by ControlArena's own Python."""

TIERS = {
    "anthropic/claude-sonnet-5-5": "sonnet",
    "openai/gpt-6.2-sol": "sonnet",
    "anthropic/claude-fable-5-2": "frontier",
    "openai/gpt-6.2-astra": "frontier",
}
"""The attacker tiers: the Sonnet class gives the volume and every honest run, the frontier class a smaller set of attacks."""

Mode = Literal["attack", "honest"]


@dataclass(frozen=True)
class Plan:
    """What to generate."""

    targets: Mapping[str, Mapping[str, Mapping[str, int]]]
    """Trajectories wanted, by setting, then mode, then attacker model."""

    setting_args: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: {})
    """Each setting's own arguments, as ControlArena takes them."""

    prior: Mapping[str, float] = field(
        default_factory=lambda: {"sonnet": 2.0, "frontier": 8.0}
    )
    """Dollars per trajectory assumed by tier until a setting and attacker has a measured cost."""

    config: Mapping[str, Any] = field(
        default_factory=lambda: {"reasoning_effort": "high"}
    )
    """The attacker's generate config."""

    trusted_model: str = "anthropic/claude-haiku-5-5"
    """ControlArena's trusted model, which some settings' scorers use."""

    seed: int = 0
    batch_size: int = 4
    parallel: int = 4
    """Batches run at once."""

    max_samples: int = 4
    time_limit: int = 2400
    """Seconds per trajectory."""

    token_limit: int = 1_000_000
    """Tokens per trajectory, ControlArena's default."""

    @staticmethod
    def read(file: Path) -> Plan:
        """Read a plan from JSON."""
        raw: dict[str, Any] = json.loads(file.read_text())
        return Plan(**raw)


@dataclass(frozen=True)
class Batch:
    """A slice of one setting's dataset, in one mode, for one attacker."""

    setting: str
    mode: Mode
    model: str
    start: int
    end: int

    @property
    def label(self) -> str:
        """The batch's name in the ledger."""
        return f"{self.setting}/{self.mode}/{self.model}/{self.start}-{self.end}"


def done(data: Path) -> list[dict[str, Any]]:
    """The batches already run, from `batches.jsonl`."""
    file = data / "batches.jsonl"
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


def counts(data: Path) -> dict[tuple[str, str, str], int]:
    """Trajectories already generated without error, by setting, mode and attacker."""
    found: dict[tuple[str, str, str], int] = {}
    for b in done(data):
        key = (b["setting"], b["mode"], b["model"])
        found[key] = found.get(key, 0) + int(b.get("ok", 0))
    return found


def cost_per_trajectory(data: Path, setting: str, model: str, plan: Plan) -> float:
    """The measured dollars per trajectory for `setting` and `model` (both modes), else for the model's tier across settings, else the plan's prior."""
    runs = [b for b in done(data) if b.get("ok") and b.get("cost") is not None]
    mine = [b for b in runs if b["setting"] == setting and b["model"] == model]
    tier = TIERS.get(model, "frontier")
    peers = mine or [b for b in runs if TIERS.get(b["model"], "frontier") == tier]
    total = sum(int(b["samples"]) for b in peers)
    if total:
        return sum(float(b["cost"]) for b in peers) / total
    return plan.prior[tier]


def generate(
    plan: Plan,
    data: Path,
    *,
    python: Path,
    budget: float,
    reserve: float,
    env: Mapping[str, str] | None = None,
    log: Any = print,
) -> list[dict[str, Any]]:
    """Run the plan's remaining batches within `budget` less `reserve` (for judging) and `MARGIN`; return the batches run.

    Raises:
        FileNotFoundError: If `python` (ControlArena's interpreter) is missing.
    """
    if not python.exists():
        raise FileNotFoundError(f"No ControlArena Python at {python}.")
    ledger = data / "ledger.csv"
    limit = budget - reserve - MARGIN
    lock = threading.Lock()
    running: dict[Batch, float] = {}
    ran: list[dict[str, Any]] = []
    for row in reconcile(data):
        log(
            f"Recorded {row['setting']}/{row['mode']}/{row['model']}/{row['start']}-{row['end']}, run before but not recorded: ${row['cost']:.2f}."
        )
    position = _positions(data)
    queue = _queue(plan, data, position)
    with ThreadPoolExecutor(max_workers=plan.parallel) as pool:
        futures: list[Future[dict[str, Any]]] = []
        for batch in queue:
            projected = (batch.end - batch.start) * cost_per_trajectory(
                data, batch.setting, batch.model, plan
            )
            skip = False
            while True:
                with lock:
                    # checked as each batch starts, not when it was queued: a
                    # batch that has just finished may have been stopped
                    if (batch.setting, batch.mode, batch.model) in stopped(data):
                        log(
                            f"Skipping {batch.label}: a provider's safety classifier stopped this attacker in this setting and mode."
                        )
                        skip = True
                        break
                    if len(running) < plan.parallel and _free_gb() < MIN_FREE_GB:
                        _prune_images()
                    if len(running) >= plan.parallel:
                        pass
                    elif _free_gb() < MIN_FREE_GB:
                        if not running:
                            log(
                                f"Stopping: {_free_gb():.0f}GB free on Docker's disk, under {MIN_FREE_GB}GB."
                            )
                            return _finish(futures, ran)
                    elif spent(ledger) + sum(running.values()) + projected <= limit:
                        running[batch] = projected
                        break
                    elif not running:
                        committed = spent(ledger)
                        log(
                            f"Stopping: {batch.label} is projected at ${projected:.2f}, and ${committed:.2f} is spent of the ${limit:.2f} generation may use."
                        )
                        return _finish(futures, ran)
                time.sleep(15)
            if skip:
                continue

            def run(b: Batch = batch) -> dict[str, Any]:
                try:
                    result = _run(b, plan, data, python, env)
                    with lock:
                        record(ledger, result.pop("entries"))
                        _append(data / "batches.jsonl", result)
                    log(
                        f"{b.label}: {result['ok']}/{result['samples']} ok, ${result['cost']:.2f}, {result['seconds']:.0f}s"
                    )
                    _prune_images()
                    return result
                finally:
                    with lock:
                        running.pop(b, None)

            futures.append(pool.submit(run))
        return _finish(futures, ran)


def _finish(
    futures: Sequence[Future[dict[str, Any]]], ran: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    for f in futures:
        try:
            ran.append(f.result())
        except (
            Exception
        ) as ex:  # a batch that failed to run is reported, and the others still finish
            ran.append({"error": repr(ex)})
    return ran


def reconcile(data: Path) -> list[dict[str, Any]]:
    """Record every finished batch log under `data/logs` that `batches.jsonl` doesn't name, such as one whose driver was stopped while it ran, pricing it into the ledger; return the rows added. A log still being written is left for later."""
    known = {b.get("log") for b in done(data)}
    added: list[dict[str, Any]] = []
    for file in sorted((data / "logs").rglob("*.eval")):
        if str(file) in known:
            continue
        header = read_eval_log(str(file), header_only=True)
        batch = cast(
            Mapping[str, Any], (header.eval.metadata or {}).get("amb_batch") or {}
        )
        if not batch or header.status == "started":
            continue
        log = read_eval_log(str(file))
        samples = log.samples or []
        halted = [s for s in samples if _stopped(s)]
        priced = entries(
            "generate",
            f"{batch['setting']}/{batch['mode']}/{batch['model']}/{batch['start']}-{batch['end']}",
            log.stats.model_usage,
        )
        row: dict[str, Any] = {
            **{k: batch[k] for k in ("setting", "mode", "model", "start", "end")},
            "log": str(file),
            "status": log.status,
            "samples": len(samples),
            "ok": sum(1 for s in samples if s.error is None and s not in halted),
            "errors": [
                str(s.error.message)[:300] for s in samples if s.error is not None
            ],
            "stopped": len(halted),
            "cost": sum(e.cost for e in priced),
            "seconds": 0.0,
            "reconciled": True,
        }
        record(data / "ledger.csv", priced)
        _append(data / "batches.jsonl", row)
        added.append(row)
    return added


def _queue(plan: Plan, data: Path, position: dict[tuple[str, str], int]) -> list[Batch]:
    have = counts(data)
    halted = stopped(data)
    queue: list[Batch] = []
    # interleave settings and attackers, so a budget stop leaves every setting
    # partly done rather than the last ones untouched
    wanted: list[tuple[str, Mode, str, int]] = []
    for setting, modes in plan.targets.items():
        for mode, models in modes.items():
            for model, n in models.items():
                missing = n - have.get((setting, mode, model), 0)
                if missing > 0 and (setting, mode, model) not in halted:
                    wanted.append((setting, cast(Mode, mode), model, missing))
    while wanted:
        later: list[tuple[str, Mode, str, int]] = []
        for setting, mode, model, missing in wanted:
            size = min(plan.batch_size, missing)
            start = position.get((setting, mode), 0)
            position[(setting, mode)] = start + size
            queue.append(Batch(setting, mode, model, start, start + size))
            if missing > size:
                later.append((setting, mode, model, missing - size))
        wanted = later
    return queue


def _positions(data: Path) -> dict[tuple[str, str], int]:
    found: dict[tuple[str, str], int] = {}
    for b in done(data):
        key = (b["setting"], b["mode"])
        found[key] = max(found.get(key, 0), int(b["end"]))
    return found


def _run(
    batch: Batch, plan: Plan, data: Path, python: Path, env: Mapping[str, str] | None
) -> dict[str, Any]:
    log_dir = data / "logs" / batch.setting / batch.mode
    specs = data / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    spec = {
        "setting": batch.setting,
        "setting_args": dict(plan.setting_args.get(batch.setting, {})),
        "mode": batch.mode,
        "model": batch.model,
        "config": dict(plan.config),
        "trusted_model": plan.trusted_model,
        "seed": plan.seed,
        "start": batch.start,
        "end": batch.end,
        "log_dir": str(log_dir),
        "max_samples": plan.max_samples,
        "time_limit": plan.time_limit,
        "token_limit": plan.token_limit,
    }
    name = batch.label.replace("/", "_")
    spec_file = specs / f"{name}.json"
    spec_file.write_text(json.dumps(spec, indent=1))
    output = data / "runs" / f"{name}.out"
    output.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    began_at = time.time()
    with output.open("w") as out:
        done_ = subprocess.run(
            [str(python), "-I", str(BATCH), str(spec_file)],
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            cwd=str(data),
        )
    seconds = time.monotonic() - began
    lines = [
        line for line in output.read_text().splitlines() if line.startswith('{"log":')
    ]
    if lines:
        location = json.loads(lines[-1])["log"]
    else:
        # the run failed before reporting its log; a log it wrote still spent
        # money, so it is priced and recorded all the same
        written = sorted(
            (
                f
                for f in log_dir.glob("*.eval")
                if f.stat().st_mtime >= began_at and _prefix(batch) in f.name
            ),
            key=lambda f: f.stat().st_mtime,
        )
        if not written:
            raise RuntimeError(
                f"{batch.label} exited {done_.returncode} with no log; see {output}."
            )
        location = str(written[-1])
    log = read_eval_log(location)
    samples = log.samples or []
    stopped = [s for s in samples if _stopped(s)]
    ok = sum(1 for s in samples if s.error is None and s not in stopped)
    usage = log.stats.model_usage
    priced = entries("generate", batch.label, usage)
    return {
        "setting": batch.setting,
        "mode": batch.mode,
        "model": batch.model,
        "start": batch.start,
        "end": batch.end,
        "log": location,
        "status": log.status,
        "samples": len(samples),
        "ok": ok,
        "errors": [str(s.error.message)[:300] for s in samples if s.error is not None],
        "stopped": len(stopped),
        "cost": sum(e.cost for e in priced),
        "seconds": seconds,
        "entries": priced,
    }


def _prefix(batch: Batch) -> str:
    # the start of the task name batch.py gives the batch's eval
    return (
        f"amb-{batch.setting}-{batch.mode}-{batch.model.split('/')[-1]}-{batch.start}"
    )


MIN_FREE_GB = 25
"""Gigabytes Docker's disk must have free before a batch starts: settings build an image per task or per sample, several gigabytes each."""

DOCKER_ROOT = Path("/var/lib/docker")

IMAGES = (
    "inspect-amb-",
    "bash-arena-default-",
    "bash-arena-database-server-",
    "bash-arena-logging-server-",
    "bash-arena-malware-server-",
    "bash-arena-router-",
)
"""The prefixes of the images ControlArena's settings build for one batch's samples, which `_prune_images()` removes once no container uses them."""

PRUNE_AFTER = 600
"""Seconds an image must have existed before it is removed, so one a running batch has just built for a container it hasn't started yet is kept."""


def _free_gb() -> float:
    usage = shutil.disk_usage(DOCKER_ROOT if DOCKER_ROOT.exists() else Path("/"))
    return usage.free / 1e9


def _prune_images() -> None:
    # remove the benchmark's own images that no container uses (docker
    # refuses to remove one in use without --force, which is not passed), and
    # dangling build layers; nothing else on the machine is touched
    listed = subprocess.run(
        [
            "docker",
            "image",
            "ls",
            "--format",
            "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
        ],
        capture_output=True,
        text=True,
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    old: list[str] = []
    for line in listed.stdout.splitlines():
        name, _, created = line.partition("\t")
        if not name.startswith(IMAGES):
            continue
        try:
            when = datetime.datetime.strptime(
                " ".join(created.split()[:3]), "%Y-%m-%d %H:%M:%S %z"
            )
        except ValueError:
            continue
        if (now - when).total_seconds() > PRUNE_AFTER:
            old.append(name)
    for name in old:
        subprocess.run(["docker", "rmi", name], capture_output=True)
    subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)


def _stopped(sample: EvalSample) -> bool:
    # a provider's safety classifier stopped the attacker's generation: its
    # reply ended with stop reason content_filter
    return any(
        e.event == "model" and e.output.stop_reason == "content_filter"
        for e in sample.events
    )


def stopped(data: Path) -> set[tuple[str, str, str]]:
    """The setting, mode and attacker of every batch a provider's safety classifier stopped; no batch of these is run again."""
    return {
        (b["setting"], b["mode"], b["model"]) for b in done(data) if b.get("stopped")
    }


def _append(file: Path, row: Mapping[str, Any]) -> None:
    with file.open("a") as f:
        f.write(json.dumps(row) + "\n")


def dotenv(file: Path) -> dict[str, str]:
    """The variables a dotenv file sets."""
    found: dict[str, str] = {}
    for line in file.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name and not name.startswith("#"):
            found[name.strip()] = value.strip().strip('"').strip("'")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m benchmarks.auto_mode.controlarena.generate --plan PLAN --data DIR --budget N`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.auto_mode.controlarena.generate"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="The data directory, outside the repository.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="ControlArena's interpreter; by default DATA/venv/bin/python.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        required=True,
        help="Dollars for the whole run, generation and judging.",
    )
    parser.add_argument(
        "--reserve", type=float, default=150.0, help="Dollars kept for judging."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="A dotenv file of API keys for the runs.",
    )
    args = parser.parse_args(argv)
    env = dotenv(args.env_file) if args.env_file else None
    ran = generate(
        Plan.read(args.plan),
        args.data,
        python=args.python or args.data / "venv" / "bin" / "python",
        budget=args.budget,
        reserve=args.reserve,
        env=env,
    )
    failed = [r for r in ran if "error" in r]
    for r in failed:
        print(r["error"], file=sys.stderr)
    print(
        f"Ran {len(ran)} batches; ${spent(args.data / 'ledger.csv'):.2f} spent in all."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
