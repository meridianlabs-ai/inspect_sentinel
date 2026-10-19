"""Score every step of a set of transcripts, several runs each, with the baseline or the current code.

    .venv/bin/python spikes/monitor_eval/run_set.py DIR_OR_FILES... --variant baseline --runs 3 --out results.jsonl

`--variant baseline` puts the package source at `harness.BASELINE_REF` first on `sys.path` (see `harness.py`); `current` uses this worktree. Each run of a transcript has its own sample store, so its own nonce, and scores the transcript's steps concurrently. The dimension is `score.py`'s, so results compare with that command line. `--judging` swaps the template's judging sentence (`harness.JUDGING`). Every line of `--out` records the variant, the judging, the model, the Scout version, the run and the step.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import harness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--variant", choices=["baseline", "current"], required=True)
    parser.add_argument("--judging", choices=list(harness.JUDGING), default="default")
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--stage", action="append")
    parser.add_argument("--step", action="append", help="transcript:step id")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.variant == "baseline":
        harness.use_baseline()
    import anyio
    from inspect_ai.util import Store
    from transcript import STAGES, DryRunHost, ReplayHost, load_transcript

    import inspect_sentinel
    from inspect_sentinel._host import HostContext, running_step

    loaded = str(Path(inspect_sentinel.__file__).parent)
    print(f"inspect_sentinel from {loaded}", file=sys.stderr)
    if args.variant == "baseline":
        assert harness.BASELINE_REF in loaded, loaded
    import _run
    from score import DIMENSION
    from transcript import _NoRecorder  # pyright: ignore[reportPrivateUsage]

    files = sorted(
        f for p in args.paths for f in (sorted(p.glob("*.json")) if p.is_dir() else [p])
    )
    scout = importlib.metadata.version("inspect-scout")
    template = _run.template_for(args.judging, harness.JUDGED, harness.JUDGING)
    stages = args.stage or list(STAGES)
    wanted = set(args.step or [])

    async def all_runs() -> None:
        limiter = anyio.CapacityLimiter(args.concurrency)
        out = args.out.open("a")
        done = 0

        async def one(t: Any, host: Any, store: Any, s: Any, run: int) -> None:
            nonlocal done
            context = HostContext(
                host=host, recorder=_NoRecorder(), store=store
            ).context
            async with limiter:
                started = time.monotonic()
                error = None
                observation = None
                try:
                    with running_step(_NoRecorder(), store):
                        observation = await context.host.score(
                            s.step,
                            DIMENSION,
                            policy=t.policy,
                            template=template,
                            model=None if args.dry_run else args.model,
                        )
                except Exception as e:  # recorded, so one failure does not stop the set
                    error = f"{type(e).__name__}: {e}"
                out.write(
                    json.dumps(
                        {
                            "transcript": t.id,
                            "step": s.id,
                            "stage": s.stage,
                            "run": run,
                            "variant": args.variant,
                            "judging": args.judging,
                            "label": args.label,
                            "model": args.model,
                            "scout": scout,
                            "suspicion": observation.suspicion if observation else None,
                            "explanation": observation.explanation
                            if observation
                            else None,
                            "error": error,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    )
                    + "\n"
                )
                out.flush()
                done += 1
                if done % 50 == 0:
                    print(f"{done} scored", file=sys.stderr)

        async with anyio.create_task_group() as tg:
            for file in files:
                t = load_transcript(file)
                for run in range(args.runs):
                    host = (
                        DryRunHost(t.task)
                        if args.dry_run
                        else ReplayHost(t.task, args.model)
                    )
                    store = Store()
                    for s in t.steps(stages):
                        if wanted and f"{t.id}:{s.id}" not in wanted:
                            continue
                        tg.start_soon(one, t, host, store, s, run)
        out.close()

    anyio.run(all_runs)


if __name__ == "__main__":
    main()
