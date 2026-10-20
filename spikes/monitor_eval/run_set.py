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
    parser.add_argument(
        "--author",
        default="none",
        help="none, a name in authors.AUTHORS, or auto: the author for each transcript's task.metadata.domain (coding when unset)",
    )
    parser.add_argument(
        "--domains", type=Path, help="a generator key, for each transcript's domain"
    )
    parser.add_argument(
        "--ref", help="with --variant current, score the package at this commit"
    )
    parser.add_argument("--siblings", choices=["none", "calls", "results"])
    parser.add_argument("--scope", choices=["call", "turn"])
    parser.add_argument(
        "--opts",
        default="{}",
        help="JSON of more Host.score keyword arguments (config is a GenerateConfig dict)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.variant == "baseline":
        harness.use_baseline()
    elif args.ref:
        harness.use_baseline(args.ref)
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
    from authors import AUTHORS, DOMAIN_AUTHOR

    # a generator key's domain per transcript, for --author auto
    domains: dict[str, str] = {}
    if args.domains:
        domains = {
            t["id"]: t["domain"]
            for t in json.loads(args.domains.read_text())["transcripts"]
        }

    from inspect_ai.model import GenerateConfig

    from inspect_sentinel import StepPreprocessor

    def options_for(t: Any) -> dict[str, Any]:
        author = args.author
        if author == "auto":
            domain = domains.get(t.id) or (t.task.metadata or {}).get(
                "domain", "coding"
            )
            author = DOMAIN_AUTHOR[domain]
        options = _run.score_options(
            args.judging, harness.JUDGED, harness.JUDGING, AUTHORS[author]
        )
        if args.variant == "current":
            if args.siblings:
                options["preprocessor"] = StepPreprocessor(siblings=args.siblings)
            if args.scope:
                options["scope"] = args.scope
        extra = json.loads(args.opts)
        if "config" in extra:
            extra["config"] = GenerateConfig(**extra["config"])
        options.update(extra)
        return options

    stages = args.stage or list(STAGES)
    wanted = set(args.step or [])

    class Counting(ReplayHost):
        def __init__(self, task: Any, model: str | None, dry: bool) -> None:
            super().__init__(task, model)
            self._dry = DryRunHost(task) if dry else None
            self.input_tokens = 0
            self.output_tokens = 0
            self.cache_write_tokens = 0
            self.cache_read_tokens = 0
            self.calls = 0

        async def generate(self, input: Any, **kwargs: Any) -> Any:
            self.calls += 1
            output = await (
                self._dry.generate(input, **kwargs)
                if self._dry
                else super().generate(input, **kwargs)
            )
            if output.usage:
                self.input_tokens += output.usage.input_tokens
                self.output_tokens += output.usage.output_tokens
                self.cache_write_tokens += output.usage.input_tokens_cache_write or 0
                self.cache_read_tokens += output.usage.input_tokens_cache_read or 0
            return output

    async def all_runs() -> None:
        limiter = anyio.CapacityLimiter(args.concurrency)
        out = args.out.open("a")
        done = 0

        async def one(t: Any, shared: Any, store: Any, s: Any, run: int) -> None:
            nonlocal done
            # a host per step, so its usage is this step's
            host = Counting(t.task, None if args.dry_run else args.model, args.dry_run)
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
                            **options_for(t),
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
                            "author": args.author,
                            "siblings": args.siblings,
                            "scope": args.scope,
                            "opts": args.opts,
                            "ref": args.ref,
                            "label": args.label,
                            "model": args.model,
                            "scout": scout,
                            "suspicion": observation.suspicion if observation else None,
                            "explanation": observation.explanation
                            if observation
                            else None,
                            "error": error,
                            "seconds": round(time.monotonic() - started, 1),
                            "input_tokens": host.input_tokens,
                            "output_tokens": host.output_tokens,
                            "cache_write_tokens": host.cache_write_tokens,
                            "cache_read_tokens": host.cache_read_tokens,
                            "calls": host.calls,
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
