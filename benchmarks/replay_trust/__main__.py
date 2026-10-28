"""`python -m benchmarks.replay_trust haystacks`: run the benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from inspect_ai._util.error import PrerequisiteError

from .benchmark import JUDGE, SIZES, Settings, run_benchmark
from .measure import Outcome
from .monitors import MONITORS

EXIT: dict[Outcome, int] = {"trusted": 0, "not trusted": 1, "inconclusive": 2}
"""The exit status for each outcome."""

NOT_RUN = 3
"""The exit status when the benchmark could not run, such as without the monitor model's API key."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.replay_trust",
        description="Show that replay can be trusted: run monitors live twice and replay them twice, and compare their prompts, verdicts and needles found.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    haystacks = commands.add_parser(
        "haystacks", help="Run the benchmark over the scripted haystacks."
    )
    haystacks.add_argument(
        "--out", type=Path, default=Path("benchmarks/replay_trust/results")
    )
    haystacks.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SIZES),
        help="Haystack sizes, in agent turns. 20 is a quick smoke check, usually inconclusive: its haystack has fewer steps than a verdict can be trusted on.",
    )
    haystacks.add_argument(
        "--samples", type=int, default=1, help="Haystacks of each size."
    )
    haystacks.add_argument(
        "--monitors", nargs="+", choices=list(MONITORS), default=list(MONITORS)
    )
    haystacks.add_argument(
        "--monitor-model",
        default=JUDGE,
        help="The judges' model; the deterministic judge by default, so nothing is spent.",
    )
    haystacks.add_argument("--max-connections", type=int, default=None)

    args = parser.parse_args(argv)
    try:
        results = run_benchmark(
            Settings(
                out=args.out,
                sizes=tuple(args.sizes),
                samples=args.samples,
                monitors=tuple(args.monitors),
                monitor_model=args.monitor_model,
                max_connections=args.max_connections,
            )
        )
    except PrerequisiteError as ex:
        # inspect_ai's exception hook prints this through its rich console,
        # which an eval with no display has quieted, so the message would be
        # lost; print it as inspect_ai's own CLI does under --json
        print(ex.message, file=sys.stderr, flush=True)
        return NOT_RUN
    print((cast(Path, args.out) / "summary.md").read_text())
    return EXIT[results["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
