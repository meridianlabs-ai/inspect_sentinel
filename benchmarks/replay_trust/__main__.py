"""`python -m benchmarks.replay_trust`: run the benchmark, or measure your own monitor on your own logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from inspect_sentinel.scout import Stage

from .benchmark import JUDGE, Settings, run_benchmark, run_on_logs
from .measure import Outcome
from .monitors import MONITORS

EXIT: dict[Outcome, int] = {"trusted": 0, "not trusted": 1, "inconclusive": 2}
"""The exit status for each outcome."""


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
        default=[20, 100, 300],
        help="Haystack sizes, in agent turns.",
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

    logs = commands.add_parser(
        "logs", help="Measure your own monitor on your own logs."
    )
    logs.add_argument("--out", type=Path, required=True)
    logs.add_argument(
        "--monitor",
        required=True,
        help="The monitor's factory, as module:name, called with no arguments.",
    )
    logs.add_argument(
        "--stage",
        required=True,
        choices=["model_input", "model_output", "tool_call", "tool_result"],
    )
    logs.add_argument(
        "--live",
        nargs="+",
        required=True,
        help="One or two eval logs of the same samples with the monitor attached observe-only.",
    )
    logs.add_argument(
        "--model-role",
        action="append",
        default=[],
        help="role=model, as the live runs had them, such as monitor=openai/gpt-4o-mini.",
    )
    logs.add_argument(
        "--needles",
        type=Path,
        default=None,
        help="A JSON list of the tool call ids that break the policy.",
    )
    logs.add_argument(
        "--path",
        default=None,
        help="The monitor's instance path in the live sentinel; its name by default.",
    )
    logs.add_argument(
        "--no-model",
        action="store_true",
        help="The monitor calls no model, so there are no prompts to compare.",
    )

    args = parser.parse_args(argv)
    if args.command == "haystacks":
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
        out = cast(Path, args.out)
    else:
        roles = dict(r.split("=", 1) for r in args.model_role)
        needles = (
            [str(n) for n in cast(list[object], json.loads(args.needles.read_text()))]
            if args.needles
            else []
        )
        results = run_on_logs(
            args.out,
            args.monitor,
            cast(Stage, args.stage),
            args.live,
            roles,
            needles=needles,
            path=args.path,
            calls_a_model=not args.no_model,
        )
        out = cast(Path, args.out)
    print((out / "summary.md").read_text())
    return EXIT[results["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
