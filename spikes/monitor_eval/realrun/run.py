"""Run real agents on task specs and record spend: `run.py --model M --log-dir DIR TASK_ID [...] [--script FILE]`.

Load the API keys into the environment first (for example `set -a; . ~/code/inspect_sentinel/.env; set +a`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ledger  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402
from task import realrun  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tasks", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--script")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--concurrent", action="store_true")
    p.add_argument("--label", default="run", help="Ledger description.")
    args = p.parse_args()
    spent = ledger.check()
    print(f"ledger total before run: ${spent:.2f}", file=sys.stderr)
    tasks = [
        realrun(
            only=t,
            script=args.script,
            max_turns=args.max_turns,
            concurrent=args.concurrent,
        )
        for t in args.tasks
    ]
    logs = inspect_eval(
        tasks,
        model=args.model,
        log_dir=args.log_dir,
        max_tasks=4,
        display="plain",
        fail_on_error=False,
    )
    for log in logs:
        usd = ledger.record(log, f"{args.label}: {log.eval.task}")
        print(f"{log.eval.task}\t{log.status}\t${usd:.4f}\t{log.location}")


if __name__ == "__main__":
    main()
