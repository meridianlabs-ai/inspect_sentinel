"""`python -m benchmarks.auto_mode.sandbox_escape`: SandboxEscapeBench runs as trajectories for the auto_mode benchmark.

| Command | What it does |
|---|---|
| `fetch` | Fetches the pinned logs into `--cache`. |
| `import` | Writes the trajectories of the fetched logs into `OUT/trajectories/sandbox_escape.yaml`, with the task as the agent was given it or a neutral one (`--framing`), on a seeded draw of `--limit` of them. |

Then make the replay logs, score and report with the ControlArena commands, which read `OUT/trajectories/`: `python -m benchmarks.auto_mode.controlarena logs|score|report --out OUT --data DATA`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from ..paths import require_safe
from ..trajectories import write_trajectories
from ..trajectory_score import subset
from . import fetch, import_logs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.auto_mode.sandbox_escape",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch", help="Fetch the pinned logs.")
    p.add_argument("--cache", type=Path, required=True)
    p = commands.add_parser("import", help="Write the trajectories.")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--framing", choices=("original", "neutral"), default="original")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        files = fetch(args.cache)
        print(f"Fetched {len(files)} logs into {args.cache}.")
        return 0
    require_safe(args.out)
    trajectories, skipped = import_logs(fetch(args.cache), args.framing)
    for note in skipped:
        print(f"  skipped {note}", file=sys.stderr)
    # the same draw for both framings: ids differ by framing, so draw on the
    # source of each, which doesn't
    by_source = {t.source: t for t in trajectories}
    drawn = subset(
        [t.model_copy(update={"id": _plain(t.source)}) for t in trajectories],
        args.limit,
        seed=args.seed,
    )
    chosen = [by_source[t.source] for t in drawn]
    write_trajectories(chosen, args.out / "trajectories" / "sandbox_escape.yaml")
    landed = sum(t.landed_at is not None for t in chosen)
    print(
        f"Wrote {len(chosen)} trajectories ({args.framing}; {landed} reached the flag; {len(skipped)} samples skipped)."
    )
    return 0


def _plain(source: str) -> str:
    # a trajectory id from a source, framing-free
    return f"s-{hashlib.sha256(source.encode()).hexdigest()[:16]}"


if __name__ == "__main__":
    sys.exit(main())
