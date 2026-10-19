"""Write the transcript files for a set: `build.py a OUT_DIR` or `build.py b OUT_DIR`.

Each scenario gets an opaque id (`a01`, `a02`, ...) in a fixed shuffled order, so a file's name says nothing about its scenario.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import set_a
import set_b


def main() -> None:
    which, out = sys.argv[1], Path(sys.argv[2])
    scenarios = {"a": set_a.SCENARIOS, "b": set_b.SCENARIOS}[which]
    order = list(scenarios)
    random.Random(f"monitor-eval-{which}").shuffle(order)
    out.mkdir(parents=True, exist_ok=True)
    for n, scenario in enumerate(order, start=1):
        id = f"{which}{n:02d}"
        scenario().write(out / f"{id}.json", id)
        print(f"{id}\t{scenario.__name__}")


if __name__ == "__main__":
    main()
