"""Per-call scores of each labelled multi-call message, by variant: the evidence behind the scorecard's turn measures.

    .venv/bin/python spikes/monitor_eval/turns_table.py KEY.json RESULTS.jsonl [...] [--stage tool_call]

KEY.json is a turn key (`turn_probes.py`); with a set key the labelled messages come from `scorecard.SET_TURNS`. Each cell is a call's mean score over runs, `*` marking a bad call and `~` a related one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from scorecard import SET_TURNS, turn_of  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--stage", default="tool_call")
    args = parser.parse_args()
    key: dict[str, Any] = json.loads(args.key.read_text())
    labelled: list[tuple[str, str, int, list[int], list[int]]] = []
    for t, k in key.items():
        if "kind" in k:
            labelled.append((t, k["kind"], k["turn"], k["bad"], k.get("related", [])))
        elif "turns" in k:
            for m in k["turns"]:
                labelled.append((t, m["kind"] or "plain", m["turn"], m["bad"], []))
        else:
            for n, bad in SET_TURNS.get(t, []):
                labelled.append((t, "set" if bad else "set-benign", n, bad, []))
    rows: list[dict[str, Any]] = []
    for path in args.results:
        rows += [json.loads(line) for line in path.open()]
    scores: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    labels: list[str] = []
    for r in rows:
        if r["stage"] != args.stage or r["suspicion"] is None:
            continue
        tc = turn_of(r["step"])
        if tc is None:
            continue
        label = r["label"] or r["variant"]
        if label not in labels:
            labels.append(label)
        scores[(label, r["transcript"], tc[0], tc[1])].append(r["suspicion"])
    print("| message | kind | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for t, kind, n, bad, related in labelled:
        cells = []
        for label in labels:
            calls = sorted(
                c for (lb, tt, nn, c) in scores if lb == label and tt == t and nn == n
            )
            cells.append(
                " ".join(
                    f"{statistics.mean(scores[(label, t, n, c)]):.2f}"
                    + ("*" if c in bad else "~" if c in related else "")
                    for c in calls
                )
            )
        print(f"| {t} t{n:02d} | {kind} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
