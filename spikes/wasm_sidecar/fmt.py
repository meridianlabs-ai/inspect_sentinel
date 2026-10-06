"""Formats demo.sh output: python3 fmt.py {asyncprobes|probes|m4} < json"""

import json
import sys


def row(r: dict) -> str:
    shown = ("ok " + str(r["result"])) if r["ok"] else r["error"]
    return f"{r['probe']:36} {shown}"[:150]


data = json.load(sys.stdin)
kind = sys.argv[1]
if kind == "asyncprobes":
    for r in data:
        print(row(r))
elif kind == "probes":
    for r in data["report"][int(sys.argv[2]) if len(sys.argv) > 2 else 0 :][
        : int(sys.argv[3]) if len(sys.argv) > 3 else None
    ]:
        print(row(r))
elif kind == "m4":
    print("decision:", data["decision"])
    for r in data["records"]:
        print(
            "  ",
            r["status"],
            r["path"] or "<root>",
            r["factory"],
            r.get("action") or r.get("suspicion") or "",
        )
