"""Generate `protocol-tree.excalidraw`: the Getting Started sentinel as a tree of instance paths.

`{"network": no_network(), "llm": threshold(suspicion(), reject_at=0.8)}` resolves to a root `concurrent()` at the empty path, with the rule at `network`, the threshold at `llm` and its monitor at `llm/suspicion`. Each node carries its path as a grey monospace tab on its top edge. Protocols are filled indigo and monitors outlined, as in `sentinel.excalidraw`; a small legend repeats the distinction.

Regenerate with `python3 docs/diagrams/build.py` (or this script alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _excalidraw import (  # noqa: E402
    ARROW,
    BODY,
    CONNECTIVE,
    GREY_FILL,
    INDIGO_FILL,
    INDIGO_STRONG,
    INDIGO_TEXT,
    MONO,
    MUTED,
    SANS,
    WHITE,
    Scene,
)

W = 700
BOX_H = 62
ROWS = (60, 186, 312)  # top of each tree level
COL = {"root": W / 2, "network": 175, "llm": 525}


def node(
    s: Scene,
    cx: float,
    y: float,
    w: float,
    path: str,
    fn: str,
    kind: str,
    protocol: bool,
) -> None:
    x = cx - w / 2
    if protocol:
        s.rect(x, y, w, BOX_H, INDIGO_FILL, INDIGO_STRONG, 1.5)
    else:
        s.rect(x, y, w, BOX_H, WHITE, INDIGO_STRONG, 1)
    s.text(cx, y + 15, fn, 15, MONO, INDIGO_TEXT, align="center")
    s.text(cx, y + 37, kind, 13, SANS, BODY if protocol else MUTED, align="center")
    # the instance path, as a tab on the top edge
    s.chip(x + 12, y - 11, path, size=12, color=BODY, fill=GREY_FILL, stroke=CONNECTIVE)


def edge(s: Scene, x1: float, y1: float, x2: float, y2: float) -> None:
    mid = (y1 + y2) / 2
    s.line([(x1, y1), (x1, mid), (x2, mid), (x2, y2)], ARROW, 1.5)


def main() -> Path:
    s = Scene(seed=20261106)

    s.text(
        W / 2,
        0,
        'sentinel={"network": no_network(), "llm": threshold(suspicion(), reject_at=0.8)}',
        12,
        MONO,
        MUTED,
        align="center",
    )

    root_y, child_y, leaf_y = ROWS
    edge(s, COL["root"], root_y + BOX_H, COL["network"], child_y)
    edge(s, COL["root"], root_y + BOX_H, COL["llm"], child_y)
    edge(s, COL["llm"], child_y + BOX_H, COL["llm"], leaf_y)

    node(
        s,
        COL["root"],
        root_y,
        280,
        '"" (top)',
        "concurrent()",
        "protocol: its decision takes effect",
        protocol=True,
    )
    node(
        s,
        COL["network"],
        child_y,
        250,
        "network",
        "no_network()",
        "rule: a protocol with no monitors",
        protocol=True,
    )
    node(
        s,
        COL["llm"],
        child_y,
        250,
        "llm",
        "threshold(reject_at=0.8)",
        "protocol: turns scores into a decision",
        protocol=True,
    )
    node(
        s,
        COL["llm"],
        leaf_y,
        250,
        "llm/suspicion",
        "suspicion()",
        "monitor",
        protocol=False,
    )

    # legend, under the empty left half of the bottom row
    lx, ly = 70, leaf_y + 6
    s.rect(lx, ly, 28, 18, INDIGO_FILL, INDIGO_STRONG, 1.5, radius=4)
    s.text(lx + 38, ly + 1, "protocol: decides", 13, SANS, BODY)
    s.rect(lx, ly + 30, 28, 18, WHITE, INDIGO_STRONG, 1, radius=4)
    s.text(lx + 38, ly + 31, "monitor: observes, never acts", 13, SANS, BODY)

    return s.write("protocol-tree")


if __name__ == "__main__":
    main()
