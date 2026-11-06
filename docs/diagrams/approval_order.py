"""Generate `approval-order.excalidraw`: the order of approval, review and the sentinel around one tool call.

A single column read top to bottom, matching inspect_ai's tool execution (`model/_call_tools.py`): Inspect's approval policy, then the sentinel at `BeforeToolCall`; the tool executes; then Inspect's review policy, then the sentinel at `AfterToolCall`; then the result goes to the model. Inspect's own policies are grey, the sentinel indigo, the tool violet. Brackets on the left group the two checks before the call and the two after it.

Regenerate with `python3 docs/diagrams/build.py` (or this script alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _excalidraw import (  # noqa: E402
    BODY,
    CONNECTIVE,
    GREY_FILL,
    INDIGO_FILL,
    INDIGO_LINE,
    INDIGO_STRONG,
    INDIGO_TEXT,
    MONO,
    MUTED,
    SANS,
    VIOLET_FILL,
    VIOLET_LINE,
    VIOLET_TEXT,
    WHITE,
    Scene,
    measure,
)

COL_X, COL_W = 170, 380
BOX_H, GAP = 52, 30
BRACKET_X = COL_X - 22

STEPS = [
    # title, code chip, fill, line, title colour
    ("Inspect approval policy", "approval=", GREY_FILL, CONNECTIVE, BODY),
    ("sentinel", "BeforeToolCall", INDIGO_FILL, INDIGO_STRONG, INDIGO_TEXT),
    ("tool executes", None, VIOLET_FILL, VIOLET_LINE, VIOLET_TEXT),
    ("Inspect review policy", "review=", GREY_FILL, CONNECTIVE, BODY),
    ("sentinel", "AfterToolCall", INDIGO_FILL, INDIGO_STRONG, INDIGO_TEXT),
]


def bracket(s: Scene, y1: float, y2: float, text: str) -> None:
    s.line(
        [(BRACKET_X + 8, y1), (BRACKET_X, y1), (BRACKET_X, y2), (BRACKET_X + 8, y2)],
        CONNECTIVE,
        1.5,
    )
    s.text(BRACKET_X - 12, (y1 + y2) / 2 - 9, text, 14, SANS, MUTED, align="right")


def main() -> Path:
    s = Scene(seed=20261108)
    cx = COL_X + COL_W / 2

    s.text(
        cx, 0, "the model's reply asks for a tool call", 13, SANS, MUTED, align="center"
    )
    y = 48
    s.arrow([(cx, 20), (cx, y - 3)])

    tops = []
    for title, code, fill, line, color in STEPS:
        tops.append(y)
        s.rect(COL_X, y, COL_W, BOX_H, fill, line, 1.5 if fill == INDIGO_FILL else 1)
        s.text(COL_X + 18, y + 15, title, 17, SANS, color)
        if code:
            w = measure(code, 13, MONO) + 16
            s.chip(
                COL_X + COL_W - 16 - w,
                y + 14,
                code,
                size=13,
                color=INDIGO_TEXT if fill == INDIGO_FILL else BODY,
                stroke=INDIGO_LINE if fill == INDIGO_FILL else CONNECTIVE,
                fill=WHITE,
            )
        y += BOX_H
        s.arrow([(cx, y + 2), (cx, y + GAP - 3)])
        y += GAP

    s.text(cx, y + 2, "the result goes to the model", 13, SANS, MUTED, align="center")

    bracket(s, tops[0], tops[1] + BOX_H, "before the call")
    bracket(s, tops[3], tops[4] + BOX_H, "after the call")

    return s.write("approval-order")


if __name__ == "__main__":
    main()
