"""Generate `stages.excalidraw`: the four stages as checkpoints on one turn of an agent loop.

The loop is drawn as a ring through two boxes, the agent's model at the top and the tool at the bottom. The four stages sit on the ring as indigo diamonds at the entrance and exit of each box: `BeforeGenerate` before the model call, `AfterGenerate` after it and before the reply reaches the agent, `BeforeToolCall` before the tool executes, `AfterToolCall` after it and before the model sees the result. The left side of the ring closes the loop: the result goes into the next request.

Regenerate with `python3 docs/diagrams/build.py` (or this script alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _excalidraw import (  # noqa: E402
    ARROW,
    BODY,
    INDIGO_MONO,
    INDIGO_TEXT,
    MONO,
    MUTED,
    SANS,
    VIOLET_FILL,
    VIOLET_LINE,
    VIOLET_TEXT,
    Scene,
)

L, R = 20, 700  # the ring's vertical sides
T, B = 86, 276  # the ring's horizontal sides, through the two boxes' centres
CX = (L + R) / 2
BOX_W, BOX_H = 160, 60
MARK_DX = 195  # checkpoint distance from the centre line


def box(s: Scene, cy: float, title: str, desc: str) -> None:
    s.rect(CX - BOX_W / 2, cy - BOX_H / 2, BOX_W, BOX_H, VIOLET_FILL, VIOLET_LINE, 1)
    s.text(CX, cy - 21, title, 18, SANS, VIOLET_TEXT, align="center")
    s.text(CX, cy + 2, desc, 13, SANS, BODY, align="center")


def checkpoint(
    s: Scene, cx: float, cy: float, name: str, desc: str, above: bool
) -> None:
    s.diamond(cx, cy, 8, INDIGO_MONO, INDIGO_TEXT)
    if above:
        s.text(cx, cy - 50, name, 15, MONO, INDIGO_TEXT, align="center")
        s.text(cx, cy - 29, desc, 13, SANS, MUTED, align="center")
    else:
        s.text(cx, cy + 15, name, 15, MONO, INDIGO_TEXT, align="center")
        s.text(cx, cy + 36, desc, 13, SANS, MUTED, align="center")


def main() -> Path:
    s = Scene(seed=20261104)
    left_edge, right_edge = CX - BOX_W / 2, CX + BOX_W / 2

    # the ring: model -> tool down the right, tool -> next request up the left
    s.arrow([(right_edge, T), (R, T), (R, B), (right_edge + 2, B)])
    s.arrow([(left_edge, B), (L, B), (L, T), (left_edge - 2, T)])

    box(s, T, "model", "generates a reply")
    box(s, B, "tool", "executes the call")

    checkpoint(
        s, CX - MARK_DX, T, "BeforeGenerate", "before the model call", above=True
    )
    checkpoint(
        s,
        CX + MARK_DX,
        T,
        "AfterGenerate",
        "before the reply reaches the agent",
        above=True,
    )
    checkpoint(
        s, CX + MARK_DX, B, "BeforeToolCall", "before the tool executes", above=False
    )
    checkpoint(
        s,
        CX - MARK_DX,
        B,
        "AfterToolCall",
        "before the model sees the result",
        above=False,
    )

    # the two sides of the ring, read inside it
    mid = (T + B) / 2
    s.text(
        R - 14,
        mid - 17,
        "the agent runs\nthe reply's tool call",
        13,
        SANS,
        MUTED,
        align="right",
    )
    s.text(
        L + 14,
        mid - 24,
        "the loop continues:\nthe result goes into\nthe next request",
        13,
        SANS,
        MUTED,
    )
    s.text(CX, mid - 8, "one turn of the agent's loop", 13, SANS, ARROW, align="center")

    return s.write("stages")


if __name__ == "__main__":
    main()
