"""Generate `concurrent-sequential.excalidraw`: the two combining protocols side by side.

Left, `concurrent()`: the step fans out to every child at once, each child's decision is shown under it, and the strongest one (here `reject`) is the panel's result, with the ranking `terminate > reject > modify > escalate > proceed` as a ladder beside it. Right, `sequential()`: the step goes down a chain, rule then threshold then a person (`human()`, amber), each `escalate` handing it to the next, and the first decision that is not `escalate` ends the chain.

Regenerate with `python3 docs/diagrams/build.py` (or this script alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _excalidraw import (  # noqa: E402
    AMBER_FILL,
    AMBER_LINE,
    AMBER_TEXT,
    ARROW,
    BODY,
    FRAME_LINE,
    INDIGO_FILL,
    INDIGO_LINE,
    INDIGO_MONO,
    INDIGO_STRONG,
    INDIGO_TEXT,
    MONO,
    MUTED,
    PANEL_FILL,
    SANS,
    VIOLET_FILL,
    VIOLET_LINE,
    VIOLET_TEXT,
    WHITE,
    Scene,
)

PANEL_W, PANEL_H, GAP = 340, 430, 20
PAD = 16
CHILD_Y = 134
STEP_Y = 86
CHILD_H = 50


def header(s: Scene, x0: float, fn: str, desc: str) -> None:
    s.rect(x0, 0, PANEL_W, PANEL_H, PANEL_FILL, FRAME_LINE, 1)
    s.text(x0 + PAD, 16, fn, 18, MONO, INDIGO_TEXT)
    s.text(x0 + PAD, 42, desc, 13, SANS, BODY, max_w=PANEL_W - 2 * PAD)


def step_chip(s: Scene, cx: float) -> float:
    """The step entering the panel; returns its bottom y."""
    _, h = s.chip(
        cx,
        STEP_Y,
        "the step",
        size=12,
        font=SANS,
        color=VIOLET_TEXT,
        fill=VIOLET_FILL,
        stroke=VIOLET_LINE,
        align="center",
    )
    return STEP_Y + h


def child(
    s: Scene, x: float, y: float, w: float, fn: str, kind: str, person: bool = False
) -> None:
    fill, line, color = (
        (AMBER_FILL, AMBER_LINE, AMBER_TEXT)
        if person
        else (INDIGO_FILL, INDIGO_STRONG, INDIGO_TEXT)
    )
    s.rect(x, y, w, CHILD_H, fill, line, 1)
    s.text(x + w / 2, y + 9, fn, 12, MONO, color, align="center")
    s.text(x + w / 2, y + 28, kind, 12, SANS, MUTED, align="center")


def decision(
    s: Scene, cx: float, y: float, action: str, chosen: bool = False
) -> tuple[float, float]:
    return s.chip(
        cx,
        y,
        action,
        size=12,
        align="center",
        fill=INDIGO_MONO if chosen else WHITE,
        stroke=INDIGO_TEXT if chosen else INDIGO_LINE,
        color=WHITE if chosen else INDIGO_MONO,
    )


def concurrent_panel(s: Scene, x0: float) -> None:
    header(
        s, x0, "concurrent()", "runs every child at once; the strongest decision wins"
    )
    cw, cg = 96, 11
    xs = [x0 + PAD + i * (cw + cg) for i in range(3)]
    centers = [x + cw / 2 for x in xs]
    fan_x = x0 + PANEL_W / 2
    top = step_chip(s, fan_x)

    # the step fans out to every child at once
    for cx in centers:
        s.arrow(
            [(fan_x, top + 2), (fan_x, top + 12), (cx, top + 12), (cx, CHILD_Y - 3)]
        )

    children = [
        ("no_network()", "rule", "proceed"),
        ("threshold()", "protocol", "reject"),
        ("triage()", "rule", "escalate"),
    ]
    chip_y = CHILD_Y + CHILD_H + 14
    for x, cx, (fn, kind, action) in zip(xs, centers, children, strict=True):
        child(s, x, CHILD_Y, cw, fn, kind)
        s.line([(cx, CHILD_Y + CHILD_H), (cx, chip_y)], ARROW, 1.5)
        decision(s, cx, chip_y, action)

    # the strongest decision is the panel's result
    res_cx, res_y = x0 + 100, 270
    for cx in centers:
        s.arrow(
            [
                (cx, chip_y + 22),
                (cx, chip_y + 40),
                (res_cx, chip_y + 40),
                (res_cx, res_y - 3),
            ]
        )
    decision(s, res_cx, res_y, "reject", chosen=True)
    s.text(
        res_cx, res_y + 30, "concurrent()'s\ndecision", 13, SANS, BODY, align="center"
    )

    # the ranking, strongest first, as a ladder with a thin arrow beside it
    lx = x0 + 258
    s.text(lx, 246, "strongest", 12, SANS, MUTED, align="center")
    y = 266
    for action in ["terminate", "reject", "modify", "escalate", "proceed"]:
        _, h = s.chip(
            lx,
            y,
            action,
            size=11,
            pad_x=6,
            pad_y=2,
            align="center",
            fill=INDIGO_MONO if action == "reject" else WHITE,
            stroke=INDIGO_TEXT if action == "reject" else INDIGO_LINE,
            color=WHITE if action == "reject" else INDIGO_MONO,
        )
        y += h + 5
    s.text(lx, y, "weakest", 12, SANS, MUTED, align="center")


def sequential_panel(s: Scene, x0: float) -> None:
    header(
        s,
        x0,
        "sequential()",
        "runs children in order; each escalate passes the step on",
    )
    cw = 150
    cx = x0 + PAD + cw / 2 + 6
    top = step_chip(s, cx)
    s.arrow([(cx, top + 2), (cx, CHILD_Y - 3)])

    links = [
        ("triage()", "rule", False),
        ("threshold()", "protocol", False),
        ("human()", "a person", True),
    ]
    pitch = CHILD_H + 40
    for i, (fn, kind, person) in enumerate(links):
        y = CHILD_Y + i * pitch
        child(s, cx - cw / 2, y, cw, fn, kind, person)
        if i < len(links) - 1:
            s.arrow([(cx, y + CHILD_H + 2), (cx, y + pitch - 3)])
            s.chip(cx + 10, y + CHILD_H + 9, "escalate", size=12)

    # the first decision that is not escalate ends the chain
    y = CHILD_Y + 2 * pitch + CHILD_H / 2
    ex = cx + cw / 2
    s.arrow([(ex + 2, y), (ex + 40, y)])
    w, _ = s.chip(
        ex + 44,
        y - 11,
        "reject",
        size=12,
        fill=INDIGO_MONO,
        stroke=INDIGO_TEXT,
        color=WHITE,
    )
    s.text(ex + 44 + w / 2, y + 16, "decides", 12, SANS, MUTED, align="center")
    s.text(
        x0 + PAD,
        CHILD_Y + 3 * pitch - 8,
        "The first decision that is not escalate ends the chain; later children do not run.",
        13,
        SANS,
        BODY,
        max_w=PANEL_W - 2 * PAD,
    )


def main() -> Path:
    s = Scene(seed=20261107)
    concurrent_panel(s, 0)
    sequential_panel(s, PANEL_W + GAP)
    return s.write("concurrent-sequential")


if __name__ == "__main__":
    main()
