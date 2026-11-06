"""Generate `replay.excalidraw`: running a live monitor over stored transcripts with Inspect Scout.

Three rows read as a Z: Inspect logs, then the steps rebuilt from each transcript (violet, the agent's steps, named by stage as in `stages.excalidraw`); down into a Scout `scan()` panel that holds the same monitor wrapped by `as_scanner()`; down to one result per step and across to the results dataframe (`scan_results_df()`). Inspect and Scout machinery is grey, the monitor and its results indigo.

Regenerate with `python3 docs/diagrams/build.py` (or this script alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _excalidraw import (  # noqa: E402
    BODY,
    CONNECTIVE,
    FRAME_LINE,
    GREY_FILL,
    INDIGO_LINE,
    INDIGO_MONO,
    INDIGO_SOFT,
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

W = 700
CARD_W, GAP = 300, 100
RIGHT_X = CARD_W + GAP
ROW_A, ROW_B, ROW_C = 8, 196, 372
CARD_H = 132
SCAN_H = 132
PAD = 16


def logs_card(s: Scene) -> None:
    x, y = 0, ROW_A
    # a stack of logs: two ghost cards behind the face
    s.rect(x + 12, y - 8, CARD_W - 24, CARD_H, PANEL_FILL, FRAME_LINE)
    s.rect(x + 6, y - 4, CARD_W - 12, CARD_H, PANEL_FILL, FRAME_LINE)
    s.rect(x, y, CARD_W, CARD_H, GREY_FILL, CONNECTIVE)
    s.text(x + PAD, y + 14, "Inspect logs", 17, SANS, BODY)
    s.text(
        x + PAD,
        y + 40,
        "eval logs you already have, one transcript per sample",
        13,
        SANS,
        BODY,
        max_w=CARD_W - 2 * PAD,
    )
    s.chip(
        x + PAD,
        y + 88,
        'transcripts_from("./logs")',
        size=12,
        color=BODY,
        stroke=CONNECTIVE,
    )


def steps_card(s: Scene) -> None:
    x, y = RIGHT_X, ROW_A
    s.rect(x, y, CARD_W, CARD_H, VIOLET_FILL, VIOLET_LINE)
    s.text(x + PAD, y + 14, "rebuilt steps", 17, SANS, VIOLET_TEXT)
    cx, cy = x + PAD, y + 46
    for name in [
        "BeforeGenerate",
        "AfterGenerate",
        "BeforeToolCall",
        "AfterToolCall",
        "...",
    ]:
        w, h = s.chip(0, -100, name, size=11, pad_x=6, pad_y=2)  # measure only
        s.els = s.els[:-2]
        if cx + w > x + CARD_W - PAD:
            cx, cy = x + PAD, cy + h + 6
        s.chip(
            cx,
            cy,
            name,
            size=11,
            pad_x=6,
            pad_y=2,
            color=VIOLET_TEXT,
            stroke=VIOLET_LINE,
        )
        cx += w + 6
    s.text(
        x + PAD,
        y + CARD_H - 36,
        "from each transcript, in order, as the agent ran",
        13,
        SANS,
        MUTED,
        max_w=CARD_W - 2 * PAD,
    )


def scan_panel(s: Scene) -> None:
    x, y = 0, ROW_B
    s.rect(x, y, W, SCAN_H, PANEL_FILL, CONNECTIVE)
    s.text(x + PAD, y + 14, "scan()", 17, MONO, BODY)
    s.text(
        x + PAD,
        y + 40,
        "Inspect Scout runs\nthe scanner over every\ntranscript",
        13,
        SANS,
        MUTED,
    )

    # as_scanner( wraps the monitor
    ox, oy, ow, oh = 210, y + 16, W - 210 - PAD, SCAN_H - 32
    s.rect(ox, oy, ow, oh, WHITE, INDIGO_LINE)
    s.text(ox + 14, oy + 12, "as_scanner(", 15, MONO, INDIGO_TEXT)
    ix, iy, iw, ih = ox + 28, oy + 42, ow - 56, 40
    s.rect(ix, iy, iw, ih, WHITE, INDIGO_STRONG)
    s.text(ix + 14, iy + 11, "deletes_files()", 14, MONO, INDIGO_TEXT)
    s.text(
        ix + iw - 14,
        iy + 12,
        "the same monitor as live",
        13,
        SANS,
        MUTED,
        align="right",
    )


def results_card(s: Scene) -> None:
    x, y = 0, ROW_C
    s.rect(x, y, CARD_W, CARD_H, INDIGO_SOFT, INDIGO_LINE)
    s.text(x + PAD, y + 14, "one result per step", 17, SANS, INDIGO_TEXT)
    rows = [
        ("step", "stage", "value"),
        ("call_1", "tool_call", "0.0"),
        ("call_2", "tool_call", "0.0"),
        ("call_3", "tool_call", "1.0"),
    ]
    ty = y + 46
    for i, (a, b, c) in enumerate(rows):
        color = MUTED if i == 0 else INDIGO_MONO
        s.text(x + PAD, ty, a, 12, MONO, color)
        s.text(x + PAD + 62, ty, b, 12, MONO, color)
        s.text(x + CARD_W - PAD, ty, c, 12, MONO, color, align="right")
        ty += 19


def dataframe_card(s: Scene) -> None:
    x, y = RIGHT_X, ROW_C
    s.rect(x, y, CARD_W, CARD_H, GREY_FILL, CONNECTIVE)
    s.text(x + PAD, y + 14, "results dataframe", 17, SANS, BODY)
    s.chip(x + PAD, y + 42, "scan_results_df()", size=12, color=BODY, stroke=CONNECTIVE)
    # a small grid: a header row and three data rows
    gx, gy, gw, gh = x + PAD, y + 76, CARD_W - 2 * PAD, 40
    s.rect(gx, gy, gw, gh, WHITE, CONNECTIVE, rounded=False)
    s.rect(gx, gy, gw, gh / 4, FRAME_LINE, CONNECTIVE, rounded=False)
    for i in range(1, 4):
        s.line([(gx, gy + gh * i / 4), (gx + gw, gy + gh * i / 4)], CONNECTIVE)
    for f in (0.25, 0.6, 0.8):
        s.line([(gx + gw * f, gy), (gx + gw * f, gy + gh)], CONNECTIVE)


def main() -> Path:
    s = Scene(seed=20261109)

    logs_card(s)
    steps_card(s)
    scan_panel(s)
    results_card(s)
    dataframe_card(s)

    mid_a = ROW_A + CARD_H / 2
    s.arrow([(CARD_W + 8, mid_a), (RIGHT_X - 8, mid_a)])
    sx = RIGHT_X + CARD_W / 2
    s.arrow([(sx, ROW_A + CARD_H + 4), (sx, ROW_B - 4)])
    s.arrow([(CARD_W / 2, ROW_B + SCAN_H + 4), (CARD_W / 2, ROW_C - 4)])
    mid_c = ROW_C + CARD_H / 2
    s.arrow([(CARD_W + 8, mid_c), (RIGHT_X - 8, mid_c)])

    return s.write("replay")


if __name__ == "__main__":
    main()
