"""Generate `sentinel.excalidraw`: monitors observe a step, the protocol decides it.

A cycle read left to right and back: the agent's step (violet) is read by two monitors (outlined indigo, the style every monitor has in these diagrams), each reports an observation, a suspicion score, to the protocol (filled indigo, the style every protocol has), and the protocol's decision returns along the bottom to take effect on the step. The decision's five actions sit on that return path, with the one this example reaches (0.91 is at or above `reject_at=0.8`) filled. Monitors have no arrow back to the step: they never act.

Regenerate with `python3 docs/diagrams/build.py` (or this script alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _excalidraw import (  # noqa: E402
    BODY,
    INDIGO_FILL,
    INDIGO_LINE,
    INDIGO_MONO,
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

STEP_X, STEP_W = 0, 160
MON_X, MON_W, MON_H = 214, 170, 62
PRO_X, PRO_W = 510, 190
TOP, BOTTOM = 0, 210  # the step box's extent
MON_YS = (18, 130)
PRO_Y, PRO_H = 54, 102
RETURN_Y = 252


def monitor(s: Scene, y: float, name: str) -> None:
    s.rect(MON_X, y, MON_W, MON_H, WHITE, INDIGO_STRONG, 1)
    s.text(MON_X + 14, y + 12, name, 14, MONO, INDIGO_TEXT)
    s.text(MON_X + 14, y + 34, "monitor: observes", 13, SANS, MUTED)


def main() -> Path:
    s = Scene(seed=20261105)

    # the agent's step
    s.rect(STEP_X, TOP, STEP_W, BOTTOM - TOP, VIOLET_FILL, VIOLET_LINE, 1)
    s.text(STEP_X + 16, TOP + 16, "agent step", 18, SANS, VIOLET_TEXT)
    s.text(
        STEP_X + 16,
        TOP + 44,
        "a tool call about to run",
        13,
        SANS,
        BODY,
        max_w=STEP_W - 32,
    )
    s.chip(
        STEP_X + 16,
        TOP + 92,
        "BeforeToolCall",
        fill=WHITE,
        stroke=VIOLET_LINE,
        color=VIOLET_TEXT,
    )
    s.text(STEP_X + 16, TOP + 128, 'bash("rm -rf\n  ~/data")', 13, MONO, BODY)

    # monitors read the step
    for y, name in zip(MON_YS, ("suspicion()", "deletes_files()"), strict=True):
        cy = y + MON_H / 2
        s.arrow([(STEP_X + STEP_W + 2, cy), (MON_X - 4, cy)], dashed=True)
        monitor(s, y, name)
    s.text(
        (STEP_X + STEP_W + MON_X) / 2,
        MON_YS[0] + MON_H / 2 - 22,
        "reads",
        12,
        SANS,
        MUTED,
        align="center",
    )

    # observations to the protocol, the score riding on each arrow
    lx = (MON_X + MON_W + PRO_X) / 2
    for (y, score), dy in zip(
        zip(MON_YS, ("0.85", "1.0"), strict=True), (-20, 20), strict=True
    ):
        cy = y + MON_H / 2
        py = PRO_Y + PRO_H / 2 + dy
        s.arrow([(MON_X + MON_W + 2, cy), (PRO_X - 4, py)])
        s.chip(lx, (cy + py) / 2 - 11, score, size=13, align="center")
    s.text(lx, MON_YS[0] - 4, "suspicion scores", 12, SANS, MUTED, align="center")

    # the protocol
    s.rect(PRO_X, PRO_Y, PRO_W, PRO_H, INDIGO_FILL, INDIGO_STRONG, 1.5)
    s.text(PRO_X + 14, PRO_Y + 14, "threshold(", 14, MONO, INDIGO_TEXT)
    s.text(PRO_X + 14, PRO_Y + 32, "  reject_at=0.8)", 14, MONO, INDIGO_TEXT)
    s.text(PRO_X + 14, PRO_Y + 62, "protocol: decides", 13, SANS, BODY)

    # the decision returns to the step
    px = PRO_X + PRO_W / 2
    sx = STEP_X + STEP_W / 2
    s.arrow(
        [(px, PRO_Y + PRO_H), (px, RETURN_Y), (sx, RETURN_Y), (sx, BOTTOM + 4)],
        color=INDIGO_STRONG,
    )
    s.text(px + 10, RETURN_Y - 26, "decision", 13, SANS, MUTED)

    actions = ["proceed", "reject", "modify", "terminate", "escalate"]
    gap = 8
    widths = [measure(a, 12, MONO) + 16 for a in actions]
    x = (sx + px) / 2 - (sum(widths) + gap * (len(actions) - 1)) / 2
    for a, w in zip(actions, widths, strict=True):
        chosen = a == "reject"
        s.chip(
            x,
            RETURN_Y + 12,
            a,
            fill=INDIGO_MONO if chosen else WHITE,
            stroke=INDIGO_TEXT if chosen else INDIGO_LINE,
            color=WHITE if chosen else INDIGO_MONO,
        )
        x += w + gap

    s.text(
        (sx + px) / 2,
        RETURN_Y + 46,
        "takes effect on the step",
        13,
        SANS,
        MUTED,
        align="center",
    )

    return s.write("sentinel")


if __name__ == "__main__":
    main()
