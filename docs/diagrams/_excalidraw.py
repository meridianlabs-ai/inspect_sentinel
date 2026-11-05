"""Shared helpers and palette for the Inspect Sentinel docs diagrams.

Each diagram is a small stdlib-only script in this directory that builds a `Scene` and writes `<name>.excalidraw` next to itself. The Quarto extension's excalidraw filter converts `![](diagrams/<name>.excalidraw)` to SVG at render time; the SVG is gitignored, the `.excalidraw` file is the committed artifact. `build.py` regenerates every diagram (and, with `--png`, renders previews).

House style, shared with the inspect_steward diagrams:

    indigo   sentinel machinery: monitors, protocols, stages
    violet   the agent and its steps
    amber    a person (`human()`)
    grey     Inspect and Scout machinery, and all connective tissue

    monospace   code: class and function names, paths, commands
    sans-serif  everything else

Clean lines (roughness 0), soft fills, text 12-20px, sized for a ~700px content column. Light theme only for now. Text is laid out with an advance-width model rather than a real font metric, so a measured width can be a few pixels off what Excalidraw would compute; centred labels use `textAlign: center` so they stay centred either way. Nothing is bound to a container, so opening a file in Excalidraw and nudging a label is safe. Ids and seeds come from a per-diagram seeded generator, so a rebuild is a no-op in git.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------- palette

INDIGO_FILL = "#e4e9fc"
INDIGO_SOFT = "#eef1fd"
INDIGO_LINE = "#c7cffb"
INDIGO_STRONG = "#8b95f2"
INDIGO_TEXT = "#3730a3"
INDIGO_MONO = "#4f46e5"

VIOLET_FILL = "#f2ecfd"
VIOLET_LINE = "#c4b5fd"
VIOLET_TEXT = "#7c3aed"

AMBER_FILL = "#fdf7dc"
AMBER_LINE = "#ecc94b"
AMBER_TEXT = "#b45309"

GREY_FILL = "#f3f4f6"
PANEL_FILL = "#f9fafb"
FRAME_LINE = "#e5e7eb"
CONNECTIVE = "#d1d5db"
ARROW = "#9ca3af"

BODY = "#374151"
MUTED = "#6b7280"
FAINT = "#9ca3af"
WHITE = "#ffffff"
NONE = "transparent"

SANS, MONO = 2, 3  # Excalidraw font ids: 1 hand-drawn, 2 Helvetica, 3 code

LINE_H = 1.25
HEAD_SEG = 14  # length of the arrow element that carries each arrowhead

# advance width as a fraction of font size, averaged over lowercase prose
ADVANCE = {SANS: 0.515, MONO: 0.600}

Align = Literal["left", "center", "right"]


def measure(s: str, size: float, font: int = SANS) -> float:
    """Estimated rendered width of one line of `s`."""
    return len(s) * size * ADVANCE[font]


def wrap(s: str, size: float, font: int, max_w: float) -> list[str]:
    """Greedy word wrap of `s` to `max_w`."""
    words, lines, line = s.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and measure(trial, size, font) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines or [""]


class Scene:
    """A list of Excalidraw elements with stable, seeded ids."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.els: list[dict[str, Any]] = []

    # ------------------------------------------------------------ primitives

    def _base(
        self, kind: str, x: float, y: float, w: float, h: float
    ) -> dict[str, Any]:
        el: dict[str, Any] = {
            "id": f"{kind}-{self.rng.getrandbits(48):012x}",
            "type": kind,
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(w, 2),
            "height": round(h, 2),
            "angle": 0,
            "strokeColor": CONNECTIVE,
            "backgroundColor": NONE,
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 0,  # house style: clean lines, no sketch roughness
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": self.rng.getrandbits(31),
            "version": 1,
            "versionNonce": self.rng.getrandbits(31),
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
        }
        self.els.append(el)
        return el

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = NONE,
        stroke: str = FRAME_LINE,
        stroke_w: float = 1,
        rounded: bool = True,
        dashed: bool = False,
        radius: float = 8,
    ) -> dict[str, Any]:
        """A box. Rounded boxes are drawn as a closed, filled line with fixed-radius corners.

        The docs build's converter (`@excalidraw/utils` 0.1.2) rounds a rectangle's corners by a quarter of its shorter side, which turns a tall card into a lozenge; a polygon keeps every corner at `radius` in any renderer.
        """
        if not rounded:
            el = self._base("rectangle", x, y, w, h)
            el["backgroundColor"] = fill
            el["strokeColor"] = stroke
            el["strokeWidth"] = stroke_w
            el["strokeStyle"] = "dashed" if dashed else "solid"
            return el
        r = min(radius, w / 2, h / 2)
        pts: list[tuple[float, float]] = []
        corners = [  # centre of each corner arc, and its start angle (degrees)
            (x + w - r, y + r, 270),
            (x + w - r, y + h - r, 0),
            (x + r, y + h - r, 90),
            (x + r, y + r, 180),
        ]
        for cx, cy, a0 in corners:
            for i in range(7):
                a = math.radians(a0 + 15 * i)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        pts.append(pts[0])
        el = self._path("line", pts, stroke, stroke_w, dashed, False)
        el["backgroundColor"] = fill
        el["startArrowhead"] = None
        el["endArrowhead"] = None
        return el

    def diamond(
        self, cx: float, cy: float, r: float, fill: str, stroke: str
    ) -> dict[str, Any]:
        el = self._base("diamond", cx - r, cy - r, 2 * r, 2 * r)
        el["backgroundColor"] = fill
        el["strokeColor"] = stroke
        el["strokeWidth"] = 1.5
        return el

    def ellipse(
        self, cx: float, cy: float, r: float, fill: str, stroke: str
    ) -> dict[str, Any]:
        el = self._base("ellipse", cx - r, cy - r, 2 * r, 2 * r)
        el["backgroundColor"] = fill
        el["strokeColor"] = stroke
        el["strokeWidth"] = 1.5
        return el

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float,
        font: int = SANS,
        color: str = BODY,
        max_w: float | None = None,
        align: Align = "left",
    ) -> float:
        """Place text and return its height.

        `x` is the left edge for `align="left"`, the centre for `"center"` and the right edge for `"right"`. With `max_w` the text is word-wrapped; explicit newlines in `s` are kept.
        """
        lines: list[str] = []
        for para in s.split("\n"):
            lines.extend(wrap(para, size, font, max_w) if max_w else [para])
        w = max(measure(line, size, font) for line in lines)
        h = len(lines) * size * LINE_H
        left = x if align == "left" else x - w / 2 if align == "center" else x - w
        el = self._base("text", left, y, w, h)
        el["strokeColor"] = color
        el["fontSize"] = size
        el["fontFamily"] = font
        el["text"] = "\n".join(lines)
        el["originalText"] = "\n".join(lines)
        el["textAlign"] = align
        el["verticalAlign"] = "top"
        el["containerId"] = None
        el["lineHeight"] = LINE_H
        el["autoResize"] = True
        # the SVG exporter derives each line's y as (i+1)*lineHeightPx - (height - baseline),
        # so baseline is the top-to-last-line-baseline distance. Absent, every y exports NaN.
        el["baseline"] = round(h - size * (LINE_H - 1), 2)
        return h

    def _path(
        self,
        kind: str,
        pts: list[tuple[float, float]],
        color: str,
        stroke_w: float,
        dashed: bool,
        curved: bool,
    ) -> dict[str, Any]:
        x0, y0 = pts[0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        el = self._base(kind, x0, y0, max(xs) - min(xs), max(ys) - min(ys))
        el["strokeColor"] = color
        el["strokeWidth"] = stroke_w
        el["strokeStyle"] = "dashed" if dashed else "solid"
        el["points"] = [[round(px - x0, 2), round(py - y0, 2)] for px, py in pts]
        el["lastCommittedPoint"] = None
        el["startBinding"] = None
        el["endBinding"] = None
        el["roundness"] = {"type": 2} if curved else None
        el["strokeSharpness"] = "round" if curved else "sharp"
        return el

    def arrow(
        self,
        pts: list[tuple[float, float]],
        color: str = ARROW,
        stroke_w: float = 1.5,
        head: str | None = "triangle",
        tail: str | None = None,
        dashed: bool = False,
        curved: bool = False,
    ) -> dict[str, Any]:
        """A straight or elbowed arrow through absolute points.

        The converter sizes an arrowhead by the arrow's length, so a long arrow gets an outsized head. Anything longer than `HEAD_SEG` is drawn as a plain line plus a short arrow over its last `HEAD_SEG` pixels, which keeps every head the same size.
        """
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        seg = math.hypot(x2 - x1, y2 - y1)
        total = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(pts, pts[1:], strict=False)
        )
        if head is None or tail is not None or curved or total <= HEAD_SEG:
            el = self._path("arrow", pts, color, stroke_w, dashed, curved)
            el["startArrowhead"] = tail
            el["endArrowhead"] = head
            return el
        if seg < HEAD_SEG:  # a short final leg carries the head by itself
            cut = (x1, y1)
            self.line(pts[:-1], color, stroke_w, dashed)
        else:
            f = (seg - HEAD_SEG) / seg
            cut = (x1 + (x2 - x1) * f, y1 + (y2 - y1) * f)
            self.line([*pts[:-1], cut], color, stroke_w, dashed)
        el = self._path("arrow", [cut, pts[-1]], color, stroke_w, False, False)
        el["startArrowhead"] = None
        el["endArrowhead"] = head
        return el

    def line(
        self,
        pts: list[tuple[float, float]],
        color: str = CONNECTIVE,
        stroke_w: float = 1,
        dashed: bool = False,
    ) -> dict[str, Any]:
        el = self._path("line", pts, color, stroke_w, dashed, False)
        el["startArrowhead"] = None
        el["endArrowhead"] = None
        return el

    # ------------------------------------------------------------ composites

    def chip(
        self,
        x: float,
        y: float,
        s: str,
        size: float = 12,
        font: int = MONO,
        color: str = INDIGO_MONO,
        fill: str = WHITE,
        stroke: str = INDIGO_LINE,
        pad_x: float = 8,
        pad_y: float = 3,
        align: Align = "left",
    ) -> tuple[float, float]:
        """A small labelled pill; returns its (width, height). `x` follows `align`."""
        w = measure(s, size, font) + 2 * pad_x
        h = size * LINE_H + 2 * pad_y
        left = x if align == "left" else x - w / 2 if align == "center" else x - w
        self.rect(left, y, w, h, fill, stroke, 1, radius=5)
        self.text(left + w / 2, y + pad_y, s, size, font, color, align="center")
        return w, h

    # ------------------------------------------------------------ output

    def write(self, name: str) -> Path:
        """Write `<name>.excalidraw` into this directory."""
        path = DIAGRAMS_DIR / f"{name}.excalidraw"
        doc = {
            "type": "excalidraw",
            "version": 2,
            "source": "inspect_sentinel docs/diagrams",
            "elements": self.els,
            "appState": {"viewBackgroundColor": WHITE, "gridSize": None},
            "files": {},
        }
        path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {path}")
        return path


DIAGRAMS_DIR = Path(__file__).parent
