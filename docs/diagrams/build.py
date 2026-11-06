"""Regenerate every diagram, and optionally render previews.

    python3 docs/diagrams/build.py                 # rewrite the .excalidraw files
    python3 docs/diagrams/build.py --png DIR       # also render SVG + PNG previews into DIR

The Quarto docs build converts `.excalidraw` images itself (the inspect-docs extension's excalidraw filter), so the `.excalidraw` files are all a docs build needs. `--png` is for looking at a diagram while working on it: it renders each one to SVG with the same converter the build uses, then to a 2x PNG with `rsvg-convert` if installed, else headless Google Chrome. It requires Node.js; the converter's npm dependencies are installed on first use.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# one script per diagram; each has a main() that writes its .excalidraw and returns the path
DIAGRAMS = [
    "stages",
    "sentinel",
    "protocol_tree",
    "concurrent_sequential",
    "approval_order",
    "replay",
]

HERE = Path(__file__).parent
CONVERTER_DIR = (
    HERE.parent / "_extensions/meridianlabs-ai/inspect-docs/resources/excalidraw"
)
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def render_svg(src: Path, svg: Path) -> None:
    if not (CONVERTER_DIR / "node_modules").is_dir():
        print("installing converter dependencies (first use) ...")
        subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund"],
            cwd=CONVERTER_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            "node",
            str(CONVERTER_DIR / "excalidraw-to-svg.mjs"),
            str(src),
            str(svg),
            "--theme",
            "light",
        ],
        check=True,
        capture_output=True,
    )


def render_png(svg: Path, png: Path) -> None:
    if shutil.which("rsvg-convert"):
        subprocess.run(
            ["rsvg-convert", "-z", "2", "-b", "white", str(svg), "-o", str(png)],
            check=True,
        )
        return
    head = svg.read_text()[:2000]
    w = float(re.search(r'width="([\d.]+)', head).group(1))  # type: ignore[union-attr]
    h = float(re.search(r'height="([\d.]+)', head).group(1))  # type: ignore[union-attr]
    page = png.with_suffix(".html")
    page.write_text(
        "<!doctype html><html><body style='margin:0;background:#fff'>"
        f"<img src='{svg.name}' width='{w}' height='{h}' style='display:block'>"
        "</body></html>"
    )
    subprocess.run(
        [
            str(CHROME),
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size={int(w) + 1},{int(h) + 1}",
            f"--screenshot={png}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    page.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--png", type=Path, metavar="DIR", help="render SVG and PNG previews into DIR"
    )
    args = parser.parse_args()

    written = [importlib.import_module(name).main() for name in DIAGRAMS]

    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
        for src in written:
            svg = args.png / f"{src.stem}.svg"
            png = args.png / f"{src.stem}.png"
            render_svg(src, svg)
            render_png(svg, png)
            print(f"  previewed {png}")


if __name__ == "__main__":
    main()
