"""The frozen trajectory corpus: fetch a version from private S3 and check every file against its committed sha256.

A version lives under `s3://<bucket>/<version>/`, the bucket named by `AUTO_MODE_CORPUS_BUCKET` (raw generation logs, imported trajectories, per-trajectory logs, and `trajectories.jsonl` labels). Only its file list with hashes is committed, as `corpus/<version>.files.jsonl`; the data itself is never committed (dataset terms: evaluation only, keep canaries). A file that is missing, extra or changed fails the load, so ablations always judge the same trajectories.

    python -m benchmarks.auto_mode.controlarena.corpus glm-v1 --dest /work/src/auto-mode-corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

BUCKET_VARIABLE = "AUTO_MODE_CORPUS_BUCKET"
"""The environment variable that names the private S3 bucket holding the corpus, which the repository does not name."""
MANIFESTS = Path(__file__).parent / "corpus"


def expected(version: str) -> dict[str, str]:
    """The committed path -> sha256 map for `version` (its canary line skipped)."""
    lines = (MANIFESTS / f"{version}.files.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return {r["path"]: r["sha256"] for r in rows if "path" in r}


def verify(root: Path, version: str) -> None:
    """Raise unless `root` holds exactly the version's files with their hashes."""
    want = expected(version)
    have = {
        str(p.relative_to(root)): p
        for p in root.rglob("*")
        if p.is_file() and p.name != "files.jsonl"
    }
    problems = [f"missing {p}" for p in sorted(want.keys() - have.keys())]
    problems += [f"unexpected {p}" for p in sorted(have.keys() - want.keys())]
    for path in sorted(want.keys() & have.keys()):
        if hashlib.sha256(have[path].read_bytes()).hexdigest() != want[path]:
            problems.append(f"changed {path}")
    if problems:
        raise ValueError(
            f"corpus {version} at {root} fails its manifest: "
            + "; ".join(problems[:10])
        )


def load(version: str, dest: Path) -> Path:
    """Fetch `version` into `dest/<version>` (if not already there) and verify it; return its directory."""
    bucket = os.environ.get(BUCKET_VARIABLE)
    if not bucket:
        raise ValueError(
            f"Set {BUCKET_VARIABLE} to the private S3 bucket that holds the corpus."
        )
    root = dest / version
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            f"s3://{bucket}/{version}/",
            str(root),
            "--only-show-errors",
        ],
        check=True,
    )
    verify(root, version)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("version")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"{load(args.version, args.dest)}: verified against {args.version}.files.jsonl"
    )


if __name__ == "__main__":
    main()
