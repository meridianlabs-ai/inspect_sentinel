"""Where the benchmark may write: its output holds the judges' prompts and the imported datasets, which must never be committed.

An output directory may be anywhere outside the repository, but inside it only where git ignores it and tracks nothing under it. That holds for where each artifact actually lands: the directory's own resolved location, and the resolved target of every link beneath it, so a link such as `out/scans -> ../published` can't carry the scans somewhere git would take them. A destination that is not a directory, a dangling link's included, is checked as a file, so a rule that ignores only a directory of that name doesn't cover it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent.parent
"""The repository the benchmark is in."""


class UnsafeOutput(ValueError):
    """An output directory, or a link beneath it, resolves to where git would take what is written there."""


def check_out(out: Path, repository: Path = REPOSITORY) -> str | None:
    """Why writing under `out` is not safe, or None where it is: `out` itself, and every link beneath it, must resolve outside `repository`, or inside it where git ignores the destination and tracks nothing under it."""
    root = repository.resolve()
    reason = _unsafe(out.resolve(), root, f"--out {out}", directory=True)
    if reason is not None or not out.is_dir():
        return reason
    seen: set[Path] = set()
    stack = [out]
    while stack:
        directory = stack.pop()
        real = directory.resolve()
        if real in seen:
            continue
        seen.add(real)
        for child in directory.iterdir():
            if child.is_symlink():
                # a link to a directory is checked as one; any other link,
                # dangling ones included, can only have a file written through
                # it, so a rule for a directory of that name doesn't cover it
                reason = _unsafe(
                    child.resolve(),
                    root,
                    f"{child}, a link under --out {out} to {child.resolve()},",
                    directory=child.is_dir(),
                )
                if reason is not None:
                    return reason
            if child.is_dir():
                stack.append(child)
    return None


def require_safe(out: Path, repository: Path = REPOSITORY) -> None:
    """Refuse to write under `out` unless `check_out()` finds it safe.

    Raises:
        UnsafeOutput: If `out`, or a link beneath it, resolves to where git would take what is written there.
    """
    reason = check_out(out, repository)
    if reason is not None:
        raise UnsafeOutput(reason)


def _unsafe(resolved: Path, root: Path, label: str, directory: bool) -> str | None:
    if not resolved.is_relative_to(root):
        return None
    relative = resolved.relative_to(root).as_posix()
    if relative == ".":
        return f"{label} is the repository itself; choose a directory git ignores, such as benchmarks/auto_mode/out, or one outside the repository."
    try:
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "check-ignore",
                "-q",
                "--no-index",
                f"{relative}/" if directory else relative,
            ],
            capture_output=True,
        )
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", relative],
            capture_output=True,
            text=True,
        )
    except OSError as ex:
        return f"{label} is inside the repository, and git could not be run to check that it is ignored ({ex}); choose a directory outside the repository."
    if tracked.returncode == 0 and tracked.stdout.strip():
        return f"{label} holds files git tracks; choose a directory with none."
    if ignored.returncode != 0 or tracked.returncode != 0:
        return f"{label} is inside the repository but git does not ignore it (or could not tell), so its prompts and datasets could be committed; use benchmarks/auto_mode/out or a directory outside the repository."
    return None
