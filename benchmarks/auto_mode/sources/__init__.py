"""Importers: cases taken from existing sources, fetched at runtime into the output directory and never committed.

Each importer turns a source's items into cases (a transcript ending at the call under test, labelled by the source's own split, with `source` naming where it came from). `import_sources()` writes each source's cases to `out/imported/<source>.yaml`, which `logs` and `run` read beside the committed cases. A source that needs a login or token reads it from the environment, and without it is skipped with a message.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..cases import Case, write_cases
from ..paths import require_safe
from . import basharena, replay_trust


class SourceUnavailable(Exception):
    """A source can't be imported here, such as without the token a gated dataset needs; the message says why and what to do."""


@dataclass(frozen=True)
class Source:
    """An importer."""

    name: str
    description: str
    fetch: Callable[[Path], list[Case]]
    """Make the source's cases, fetching into the given cache directory; raises `SourceUnavailable` when it can't run here."""


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source("replay_trust", replay_trust.DESCRIPTION, replay_trust.fetch),
        Source("basharena", basharena.DESCRIPTION, basharena.fetch),
    ]
}
"""The importers, by name."""


def import_sources(names: Sequence[str], out: Path) -> tuple[list[Path], list[str]]:
    """Import each source in `names` into `out/imported/<name>.yaml`; return the files written and a message for each source skipped.

    Raises:
        UnsafeOutput: If `out`, or a link beneath it, resolves to where git would take what is written there.
    """
    require_safe(out)
    written: list[Path] = []
    skipped: list[str] = []
    for name in names:
        source = SOURCES[name]
        try:
            cases = source.fetch(out / "cache" / name)
        except SourceUnavailable as ex:
            skipped.append(f"{name} skipped: {ex}")
            continue
        file = out / "imported" / f"{name}.yaml"
        write_cases(cases, file)
        written.append(file)
    return written, skipped
