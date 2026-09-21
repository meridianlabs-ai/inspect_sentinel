# Contributing to Inspect Sentinel

Thanks for your interest in contributing. Bug reports, doc fixes, and core changes are all welcome.

## Development setup

Inspect Sentinel uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, and requires Python 3.10+.

The development environment pins 3.13 (`.python-version`). CI covers both ends of the supported range: 3.10 proves the floor still works, and 3.14 exercises the newest interpreter. `make check` only ever sees the 3.13 environment, so a construct that is new in 3.11 or later passes locally and fails in CI on 3.10. To reproduce the floor locally:

```bash
uv venv --python 3.10 /tmp/py310 && uv pip install --python /tmp/py310/bin/python -e . --group dev
.venv/bin/pyright --pythonpath /tmp/py310/bin/python
```

```bash
git clone https://github.com/meridianlabs-ai/inspect_sentinel
cd inspect_sentinel
uv sync --group dev
```

This installs the package in editable mode along with the dev tools (ruff, pyright, pytest).

## Checks and tests

Before opening a PR, make sure these pass:

```bash
make check   # pyright + ruff check --fix + ruff format
make test    # pytest
```

To run a single test:

```bash
uv run pytest tests/path/to/test.py::test_name -v
```

## Code style

- Type-annotate all functions (including tests). Use modern syntax (`X | None`, `list[str]`).
- Google-style docstrings for public APIs.
- `ruff` handles formatting and import order; don't fight it.
- Don't catch exceptions defensively — let unexpected errors propagate.

See [`AGENTS.md`](AGENTS.md) for the full conventions used in this repo.

## Design documents

`design/` holds the design this package implements. A change to public API should cite the section it implements, and a change that departs from the design should update the design document in the same PR.

## Building the docs

Docs live in `docs/` and are built with [Quarto](https://quarto.org). The Quarto CLI and the doc-build dependencies are in the `doc` group:

```bash
uv sync --group doc
cd docs
source ../.venv/bin/activate
quarto render        # outputs to docs/_site/
quarto preview       # live-reload server
```

The venv must be activated (not just `uv run quarto`) so the reference-page filter can import `griffe` and the `inspect_sentinel` package itself.

## Updating the docs extension

`docs/_extensions/` holds a checked-in copy of the [inspect-docs](https://github.com/meridianlabs-ai/inspect-docs) Quarto extension, which contributes the project type `_quarto.yml` declares. It is committed (per Quarto convention) so that a fresh clone can render without any install step.

To pull a newer version:

```bash
cd docs
mv _quarto.yml _quarto.yml.bak
quarto update meridianlabs-ai/inspect-docs --no-prompt
mv _quarto.yml.bak _quarto.yml
```

The `mv` is a bootstrap workaround, not superstition: `quarto add`/`update` parses `_quarto.yml` to choose an install directory, so it chokes on the very project type the extension provides. Commit the resulting diff.

## Pull requests

- Branch from `main`; we squash-merge.
- Keep PRs focused on one change.
- Add or update tests for behavior changes.
- Update `CHANGELOG.md` under the `## Unreleased` heading if the change is user-visible.

## Reporting issues

Use the [issue tracker](https://github.com/meridianlabs-ai/inspect_sentinel/issues). For bugs, include the `inspect_sentinel` and `inspect_ai` versions (`pip show inspect-sentinel inspect-ai`) and a minimal repro.
