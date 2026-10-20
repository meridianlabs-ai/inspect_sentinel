# Contributing to Inspect Sentinel

Thanks for your interest in contributing. Bug reports, doc fixes, and core changes are all welcome.

## Development setup

Inspect Sentinel uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, and requires Python 3.11+.

The development environment pins 3.13 (`.python-version`). CI covers both ends of the supported range: 3.11 proves the floor still works, and 3.14 exercises the newest interpreter. `make check` only ever sees the 3.13 environment, so a construct that is new in 3.12 or later passes locally and fails in CI on 3.11. To reproduce the floor locally:

```bash
uv venv --python 3.11 /tmp/py311
uv pip install --python /tmp/py311/bin/python git+https://github.com/UKGovernmentBEIS/inspect_ai.git
uv pip install --python /tmp/py311/bin/python -e ".[dev]"
.venv/bin/pyright --pythonpath /tmp/py311/bin/python
```

Development tracks `inspect_ai` from `main`, so install that first (this is what CI does), then the package in editable mode with the dev tools (ruff, pyright, pytest):

```bash
git clone https://github.com/meridianlabs-ai/inspect_sentinel
cd inspect_sentinel
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/UKGovernmentBEIS/inspect_ai.git
uv pip install -e ".[dev]"
```

Tracking `main` is deliberate: sentinel and inspect_ai co-develop, and most sentinel changes need hooks that have not shipped yet. `pyproject.toml` names inspect_ai by a PyPI floor, which the git install satisfies, so the last line leaves it in place. To pick up new inspect_ai commits, run `uv pip install --upgrade git+https://github.com/UKGovernmentBEIS/inspect_ai.git`. To develop against a local inspect_ai checkout instead, install it editable: `uv pip install -e ../inspect_ai`.

Run Python commands from the activated venv, or reference `.venv/bin/` directly; `make check` and `make test` call the tools directly. Do **not** use `uv sync` or `uv run`: there is no `uv.lock`, so they lock and sync the project, and silently replace inspect_ai's `main` (or your editable checkout) with a PyPI release.

## Checks and tests

Before opening a PR, make sure these pass:

```bash
make check   # pyright + ruff check --fix + ruff format
make test    # pytest
```

To run a single test:

```bash
pytest tests/path/to/test.py::test_name -v
```

The Envoy shim's tests run a real Envoy in Docker, and are skipped where Docker isn't running. To test against another Envoy release, set `SENTINEL_ENVOY_IMAGE`, such as `envoyproxy/envoy:v1.39.1`.

The shim's gRPC code in `src/inspect_sentinel/sidecar/envoy/_proto` is generated from Envoy's protos. Don't edit it: change `scripts/generate_envoy_protos.py`, or the Envoy release it names, and run `make envoy-protos`.

## Code style

- Type-annotate all functions (including tests). Use modern syntax (`X | None`, `list[str]`).
- Google-style docstrings for public APIs.
- `ruff` handles formatting and import order; don't fight it.
- Don't catch exceptions defensively — let unexpected errors propagate.

See [`AGENTS.md`](AGENTS.md) for the full conventions used in this repo.

## Design documents

`design/` holds the design this package implements. A change to public API should cite the section it implements, and a change that departs from the design should update the design document in the same PR.

## Building the docs

Docs live in `docs/` and are built with [Quarto](https://quarto.org). The Quarto CLI and the doc-build dependencies are in the `doc` extra, which `dev` includes:

```bash
uv pip install -e ".[doc]"
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

## Commit messages and releases

We use [Conventional Commits](https://www.conventionalcommits.org/). Because we squash-merge, **the PR title becomes the commit message**, so the title is what matters. Format it as `<type>: <description>`; `pr-title-lint` enforces it.

Releases are automated with [Release Please](https://github.com/googleapis/release-please): **don't edit `CHANGELOG.md` or bump the version by hand.** Release Please reads the merged commit types, opens a release PR that updates the changelog and version, and merging that PR tags the release; the publish then runs once a maintainer approves the deployment.

Choose the type deliberately. `feat:` and `fix:` drive the version bump and headline the release notes:

| Type | Effect |
| --- | --- |
| `feat:` | a user-facing feature; bumps the minor version |
| `fix:` | a user-facing bug fix; bumps the patch version |
| `perf:`, `revert:` | appear in the release notes (no bump on their own) |
| `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` | hidden from the release notes |

Anything that isn't a user-facing feature or fix should avoid `feat:`/`fix:` so it stays out of the headline sections.

### The inspect_ai pin at release time

`pyproject.toml` names inspect_ai by a PyPI floor, and every CI build installs inspect_ai's `main` over it, except on the Release Please release PR. That PR tests against PyPI, the way a published sentinel is installed, and two workflows keep its floor honest:

- `release-pin-deps.yml` raises the `inspect-ai` floor to the latest PyPI version and pushes it to the release PR. It runs on release-PR events and every two hours, because an inspect_ai release produces no event in this repository. The raised floor lands on `main` with the release, which changes nothing for development.
- `release-dep-guard.yml` fails the release PR if any runtime dependency points at a git ref, as a backstop: a published package cannot depend on one (PyPI rejects it, and a moving ref is not reproducible).

If the release PR then fails its build or tests, sentinel needs an inspect_ai change that has not shipped: cut an inspect_ai release first, and the scheduled run raises the floor once it is on PyPI.

## Reporting issues

Use the [issue tracker](https://github.com/meridianlabs-ai/inspect_sentinel/issues). For bugs, include the `inspect_sentinel` and `inspect_ai` versions (`pip show inspect-sentinel inspect-ai`) and a minimal repro.
