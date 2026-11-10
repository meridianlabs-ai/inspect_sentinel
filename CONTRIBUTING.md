# Contributing to Inspect Sentinel

Thanks for your interest in contributing. Bug reports, doc fixes, and core changes are all welcome.

<!-- BEGIN shared text from meridianlabs-ai/actions policy/CONTRIBUTING.md. Change it there, not here. -->
## How contributions work

There are two paths:

- **Qualified contributors**: maintainers and individuals recorded (by
  account id) in [`.github/qualified.yml`](.github/qualified.yml) — may open
  PRs directly.

- **Everyone else**: start from an issue a maintainer has labeled `accepted`
  before writing code, and reference it from your PR (`Fixes #NNN`). This
  applies regardless of your contribution history here, and regardless of
  who filed the issue — what matters is that a maintainer has accepted it.
  PRs that aren't linked to an accepted issue may be closed; if a linked
  issue is later accepted, reopen the PR.
  **Exception:** trivial documentation fixes (typos, broken links — docs
  files only, under 25 changed lines) are always welcome directly.

Whatever your path, check for an existing open PR addressing the same
problem before opening yours: duplicates are closed in favor of the earlier
PR unless a maintainer has asked for an alternative.

Issues a maintainer has labeled `deferred` are decisions not to prioritize
that work right now. PRs against a deferred issue are closed —
comment on the issue with new evidence or demand if you think it should be
revisited.

**Why:** like most open-source projects, we now receive a large volume of
unrequested, often agent-generated PRs. Reviewing a PR is time-consuming;
agreeing on a direction in an issue first is much easier. This policy spends
our review time on work we've agreed we want and spares you writing code we
can't merge.
<!-- END shared text from meridianlabs-ai/actions policy/CONTRIBUTING.md -->

## Development setup

Inspect Sentinel uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, and requires Python 3.11+. Install uv 0.13.4 or later, then clone the repository and create the environment:

```bash
git clone https://github.com/meridianlabs-ai/inspect_sentinel
cd inspect_sentinel
uv sync --locked --extra dev
```

`uv sync --locked` creates `.venv` with the exact versions in `uv.lock`, the `dev` extra included, and installs sentinel editable. It fails if `uv.lock` is out of date with `pyproject.toml`. Run Python commands from the environment: `.venv/bin/<command>`, `uv run --no-sync <command>`, or an activated venv (`source .venv/bin/activate`). A plain `uv run` also works, but it reinstalls the locked inspect_ai and inspect_scout over a local editable checkout (below). `make check` and `make test` call the tools directly. `uv sync --locked` without `--extra dev` removes the dev tools again.

The development environment pins 3.13 (`.python-version`). CI covers both ends of the supported range: 3.11 proves the floor still works, and 3.14 exercises the newest interpreter. `make check` only ever sees the 3.13 environment, so a construct that is new in 3.12 or later passes locally and fails in CI on 3.11. To reproduce the floor locally, install the same lock into a second environment:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/py311 uv sync --locked --extra dev --python 3.11
.venv/bin/pyright --pythonpath /tmp/py311/bin/python
/tmp/py311/bin/pytest
```

### Dependencies

[DEPENDENCIES.md](https://github.com/meridianlabs-ai/actions/blob/main/DEPENDENCIES.md) in meridianlabs-ai/actions tells how the Inspect repositories install and change dependencies. It covers the lock file, the cooldown, the canaries, inspect_ai `main`, the release build tools, and the procedures to add or update a dependency. Follow it. It is the same for every Inspect repository, so this file does not repeat it.

For this repository:

- inspect_sentinel needs unreleased inspect_ai and inspect_scout changes. `[tool.uv.sources]` in `pyproject.toml` points `inspect-ai` and `inspect-scout` at their `main` branches. Tracking `main` is deliberate: sentinel and inspect_ai co-develop, and most sentinel changes need hooks that have not shipped yet. Scout is tracked the same way, so a Scout change that alters what a monitor's model is shown reaches CI at once.
- The 6-hour packages are in `[tool.uv.exclude-newer-package]` in `pyproject.toml`.
- The procedure for a local inspect_ai checkout is also the procedure for a local inspect_scout checkout.

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

Docs live in `docs/` and are built with [Quarto](https://quarto.org). The Quarto CLI and the doc-build dependencies are in the `doc` extra. `dev` leaves it out, so CI doesn't install them. `make docs` installs the `doc` extra and renders the docs to `docs/_site/`. To run Quarto yourself:

```bash
uv sync --locked --extra dev --extra doc
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

### The inspect_ai and inspect_scout pins at release time

`pyproject.toml` names inspect_ai and inspect_scout by PyPI floors, and every CI build installs their `main` through `[tool.uv.sources]`, except on the Release Please release PR. There, CI ignores the git sources (`UV_NO_SOURCES=1`) and installs the newest inspect-ai and inspect-scout releases that have cleared their cooldown, the way a published sentinel is installed. Two workflows keep its floors honest:

- `release-pin-deps.yml` raises the `inspect-ai` and `inspect-scout` floors to those releases and pushes them to the release PR. It runs on release-PR events and every two hours, because an inspect_ai or inspect_scout release produces no event in this repository. The raised floors land on `main` with the release. Main keeps the git sources, so nothing has to be restored after a release.
- `release-dep-guard.yml` fails the release PR if any runtime dependency points at a git ref, as a backstop: a published package cannot depend on one (PyPI rejects it, and a moving ref is not reproducible).

If the release PR then fails its build or tests, sentinel needs an inspect_ai or inspect_scout change that has not shipped: cut a release of that package first, and the scheduled run raises the floor once it is on PyPI and has cleared its cooldown.

## Reporting issues

Use the [issue tracker](https://github.com/meridianlabs-ai/inspect_sentinel/issues). For bugs, include the `inspect_sentinel`, `inspect_ai` and `inspect_scout` versions (`pip show inspect-sentinel inspect-ai inspect-scout`) and a minimal repro.
