# Contributing to Inspect Sentinel

Thanks for your interest in contributing. Bug reports, doc fixes, and core changes are all welcome.

## Start with an issue

Open an issue before you write code, and wait for a maintainer to say on the issue that we want the change. Then open a PR that links the issue (`Fixes #NNN`). This applies whoever filed the issue. Maintainers may open PRs directly, and trivial documentation fixes, such as typos and broken links, are welcome without an issue.

Reviewing a PR takes much longer than agreeing on a direction in an issue. Starting with an issue spends review time on work we have agreed we want, and spares you writing code we can't merge.

## Development setup

Inspect Sentinel uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, and requires Python 3.11+. Install uv 0.13.4 or later, then clone the repository and create the environment:

```bash
git clone https://github.com/meridianlabs-ai/inspect_sentinel
cd inspect_sentinel
uv sync --extra dev
```

`uv sync` creates `.venv` with the exact versions in `uv.lock`, the `dev` extra included, and installs sentinel editable. Run Python commands from it: `.venv/bin/<command>`, `uv run --no-sync <command>`, or an activated venv (`source .venv/bin/activate`). A plain `uv run` also works, but it reinstalls the locked inspect_ai and inspect_scout over a local editable checkout (below). `make check` and `make test` call the tools directly. `uv sync` without `--extra dev` removes the dev tools again.

The development environment pins 3.13 (`.python-version`). CI covers both ends of the supported range: 3.11 proves the floor still works, and 3.14 exercises the newest interpreter. `make check` only ever sees the 3.13 environment, so a construct that is new in 3.12 or later passes locally and fails in CI on 3.11. To reproduce the floor locally, install the same lock into a second environment:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/py311 uv sync --locked --extra dev --python 3.11
.venv/bin/pyright --pythonpath /tmp/py311/bin/python
/tmp/py311/bin/pytest
```

### Dependencies and the cooldown

`uv.lock` pins every Python dependency. CI's lint, type-check, test and build jobs install their dependencies from it, and fail when `uv.lock` is out of date with `pyproject.toml`. When you change dependencies in `pyproject.toml`, run `uv lock` and commit both files.

The build tools (`build`, `hatchling`, `hatch-vcs`) come from `scripts/build-dist/requirements.txt` instead, pinned with hashes, and CI and the publish job build with `python -m build --no-isolation`, so the build installs nothing else. Dependabot updates the file weekly. To update it by hand, run `uv pip compile --no-config --generate-hashes --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --exclude-newer "7 days" --only-binary :all: --upgrade requirements.in -o requirements.txt` in `scripts/build-dist`.

`pyproject.toml` sets `exclude-newer = "7 days"` under `[tool.uv]`, so `uv lock`, `uv sync` and `uv pip` only pick releases that have been on PyPI for at least seven days. A compromised release is usually found and removed within that time, before it reaches `uv.lock`, CI or your machine. A few packages wait 6 hours instead, so sentinel can adopt a new release the day it ships: `inspect-ai`, `inspect-scout`, and the `anthropic` and `openai` SDKs. They are listed with `"6 hours"` in `[tool.uv.exclude-newer-package]`. Only a direct dependency (a package in `dependencies` or an extra in `pyproject.toml`) that is a model provider SDK or an Inspect package gets this exception. Every other package keeps the seven-day cooldown, including the CA bundle (`certifi`). Dependabot's `uv` entry in `.github/dependabot.yml` repeats the cooldown and the list, except `inspect-ai` and `inspect-scout`, which it ignores. When you add a `"6 hours"` entry, also add the package to `cooldown.exclude` and the `six-hour-cooldown` group there. `tests/test_dependency_cooldown.py` checks that each entry is a direct dependency, that dated entries (below) are less than seven days old, and that the two files match. Whether a package is a provider SDK or an Inspect package is up to review.

Dependabot proposes `uv.lock` updates daily, once a release has cleared its cooldown. To adopt a release sooner, run `uv lock --upgrade-package <package>` and commit `uv.lock`. A release still inside its cooldown can't be locked; wait for it rather than overriding `exclude-newer`, unless it's an emergency (below).

In an emergency, for example a security fix, you can adopt a release that is less than seven days old with a one-time dated entry:

1. Under `[tool.uv.exclude-newer-package]` in `pyproject.toml`, add `<package> = "<timestamp>"`. The package must be a direct dependency. Use an RFC 3339 timestamp just after the upload time of the fixed release. Don't add a dated entry to `.github/dependabot.yml`.
2. Run `uv lock --upgrade-package <package>` and commit both files.
3. Within seven days, once the normal cooldown includes the release, delete the entry, run `uv lock --upgrade-package <package>`, and commit both files. A plain `uv lock` doesn't reliably remove the entry from `uv.lock`, and CI then fails. A test fails when a dated entry is seven days old or older.

Testing against the newest release of every dependency happens in the nightly canary in [meridianlabs-ai/canaries](https://github.com/meridianlabs-ai/canaries), so sentinel's own CI doesn't need to.

### inspect_ai and inspect_scout main

Development and CI track inspect_ai's and inspect_scout's `main` branches. Tracking `main` is deliberate: sentinel and inspect_ai co-develop, and most sentinel changes need hooks that have not shipped yet. Scout is tracked the same way, so a Scout change that alters what a monitor's model is shown reaches CI at once. `[tool.uv.sources]` in `pyproject.toml` points `inspect-ai` and `inspect-scout` at their git repositories, and `uv.lock` records a commit of each. Only uv reads `[tool.uv.sources]`: the published package requires the `inspect-ai>=` and `inspect-scout>=` floors in `[project]`.

- CI moves both to the latest `main` commit before every install (`.github/actions/uv-sync`), so it always tests against their `main`. Every other package stays at its locked version.
- `uv sync --extra dev` installs the commits in `uv.lock`. To move to the latest `main` commits, run `uv sync --upgrade-package inspect-ai --upgrade-package inspect-scout --extra dev`. Committing the updated `uv.lock` is optional.
- On the release PR, CI ignores the git sources and installs the newest releases that have cleared their cooldown (below).

To work against a local inspect_ai or inspect_scout checkout, install it over the locked one:

```bash
uv sync --extra dev
uv pip install -e ../inspect_ai
```

Then run commands with `.venv/bin/<command>` or `uv run --no-sync <command>`. A plain `uv run` or `uv sync` reinstalls the locked version.

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
uv sync --extra dev --extra doc
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
