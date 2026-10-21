# AGENTS.md

## Principles

- Shortcuts become someone else's problem; hacks compound into debt
- Patterns get copied—establish good ones
- Flag issues; ask before fixing
- Strict typing required
  - Python: modern syntax (`X | None`, `dict[str, Any]`)
- Error handling: do not use Exceptions gratuitously, unless an error is expected in a context and needs to be computed on it should be allowed to propagate.
- Never swallow an exception from a monitor, a protocol, or the recorder, and never log-and-continue in their place. A raising protocol or recorder fails the layer. A raising monitor is recorded as failed by the runner, and reading the scores of a run with a failed monitor raises; that failure policy (design/sentinel.md, "Failure semantics") is the one sanctioned place a failure is tolerated, and only where a protocol handles `observations.failed` explicitly.
- Respect existing patterns
- Before committing, run `make check` and `make test` and report results honestly

### Testing
- Test observable behavior, not internal implementation details
- Do not test things that are enforced by the type system
- Test through the narrowest public API that covers the behavior
- Be efficient; avoid duplicate coverage
- Prefer data/table driven tests for maintainability
- Tests must be isolated; no shared mutable state or order dependencies
- Tests must be deterministic; control randomness with seeds
- Prefer real objects over mocks when possible
- Async tests: write `async def test_...` and mark it `@pytest.mark.anyio`. Use `anyio.sleep()` and `anyio.Event()`, never the `asyncio` equivalents. The `anyio_backend` fixture in `tests/conftest.py` runs every async test on asyncio and trio.

### Common Pitfalls
- Stay within scope—don't make unrequested changes
- During development, run only implicated tests; run the full suite when the work is complete

## Pull Requests

- Unless the account you operate for is a maintainer, open a PR only for an issue where a maintainer has said they want the change, and link it (`Fixes #NNN`). Without an accepted issue, file your evidence on an issue and stop. Trivial documentation fixes (typos, broken links) are exempt
- Title PRs as Conventional Commits (`<type>: <description>`)—we squash-merge, so the PR title becomes the commit message that drives releases; `pr-title-lint` enforces it
- `feat:`/`fix:` are for user-facing changes only: they headline the release notes and bump the version. `perf:`/`revert:` also appear in the notes (no bump); `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` are hidden
- Body lines starting with `<type>:` are parsed as extra changelog entries—don't begin description lines with a conventional-commit prefix unless that's intended
- Never edit `CHANGELOG.md`, version numbers, or `.release-please-manifest.json`—Release Please owns them
- Keep the `inspect-ai` and `inspect-scout` dependencies in `pyproject.toml` PyPI floors, never git references—development installs their `main` over the floors (CONTRIBUTING.md), and `release-pin-deps.yml` raises the floors on the release PR
- Before opening a non-trivial PR, run at least one code review pass in a fresh context (e.g. `/code-review` or a subagent that hasn't seen the authoring conversation) using a frontier-class model — the authoring context is blind to its own assumptions, and in our experience reviews from small fast-tier models rarely surface real issues. Fix or explicitly dismiss each finding before opening the PR, and disclose the pass in the description (see "Agent review disclosure" below).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines

### Agent review disclosure

PRs prepared by AI coding agents must disclose agent involvement in the description, including an `### Agent review` section summarizing pre-PR review passes: what model/tool reviewed, whether the review ran in a fresh context and/or used a different model from the author, how many passes, and the findings—issues found, which were fixed, and which were dismissed with a one-line reason each. Multiple passes, each in a fresh context, often catch issues a single pass misses—prefer that for non-trivial changes. If no review pass was run, say so explicitly. Never report a review that didn't happen—a fabricated or content-free claim ("reviewed, looks good") is worse than disclosing none. Example:

```
### Agent review
- Reviewer: Claude Opus 5 via /code-review (fresh context), 2 passes
- Findings: 3 — 2 fixed, 1 dismissed (flagged a missing None check that is
  guarded upstream)
```

## Design

`design/` holds the design documents this package implements. Start with `design/sentinel-overview.md`; `design/sentinel.md` is the full design and `design/sentinel-reference.md` its reference form. Read the relevant section before adding or changing public API, and when an implementation decision departs from the design, update the design document in the same change.

## Python

Directory: `src/inspect_sentinel/`

Supports Python 3.11+ (the same floor as `inspect-ai`, which is intended to depend on this package). Do not use syntax or stdlib features newer than 3.11.

### Scripts
| Command | Description |
|---------|-------------|
| `make check` | Run all checks (lint, format, typecheck) |
| `make typecheck` | Run typechecking only |
| `make test` | Run all tests |
| `make docs` | Render the Quarto docs to `docs/_site/` |
| `pytest` | Run all tests |
| `pytest tests/path/to/test.py::test_name -v` | Run single test |
| `ruff format` | Format code |
| `ruff check --fix` | Lint and auto-fix |

### Style
- **Formatting**: Follow Google style convention. Use ruff for formatting
- **Imports**: Use isort order (enforced by ruff)
- **Types**: All functions must have type annotations, including in tests.
- **Naming**: Use snake_case for variables, functions, methods; PascalCase for classes
- **Docstrings**: Google-style docstrings required for public APIs; none on private functions, private module-level names, or the `_`-prefixed modules themselves (the inspect_ai and inspect_scout pattern). Use single backtick (`) around symbols. Do not use hard line breaks within paragraphs or list items — each paragraph should be a single unwrapped line, separated by blank lines.
- **Error Handling**: Use appropriate exception types; include context in error messages
- **Testing**: Write tests with pytest; maintain high coverage

### Common Pitfalls
- Use the venv for all Python commands: either reference `.venv/bin/` directly or run `source .venv/bin/activate`
- **Never use `uv sync` or `uv run`**—there is no `uv.lock`, so they lock and sync the project and silently replace inspect_ai's and inspect_scout's `main` (or a local editable checkout) with a PyPI release. Set up with `uv pip install git+https://github.com/UKGovernmentBEIS/inspect_ai.git git+https://github.com/meridianlabs-ai/inspect_scout.git` then `uv pip install -e ".[dev]"`; refresh them with `uv pip install --upgrade git+https://github.com/UKGovernmentBEIS/inspect_ai.git git+https://github.com/meridianlabs-ai/inspect_scout.git`

## Reviewing other people's pull requests

Unless a maintainer asked for your review, submit any review on someone else's pull request as a **Comment**, never **Approve** or **Request changes**. In this repository those are maintainer decisions: a non-maintainer's verdict doesn't count toward merging and misleads the author about where their PR stands. Put any confirmation or concern in the comment text, and say that you are an AI agent and not a maintainer. This does not apply to agents acting for project maintainers.
