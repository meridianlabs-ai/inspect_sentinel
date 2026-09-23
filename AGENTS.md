# AGENTS.md

## Principles

- Shortcuts become someone else's problem; hacks compound into debt
- Patterns get copied—establish good ones
- Flag issues; ask before fixing
- Strict typing required
  - Python: modern syntax (`X | None`, `dict[str, Any]`)
- Error handling: do not use Exceptions gratuitously, unless an error is expected in a context and needs to be computed on it should be allowed to propagate.
- Respect existing patterns
- Before committing, run the appropriate checks for code you touched (lint, typecheck, test)

### Testing
- Test observable behavior, not internal implementation details
- Do not test things that are enforced by the type system
- Test through the narrowest public API that covers the behavior
- Be efficient; avoid duplicate coverage
- Prefer data/table driven tests for maintainability
- Tests must be isolated; no shared mutable state or order dependencies
- Tests must be deterministic; control randomness with seeds
- Prefer real objects over mocks when possible
- Async tests: write `async def test_...` and mark it `@pytest.mark.anyio`. Use `anyio.sleep()` and `anyio.Event()`, never the `asyncio` equivalents. The `anyio_backend` fixture in `tests/conftest.py` pins the backend to asyncio.

### Common Pitfalls
- Stay within scope—don't make unrequested changes
- During development, run only implicated tests; run the full suite when the work is complete

## Pull Requests

- Title PRs as Conventional Commits (`<type>: <description>`)—we squash-merge, so the PR title becomes the commit message that drives releases; `pr-title-lint` enforces it
- `feat:`/`fix:` are for user-facing changes only: they headline the release notes and bump the version. `perf:`/`revert:` also appear in the notes (no bump); `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` are hidden
- Body lines starting with `<type>:` are parsed as extra changelog entries—don't begin description lines with a conventional-commit prefix unless that's intended
- Never edit `CHANGELOG.md`, version numbers, or `.release-please-manifest.json`—Release Please owns them
- Never replace the `inspect-ai` git reference in `pyproject.toml` with a version floor—`release-pin-deps.yml` does that on the release PR
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines

## Design

`design/` holds the design documents this package implements. Start with `design/sentinel-overview.md`; `design/sentinel.md` is the full design and `design/sentinel-reference.md` its reference form. Read the relevant section before adding or changing public API, and when an implementation decision departs from the design, update the design document in the same change.

## Python

Directory: `src/inspect_sentinel/`

Supports Python 3.10+ (the same floor as `inspect-ai`, which is intended to depend on this package). Do not use syntax or stdlib features newer than 3.10.

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
