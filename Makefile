.PHONY: typecheck
typecheck:
	uv run pyright

.PHONY: check
check: typecheck
	uv run ruff check --fix
	uv run ruff format

.PHONY: test
test:
	uv run pytest

.PHONY: docs
docs:
	cd docs && PATH="$(CURDIR)/.venv/bin:$$PATH" ../.venv/bin/quarto render
