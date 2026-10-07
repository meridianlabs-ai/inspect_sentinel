.PHONY: typecheck
typecheck:
	pyright

.PHONY: check
check: typecheck
	ruff check --fix
	ruff format

.PHONY: test
test:
	pytest

.PHONY: docs
docs:
	cd docs && PATH="$(CURDIR)/.venv/bin:$$PATH" ../.venv/bin/quarto render
