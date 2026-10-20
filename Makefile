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

# The LiteLLM guardrail is a package of its own, tested and checked in an
# environment of its own with one LiteLLM version, which this repository's
# environment never has. Its tests start a real sidecar from this repository's
# environment, `.venv`, so set that up first (CONTRIBUTING.md). `uv run
# --isolated --no-project` never reads or installs this repository's project.
LITELLM ?= 1.105.0
LITELLM_ENV = cd src/proxies/litellm && uv run --isolated --no-project \
	--with "litellm[proxy]==$(LITELLM)" --with-editable . \
	--with pytest --with pytest-timeout --with jsonschema --with httpx \
	--with anthropic

.PHONY: test-litellm
test-litellm:
	$(LITELLM_ENV) pytest

.PHONY: check-litellm
check-litellm:
	$(LITELLM_ENV) --with pyright pyright

# The Envoy shim's gRPC code, generated from Envoy's protos into
# src/inspect_sentinel/sidecar/envoy/_proto. The versions of grpcio-tools and
# its protoc are the floors of the `envoy` extra: generated code needs a
# runtime at least as new as what generated it.
.PHONY: envoy-protos
envoy-protos:
	uv run --python 3.13 --isolated --no-project \
		--with grpcio-tools==1.74.0 --with mypy-protobuf==5.1.0 \
		python scripts/generate_envoy_protos.py

.PHONY: docs
docs:
	cd docs && PATH="$(CURDIR)/.venv/bin:$$PATH" ../.venv/bin/quarto render
