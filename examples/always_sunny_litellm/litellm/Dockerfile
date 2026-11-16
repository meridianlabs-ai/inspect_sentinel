# LiteLLM's proxy, with inspect_sentinel's guardrail for it installed.
# Built from this example's folder, and from nothing outside it.
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM python:3.13.18-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# uv.lock names the guardrail by a commit of inspect_sentinel's repository,
# which uv fetches
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# the image's own Python, and never one that uv downloads
ENV UV_PYTHON_DOWNLOADS=never
# where uv puts what it installs
ENV UV_PROJECT_ENVIRONMENT=/opt/litellm

# Exactly the versions and hashes in uv.lock: uv resolves no versions here, and
# the build fails if uv.lock doesn't agree with pyproject.toml. A LiteLLM image
# of your own would get the guardrail the same way, or from a release of it.
WORKDIR /example/litellm
COPY litellm/pyproject.toml litellm/uv.lock ./
RUN uv sync --locked
ENV PATH=/opt/litellm/bin:$PATH

EXPOSE 4000
ENTRYPOINT ["litellm", "--host", "0.0.0.0", "--port", "4000", "--config", "/config/litellm.yaml"]
