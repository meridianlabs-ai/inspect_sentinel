# The sidecar, with what it needs to read Anthropic's and OpenAI's traffic, and
# the example's rule.
# Built from this example's folder, and from nothing outside it.
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM python:3.13.18-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

COPY --from=uv /uv /usr/local/bin/uv

# the image's own Python, and never one that uv downloads
ENV UV_PYTHON_DOWNLOADS=never
# where uv puts what it installs
ENV UV_PROJECT_ENVIRONMENT=/opt/sidecar

# The image keeps this example's folders as they are in the repository, since
# pyproject.toml names the rule by the folder beside it, ../rule.
WORKDIR /example/sidecar

# Exactly the versions and hashes in uv.lock: uv resolves no versions here, and
# the build fails if uv.lock doesn't agree with pyproject.toml. First everything
# but the rule, so that a change to the rule doesn't install the rest again.
COPY sidecar/pyproject.toml sidecar/uv.lock ./
COPY rule/pyproject.toml /example/rule/pyproject.toml
RUN uv sync --locked --no-install-package weather-rules

# the example's rule, which the sentinel names as weather_rules/always_sunny
COPY rule /example/rule
RUN uv sync --locked
ENV PATH=/opt/sidecar/bin:$PATH

EXPOSE 8900
ENTRYPOINT ["python", "-m", "inspect_sentinel.sidecar", "--host", "0.0.0.0", "--port", "8900"]
