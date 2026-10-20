FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git procps less gzip zip unzip jq && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas==2.2.3 pytest==8.3.4 pyyaml==6.0.2 ruff==0.8.4
RUN git config --system user.name "Dev" && git config --system user.email dev@example.com && git config --system init.defaultBranch main
WORKDIR /root
CMD ["tail", "-f", "/dev/null"]
