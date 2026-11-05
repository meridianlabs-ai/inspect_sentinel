# Jobs that hold credentials restore no cache. A malicious package installed by
# any job can write to the shared Actions cache, so a job holding secrets, an
# id-token or write permissions, or one whose output is published, must not
# restore one.
#
# Jobs that install packages with no cooldown restore no cache either. Every
# install in a job uses the job's one setup-uv, so all of them say none.

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
CREDENTIAL_JOBS = [
    ("publish.yaml", "publish"),
    ("publish.yaml", "notify-approval"),
    # Runs `uv lock`, which may build sdists, and hands push-floor versions.
    ("release-pin-deps.yml", "floor"),
    ("release-pin-deps.yml", "push-floor"),
]
NO_COOLDOWN_JOBS = [
    ("build.yaml", "py-build"),
    ("build.yaml", "litellm-guardrail"),
]


def job_steps(workflow: str, job: str) -> list[dict[str, Any]]:
    config = yaml.safe_load((WORKFLOWS / workflow).read_text())
    steps: list[dict[str, Any]] = config["jobs"][job]["steps"]
    return steps


@pytest.mark.parametrize("workflow,job", CREDENTIAL_JOBS + NO_COOLDOWN_JOBS)
def test_jobs_restore_no_cache(workflow: str, job: str) -> None:
    for step in job_steps(workflow, job):
        action = step.get("uses", "").split("@")[0]
        inputs: dict[str, Any] = step.get("with") or {}
        assert action not in {"actions/cache", "actions/cache/restore"}
        if action == "astral-sh/setup-uv":
            assert inputs.get("enable-cache") is False
        if action in {
            "meridianlabs-ai/actions/setup-uv",
            "meridianlabs-ai/actions/install-python-dependencies",
            "./.github/actions/uv-sync",
        }:
            assert inputs.get("actions-cache") == "none"
        if action == "actions/setup-python":
            assert "cache" not in inputs
