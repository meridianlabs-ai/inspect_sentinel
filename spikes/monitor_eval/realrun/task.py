"""The inspect_ai task for real runs: `inspect eval task.py -T set=c -T only=c-sales --model ...`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import scripted_agent  # noqa: E402
from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import Sample  # noqa: E402
from registry import task_spec  # noqa: E402

COMPOSE = str(Path(__file__).parent / "sandbox" / "compose.yaml")


@task
def realrun(
    only: str,
    script: str | None = None,
    max_turns: int = 40,
    concurrent: bool = False,
) -> Task:
    spec = task_spec(only)
    return Task(
        dataset=[
            Sample(
                id=spec.id,
                input=spec.input,
                setup=spec.setup,
                metadata={"domain": spec.domain, "workspace": spec.workspace},
            )
        ],
        solver=scripted_agent(
            spec.policy, script=script, max_turns=max_turns, concurrent=concurrent
        ),
        sandbox=("docker", COMPOSE),
        name=spec.id,
    )
