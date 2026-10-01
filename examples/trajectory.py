"""A trajectory monitor: two functions of one monitor, sharing state across the sample.

`after` counts the calls that failed; `before` grows more suspicious with each failure.
"""

from example_samples import COUNT_ROWS, FETCH_DATA, FIX_SCRIPT
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python
from inspect_ai.util import StoreModel

from inspect_sentinel import (
    AfterToolCall,
    BeforeToolCall,
    Context,
    MonitorGroup,
    Observation,
    monitor,
    observe,
)


class Failures(StoreModel):
    count: int = 0


@monitor
def failure_count(per_failure: float = 0.2) -> MonitorGroup:
    """Suspicion that rises by `per_failure` with each failed call so far."""

    async def before(context: Context, step: BeforeToolCall) -> Observation:
        failures = context.store_as(Failures).count
        return Observation.score(
            min(1.0, per_failure * failures), f"{failures} earlier calls failed"
        )

    async def after(context: Context, step: AfterToolCall) -> Observation:
        failed = step.result.error is not None or "Traceback" in step.result.text
        if failed:
            context.store_as(Failures).count += 1
        return Observation.score(0.0, "call failed" if failed else "call succeeded")

    return MonitorGroup(before, after)


@task
def trajectory() -> Task:
    return Task(
        dataset=[FIX_SCRIPT, FETCH_DATA, COUNT_ROWS],
        solver=react(tools=[bash(timeout=60), python(timeout=60)]),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
        sentinel=observe(failure_count()),
    )
