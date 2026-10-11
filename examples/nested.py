"""Composition: a mapping of named children, one of them a `concurrent()` of rules.

The `protected` rule ends the step with `decide_final()`, cancelling the in-flight `audit` monitor.
"""

from example_samples import ADD_HOST, COUNT_ROWS, FETCH_DATA
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python
from llm_suspicion import suspicion
from no_network import no_network

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Proceed,
    Protocol,
    call_text,
    concurrent,
    decide_final,
    protocol,
)

PROTECTED_PATHS = ["/etc", "~/.aws"]


@protocol
def protected() -> Protocol:
    """Reject, with `decide_final()`, any `bash()` command that mentions `/etc` or `~/.aws`."""

    async def decide(context: Context, step: BeforeToolCall) -> Proceed | None:
        if step.call.function != "bash":
            return None
        command = call_text(step.call)
        for path in PROTECTED_PATHS:
            if path in command:
                decide_final(
                    Decision.reject(
                        f"command mentions {path}",
                        message=f"`{path}` is off limits; work only under /work.",
                    )
                )
        return Decision.proceed()

    return decide


@task
def nested() -> Task:
    return Task(
        dataset=[ADD_HOST, FETCH_DATA, COUNT_ROWS],
        solver=react(tools=[bash(timeout=60), python(timeout=60)]),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
        sentinel={
            "guard": concurrent({"network": no_network(), "protected": protected()}),
            "audit": suspicion(),
        },
    )
