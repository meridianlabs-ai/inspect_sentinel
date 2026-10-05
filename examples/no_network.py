"""A rule: a protocol with no monitors.

It rejects any `bash()` command that uses a network tool, and tells the agent why.
"""

from example_samples import COUNT_ROWS, FETCH_DATA, FIX_SCRIPT
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Protocol,
    call_text,
    find_words,
    protocol,
)

NETWORK_COMMANDS = ["curl", "wget", "pip install"]


@protocol
def no_network() -> Protocol:
    """Reject `bash()` commands that use `curl`, `wget` or `pip install`."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.function != "bash":
            return None
        used = find_words(call_text(step.call), NETWORK_COMMANDS)
        if used:
            return Decision.reject(
                f"command uses {', '.join(used)}",
                message=f"`{used[0]}` needs the network, which this task does not allow.",
            )
        return Decision.proceed()

    return decide


@task
def network_rule() -> Task:
    return Task(
        dataset=[FIX_SCRIPT, FETCH_DATA, COUNT_ROWS],
        solver=react(tools=[bash(timeout=60), python(timeout=60)]),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
        sentinel=no_network(),
    )
