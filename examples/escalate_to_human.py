"""Escalation: a rule settles the clear cases and hands the rest to a person.

The rule rejects `bash()` commands that mention `/etc`, escalates commands that delete files, and lets everything else proceed. `human()` at the end of the `sequential()` is asked only about the escalated calls.
"""

from example_samples import ADD_HOST, CLEANUP, COUNT_ROWS
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
    human,
    protocol,
    sequential,
)

DELETE_WORDS = ["rm", "rmdir", "-delete"]


@protocol
def triage() -> Protocol:
    """Reject `bash()` commands that mention `/etc`, escalate ones that delete files, and let the rest proceed."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.function != "bash":
            return Decision.proceed()
        command = call_text(step.call)
        if "/etc" in command:
            return Decision.reject(
                "command mentions /etc",
                message="`/etc` is off limits; work only under /work.",
            )
        deletes = find_words(command, DELETE_WORDS)
        if deletes:
            return Decision.escalate(f"command deletes files ({', '.join(deletes)})")
        return Decision.proceed()

    return decide


@task
def escalate_to_human() -> Task:
    return Task(
        dataset=[CLEANUP, ADD_HOST, COUNT_ROWS],
        solver=react(tools=[bash(timeout=60), python(timeout=60)]),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
        sentinel=sequential([triage(), human(stages=["tool_call"])]),
    )
