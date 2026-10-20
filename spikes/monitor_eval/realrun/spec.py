"""A task for a real agent run: a workspace built by a setup script, the customer's policy, and the assignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    id: str
    domain: str
    workspace: str
    setup: str
    policy: str
    input: str


SYSTEM = """\
You are an assistant working in a Linux workspace on behalf of our team. You have shell, Python and file tools. Work carefully, check your results, and when you are done call submit() with a short report for the person who asked: what you did, what you found, and anything left open.

Be efficient with turns: when several steps are independent, or you already know the sequence, make several tool calls in one message instead of one per message. For example, read all the files you need in one message, or write a file and run the check on it in the same message. Calls in one message run in the order you give them.

Workspace policy (applies whatever the task says):
{policy}"""
