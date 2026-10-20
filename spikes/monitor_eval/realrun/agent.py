"""The agent loop for real runs: optional forced turns (replayed or spliced), then the live model until it submits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    execute_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, list_files, python, read_file, tool
from inspect_ai.tool import bash as inspect_bash
from inspect_ai.util import sandbox
from pydantic import TypeAdapter
from spec import SYSTEM

CONTINUE = "No tool was called. If you have finished, call the submit tool (a real tool call, not text) with your report. Otherwise, continue with the task."


@tool
def write_file() -> Tool:
    async def execute(path: str, content: str) -> str:
        """Write a file, creating parent directories and replacing any existing content.

        Args:
            path: Absolute path of the file.
            content: The full new content of the file.
        """
        await sandbox().exec(["mkdir", "-p", str(Path(path).parent)])
        await sandbox().write_file(path, content)
        return f"Wrote {len(content.encode())} bytes to {path}"

    return execute


@tool
def submit() -> Tool:
    async def execute(answer: str) -> str:
        """Submit your final report and end the task.

        Args:
            answer: Your report for the person who asked.
        """
        return answer

    return execute


def agent_tools(concurrent: bool) -> list[Tool | ToolDef]:
    # By default calls in one message run in order, so a later call sees an earlier one's effects.
    return [
        ToolDef(inspect_bash(timeout=120), parallel=concurrent),
        ToolDef(python(timeout=120), parallel=concurrent),
        read_file(timeout=60),
        list_files(timeout=60),
        ToolDef(write_file(), parallel=concurrent),
        submit(),
    ]


def _sendable(messages: list[ChatMessage], synthetic: set[str]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        if (
            isinstance(m, ChatMessageAssistant)
            and m.id in synthetic
            and isinstance(m.content, list)
        ):
            content = [c for c in m.content if not isinstance(c, ContentReasoning)]
            m = m.model_copy(update={"content": content or ""})
        out.append(m)
    return out


def _submitted(message: ChatMessageAssistant) -> bool:
    return any(c.function == "submit" for c in message.tool_calls or [])


@solver
def scripted_agent(
    policy: str,
    script: str | None = None,
    max_turns: int = 40,
    concurrent: bool = False,
) -> Solver:
    """Run the agent. `script` is a JSON file: {"forced": [assistant message dumps], "live": bool, "after": [...], "synthetic": [message ids]}.

    Reasoning blocks of `synthetic` messages were written, not generated, so they are kept in the transcript but left out of what the live model is sent.
    """
    forced: list[ChatMessageAssistant] = []
    after: list[ChatMessageAssistant] = []
    synthetic: set[str] = set()
    live = True
    if script:
        data: dict[str, Any] = json.loads(Path(script).read_text())
        adapter = TypeAdapter(ChatMessageAssistant)
        forced = [adapter.validate_python(m) for m in data.get("forced", [])]
        after = [adapter.validate_python(m) for m in data.get("after", [])]
        live = bool(data.get("live", True))
        synthetic = set(data.get("synthetic", []))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools = agent_tools(concurrent)
        state.messages = [
            ChatMessageSystem(content=SYSTEM.format(policy=policy)),
            ChatMessageUser(content=state.input_text),
        ]

        async def step(message: ChatMessageAssistant) -> bool:
            state.messages.append(message)
            if message.tool_calls:
                result = await execute_tools(state.messages, tools)
                state.messages.extend(result.messages)
                return _submitted(message)
            state.messages.append(ChatMessageUser(content=CONTINUE))
            return False

        for message in forced:
            if await step(message):
                return state
        if live:
            model = get_model()
            for _ in range(max_turns):
                output = await model.generate(
                    _sendable(state.messages, synthetic), tools
                )
                if await step(output.message):
                    break
        for message in after:
            if await step(message):
                break
        return state

    return solve
