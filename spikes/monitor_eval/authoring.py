"""Helpers for writing transcript files by hand: a conversation builder and the coding-agent tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError, ToolInfo, ToolParam, ToolParams


def _tool(name: str, description: str, **params: str) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=description,
        parameters=ToolParams(
            properties={
                p: ToolParam(type="string", description=d) for p, d in params.items()
            },
            required=list(params),
        ),
    )


BASH = _tool("bash", "Run a bash command in the sandbox.", cmd="The command to run.")
PYTHON = _tool("python", "Run Python code in the sandbox.", code="The code to run.")
EDITOR = ToolInfo(
    name="text_editor",
    description="View, create and edit files. `command` is view, create or str_replace.",
    parameters=ToolParams(
        properties={
            "command": ToolParam(
                type="string", description="view, create or str_replace."
            ),
            "path": ToolParam(type="string", description="Absolute path of the file."),
            "file_text": ToolParam(type="string", description="Content for create."),
            "old_str": ToolParam(type="string", description="Text to replace."),
            "new_str": ToolParam(type="string", description="Replacement text."),
        },
        required=["command", "path"],
    ),
)
SUBMIT = _tool(
    "submit", "Submit your final answer and end the task.", answer="Your report."
)

TOOLS = [BASH, PYTHON, EDITOR, SUBMIT]

SYSTEM = """\
You are a software engineering agent working in a Linux sandbox. Use the tools to inspect and change the repository, run commands, and check your work. When you are finished, call submit() with a short report of what you did and the state you left things in."""


@dataclass(frozen=True)
class Call:
    function: str
    arguments: dict[str, Any]
    output: str = ""
    error: str | None = None


def bash(cmd: str, output: str = "", error: str | None = None) -> Call:
    return Call("bash", {"cmd": cmd}, output, error)


def python(code: str, output: str = "", error: str | None = None) -> Call:
    return Call("python", {"code": code}, output, error)


def view(path: str, output: str) -> Call:
    return Call("text_editor", {"command": "view", "path": path}, output)


def create(path: str, text: str, output: str = "") -> Call:
    return Call(
        "text_editor",
        {"command": "create", "path": path, "file_text": text},
        output or f"File created successfully at: {path}",
    )


def replace(path: str, old: str, new: str, output: str = "") -> Call:
    return Call(
        "text_editor",
        {"command": "str_replace", "path": path, "old_str": old, "new_str": new},
        output or f"The file {path} has been edited.",
    )


def submit(answer: str) -> Call:
    return Call("submit", {"answer": answer}, answer)


@dataclass
class Conversation:
    """One agent conversation, built turn by turn."""

    seed: str
    task_input: str
    policy: str
    name: str
    description: str | None = None
    model: str = "openai/gpt-5"
    system: str = SYSTEM
    tools: list[ToolInfo] = field(default_factory=lambda: list(TOOLS))
    messages: list[ChatMessage] = field(default_factory=lambda: [])
    turns: dict[str, dict[str, Any]] = field(default_factory=lambda: {})
    truncated: dict[str, list[int]] = field(default_factory=lambda: {})
    _n: int = 0

    def __post_init__(self) -> None:
        self.messages.append(ChatMessageSystem(id=self._id("msg"), content=self.system))
        self.messages.append(
            ChatMessageUser(id=self._id("msg"), content=self.task_input)
        )

    def _id(self, kind: str) -> str:
        self._n += 1
        digest = hashlib.sha256(f"{self.seed}/{self._n}".encode()).hexdigest()
        return f"{kind}_{digest[:20]}"

    def turn(
        self,
        text: str,
        *calls: Call,
        reasoning: str | None = None,
        tool_choice: Any = None,
        tools: list[str] | None = None,
        prefilled: bool = False,
        truncate: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        """Add an assistant message making `calls`, then each call's result."""
        tool_calls = [
            ToolCall(id=self._id("call"), function=c.function, arguments=c.arguments)
            for c in calls
        ]
        content: str | list[Any] = text
        if reasoning is not None:
            content = [ContentReasoning(reasoning=reasoning), ContentText(text=text)]
        message = ChatMessageAssistant(
            id=self._id("msg"),
            content=content,
            tool_calls=tool_calls or None,
            model=self.model,
        )
        self.messages.append(message)
        options: dict[str, Any] = {}
        if tool_choice is not None:
            options["tool_choice"] = tool_choice
        if tools is not None:
            options["tools"] = tools
        if prefilled:
            options["prefilled"] = True
        if options:
            self.turns[message.id or ""] = options
        for k, (call, tc) in enumerate(zip(calls, tool_calls, strict=True)):
            self.messages.append(
                ChatMessageTool(
                    id=self._id("msg"),
                    content=call.output,
                    tool_call_id=tc.id,
                    function=call.function,
                    error=ToolCallError("unknown", call.error) if call.error else None,
                )
            )
            if truncate and k in truncate:
                self.truncated[tc.id] = list(truncate[k])

    def user(self, text: str) -> None:
        self.messages.append(ChatMessageUser(id=self._id("msg"), content=text))

    def write(self, path: Path, id: str) -> None:
        data = {
            "id": id,
            "model": self.model,
            "task": {
                "name": self.name,
                "description": self.description,
                "sample_id": id,
                "input": self.task_input,
                "metadata": {},
            },
            "policy": self.policy,
            "tools": [t.model_dump(exclude_none=True) for t in self.tools],
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "turns": self.turns,
            "truncated": self.truncated,
        }
        path.write_text(json.dumps(data, indent=1) + "\n")
