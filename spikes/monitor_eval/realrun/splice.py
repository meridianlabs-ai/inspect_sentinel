"""Splice new turns into a copy of a real run, so that every spliced action really executes in a fresh sandbox.

A splice is a script for `agent.scripted_agent`: assistant messages to force in order (the base run's own turns, replayed, plus new or edited ones), then optionally the live model, then optionally more forced turns. Each forced message's tool calls execute for real, so tool results always match the actions, and later results reflect the state an earlier spliced call left behind, also within one message.

New messages follow the base model's texture: the same message and tool call id styles, `source="generate"`, the model name, `view` blocks on bash and python calls, and for a model whose turns carry reasoning, a reasoning block with a written summary. Such a reasoning block cannot be replayed to that provider's live API, so list the message in `Splice.synthetic`; the agent then leaves its reasoning out of what the live model is sent.
"""

from __future__ import annotations

import copy
import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, read_eval_log

Message = dict[str, Any]

_SHORTUUID = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B62 = string.ascii_letters + string.digits


@dataclass
class Turn:
    """One assistant message of the base run and the tool results that followed it."""

    index: int
    """1-based, as in step ids (`t03`)."""

    message: Message
    results: list[Message]

    @property
    def calls(self) -> list[dict[str, Any]]:
        return list(self.message.get("tool_calls") or [])

    def text(self) -> str:
        content = self.message.get("content")
        if isinstance(content, str):
            return content
        return "".join(
            c.get("text", "") for c in content or [] if c.get("type") == "text"
        )


class Base:
    """A real run's eval log, read for splicing."""

    def __init__(self, log_path: str | Path, seed: str | None = None) -> None:
        self.path = Path(log_path)
        log = read_eval_log(str(self.path))
        assert log.samples
        self.sample: EvalSample = log.samples[0]
        self.task_id = str(self.sample.id)
        self.model = log.eval.model
        self.messages: list[Message] = [
            m.model_dump(exclude_none=True) for m in self.sample.messages
        ]
        self._rng = random.Random(seed or self.path.name)
        self.turns: list[Turn] = []
        for i, m in enumerate(self.messages):
            if m["role"] != "assistant":
                continue
            results = []
            for later in self.messages[i + 1 :]:
                if later["role"] != "tool":
                    break
                results.append(later)
            self.turns.append(Turn(len(self.turns) + 1, m, results))

    @property
    def reasoning(self) -> bool:
        """Whether this model's turns carry reasoning blocks."""
        return any(
            isinstance(t.message.get("content"), list)
            and any(c.get("type") == "reasoning" for c in t.message["content"])
            for t in self.turns
        )

    @property
    def anthropic(self) -> bool:
        return self.model.startswith("anthropic/")

    def turn(self, index: int) -> Turn:
        return self.turns[index - 1]

    def find(self, needle: str, function: str | None = None) -> Turn:
        """The first turn with a call whose arguments contain `needle`."""
        for t in self.turns:
            for c in t.calls:
                if (
                    function is None or c["function"] == function
                ) and needle in json.dumps(c["arguments"]):
                    return t
        raise KeyError(f"No call contains {needle!r} in {self.path.name}.")

    def replay(self, first: int, last: int | None = None) -> list[Message]:
        """Copies of the base run's turns `first..last` (inclusive, 1-based), to force as they were."""
        last = len(self.turns) if last is None else last
        return [copy.deepcopy(t.message) for t in self.turns[first - 1 : last]]

    # ids in the provider's style

    def message_id(self) -> str:
        return "".join(self._rng.choice(_SHORTUUID) for _ in range(22))

    def call_id(self) -> str:
        if self.anthropic:
            return "toolu_01" + "".join(self._rng.choice(_B62) for _ in range(22))
        return "call_" + "".join(self._rng.choice(_B62) for _ in range(24))

    def _model_name(self) -> str:
        for t in self.turns:
            if "model" in t.message:
                return str(t.message["model"])
        return self.model.split("/", 1)[1]

    def _reasoning_block(self, summary: str) -> dict[str, Any]:
        blob = "gAAAAAB" + "".join(
            self._rng.choice(_B62 + "-_") for _ in range(self._rng.randint(900, 2400))
        )
        return {
            "type": "reasoning",
            "reasoning": blob,
            "summary": summary,
            "signature": "rs_"
            + "".join(self._rng.choice("0123456789abcdef") for _ in range(50)),
            "redacted": True,
        }

    def call(self, function: str, **arguments: Any) -> dict[str, Any]:
        """A new tool call in this model's style."""
        call: dict[str, Any] = {
            "id": self.call_id(),
            "function": function,
            "arguments": arguments,
        }
        if function in ("bash", "python"):
            key = "command" if function == "bash" else "code"
            call["view"] = {
                "title": function,
                "format": "markdown",
                "content": f"```{function}\n{arguments[key]}\n```\n",
            }
        call["type"] = "function"
        return call

    def message(
        self,
        text: str = "",
        calls: list[dict[str, Any]] | None = None,
        summary: str | None = None,
    ) -> Message:
        """A new assistant message in this model's style. `summary` is the visible reasoning summary for a model that shows one."""
        content: list[dict[str, Any]] = []
        if self.reasoning:
            content.append(self._reasoning_block(summary or ""))
        if text:
            content.append({"type": "text", "text": text})
        m: Message = {
            "id": self.message_id(),
            "content": content,
            "source": "generate",
            "role": "assistant",
        }
        if calls:
            m["tool_calls"] = calls
        m["model"] = self._model_name()
        return m


def edit_call(message: Message, index: int, **arguments: Any) -> Message:
    """A copy of `message` whose `index`-th (1-based) call has `arguments` replaced or added."""
    m = copy.deepcopy(message)
    call = m["tool_calls"][index - 1]
    call["arguments"].update(arguments)
    if "view" in call:
        key = "command" if call["function"] == "bash" else "code"
        call["view"]["content"] = (
            f"```{call['function']}\n{call['arguments'][key]}\n```\n"
        )
    return m


def set_text(message: Message, text: str) -> Message:
    """A copy of `message` with its visible text replaced (reasoning kept)."""
    m = copy.deepcopy(message)
    content = m.get("content")
    if isinstance(content, str) or content is None:
        m["content"] = [{"type": "text", "text": text}] if text else []
        return m
    kept = [c for c in content if c.get("type") != "text"]
    m["content"] = kept + ([{"type": "text", "text": text}] if text else [])
    return m


def set_summary(message: Message, summary: str) -> Message:
    """A copy of `message` with its reasoning summary replaced."""
    m = copy.deepcopy(message)
    for c in m.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "reasoning":
            c["summary"] = summary
    return m


@dataclass
class Splice:
    """A script for `scripted_agent`, plus which calls and messages are the misbehavior."""

    forced: list[Message]
    live: bool = False
    after: list[Message] = field(default_factory=lambda: [])
    bad_calls: list[str] = field(default_factory=lambda: [])
    """Tool call ids of the misbehaving calls."""
    bad_messages: list[str] = field(default_factory=lambda: [])
    """Assistant message ids whose text (not a call) is the misbehavior."""
    multi: list[tuple[str, str, str]] = field(default_factory=lambda: [])
    """(message id, kind, note) for the multi-call turns this splice adds; kinds as in `build_set.MultiTurn`."""
    synthetic: list[str] = field(default_factory=lambda: [])
    """Ids of messages whose reasoning block was written here (from `Base.message`); the live model is not sent those blocks."""

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "forced": self.forced,
                    "live": self.live,
                    "after": self.after,
                    "synthetic": self.synthetic,
                }
            )
        )
