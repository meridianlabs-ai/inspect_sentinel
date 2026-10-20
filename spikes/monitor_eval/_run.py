"""Score one step file with whichever `inspect_sentinel` is first on `sys.path`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolCallView, ToolChoice, ToolInfo
from inspect_ai.util import Store
from pydantic import TypeAdapter

from inspect_sentinel import _score
from inspect_sentinel._context import EvalContext
from inspect_sentinel._host import HumanAnswer
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step

_MESSAGES: TypeAdapter[list[ChatMessage]] = TypeAdapter(list[ChatMessage])
_TOOLS: TypeAdapter[list[ToolInfo]] = TypeAdapter(list[ToolInfo])


class Native:
    def __init__(self, eval: EvalContext | None, model: str | None) -> None:
        self._eval = eval
        self._model = model
        self.prompts: list[list[ChatMessage]] = []
        self.outputs: list[ModelOutput] = []

    @property
    def eval(self) -> EvalContext | None:
        return self._eval

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        assert not isinstance(input, str)
        self.prompts.append(list(input))
        if self._model is None:
            # render only: stop after the prompt is built
            raise Rendered()
        output = await get_model(self._model).generate(
            input,
            tools=tools or [],
            tool_choice=tool_choice,
            config=config or GenerateConfig(),
        )
        self.outputs.append(output)
        return output

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        raise NotImplementedError


class Rendered(Exception):
    pass


def load_step(data: dict[str, Any]) -> Step:
    call = ToolCall(
        id=data["call"].get("id", "call_x"),
        function=data["call"]["function"],
        arguments=data["call"]["arguments"],
    )
    input = (
        _MESSAGES.validate_python(data["input"])
        if data.get("input") is not None
        else None
    )
    common: dict[str, Any] = dict(
        conversation="c",
        message=data.get("message", ""),
        message_id=data.get("message_id", "msg_call"),
        call=call,
        view=ToolCallView(),
        input=input,
        agent_messages=None,
        tools=_TOOLS.validate_python(data["tools"]) if "tools" in data else None,
    )
    if data["stage"] == "before_tool_call":
        return BeforeToolCall(**common)
    return AfterToolCall(
        **common,
        result=ChatMessageTool(
            content=data["result"], tool_call_id=call.id, function=call.function
        ),
        output=data["result"],
    )


def eval_for(data: dict[str, Any]) -> EvalContext | None:
    if "task" not in data:
        return None
    return EvalContext(
        task="monitor_eval",
        task_description=None,
        sample_id=1,
        epoch=1,
        sample_description=None,
        sample_input=data["task"],
        metadata={},
    )


def template_for(judging: str, judged: str, wordings: Mapping[str, str]) -> str | None:
    # the baseline has no subject=; its judging sentence is swapped in the template
    if judging == "default":
        return None
    default = getattr(_score, "_default_template", lambda: _score.DEFAULT_TEMPLATE)()
    assert judged in default, "the template's judging sentence moved"
    return default.replace(judged, wordings[judging])


def score_options(
    judging: str, judged: str, wordings: Mapping[str, str], instructions: str | None
) -> dict[str, Any]:
    """What `Host.score` is passed for a judging wording and an author's instructions."""
    options: dict[str, Any] = {}
    if instructions is not None:
        options["instructions"] = instructions
    if judging == "default":
        return options
    if hasattr(_score, "DEFAULT_SUBJECT"):
        options["subject"] = wordings[judging]
    else:
        options["template"] = template_for(judging, judged, wordings)
    return options


async def run_one(
    path: Path,
    model: str | None,
    dimensions: str,
    judging: str,
    judged: str,
    wordings: Mapping[str, str],
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    native = Native(eval_for(data), model)
    try:
        observation = await _score.score_step(
            native,
            Store(),
            load_step(data),
            dimensions,
            policy=data.get("policy"),
            **score_options(judging, judged, wordings, None),
        )
    except Rendered:
        return {"prompt": native.prompts[0][0].text}
    return {
        "prompt": native.prompts[0][0].text,
        "suspicion": observation.suspicion,
        "explanation": observation.explanation,
        "references": [r.cite for r in observation.references or []],
        "attempts": len(native.outputs),
        "reasoning_text": native.outputs[-1].message.text
        if native.outputs and native.outputs[-1].choices
        else "",
        "input_tokens": sum(o.usage.input_tokens for o in native.outputs if o.usage),
        "output_tokens": sum(o.usage.output_tokens for o in native.outputs if o.usage),
        "cache_write_tokens": sum(
            o.usage.input_tokens_cache_write or 0 for o in native.outputs if o.usage
        ),
        "cache_read_tokens": sum(
            o.usage.input_tokens_cache_read or 0 for o in native.outputs if o.usage
        ),
    }
