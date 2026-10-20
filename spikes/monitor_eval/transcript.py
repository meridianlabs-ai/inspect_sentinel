"""Load a stored agent conversation and score its steps through `Host.score`.

A transcript file is JSON: the task, the policy, the tools the agent was offered, and the agent's conversation as inspect_ai `ChatMessage`s. `load_transcript()` reads one, and `Transcript.steps()` rebuilds the four step types the host would have handed a monitor at each turn: `BeforeGenerate` and `AfterGenerate` for each assistant message, then `BeforeToolCall` and `AfterToolCall` for each of its tool calls. `score_transcript()` runs `Host.score` on chosen steps inside `running_step`, with one sample store for the whole transcript, as a live host does.

File fields (all but `messages` optional):

- `id`: the transcript's name; also its conversation id.
- `model`: the agent's model, as `provider/name`.
- `task`: `name`, `description`, `sample_id`, `input` (the assignment) and `metadata`, as `EvalContext` holds them.
- `policy`: what the agent must follow whatever its task says; passed to `score(policy=...)`.
- `tools`: `ToolInfo` dumps, the tools offered to every turn.
- `messages`: `ChatMessage` dumps in conversation order. Each tool message follows the assistant message that made its call.
- `turns`: per assistant message id, any of `tool_choice` (`"auto"`, `"none"`, `"any"` or `{"name": ...}`), `tools` (the names offered to that turn, a subset of `tools`), and `prefilled` (true when no model call produced the message, so its tool stages have no `input` or `tools`).
- `truncated`: per tool call id, `[original_bytes, limit_bytes]` for a result the host cut.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCallView, ToolChoice, ToolFunction, ToolInfo
from inspect_ai.util import Store
from pydantic import TypeAdapter

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Observation,
    Step,
    StepPreprocessor,
)
from inspect_sentinel._context import Context, EvalContext
from inspect_sentinel._host import HostContext, HumanAnswer, running_step
from inspect_sentinel._report import Decision, Failed, Report, Reported

Stage = Literal["model_input", "model_output", "tool_call", "tool_result"]

STAGES: tuple[Stage, ...] = ("model_input", "model_output", "tool_call", "tool_result")

_MESSAGES = TypeAdapter(list[ChatMessage])
_TOOLS = TypeAdapter(list[ToolInfo])


@dataclass(frozen=True)
class TranscriptStep:
    """One step of a transcript, as a host would hand it to a monitor."""

    id: str
    """`t03.model_input`, `t03.model_output`, `t03.c2.tool_call` or `t03.c2.tool_result`: turn 3 is the third assistant message, c2 its second tool call."""

    stage: Stage
    step: Step


@dataclass(frozen=True)
class Transcript:
    id: str
    model: str
    task: EvalContext
    policy: str | None
    tools: list[ToolInfo]
    messages: list[ChatMessage]
    turns: dict[str, dict[str, Any]] = field(default_factory=lambda: {})
    truncated: dict[str, tuple[int, int]] = field(default_factory=lambda: {})

    def steps(self, stages: Sequence[Stage] = STAGES) -> list[TranscriptStep]:
        """Every step of the conversation at `stages`, in the order a host meets them."""
        steps: list[TranscriptStep] = []
        history: list[ChatMessage] = []
        seen: set[str] = set()
        turn = 0
        for i, message in enumerate(self.messages):
            if not isinstance(message, ChatMessageAssistant):
                continue
            turn += 1
            options = self.turns.get(message.id or "", {})
            prefilled = bool(options.get("prefilled", False))
            input = list(self.messages[:i])
            before = list(history)
            if not prefilled:
                for m in input:
                    if m.id not in seen:
                        seen.add(m.id or "")
                        history.append(m)
            offered = self._offered(options)
            choice = _tool_choice(options.get("tool_choice", "auto"))
            prefix = f"t{turn:02d}"
            if not prefilled:
                steps.append(
                    TranscriptStep(
                        f"{prefix}.model_input",
                        "model_input",
                        BeforeGenerate(
                            model=self.model,
                            conversation=self.id,
                            input=input,
                            agent_messages=input,
                            history=before,
                            human_decisions=(),
                            tools=offered,
                            tool_choice=choice,
                            config=GenerateConfig(),
                        ),
                    )
                )
                steps.append(
                    TranscriptStep(
                        f"{prefix}.model_output",
                        "model_output",
                        AfterGenerate(
                            model=self.model,
                            conversation=self.id,
                            input=input,
                            agent_messages=input,
                            history=list(history),
                            human_decisions=(),
                            tools=offered,
                            tool_choice=choice,
                            config=GenerateConfig(),
                            output=ModelOutput(
                                model=self.model,
                                choices=[
                                    ChatCompletionChoice(
                                        message=message,
                                        stop_reason="tool_calls"
                                        if message.tool_calls
                                        else "stop",
                                    )
                                ],
                            ),
                        ),
                    )
                )
            results = {
                m.tool_call_id: m
                for m in self.messages[i + 1 :]
                if isinstance(m, ChatMessageTool)
            }
            done: list[ChatMessage] = []
            for k, call in enumerate(message.tool_calls or [], start=1):
                agent_messages = [*self.messages[:i], message, *done]
                common: dict[str, Any] = dict(
                    conversation=self.id,
                    message=message.text,
                    message_id=message.id or "",
                    call=call,
                    view=ToolCallView(),
                    input=None if prefilled else input,
                    agent_messages=agent_messages,
                    history=list(history),
                    human_decisions=(),
                    tools=None if prefilled else offered,
                )
                steps.append(
                    TranscriptStep(
                        f"{prefix}.c{k}.tool_call",
                        "tool_call",
                        BeforeToolCall(**common),
                    )
                )
                result = results.get(call.id)
                if result is None:
                    continue
                cut = self.truncated.get(call.id)
                steps.append(
                    TranscriptStep(
                        f"{prefix}.c{k}.tool_result",
                        "tool_result",
                        AfterToolCall(
                            **common,
                            result=result,
                            output=result.text,
                            truncated=False if cut is None else _truncation(cut),
                        ),
                    )
                )
                done.append(result)
        return [s for s in steps if s.stage in stages]

    def step(self, id: str) -> TranscriptStep:
        """The step named `id`."""
        for s in self.steps():
            if s.id == id:
                return s
        raise KeyError(f"{self.id} has no step {id!r}.")

    def _offered(self, options: dict[str, Any]) -> list[ToolInfo]:
        names = options.get("tools")
        return (
            self.tools
            if names is None
            else [t for t in self.tools if t.name in set(names)]
        )


def _tool_choice(value: Any) -> ToolChoice:
    if isinstance(value, dict):
        return ToolFunction(name=str(value["name"]))  # pyright: ignore[reportUnknownArgumentType]
    if value in ("auto", "none", "any"):
        return value
    raise ValueError(f"Unknown tool choice {value!r}.")


def _truncation(cut: tuple[int, int]) -> Any:
    from inspect_sentinel._step import Truncation

    return Truncation(*cut)


def load_transcript(path: str | Path) -> Transcript:
    """Read a transcript file."""
    data = json.loads(Path(path).read_text())
    task = data.get("task", {})
    return Transcript(
        id=data.get("id", Path(path).stem),
        model=data.get("model", "unknown/agent"),
        task=EvalContext(
            task=task.get("name", "task"),
            task_description=task.get("description"),
            sample_id=task.get("sample_id", data.get("id", Path(path).stem)),
            epoch=1,
            sample_description=None,
            sample_input=task.get("input", ""),
            metadata=task.get("metadata", {}),
        ),
        policy=data.get("policy"),
        tools=_TOOLS.validate_python(data.get("tools", [])),
        messages=_MESSAGES.validate_python(data["messages"]),
        turns=data.get("turns", {}),
        truncated={k: (v[0], v[1]) for k, v in data.get("truncated", {}).items()},
    )


class ReplayHost:
    """A host for scoring stored steps: the transcript's task as the eval, and inference through inspect_ai's `get_model`."""

    def __init__(self, eval: EvalContext, default_model: str | None = None) -> None:
        self._eval = eval
        self._default_model = default_model
        self.prompts: list[list[ChatMessage]] = []

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
        if not isinstance(input, str):
            self.prompts.append(input)
        name = model or self._default_model
        m = get_model(name) if name else get_model(role=role or "monitor")
        return await m.generate(
            input,
            tools=tools or [],
            tool_choice=tool_choice,
            config=config or GenerateConfig(),
        )

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        raise RuntimeError("A replayed transcript has nobody to ask.")


class DryRunHost(ReplayHost):
    """A `ReplayHost` that never calls a model: it keeps each prompt and answers 0.0, to check that a transcript loads and renders."""

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
        if not isinstance(input, str):
            self.prompts.append(input)
        names = [
            name
            for t in tools or []
            for name in (t.parameters.properties or {})
            if name != "explanation"
        ]
        return ModelOutput.for_tool_call(
            model="dry-run",
            tool_name="answer",
            tool_arguments={"explanation": "dry run", **{n: 0.0 for n in names}},
        )


class _NoRecorder:
    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        pass

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        pass

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        pass

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        pass

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        pass


async def score_transcript(
    transcript: Transcript,
    dimensions: str | dict[str, str],
    *,
    host: ReplayHost | None = None,
    stages: Sequence[Stage] = STAGES,
    step_ids: Sequence[str] | None = None,
    preprocessor: StepPreprocessor[Any] | None = None,
    **score_options: Any,
) -> list[tuple[TranscriptStep, Observation]]:
    """Run `Host.score` on each chosen step, in order, with one sample store for the transcript (so one nonce, as in a live sample).

    `score_options` go to `Host.score` (`instructions`, `template`, `model`, `role`, ...); `policy` defaults to the transcript's.
    """
    host = host or ReplayHost(transcript.task)
    store = Store()
    recorder = _NoRecorder()
    context = HostContext(host=host, recorder=recorder, store=store).context
    score_options.setdefault("policy", transcript.policy)
    chosen = [
        s for s in transcript.steps(stages) if step_ids is None or s.id in set(step_ids)
    ]
    scored: list[tuple[TranscriptStep, Observation]] = []
    for s in chosen:
        with running_step(recorder, store):
            observation = await context.host.score(
                s.step, dimensions, preprocessor=preprocessor, **score_options
            )
        scored.append((s, observation))
    return scored
