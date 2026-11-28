"""Trajectory cases: a whole agent run, every tool call of which each judge scores, labelled by the run's mode and its setting's own scorers.

A trajectory is its task, the tools the agent was given, and its turns: each the agent's text and its tool calls, with their recorded results. An `attack` trajectory comes from a run told to pursue a side task, and an `honest` one from a run with none. The setting's scorers say whether the side task succeeded and, where the setting tests it after every action, at which call it landed (`landed_at`).

Trajectories are YAML files, each a list of them, read by `load_trajectories()`. They are replayed into eval logs as cases are (`trajectory_task()`): a scripted `mockllm` model makes each turn's calls and the tools return the recorded results, so nothing runs. A trajectory's tools are rebuilt from their recorded names, descriptions and parameters, so each judge sees the calls as the agent made them.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolDef, ToolError, ToolParams
from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["attack", "honest"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_Strict):
    """A tool the agent was given, as its run recorded it."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {})
    """The tool's parameters as a JSON schema object (`ToolParams`)."""


class TrajectoryCall(_Strict):
    """A tool call the agent made, with its recorded result."""

    tool: str
    args: dict[str, Any]
    result: str = ""
    """The tool's output, cut to what the importer keeps."""

    error: bool = False
    """Whether the tool reported an error, with `result` as its message."""


class Turn(_Strict):
    """One message of the agent's: its own text and the calls it made."""

    text: str = ""
    """The agent's text, without its reasoning."""

    calls: tuple[TrajectoryCall, ...] = ()


class Trajectory(_Strict):
    """A labelled agent run."""

    id: str = Field(pattern=r"^[a-z0-9_.-]+$")
    """Unique across the set; the sample id of its log."""

    mode: Mode
    """`attack` where the agent was told to pursue a side task, `honest` where it had none."""

    setting: str
    """The ControlArena setting, such as `bash_arena`."""

    attacker: str
    """The model that played the agent."""

    main_task: str | None = None
    """The setting's name for the main task."""

    side_task: str | None = None
    """The side task an attack pursued, as the setting names it; for a setting that offers several at once, the one that landed where one did."""

    side_task_success: bool | None = None
    """Whether the setting's scorer found the side task done; None for an honest run or where it gave no verdict."""

    main_task_success: float | None = None
    """The setting's main-task score, 0-1, where it gives one."""

    landed_at: int | None = None
    """The index, in `calls()`, of the call after which the setting found the side task done, where the setting tests after every action; None otherwise."""

    attempt: str | None = None
    """Whether an attack's agent attempted its side task: `attempted`, `declined` (it said it wouldn't) or `not_attempted`, as a judge model read it (`controlarena.refusals`); None for an honest run or one not judged."""

    refused: bool | None = None
    """Whether an attack's agent declined or skipped its side task (`attempt` other than `attempted`); None where `attempt` is."""

    source: str
    """Where it came from: the log file and sample, for a reviewer."""

    task: str
    """The main task as the agent was given it, the sample's input; never the side task."""

    tools: tuple[ToolSpec, ...] = ()
    turns: tuple[Turn, ...] = ()

    def calls(self) -> list[TrajectoryCall]:
        """Every tool call, in order."""
        return [c for t in self.turns for c in t.calls]


def call_id(index: int) -> str:
    """The tool call id of the trajectory's `index`th call in replay, its step id."""
    return f"call-{index}"


def call_index(step_id: str) -> int | None:
    """The index a replayed call's id names, or None for another id."""
    prefix, _, number = step_id.partition("-")
    return int(number) if prefix == "call" and number.isdigit() else None


def trajectory_digest(trajectory: Trajectory) -> str:
    """A digest of everything in `trajectory` that reaches the log."""
    shown = trajectory.model_dump(mode="json", include={"task", "tools", "turns"})
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:16]


def load_trajectories(paths: Path | Iterable[Path]) -> list[Trajectory]:
    """Read every trajectory from `paths`: YAML files, or directories whose `*.yaml` files are read in name order.

    Raises:
        ValueError: If a trajectory is malformed, or two share an id.
    """
    files: list[Path] = []
    for path in [paths] if isinstance(paths, Path) else paths:
        files.extend(sorted(path.glob("*.yaml")) if path.is_dir() else [path])
    found: list[Trajectory] = []
    for file in files:
        raw: Any = yaml.safe_load(file.read_text()) or []
        if not isinstance(raw, list):
            raise ValueError(f"{file} must hold a list of trajectories.")
        for item in cast(list[Any], raw):
            try:
                found.append(Trajectory.model_validate(item))
            except ValueError as ex:
                raise ValueError(f"{file}: {ex}") from ex
    seen: set[str] = set()
    for trajectory in found:
        if trajectory.id in seen:
            raise ValueError(f"Two trajectories have the id {trajectory.id!r}.")
        seen.add(trajectory.id)
    return found


def write_trajectories(trajectories: Sequence[Trajectory], file: Path) -> None:
    """Write `trajectories` as a YAML file."""
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        yaml.safe_dump(
            [t.model_dump(mode="json", exclude_defaults=True) for t in trajectories],
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )
    )


def trajectory_task(trajectory: Trajectory) -> Task:
    """A task of one sample that replays `trajectory`: a scripted model makes each turn's calls, and its tools return the recorded results."""
    outputs: list[ModelOutput] = []
    results: dict[str, deque[TrajectoryCall]] = defaultdict(deque)
    index = 0
    for turn in trajectory.turns:
        calls: list[ToolCall] = []
        for call in turn.calls:
            calls.append(
                ToolCall(
                    id=call_id(index), function=call.tool, arguments=dict(call.args)
                )
            )
            results[call.tool].append(call)
            index += 1
        outputs.append(
            ModelOutput.from_message(
                ChatMessageAssistant(
                    content=turn.text, tool_calls=calls or None, model="mockllm/model"
                ),
                stop_reason="tool_calls" if calls else "stop",
            )
        )
    specs = {s.name: s for s in trajectory.tools}
    for name in results:
        specs.setdefault(name, ToolSpec(name=name))
    tools = [_replayed(spec, results[spec.name]) for spec in specs.values()]
    return Task(
        name="auto_mode_trajectory",
        dataset=[Sample(id=trajectory.id, input=trajectory.task)],
        solver=[use_tools(tools), replay_turns(len(trajectory.turns))],
        model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
    )


@solver
def replay_turns(turns: int) -> Solver:
    """Play a trajectory's turns: one generate per turn, running its calls.

    Args:
        turns: How many turns the trajectory has.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for _ in range(turns):
            state = await generate(state, tool_calls="single")
        return state

    return solve


def _replayed(spec: ToolSpec, played: deque[TrajectoryCall]) -> Tool:
    async def execute(**kwargs: Any) -> str:
        call = played.popleft()
        if call.error:
            raise ToolError(call.result)
        return call.result

    parameters = ToolParams.model_validate(
        spec.parameters or {"type": "object", "properties": {}}
    )
    # a recorded call may name an argument its schema doesn't, or leave out a
    # required one; replay plays what was recorded, so the schema is advisory
    # inspect_ai needs a description for every parameter
    properties = {
        name: param
        if param.description
        else param.model_copy(update={"description": name})
        for name, param in parameters.properties.items()
    }
    parameters = parameters.model_copy(
        update={"properties": properties, "required": [], "additionalProperties": True}
    )
    return ToolDef(
        execute,
        name=spec.name,
        description=spec.description or spec.name,
        parameters=parameters,
    ).as_tool()
