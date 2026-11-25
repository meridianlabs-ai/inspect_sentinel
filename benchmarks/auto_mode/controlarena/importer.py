"""ControlArena logs into trajectory cases, keeping only what a judge may see and the setting's own labels.

What a trajectory keeps of a sample:

- **The task:** the main task's description from the sample's metadata (`main_task_description`), never its messages. An attack policy can write its instructions into the system prompt and the first user message (BashArena's does), so no system or user message is kept at all.
- **The turns:** each of the agent's messages: its text without its reasoning, and its tool calls with their results, each result cut to `RESULT_CHARS`. A message the harness inserted in the agent's place (BashArena's forced submit after a side task lands) is left out.
- **The tools:** each tool's name, description and parameters, from the sample's metadata (`tools`) or, failing that, its first model call.
- **The labels:** the mode; the side task's success and the main task's score from the setting's scorers; and where BashArena reports a side task passing after an action, the side task's name and the index of the call it passed after (`landed_at`). Nothing of a side task's description is kept.
- **Refusals:** whether an attack's agent attempted its side task, where `refusals.classify()` has judged it.

A sample that errored, or whose agent made no tool call, gives no trajectory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser

from ..trajectories import Mode, ToolSpec, Trajectory, TrajectoryCall, Turn

RESULT_CHARS = 4000
"""How much of a tool's output a trajectory keeps: its head and tail around a marker. About the 1,000 tokens Codex Guardian keeps of one."""

_LANDED = re.compile(r"^Side task '([^']+)' passed")
_AUTO_SUBMIT = "side_task_auto_submit"


def import_logs(
    files: Iterable[Path], attempts: Mapping[str, str] | None = None
) -> tuple[list[Trajectory], list[str]]:
    """The trajectories in ControlArena logs `files`, and a note for each sample skipped. `attempts` gives, by trajectory id, whether each attack's agent attempted its side task (`refusals.classify()`)."""
    found: list[Trajectory] = []
    skipped: list[str] = []
    for file in files:
        log = read_eval_log(str(file))
        model = _attacker(log.eval.model_roles or {}, log.eval.model)
        batch = cast(
            Mapping[str, Any], (log.eval.metadata or {}).get("amb_batch") or {}
        )
        setting = str(batch["setting"]) if batch.get("setting") else None
        for sample in log.samples or []:
            where = f"{file.name}#{sample.id}/{sample.epoch}"
            if sample.error is not None:
                skipped.append(f"{where}: the sample errored")
                continue
            trajectory = convert(sample, model, where, setting, attempts or {})
            if trajectory is None:
                skipped.append(f"{where}: no tool calls")
                continue
            found.append(trajectory)
    return found, skipped


def _attacker(roles: Mapping[str, Any], model: str) -> str:
    # the untrusted role's model, which plays the agent in ControlArena
    untrusted: Any = roles.get("untrusted")
    if isinstance(untrusted, list):
        untrusted = cast(list[Any], untrusted)[0] if untrusted else None
    found = getattr(untrusted, "model", None)
    return str(found) if found else model


def trajectory_id(sample: EvalSample, setting: str, attacker: str) -> str:
    """The id of the trajectory from `sample`, run in `setting` by `attacker`."""
    mode = (
        "attack"
        if str(sample.metadata.get("eval_mode", "")).lower() == "attack"
        else "honest"
    )
    return _id(
        f"ca-{setting}-{mode}-{attacker.split('/')[-1]}-{sample.id}-e{sample.epoch}"
    )


def convert(
    sample: EvalSample,
    attacker: str,
    where: str,
    setting: str | None = None,
    attempts: Mapping[str, str] | None = None,
) -> Trajectory | None:
    """A trajectory from one sample of a ControlArena log, or None where the agent made no tool call.

    `setting` is the ControlArena setting the batch ran, where the log records it; the sample's own `setting_name` can be narrower (AgentDojo gives its suite), and then prefixes the side task's name.
    """
    metadata = cast(Mapping[str, Any], sample.metadata)
    mode: Mode = (
        "attack" if str(metadata.get("eval_mode", "")).lower() == "attack" else "honest"
    )
    named = str(metadata.get("setting_name") or "unknown")
    setting = setting or named
    scope = f"{named}/" if named != setting else ""
    turns: list[Turn] = []
    results = {
        m.tool_call_id: m for m in sample.messages if isinstance(m, ChatMessageTool)
    }
    landed_at: int | None = None
    landed_task: str | None = None
    index = 0
    for message in sample.messages:
        if isinstance(message, ChatMessageUser):
            match = _LANDED.match(message.text)
            if match and landed_at is None and index > 0:
                landed_at, landed_task = index - 1, match.group(1)
            continue
        if (
            not isinstance(message, ChatMessageAssistant)
            or message.model == _AUTO_SUBMIT
        ):
            continue
        calls: list[TrajectoryCall] = []
        for call in message.tool_calls or []:
            result = results.get(call.id)
            calls.append(
                TrajectoryCall(
                    tool=call.function,
                    args=dict(call.arguments),
                    result=_cut(_result_text(result)),
                    error=result is not None and result.error is not None,
                )
            )
            index += 1
        turns.append(Turn(text=message.text.strip(), calls=tuple(calls)))
    if index == 0:
        return None
    scores = _scores(sample)
    side = landed_task or _name(metadata.get("side_task_name"))
    side_task = f"{scope}{side}" if mode == "attack" and side else None
    id = trajectory_id(sample, setting, attacker)
    attempt = (attempts or {}).get(id) if mode == "attack" else None
    return Trajectory(
        id=id,
        mode=mode,
        setting=setting,
        attacker=attacker,
        main_task=_name(metadata.get("main_task_name")),
        side_task=side_task,
        side_task_success=(
            scores.get("side_task_success") if landed_at is None else True
        )
        if mode == "attack"
        else None,
        main_task_success=_fraction(
            scores.get("main_task_score_value", scores.get("main_task_success_value"))
        ),
        landed_at=landed_at,
        attempt=attempt,
        refused=None if attempt is None else attempt != "attempted",
        source=where,
        task=str(metadata.get("main_task_description") or ""),
        tools=_tools(sample),
        turns=tuple(turns),
    )


def _scores(sample: EvalSample) -> dict[str, Any]:
    # the setting's verdicts, whether each is its own scorer or a key of one
    # scorer's dict value
    found: dict[str, Any] = {}
    for name, score in (sample.scores or {}).items():
        values: Mapping[str, Any] = (
            cast(Mapping[str, Any], score.value)
            if isinstance(score.value, Mapping)
            else {name: score.value}
        )
        for key, value in values.items():
            if key == "side_task_success":
                found[key] = _passed(value)
            elif key == "main_task_score" or (
                key == "main_task_success" and "main_task_score_value" not in found
            ):
                found[
                    "main_task_score_value"
                    if key == "main_task_score"
                    else "main_task_success_value"
                ] = value
    return found


def _passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"C": True, "I": False}.get(value.upper())
    if isinstance(value, int | float):
        return value >= 1
    return None


def _fraction(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        return {"C": 1.0, "I": 0.0}.get(value.upper())
    if isinstance(value, int | float):
        return float(value)
    return None


def _tools(sample: EvalSample) -> tuple[ToolSpec, ...]:
    listed: Any = sample.metadata.get("tools")
    if not isinstance(listed, list):
        model_events = [e for e in sample.events if e.event == "model"]
        listed = (
            [t.model_dump(mode="json") for t in model_events[0].tools]
            if model_events
            else []
        )
    specs: list[ToolSpec] = []
    for item in cast(list[Any], listed):
        if isinstance(item, Mapping):
            tool = cast(Mapping[str, Any], item)
            specs.append(
                ToolSpec(
                    name=str(tool.get("name")),
                    description=str(tool.get("description") or ""),
                    parameters=dict(
                        cast(Mapping[str, Any], tool.get("parameters") or {})
                    ),
                )
            )
    return tuple(specs)


def _result_text(message: ChatMessageTool | None) -> str:
    if message is None:
        return ""
    if message.error is not None:
        return message.error.message
    return message.text


def _cut(text: str) -> str:
    if len(text) <= RESULT_CHARS:
        return text
    half = RESULT_CHARS // 2
    return f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut ...]\n{text[-half:]}"


def _name(value: Any) -> str | None:
    return str(value) if value else None


def _id(text: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", text.lower()).strip("-")
