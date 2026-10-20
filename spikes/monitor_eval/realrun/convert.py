"""Convert a real run's eval log into a transcript file the loader reads (`../transcript.py`).

    convert.py LOG.eval OUT.json [--id ID]

The transcript keeps the conversation exactly as the log has it, including the system message, the user nudges a model got when it answered without a tool call, reasoning blocks and message ids. Tools are the ones the agent was offered; a tool result the host cut is recorded under `truncated`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from agent import agent_tools  # noqa: E402
from inspect_ai.event import ModelEvent, ToolEvent  # noqa: E402
from inspect_ai.log import EvalLog, EvalSample, read_eval_log  # noqa: E402
from inspect_ai.tool import ToolDef, ToolInfo  # noqa: E402
from registry import task_spec  # noqa: E402


def offered_tools(sample: EvalSample) -> list[ToolInfo]:
    for event in sample.events:
        if isinstance(event, ModelEvent) and event.tools:
            return list(event.tools)
    # A run with no live model call (every turn forced) offered the same tools.
    infos = []
    for t in agent_tools(concurrent=False):
        d = t if isinstance(t, ToolDef) else ToolDef(t)
        infos.append(
            ToolInfo(name=d.name, description=d.description, parameters=d.parameters)
        )
    return infos


def transcript_data(log: EvalLog, id: str) -> dict[str, Any]:
    assert log.samples, f"{log.location} has no samples."
    sample = log.samples[0]
    spec = task_spec(str(sample.id))
    truncated: dict[str, list[int]] = {}
    for event in sample.events:
        if isinstance(event, ToolEvent) and event.truncated:
            truncated[event.id] = list(event.truncated)
    return {
        "id": id,
        "model": log.eval.model,
        "task": {
            "name": spec.id,
            "description": None,
            "sample_id": id,
            "input": spec.input,
            "metadata": {},
        },
        "policy": spec.policy,
        "tools": [t.model_dump(exclude_none=True) for t in offered_tools(sample)],
        "messages": [m.model_dump(exclude_none=True) for m in sample.messages],
        "turns": {},
        "truncated": truncated,
    }


def write_transcript(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log")
    p.add_argument("out")
    p.add_argument("--id")
    args = p.parse_args()
    out = Path(args.out)
    write_transcript(transcript_data(read_eval_log(args.log), args.id or out.stem), out)


if __name__ == "__main__":
    main()
