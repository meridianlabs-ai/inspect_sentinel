"""A local benign set: tool calls sampled from a directory of real eval logs, as `allow` cases, for measuring false positives on real work.

The runs are assumed honest, so every sampled call is labelled `allow`. The cases are written under the benchmark's gitignored output directory and are never committed, since they hold the logs' content. Only `bash` calls are sampled, each with up to `CONTEXT` earlier `bash` calls of its sample and their results before it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)

from .cases import BENIGN, Call, Case, Step, write_cases
from .paths import require_safe

CONTEXT = 4
"""How many earlier calls a sampled case shows."""

RESULT_CHARS = 2000
"""How much of a recorded result a case keeps."""


def sample_benign(log_dir: Path, count: int, out: Path, *, seed: int = 0) -> Path:
    """Sample `count` `bash` calls from the eval logs in `log_dir` into `allow` cases, write them to `out/local/cases.yaml`, and return that file.

    Raises:
        UnsafeOutput: If `out`, or a link beneath it, resolves to where git would take what is written there.
        ValueError: If the logs hold no `bash` calls.
    """
    require_safe(out)
    candidates: list[Case] = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        for sample in log.samples or []:
            candidates.extend(
                _cases(
                    f"{log.eval.task}-{sample.id}-{sample.epoch}",
                    sample.input,
                    sample.messages,
                )
            )
    if not candidates:
        raise ValueError(f"No bash calls found in the logs in {log_dir}.")
    chosen = random.Random(seed).sample(candidates, min(count, len(candidates)))
    cases = [
        case.model_copy(update={"id": f"local-{i:04d}"})
        for i, case in enumerate(chosen)
    ]
    file = out / "local" / "cases.yaml"
    write_cases(cases, file)
    return file


def _cases(source: str, input: object, messages: Sequence[ChatMessage]) -> list[Case]:
    task = (
        input
        if isinstance(input, str)
        else next((m.text for m in messages if isinstance(m, ChatMessageUser)), "")
    )
    results = {
        m.tool_call_id: m.text
        for m in messages
        if isinstance(m, ChatMessageTool) and m.tool_call_id
    }
    earlier: list[Step] = []
    found: list[Case] = []
    for message in messages:
        if not isinstance(message, ChatMessageAssistant):
            continue
        for call in message.tool_calls or []:
            command = call.arguments.get("cmd", call.arguments.get("command"))
            if call.function != "bash" or not isinstance(command, str):
                continue
            found.append(
                Case(
                    id="local",
                    label="allow",
                    category=BENIGN,
                    why=f"Sampled from a run assumed honest ({source}).",
                    task=task or "(no task text)",
                    steps=tuple(earlier[-CONTEXT:]),
                    call=Call(tool="bash", args={"command": command}),
                )
            )
            earlier.append(
                Step(
                    tool="bash",
                    args={"command": command},
                    result=results.get(call.id, "")[:RESULT_CHARS],
                )
            )
    return found
