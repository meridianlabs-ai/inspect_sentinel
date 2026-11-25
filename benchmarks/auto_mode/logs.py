"""Cases into eval logs, once: each case replayed by a scripted model, so its log holds the transcript with the call under test.

Each case is a task of one sample. Its model is `mockllm` playing the case's tool calls in order, and its tools return the recorded results, so no command runs and the transcript is the case's exactly. The solver sends any later user message where the case has it. No sentinel is attached: the monitors run later, in replay (`score.py`), so the logs are made once and every monitor is scored on the same ones.

The sample's input is the case's task, and its metadata is empty, so nothing a judge is shown carries the label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools
from inspect_ai.tool import Tool, ToolCall, tool

from .cases import CALL_UNDER_TEST, Call, Case
from .paths import require_safe

MANIFEST = "cases.json"
"""The file in the log directory recording, for each case, its digest, its log file and that file's sha256, so a scan can check the logs are exactly the cases' as they are now."""

MANIFEST_VERSION = 2
"""The manifest's format; logs with an older one are made again."""


def case_digest(case: Case) -> str:
    """A digest of everything in `case` that reaches the log."""
    shown = case.model_dump(mode="json", include={"task", "steps", "call"})
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:16]


def case_task(case: Case) -> Task:
    """A task of one sample that replays `case`: a scripted model makes its calls, and its tools return its recorded results."""
    calls = [step.as_call() for step in case.steps if step.user is None] + [case.call]
    results = [step.result or "" for step in case.steps if step.user is None] + [""]
    ids = [f"call-{i + 1}" for i in range(len(calls) - 1)] + [CALL_UNDER_TEST]
    outputs = [_turn(call, id) for call, id in zip(calls, ids, strict=True)]
    played = iter(results)
    return Task(
        name="auto_mode_case",
        dataset=[Sample(id=case.id, input=case.task)],
        solver=[
            use_tools(
                bash(played), write_file(played), python(played), text_editor(played)
            ),
            replay_case([step.user for step in case.steps]),
        ],
        model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
    )


@solver
def replay_case(users: list[str | None]) -> Solver:
    """Play a case's earlier steps, then its call under test: one generate per tool call, each running its call, and a user message sent where `users` has one.

    Args:
        users: Each earlier step in order: its user message, or `None` for a tool call. The solver's arguments are recorded in the log, so they hold nothing the transcript doesn't.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for user in users:
            if user is not None:
                state.messages.append(ChatMessageUser(content=user))
            else:
                state = await generate(state, tool_calls="single")
        return await generate(state, tool_calls="single")

    return solve


@tool
def bash(results: Iterator[str]) -> Tool:
    async def execute(command: str) -> str:
        """Run a bash command in the agent's container and return its output.

        Args:
            command: The command to run.
        """
        return next(results)

    return execute


@tool
def write_file(results: Iterator[str]) -> Tool:
    async def execute(path: str, content: str) -> str:
        """Write `content` to the file at `path`, replacing it if it exists.

        Args:
            path: The file's path.
            content: What to write.
        """
        return next(results)

    return execute


@tool
def python(results: Iterator[str]) -> Tool:
    async def execute(code: str) -> str:
        """Run Python code in the agent's container and return what it prints.

        Args:
            code: The code to run.
        """
        return next(results)

    return execute


@tool
def text_editor(results: Iterator[str]) -> Tool:
    async def execute(
        command: str,
        path: str,
        file_text: str | None = None,
        insert_line: int | None = None,
        new_str: str | None = None,
        old_str: str | None = None,
        view_range: list[int] | None = None,
    ) -> str:
        """View, create or edit a file: `view` shows it, `create` writes `file_text` to it, `str_replace` replaces `old_str` with `new_str`, and `insert` puts `new_str` after line `insert_line`.

        Args:
            command: `view`, `create`, `str_replace`, `insert` or `undo_edit`.
            path: The file's path.
            file_text: The file's content, for `create`.
            insert_line: The line to insert after, for `insert`.
            new_str: The new text, for `str_replace` and `insert`.
            old_str: The text to replace, for `str_replace`.
            view_range: The lines to show, for `view`.
        """
        return next(results)

    return execute


def generate_logs(
    cases: Sequence[Case], log_dir: Path, *, max_tasks: int = 32, force: bool = False
) -> Path:
    """Write a log for each case under `log_dir`, with the manifest naming each case's digest and log, and return `log_dir`.

    Raises:
        UnsafeOutput: If `log_dir`, or a link beneath it, resolves to where git would take what is written there.
        FileExistsError: If `log_dir` already holds logs and `force` is not set; logs are made once and reused.
        RuntimeError: If a case's log did not succeed.
    """
    require_safe(log_dir)
    if (log_dir / MANIFEST).exists() and not force:
        raise FileExistsError(
            f"{log_dir} already holds logs; they are made once and reused. Pass --force to make them again."
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / MANIFEST).unlink(missing_ok=True)
    for old in _logs_in(log_dir):
        old.unlink()
    logs = eval(
        [case_task(case) for case in cases],
        log_dir=str(log_dir),
        display="none",
        max_tasks=max_tasks,
        fail_on_error=True,
        score=False,
    )
    failed = [log.location for log in logs if log.status != "success"]
    if failed:
        raise RuntimeError(f"{len(failed)} case logs did not succeed.")
    files: dict[str, str] = {}
    for log in logs:
        [sample_id] = log.eval.dataset.sample_ids or [None]
        files[str(sample_id)] = Path(log.location).name
    entries = {
        case.id: {
            "digest": case_digest(case),
            "log": files[case.id],
            "sha256": _sha256(log_dir / files[case.id]),
        }
        for case in cases
    }
    (log_dir / MANIFEST).write_text(
        json.dumps({"version": MANIFEST_VERSION, "cases": entries}, indent=1)
    )
    return log_dir


def check_manifest(cases: Sequence[Case], log_dir: Path) -> dict[str, str]:
    """Check that `log_dir` holds exactly one log for each of `cases`, made from it as it is now and unchanged since, and no other log at any depth; return each case's digest by id.

    Raises:
        FileNotFoundError: If `log_dir` has no manifest.
        ValueError: If the manifest doesn't name exactly `cases` as they are now, or the logs in `log_dir` aren't exactly the ones it names.
    """
    manifest: dict[str, Any] = json.loads((log_dir / MANIFEST).read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"The logs in {log_dir} were made by an older version of the benchmark; make them again with `logs --force`."
        )
    entries: dict[str, dict[str, str]] = manifest["cases"]
    ids = {c.id for c in cases}
    missing = sorted(ids - set(entries))
    extra = sorted(set(entries) - ids)
    changed = sorted(
        c.id
        for c in cases
        if c.id in entries and entries[c.id]["digest"] != case_digest(c)
    )
    named = {e["log"] for e in entries.values()}
    present = {f.relative_to(log_dir).as_posix() for f in _logs_in(log_dir)}
    gone = sorted(i for i, e in entries.items() if e["log"] not in present)
    altered = sorted(
        i
        for i, e in entries.items()
        if e["log"] in present and _sha256(log_dir / e["log"]) != e["sha256"]
    )
    unnamed = sorted(present - named)
    if missing or extra or changed or gone or altered or unnamed:
        raise ValueError(
            f"The logs in {log_dir} don't match the cases (cases with no log: {missing[:5]}; logs of other cases: {extra[:5]}; cases changed: {changed[:5]}; log files missing: {gone[:5]}; log files changed: {altered[:5]}; log files not in the manifest: {unnamed[:5]}); make them again with `logs --force`."
        )
    return {i: e["digest"] for i, e in entries.items()}


def log_files(cases: Sequence[Case], log_dir: Path) -> list[Path]:
    """The log of each of `cases` in `log_dir`, in their order, once `check_manifest()` finds the directory is exactly theirs; a scan is given these files, never the directory.

    Raises:
        FileNotFoundError: If `log_dir` has no manifest.
        ValueError: If the logs in `log_dir` are not exactly those of `cases` as they are now.
    """
    check_manifest(cases, log_dir)
    manifest: dict[str, Any] = json.loads((log_dir / MANIFEST).read_text())
    return [log_dir / manifest["cases"][c.id]["log"] for c in cases]


def _logs_in(log_dir: Path) -> list[Path]:
    # every eval log under log_dir at any depth, in either of Inspect's
    # formats, the manifest aside
    return sorted(
        f
        for pattern in ("*.eval", "*.json")
        for f in log_dir.rglob(pattern)
        if f != log_dir / MANIFEST
    )


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _turn(call: Call, id: str) -> ModelOutput:
    return ModelOutput.from_message(
        ChatMessageAssistant(
            content=call.text,
            tool_calls=[ToolCall(id=id, function=call.tool, arguments=dict(call.args))],
            model="mockllm/model",
        ),
        stop_reason="tool_calls",
    )
