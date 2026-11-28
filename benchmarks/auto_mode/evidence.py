"""What a judge is shown of a trajectory, beyond each reproduction's own transcript: a cap on its prompt, and one shared rendering for the "same evidence" ablation.

**The cap.** A judge prompt is kept under `CAP` tokens, where Haiku's price rises fivefold. `auto_mode()` is held under it by the host, which fits `Host.score`'s prompt to the judge model's input budget by cutting the oldest messages and records what it cut in the observation's `left_out`; `cap_judge_model()` sets that budget. The Guardian and Claude Code reproductions build their own prompts, so `fit()` cuts their oldest transcript entries, after the task, to an estimate of the same budget, and they record what they cut the same way. The estimate counts `CHARS_PER_TOKEN` characters a token, fewer than most text takes, so it errs toward cutting.

**Same evidence.** `shared_transcript()` renders a step as `auto_mode()` is shown it, with its default window and omissions, so each judge can be given that one rendering in place of its own transcript.
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.model import ModelInfo, get_model_info, set_model_info
from inspect_ai.tool import ToolCall

from inspect_sentinel import (
    BeforeToolCall,
    LastTurns,
    StepPreprocessor,
    preprocess_step,
    step_as_str,
)

CAP = 100_000
"""Tokens a judge prompt is kept under."""

CHARS_PER_TOKEN = 2.5
"""Characters per token for estimating a prompt's size: fewer than English or code takes, so an estimate under the cap is usually under it. Dense text such as base64 can take fewer, so the usage files record each trajectory's largest prompt, and the ledger prices a trajectory whose prompt passed the cap at the higher rate."""

SHARED = StepPreprocessor[BeforeToolCall](
    input_window=LastTurns(20), omit_assistant_text=True, omit_reasoning=True
)
"""What `auto_mode()` shows by default: the last 20 turns, without the agent's text and reasoning."""


def bash_command(call: ToolCall) -> str:
    """A `bash` call's command: `command` in the cases' own tool, `cmd` in inspect_ai's, which ControlArena's settings use."""
    return str(call.arguments.get("command", call.arguments.get("cmd", "")))


def cap_judge_model(model: str, cap: int = CAP) -> None:
    """Give `model` an input budget of `cap` tokens in this process, so `Host.score` fits each prompt under it, cutting the oldest messages."""
    info = get_model_info(model) or ModelInfo()
    set_model_info(model, info.model_copy(update={"input_tokens": cap}))


def fit(entries: Sequence[str], fixed: int, cap: int = CAP) -> tuple[list[str], int]:
    """`entries` cut to fit a prompt of `cap` tokens around `fixed` characters of the rest of it: the oldest after the first (the task) are left out until the estimate fits, and the last (the call under review) is always kept. Returns the entries kept and how many were left out."""
    budget = int(cap * 0.95 * CHARS_PER_TOKEN) - fixed
    kept = list(entries)
    total = sum(len(e) + 1 for e in kept)
    left_out = 0
    while total > budget and len(kept) > 2:
        total -= len(kept.pop(1)) + 1
        left_out += 1
    return kept, left_out


async def shared_transcript(step: BeforeToolCall) -> str:
    """`step` rendered as `auto_mode()` is shown it by default (`SHARED`), without its nonce."""
    rendered, _ = await step_as_str(await preprocess_step(step, SHARED))
    return rendered
