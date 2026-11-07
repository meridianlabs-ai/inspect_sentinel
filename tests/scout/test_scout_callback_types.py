# pyright: reportUnnecessaryTypeIgnoreComment=true

# `as_scanner(steps=, human=)` takes any module-level function, and a wrongly
# shaped one is a type error at the call. `make typecheck` checks this file:
# each call marked with an ignore must be an error, since an ignore with
# nothing to ignore is one here, and every other call must not be.

from __future__ import annotations

from collections.abc import Sequence

import pandas  # pyright: ignore[reportMissingTypeStubs]
import pytest
from inspect_scout import Scanner, Transcript

from inspect_sentinel import Escalations, HumanAnswer, HumanStandIn, ReplayStep, human
from inspect_sentinel.scout import StepSelector, StepTable, as_scanner, replay_check


def selects(transcript_id: str, step: ReplayStep) -> bool:
    return True


async def answers(
    step: ReplayStep, escalations: Escalations, choices: Sequence[str]
) -> HumanAnswer:
    return HumanAnswer("approve")


def selects_one_argument(step: ReplayStep) -> bool:
    return True


def selects_with_text(transcript_id: str, step: ReplayStep) -> str:
    return "yes"


async def selects_later(transcript_id: str, step: ReplayStep) -> bool:
    return True


def answers_now(
    step: ReplayStep, escalations: Escalations, choices: Sequence[str]
) -> HumanAnswer:
    return HumanAnswer("approve")


async def answers_with_text(
    step: ReplayStep, escalations: Escalations, choices: Sequence[str]
) -> str:
    return "approve"


async def answers_two_arguments(
    step: ReplayStep, escalations: Escalations
) -> HumanAnswer:
    return HumanAnswer("approve")


async def answers_from_a_list(
    step: ReplayStep, escalations: Escalations, choices: list[str]
) -> HumanAnswer:
    return HumanAnswer("approve")


_HERE = "tests.scout.test_scout_callback_types"


@pytest.mark.parametrize(
    "steps",
    [
        selects,
        f"{_HERE}:selects",
        pandas.DataFrame([{"transcript_id": "t", "step_id": "s"}]),
        [("t", "s")],
        [{"transcript_id": "t", "step_id": "s", "stage": None}],
        None,
    ],
)
def test_a_scan_takes_any_table_or_module_level_selector(
    steps: StepTable | StepSelector | str | None,
) -> None:
    assert as_scanner(replay_check(), steps=steps)


@pytest.mark.parametrize(
    "answer", [answers, f"{_HERE}:answers", "approve", "reject", "terminate"]
)
def test_a_scan_takes_a_fixed_answer_or_module_level_stand_in(
    answer: HumanStandIn | str,
) -> None:
    assert as_scanner(human(stages=["tool_call"]), human=answer)


def wrongly_shaped() -> list[Scanner[Transcript]]:
    # never run: each call is a type error, and would fail when the scan runs
    asking = human(stages=["tool_call"])
    return [
        as_scanner(replay_check(), steps=selects_one_argument),  # pyright: ignore[reportArgumentType]
        as_scanner(replay_check(), steps=selects_with_text),  # pyright: ignore[reportArgumentType]
        as_scanner(replay_check(), steps=selects_later),  # pyright: ignore[reportArgumentType]
        as_scanner(replay_check(), steps=answers),  # pyright: ignore[reportArgumentType]
        as_scanner(asking, human=answers_now),  # pyright: ignore[reportArgumentType]
        as_scanner(asking, human=answers_with_text),  # pyright: ignore[reportArgumentType]
        as_scanner(asking, human=answers_two_arguments),  # pyright: ignore[reportArgumentType]
        as_scanner(asking, human=answers_from_a_list),  # pyright: ignore[reportArgumentType]
        as_scanner(asking, human=selects),  # pyright: ignore[reportArgumentType]
    ]
