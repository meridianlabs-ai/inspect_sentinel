from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from inspect_sentinel import HumanAnswer, ReplayStep, Step
from tests._fakes import (
    after_generate_step,
    after_step,
    before_generate_step,
    before_step,
)

STEPS: list[tuple[str, Step]] = [
    ("model_input", before_generate_step()),
    ("model_output", after_generate_step()),
    ("tool_call", before_step()),
    ("tool_result", after_step()),
]


@pytest.mark.parametrize(("stage", "step"), STEPS)
def test_a_replay_step_holds_its_step_and_no_answers_by_default(
    stage: Any, step: Step
) -> None:
    replayed = ReplayStep(
        id="s1", stage=stage, step=step, segment=(step.conversation, 0)
    )
    assert replayed.step is step
    assert dict(replayed.recorded_answers) == {}


def test_recorded_answers_are_a_read_only_copy() -> None:
    answers = {"attempt/human": HumanAnswer("approve")}
    step = after_step()
    replayed = ReplayStep(
        id="call_1",
        stage="tool_result",
        step=step,
        segment=(step.conversation, 0),
        recorded_answers=answers,
    )
    answers["attempt/human"] = HumanAnswer("reject")
    assert dict(replayed.recorded_answers) == {"attempt/human": HumanAnswer("approve")}
    with pytest.raises(TypeError):
        replayed.recorded_answers["other"] = HumanAnswer("approve")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        replayed.recorded_answers = {}  # type: ignore[misc]


@pytest.mark.parametrize(
    ("stage", "step", "segment", "error"),
    [
        (
            "tool_call",
            after_step(),
            ("conversation", 0),
            "whose stage is 'tool_result'",
        ),
        ("model_input", after_generate_step(), ("c", 0), "AfterGenerate"),
        ("tool_result", after_step(), ("other", 0), "conversation 'conversation'"),
    ],
)
def test_a_replay_step_must_agree_with_its_step(
    stage: Any, step: Step, segment: tuple[str, int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ReplayStep(id="s1", stage=stage, step=step, segment=segment)
