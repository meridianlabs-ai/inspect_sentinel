from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from inspect_ai.tool import ToolCall, ToolCallContent
from pydantic import ValidationError

from inspect_sentinel._report import Decision, Observation, Report, Reported

if TYPE_CHECKING:
    _widens: Reported[Report] = Reported("n", "p", Observation.score(0.0))


def test_observation_score_carries_scalar_and_dict_suspicion() -> None:
    assert Observation.score(0.7).suspicion == 0.7
    assert Observation.score({"sabotage": 0.9}, "why").suspicion == {"sabotage": 0.9}
    assert Observation.score(0.7, "why").explanation == "why"


def test_observation_flag_is_full_suspicion() -> None:
    flagged = Observation.flag("definite")
    assert flagged.suspicion == 1.0
    assert flagged.explanation == "definite"


@pytest.mark.parametrize(
    ("make", "action"),
    [
        (Decision.clear, "continue"),
        (Decision.reject, "reject"),
        (Decision.terminate, "terminate"),
        (Decision.escalate, "escalate"),
    ],
)
def test_decision_constructors_set_action(
    make: Callable[[str], Decision], action: str
) -> None:
    decision = make("because")
    assert decision.action == action
    assert decision.explanation == "because"
    assert decision.audit is False
    assert decision.authoritative is False
    assert decision.modified is None


def test_reports_round_trip_through_json() -> None:
    observation = Observation(
        suspicion={"a": 0.1, "b": 0.8}, explanation="two dims", metadata={"k": 1}
    )
    assert Observation.model_validate_json(observation.model_dump_json()) == observation

    call = ToolCall(
        id="c1",
        function="bash",
        arguments={"cmd": "ls"},
        parse_error="bad json",
        view=ToolCallContent(title="bash", format="markdown", content="`ls`"),
        type="custom",
    )
    decision = Decision(action="modify", modified=call, audit=True, authoritative=True)
    restored = Decision.model_validate_json(decision.model_dump_json())
    assert restored == decision
    assert restored.modified == call
    assert Decision.model_validate(decision.model_dump()) == decision


@pytest.mark.parametrize("bad", ["high", {"a": "high"}, [0.1], None])
def test_observation_rejects_non_numeric_suspicion(bad: object) -> None:
    with pytest.raises(ValidationError):
        Observation.model_validate({"suspicion": bad})


def test_reported_attaches_identity() -> None:
    reported = Reported(
        name="judge", path="attempt/judge", report=Observation.score(0.2)
    )
    assert reported.name == "judge"
    assert reported.path == "attempt/judge"
    assert reported.report.suspicion == 0.2
