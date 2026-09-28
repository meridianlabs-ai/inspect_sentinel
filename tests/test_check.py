from typing import Any, cast

import pytest
from inspect_ai.tool import ToolCall

from inspect_sentinel._check import check_decision_shape
from inspect_sentinel._final import Final, final
from inspect_sentinel._report import Action, Decision
from tests._fakes import after_step, before_step


def _call() -> ToolCall:
    return ToolCall(id="c1", function="bash", arguments={"cmd": "ls"})


def _decision(action: Action) -> Decision:
    return Decision(action=action, modified=_call() if action == "modify" else None)


@pytest.mark.parametrize(
    "action", ["continue", "modify", "reject", "terminate", "escalate"]
)
def test_every_action_is_legal_before_a_tool_call(action: Action) -> None:
    check_decision_shape(_decision(action), before_step())


@pytest.mark.parametrize("action", ["reject", "modify"])
def test_intervening_after_a_tool_call_is_a_bug(action: Action) -> None:
    with pytest.raises(ValueError, match=action):
        check_decision_shape(_decision(action), after_step())


@pytest.mark.parametrize("action", ["continue", "terminate", "escalate"])
def test_the_other_actions_are_legal_after_a_tool_call(action: Action) -> None:
    check_decision_shape(_decision(action), after_step())


def test_a_modify_must_carry_the_replacement() -> None:
    with pytest.raises(ValueError, match="modified"):
        check_decision_shape(Decision(action="modify"), before_step())


def test_only_a_modify_may_carry_a_replacement() -> None:
    with pytest.raises(ValueError, match="modified"):
        check_decision_shape(
            Decision(action="continue", modified=_call()), before_step()
        )


def test_a_modify_before_a_tool_call_replaces_the_call() -> None:
    decision = Decision.model_construct(action="modify", modified=cast(Any, "rm -rf /"))
    with pytest.raises(ValueError, match="ToolCall"):
        check_decision_shape(decision, before_step())


def test_final_raises_final_carrying_the_decision() -> None:
    decision = Decision.reject("a person said no")
    with pytest.raises(Final) as info:
        final(decision)
    assert info.value.decision is decision
