from typing import Any, cast

import pytest
from inspect_ai.tool import ToolCall

from inspect_sentinel._check import validate_decision
from inspect_sentinel._report import Action, Decision, Reported
from tests._fakes import after_step, before_step


def _call() -> ToolCall:
    return ToolCall(id="c1", function="bash", arguments={"cmd": "ls"})


def _decision(action: Action) -> Decision:
    return Decision(action=action, modified=_call() if action == "modify" else None)


def _child(action: Action, binding: bool, name: str = "human") -> Reported[Decision]:
    return Reported(
        name=name,
        path=name,
        report=Decision(
            action=action,
            binding=binding,
            modified=_call() if action == "modify" else None,
        ),
    )


@pytest.mark.parametrize(
    "action", ["continue", "modify", "reject", "terminate", "escalate"]
)
def test_every_action_is_legal_before_a_tool_call(action: Action) -> None:
    validate_decision(_decision(action), before_step())


@pytest.mark.parametrize("action", ["reject", "modify"])
def test_intervening_after_a_tool_call_is_a_bug(action: Action) -> None:
    with pytest.raises(ValueError, match=action):
        validate_decision(_decision(action), after_step())


@pytest.mark.parametrize("action", ["continue", "terminate", "escalate"])
def test_the_other_actions_are_legal_after_a_tool_call(action: Action) -> None:
    validate_decision(_decision(action), after_step())


def test_a_modify_must_carry_the_replacement() -> None:
    with pytest.raises(ValueError, match="modified"):
        validate_decision(Decision(action="modify"), before_step())


def test_only_a_modify_may_carry_a_replacement() -> None:
    with pytest.raises(ValueError, match="modified"):
        validate_decision(Decision(action="continue", modified=_call()), before_step())


def test_a_modify_before_a_tool_call_replaces_the_call() -> None:
    decision = Decision.model_construct(action="modify", modified=cast(Any, "rm -rf /"))
    with pytest.raises(ValueError, match="ToolCall"):
        validate_decision(decision, before_step())


def test_a_decision_may_not_weaken_a_binding_child() -> None:
    with pytest.raises(ValueError, match="binding") as info:
        validate_decision(
            _decision("continue"), before_step(), [_child("reject", binding=True)]
        )
    assert "human" in str(info.value)


def test_abstaining_after_a_binding_child_is_a_bug() -> None:
    with pytest.raises(ValueError, match="binding") as info:
        validate_decision(None, before_step(), [_child("reject", binding=True)])
    assert "human" in str(info.value)


def test_abstaining_is_legal_when_no_child_was_binding() -> None:
    validate_decision(None, before_step(), [_child("reject", binding=False)])


@pytest.mark.parametrize("action", ["reject", "terminate"])
def test_a_decision_at_or_above_the_binding_floor_stands(action: Action) -> None:
    validate_decision(
        _decision(action), before_step(), [_child("reject", binding=True)]
    )


def test_a_binding_escalate_imposes_no_floor() -> None:
    validate_decision(
        _decision("continue"), before_step(), [_child("escalate", binding=True)]
    )


def test_an_advisory_child_imposes_no_floor() -> None:
    validate_decision(
        _decision("continue"), before_step(), [_child("reject", binding=False)]
    )


def test_a_modify_must_carry_the_binding_childs_replacement() -> None:
    other = Decision(
        action="modify",
        modified=ToolCall(id="c1", function="bash", arguments={"cmd": "rm -rf /"}),
    )
    with pytest.raises(ValueError, match="replacement") as info:
        validate_decision(other, before_step(), [_child("modify", binding=True)])
    assert "human" in str(info.value)


def test_passing_the_binding_replacement_on_stands() -> None:
    child = _child("modify", binding=True)
    validate_decision(child.report.model_copy(), before_step(), [child])


@pytest.mark.parametrize("action", ["reject", "terminate"])
def test_a_stronger_action_over_a_binding_modify_stands(action: Action) -> None:
    validate_decision(
        _decision(action), before_step(), [_child("modify", binding=True)]
    )


def test_the_floor_is_the_strongest_binding_child() -> None:
    children = [
        _child("continue", binding=True, name="a"),
        _child("terminate", binding=True, name="b"),
    ]
    with pytest.raises(ValueError, match="binding") as info:
        validate_decision(_decision("reject"), before_step(), children)
    assert "'b'" in str(info.value)
