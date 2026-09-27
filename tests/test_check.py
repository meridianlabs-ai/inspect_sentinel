from typing import Any, cast

import pytest
from inspect_ai.tool import ToolCall

from inspect_sentinel._check import apply_binding_floor, check_decision_shape
from inspect_sentinel._report import Action, Decision, Reported
from tests._fakes import after_step, before_step


def _call(cmd: str = "ls") -> ToolCall:
    return ToolCall(id="c1", function="bash", arguments={"cmd": cmd})


def _decision(action: Action, binding: bool = False, cmd: str = "ls") -> Decision:
    return Decision(
        action=action,
        binding=binding,
        modified=_call(cmd) if action == "modify" else None,
    )


def _child(
    action: Action,
    binding: bool,
    name: str = "human",
    cmd: str = "ls",
    explanation: str | None = None,
) -> Reported[Decision]:
    decision = _decision(action, binding, cmd)
    return Reported(
        name=name,
        path=name,
        report=decision.model_copy(update={"explanation": explanation}),
    )


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


@pytest.mark.parametrize(
    "children",
    [[], [_child("reject", binding=False)], [_child("escalate", binding=True)]],
)
def test_no_binding_decision_imposes_no_floor(
    children: list[Reported[Decision]],
) -> None:
    decision = _decision("continue")
    assert apply_binding_floor(decision, children) == (decision, None)
    assert apply_binding_floor(None, children) == (None, None)


@pytest.mark.parametrize(
    ("action", "binding"), [("reject", False), ("terminate", False), ("reject", True)]
)
def test_a_decision_at_or_above_the_floor_stands_with_its_own_flag(
    action: Action, binding: bool
) -> None:
    decision = _decision(action, binding)
    resolved, overrode = apply_binding_floor(decision, [_child("reject", binding=True)])
    assert resolved is decision and overrode is None


@pytest.mark.parametrize(
    "decision", [None, _decision("continue"), _decision("escalate")]
)
def test_a_decision_below_the_floor_is_replaced_by_it(
    decision: Decision | None,
) -> None:
    child = _child("reject", binding=True, explanation="a person said no")
    resolved, overrode = apply_binding_floor(decision, [child])
    assert resolved is not None
    assert resolved.action == "reject" and resolved.binding is True
    expected = decision.action if decision is not None else "abstention"
    assert resolved.explanation == f"a person said no (overrides {expected})"
    assert overrode is decision


def test_the_floor_explanation_falls_back_to_its_action() -> None:
    resolved, _ = apply_binding_floor(
        _decision("continue"), [_child("reject", binding=True)]
    )
    assert (
        resolved is not None and resolved.explanation == "reject (overrides continue)"
    )


def test_the_floor_is_the_strongest_binding_child() -> None:
    children = [
        _child("continue", binding=True, name="a"),
        _child("terminate", binding=True, name="b"),
    ]
    resolved, overrode = apply_binding_floor(_decision("reject"), children)
    assert resolved is not None and resolved.action == "terminate"
    assert overrode is not None and overrode.action == "reject"


def test_a_tie_at_the_floor_takes_the_first_child() -> None:
    children = [
        _child("reject", binding=True, name="a", explanation="a"),
        _child("reject", binding=True, name="b", explanation="b"),
    ]
    resolved, _ = apply_binding_floor(None, children)
    assert resolved is not None and resolved.explanation == "a (overrides abstention)"


def test_a_modify_with_another_replacement_is_below_the_floor() -> None:
    child = _child("modify", binding=True, cmd="echo safe")
    resolved, overrode = apply_binding_floor(_decision("modify", cmd="rm"), [child])
    assert resolved is not None and resolved.modified == child.report.modified
    assert overrode is not None and overrode.modified == _call("rm")


def test_passing_the_binding_replacement_on_stands() -> None:
    child = _child("modify", binding=True)
    decision = child.report.model_copy(update={"binding": False})
    assert apply_binding_floor(decision, [child]) == (decision, None)


@pytest.mark.parametrize("action", ["reject", "terminate"])
def test_a_stronger_action_over_a_binding_modify_stands(action: Action) -> None:
    decision = _decision(action)
    assert apply_binding_floor(decision, [_child("modify", binding=True)]) == (
        decision,
        None,
    )


def test_binding_modifies_that_disagree_resolve_to_a_binding_reject() -> None:
    children = [
        _child("modify", binding=True, name="alice", cmd="echo a"),
        _child("modify", binding=True, name="bob", cmd="echo b"),
    ]
    resolved, overrode = apply_binding_floor(
        _decision("modify", cmd="echo a"), children
    )
    assert resolved is not None
    assert resolved.action == "reject" and resolved.binding is True
    assert resolved.modified is None
    assert resolved.explanation == (
        "binding replacements disagree: alice, bob (overrides modify)"
    )
    assert overrode is not None and overrode.action == "modify"
    stronger = _decision("terminate")
    assert apply_binding_floor(stronger, children) == (stronger, None)
