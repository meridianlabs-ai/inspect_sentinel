from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import inspect_ai
import pytest
from inspect_ai import Task
from inspect_ai.approval import (
    Approval,
    ApprovalDecision,
    ApprovalPolicy,
    Approver,
    approver,
    auto_approver,
    human_approver,
)
from inspect_ai.approval._human import acp as acp_module
from inspect_ai.approval._human import approver as approver_module
from inspect_ai.dataset import Sample
from inspect_ai.event import ApprovalEvent
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, ChatMessageTool, ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolCallView, tool

from inspect_sentinel import Sentinels, as_protocol, sequential
from inspect_sentinel._integration import config_from_sentinel, sentinel_from_config

# eval's `scanner` parameter is partially unknown to pyright without inspect_scout installed
eval_tasks = cast(Callable[..., list[EvalLog]], vars(inspect_ai)["eval"])


@tool
def addition() -> Tool:
    async def execute(x: int, y: int) -> str:
        """Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.
        """
        return str(x + y)

    return execute


@approver
def scripted(
    decision: ApprovalDecision = "approve",
    explanation: str | None = None,
    modify: bool = False,
) -> Approver:
    async def approve(
        message: str, call: ToolCall, view: ToolCallView, history: list[ChatMessage]
    ) -> Approval:
        modified = (
            ToolCall(id=call.id, function=call.function, arguments={"x": 2, "y": 2})
            if modify
            else None
        )
        return Approval(decision=decision, explanation=explanation, modified=modified)

    return approve


class Outcome(NamedTuple):
    tools: list[tuple[str, str | None]]
    approvals: list[tuple[str, str, str | None]]
    terminated: bool


def run(tmp_path: Path, **task_args: Any) -> EvalLog:
    outputs = [
        ModelOutput.for_tool_call(
            "mockllm/model",
            tool_name="addition",
            tool_arguments={"x": 1, "y": 1},
            content="Adding now.",
        ),
        ModelOutput.from_content("mockllm/model", content="done"),
    ]
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target="2")],
        solver=[use_tools(addition()), generate()],
        **task_args,
    )
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    [log] = eval_tasks(task, model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    return log


def outcome(log: EvalLog) -> Outcome:
    assert log.samples
    [sample] = log.samples
    return Outcome(
        tools=[
            (m.text, m.error.message if m.error else None)
            for m in sample.messages
            if isinstance(m, ChatMessageTool)
        ],
        approvals=[
            (e.approver, e.decision, e.explanation)
            for e in sample.events
            if isinstance(e, ApprovalEvent)
        ],
        terminated=sample.limit is not None and sample.limit.type == "operator",
    )


def assert_equivalent(
    tmp_path: Path, policies: list[ApprovalPolicy] | str, *sentinels: Sentinels
) -> Outcome:
    expected = outcome(run(tmp_path, approval=policies))
    for sentinel in sentinels:
        assert outcome(run(tmp_path, sentinel=sentinel)) == expected
    return expected


@pytest.mark.parametrize(
    ("make", "tools"),
    [
        (lambda: auto_approver(), [("2", None)]),
        (lambda: scripted("approve", "fine"), [("2", None)]),
        (lambda: scripted("modify", "use 2s", modify=True), [("4", None)]),
        (lambda: scripted("approve", "use 2s", modify=True), [("4", None)]),
        (lambda: scripted("modify", "nothing to change"), [("2", None)]),
        (lambda: scripted("reject", "no adding"), [("", "no adding")]),
        (lambda: scripted("reject"), [("", "Tool call not approved.")]),
        (lambda: scripted("terminate", "stop"), []),
    ],
)
def test_an_approver_decides_as_under_approval(
    tmp_path: Path, make: Any, tools: list[tuple[str, str | None]]
) -> None:
    expected = assert_equivalent(
        tmp_path,
        [ApprovalPolicy(make(), "*")],
        as_protocol(make()),
        as_protocol([ApprovalPolicy(make(), "*")]),
    )
    assert expected.tools == tools
    assert expected.terminated == (not tools)


@pytest.mark.parametrize(
    ("policies", "tools"),
    [
        (
            lambda: [
                ApprovalPolicy(scripted("escalate", "unsure"), "*"),
                ApprovalPolicy(scripted("reject", "not addition"), "addition"),
            ],
            [("", "not addition")],
        ),
        (
            lambda: [
                ApprovalPolicy(scripted("reject", "not bash"), "bash"),
                ApprovalPolicy(scripted("escalate", "unsure"), "add*"),
                ApprovalPolicy(auto_approver(), "*"),
            ],
            [("2", None)],
        ),
        (
            lambda: [ApprovalPolicy(scripted("escalate", "unsure"), "*")],
            [("", "No approval granted for tool addition")],
        ),
        (
            lambda: [ApprovalPolicy(auto_approver(), "bash")],
            [("", "No approvers registered for tool addition")],
        ),
        (
            lambda: [ApprovalPolicy(auto_approver(), "addition(x=2*")],
            [("", "No approvers registered for tool addition")],
        ),
        (
            lambda: [ApprovalPolicy(auto_approver(), ["bash", "addition(x=1*"])],
            [("2", None)],
        ),
    ],
)
def test_policies_match_and_fall_through_as_under_approval(
    tmp_path: Path, policies: Any, tools: list[tuple[str, str | None]]
) -> None:
    expected = assert_equivalent(tmp_path, policies(), as_protocol(policies()))
    assert expected.tools == tools


def test_an_escalating_approver_passes_the_step_on_in_a_sequential(
    tmp_path: Path,
) -> None:
    assert_equivalent(
        tmp_path,
        [
            ApprovalPolicy(scripted("escalate", "unsure"), "*"),
            ApprovalPolicy(scripted("reject", "no"), "*"),
        ],
        sequential(
            {
                "unsure": as_protocol(scripted("escalate", "unsure")),
                "no": as_protocol(scripted("reject", "no")),
            }
        ),
    )


def test_a_config_file_or_approver_name_decides_as_under_approval(
    tmp_path: Path,
) -> None:
    config = tmp_path / "approval.yaml"
    config.write_text(
        "approvers:\n"
        "  - name: scripted\n"
        "    tools: bash\n"
        "    decision: reject\n"
        "  - name: auto\n"
        "    tools: '*'\n"
    )
    for spec in [str(config), "auto"]:
        expected = assert_equivalent(
            tmp_path,
            spec,
            as_protocol(spec),
            sentinel_from_config({"name": "approval", "params": {"policies": spec}}),
        )
        assert expected.tools == [("2", None)]


class Panel:
    def __init__(self, decision: ApprovalDecision, explanation: str | None) -> None:
        self.decision: ApprovalDecision = decision
        self.explanation = explanation
        self.choices: list[list[str]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def acp(**kwargs: object) -> Approval | None:
            return None

        async def panel(
            message: str,
            call: ToolCall,
            view: ToolCallView,
            history: object,
            choices: list[str],
        ) -> Approval:
            self.choices.append(choices)
            return Approval(decision=self.decision, explanation=self.explanation)

        monkeypatch.setattr(acp_module, "request_human_approval_via_acp", acp)
        monkeypatch.setattr(approver_module, "panel_approval", panel)


@pytest.mark.parametrize(
    ("decision", "explanation", "tools"),
    [
        ("approve", None, [("2", None)]),
        (
            "reject",
            "Rejected by human approver.",
            [("", "Rejected by human approver.")],
        ),
        ("terminate", "Terminated by human approver.", []),
    ],
)
def test_a_human_approver_decides_as_under_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ApprovalDecision,
    explanation: str | None,
    tools: list[tuple[str, str | None]],
) -> None:
    panel = Panel(decision, explanation)
    panel.install(monkeypatch)
    expected = assert_equivalent(
        tmp_path,
        [ApprovalPolicy(human_approver(), "*")],
        as_protocol(human_approver()),
    )
    assert expected.tools == tools
    assert panel.choices == [["approve", "reject", "terminate"]] * 2


@pytest.mark.parametrize(
    "make",
    [
        lambda: as_protocol(auto_approver("reject")),
        lambda: as_protocol(scripted("reject", "no adding")),
        lambda: as_protocol(
            [
                ApprovalPolicy(scripted("escalate", "unsure"), "*"),
                ApprovalPolicy(scripted("modify", "use 2s", modify=True), ["add*"]),
            ]
        ),
    ],
)
def test_the_log_rebuilds_the_protocol(tmp_path: Path, make: Any) -> None:
    sentinel = make()
    log = run(tmp_path, sentinel=sentinel)
    assert log.eval.config.sentinel == config_from_sentinel(sentinel)
    assert log.eval.config.sentinel is not None
    rebuilt = sentinel_from_config(log.eval.config.sentinel)
    assert outcome(run(tmp_path, sentinel=rebuilt)) == outcome(log)


def test_an_unregistered_approver_is_a_configuration_error() -> None:
    async def approve(
        message: str, call: ToolCall, view: ToolCallView, history: list[ChatMessage]
    ) -> Approval:
        return Approval(decision="approve")

    with pytest.raises(TypeError, match="'approve' is not a configured approver"):
        as_protocol(approve)
    with pytest.raises(TypeError, match="'approve' is not a configured approver"):
        as_protocol([ApprovalPolicy(approve, "*")])


@pytest.mark.parametrize(
    ("policies", "match"),
    [
        ([], "at least one policy"),
        ([{"approver": auto_approver()}], r"policies\[0\].*'approver' and 'tools'"),
        ([{"name": "auto", "tools": "*"}], r"policies\[0\].*'approver' and 'tools'"),
    ],
)
def test_invalid_policies_are_a_configuration_error(
    policies: list[dict[str, Any]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        sentinel_from_config({"name": "approval", "params": {"policies": policies}})


@approver
def renaming() -> Approver:
    async def approve(
        message: str, call: ToolCall, view: ToolCallView, history: list[ChatMessage]
    ) -> Approval:
        return Approval(
            decision="modify",
            modified=ToolCall(
                id=call.id, function="subtract", arguments=call.arguments
            ),
        )

    return approve


def test_a_replacement_call_with_another_function_fails_the_sample(
    tmp_path: Path,
) -> None:
    task = Task(
        dataset=[Sample(input="What is 1 + 1?")],
        solver=[use_tools(addition()), generate()],
        sentinel=as_protocol(renaming()),
    )
    outputs = [
        ModelOutput.for_tool_call(
            "mockllm/model", tool_name="addition", tool_arguments={"x": 1, "y": 1}
        )
    ]
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    [log] = eval_tasks(task, model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "error"
    assert log.error is not None
    assert "may change only the call's arguments" in log.error.message
