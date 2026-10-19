from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.approval._approval import Approval, ApprovalDecision
from inspect_ai.approval._human import acp as acp_module
from inspect_ai.approval._human import approver as approver_module
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolCallContent, ToolCallView, tool

from inspect_sentinel import (
    AfterGenerate,
    BeforeToolCall,
    Context,
    Decision,
    Protocol,
    human,
    protocol,
    sequential,
)


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


def shout_viewer(call: ToolCall) -> ToolCallView:
    return ToolCallView(
        call=ToolCallContent(title="Shout", format="markdown", content="**{{text}}**")
    )


@tool(viewer=shout_viewer)
def shout() -> Tool:
    async def execute(text: str) -> str:
        """Shout some text.

        Args:
            text: Text to shout.
        """
        return text.upper()

    return execute


@protocol
def h_escalate_turn() -> Protocol:
    async def unsure(context: Context, step: AfterGenerate) -> Decision | None:
        if step.output.message.tool_calls:
            return Decision.escalate("two calls together, {{x}}")
        return Decision.proceed()

    return unsure


@protocol
def h_escalate() -> Protocol:
    async def unsure(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.escalate("adds {{x}} suspiciously")

    return unsure


class Panel:
    def __init__(
        self, decision: ApprovalDecision, explanation: str | None = None
    ) -> None:
        self.decision: ApprovalDecision = decision
        self.explanation = explanation
        self.views: list[ToolCallView] = []
        self.choices: list[list[str]] = []
        self.messages: list[str] = []
        self.histories: list[object] = []

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
            self.messages.append(message)
            self.histories.append(history)
            self.views.append(view)
            self.choices.append(choices)
            return Approval(decision=self.decision, explanation=self.explanation)

        monkeypatch.setattr(acp_module, "request_human_approval_via_acp", acp)
        monkeypatch.setattr(approver_module, "panel_approval", panel)

    def content(self) -> str:
        [view] = self.views
        assert view.call is not None
        return view.call.content


def run(sentinel: Any, **kwargs: Any) -> EvalLog:
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
        sentinel=sentinel,
    )
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    return eval(task, model=model, **kwargs)[0]


def tool_messages(log: EvalLog) -> list[ChatMessageTool]:
    assert log.samples
    return [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]


def escalated_to_human() -> Any:
    return sequential([h_escalate(), human(stages=["tool_call"])])


def test_an_escalation_asks_the_person_and_approve_runs_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = Panel("approve")
    panel.install(monkeypatch)

    log = run(escalated_to_human())

    assert log.status == "success", log.error
    assert panel.choices == [["approve", "reject", "terminate"]]
    assert panel.messages == ["Adding now."]
    content = panel.content()
    assert (
        "**Escalated by**\n\n```\n- h_escalate: adds { {x}} suspiciously\n```"
        in content
    )
    assert content.index("Escalated by") < content.index("addition(")
    [message] = tool_messages(log)
    assert message.error is None
    assert message.text == "2"


def test_after_a_call_the_person_sees_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = Panel("approve")
    panel.install(monkeypatch)

    log = run(human(stages=["tool_result"]))

    assert log.status == "success", log.error
    assert panel.choices == [["approve", "terminate"]]
    assert "**Result**\n\n```\n2\n```" in panel.content()


ADD = ToolCall(id="c1", function="addition", arguments={"x": 1, "y": 1})


SHOUT = ToolCall(id="c2", function="shout", arguments={"text": "hello"})


def turn_output(*calls: ToolCall, content: str = "Adding and shouting.") -> ModelOutput:
    message = ChatMessageAssistant(
        content=content, model="mockllm/model", tool_calls=list(calls) or None
    )
    return ModelOutput(
        model="mockllm/model",
        choices=[ChatCompletionChoice(message=message, stop_reason="tool_calls")],
    )


def run_turn(sentinel: Any) -> EvalLog:
    outputs = [
        turn_output(ADD, SHOUT),
        ModelOutput.from_content("mockllm/model", content="done"),
    ]
    task = Task(
        dataset=[Sample(input="Add 1 and 1, and shout hello.")],
        solver=[use_tools(addition(), shout()), generate()],
        sentinel=sentinel,
    )
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    return eval(task, model=model)[0]


def turn_escalated_to_human() -> Any:
    return sequential([h_escalate_turn(), human(stages=["model_output"])])


def test_an_escalated_turn_shows_every_call_and_approve_runs_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = Panel("approve")
    panel.install(monkeypatch)

    log = run_turn(turn_escalated_to_human())

    assert log.status == "success", log.error
    assert panel.choices == [["approve", "terminate"]]
    assert panel.messages == ["Adding and shouting."]
    content = panel.content()
    escalated = (
        "**Escalated by**\n\n```\n- h_escalate_turn: two calls together, { {x}}\n```"
    )
    assert content.startswith(escalated)
    # each call as the transcript shows it: the default view, and the tool's
    # own viewer with its placeholders filled in
    assert content.index("addition(x=1, y=1)") < content.index("**hello**")
    assert [m.text for m in tool_messages(log)] == ["2", "HELLO"]
