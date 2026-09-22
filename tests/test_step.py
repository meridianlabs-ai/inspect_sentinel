from typing import get_args

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall, ToolCallView

from inspect_sentinel._report import Decision, Reported
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Stage, Step


def _call() -> ToolCall:
    return ToolCall(id="c1", function="bash", arguments={"cmd": "curl x"})


def _history() -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        ChatMessageUser(content="do it"),
        ChatMessageAssistant(content="ok"),
    ]
    return messages


def test_before_tool_call_defaults_to_no_escalations() -> None:
    step = BeforeToolCall(
        conversation="conv",
        message="ok",
        call=_call(),
        view=ToolCallView(),
        input=_history(),
        history=_history(),
    )
    assert step.escalations == ()


def test_before_tool_call_carries_escalations() -> None:
    escalation = Reported(
        name="guard", path="attempt/guard", report=Decision.escalate("unsure")
    )
    step = BeforeToolCall(
        conversation="conv",
        message="ok",
        call=_call(),
        view=ToolCallView(),
        input=_history(),
        history=_history(),
        escalations=(escalation,),
    )
    assert step.escalations[0].report.action == "escalate"


def test_after_tool_call_holds_result_and_untruncated_output() -> None:
    step = AfterToolCall(
        conversation="conv",
        message="ok",
        call=_call(),
        result=ChatMessageTool(content="short", tool_call_id="c1"),
        output="the full untruncated output",
        view=ToolCallView(),
        input=_history(),
        history=_history(),
    )
    assert step.result.text == "short"
    assert step.output == "the full untruncated output"


def test_step_union_and_stage_literal_cover_tool_stages() -> None:
    assert set(get_args(Step)) == {BeforeToolCall, AfterToolCall}
    assert set(get_args(Stage)) == {"tool_call", "tool_result"}
