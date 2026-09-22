from functools import cache
from typing import get_args

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallView, ToolResult

from inspect_sentinel._report import Decision, Reported
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step


def _call() -> ToolCall:
    return ToolCall(id="c1", function="bash", arguments={"cmd": "curl x"})


@cache
def _input() -> list[ChatMessage]:
    return [
        ChatMessageSystem(content="be careful"),
        ChatMessageUser(content="do it"),
        ChatMessageAssistant(content="ok"),
    ]


@cache
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
        input=_input(),
        history=_history(),
    )
    assert step.escalations == ()
    assert step.input == _input()
    assert step.history == _history()
    assert step.input != step.history


def test_before_tool_call_carries_escalations() -> None:
    escalation = Reported(
        name="guard", path="attempt/guard", report=Decision.escalate("unsure")
    )
    step = BeforeToolCall(
        conversation="conv",
        message="ok",
        call=_call(),
        view=ToolCallView(),
        input=_input(),
        history=_history(),
        escalations=(escalation,),
    )
    assert step.escalations == (escalation,)


def test_after_tool_call_holds_result_and_untruncated_output() -> None:
    step = AfterToolCall(
        conversation="conv",
        message="ok",
        call=_call(),
        result=ChatMessageTool(content="short", tool_call_id="c1"),
        output="the full untruncated output",
        view=ToolCallView(),
        input=_input(),
        history=_history(),
    )
    assert step.result.content == "short"
    assert step.output == "the full untruncated output"


def test_after_tool_call_output_may_be_structured_content() -> None:
    output: ToolResult = [ContentText(text="a"), ContentText(text="b")]
    step = AfterToolCall(
        conversation="conv",
        message="ok",
        call=_call(),
        result=ChatMessageTool(content="a b", tool_call_id="c1"),
        output=output,
        view=ToolCallView(),
        input=_input(),
        history=_history(),
    )
    assert step.output == output


def test_step_union_covers_tool_stages() -> None:
    assert set(get_args(Step)) == {BeforeToolCall, AfterToolCall}
