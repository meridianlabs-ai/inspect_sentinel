from typing import get_args

import pytest
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
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
    Truncation,
)


def _call() -> ToolCall:
    return ToolCall(id="c1", function="bash", arguments={"cmd": "curl x"})


def _input() -> list[ChatMessage]:
    return [
        ChatMessageSystem(content="be careful"),
        ChatMessageUser(content="do it"),
        ChatMessageAssistant(content="ok"),
    ]


def _history() -> list[ChatMessage]:
    return [
        ChatMessageUser(content="do it"),
        ChatMessageAssistant(content="ok"),
    ]


def test_before_tool_call_defaults_to_no_escalations() -> None:
    model_input, history = _input(), _history()
    step = BeforeToolCall(
        conversation="conv",
        message="ok",
        message_id="msg_1",
        call=_call(),
        view=ToolCallView(),
        input=model_input,
        history=history,
    )
    assert step.escalations == ()
    assert step.input is model_input
    assert step.history is history


def test_before_tool_call_carries_escalations() -> None:
    escalation = Reported(
        name="guard",
        path="attempt/guard",
        report=Decision.escalate("unsure"),
        function="decide",
    )
    step = BeforeToolCall(
        conversation="conv",
        message="ok",
        message_id="msg_1",
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
        message_id="msg_1",
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
        message_id="msg_1",
        call=_call(),
        result=ChatMessageTool(content="a b", tool_call_id="c1"),
        output=output,
        view=ToolCallView(),
        input=_input(),
        history=_history(),
    )
    assert step.output == output


def test_tool_steps_default_to_no_tools_and_no_truncation() -> None:
    before = BeforeToolCall(
        conversation="conv",
        message="ok",
        message_id="msg_1",
        call=_call(),
        view=ToolCallView(),
        input=None,
        history=_history(),
    )
    after = AfterToolCall(
        conversation="conv",
        message="ok",
        message_id="msg_1",
        call=_call(),
        result=ChatMessageTool(content="out", tool_call_id="c1"),
        output="out",
        view=ToolCallView(),
        input=None,
        history=_history(),
    )
    assert before.tools is None
    assert after.tools is None
    assert after.truncated is None


@pytest.mark.parametrize(
    "truncated,cut",
    [
        (Truncation(original_bytes=100, limit_bytes=10), True),
        (False, False),
        (None, False),
    ],
)
def test_only_a_cut_is_truthy(truncated: Truncation | bool | None, cut: bool) -> None:
    assert bool(truncated) is cut


def test_step_union_covers_every_stage() -> None:
    assert set(get_args(Step)) == {
        BeforeGenerate,
        AfterGenerate,
        BeforeToolCall,
        AfterToolCall,
    }
