from datetime import date

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
    ContentToolUse,
)
from inspect_ai.tool import ToolCall, ToolCallError, ToolCallView

from inspect_sentinel import (
    AfterToolCall,
    BeforeToolCall,
    call_as_str,
    last_turns,
    message_as_str,
    messages_as_str,
    step_as_str,
)


def _call(function: str = "bash", **arguments: object) -> ToolCall:
    return ToolCall(id=f"{function}-1", function=function, arguments=arguments)


def _assistant(text: str, *calls: ToolCall) -> ChatMessageAssistant:
    return ChatMessageAssistant(content=text, tool_calls=list(calls) or None)


def _tool(call: ToolCall, text: str) -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call.id, function=call.function)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (
            _call(command="ls -la /etc", timeout=30),
            'bash(command="ls -la /etc", timeout=30)',
        ),
        (_call("submit"), "submit()"),
        (
            _call(command='echo "hi" \\ there'),
            'bash(command="echo \\"hi\\" \\\\ there")',
        ),
        (
            _call("python", code="import os\nprint(1)"),
            'python(code="import os\nprint(1)")',
        ),
        (
            _call("edit", lines=[1, 2], options={"dry": True, "name": "é"}, x=None),
            'edit(lines=[1, 2], options={"dry": true, "name": "é"}, x=null)',
        ),
    ],
)
def test_call_as_str(call: ToolCall, expected: str) -> None:
    assert call_as_str(call) == expected


def test_call_as_str_unusual_arguments() -> None:
    assert call_as_str(_call(when=date(2026, 10, 3))) == 'bash(when="2026-10-03")'
    malformed = ToolCall(
        id="c1", function="bash", arguments={}, parse_error="invalid JSON"
    )
    assert call_as_str(malformed) == "bash() [parse error: invalid JSON]"


LS = _call(command="ls")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (ChatMessageSystem(content="be careful"), "SYSTEM:\nbe careful\n"),
        (ChatMessageUser(content="do it"), "USER:\ndo it\n"),
        (
            _assistant("looking", LS, _call("python", code="1")),
            'ASSISTANT:\nlooking\n\nTool Call: bash(command="ls")\n\nTool Call: python(code="1")\n',
        ),
        (_assistant("", LS), 'ASSISTANT:\nTool Call: bash(command="ls")\n'),
        (_tool(LS, "a.txt"), "TOOL:\na.txt\n"),
        (
            ChatMessageTool(
                content="",
                tool_call_id=LS.id,
                function="bash",
                error=ToolCallError(type="timeout", message="timed out"),
            ),
            "TOOL:\nError in tool call 'bash':\ntimed out\n",
        ),
        (
            ChatMessageUser(
                content=[
                    ContentText(text="what is this?"),
                    ContentImage(image="data:image/png;base64,AAAA"),
                ]
            ),
            "USER:\nwhat is this?\n\n<image />\n",
        ),
        (
            ChatMessageAssistant(
                content=[
                    ContentReasoning(reasoning="plan"),
                    ContentReasoning(reasoning="", summary="short", redacted=True),
                    ContentReasoning(reasoning="", redacted=True),
                    ContentText(text="done"),
                ]
            ),
            "ASSISTANT:\n<thinking>plan</thinking>\n\n<thinking_summary>short</thinking_summary>\n\n<thinking_redacted/>\n\ndone\n",
        ),
        (
            ChatMessageAssistant(
                content=[
                    ContentToolUse(
                        tool_type="web_search",
                        id="w1",
                        name="search",
                        arguments='{"q": "x"}',
                        result="found",
                    ),
                ]
            ),
            'ASSISTANT:\nTool Use: search({"q": "x"}) -> found\n',
        ),
    ],
)
def test_message_as_str(message: ChatMessage, expected: str) -> None:
    assert message_as_str(message) == expected


def test_message_as_str_exclusions() -> None:
    message = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning="plan"),
            ContentToolUse(
                tool_type="web_search", id="w1", name="search", arguments="", result=""
            ),
            ContentText(text="running"),
        ],
        tool_calls=[LS],
    )
    assert (
        message_as_str(message, exclude_reasoning=True, exclude_tool_usage=True)
        == "ASSISTANT:\nrunning\n"
    )
    assert message_as_str(_tool(LS, "a.txt"), exclude_tool_usage=True) == (
        "TOOL:\na.txt\n"
    )


CONVERSATION: list[ChatMessage] = [
    ChatMessageSystem(content="be careful"),
    ChatMessageUser(content="list files"),
    _assistant("looking", LS),
    _tool(LS, "a.txt"),
]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            {},
            'USER:\nlist files\n\nASSISTANT:\nlooking\n\nTool Call: bash(command="ls")\n\nTOOL:\na.txt\n',
        ),
        (
            {"exclude_system": False},
            'SYSTEM:\nbe careful\n\nUSER:\nlist files\n\nASSISTANT:\nlooking\n\nTool Call: bash(command="ls")\n\nTOOL:\na.txt\n',
        ),
        (
            {"exclude_tool_usage": True},
            "USER:\nlist files\n\nASSISTANT:\nlooking\n",
        ),
    ],
)
def test_messages_as_str(options: dict[str, bool], expected: str) -> None:
    assert messages_as_str(CONVERSATION, **options) == expected


def test_messages_as_str_empty() -> None:
    assert messages_as_str([]) == ""
    assert messages_as_str([ChatMessageSystem(content="be careful")]) == ""


CAT = _call(command="cat a.txt")
WC = _call("python", code="len(open('a.txt').read())")
TURNS: list[ChatMessage] = [
    ChatMessageSystem(content="be careful"),
    ChatMessageUser(content="list files"),
    _assistant("looking", LS),
    _tool(LS, "a.txt"),
    _assistant("reading both ways", CAT, WC),
    _tool(CAT, "hello"),
    _tool(WC, "5"),
    ChatMessageUser(content="keep going"),
    _assistant("done"),
]


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, []),
        (1, TURNS[8:]),
        (2, TURNS[4:]),
        (3, TURNS),
        (10, TURNS),
    ],
)
def test_last_turns(n: int, expected: list[ChatMessage]) -> None:
    assert last_turns(TURNS, n) == expected


def test_last_turns_keeps_pending_calls_whole() -> None:
    pending: list[ChatMessage] = [*TURNS[:4], _assistant("reading", CAT, WC)]
    assert last_turns(pending, 1) == pending[4:]


def test_last_turns_rejects_negative() -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        last_turns(TURNS, -1)


def _before(input: list[ChatMessage]) -> BeforeToolCall:
    return BeforeToolCall(
        conversation="conv",
        message="reading",
        call=CAT,
        view=ToolCallView(),
        input=input,
        history=input,
    )


def test_step_as_str_before_tool_call() -> None:
    assert step_as_str(_before(CONVERSATION)) == (
        "USER:\nlist files\n\n"
        'ASSISTANT:\nlooking\n\nTool Call: bash(command="ls")\n\n'
        "TOOL:\na.txt\n\n"
        'ASSISTANT:\nreading\n\nTool Call: bash(command="cat a.txt")\n'
    )


def test_step_as_str_after_tool_call() -> None:
    step = AfterToolCall(
        conversation="conv",
        message="reading",
        call=CAT,
        result=_tool(CAT, "hello"),
        output="hello",
        view=ToolCallView(),
        input=CONVERSATION,
        history=CONVERSATION,
    )
    assert step_as_str(step, turns=0) == (
        'ASSISTANT:\nreading\n\nTool Call: bash(command="cat a.txt")\n\nTOOL:\nhello\n'
    )


def test_step_as_str_turns() -> None:
    assert step_as_str(_before(TURNS[:7]), turns=1) == (
        "ASSISTANT:\nreading both ways\n\n"
        'Tool Call: bash(command="cat a.txt")\n\n'
        "Tool Call: python(code=\"len(open('a.txt').read())\")\n\n"
        "TOOL:\nhello\n\n"
        "TOOL:\n5\n\n"
        'ASSISTANT:\nreading\n\nTool Call: bash(command="cat a.txt")\n'
    )
