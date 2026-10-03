from dataclasses import replace
from typing import Any

import anyio
import pytest
from inspect_ai.approval import Approval, ApprovalPolicy, Approver, approver
from inspect_ai.approval._policy import policy_approver
from inspect_ai.model import ChatMessage, ChatMessageTool, ContentImage, ContentText
from inspect_ai.tool import ToolCall, ToolCallError, ToolCallView

from inspect_sentinel import (
    AfterToolCall,
    after_tool_call,
    before_tool_call,
    call_text,
    find_words,
    result_text,
    tool_matches,
)


def _call(function: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", function=function, arguments=arguments)


@pytest.mark.parametrize(
    "call, expected",
    [
        (_call("bash", command="ls /work"), "ls /work"),
        (_call("python", code="print(1)"), "print(1)"),
        (_call("code_execution", code="1 + 1"), "1 + 1"),
        (_call("bash_session", action="type_submit", input="ls"), "ls"),
        (_call("bash_session", action="read"), ""),
        (
            _call(
                "text_editor",
                command="str_replace",
                new_str="b",
                path="/w/f.py",
                old_str="a",
            ),
            "/w/f.py\na\nb",
        ),
        (
            _call("text_editor", command="create", path="/w/f", file_text="x"),
            "/w/f\nx",
        ),
        (
            _call(
                "text_editor",
                command="insert",
                path="/w/f",
                insert_line=2,
                insert_text="y",
            ),
            "/w/f\ny",
        ),
        (
            _call("memory", command="rename", old_path="/m/a", new_path="/m/b"),
            "/m/a\n/m/b",
        ),
        (_call("read_file", file_path="/w/f", offset=3), "/w/f"),
        (_call("list_files", path="/w", depth=1), "/w"),
        (_call("grep", pattern="TODO", path="/w", glob="*.py"), "TODO\n/w\n*.py"),
        (_call("skill", command="pdf"), "pdf"),
        (_call("think", thought="hmm"), "hmm"),
        (_call("web_search", query="weather"), "weather"),
        (_call("web_browser_go", url="https://x.org"), "https://x.org"),
        (_call("web_browser_type", element_id=3, text="hi"), "hi"),
        (_call("web_browser_type_submit", element_id=3, text="hi"), "hi"),
        (_call("computer", action="type", text="hello"), "hello"),
        (_call("computer", action="left_click", coordinate=[1, 2]), ""),
        (
            _call(
                "computer",
                actions=[
                    {"action": "left_click", "coordinate": [1, 2]},
                    {"action": "type", "text": "rm -rf /"},
                    "junk",
                ],
            ),
            "rm -rf /",
        ),
        (_call("ask_user", message="ok?", schema={"type": "string"}), "ok?"),
        (_call("notify_user", title="t", message="m"), "t\nm"),
        (_call("submit", answer="42"), "42"),
        (_call("mine", b="x", n=3, items=["y", {"k": "z"}], a=""), "x\ny\nz"),
        (_call("bash", cmd="rm x", timeout=5), "rm x"),
        (_call("bash"), ""),
    ],
)
def test_call_text(call: ToolCall, expected: str) -> None:
    assert call_text(call) == expected


@approver(name="sentinel_test_approve_all")
def _approve_all() -> Approver:
    async def approve(
        message: str,
        call: ToolCall,
        view: ToolCallView,
        history: list[ChatMessage],
    ) -> Approval:
        return Approval(decision="approve")

    return approve


def _approval_matches(function: str, pattern: str) -> bool:
    policy = policy_approver([ApprovalPolicy(_approve_all(), pattern)])

    async def decide() -> bool:
        approval = await policy("", _call(function, x="1"), ToolCallView(), [])
        return approval.decision == "approve"

    return anyio.run(decide)


@pytest.mark.parametrize(
    "function, pattern, expected",
    [
        ("bash", "bash", True),
        ("bash_session", "bash", True),
        ("python", "bash", False),
        ("python", "bash, python", True),
        ("python", "bash,python", True),
        ("web_browser_go", "web_browser*", True),
        ("web_browser_go", "web_*_go", True),
        ("web_search", "web_*_go", False),
        ("text_editor", "*", True),
        ("bash", "b?sh", True),
        ("bash", "[bc]ash", True),
        ("dash", "[bc]ash", False),
        ("mybash", "bash", False),
        ("bash", "Bash", False),
    ],
)
def test_tool_matches_as_an_approval_policy_does(
    function: str, pattern: str, expected: bool
) -> None:
    assert tool_matches(_call(function, x="1"), pattern) is expected
    assert _approval_matches(function, pattern) is expected


def test_tool_matches_any_of_several_patterns() -> None:
    assert tool_matches(_call("python"), "bash", "python")
    assert not tool_matches(_call("python"), "bash", "think")
    assert not tool_matches(_call("python"))


def test_tool_matches_ignores_arguments_where_approval_does_not() -> None:
    assert not tool_matches(_call("bash", command="python x"), "python")
    assert not tool_matches(_call("bash", x="1"), "*1'")
    assert _approval_matches("bash", "*1'")


def test_tool_matches_patterns_with_spaces_around_commas() -> None:
    assert tool_matches(_call("think"), " bash , think ", "python")


@pytest.mark.parametrize(
    "text, words, expected",
    [
        ("rm -rf /tmp/x", ["rm"], ["rm"]),
        ("python platform.py", ["rm"], []),
        ("/bin/rm x", ["rm"], ["rm"]),
        ("rm_all x", ["rm"], []),
        ("find . -name x -delete", ["rm", "-delete"], ["-delete"]),
        ("find . -deleted", ["-delete"], []),
        ("pip install x", ["pip install"], ["pip install"]),
        ("pip  install x", ["pip install"], ["pip install"]),
        ("pip\tinstall x", ["pip install"], ["pip install"]),
        ("pip installer", ["pip install"], []),
        ("wget a && curl b", ["curl", "wget"], ["wget", "curl"]),
        ("curl a; curl b", ["curl", "curl"], ["curl"]),
        ("a.b(c)", ["a.b(c)"], ["a.b(c)"]),
        ("Curl x", ["curl"], []),
        ("anything", ["", " "], []),
        ("axb a+b", ["a.b", "a+b"], ["a+b"]),
    ],
)
def test_find_words(text: str, words: list[str], expected: list[str]) -> None:
    assert find_words(text, words) == expected


def test_find_words_ignoring_case() -> None:
    assert find_words("Curl x", ["curl"], case_sensitive=False) == ["curl"]


@pytest.mark.parametrize(
    "step, expected",
    [
        (after_tool_call("bash", "out\n", command="ls"), "out\n"),
        (
            after_tool_call(
                "bash",
                error=ToolCallError("timeout", "Command timed out after 60s"),
                command="sleep 99",
            ),
            "Error (timeout): Command timed out after 60s",
        ),
    ],
)
def test_result_text(step: AfterToolCall, expected: str) -> None:
    assert result_text(step) == expected


def test_result_text_names_content_that_is_not_text() -> None:
    step = replace(
        after_tool_call("computer", action="screenshot"),
        result=ChatMessageTool(
            content=[ContentText(text="shot"), ContentImage(image="data:,")],
            tool_call_id="call_1",
        ),
    )
    assert result_text(step) == "shot\n[image]"


def test_before_tool_call_builds_a_call() -> None:
    step = before_tool_call("bash", command="ls")
    assert step.call == ToolCall(
        id="call_1", function="bash", arguments={"command": "ls"}
    )
    assert (step.message, step.input, step.history) == ("", [], [])


def test_after_tool_call_builds_a_result() -> None:
    step = after_tool_call("bash", "out", command="ls")
    assert step.call == before_tool_call("bash", command="ls").call
    assert (step.result.text, step.result.tool_call_id, step.output) == (
        "out",
        "call_1",
        "out",
    )
    assert step.result.error is None


def test_after_tool_call_builds_a_failure() -> None:
    error = ToolCallError("permission", "denied")
    step = after_tool_call("bash", error=error, command="cat /x")
    assert step.result.error == error
    assert step.call.arguments == {"command": "cat /x"}
