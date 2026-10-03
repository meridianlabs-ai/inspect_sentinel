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
    path_matches,
    paths_in,
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


def test_tool_matches_ignores_arguments() -> None:
    assert not tool_matches(_call("bash", command="python x"), "python")


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


@pytest.mark.parametrize(
    "call, expected",
    [
        (_call("text_editor", command="view", path="w/f.py"), ["w/f.py"]),
        (
            _call("memory", command="rename", old_path="/m/a", new_path="/m/b"),
            ["/m/a", "/m/b"],
        ),
        (_call("read_file", file_path="/w/f"), ["/w/f"]),
        (_call("grep", pattern="x", path="/w"), ["/w"]),
        (
            _call("bash", command="cat /etc/passwd ~/.aws/creds"),
            ["/etc/passwd", "~/.aws/creds"],
        ),
        (_call("bash", command="cd ~ && ls"), ["~"]),
        (_call("bash", command="rm -rf /"), ["/"]),
        (_call("bash", command="cp '/a b' \"/c\";ls>/d"), ["/a", "/c", "/d"]),
        (_call("bash", command="tar --file=/x.tar a"), ["/x.tar"]),
        (_call("bash", command="PATH=/usr/bin:/bin x"), ["/usr/bin", "/bin"]),
        (_call("bash", command="cat ./a ../b c/d ~user/e"), []),
        (_call("bash", command="curl https://x.org/a file:///b"), []),
        (_call("bash", command="cat //etc/passwd"), ["//etc/passwd"]),
        (_call("bash", command="cat /a /a"), ["/a"]),
        (_call("python", code="open('/etc/hosts').read()"), ["/etc/hosts"]),
        (_call("bash_session", action="type_submit", input="ls /w"), ["/w"]),
        (_call("think", thought="look at /etc"), []),
        (_call("mine", path="/w"), []),
    ],
)
def test_paths_in(call: ToolCall, expected: list[str]) -> None:
    assert paths_in(call) == expected


@pytest.mark.parametrize(
    "path, patterns, cwd, expected",
    [
        ("/etc/passwd", ["/etc/**"], None, True),
        ("/etc", ["/etc/**"], None, True),
        ("/etcetera", ["/etc/**"], None, False),
        ("/work/../etc/passwd", ["/etc/**"], None, True),
        ("//etc///passwd", ["/etc/*"], None, True),
        ("/etc/./ssh/key", ["/etc/*"], None, False),
        ("/etc/./ssh/key", ["/etc/*/*"], None, True),
        ("/w/a/b/c.env", ["**/*.env"], None, True),
        ("c.env", ["**/*.env"], None, True),
        ("/w/a/b/c.py", ["/w/**/c.py"], None, True),
        ("/w/c.py", ["/w/**/c.py"], None, True),
        ("/w/c.py", ["/w/c.p?"], None, True),
        ("/w/c.py", ["/w/c.p"], None, False),
        ("~/.aws/credentials", ["~/.aws/**"], None, True),
        ("~/x/../.aws/credentials", ["~/.aws/**"], None, True),
        ("~/../../.aws", ["~/.aws"], None, True),
        ("~", ["~"], None, True),
        ("/root/.aws/c", ["~/.aws/**"], None, False),
        ("../etc/passwd", ["/etc/**"], "/work", True),
        ("../etc/passwd", ["/etc/**"], None, False),
        ("../etc/passwd", ["../etc/**"], None, True),
        ("./a/../b", ["b"], None, True),
        (".aws/c", ["~/.aws/**"], "~", True),
        ("/etc/passwd", ["/etc/**"], "/work", True),
        ("/w/x", [], None, False),
        ("/w/x", ["/tmp/**", "/w/*"], None, True),
    ],
)
def test_path_matches(
    path: str, patterns: list[str], cwd: str | None, expected: bool
) -> None:
    assert path_matches(path, patterns, cwd=cwd) is expected


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
