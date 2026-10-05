from dataclasses import replace
from typing import Any

import pytest
from inspect_ai.model import ChatMessageTool, ContentImage, ContentText
from inspect_ai.tool import ToolCall, ToolCallError

from inspect_sentinel import (
    AfterToolCall,
    call_text,
    find_words,
    result_text,
    tool_matches,
)
from tests._fakes import after_tool_call


def _call(function: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", function=function, arguments=arguments)


@pytest.mark.parametrize(
    "call, expected",
    [
        (_call("bash", command="ls /work"), "ls /work"),
        (_call("bash", cmd="rm x", timeout=5), "rm x"),
        (_call("bash"), ""),
        (
            _call("text_editor", command="create", path="/w/f", file_text="x"),
            "create\n/w/f\nx",
        ),
        (_call("read_file", offset=3, file_path="/w/f"), "/w/f"),
        (_call("mine", b="x", n=3, items=["y", {"k": "z"}], a=""), "x\ny\nz"),
        (_call("mine", d={"b": "1", "a": ["2", ("3",)]}, flag=True), "1\n2\n3"),
        (_call("mine", s="two\nlines", t="z"), "two\nlines\nz"),
    ],
)
def test_call_text(call: ToolCall, expected: str) -> None:
    assert call_text(call) == expected


@pytest.mark.parametrize(
    "function, patterns, expected",
    [
        ("bash", ["bash"], True),
        ("bash_session", ["bash"], False),
        ("bash_session", ["bash*"], True),
        ("bash", ["bash*"], True),
        ("mybash", ["bash"], False),
        ("bash", ["Bash"], False),
        ("bash", ["b?sh"], True),
        ("bash", ["[bc]ash"], True),
        ("dash", ["[bc]ash"], False),
        ("web_browser_go", ["web_*_go"], True),
        ("web_search", ["web_*_go"], False),
        ("text_editor", ["*"], True),
        ("python", ["bash", "python"], True),
        ("python", ["bash", "think"], False),
        ("python", [], False),
        ("python", ["bash,python"], False),
        ("python", ["bash, python"], False),
    ],
)
def test_tool_matches(function: str, patterns: list[str], expected: bool) -> None:
    assert tool_matches(_call(function, x="1"), *patterns) is expected


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
