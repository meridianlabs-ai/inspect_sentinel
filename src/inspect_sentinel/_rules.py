from __future__ import annotations

import fnmatch
import posixpath
import re
from collections.abc import Iterable
from typing import Any, NamedTuple, cast

from inspect_ai.model import ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallError, ToolCallView

from ._step import AfterToolCall, BeforeToolCall


class _Builtin(NamedTuple):
    text: tuple[str, ...]
    other: tuple[str, ...] = ()


_BUILTINS: dict[str, _Builtin] = {
    "bash": _Builtin(("command",)),
    "python": _Builtin(("code",)),
    "code_execution": _Builtin(("code",)),
    "bash_session": _Builtin(("input",), ("action",)),
    "text_editor": _Builtin(
        ("path", "file_text", "old_str", "new_str", "insert_text"),
        ("command", "insert_line", "view_range"),
    ),
    "memory": _Builtin(
        (
            "path",
            "old_path",
            "new_path",
            "file_text",
            "old_str",
            "new_str",
            "insert_text",
        ),
        ("command", "insert_line", "view_range"),
    ),
    "read_file": _Builtin(("file_path",), ("offset", "limit")),
    "list_files": _Builtin(("path",), ("depth",)),
    "grep": _Builtin(
        ("pattern", "path", "glob"),
        ("fixed_strings", "extended_regexp", "output_mode"),
    ),
    "skill": _Builtin(("command",)),
    "think": _Builtin(("thought",)),
    "web_search": _Builtin(("query",)),
    "web_browser_go": _Builtin(("url",)),
    "web_browser_type": _Builtin(("text",), ("element_id",)),
    "web_browser_type_submit": _Builtin(("text",), ("element_id",)),
    "computer": _Builtin(
        ("text",),
        (
            "actions",
            "action",
            "coordinate",
            "duration",
            "region",
            "scroll_amount",
            "scroll_direction",
            "start_coordinate",
            "repeat",
            "press_enter",
        ),
    ),
    "ask_user": _Builtin(("message",), ("schema",)),
    "notify_user": _Builtin(("title", "message")),
}

_PATH_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "text_editor": ("path",),
    "memory": ("path", "old_path", "new_path"),
    "read_file": ("file_path",),
    "list_files": ("path",),
    "grep": ("path",),
}

_CODE_TOOLS = {"bash", "python", "code_execution", "bash_session"}

_STOP = r"\s'\"`;|&<>(){}\[\],:=$"
_PATH_TOKEN = re.compile(
    rf"(?<![\w.~/\-])(?:~(?=/|[{_STOP}]|$)|(?!(?<=:)//)(?=/))[^{_STOP}]*"
)


def call_text(call: ToolCall) -> str:
    """The text of a tool call, whichever tool it is.

    For inspect_ai's built-in tools, the arguments that carry text, in this order, joined with newlines:

    | Tool | Arguments |
    |---|---|
    | `bash()` | `command` |
    | `python()`, `code_execution()` | `code` |
    | `bash_session()` | `input` |
    | `text_editor()` | `path`, `file_text`, `old_str`, `new_str`, `insert_text` |
    | `memory()` | `path`, `old_path`, `new_path`, `file_text`, `old_str`, `new_str`, `insert_text` |
    | `read_file()` | `file_path` |
    | `list_files()` | `path` |
    | `grep()` | `pattern`, `path`, `glob` |
    | `skill()` | `command` |
    | `think()` | `thought` |
    | `web_search()` | `query` |
    | `web_browser_go()` | `url` |
    | `web_browser_type()`, `web_browser_type_submit()` | `text` |
    | `computer()` | `text`, then each of `actions`' `text` |
    | `ask_user()` | `message` |
    | `notify_user()` | `title`, `message` |

    Arguments that select an action or a mode, such as `text_editor()`'s `command`, are left out. For any other tool, and for a call whose arguments are not all parameters of the built-in tool of its name (a tool of your own named `bash`, say), it is every string argument value in the order the arguments were given. Either way, strings inside lists and dicts are included, in order, and missing or empty arguments are skipped.

    Args:
        call: The tool call.

    Returns:
        The call's text, or `""` when it has none.
    """
    builtin = _BUILTINS.get(call.function)
    if builtin is not None and set(call.arguments) <= {
        *builtin.text,
        *builtin.other,
    }:
        values = [call.arguments.get(name) for name in builtin.text]
        if call.function == "computer":
            values.extend(_action_texts(call.arguments.get("actions")))
    else:
        values = list(call.arguments.values())
    return "\n".join(text for value in values for text in _strings(value) if text)


def _action_texts(actions: object) -> list[object]:
    if not isinstance(actions, list):
        return []
    return [
        cast("dict[str, object]", action).get("text")
        for action in cast("list[object]", actions)
        if isinstance(action, dict)
    ]


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list | tuple | dict):
        items = cast(
            "Iterable[object]", value.values() if isinstance(value, dict) else value
        )
        for item in items:
            yield from _strings(item)


def tool_matches(call: ToolCall, *patterns: str) -> bool:
    """Whether the call's tool matches any of the patterns.

    Patterns match the tool name as an `ApprovalPolicy`'s `tools` do in inspect_ai: a glob (`*`, `?`, `[seq]`) that also matches any name it is a prefix of, so `"bash"` matches `bash_session` too. To match exactly one tool, compare `call.function` instead. A pattern may hold several, separated by commas (`"bash, python"`). Unlike an approval policy, a pattern does not match against the call's arguments.

    Args:
        call: The tool call.
        *patterns: Tool name globs.

    Returns:
        True if any pattern matches.
    """
    for spec in patterns:
        for pattern in (p.strip() for p in spec.split(",")):
            if not pattern:
                continue
            glob = pattern if pattern.endswith("*") else f"{pattern}*"
            if fnmatch.fnmatch(call.function, glob):
                return True
    return False


def find_words(
    text: str, words: Iterable[str], *, case_sensitive: bool = True
) -> list[str]:
    """The words that appear in the text as whole words.

    A word matches where it is not part of a longer word: the characters on either side of it are not letters, digits or underscores. So `"rm"` matches `rm -rf x` and `/bin/rm x` but not `platform`, and a word that starts or ends with punctuation works too: `"-delete"` matches `find . -delete`. A space in a word matches any run of whitespace, so `"pip install"` also matches `pip` and `install` separated by a tab or several spaces.

    Args:
        text: The text to search, such as `call_text(step.call)`.
        words: The words to look for.
        case_sensitive: Whether case must match.

    Returns:
        The words found, as given, in the order they first appear in the text.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    found: list[tuple[int, str]] = []
    for word in dict.fromkeys(words):
        if not word.strip():
            continue
        pattern = r"\s+".join(re.escape(part) for part in word.split())
        match = re.search(rf"(?<!\w){pattern}(?!\w)", text, flags)
        if match is not None:
            found.append((match.start(), word))
    return [word for _, word in sorted(found, key=lambda item: item[0])]


def result_text(step: AfterToolCall) -> str:
    """The tool call's result as text, as inspect_ai's human review shows it.

    Content other than text, such as an image, appears as its type in brackets (`[image]`). A call that failed reads `"Error (<type>): <message>"`.

    Args:
        step: The step after the tool call.

    Returns:
        The result's text.
    """
    error = step.result.error
    if error is not None:
        return f"Error ({error.type}): {error.message}"
    return "\n".join(
        content.text if content.type == "text" else f"[{content.type}]"
        for content in step.result.content_list
    )


def paths_in(call: ToolCall) -> list[str]:
    """Paths the tool call names, as best it can tell.

    Two sources, in this order: the path arguments of inspect_ai's built-in file tools (`text_editor()`'s and `list_files()`'s `path`, `memory()`'s `path`, `old_path` and `new_path`, `read_file()`'s `file_path`, `grep()`'s `path`); and, in the text of `bash()`, `bash_session()`, `python()` and `code_execution()` calls, tokens that begin with `/` or `~/`, or are `~` alone, ending at whitespace, quotes or shell punctuation. Relative paths in commands, paths built from variables (`$HOME/x`), and paths inside URLs, `file:///etc/passwd` included, are not found, and a `/` used as an operator is found as the root.

    Args:
        call: The tool call.

    Returns:
        The paths, as written and without duplicates, in order.
    """
    paths: list[str] = []
    for name in _PATH_ARGUMENTS.get(call.function, ()):
        value = call.arguments.get(name)
        if isinstance(value, str) and value:
            paths.append(value)
    if call.function in _CODE_TOOLS:
        paths.extend(_PATH_TOKEN.findall(call_text(call)))
    return list(dict.fromkeys(paths))


def path_matches(path: str, patterns: Iterable[str], *, cwd: str | None = None) -> bool:
    """Whether a path, once normalised, matches any of the glob patterns.

    The path is normalised first: repeated slashes collapse, and `.` and `..` resolve, so `/work/../etc/passwd` is `/etc/passwd`. A relative path is resolved against `cwd` when given, and otherwise matched as it is. `~` is not expanded, since the home directory is the agent's and not known here: `~/x` matches a `~/...` pattern, and a path whose `..` climbs out of `~` keeps it (`~/../etc`), so it matches neither form. List both forms (`"~/.aws/**"` and `"/root/.aws/**"`) to catch either.

    In a pattern, `*` matches within one path segment, `?` matches one character other than `/`, and `**` as a whole segment matches any number of segments, including none: `"/etc/**"` matches `/etc` and everything under it. Other characters, `[` included, match themselves. Patterns have repeated and trailing slashes removed but are not otherwise normalised.

    Args:
        path: The path, such as one from `paths_in()`.
        patterns: Glob patterns.
        cwd: Directory that a relative path is relative to.

    Returns:
        True if any pattern matches.
    """
    normalized = _normalize(path, cwd)
    return any(
        re.fullmatch(_glob_regex(_pattern(pattern)), normalized) for pattern in patterns
    )


def _pattern(pattern: str) -> str:
    pattern = _collapse(pattern)
    return pattern.rstrip("/") if len(pattern) > 1 else pattern


def _collapse(path: str) -> str:
    return re.sub(r"/{2,}", "/", path)


def _normalize(path: str, cwd: str | None) -> str:
    path = _collapse(path)
    home = path == "~" or path.startswith("~/")
    if cwd is not None and not home and not path.startswith("/"):
        return _normalize(f"{_collapse(cwd)}/{path}", None)
    if home:
        rest = posixpath.normpath(path[2:] or ".")
        return "~" if rest == "." else f"~/{rest}"
    return posixpath.normpath(path)


def _glob_regex(pattern: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if (
            pattern.startswith("**", i)
            and (i == 0 or pattern[i - 1] == "/")
            and (i + 2 == len(pattern) or pattern[i + 2] == "/")
        ):
            if i + 2 < len(pattern):
                out.append("(?:.*/)?")
                i += 3
            elif i == 0:
                out.append(".*")
                i += 2
            else:
                out[-1] = "(?:/.*)?"
                i += 2
            continue
        char = pattern[i]
        out.append(
            "[^/]*" if char == "*" else "[^/]" if char == "?" else re.escape(char)
        )
        i += 1
    return "".join(out)


def before_tool_call(function: str, /, **arguments: Any) -> BeforeToolCall:
    """A `BeforeToolCall` for unit-testing a rule.

    The call has id `"call_1"` and the conversation id `"conversation"`; the other fields are empty: no message, an empty view, and no input or history.

    Args:
        function: The tool's name.
        **arguments: The call's arguments.

    Returns:
        The step.
    """
    return BeforeToolCall(
        conversation="conversation",
        message="",
        call=ToolCall(id="call_1", function=function, arguments=arguments),
        view=ToolCallView(),
        input=[],
        history=[],
    )


def after_tool_call(
    function: str,
    /,
    result: str = "",
    *,
    error: ToolCallError | None = None,
    **arguments: Any,
) -> AfterToolCall:
    """An `AfterToolCall` for unit-testing a rule.

    The call is built as `before_tool_call()` builds it, and `result` is both the tool's output and what the model sees. A tool with an argument named `result` or `error` cannot be built this way.

    Args:
        function: The tool's name.
        result: The tool's output.
        error: The error the call failed with, if it failed.
        **arguments: The call's arguments.

    Returns:
        The step.
    """
    before = before_tool_call(function, **arguments)
    return AfterToolCall(
        conversation=before.conversation,
        message=before.message,
        call=before.call,
        result=ChatMessageTool(
            content=result,
            tool_call_id=before.call.id,
            function=function,
            error=error,
        ),
        output=result,
        view=before.view,
        input=before.input,
        history=before.history,
    )
