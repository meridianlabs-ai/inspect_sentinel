from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable
from typing import cast

from inspect_ai.tool import ToolCall

from ._step import AfterToolCall


def call_text(call: ToolCall) -> str:
    """The text of a tool call: every string in its arguments.

    The string argument values, in the order the arguments were given, joined with newlines. Strings inside lists and dicts are included, in order (a dict's values in insertion order); empty strings and values that are not strings, such as numbers, are left out. This knows nothing about particular tools: for `bash()` it is the `command`, and for `text_editor()` it is the `command` (such as `"create"`) as well as the path and the file's text.

    Args:
        call: The tool call.

    Returns:
        The call's text, or `""` when it has none.
    """
    return "\n".join(text for text in _strings(call.arguments) if text)


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list | tuple | dict):
        items = cast(
            "Iterable[object]", value.values() if isinstance(value, dict) else value
        )
        for item in items:
            yield from _strings(item)


def tool_matches(call: ToolCall, *patterns: str, case_sensitive: bool = True) -> bool:
    """Whether the call's tool name matches any of the patterns.

    A pattern is a tool name or a glob (`*`, `?`, `[seq]`) that must match the whole name: `"bash"` matches `bash` and not `bash_session`, while `"bash*"` matches both. Matching is case-sensitive by default and the same on every platform; pass `case_sensitive=False` for tools whose names differ only in case, such as inspect_ai's `bash` and a bridged agent's `Bash`. Unlike the `tools` of an `ApprovalPolicy` in inspect_ai, a pattern is not extended with a trailing `*` and is not split on commas.

    Args:
        call: The tool call.
        *patterns: Tool names or globs.
        case_sensitive: Whether case must match.

    Returns:
        True if any pattern matches.
    """
    function = call.function if case_sensitive else call.function.casefold()
    return any(
        fnmatch.fnmatchcase(function, pattern if case_sensitive else pattern.casefold())
        for pattern in patterns
    )


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
