from __future__ import annotations

import fnmatch
import posixpath
import re
from collections.abc import Iterable
from typing import cast

from inspect_ai.tool import ToolCall
from wcmatch.glob import (
    CASE,
    DOTGLOB,
    FORCEUNIX,
    GLOBSTAR,
    NODOTDIR,
    escape,
    globmatch,
    is_magic,
)

from ._step import AfterToolCall

_UNPLACEABLE = re.compile(r"^(?:file:|~[^/])", re.IGNORECASE)


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


def path_matches(
    path: str,
    patterns: Iterable[str],
    *,
    cwd: str | None = None,
    home: str | None = None,
) -> bool:
    r"""Whether a path, once normalised, matches any of the glob patterns.

    The path is normalised first, by its text alone (symlinks are not followed): repeated slashes collapse, and `.` and `..` resolve, so `/work/../etc/passwd` is `/etc/passwd` and `/..` is `/`. A relative path is joined to `cwd` when given. With `home`, a leading `~` is replaced by it, in the path, in `cwd` and in each pattern, so `~/.ssh/**` matches `/root/.ssh/id_rsa` when `home="/root"`. Without `home`, `~` is kept as written: `~/x` matches `~/**` and not `/root/**`. `~user` is never expanded. Patterns are not joined to `cwd`: a relative pattern such as `secrets/**` matches only a relative path, so with `cwd` write patterns absolute, or begin them with `**/`.

    A path can keep a `..` after normalising: a relative one without `cwd` (`../etc`), or one that climbs out of `~` without `home` (`~/../etc`). Wildcards never match a `.` or `..` segment, so such a path matches only a pattern that spells its `..` segments out, never `/etc/**`, `~/**` or `**`. An allow-list therefore rejects it. A deny-list would let it through, so a rule that denies should also escalate or reject a path for which `path_resolves()` is false.

    Patterns are globs: `*` matches any characters within one segment, `?` one character other than `/`, `[seq]` one character in `seq` (ranges such as `a-z` and POSIX classes such as `[:alpha:]` included), `[!seq]` or `[^seq]` one character not in it, and `\` escapes the next character. All of these match a leading `.`, so `/home/*/.ssh` and `/etc/*` see dotfiles. `**` as a whole segment matches any number of segments, and a pattern ending in `/**` also matches the directory itself: `"/etc/**"` matches `/etc` and everything under it, and `"**/*.env"` matches `.env` files anywhere, absolute paths included. Matching is case-sensitive. Braces and `!` at the start of a pattern have no special meaning. A pattern is normalised too: repeated and trailing slashes and `.` segments are removed, and a `..` resolves against the segment before it when that segment has no wildcard (`/work/../etc/**` is `/etc/**`); other `..` segments are kept.

    Args:
        path: The path, such as a file tool's `path` argument.
        patterns: Glob patterns.
        cwd: Directory that a relative path is relative to.
        home: The agent's home directory, an absolute path, to expand `~` to.

    Returns:
        True if any pattern matches.

    Raises:
        ValueError: If `home` is not an absolute path.
    """
    home = _resolve_home(home)
    normalized = _normalize(path, cwd, home)
    return any(
        globmatch(normalized, form, flags=_GLOB_FLAGS)
        for pattern in patterns
        for form in _pattern_forms(pattern, home)
    )


def path_resolves(
    path: str, *, cwd: str | None = None, home: str | None = None
) -> bool:
    r"""Whether a path normalises to a place: an absolute path, or one under `~`, with no `..` left.

    The path is normalised as `path_matches()` does. It cannot be placed when it is relative and there is no `cwd` (`passwd`, `../etc`), when it climbs out of `~` and there is no `home` (`~/../etc`), or when it is `~user/x` or a `file:` URL. The path is otherwise taken literally, as a file tool's argument is: `$`, `*` and `\` are characters of a name.

    Args:
        path: The path.
        cwd: Directory that a relative path is relative to.
        home: The agent's home directory, an absolute path, to expand `~` to.

    Returns:
        True if the path is absolute or under `~` once normalised, with no `..` segment.

    Raises:
        ValueError: If `home` is not an absolute path.
    """
    home = _resolve_home(home)
    if _UNPLACEABLE.search(path):
        return False
    normalized = _normalize(path, cwd, home)
    anchored = normalized.startswith("/") or _is_home(normalized)
    return anchored and ".." not in normalized.split("/")


_GLOB_FLAGS = GLOBSTAR | DOTGLOB | NODOTDIR | FORCEUNIX | CASE


def _pattern_forms(pattern: str, home: str | None) -> Iterable[str]:
    if home is not None:
        pattern = _expand_home(pattern, escape(home))
    anchor = "/" if pattern.startswith("/") else ""
    segments: list[str] = []
    for segment in pattern.split("/"):
        if segment in ("", "."):
            continue
        if segment == ".." and (anchor or segments) and _resolvable(segments):
            segments = segments[:-1]
        else:
            segments.append(segment)
    pattern = anchor + "/".join(segments) or ("." if pattern else "")
    yield pattern
    while pattern.endswith("/**"):
        pattern = pattern[:-3] or "/"
        yield pattern


def _resolvable(segments: list[str]) -> bool:
    if not segments:
        return True
    last = segments[-1]
    if last == ".." or (last == "~" and len(segments) == 1):
        return False
    return not is_magic(last, flags=_GLOB_FLAGS)


def _resolve_home(home: str | None) -> str | None:
    if home is None:
        return None
    if not home.startswith("/"):
        raise ValueError(f"home must be an absolute path, not {home!r}")
    return posixpath.normpath(_collapse(home))


def _collapse(path: str) -> str:
    return re.sub(r"/{2,}", "/", path)


def _is_home(path: str) -> bool:
    return path == "~" or path.startswith("~/")


def _expand_home(path: str, home: str) -> str:
    return home + path[1:] if _is_home(path) else path


def _normalize(path: str, cwd: str | None, home: str | None) -> str:
    if home is not None:
        path = _expand_home(path, home)
        cwd = None if cwd is None else _expand_home(cwd, home)
    if cwd is not None and not (path.startswith("/") or _is_home(path)):
        path = posixpath.join(cwd, path)
    path = _collapse(path)
    if _is_home(path):
        rest = posixpath.normpath(path[2:] or ".")
        return "~" if rest == "." else f"~/{rest}"
    return posixpath.normpath(path)
