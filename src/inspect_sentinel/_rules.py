from __future__ import annotations

import bisect
import fnmatch
import posixpath
import re
from collections.abc import Iterable
from typing import Any, NamedTuple, cast

from inspect_ai.model import ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallError, ToolCallView
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

from ._shell import LITERAL, UNQUOTED, Command, Shell, lex, literal_spans
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
    rf"(?<![\w.~/\-+}})*?])(?:~(?=/|[{_STOP}]|$)|(?!(?<=:)//)(?=/))[^{_STOP}]*"
)
_OPTION_PATH = re.compile(rf"(?<![^\s'\"])-[A-Za-z]+(/[^{_STOP}]*)")
_RELATIVE_TOKEN = re.compile(rf"(?<![^\s'\"(=:,;|&<>\[{{])[\w.*?][^{_STOP}]*")
_EXPANSION = (
    r"\$[\w@*#?$!-]+|\$\{[^}]{0,256}\}|\$\((?:[^()]|\([^()]*\)){0,256}\)|`[^`]{0,256}`"
)
_BUILT_PATH = re.compile(
    rf"(?P<expansion>{_EXPANSION}|\{{[^{{}}\s]*\}})[\"'\w.~-]*/[^{_STOP}]*"
)
_SHELL_BUILT_PATH = re.compile(
    rf"(?P<expansion>{_EXPANSION}|\{{[^{{}}\s]*(?:,|\.\.)[^{{}}\s]*\}})[\"'\w.~-]*/[^{_STOP}]*"
)
_URL = re.compile(r"(?<![\w+.-])[A-Za-z][\w+.-]*://[^\s'\"`<>]*")
_USER_PATH = re.compile(rf"(?<![^\s'\"(=:,;|&<>])~[^/{_STOP}]+/[^{_STOP}]*")
_FILE_URL = re.compile(r"(?<![\w+.-])file:/[^\s'\"`<>;|&()]*", re.IGNORECASE)
_CHANGES_DIRECTORY = re.compile(
    r"(?:^|[;&|({\n\"'`]|\b(?:then|do|else)\b)\s*(?:cd|pushd|popd)(?![\w.-])"
    r"|\b(?:git|tar|make|env)\b[^;&|\n]*\s-[A-Za-z]*C|--directory\b|--chdir\b"
    r"|\bchdir\b|\bcwd\s*="
)
_SHELL_DIRECTORY = re.compile(
    r"(?:^|[;&|({\n\"'`]|\b(?:then|do|else)\b)\s*(?:cd|pushd|popd)(?![\w.-])"
    r"|\bchdir\b|\bcwd\s*="
)
_SCOPED_DIRECTORY = re.compile(r"(?<![^\s'\"=])(?:-[A-Za-z]*C|--directory\b|--chdir\b)")
_SCOPING_PROGRAM = re.compile(r"\b(?:git|tar|make|env)\b")
_SCOPING_PROGRAMS = frozenset({"git", "tar", "make", "env"})
_DIRECTORY_COMMANDS = frozenset({"cd", "pushd", "popd"})
_VARIABLE_WORD = {name: re.compile(rf"(?<!\w){name}(?!\w)") for name in ("HOME", "PWD")}
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*\+?=")
_PATH_START = frozenset(" \t\n=:;|&<>(")
_STRING_LITERAL = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"")
_GLUED_AFTER = re.compile(
    r"""["']*(?:\$[\w{(@*#?$!-]|\{)|(['"])(?!\1\1)[\w/$~*?\\+`'"-]"""
)
_GLUE_BEFORE = frozenset("/.$~*?+)}]`'\"-")
_STRING_PREFIX = re.compile(r"""(?<!\w)[rRbBuUfF]{1,2}['"]$""")
_CONCATENATED_AFTER = re.compile(r"""(['"])\s*(?:[+/]|[rRbBuUfF]{0,2}['"])""")
_CONCATENATED_BEFORE = re.compile(r"""(?:[+/]|['"])\s*[rRbBuUfF]{0,2}['"]$""")
_BRACKET_DIRECTORY = re.compile(r"\[[^\]/\s]*\]/")
_WORD_STOPS = frozenset(" \t\n\r;|&<>(),=:")
_WORD_END = re.compile(r"(?:\$\((?:[^()]|\([^()]*\)){0,256}\)|[^\s;|&<>(),=:]){0,256}")
_UNPLACEABLE = re.compile(r"^(?:file:|~[^/])", re.IGNORECASE)
_SHELL_TOOLS = {"bash", "bash_session"}


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
    r"""Paths the tool call names, as best it can tell.

    Two sources, in this order: the path arguments of inspect_ai's built-in file tools (`text_editor()`'s and `list_files()`'s `path`, `memory()`'s `path`, `old_path` and `new_path`, `read_file()`'s `file_path`, `grep()`'s `path`); and paths in the text of `bash()`, `bash_session()`, `python()` and `code_execution()` calls. In that text a path is a token that begins with `/` or `~/`, or is `~` alone, or is glued to a short option (`-C/etc`); or a relative one with a `/` in it (`a/b`, `./x`, `../x`), which in `python()` and `code_execution()` counts only inside a string literal, and in `bash()` and `bash_session()` includes `.` and `..` alone. A token ends at whitespace, quotes or shell punctuation.

    A reference whose path cannot be read from the text is left out, and `unresolved_paths()` reports it instead: one built at run time (`/home/$USER/x`, `$(pwd)/x`, backticks, `f"/x/{name}"`, `{a,b}/x`, `"/x/" + name`, `Path.home() / ".ssh"`), one with a shell wildcard (`/etc/pass*`, `/e*/passwd`, `@(x)`) or a backslash in it, a `~` prefix other than `~/` (`~user/x`, `~+/x`), a quoted `~` (`'~/x'`), `file://` URLs, and relative paths in code that changes directory (`cd`, `pushd`, `chdir`, `git -C`, `cwd=`). In a shell, a backslash at the end of a line joins it to the next, as the shell does.

    In `bash()` text, `$HOME/x`, `${HOME}/x` and `"$HOME"/x` are given as `~/x`, and `$PWD/x` as `./x`, unless the call names the variable another way (`HOME=/etc`). After a `cd` that `unresolved_paths()` can follow, a relative path is given both as written and joined to the new directory (`cd /w && cat a/b` gives `a/b` and `/w/a/b`). A quoted argument that the shell and its command both read literally is given as written, wildcards and `$` included (`cat '/e*c/x'` gives `/e*c/x`); `unresolved_paths()` says when.

    Not found at all: a path with no `/` in a command (`cat passwd`), a path held whole in a variable (`cat $F`), a path built by a function (`os.path.join(home, ".ssh")`), a path whose `/` is written as an escape (`"\x2fetc\x2fpasswd"`), and paths in the arguments of any other tool. Misread: a quoted path with a space is cut at the space (`"/a b"` as `/a`), and a `/` or `//` used as an operator is found as a path (`$((4 / 2))` as `/`).

    Args:
        call: The tool call.

    Returns:
        The paths, as written and without duplicates, in order.
    """
    paths = _argument_paths(call)
    if call.function in _CODE_TOOLS:
        paths.extend(
            ref.text for ref in _scan(call.function, call_text(call)) if ref.readable
        )
    return list(dict.fromkeys(paths))


def unresolved_paths(
    call: ToolCall, *, cwd: str | None = None, home: str | None = None
) -> list[str]:
    r"""References to paths in the tool call that cannot be resolved to a place.

    These are the paths from `paths_in()` that `path_resolves()` cannot place with this `cwd` and `home`, and the references `paths_in()` leaves out because no path can be read from them: those built at run time (`/home/$USER/x`, `$D/x`, `$(pwd)/x`, `` `pwd`/x ``, `f"/x/{name}"`, `{a,b}/x`, `"/x/" + name`, `Path.home() / ".ssh"`), those with a shell wildcard (`/etc/pass*`, `/e*/passwd`, `/[e]tc/passwd`, `@(x)`) or a backslash, `~user/x`, `~+/x` and a quoted `~` (`'~/x'`), `file://` URLs, and relative paths in code that changes directory (`cd`, `git -C`, `cwd=`), which `cwd` no longer places. A shell call whose quotes or brackets do not close is reported whole. A rule that denies paths escalates or rejects a call with any of them, since a deny-list cannot see where they lead:

    ```python
    for path in paths_in(step.call):
        if path_matches(path, PROTECTED, cwd="/work", home="/root"):
            return Decision.reject(f"call touches {path}")
    unresolved = unresolved_paths(step.call, cwd="/work", home="/root")
    if unresolved:
        return Decision.escalate(f"cannot resolve {', '.join(unresolved)}")
    ```

    In `bash()`, `$HOME/x`, `${HOME}/x` and `"$HOME"/x` are read as `~/x`, so they are reported without `home` and resolved with it, and `$PWD/x` is read as `./x`. They are reported as written when the call names the variable another way (`HOME=/etc; cat $HOME/x`), and in `bash_session()`, whose shell keeps variables from earlier calls.

    A `cd` to an absolute path or `~`, or a leading `cd` to a path that begins with `.` or `..`, leaves later relative paths resolved: `paths_in()` gives each both as written and joined to the new directory, in case the `cd` fails. Any other change of directory makes them unresolved, as does `cd name` (which `CDPATH` can redirect), a second `cd`, and a command whose name is not written out (`$CMD`). `-C` and `--directory` of `tar`, `git`, `make` and `env` affect only the rest of that command.

    Quoting is taken into account in `bash()`. A whole argument in single quotes, or in double quotes without `$`, `` ` `` or `\`, is not expanded by the shell. When it is an argument of a command that also reads it literally (`awk`, `cat`, `cut`, `diff`, `echo`, `egrep`, `fgrep`, `grep`, `head`, `jq`, `ls`, `nl`, `printf`, `rg`, `sed`, `sort`, `tail`, `tee`, `tr`, `uniq`, `wc`), its tokens are paths as written rather than unresolved references, so `sed 's/ *$//' f` and `cat '/e*c/x'` resolve. The same holds for a heredoc body with a quoted delimiter given to such a command. This applies only when nothing in the call could run the text: no command or process substitution; only those commands, `cd`, `true`, `false` and `:`; no variable assignment; no `sed` script with `e` or read from a file; no `awk` program with `system`, `|` or `@`, or read from a file; and none of `printf -v`, `sort --compress-program`, `--files0-from` and `rg --pre`. An argument with a `..` in it is not read this way, since a quoted path cut at a space could climb anywhere. A quoted glob given to any other program, such as `find -path '/e*'` or `eslint 'src/**/*.ts'`, is reported, since that program expands it.

    Without `cwd`, every relative path is unresolved, including the operands of `/` in a shell's arithmetic and words such as `origin/main`, so pass the agent's working directory. Other quoted text, a heredoc's body and a variable are read as shell, so a wildcard in a path (`wc -l src/*.py`), a variable in a path (`cp $f out/$f`) and a Python f-string or `+` with a `/` are reported. What `paths_in()` does not find at all, this does not report either.

    Args:
        call: The tool call.
        cwd: Directory that a relative path is relative to.
        home: The agent's home directory, an absolute path, to expand `~` to.

    Returns:
        The references, as written and without duplicates, in order.

    Raises:
        ValueError: If `home` is not an absolute path.
    """
    home = _resolve_home(home)
    references = [
        path
        for path in _argument_paths(call)
        if not path_resolves(path, cwd=cwd, home=home)
    ]
    if call.function in _CODE_TOOLS:
        for ref in _scan(call.function, call_text(call)):
            resolved = ref.readable and path_resolves(ref.text, cwd=cwd, home=home)
            if not resolved or (ref.variable == "HOME" and home is None):
                references.append(ref.written or ref.text)
    return list(dict.fromkeys(references))


def _argument_paths(call: ToolCall) -> list[str]:
    paths: list[str] = []
    for name in _PATH_ARGUMENTS.get(call.function, ()):
        value = call.arguments.get(name)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


class _Reference(NamedTuple):
    start: int
    text: str
    readable: bool
    end: int = 0
    variable: str | None = None
    written: str | None = None


class _Span(NamedTuple):
    start: int
    end: int


def _scan(function: str, text: str) -> list[_Reference]:
    shell = function in _SHELL_TOOLS
    if shell:
        text = text.replace("\\\n", "")
    lexed = lex(text) if shell else None
    bash = lexed is not None and function == "bash"
    home_mentioned = _mentioned(text, "HOME")
    pwd_mentioned = _mentioned(text, "PWD")
    literal = _Spans(literal_spans(text, lexed) if lexed is not None and bash else [])
    unreadable: list[_Span] = []
    blocked: list[_Span] = []
    found: list[_Reference] = []
    homes: list[_Span] = []
    urls = _Spans(
        m.span() for m in _URL.finditer(text) if not _FILE_URL.match(m.group())
    )

    def mark(start: int, end: int, block_from: int | None = None) -> None:
        span = _Span(_word_start(text, start), _word_end(text, end))
        content = literal.enclosing(start, end)
        if content is None:
            unreadable.append(span)
        else:
            begin, finish = max(span.start, content[0]), min(span.end, content[1])
            found.append(
                _Reference(begin, _literal(text[begin:finish]), True, end=finish)
            )
        blocked.append(span if block_from is None else _Span(block_from, span.end))

    for match in (_SHELL_BUILT_PATH if shell else _BUILT_PATH).finditer(text):
        if urls.contains(match.start()):
            continue
        home = None
        if lexed is not None and bash:
            home = _variable_path(text, match, lexed, home_mentioned, pwd_mentioned)
        if home is None:
            mark(match.start(), match.end(), block_from=match.end("expansion"))
        else:
            found.append(home)
            homes.append(_Span(home.start, match.end()))
    for pattern in (_USER_PATH, _FILE_URL):
        for match in pattern.finditer(text):
            mark(*match.span())

    candidates = [match.span() for match in _PATH_TOKEN.finditer(text)]
    candidates.extend(match.span(1) for match in _OPTION_PATH.finditer(text))
    literals = _Spans(
        [] if shell else (m.span() for m in _STRING_LITERAL.finditer(text))
    )
    changes = _directory_changes(text, lexed, home_mentioned)
    relative: list[_Span] = []
    for match in _RELATIVE_TOKEN.finditer(text):
        token, start = match.group(), match.start()
        if shell:
            is_relative = "/" in token or token in (".", "..")
        else:
            is_relative = "/" in token and literals.contains(start)
        if is_relative:
            candidates.append(match.span())
            relative.append(_Span(*match.span()))

    blocked.extend(homes)
    home_spans = _Spans(homes)
    readable: list[_Span] = []
    for start, end in candidates:
        if home_spans.overlaps(start, end):
            continue
        token_end = _readable_end(text, start, end, shell)
        if (
            token_end is None
            or _glued(text, start, token_end, shell)
            or (text.startswith("~", start) and home_mentioned)
            or (
                lexed is not None
                and lexed.quoting[start] != UNQUOTED
                and text.startswith("~", start)
            )
        ):
            mark(start, end)
        else:
            readable.append(_Span(start, token_end))

    for span in relative:
        if changes.moved(span.start):
            unreadable.append(
                _Span(_word_start(text, span.start), _word_end(text, span.end))
            )
            blocked.append(unreadable[-1])
    kept: list[_Reference] = []
    for ref in found:
        if _is_relative(ref.text) and changes.moved(ref.start):
            unreadable.append(_Span(ref.start, ref.end))
        else:
            kept.append(ref)

    unreachable = _Spans(_merge(blocked))
    references = [
        _Reference(start, text[start:end], True)
        for start, end in readable
        if not unreachable.overlaps(start, end)
    ]
    references.extend(kept)
    if changes.target is not None:
        target = changes.target
        references.extend(
            [
                _Reference(ref.start, posixpath.join(target, ref.text), True)
                for ref in references
                if _is_relative(ref.text) and ref.start != changes.target_start
            ]
        )
    references.extend(
        _Reference(start, _unquote(text[start:end]), False)
        for start, end in _merge(unreadable)
    )
    if shell and lexed is None and text.strip():
        references.append(_Reference(0, text.strip(), False))
    return sorted(references, key=lambda ref: ref.start)


def _is_relative(path: str) -> bool:
    return not (path.startswith("/") or _is_home(path))


def _literal(text: str) -> str:
    return f"./{text}" if text.startswith("~") else text


def _mentioned(text: str, name: str) -> bool:
    for match in _VARIABLE_WORD[name].finditer(text):
        start = match.start()
        if text.endswith("$", 0, start) and not text.endswith("$$", 0, start):
            continue
        if text.endswith("${", 0, start) and text.startswith("}", match.end()):
            continue
        return True
    return False


def _variable_path(
    text: str,
    match: re.Match[str],
    lexed: Shell,
    home_mentioned: bool,
    pwd_mentioned: bool,
) -> _Reference | None:
    variable = match.group("expansion").strip("${}")
    if match.group("expansion") not in (f"${variable}", f"${{{variable}}}"):
        return None
    if not (
        (variable == "HOME" and not home_mentioned)
        or (variable == "PWD" and not pwd_mentioned)
    ):
        return None
    start, after, end = match.start("expansion"), match.end("expansion"), match.end()
    quoting = lexed.quoting[start]
    if quoting == LITERAL:
        return None
    begin = start
    if quoting != UNQUOTED:
        if not text.endswith('"', 0, start) or lexed.quoting[start - 1] != UNQUOTED:
            return None
        begin = start - 1
        if text.startswith('"/', after):
            after += 1
        elif not (text.startswith("/", after) and text.startswith('"', end)):
            return None
    elif not text.startswith("/", after):
        return None
    if begin > 0 and text[begin - 1] not in _PATH_START:
        return None
    if _readable_end(text, after, end, True) != end or _glued_after(text, after, end):
        return None
    if text.startswith('"', end) and quoting == UNQUOTED:
        return None
    finish = end + (1 if text.startswith('"', end) else 0)
    path = ("~" if variable == "HOME" else ".") + text[after:end]
    return _Reference(begin, path, True, finish, variable, _unquote(text[begin:finish]))


class _Changes(NamedTuple):
    everywhere: bool
    scopes: list[_Span]
    target: str | None = None
    target_start: int | None = None

    def moved(self, position: int) -> bool:
        return self.everywhere or any(
            span.start <= position < span.end for span in self.scopes
        )


def _directory_changes(
    text: str, lexed: Shell | None, home_mentioned: bool
) -> _Changes:
    if lexed is None:
        return _Changes(_CHANGES_DIRECTORY.search(text) is not None, [])
    words = {
        word.start: (command, index)
        for command in lexed.commands
        for index, word in enumerate(command.words)
        if word.plain
    }
    everywhere = False
    scopes: list[_Span] = []
    for count, match in enumerate(_SCOPED_DIRECTORY.finditer(text)):
        position = match.start()
        owner = words.get(position)
        long = match.group().startswith("--")
        if count >= 64:
            everywhere = True
            break
        if owner is not None:
            command, index = owner
            if long or command.words[0].value in _SCOPING_PROGRAMS:
                start = _scope_start(command, index)
                scopes.append(_Span(start, _command_end(command)))
        elif long or _SCOPING_PROGRAM.search(
            text, _segment_start(text, position), position
        ):
            everywhere = True
    hits = list(_SHELL_DIRECTORY.finditer(text))
    changes = [
        (command, index)
        for command in lexed.commands
        for index, word in enumerate(command.words)
        if word.unquoted in _DIRECTORY_COMMANDS
    ]
    unknown = any(_name_unknown(text, command) for command in lexed.commands)
    if not hits and not changes and not unknown:
        return _Changes(everywhere, scopes)
    if unknown:
        return _Changes(True, scopes)
    if everywhere or len(hits) != 1 or len(changes) != 1:
        return _Changes(True, scopes)
    command, index = changes[0]
    target = _cd_target(command, index, home_mentioned)
    if target is None or not (
        hits[0].start() <= command.words[0].start < hits[0].end()
    ):
        return _Changes(True, scopes)
    if _is_relative(target):
        first = min(c.words[0].start for c in lexed.commands if c.words)
        if first != command.words[0].start:
            return _Changes(True, scopes)
    return _Changes(False, scopes, target, command.words[1].start)


def _name_unknown(text: str, command: Command) -> bool:
    for word in command.words:
        if not _ASSIGNMENT.match(text, word.start, word.end):
            return word.unquoted is None
    return False


def _cd_target(command: Command, index: int, home_mentioned: bool) -> str | None:
    name = command.words[0]
    if index != 0 or len(command.words) != 2 or not (name.plain and name.value == "cd"):
        return None
    target = command.words[1].value
    if target is None or not command.words[1].plain:
        return None
    if (
        target.startswith("/")
        or target in (".", "..")
        or target.startswith(("./", "../"))
    ):
        return target
    if _is_home(target) and not home_mentioned:
        return target
    return None


def _scope_start(command: Command, index: int) -> int:
    option = command.words[index]
    takes_next = re.fullmatch(r"-[A-Za-z]*C|--directory|--chdir", option.value or "")
    if takes_next and index + 1 < len(command.words):
        return command.words[index + 1].end
    return option.end


def _command_end(command: Command) -> int:
    ends = [word.end for word in command.words]
    ends.extend(heredoc.end for heredoc in command.heredocs)
    return max(ends)


def _segment_start(text: str, position: int) -> int:
    return max(text.rfind(c, 0, position) for c in ";&|\n") + 1


class _Spans:
    def __init__(self, spans: Iterable[tuple[int, int]]) -> None:
        self._spans = sorted(spans)
        self._starts = [start for start, _ in self._spans]

    def contains(self, position: int) -> bool:
        return self.overlaps(position, position + 1)

    def overlaps(self, start: int, end: int) -> bool:
        index = bisect.bisect_left(self._starts, end) - 1
        return index >= 0 and self._spans[index][1] > start

    def enclosing(self, start: int, end: int) -> tuple[int, int] | None:
        index = bisect.bisect_right(self._starts, start) - 1
        if index >= 0 and end <= self._spans[index][1]:
            return self._spans[index]
        return None


def _readable_end(text: str, start: int, end: int, shell: bool) -> int | None:
    token = text[start:end]
    if shell and token.endswith("\\"):
        end -= 1
    elif not shell and (escape := re.search(r"\\[ntr]", token)):
        end = start + escape.start()
    token = text[start:end]
    if any(c in token for c in ("\\*?" if shell else "\\*?%")):
        return None
    if shell and text.startswith("[", end):
        if "/" in token or _BRACKET_DIRECTORY.match(text, end):
            return None
        return end
    if shell and text.endswith("]", 0, start):
        return None
    if shell and text.startswith("(", end) and token[-1:] in ("@", "!", "+"):
        return None
    return end


def _glued_after(text: str, start: int, end: int) -> bool:
    if _GLUED_AFTER.match(text, end):
        return True
    return text.startswith("`", end) and text.count("`", 0, start) % 2 == 0


def _glued(text: str, start: int, end: int, shell: bool) -> bool:
    if _glued_after(text, start, end):
        return True
    if not shell and _concatenated(text, start, end):
        return True
    if start < 2 or text[start - 1] not in "'\"":
        return False
    quote, glue = text[start - 1], text[start - 2]
    if not (glue.isalnum() or glue == "_" or glue in _GLUE_BEFORE):
        return False
    prefix = _STRING_PREFIX.search(text, max(0, start - 3), start)
    return prefix is None and not text.endswith(quote * 3, 0, start)


def _concatenated(text: str, start: int, end: int) -> bool:
    after = _CONCATENATED_AFTER.match(text, end)
    if after and not text.startswith(after.group(1) * 3, end):
        return True
    before = _CONCATENATED_BEFORE.search(text, max(0, start - 64), start)
    return before is not None and not text.endswith(text[start - 1] * 3, 0, start)


def _unquote(word: str) -> str:
    for quote in ("'", '"'):
        if word.count(quote) % 2 and (word.startswith(quote) or word.endswith(quote)):
            word = word[1:] if word.startswith(quote) else word[:-1]
    quote = word[:1]
    if quote in ("'", '"') and len(word) > 1 and word.find(quote, 1) == len(word) - 1:
        return word[1:-1]
    return word


def _word_start(text: str, start: int) -> int:
    limit = max(0, start - 256)
    while start > limit and text[start - 1] not in _WORD_STOPS:
        start -= 1
    return start


def _word_end(text: str, end: int) -> int:
    match = _WORD_END.match(text, end)
    return end if match is None else match.end()


def _merge(spans: list[_Span]) -> list[_Span]:
    merged: list[_Span] = []
    for span in sorted(spans):
        if merged and span.start <= merged[-1].end:
            merged[-1] = _Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def path_matches(
    path: str,
    patterns: Iterable[str],
    *,
    cwd: str | None = None,
    home: str | None = None,
) -> bool:
    r"""Whether a path, once normalised, matches any of the glob patterns.

    The path is normalised first, by its text alone (symlinks are not followed): repeated slashes collapse, and `.` and `..` resolve, so `/work/../etc/passwd` is `/etc/passwd` and `/..` is `/`. A relative path is joined to `cwd` when given. With `home`, a leading `~` is replaced by it, in the path, in `cwd` and in each pattern, so `~/.ssh/**` matches `/root/.ssh/id_rsa` when `home="/root"`. Without `home`, `~` is kept as written: `~/x` matches `~/**` and not `/root/**`. `~user` is never expanded. Patterns are not joined to `cwd`: a relative pattern such as `secrets/**` matches only a relative path, so with `cwd` write patterns absolute, or begin them with `**/`.

    A path can keep a `..` after normalising: a relative one without `cwd` (`../etc`), or one that climbs out of `~` without `home` (`~/../etc`). Wildcards never match a `.` or `..` segment, so such a path matches only a pattern that spells its `..` segments out, never `/etc/**`, `~/**` or `**`. An allow-list therefore rejects it. A deny-list would let it through, so a rule that denies should also escalate or reject a call for which `unresolved_paths()` reports anything.

    Patterns are globs: `*` matches any characters within one segment, `?` one character other than `/`, `[seq]` one character in `seq` (ranges such as `a-z` and POSIX classes such as `[:alpha:]` included), `[!seq]` or `[^seq]` one character not in it, and `\` escapes the next character. All of these match a leading `.`, so `/home/*/.ssh` and `/etc/*` see dotfiles. `**` as a whole segment matches any number of segments, and a pattern ending in `/**` also matches the directory itself: `"/etc/**"` matches `/etc` and everything under it, and `"**/*.env"` matches `.env` files anywhere, absolute paths included. Matching is case-sensitive. Braces and `!` at the start of a pattern have no special meaning. A pattern is normalised too: repeated and trailing slashes and `.` segments are removed, and a `..` resolves against the segment before it when that segment has no wildcard (`/work/../etc/**` is `/etc/**`); other `..` segments are kept.

    Args:
        path: The path, such as one from `paths_in()`.
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

    The path is normalised as `path_matches()` does. It cannot be placed when it is relative and there is no `cwd` (`passwd`, `../etc`), when it climbs out of `~` and there is no `home` (`~/../etc`), or when it is `~user/x` or a `file:` URL. The path is otherwise taken literally, as a file tool's argument is: `$`, `*` and `\` are characters of a name. For a call, `unresolved_paths()` reports the paths from `paths_in()` that this cannot place, together with the references in code that name no single path, such as `$HOME/x` or `/etc/pass*`.

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
