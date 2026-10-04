from __future__ import annotations

import re
from typing import NamedTuple

UNQUOTED = 0
LITERAL = 1
DOUBLE = 2


class Word(NamedTuple):
    start: int
    end: int
    value: str | None
    literal: bool
    plain: bool
    unquoted: str | None


class Heredoc(NamedTuple):
    start: int
    end: int
    quoted: bool


class Command(NamedTuple):
    words: list[Word]
    heredocs: list[Heredoc]


class Shell(NamedTuple):
    commands: list[Command]
    quoting: bytearray
    substitution: bool


class LexError(Exception):
    pass


class _Pending(NamedTuple):
    delimiter: str
    quoted: bool
    strip_tabs: bool
    heredocs: list[Heredoc]


_BLANK = " \t"
_WORD_END = " \t\n;&|<>()"
_SPECIAL = set("$`\\*?[]{}'\"")


def lex(text: str) -> Shell | None:
    try:
        return _Lexer(text).run()
    except (LexError, RecursionError):
        return None


class _Code:
    def __init__(self, closer: str) -> None:
        self.closer = closer
        self.parens = 0
        self.cases = 0
        self.words: list[Word] = []
        self.heredocs: list[Heredoc] = []
        self.word_start: int | None = None
        self.parts: list[str] = []
        self.special = False
        self.pieces: list[str] | None = []


class _Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.quoting = bytearray(len(text))
        self.commands: list[Command] = []
        self.substitution = False
        self.pending: list[_Pending] = []

    def run(self) -> Shell:
        end = self.code(0, _Code(""))
        if end != len(self.text):
            raise LexError
        return Shell(self.commands, self.quoting, self.substitution)

    def begin(self, frame: _Code, i: int) -> None:
        if frame.word_start is None:
            frame.word_start = i

    def end_word(self, frame: _Code, i: int) -> None:
        start = frame.word_start
        if start is None:
            return
        raw = self.text[start:i]
        literal = frame.parts in (["single"], ["double"])
        value: str | None = None
        if literal:
            value = raw[1:-1]
        elif frame.parts == ["bare"] and not frame.special:
            value = raw
        unquoted = None if frame.pieces is None else "".join(frame.pieces)
        frame.words.append(Word(start, i, value, literal, value == raw, unquoted))
        if len(frame.words) == 1 and raw == "case":
            frame.cases += 1
        elif raw == "esac" and frame.cases:
            frame.cases -= 1
        frame.word_start = None
        frame.parts = []
        frame.special = False
        frame.pieces = []

    def end_command(self, frame: _Code, i: int) -> None:
        self.end_word(frame, i)
        if frame.words or frame.heredocs:
            self.commands.append(Command(frame.words, frame.heredocs))
        frame.words = []
        frame.heredocs = []

    def code(self, i: int, frame: _Code) -> int:
        text = self.text
        n = len(text)
        while i < n:
            c = text[i]
            if frame.closer == "`" and c == "`":
                self.end_command(frame, i)
                return i + 1
            if c in _BLANK:
                self.end_word(frame, i)
                i += 1
            elif c == "\n":
                self.end_command(frame, i)
                i = self.heredoc_bodies(i + 1)
            elif c == "#" and frame.word_start is None:
                while i < n and text[i] != "\n":
                    self.quoting[i] = LITERAL
                    i += 1
            elif c == ")":
                if frame.parens:
                    frame.parens -= 1
                    self.end_command(frame, i)
                    i += 1
                elif frame.cases:
                    self.end_word(frame, i)
                    i += 1
                elif frame.closer == ")":
                    self.end_command(frame, i)
                    return i + 1
                else:
                    raise LexError
            elif c == "(":
                if frame.word_start is not None:
                    i = self.extglob(frame, i)
                elif text.startswith("(", i + 1):
                    i = self.arithmetic(i)
                else:
                    self.end_command(frame, i)
                    frame.parens += 1
                    i += 1
            elif c in ";&|":
                if c == "&" and text.startswith(">", i + 1):
                    self.end_word(frame, i)
                    i += 2
                    continue
                self.end_command(frame, i)
                i += 1
                while i < n and text[i] in ";&|":
                    i += 1
            elif c in "<>":
                self.end_word(frame, i)
                if text.startswith("(", i + 1):
                    self.substitution = True
                    i = self.code(i + 2, _Code(")"))
                    continue
                if text.startswith("<<", i) and not text.startswith("<<<", i):
                    i = self.heredoc(frame, i + 2)
                    continue
                i += 1
                while i < n and text[i] in "<>&|":
                    i += 1
            else:
                i = self.word_char(frame, i)
        if frame.closer or frame.parens or frame.cases:
            raise LexError
        self.end_command(frame, n)
        return n

    def word_char(self, frame: _Code, i: int) -> int:
        text = self.text
        c = text[i]
        self.begin(frame, i)
        if c == "'":
            close = text.find("'", i + 1)
            if close < 0:
                raise LexError
            self.quoting[i : close + 1] = bytes([LITERAL]) * (close + 1 - i)
            frame.parts.append("single")
            self.piece(frame, text[i + 1 : close])
            return close + 1
        if c == "$" and text.startswith("'", i + 1):
            j = i + 2
            while j < len(text) and text[j] != "'":
                j += 2 if text[j] == "\\" else 1
            if j >= len(text):
                raise LexError
            self.quoting[i : j + 1] = bytes([LITERAL]) * (j + 1 - i)
            frame.parts.append("ansi")
            frame.pieces = None
            return j + 1
        if c == '"' or (c == "$" and text.startswith('"', i + 1)):
            start = i + (1 if c == '"' else 2)
            end, simple = self.double(start)
            if simple and c == '"':
                frame.parts.append("double")
                self.piece(frame, text[start : end - 1])
            else:
                frame.parts.append("other")
                frame.pieces = None
            return end
        if c == "\\":
            self.quoting[i : i + 2] = bytes([LITERAL]) * len(self.quoting[i : i + 2])
            frame.parts.append("other")
            self.piece(frame, text[i + 1 : i + 2])
            return i + 2
        if c == "`" or c == "$":
            frame.parts.append("other")
            frame.pieces = None
            return self.dollar(i)
        if c in _SPECIAL:
            frame.special = True
            frame.pieces = None
        self.piece(frame, c)
        if frame.parts[-1:] != ["bare"]:
            frame.parts.append("bare")
        return i + 1

    def arithmetic(self, i: int) -> int:
        text = self.text
        depth = 0
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            elif c in "'\"\\":
                raise LexError
            elif c == "`" or text.startswith("$(", i):
                self.substitution = True
            i += 1
        raise LexError

    def piece(self, frame: _Code, piece: str) -> None:
        if frame.pieces is not None:
            frame.pieces.append(piece)

    def extglob(self, frame: _Code, i: int) -> int:
        depth = 0
        text = self.text
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    frame.special = True
                    frame.pieces = None
                    frame.parts.append("extglob")
                    return i + 1
            elif c in "'\"":
                close = text.find(c, i + 1)
                if close < 0 or (c == '"' and any(x in text[i:close] for x in "$`\\")):
                    raise LexError
                kind = LITERAL if c == "'" else DOUBLE
                self.quoting[i : close + 1] = bytes([kind]) * (close + 1 - i)
                i = close
            elif c in "\n`$\\":
                raise LexError
            i += 1
        raise LexError

    def dollar(self, i: int) -> int:
        text = self.text
        if text[i] == "`":
            self.substitution = True
            return self.code(i + 1, _Code("`"))
        if text.startswith("((", i + 1):
            return self.arithmetic(i + 1)
        if text.startswith("(", i + 1):
            self.substitution = True
            return self.code(i + 2, _Code(")"))
        if text.startswith("{", i + 1):
            depth = 0
            j = i + 1
            while j < len(text):
                ch = text[j]
                if ch in "'\"`\\\n" or text.startswith("$(", j):
                    raise LexError
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return j + 1
                j += 1
            raise LexError
        return i + 1

    def double(self, i: int) -> tuple[int, bool]:
        text = self.text
        simple = True
        while i < len(text):
            c = text[i]
            if c == '"':
                return i + 1, simple
            if c == "\\":
                simple = False
                if text[i + 1 : i + 2] in ("$", "`", '"', "\\", "\n"):
                    self.quoting[i : i + 2] = bytes([LITERAL]) * 2
                    i += 2
                else:
                    self.quoting[i] = DOUBLE
                    i += 1
            elif c == "`" or c == "$":
                simple = False
                end = self.dollar(i)
                if not text.startswith(("`", "$("), i):
                    self.quoting[i:end] = bytes([DOUBLE]) * (end - i)
                i = end
            else:
                self.quoting[i] = DOUBLE
                i += 1
        raise LexError

    def heredoc(self, frame: _Code, i: int) -> int:
        text = self.text
        strip_tabs = text.startswith("-", i)
        if strip_tabs:
            i += 1
        while i < len(text) and text[i] in _BLANK:
            i += 1
        start = i
        delimiter: list[str] = []
        quoted = False
        while i < len(text) and text[i] not in _WORD_END:
            c = text[i]
            if c in "'\"":
                close = text.find(c, i + 1)
                if close < 0:
                    raise LexError
                delimiter.append(text[i + 1 : close])
                quoted = True
                i = close + 1
            elif c == "\\":
                delimiter.append(text[i + 1 : i + 2])
                quoted = True
                i += 2
            else:
                delimiter.append(c)
                i += 1
        if i == start or any(c in "$`" for c in text[start:i]):
            raise LexError
        self.pending.append(
            _Pending("".join(delimiter), quoted, strip_tabs, frame.heredocs)
        )
        return i

    def heredoc_bodies(self, i: int) -> int:
        text = self.text
        pending, self.pending = self.pending, []
        for heredoc in pending:
            start = i
            end = len(text)
            while i < len(text):
                eol = text.find("\n", i)
                eol = len(text) if eol < 0 else eol
                line = text[i:eol]
                if (
                    line.lstrip("\t") if heredoc.strip_tabs else line
                ) == heredoc.delimiter:
                    end = i
                    i = min(eol + 1, len(text))
                    break
                i = eol + 1
            else:
                i = len(text)
            kind = LITERAL if heredoc.quoted else DOUBLE
            self.quoting[start:end] = bytes([kind]) * (end - start)
            body = text[start:end]
            if not heredoc.quoted and ("$(" in body or "`" in body):
                self.substitution = True
            heredoc.heredocs.append(Heredoc(start, end, heredoc.quoted))
        return i


def sed_executes(script: str) -> bool:
    return _sed_executes(script, True) or _sed_executes(script, False)


_SED_SIMPLE = set("=dDgGhHnNpPxzF")


def _sed_executes(script: str, brackets: bool) -> bool:
    n = len(script)
    i = 0

    def skip(i: int, chars: str = " \t") -> int:
        while i < n and script[i] in chars:
            i += 1
        return i

    def line_end(i: int) -> int:
        while i < n and script[i] != "\n":
            i += 2 if script[i] == "\\" else 1
        return i

    def part(i: int, delimiter: str, in_regex: bool) -> int:
        while i < n:
            c = script[i]
            if c == "\\":
                i += 2
            elif c == delimiter:
                return i + 1
            elif c == "[" and in_regex and brackets:
                i = bracket(i + 1)
                if i < 0:
                    return -1
            else:
                i += 1
        return -1

    def bracket(i: int) -> int:
        if i < n and script[i] == "^":
            i += 1
        if i < n and script[i] == "]":
            i += 1
        while i < n:
            if script[i] == "[" and script[i + 1 : i + 2] in (":", "=", "."):
                close = script.find(script[i + 1] + "]", i + 2)
                if close < 0:
                    return -1
                i = close + 2
            elif script[i] == "]":
                return i + 1
            else:
                i += 1
        return -1

    def address(i: int) -> int:
        if i < n and script[i].isdigit():
            i = skip(i, "0123456789")
            if i < n and script[i] == "~":
                i = skip(i + 1, "0123456789")
            return i
        if i < n and script[i] == "$":
            return i + 1
        if i < n and script[i] in "/\\":
            delimiter = "/"
            if script[i] == "\\":
                if i + 1 >= n:
                    return -1
                delimiter = script[i + 1]
                i += 1
            i = part(i + 1, delimiter, True)
            if i < 0:
                return -1
            return skip(i, "IM")
        return i

    while i < n:
        i = skip(i, " \t\n;")
        if i >= n:
            break
        if script[i] == "#":
            i = line_end(i)
            continue
        i = address(i)
        if i < 0:
            return True
        i = skip(i)
        if i < n and script[i] == ",":
            i = skip(i + 1)
            if i < n and script[i] in "+~":
                i = skip(i + 1, "0123456789")
            else:
                i = address(i)
                if i < 0:
                    return True
        i = skip(i, " \t!")
        if i >= n:
            return True
        command = script[i]
        i += 1
        if command in "{}" or command in _SED_SIMPLE:
            continue
        if command == "e":
            return True
        if command in "qQlL":
            i = skip(skip(i), "0123456789")
        elif command in "aicrRwW":
            i = line_end(i)
        elif command in "btTv:":
            while i < n and script[i] not in ";\n":
                i += 1
        elif command in "sy":
            if i >= n or script[i] in "\n\\":
                return True
            delimiter = script[i]
            i = part(i + 1, delimiter, command == "s")
            if i < 0:
                return True
            i = part(i, delimiter, False)
            if i < 0:
                return True
            while command == "s" and i < n and script[i] in "gpiImMew0123456789":
                if script[i] == "e":
                    return True
                if script[i] == "w":
                    i = line_end(i)
                    break
                i += 1
        else:
            return True
    return False


_LITERAL_PROGRAMS = frozenset(
    {
        "awk",
        "cat",
        "cut",
        "diff",
        "echo",
        "egrep",
        "fgrep",
        "gawk",
        "grep",
        "head",
        "jq",
        "ls",
        "mawk",
        "nawk",
        "nl",
        "printf",
        "rg",
        "sed",
        "sort",
        "tail",
        "tee",
        "tr",
        "uniq",
        "wc",
    }
)
_INERT = frozenset({"cd", "true", "false", ":"})
_FORBIDDEN_OPTIONS = {
    "printf": ("-v",),
    "rg": ("--pre",),
    "sort": ("--co", "--fil"),
    "wc": ("--fil",),
}
_AWK = frozenset({"awk", "gawk", "mawk", "nawk"})


_CLIMBS = re.compile(r"(?<![\w.])\.\.(?![\w.])")


def literal_spans(text: str, shell: Shell) -> list[tuple[int, int]]:
    if shell.substitution or not all(_inert(command) for command in shell.commands):
        return []
    spans: list[tuple[int, int]] = []
    for command in shell.commands:
        if not command.words or command.words[0].value not in _LITERAL_PROGRAMS:
            continue
        spans.extend((w.start + 1, w.end - 1) for w in command.words[1:] if w.literal)
        spans.extend((h.start, h.end) for h in command.heredocs if h.quoted)
    return [span for span in spans if not _CLIMBS.search(text, *span)]


def _inert(command: Command) -> bool:
    if not command.words:
        return True
    name = command.words[0]
    if not name.plain or name.value not in _LITERAL_PROGRAMS | _INERT:
        return False
    arguments = command.words[1:]
    forbidden = _FORBIDDEN_OPTIONS.get(name.value, ())
    if any(w.value is not None and w.value.startswith(forbidden) for w in arguments):
        return False
    if any(w.value is None and not w.literal for w in arguments) and (
        name.value == "sed" or name.value in _AWK or forbidden
    ):
        return False
    if name.value == "sed":
        scripts = _sed_scripts(arguments)
        return scripts is not None and not any(sed_executes(s) for s in scripts)
    if name.value in _AWK:
        return _awk_inert(arguments)
    return True


def _sed_scripts(arguments: list[Word]) -> list[str] | None:
    scripts: list[str] = []
    operands: list[str] = []
    values = [w.value for w in arguments]
    i = 0
    while i < len(values):
        value = values[i]
        i += 1
        if value is None:
            return None
        if value == "--":
            operands.extend(v for v in values[i:] if v is not None)
            break
        if value.startswith("--"):
            if value.startswith("--e"):
                if "=" in value:
                    scripts.append(value.split("=", 1)[1])
                elif i < len(values) and values[i] is not None:
                    scripts.append(values[i] or "")
                    i += 1
                else:
                    return None
            elif value.startswith(("--f", "--l")):
                return None
            continue
        if value.startswith("-") and value != "-":
            for j, flag in enumerate(value[1:], start=1):
                if flag == "f":
                    return None
                if flag in "el":
                    rest = value[j + 1 :]
                    if not rest:
                        if i >= len(values) or values[i] is None:
                            return None
                        rest = values[i] or ""
                        i += 1
                    if flag == "e":
                        scripts.append(rest)
                    break
                if flag == "i":
                    break
            continue
        operands.append(value)
    if not scripts:
        if not operands:
            return None
        scripts.append(operands[0])
    return scripts


def _awk_inert(arguments: list[Word]) -> bool:
    expect_value = False
    for word in arguments:
        value = word.value
        if value is None:
            return False
        if any(marker in value for marker in ("system", "|", "@")):
            return False
        if expect_value:
            expect_value = False
        elif value in ("-F", "-v"):
            expect_value = True
        elif (
            value.startswith("-")
            and not value.startswith(("-F", "-v"))
            and value not in ("-", "--")
        ):
            return False
    return True
