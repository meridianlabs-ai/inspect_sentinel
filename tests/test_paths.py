import sys
from pathlib import PurePosixPath
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from inspect_ai.tool import ToolCall
from wcmatch.glob import escape

from inspect_sentinel import path_matches, path_resolves, paths_in
from inspect_sentinel._rules import _normalize


def _call(function: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="c1", function=function, arguments=arguments)


def _bash(command: str) -> ToolCall:
    return _call("bash", command=command)


@pytest.mark.parametrize(
    "call, expected",
    [
        # file tools: their path arguments, as written
        (_call("text_editor", command="view", path="w/f.py"), ["w/f.py"]),
        (_call("text_editor", command="create", path="/w/f", file_text="/x"), ["/w/f"]),
        (_call("memory", command="view", path="/memories"), ["/memories"]),
        (
            _call("memory", command="rename", old_path="/m/a", new_path="/m/b"),
            ["/m/a", "/m/b"],
        ),
        (_call("read_file", file_path="/w/f", offset=1), ["/w/f"]),
        (_call("read_file", file_path="~/.ssh/id_rsa"), ["~/.ssh/id_rsa"]),
        (_call("list_files", path=".", depth=2), ["."]),
        (_call("grep", pattern="/etc/x", path="/w", glob="/y/*"), ["/w"]),
        (_call("grep", pattern="x"), []),
        (_call("text_editor", command="view", path=""), []),
        (_call("text_editor", command="view", path=3), []),
        # code tools: tokens in the text
        (_bash("cat /etc/passwd ~/.aws/creds"), ["/etc/passwd", "~/.aws/creds"]),
        (_bash("cd ~ && ls"), ["~"]),
        (_bash("rm -rf ~/"), ["~/"]),
        (_bash("rm -rf /"), ["/"]),
        (_bash("cp '/a' \"/c\";ls>/d"), ["/a", "/c", "/d"]),
        (_bash("cat `/x`"), ["/x"]),
        (_bash("echo $(cat /y)"), ["/y"]),
        (_bash("(cd /x)"), ["/x"]),
        (_bash("cat /a|grep x"), ["/a"]),
        (_bash("cmd 2>/e </f"), ["/e", "/f"]),
        (_bash("ls {/a,/b}"), ["/a", "/b"]),
        (_bash("tar --file=/x.tar a"), ["/x.tar"]),
        (_bash("PATH=/usr/bin:/bin x"), ["/usr/bin", "/bin"]),
        (_bash("scp host:/etc/x ."), ["/etc/x"]),
        (_bash("cat //etc/passwd"), ["//etc/passwd"]),
        (_bash("cat /a /a"), ["/a"]),
        (_bash("cat /données/été.txt"), ["/données/été.txt"]),
        (_call("bash", cmd="cat /x"), ["/x"]),
        (_call("python", code="open('/etc/hosts').read()"), ["/etc/hosts"]),
        (_call("python", code='Path("~/.ssh").expanduser()'), ["~/.ssh"]),
        (_call("python", code='open(f"/x/{name}")'), ["/x/"]),
        (_call("code_execution", code="open('/w/f')"), ["/w/f"]),
        (_call("bash_session", action="type_submit", input="ls /w"), ["/w"]),
        (_call("bash_session", action="read"), []),
        # documented misses
        (_bash("cat ./a ../b c/d"), []),
        (_bash("cat ~user/e"), []),
        (_bash("cat a~/x"), []),
        (_bash("cat -f/x"), []),
        (_bash("cat $HOME/x ${HOME}/y $(pwd)/z"), []),
        (_bash("curl https://x.org/a file:///b"), []),
        (_call("think", thought="look at /etc"), []),
        (_call("web_search", query="/etc/passwd"), []),
        (_call("mine", path="/w"), []),
        # documented misreads
        (_bash('cat "/a b"'), ["/a"]),
        (_bash('cat "$HOME"/x'), ["/x"]),
        (_bash("echo $((4 / 2))"), ["/"]),
        (_call("python", code="x = 1 // 2"), ["//"]),
        (_bash("cat /etc/pass*"), ["/etc/pass*"]),
    ],
)
def test_paths_in(call: ToolCall, expected: list[str]) -> None:
    assert paths_in(call) == expected


def test_a_wildcard_found_in_a_command_is_compared_as_text() -> None:
    [path] = paths_in(_bash("cat /etc/pass*"))
    assert not path_matches(path, ["/etc/passwd"])
    assert path_matches(path, ["/etc/**"])


@pytest.mark.parametrize(
    "path, patterns, cwd, expected",
    [
        # `..` resolves
        ("/work/../etc/passwd", ["/etc/**"], None, True),
        ("/work/../etc/passwd", ["/work/**"], None, False),
        ("/work/sub/../../etc/shadow", ["/work/**"], None, False),
        ("/a/b/../../etc/x", ["/etc/*"], None, True),
        ("/a/./../etc/x", ["/etc/x"], None, True),
        ("/../etc", ["/etc"], None, True),
        ("/../../../etc/passwd", ["/etc/passwd"], None, True),
        ("/a/../../..", ["/"], None, True),
        ("/work/..", ["/"], None, True),
        ("/work/..", ["/*"], None, False),
        ("../etc/passwd", ["/etc/**"], "/work", True),
        ("../../../../etc", ["/etc"], "/work", True),
        ("a/../../etc", ["/etc"], "/work", True),
        ("a/../../etc", ["/etc"], "/w/x", False),
        ("a/../../etc", ["/w/etc"], "/w/x", True),
        # `..` that cannot resolve matches only where spelled out
        ("~/../etc/passwd", ["~/**"], None, False),
        ("~/../passwd", ["~/*/passwd"], None, False),
        ("~/../etc/passwd", ["/etc/**"], None, False),
        ("~/../etc/passwd", ["**"], None, False),
        ("~/../etc/passwd", ["~/*/etc/passwd", "~/?/etc/passwd"], None, False),
        ("~/../etc/passwd", ["~/.*/etc/passwd", "~/[.][.]/etc/*"], None, False),
        ("~/x/../../etc/passwd", ["~/**"], None, False),
        ("~/../etc/passwd", ["~/../etc/passwd"], None, True),
        ("~/../etc/passwd", ["~/../etc/*"], None, True),
        ("~/../../etc", ["~/../**"], None, False),
        ("~/../../etc", ["~/../../*"], None, True),
        ("../etc/passwd", ["/etc/**"], None, False),
        ("../etc/passwd", ["**"], None, False),
        ("../etc/passwd", ["*/etc/passwd"], None, False),
        ("../etc/passwd", ["../etc/**"], None, True),
        ("../../etc", ["../**"], None, False),
        ("a/../../etc", ["**"], None, False),
        ("..", ["**"], None, False),
        ("..", [".."], None, True),
        # `.` segments
        ("/etc/./ssh/key", ["/etc/*"], None, False),
        ("/etc/./ssh/key", ["/etc/*/*"], None, True),
        ("/etc/.", ["/etc"], None, True),
        ("./a/../b", ["b"], None, True),
        ("./x", ["x"], None, True),
        (".", ["**"], None, False),
        (".", ["."], None, True),
        (".", ["/w"], "/w", True),
        # repeated and trailing slashes
        ("//etc///passwd", ["/etc/*"], None, True),
        ("/etc/passwd/", ["/etc/passwd"], None, True),
        ("/etc/", ["/etc"], None, True),
        ("///", ["/"], None, True),
        ("//", ["/**"], None, True),
        ("/etc/passwd", ["/etc//passwd"], None, True),
        ("/a/b", ["/a/b/"], None, True),
        ("/a/b", ["/a/b//"], None, True),
        ("/x/y", ["/x/**/"], None, True),
        ("/x", ["/x/**/"], None, True),
        ("/a/b", ["/a/./b", "./a"], None, True),
        ("a", ["./a"], None, True),
        ("/", ["/."], None, True),
        ("", ["."], None, True),
        ("", ["./"], None, True),
        # root
        ("/", ["/"], None, True),
        ("/a", ["/"], None, False),
        ("/", ["/**"], None, True),
        ("/", ["**"], None, True),
        ("/", ["/*"], None, False),
        ("/a", ["/*"], None, True),
        ("/a/b", ["/*"], None, False),
        # empty
        ("", ["**"], None, False),
        ("", [""], None, False),
        ("/a", [""], None, False),
        ("", ["/w"], "/w", True),
        # relative paths, with and without `cwd`
        ("w/f.py", ["/work/**"], None, False),
        ("w/f.py", ["/work/**"], "/work", True),
        ("f.py", ["*.py"], None, True),
        ("f.py", ["*.py"], "/work", False),
        ("f.py", ["/work/*.py"], "/work", True),
        ("f.py", ["/work/f.py"], "/work/", True),
        ("f.py", ["/work/f.py"], "/work/./sub/..", True),
        ("f.py", ["/work/f.py"], "//work", True),
        ("f", ["w/*"], "w", True),
        ("/etc/passwd", ["/etc/**"], "/work", True),
        # `~` is literal without `home`
        ("~", ["~"], None, True),
        ("~", ["~/**"], None, True),
        ("~/", ["~"], None, True),
        ("~/.ssh/id_rsa", ["~/.ssh/**"], None, True),
        ("~/.ssh/id_rsa", ["/root/.ssh/**"], None, False),
        ("/root/.ssh/id_rsa", ["~/.ssh/**"], None, False),
        ("~/x/../.aws/credentials", ["~/.aws/**"], None, True),
        ("~//x", ["~/x"], None, True),
        (".aws/c", ["~/.aws/**"], "~", True),
        ("~/a", ["~/a"], "/work", True),
        # `~user` is a name like any other
        ("~user/x", ["~/**"], None, False),
        ("~user/x", ["~user/*"], None, True),
        ("~user/x", ["/w/~user/x"], "/w", True),
        ("~x", ["/w/~x"], "/w", True),
        # unicode and spaces
        ("/données/été.txt", ["/données/*.txt"], None, True),
        ("/a b/c d", ["/a b/*"], None, True),
        ("/a b/c d", ["/a?b/c?d"], None, True),
        ("/日本/ファイル", ["/日本/????"], None, True),
        ("/日本/ファイル", ["/日本/???"], None, False),
        ("/emoji/😀.txt", ["/emoji/?.txt"], None, True),
        # dotfiles match wildcards
        ("/home/u/.ssh", ["/home/u/*"], None, True),
        ("/home/u/.ssh/id", ["/home/*/.ssh/*"], None, True),
        ("/home/.u/.ssh/id", ["/home/*/*/id"], None, True),
        ("/home/u/.ssh/id", ["/home/**"], None, True),
        ("/home/u/.ssh/id", ["**/.ssh/**"], None, True),
        ("/home/u/.ssh/id", ["/home/u/?ssh/id"], None, True),
        ("/w/.env", ["**/*.env"], None, True),
        ("/w/.git/config", ["/w/**/config"], None, True),
        ("/w/..hidden", ["/w/*"], None, True),
        ("/w/...", ["/w/*"], None, True),
        # `**`
        ("/w/a/b/c.env", ["**/*.env"], None, True),
        ("c.env", ["**/*.env"], None, True),
        ("/c.env", ["**/*.env"], None, True),
        ("/w/a/b/c.py", ["/w/**/c.py"], None, True),
        ("/w/c.py", ["/w/**/c.py"], None, True),
        ("/x/w/c.py", ["/w/**/c.py"], None, False),
        ("/etc/passwd", ["/etc/**"], None, True),
        ("/etc/a/b", ["/etc/**"], None, True),
        ("/etc", ["/etc/**"], None, True),
        ("/etcetera", ["/etc/**"], None, False),
        ("/", ["/etc/**"], None, False),
        ("/a/b", ["**"], None, True),
        ("a", ["**"], None, True),
        ("~/x", ["**"], None, True),
        ("~", ["**"], None, True),
        ("/a/c", ["/a/**/**/c"], None, True),
        ("/a/x/b", ["/a/**/b/**"], None, True),
        ("/a", ["/a/**/**"], None, True),
        ("/a/bc", ["/a/b**"], None, True),
        ("/a/b/c", ["/a/b**"], None, False),
        ("/a/b/c", ["/a/b**/c"], None, True),
        # character classes
        ("/etc/passwd", ["/etc/[ps]*"], None, True),
        ("/etc/shadow", ["/etc/[ps]*"], None, True),
        ("/etc/hosts", ["/etc/[ps]*"], None, False),
        ("/etc/passwd", ["/etc/[!p]*"], None, False),
        ("/etc/hosts", ["/etc/[!p]*"], None, True),
        ("/etc/hosts", ["/etc/[^p]*"], None, True),
        ("/v/a3", ["/v/a[0-9]"], None, True),
        ("/v/ax", ["/v/a[0-9]"], None, False),
        ("/a/b", ["/a[/]b"], None, False),
        ("/a/.b", ["/a/[.]b"], None, True),
        ("/a/-", ["/a/[a-]"], None, True),
        # escaping
        ("/a/*", ["/a/\\*"], None, True),
        ("/a/b", ["/a/\\*"], None, False),
        ("/a/*", ["/a/[*]"], None, True),
        ("/a/b", ["/a/[*]"], None, False),
        ("/a/?", ["/a/[?]"], None, True),
        ("/a/b", ["/a/[?]"], None, False),
        ("/a/[x]", ["/a/\\[x]"], None, True),
        ("/a/x", ["/a/\\[x]"], None, False),
        ("/a/[x]", ["/a/[[]x]"], None, True),
        ("/a/[x", ["/a/[x"], None, True),
        ("/a/\\b", ["/a/\\\\b"], None, True),
        # no braces, negation or case folding
        ("/etc/passwd", ["/etc/{passwd,shadow}"], None, False),
        ("/etc/{a,b}", ["/etc/{a,b}"], None, True),
        ("/etc/passwd", ["!/etc/shadow"], None, False),
        ("/a|b", ["/a|b"], None, True),
        ("/ETC/passwd", ["/etc/**"], None, False),
        ("/etc/passwd", ["/ETC/**"], None, False),
        # several patterns
        ("/w/x", [], None, False),
        ("/w/x", ["/tmp/**", "/w/*"], None, True),
        ("/c", ["/a", "/b"], None, False),
        ("/b", iter(["/a", "/b"]), None, True),
    ],
)
def test_path_matches(
    path: str, patterns: list[str], cwd: str | None, expected: bool
) -> None:
    assert path_matches(path, patterns, cwd=cwd) is expected


@pytest.mark.parametrize(
    "path, patterns, cwd, home, expected",
    [
        ("~/.ssh/id", ["/root/.ssh/**"], None, "/root", True),
        ("~/.ssh/id", ["~/.ssh/**"], None, "/root", True),
        ("/root/.ssh/id", ["~/.ssh/**"], None, "/root", True),
        ("/home/u/.ssh/id", ["~/.ssh/**"], None, "/root", False),
        ("~/../etc/passwd", ["/etc/**"], None, "/root", True),
        ("~/../etc/passwd", ["~/**"], None, "/root", False),
        ("~/../../../etc", ["/etc"], None, "/home/u", True),
        ("~", ["/root"], None, "/root", True),
        ("~", ["~/**"], None, "/root", True),
        ("~", ["/"], None, "/", True),
        ("~/x", ["/x"], None, "/", True),
        ("~/x", ["~/x", "/root/x"], None, "/root/", True),
        ("~/x", ["~/*"], None, "/home/[a]", True),
        ("/home/a/x", ["~/*"], None, "/home/[a]", False),
        ("/home/*/x", ["~/x"], None, "/home/*", True),
        ("/home/u/x", ["~/x"], None, "/home/*", False),
        ("f", ["/root/w/*"], "~/w", "/root", True),
        ("../.ssh/k", ["~/.ssh/*"], "~/w", "/root", True),
        ("f", ["/work/f"], "/work", "/root", True),
        ("~/f", ["/root/f"], "/work", "/root", True),
        ("~user/x", ["~user/**"], None, "/root", True),
        ("~user/x", ["/root/**"], None, "/root", False),
        ("~user/x", ["/w/~user/x"], "/w", "/root", True),
        ("x/~/y", ["/w/x/~/y"], "/w", "/root", True),
    ],
)
def test_path_matches_with_home(
    path: str, patterns: list[str], cwd: str | None, home: str, expected: bool
) -> None:
    assert path_matches(path, patterns, cwd=cwd, home=home) is expected


@pytest.mark.parametrize("home", ["root", "~", "", "~/x", "./root"])
def test_home_must_be_absolute(home: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        path_matches("/x", ["/x"], home=home)
    with pytest.raises(ValueError, match="absolute"):
        path_resolves("/x", home=home)


@pytest.mark.parametrize(
    "path, cwd, home, expected",
    [
        ("/etc/passwd", None, None, True),
        ("/work/../../etc", None, None, True),
        ("~/x", None, None, True),
        ("~/x/..", None, None, True),
        ("~/..", None, None, False),
        ("~/../etc", None, None, False),
        ("~/a/../../etc", None, None, False),
        ("~/../etc", None, "/root", True),
        ("a/b", None, None, True),
        ("a/../b", None, None, True),
        ("..", None, None, False),
        ("../etc", None, None, False),
        ("a/../../etc", None, None, False),
        ("../etc", "/work", None, True),
        ("../../etc", "w", None, False),
        ("../etc", "~", None, False),
        ("../etc", "~/w", None, True),
        ("../etc", "~", "/root", True),
        ("/a/..b", None, None, True),
        ("..b/c", None, None, True),
        ("", None, None, True),
    ],
)
def test_path_resolves(
    path: str, cwd: str | None, home: str | None, expected: bool
) -> None:
    assert path_resolves(path, cwd=cwd, home=home) is expected


def test_a_deny_list_rejects_paths_it_cannot_resolve() -> None:
    denied = ["/etc/**", "~/.ssh/**"]

    def allowed(path: str) -> bool:
        return path_resolves(path) and not path_matches(path, denied)

    assert allowed("/work/f")
    assert allowed("~/notes")
    assert not allowed("/work/../etc/passwd")
    assert not allowed("~/.ssh/id_rsa")
    assert not allowed("~/../etc/passwd")
    assert not allowed("../etc/passwd")


# Properties. A path is built from segments; `_SEGMENT` holds no `.` or `..`.

_PROPERTY_SETTINGS = settings(derandomize=True, database=None, max_examples=300)

_SEGMENT = st.text(
    alphabet=st.sampled_from([*"ab.-_ *?[]!", "é", "日", "\\"]), min_size=1, max_size=4
).filter(lambda s: s not in (".", ".."))
_PART = st.one_of(_SEGMENT, _SEGMENT, st.just("."), st.just(".."))
_ANCHOR = st.sampled_from(["/", "", "~/"])
_SEPARATOR = st.sampled_from(["/", "//", "/./"])


@st.composite
def _paths(draw: st.DrawFn) -> str:
    return draw(_ANCHOR) + "/".join(draw(st.lists(_PART, max_size=6)))


@st.composite
def _patterns(draw: st.DrawFn) -> str:
    token = st.one_of(
        st.sampled_from([*"ab.-é", "*", "?", "[ab]", "[!a]", "[.]", "[a-c]"]),
        st.just("\\*"),
    )
    segment = st.one_of(
        st.just("**"),
        st.lists(token, min_size=1, max_size=3).map("".join),
        st.just(".."),
    )
    return draw(_ANCHOR) + "/".join(draw(st.lists(segment, min_size=1, max_size=4)))


def _resolve(parts: list[str], base: list[str]) -> list[str]:
    stack = list(base)
    for part in parts:
        if part == "..":
            if stack:
                stack.pop()
        elif part not in ("", "."):
            stack.append(part)
    return stack


@_PROPERTY_SETTINGS
@given(
    _paths(), st.sampled_from([None, "/w", "~/w", "w"]), st.sampled_from([None, "/h"])
)
def test_normalising_is_idempotent(
    path: str, cwd: str | None, home: str | None
) -> None:
    once = _normalize(path, cwd, home)
    assert _normalize(once, None, None) == once


@_PROPERTY_SETTINGS
@given(
    _ANCHOR,
    st.lists(_PART, min_size=1, max_size=6),
    st.data(),
    st.lists(_patterns(), min_size=1, max_size=3),
)
def test_redundant_separators_do_not_change_a_match(
    anchor: str, parts: list[str], data: st.DataObject, patterns: list[str]
) -> None:
    plain = anchor + "/".join(parts)
    padded = anchor + parts[0]
    for part in parts[1:]:
        padded += data.draw(_SEPARATOR) + part
    padded = data.draw(st.sampled_from(["", "./"])) + padded if not anchor else padded
    padded += data.draw(st.sampled_from(["", "/", "/.", "//"]))
    assert _normalize(padded, None, None) == _normalize(plain, None, None)
    for pattern in patterns:
        assert path_matches(padded, [pattern]) == path_matches(plain, [pattern])
    assert path_resolves(padded) == path_resolves(plain)


@_PROPERTY_SETTINGS
@given(st.lists(_PART, max_size=8), _SEGMENT)
def test_an_absolute_path_matches_only_the_directory_it_resolves_into(
    parts: list[str], root: str
) -> None:
    path = "/" + "/".join(parts)
    resolved = _resolve(parts, [])
    pattern = f"/{escape(root)}/**"
    assert path_matches(path, [pattern]) == (resolved[:1] == [root])
    assert path_matches(path, ["/" + "/".join(escape(p) for p in resolved)])
    assert path_resolves(path)


@_PROPERTY_SETTINGS
@given(st.lists(_PART, max_size=8), _SEGMENT)
def test_a_home_path_resolves_into_home_or_not_at_all(
    parts: list[str], root: str
) -> None:
    path = "~/" + "/".join(parts)
    climbs_out = _climbs_out(parts)
    assert path_resolves(path) is not climbs_out
    assert path_matches(path, ["~/**"]) is not climbs_out
    assert not path_matches(path, ["/**", f"/{escape(root)}/**"])
    if climbs_out:
        assert not path_matches(path, ["**", "*/**", "~/*/**"])


@_PROPERTY_SETTINGS
@given(
    st.lists(_PART, max_size=8), st.lists(_SEGMENT, min_size=1, max_size=3), _SEGMENT
)
def test_home_resolves_a_home_path_as_an_absolute_one(
    parts: list[str], home_parts: list[str], root: str
) -> None:
    home = "/" + "/".join(home_parts)
    path = "~/" + "/".join(parts)
    resolved = _resolve(parts, home_parts)
    assert path_resolves(path, home=home)
    assert path_matches(path, [f"/{escape(root)}/**"], home=home) == (
        resolved[:1] == [root]
    )
    assert path_matches(path, [escape(home) + "/**"], home=home) == (
        resolved[: len(home_parts)] == home_parts
    )


@_PROPERTY_SETTINGS
@given(st.lists(_PART, min_size=1, max_size=8))
def test_a_relative_path_that_climbs_out_matches_no_wildcard(parts: list[str]) -> None:
    path = "/".join(parts)
    climbs_out = _climbs_out(parts)
    assert path_resolves(path) is not climbs_out
    if climbs_out:
        assert not path_matches(path, ["**", "/**", "*/**", "~/**"])


def _climbs_out(parts: list[str]) -> bool:
    depth = 0
    for part in parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                return True
        elif part != ".":
            depth += 1
    return False


_ORACLE_SEGMENT = st.text(
    alphabet=st.sampled_from([*"ab.-!", "é"]), min_size=1, max_size=4
).filter(lambda s: s not in (".", ".."))


@st.composite
def _oracle_cases(draw: st.DrawFn) -> tuple[str, str]:
    anchor = draw(st.sampled_from(["/", ""]))
    parts = draw(
        st.lists(st.one_of(_ORACLE_SEGMENT, st.just(".")), min_size=1, max_size=5)
    )
    path = anchor + "/".join(parts)
    token = st.sampled_from([*"ab.-é", "*", "?", "[ab]", "[!a]", "[.]", "[a-c]"])
    random_segment = st.one_of(
        st.just("**"), st.lists(token, min_size=1, max_size=3).map("".join)
    )
    segments: list[str] = []
    for part in (p for p in parts if p != "."):
        if draw(st.booleans()):
            segments.extend(draw(st.lists(random_segment, max_size=2)))
        else:
            segments.append(
                draw(
                    st.sampled_from(
                        [
                            part,
                            "*",
                            "**",
                            part[0] + "*",
                            "*" + part[-1],
                            "?" * len(part),
                            f"[{part[0]}]{part[1:]}",
                            f"[!{part[0]}]{part[1:]}",
                        ]
                    )
                )
            )
    pattern_anchor = draw(st.sampled_from([anchor, anchor, "/", ""]))
    pattern = pattern_anchor + "/".join(segments)
    assume(segments and not pattern.endswith("/**") and pattern != "**")
    assume("***" not in pattern and not (anchor and pattern.startswith("**/")))
    return path, pattern


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="PurePosixPath.full_match is new in 3.13"
)
@settings(derandomize=True, database=None, max_examples=1000)
@given(_oracle_cases())
def test_agrees_with_pathlib_full_match(case: tuple[str, str]) -> None:
    # Excluded from the oracle, as intended differences: `..` segments, which
    # wildcards never match here; a trailing `/**`, which here also matches the
    # directory itself; `\` escapes; `[^...]` as negation; a leading `**/`
    # against an absolute path, which here also matches at the root (`/a`); and
    # runs of three or more `*`, where pathlib matches `/***` against the root.
    path, pattern = case
    if sys.version_info >= (3, 13):
        expected = PurePosixPath(path).full_match(pattern)
        assert path_matches(path, [pattern]) is expected
