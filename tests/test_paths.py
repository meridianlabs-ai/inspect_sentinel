import sys
from pathlib import PurePosixPath
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from inspect_ai.tool import ToolCall
from wcmatch.glob import escape

from inspect_sentinel import (
    Action,
    BeforeToolCall,
    Context,
    Decision,
    HumanAnswer,
    Protocol,
    before_tool_call,
    call_text,
    human,
    path_matches,
    path_resolves,
    paths_in,
    protocol,
    sequential,
    unresolved_paths,
)
from inspect_sentinel._rules import _normalize
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._shell import sed_executes
from tests._benign_corpus import BASH, PYTHON
from tests._fakes import FakeHost, host_context


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
        (_bash("tar -C/etc -xf a.tar"), ["/etc"]),
        (_bash("PATH=/usr/bin:/bin x"), ["/usr/bin", "/bin"]),
        (_bash("scp host:/etc/x /w"), ["/etc/x", "/w"]),
        (_bash("cat //etc/passwd"), ["//etc/passwd"]),
        (_bash("cat /a /a"), ["/a"]),
        (_bash("cat /données/été.txt"), ["/données/été.txt"]),
        (_bash('bash -c "cat \\"/etc/x\\""'), ["/etc/x"]),
        (_bash("cat /x\\\n /y"), ["/x", "/y"]),
        (_bash("cat /root/.ss\\\nh/id"), ["/root/.ssh/id"]),
        (_bash("awk '{print $1}' /w/x"), ["/w/x"]),
        (_bash("grep -E '^/usr/bin$' /w/x"), ["/usr/bin", "/w/x"]),
        (_call("bash", cmd="cat /x"), ["/x"]),
        (_call("python", code="open('/etc/hosts').read()"), ["/etc/hosts"]),
        (_call("python", code='Path("~/.ssh").expanduser()'), ["~/.ssh"]),
        (_call("python", code='open(f"/etc/hosts")'), ["/etc/hosts"]),
        (_call("python", code="Path(r'/etc/x'), b'/y'"), ["/etc/x", "/y"]),
        (_call("python", code='p = """/etc/x"""'), ["/etc/x"]),
        (_call("python", code="d = {'/a': 1, \"/b\": [2]}"), ["/a", "/b"]),
        (_call("python", code='print("/w/out\\n")'), ["/w/out"]),
        (_call("python", code='"/".join(parts)'), ["/"]),
        (_call("code_execution", code="open('/w/f')"), ["/w/f"]),
        (_call("bash_session", action="type_submit", input="ls /w"), ["/w"]),
        (_call("bash_session", action="read"), []),
        # relative paths: words with a `/` in a shell, string literals in Python
        (_bash("cat ./a ../b c/d"), ["./a", "../b", "c/d"]),
        (_bash("ls . .. .x"), [".", ".."]),
        (_bash("git diff origin/main"), ["origin/main"]),
        (_bash("cat a~/x"), ["a~/x"]),
        (_call("python", code="open('data/x.csv')"), ["data/x.csv"]),
        (_call("python", code="x = a/b + c/d"), []),
        (_call("python", code="'.'.join(x) # see a/b"), []),
        # left to unresolved_paths()
        (_bash("cat $HOME/x ${HOME}/y $(pwd)/z `pwd`/w"), ["~/x", "~/y"]),
        (_bash('cat "$HOME"/x "$HOME/y" /home/$USER/z'), ["~/x", "~/y"]),
        (_bash("cp '$HOME/x' '~/y' \"~/z\""), []),
        (_bash("cat '$HOME/x' '~/y' \"~/z\""), ["$HOME/x", "./~/y", "./~/z"]),
        (_call("bash_session", input="cat $HOME/x"), []),
        (_bash("cat /etc/pass* /e?c/passwd /[e]tc/passwd */x"), []),
        (_bash("cat ~user/e /e\\tc/passwd '/e'tc/passwd"), []),
        (_bash("curl https://x.org/a file:///b"), []),
        (_bash("cd /w && cat a/b"), ["/w", "a/b", "/w/a/b"]),
        (_call("python", code='open(f"/x/{name}")'), []),
        (_call("python", code="home + '/.ssh'"), []),
        # documented misses
        (_bash("cat passwd"), []),
        (_bash("cat $F"), []),
        (_call("think", thought="look at /etc"), []),
        (_call("web_search", query="/etc/passwd"), []),
        (_call("mine", path="/w"), []),
        # documented misreads
        (_bash('cat "/a b"'), ["/a"]),
        (_bash("echo $((4 / 2))"), ["/"]),
        (_call("python", code="x = 1 // 2"), ["//"]),
    ],
)
def test_paths_in(call: ToolCall, expected: list[str]) -> None:
    assert paths_in(call) == expected


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
        # `..` in a pattern resolves after a segment without wildcards
        ("/etc/passwd", ["/work/../etc/**"], None, True),
        ("/etc/passwd", ["/a/b/../../etc/passwd"], None, True),
        ("/etc", ["/../etc"], None, True),
        ("/etc", ["/a/../../etc"], None, True),
        ("/etc/passwd", ["/*/../etc/passwd"], None, False),
        ("/etc/passwd", ["/**/../etc/passwd"], None, False),
        ("/etc/passwd", ["/[w]/../etc/passwd"], None, False),
        ("/x/../etc", ["/*/../etc"], None, False),
        ("a/../../etc", ["../etc"], None, True),
        ("../etc", ["a/../../etc"], None, True),
        ("../../etc", ["../../etc"], None, True),
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
        ("secrets/k", ["secrets/**"], "/work", False),
        ("secrets/k", ["**/secrets/**"], "/work", True),
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
        ("~/.ssh/id", ["~/.ssh/**"], None, "/home/u/../u", True),
        ("~/.ssh/id", ["/home/u/.ssh/**"], None, "/home/u/../u", True),
        ("/home/u/.ssh/id", ["~/.ssh/**"], None, "/home/./u/", True),
        ("/x", ["~/x"], None, "/root/..", True),
        ("~/x", ["/x"], None, "//", True),
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
        ("a/b", None, None, False),
        ("a/b", "/w", None, True),
        ("a/../b", None, None, False),
        ("a/../b", "/w", None, True),
        ("passwd", None, None, False),
        ("~root/.ssh/id", None, None, False),
        ("~root/.ssh/id", "/w", None, False),
        ("~", None, None, True),
        ("file:///etc/passwd", "/w", None, False),
        ("/w/$x", None, None, True),
        ("/w/a*b", None, None, True),
        ("~root/.ssh/id", None, "/root", False),
        ("../etc/passwd", "work", None, False),
        ("..", None, None, False),
        ("../etc", None, None, False),
        ("a/../../etc", None, None, False),
        ("../etc", "/work", None, True),
        ("../../etc", "w", None, False),
        ("../etc", "~", None, False),
        ("../etc", "~/w", None, True),
        ("../etc", "~", "/root", True),
        ("/a/..b", None, None, True),
        ("..b/c", None, None, False),
        ("..b/c", "/w", None, True),
        ("", None, None, False),
        ("", "/w", None, True),
    ],
)
def test_path_resolves(
    path: str, cwd: str | None, home: str | None, expected: bool
) -> None:
    assert path_resolves(path, cwd=cwd, home=home) is expected


@pytest.mark.parametrize(
    "call, cwd, home, expected",
    [
        # built at run time
        (_bash("cat $HOME/x"), "/w", None, ["$HOME/x"]),
        (_bash("cat ${HOME}/x"), "/w", None, ["${HOME}/x"]),
        (_bash("cat ${HOME}x/y"), "/w", "/root", ["${HOME}x/y"]),
        (_bash("cat $(pwd)/x"), "/w", "/root", ["$(pwd)/x"]),
        (_bash("cat $(dirname $(pwd))/x"), "/w", "/root", ["$(dirname $(pwd))/x"]),
        (_bash("cat `pwd`/x"), "/w", "/root", ["`pwd`/x"]),
        (_bash('cat "$HOME"/x'), "/w", None, ['"$HOME"/x']),
        (_bash('cat "$HOME/x"'), "/w", None, ["$HOME/x"]),
        (_bash("cat /home/$USER/.ssh/id"), "/w", "/root", ["/home/$USER/.ssh/id"]),
        (_bash('cat "/home/"$USER/id'), "/w", "/root", ['"/home/"$USER/id']),
        (_bash("cat /x/${D}"), "/w", "/root", ["/x/${D}"]),
        (_bash("cat /x/`id`"), "/w", "/root", ["/x/`id`"]),
        (
            _bash("ls {a,b}/x /home/{a,b}/y"),
            "/w",
            "/root",
            ["{a,b}/x", "/home/{a,b}/y"],
        ),
        (_bash("echo $(cat /etc/passwd)/x"), "/w", "/root", ["$(cat /etc/passwd)/x"]),
        (_bash("PATH=/a:$HOME/bin x"), "/w", None, ["$HOME/bin"]),
        (
            _bash("cat /root/.ss$@h/id ~/.ss$*h/id"),
            "/w",
            "/root",
            ["/root/.ss$@h/id", "~/.ss$*h/id"],
        ),
        (_bash("cat /root$@/.ssh/id"), "/w", "/root", ["/root$@/.ssh/id"]),
        (_bash("git clone https://x.org/$ORG/r.git $D/x"), "/w", "/root", ["$D/x"]),
        (
            _call("python", code="open('/root/.ss' 'h/id')"),
            "/w",
            "/root",
            ["/root/.ss", "h/id"],
        ),
        (_call("python", code='Path("/root") / ".ssh/id"'), "/w", "/root", [".ssh/id"]),
        (_call("python", code='(Path.home() / ".ssh/id")'), "/w", "/root", [".ssh/id"]),
        (_call("python", code='open(f"/x/{name}")'), "/w", "/root", ['f"/x/{name}"']),
        (
            _call("python", code='open("/x/{}".format(n))'),
            "/w",
            "/root",
            ['"/x/{}".format'],
        ),
        (_call("python", code="open('/x/%s' % n)"), "/w", "/root", ["/x/%s"]),
        (_call("python", code="open(home + '/.ssh/id')"), "/w", "/root", ["/.ssh/id"]),
        (_call("python", code="open('/x/'+name)"), "/w", "/root", ["'/x/'+name"]),
        # wildcards and escapes
        (_bash("cat /e*/passwd"), "/w", "/root", ["/e*/passwd"]),
        (_bash("cat /etc/pass*"), "/w", "/root", ["/etc/pass*"]),
        (_bash("cat /e?c/passwd"), "/w", "/root", ["/e?c/passwd"]),
        (_bash("cat /[e]tc/passwd"), "/w", "/root", ["/[e]tc/passwd"]),
        (_bash("cat */passwd ../*/x"), "/w", "/root", ["*/passwd", "../*/x"]),
        (_bash("cat ~/.ssh/*"), "/w", "/root", ["~/.ssh/*"]),
        (
            _bash("cat /root/.ss@(h)/id /root/.ss!(x)/id"),
            "/w",
            "/root",
            ["/root/.ss@", "/root/.ss!"],
        ),
        (_bash("ls [a]/x"), "/w", "/root", ["[a]/x"]),
        (_bash("cat /e\\tc/passwd"), "/w", "/root", ["/e\\tc/passwd"]),
        (_bash("cat '/e'tc/passwd"), "/w", "/root", ["'/e'tc/passwd"]),
        (_bash('cat "/e""tc/passwd"'), "/w", "/root", ['"/e""tc/passwd"']),
        (_bash("cat /a'/b'"), "/w", "/root", ["/a'/b'"]),
        (
            _call("python", code="glob.glob('/e*/passwd')"),
            "/w",
            "/root",
            ["/e*/passwd"],
        ),
        (
            _call("python", code="open('/e\\x74c/passwd')"),
            "/w",
            "/root",
            ["/e\\x74c/passwd"],
        ),
        # other users' homes and URLs
        (_bash("cat ~root/.ssh/id"), "/w", "/root", ["~root/.ssh/id"]),
        (
            _bash("cat ~+/.ssh/id ~-/.ssh/id ~0/x"),
            "/w",
            "/root",
            ["~+/.ssh/id", "~-/.ssh/id", "~0/x"],
        ),
        (_bash("curl file:///etc/passwd"), "/w", "/root", ["file:///etc/passwd"]),
        (_bash("curl FILE:/etc/passwd"), "/w", "/root", ["FILE:/etc/passwd"]),
        (
            _call("python", code="urlopen('file:///etc/passwd')"),
            "/w",
            "/root",
            ["file:///etc/passwd"],
        ),
        # relative paths: unresolved without `cwd`, or after a change of directory
        (_bash("cat a/b ./x ../y"), None, None, ["a/b", "./x", "../y"]),
        (_bash("cat a/b ./x ../y"), "/w", None, []),
        (_bash("ls ."), None, None, ["."]),
        (_bash("cd x && cat a/b"), "/w", None, ["a/b"]),
        (_bash("ls && cd .. && cat a/b"), "/w", None, ["..", "a/b"]),
        (_bash("cd /x && cd y && cat a/b"), "/w", None, ["a/b"]),
        (_bash("cd -P /x && cat a/b"), "/w", None, ["a/b"]),
        (_bash("\\cd /x; cat a/b"), "/w", None, ["a/b"]),
        (_bash("c''d /x; cat a/b"), "/w", None, ["a/b"]),
        (_bash("builtin cd /x; cat a/b"), "/w", None, ["a/b"]),
        (_bash('bash -c "cd /x; cat a/b"'), "/w", None, ["a/b"]),
        (_bash("$CMD /x; cat a/b"), "/w", None, ["a/b"]),
        (_bash("X=1 $CMD /x; cat a/b"), "/w", None, ["a/b"]),
        (_bash("X=$(ls) Y=1; cat a/b"), "/w", None, []),
        (_bash("cd $D && cat a/b"), "/w", None, ["a/b"]),
        (_bash("tar -xzf a.tgz -C out/ b/c"), "/w", None, ["b/c"]),
        (_bash("tar -xzC out/ -f a.tgz b/c"), "/w", None, ["b/c"]),
        (_bash("env -C /x cat a/b"), "/w", None, ["a/b"]),
        (_bash("env --chdir=/x cat a/b"), "/w", None, ["a/b"]),
        (_bash("bash -c 'tar -C /x -cf - a/b'"), "/w", None, ["a/b"]),
        (_bash("pushd /tmp; cat x/y"), "/w", None, ["x/y"]),
        # unbalanced quotes: the whole call
        (_bash("cat 'a"), "/w", "/root", ["cat 'a"]),
        (_bash('echo "a $(b'), "/w", "/root", ['echo "a $(b']),
        (_bash("cat <(ls"), "/w", "/root", ["cat <(ls"]),
        (_call("bash_session", input="echo 'x"), "/w", "/root", ["echo 'x"]),
        (_bash("tar -C /root -cf - .ssh/id"), "/w", None, [".ssh/id"]),
        (_bash("git -C /root show HEAD:.ssh/id"), "/w", None, [".ssh/id"]),
        (_bash("make --directory=/x a/b"), "/w", None, ["a/b"]),
        (
            _call("python", code="run(['cat', 'a/b'], cwd='/root')"),
            "/w",
            None,
            ["'a/b']"],
        ),
        (
            _call("python", code="os.chdir('/'); open('etc/passwd')"),
            "/w",
            None,
            ["etc/passwd"],
        ),
        (_call("python", code="open('data/x.csv')"), None, None, ["data/x.csv"]),
        (_call("python", code="open('data/x.csv')"), "/w", None, []),
        (_call("text_editor", command="view", path="w/f.py"), None, None, ["w/f.py"]),
        (_call("text_editor", command="view", path="w/f.py"), "/w", None, []),
        # `~` that climbs out without `home`
        (_bash("cat ~/../etc/passwd"), "/w", None, ["~/../etc/passwd"]),
        (_bash("cat ~/../etc/passwd"), "/w", "/root", []),
        (
            _call("read_file", file_path="~/../etc/passwd"),
            "/w",
            None,
            ["~/../etc/passwd"],
        ),
        # several, in order, without duplicates
        (_bash("cat $A/x /ok ~u/y $A/x"), "/w", "/root", ["$A/x", "~u/y"]),
        (
            _call("memory", command="rename", old_path="a", new_path="/m/b"),
            None,
            None,
            ["a"],
        ),
        # `$HOME` with `home`, unless the call sets HOME
        (_bash("cat $HOME/x ${HOME}/y"), "/w", "/root", []),
        (_bash('cat "$HOME"/x "$HOME/y" "${HOME}/z"'), "/w", "/root", []),
        (_bash("PATH=/a:$HOME/bin x"), "/w", "/root", []),
        (_bash("HOME=/etc; cat $HOME/x"), "/w", "/root", ["$HOME/x"]),
        (_bash("export HOME=/etc; cat ~/x"), "/w", "/root", ["~/x"]),
        (_bash("cat ${HOME:=/etc}/x"), "/w", "/root", ["${HOME:=/etc}/x"]),
        (
            _call("python", code="os.environ['HOME'] = '/'; open('~/x')"),
            "/w",
            "/root",
            ["~/x"],
        ),
        (
            _bash("cat $HOME/.ss*/x a$HOME/x"),
            "/w",
            "/root",
            ["$HOME/.ss*/x", "a$HOME/x"],
        ),
        (_bash('cat "$HOME/x"y $HOME/"x"'), "/w", "/root", ['"$HOME/x"y', '$HOME/"x"']),
        (_call("bash_session", input="cat $HOME/x"), "/w", "/root", ["$HOME/x"]),
        (
            _call("python", code="os.path.expandvars('$HOME/x')"),
            "/w",
            "/root",
            ["$HOME/x"],
        ),
        # `$PWD` is the working directory
        (_bash('cat $PWD/x "${PWD}/y"'), "/w", None, []),
        (_bash("cat $PWD/x"), None, None, ["$PWD/x"]),
        (_bash("cd / && cat $PWD/x"), "/w", None, []),
        (_bash("cd x && cat $PWD/y"), "/w", None, ["$PWD/y"]),
        (_bash("PWD=/; cat $PWD/x"), "/w", None, ["$PWD/x"]),
        (_call("bash_session", input="cat $PWD/x"), "/w", None, ["$PWD/x"]),
        # quoted `~` is not expanded
        (_bash("cp '~/x' \"~/y\" z"), "/w", "/root", ["~/x", "~/y"]),
        # a leading `cd` to a place: relative paths are read both ways
        (_bash("cd /w/api && pytest tests/unit"), "/w", None, []),
        (_bash("cd .. && ls a/b"), "/w", None, []),
        (_bash("cd ~/p && cat a/b"), "/w", "/root", []),
        # single-quoted programs of commands that read them literally
        (_bash("sed -i 's/[[:space:]]*$//' a.py"), "/w", None, []),
        (_bash("sed -E 's/\\/usr\\/local/\\/opt/g' a.sh"), "/w", None, []),
        (_bash("sed -e 's/^ *//' -e 's/ *$//' a"), "/w", None, []),
        (_bash("awk -F/ 'NR > 1 {print $2/$3}' a | sort -rn"), "/w", None, []),
        (_bash("printf 'a/b/c\\n' | cut -d/ -f2"), "/w", None, []),
        (
            _bash("echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc"),
            "/w",
            "/root",
            [],
        ),
        (_bash('grep -E "a*/b" x'), "/w", None, []),
        (_bash("cat <<'X' > a.sh\nls $D/*.txt\nX"), "/w", None, []),
        # ... but not when the call could run them, or after `cd`
        (_bash("sed '1e cat /e*/x' f"), "/w", None, ["/e*/x"]),
        (_bash("sed 's/a/cat \\/e*\\/x/e' f"), "/w", None, ["\\/e*\\/x/e"]),
        (_bash("sed -n 'p;e ls /e*' f"), "/w", None, ["/e*"]),
        (_bash("sed 's/[/]/x/e' f"), "/w", None, ["s/[/]/x/e"]),
        (_bash("sed -f s.sed 'a*/b'"), "/w", None, ["a*/b"]),
        (_bash("awk 'BEGIN{system(\"ls $D/x\")}'"), "/w", None, ["$D/x"]),
        (_bash("awk '{print | \"cat /e*/x\"}'"), "/w", None, ['/e*/x"}']),
        (_bash("echo 'cat $D/x' | sh"), "/w", None, ["$D/x"]),
        (_bash("echo 'cat $D/x' > a.sh; bash a.sh"), "/w", None, ["$D/x"]),
        (_bash("cat $(echo '/e*/x')"), "/w", None, ["/e*/x"]),
        (_bash("printf -v P '/e*/x'"), "/w", None, ["/e*/x"]),
        (_bash("X=1 grep '/e*/x' f"), "/w", None, ["/e*/x"]),
        (_bash("find / -path '/e*/x'"), "/w", None, ["/e*/x"]),
        (_bash("eslint 'src/**/*.ts'"), "/w", None, ["src/**/*.ts"]),
        (_bash('cat "$D/x"'), "/w", None, ["$D/x"]),
        (_bash("cat '/e*'/x"), "/w", None, ["'/e*'/x"]),
        (_bash("cat '/a b*/../etc/x'"), "/w", None, ["b*/../etc/x"]),
        (_bash("cd /w && cat 'a*/b'"), "/w", None, []),
        (_bash("cd x && cat 'a*/b'"), "/w", None, ["a*/b"]),
        (_call("bash_session", input="grep '/e*/x' f"), "/w", None, ["/e*/x"]),
        # near misses that resolve
        (_bash("cat /etc/passwd ~/.ssh/id"), None, "/root", []),
        (_bash("cat ~/.ssh/id ~"), None, None, []),
        (_bash("cat '/etc/passwd' \"/etc/hosts\""), None, None, []),
        (_bash('bash -c "cat \\"/etc/x\\""'), None, None, []),
        (_bash("echo $(cat /y) `cat /z` $HOME $PATH"), None, None, []),
        (_bash("echo $((4 / 2)) ${#x}"), None, None, []),
        (_bash("awk '{print $1}' /w/x"), None, None, []),
        (_bash("find /w -name '*.py' -exec rm {} +"), None, None, []),
        (_bash("grep -E '^/usr/bin$' /w/x"), None, None, []),
        (_bash("tar -C/etc -xf /w/a.tar"), None, None, []),
        (_bash("pip install -e .[dev] && pip install -e '.[test]'"), "/w", None, []),
        (_bash('git commit -m "fix cd in src/a.py"'), "/w", None, []),
        (_bash("grep -C 3 foo src/a.py"), "/w", None, []),
        (_bash("cat > a.py <<'X'\nprint(f\"{os.getcwd()}/out\")\nX"), "/w", None, []),
        (_bash("echo $? $$ $# done"), None, None, []),
        (_bash("curl https://x.org/a?f=/b"), None, None, []),
        (_bash("pip install -r /w/req.txt && python /w/main.py"), None, None, []),
        (_bash("git diff origin/main"), "/w", None, []),
        (_bash("sed -i 's/a/b/g' /w/x"), "/w", None, []),
        (_call("python", code="open('/etc/hosts').read()"), None, None, []),
        (
            _call("python", code='open(f"/etc/hosts"), Path(r"/x"), b"/y"'),
            None,
            None,
            [],
        ),
        (_call("python", code="x = a/b; y = d['a']/d['b']; z = n*2/3"), None, None, []),
        (_call("python", code='"/".join(parts); s.split("/")'), None, None, []),
        (_call("python", code='print("/w/out\\n")'), None, None, []),
        (_call("python", code='p = """/etc/x"""'), None, None, []),
        (_call("python", code='x = ["/a",\n "/b"]\nprint(\'/c\')'), None, None, []),
        (_call("python", code="d = {'/a': [1], \"/b\": 2}"), None, None, []),
        (_call("python", code="'.'.join(x)  # see a/b"), None, None, []),
        (_call("python", code="print(f'{a} of {b}')"), None, None, []),
        (_call("text_editor", command="view", path="/w/$x*"), None, None, []),
        (_call("think", thought="cat $HOME/x"), None, None, []),
    ],
)
def test_unresolved_paths(
    call: ToolCall, cwd: str | None, home: str | None, expected: list[str]
) -> None:
    assert unresolved_paths(call, cwd=cwd, home=home) == expected


@pytest.mark.parametrize(
    "call",
    [
        _bash("cat a/b $HOME/x /etc/pass* ~/../y"),
        _call("python", code="open(f'/x/{n}'); open('a/b')"),
        _call("text_editor", command="view", path="a/b"),
    ],
)
def test_unresolved_paths_reports_what_path_resolves_cannot_place(
    call: ToolCall,
) -> None:
    unresolved = unresolved_paths(call)
    for path in paths_in(call):
        assert (path in unresolved) is not path_resolves(path)


def test_unresolved_paths_rejects_a_relative_home() -> None:
    with pytest.raises(ValueError, match="absolute"):
        unresolved_paths(_bash("ls"), home="root")


PROTECTED = ["/etc/**", "~/.ssh/**"]


@protocol
def protected_paths() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        for path in paths_in(step.call):
            if path_matches(path, PROTECTED, cwd="/work", home="/root"):
                return Decision.reject(f"call touches {path}")
        unresolved = unresolved_paths(step.call, cwd="/work", home="/root")
        if unresolved:
            return Decision.escalate(f"cannot resolve {', '.join(unresolved)}")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    "call, action, asked",
    [
        (_bash("cat /etc/passwd"), "reject", False),
        (_bash("cat /work/../etc/passwd"), "reject", False),
        (_bash("cat ../etc/passwd"), "reject", False),
        (_bash("cat ~/.ssh/id_rsa"), "reject", False),
        (_bash("cat /root/.ssh/id_rsa"), "reject", False),
        (_bash("cat $HOME/x /etc/passwd"), "reject", False),
        (_call("python", code="open('/etc/hosts').read()"), "reject", False),
        (_call("text_editor", command="view", path="../etc/passwd"), "reject", False),
        (_bash("cat $HOME/.ssh/id_rsa"), "reject", False),
        (_bash('cat "${HOME}/.ssh/id_rsa"'), "reject", False),
        (_bash("HOME=/x; cat $HOME/.ssh/id_rsa"), "reject", True),
        (_bash("cat '/e'\"tc/passwd\""), "reject", True),
        (_bash("cat \"/e\"'tc/passwd'"), "reject", True),
        (_bash("cat '/etc/passwd'"), "reject", False),
        (_bash("grep x '/etc/pass*'"), "reject", False),
        (_bash("grep x '/e*/../etc/passwd'"), "reject", True),
        (_bash("cat '/x y*/../etc/passwd'"), "reject", True),
        (_bash("sed '1e cat ~/.ss*/id_rsa' f"), "reject", True),
        (_bash("echo 'cat $HOME/.ssh/id_rsa' | sh"), "reject", True),
        (_bash("cat 'x"), "reject", True),
        (_bash("cat $(echo ~)/.ssh/id_rsa"), "reject", True),
        (_bash("cat `echo ~`/.ssh/id_rsa"), "reject", True),
        (_bash("cat /e*/passwd"), "reject", True),
        (_bash("cat /etc/pass*"), "reject", True),
        (_bash("cat '/e'tc/passwd"), "reject", True),
        (_bash("cat ~root/.ssh/id_rsa"), "reject", True),
        (_bash("curl file:///etc/passwd"), "reject", True),
        (_bash("cd / && cat etc/passwd"), "reject", False),
        (_bash("cd .. && cat ../etc/passwd"), "reject", False),
        (_bash("cd / && cat $PWD/etc/passwd"), "reject", False),
        (_bash("cd /tmp && cd / && cat etc/passwd"), "reject", True),
        (_bash("tar -C / -cf - etc/passwd"), "reject", True),
        (_call("python", code="open(f'/{d}/passwd')"), "reject", True),
        (_bash("wc -l /work/data.csv > /work/count.txt"), "continue", False),
        (_bash("python report.py && cat out/a.txt"), "continue", False),
        (_bash("ls ~/notes"), "continue", False),
        (_bash("cat $HOME/notes.txt"), "continue", False),
        (_bash("sed -i 's/ *$//' src/a.py"), "continue", False),
        (_bash("cd /work/api && pytest tests/unit"), "continue", False),
        (_call("python", code="open('data.csv').read()"), "continue", False),
    ],
)
async def test_a_deny_list_escalates_what_it_cannot_resolve(
    call: ToolCall, action: Action, asked: bool
) -> None:
    host = FakeHost(HumanAnswer("reject"))
    sentinel = sequential([protected_paths(), human(stages=["tool_call"])])
    step = before_tool_call(call.function, **call.arguments)
    decision = await run_sentinel(sentinel, host_context(host=host), step)
    assert decision is not None
    assert decision.action == action
    assert bool(host.asked) is asked


# Escalated by the deny-list idiom with `cwd` and `home`: wildcards (`src/*.py`,
# `glob.glob`), a quoted glob given to a program that expands it (`eslint`), a
# relative `cd` whose target `CDPATH` could change, and Python f-strings or `+`
# with a `/`. Before quoting and `$HOME` were taken into account, 25 of the 286.
EVERYDAY_ESCALATED = [
    "eslint 'src/**/*.ts'",
    "prettier --write 'src/**/*.{ts,tsx}'",
    "cd build && cmake .. && make",
    'for f in data/*.csv; do wc -l "$f"; done',
    "wc -l src/*.py",
    "import glob\nprint(glob.glob('data/*.csv'))",
    "out = f'out/{name}.csv'\ndf.to_csv(out)",
    "for i in range(3):\n    print(f'epoch {i}/{epochs}')",
    "print(f'{a}/{b}')",
    "s = 'a' + '/' + 'b'",
    'print(f"{os.getcwd()}/out")',
]


def test_everyday_calls_rarely_escalate() -> None:
    calls = [_bash(command) for command in BASH]
    calls.extend(_call("python", code=code) for code in PYTHON)
    assert len(calls) >= 200
    for call in calls:
        assert not any(
            path_matches(path, PROTECTED, cwd="/work", home="/root")
            for path in paths_in(call)
        ), call
    escalated = [
        call_text(call)
        for call in calls
        if unresolved_paths(call, cwd="/work", home="/root")
    ]
    assert escalated == EVERYDAY_ESCALATED


@pytest.mark.parametrize(
    "script, executes",
    [
        ("s/a/b/g", False),
        ("s/[[:space:]]*$//", False),
        ("s|/usr|/opt|; /^#/d; 1,3p", False),
        ("/x/{s/a/b/;p}", False),
        ("y/abc/xyz/", False),
        ("a\\\ntext e\n$d", False),
        ("1~2d; 0,/re/s//x/w out.txt", False),
        ("s/e/E/g # e", False),
        ("e", True),
        ("1e ls", True),
        ("s/a/b/e", True),
        ("s/a/b/ge", True),
        ("/x/e ls", True),
        ("p;e ls", True),
        ("s/[/]/x/e", True),
        ("s/x/[/e", True),
        ("s/a/b", True),
        ("s/a/b/;Z", True),
        ("\\%x%e", True),
        ("{e ls\n}", True),
    ],
)
def test_sed_executes(script: str, executes: bool) -> None:
    assert sed_executes(script) is executes


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
@given(st.lists(_PART, max_size=8), st.lists(_PART, max_size=4), _SEGMENT)
def test_home_resolves_a_home_path_as_an_absolute_one(
    parts: list[str], home_written: list[str], root: str
) -> None:
    home = "/" + "/".join(home_written)
    home_parts = _resolve(home_written, [])
    path = "~/" + "/".join(parts)
    resolved = _resolve(parts, home_parts)
    assert path_resolves(path, home=home)
    assert path_matches(path, [f"/{escape(root)}/**"], home=home) == (
        resolved[:1] == [root]
    )
    under_home = resolved[: len(home_parts)] == home_parts
    assert path_matches(path, ["~/**"], home=home) == under_home
    clean_home = "/" + "/".join(escape(part) for part in home_parts)
    assert path_matches(path, [clean_home + "/**"], home=home) == under_home


@_PROPERTY_SETTINGS
@given(st.lists(_PART, min_size=1, max_size=8))
def test_a_relative_path_that_climbs_out_matches_no_wildcard(parts: list[str]) -> None:
    path = "/".join(parts)
    climbs_out = _climbs_out(parts)
    assert not path_resolves(path)
    assert path_resolves(path, cwd="/w")
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
