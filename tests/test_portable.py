import importlib
import importlib.machinery
import importlib.util
import linecache
import re
import sys
import textwrap
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from inspect_ai._util.registry import registry_info

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Monitor,
    Observation,
    PortabilityError,
    concurrent,
    human,
    monitor,
    observe_only,
    sequential,
    threshold,
)
from inspect_sentinel._integration import sentinel_from_config

Load = Callable[..., ModuleType]

EXAMPLES = Path(__file__).parent.parent / "examples"

HEADER = """\
from inspect_sentinel import BeforeToolCall, Context, Monitor, Observation, monitor
"""

MONITOR = """
@monitor{decorator}
def watched() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
{body}
        return Observation.score(0.0)

    return check
"""


@pytest.fixture
def load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Load:
    prepend_path(monkeypatch, str(tmp_path))

    def load(source: str, name: str | None = None) -> ModuleType:
        name = name or f"fixture_{uuid.uuid4().hex}"
        path = tmp_path / f"{name}.py"
        path.write_text(textwrap.dedent(source))
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def third_party(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = f"fakereq_{uuid.uuid4().hex}"
    package = tmp_path / "site-packages" / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("def get(url):\n    return url\n")
    prepend_path(monkeypatch, str(tmp_path / "site-packages"))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return name


def prepend_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(sys, "path", [path, *sys.path])
    importlib.invalidate_caches()


def monitor_source(
    body: str, setup: str = "", decorator: str = "", header: str = HEADER
) -> str:
    indented = textwrap.indent(textwrap.dedent(body).strip(), " " * 8)
    return f"{header}{setup}\n{MONITOR.format(decorator=decorator, body=indented)}"


def configure(module: ModuleType, name: str = "watched") -> object:
    return cast(Callable[[], object], getattr(module, name))()


@pytest.mark.parametrize(
    "setup,body",
    [
        ("import json", "json.dumps(step.call.arguments)"),
        ("import re", "re.findall('rm', step.call.function)"),
        ("import datetime", "datetime.datetime.now(datetime.timezone.utc)"),
        ("import math", "math.sqrt(2.0)"),
        ("import os", "os.path.join('a', 'b')"),
        ("import posixpath", "posixpath.join('a', 'b')"),
        ("from os import path", "path.join('a', 'b')"),
        ("from os.path import join", "join('a', 'b')"),
        (
            "from inspect_sentinel import call_text, find_words",
            "find_words(call_text(step.call), ['rm'])",
        ),
        ("import inspect_sentinel", "inspect_sentinel.Observation.score(0.1)"),
        ("from pydantic import BaseModel", "BaseModel.model_validate({})"),
        ("import os", "os.path.normpath('a/../b')"),
        ("from os.path import normpath", "normpath('a/../b')"),
        ("import os\nlast = os.environ", "[(last := w) for w in ['a']]\nlast.upper()"),
        ("import random", "random.Random(0).random()\nrandom.random()"),
        ("import heapq, bisect", "heapq.heappush([], 1)\nbisect.bisect([1], 0)"),
        ("import hashlib", "hashlib.sha256(b'a').hexdigest()"),
        ("import statistics", "statistics.mean([1.0, 2.0])"),
        ("import asyncio", "asyncio.Lock()\nasyncio.wait_for"),
        ("import anyio", "anyio.create_task_group()\nanyio.sleep(0)"),
        ("from collections import Counter", "Counter('abc').most_common(1)"),
        ("import logging", "logging.getLogger(__name__).info('a')"),
        ("import builtins", "builtins.len('a')"),
        ("import string", "'a'.translate(str.maketrans('', '', string.punctuation))"),
        ("", "from os import path\npath.join('a', 'b')"),
        (
            "from typing import TYPE_CHECKING",
            "if TYPE_CHECKING:\n    from inspect_ai.model import Model",
        ),
        ("", "import os\nos.path.join('a', 'b')"),
        ("", "from inspect_ai import core"),
        (
            "import os",
            "def inner(path: os.PathLike[str]) -> os.PathLike[str]:\n    return path",
        ),
        (
            "import os as names\nclass Words:\n    names = ('a', 'b')\n    upper = [n.upper() for n in names]",
            "Words.upper",
        ),
        ("import ipaddress", "ipaddress.ip_address('10.0.0.1').is_private"),
        ("import graphlib", "graphlib.TopologicalSorter({}).static_order()"),
        ("import codecs", "codecs.decode('nop', 'rot13')\ncodecs.strict_errors"),
        ("import csv", "list(csv.reader(['a,b']))"),
        ("import errno", "errno.ENOENT"),
        ("import colorsys", "colorsys.rgb_to_hsv(0.1, 0.2, 0.3)"),
        ("import traceback", "traceback.format_exc()"),
        ("import io", "io.StringIO('a').read()\nio.BytesIO(b'a')"),
        *(
            [("import tomllib", "tomllib.loads('a = 1')")]
            if sys.version_info >= (3, 11)
            else []
        ),
    ],
)
def test_allowed_references_pass(load: Load, setup: str, body: str) -> None:
    configure(load(monitor_source(body, setup)))


STDLIB = "is not on the portable standard-library list"
BUILTIN = "which a portable function cannot call"
EFFECTS = "which has effects a portable function cannot have"
CORE = "inspect_ai is portable only through `inspect_ai.core`"


@pytest.mark.parametrize(
    "setup,body,reference,reason",
    [
        ("import os", "os.environ.get('HOME')", "os.environ", STDLIB),
        ("import os", "os.getenv('HOME')", "os.getenv", STDLIB),
        ("import os", "os.system('ls')", "os.system", f"`os` {STDLIB}"),
        ("import os", "os.sep", "os.sep", f"`os` {STDLIB}"),
        ("from os import sep", "sep", "sep", f"`os` {STDLIB}"),
        ("import sys", "sys.modules", "sys.modules", STDLIB),
        ("import sys", "sys", "sys", STDLIB),
        ("import subprocess", "subprocess.run(['ls'])", "subprocess.run", STDLIB),
        (
            "import socket",
            "socket.create_connection(('x', 1))",
            "socket.create_connection",
            STDLIB,
        ),
        ("import pathlib", "pathlib.Path('f').read_text()", "pathlib.Path", STDLIB),
        ("import pickle", "pickle.loads(b'')", "pickle.loads", STDLIB),
        ("import io", "io.open('f')", "io.open", STDLIB),
        (
            "import asyncio",
            "asyncio.create_subprocess_exec('ls')",
            "asyncio.create_subprocess_exec",
            STDLIB,
        ),
        ("reader = open", "reader('f')", "reader", STDLIB),
        ("import builtins", "builtins.open('f')", "builtins.open", STDLIB),
        (
            "import importlib",
            "importlib.import_module('os')",
            "importlib.import_module",
            STDLIB,
        ),
        ("", "open('f')", "open", BUILTIN),
        ("", "input()", "input", BUILTIN),
        ("", "eval('1')", "eval", BUILTIN),
        ("", "exec('1')", "exec", BUILTIN),
        ("", "compile('1', 'f', 'eval')", "compile", BUILTIN),
        ("", "__import__('os')", "__import__", BUILTIN),
        ("", "breakpoint()", "breakpoint", BUILTIN),
        ("from inspect_ai.model import get_model", "get_model()", "get_model", CORE),
        (
            "import inspect_ai.model",
            "inspect_ai.model.get_model()",
            "inspect_ai.model.get_model",
            CORE,
        ),
        ("import os as o", "o.environ['HOME']", "o.environ", STDLIB),
        ("from os import environ as e", "e.get('HOME')", "e", STDLIB),
        ("from os import *", "getenv('HOME')", "getenv", STDLIB),
        (
            "import functools, os",
            "functools.partial(os.system, 'ls')",
            "os.system",
            STDLIB,
        ),
        ("", "import subprocess\nsubprocess.run(['ls'])", "subprocess.run", STDLIB),
        ("", "from os import environ", "from os import environ", STDLIB),
        ("", "from json import tool", "from json import tool", STDLIB),
        ("import os", "(lambda: os.getenv('HOME'))()", "os.getenv", STDLIB),
        ("import os", "[os.getenv(k) for k in ('A', 'B')]", "os.getenv", STDLIB),
        ("import os", "[os for os in os.listdir('.')]", "os.listdir", STDLIB),
        (
            "import os",
            "def inner() -> str | None:\n    return os.getenv('HOME')\ninner()",
            "os.getenv",
            STDLIB,
        ),
        (
            "",
            "def inner() -> object:\n    return subprocess.run\nimport subprocess",
            "subprocess.run",
            STDLIB,
        ),
        (
            "import os",
            "os = 1\ndef inner() -> object:\n    global os\n    return os.getcwd()",
            "os.getcwd",
            STDLIB,
        ),
        ("", "import os.path\nos.system('ls')", "os.system", STDLIB),
        (
            "",
            "import posixpath\nposixpath.os.system('ls')",
            "posixpath.os.system",
            STDLIB,
        ),
        (
            "from inspect_sentinel._runner import run_sentinel",
            "run_sentinel",
            "run_sentinel",
            "only `inspect_sentinel`'s public API is portable",
        ),
        (
            "import functools, os\n@functools.cache\ndef helper() -> None:\n    os.system('ls')",
            "helper()",
            "os.system",
            STDLIB,
        ),
        (
            "def make() -> type:\n    import subprocess\n    class Helper:\n        def run(self) -> object:\n            return subprocess.run\n    return Helper\nHelper = make()",
            "Helper().run()",
            "subprocess.run",
            STDLIB,
        ),
        (
            "import os\nclass Paths:\n    home: os.PathLike[str]",
            "Paths",
            "os.PathLike",
            STDLIB,
        ),
        ("import logging", "logging.FileHandler('f')", "logging.FileHandler", EFFECTS),
        (
            "import logging",
            "logging.basicConfig(filename='f')",
            "logging.basicConfig",
            EFFECTS,
        ),
        ("import uuid", "uuid.uuid1()", "uuid.uuid1", EFFECTS),
        ("import uuid", "uuid.getnode()", "uuid.getnode", EFFECTS),
        ("import calendar", "calendar.main", "calendar.main", EFFECTS),
        ("import codecs", "codecs.open('f')", "codecs.open", EFFECTS),
        *(
            [
                (
                    "import contextlib",
                    "contextlib.chdir('/')",
                    "contextlib.chdir",
                    EFFECTS,
                )
            ]
            if sys.version_info >= (3, 11)
            else []
        ),
        *(
            [("import time", "time.tzset()", "time.tzset", EFFECTS)]
            if hasattr(time, "tzset")
            else []
        ),
    ],
)
def test_disallowed_references_fail(
    load: Load, setup: str, body: str, reference: str, reason: str
) -> None:
    match = f"`{re.escape(reference)}` is [^\\n]*{re.escape(reason)}"
    with pytest.raises(PortabilityError, match=match):
        configure(load(monitor_source(body, setup)))


@pytest.mark.parametrize(
    "setup,body",
    [
        ("import builtins", "getattr(builtins, 'open')('f')"),
        ("import functools, os\nRUN = functools.partial(os.system, 'ls')", "RUN()"),
        (
            "import os\nfrom typing import NamedTuple\nclass Holder(NamedTuple):\n    fn: object\nHOLDER = Holder(os.system)",
            "HOLDER.fn('ls')",
        ),
    ],
)
def test_deliberate_indirection_is_not_caught(
    load: Load, setup: str, body: str
) -> None:
    configure(load(monitor_source(body, setup)))


def test_a_third_party_package_fails(load: Load, third_party: str) -> None:
    module = load(monitor_source(f"{third_party}.get('u')", f"import {third_party}"))
    with pytest.raises(PortabilityError, match="is not a portable dependency"):
        configure(module)


def test_inspect_ai_core_passes(load: Load) -> None:
    # importing inspect_ai.model replaces ModelOutput.from_message with its own
    setup = "import inspect_ai.model\nfrom inspect_ai.core import ChatMessageUser, ModelOutput\nfrom inspect_ai.model import ChatMessageAssistant"
    body = "ChatMessageUser(content='a')\nModelOutput.from_message(ChatMessageAssistant(content='b'))"
    configure(load(monitor_source(body, setup)))


def test_a_helper_in_the_same_module_is_followed(load: Load) -> None:
    setup = """
def clean(text: str) -> str:
    return text.strip()

def read(path: str) -> str:
    return open(path).read()
"""
    configure(load(monitor_source("clean(step.call.function)", setup)))
    with pytest.raises(PortabilityError, match=r"in read: `open`.*reached from"):
        configure(load(monitor_source("read('f')", setup)))


def test_a_helper_in_another_local_module_is_followed(load: Load) -> None:
    helpers = load(
        "import subprocess\n\ndef run() -> None:\n    subprocess.run(['ls'])\n"
    )
    module = load(monitor_source("run()", f"from {helpers.__name__} import run"))
    with pytest.raises(PortabilityError, match="`subprocess.run`"):
        configure(module)


def test_allowed_data_in_a_helper_module_is_not_looked_into(load: Load) -> None:
    helpers = load(
        "import logging\nfrom pydantic import TypeAdapter\nlog = logging.getLogger('x')\nSCORES = TypeAdapter(list[float])\n"
    )
    body = f"{helpers.__name__}.log.info('a')\n{helpers.__name__}.SCORES.validate_json('[0.1]')"
    configure(load(monitor_source(body, f"import {helpers.__name__}")))


def test_a_class_in_a_module_outside_sys_modules_is_checked(tmp_path: Path) -> None:
    setup = "import os\n\nclass Helper:\n    def home(self) -> str | None:\n        return os.getenv('HOME')\n"
    path = tmp_path / "task_file.py"
    path.write_text(monitor_source("Helper().home()", setup))
    # how `inspect eval` loads a task file: named by its path, not in sys.modules
    loader = importlib.machinery.SourceFileLoader(path.as_posix(), path.as_posix())
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    with pytest.raises(PortabilityError, match=r"in Helper\.home: `os\.getenv`"):
        configure(module)


def test_a_module_imported_inside_the_function_is_followed_only_if_loaded(
    load: Load, tmp_path: Path
) -> None:
    helpers = load(
        "import subprocess\n\ndef run() -> None:\n    subprocess.run(['ls'])\n"
    )
    module = load(
        monitor_source(f"import {helpers.__name__}\n{helpers.__name__}.run()")
    )
    with pytest.raises(PortabilityError, match="`subprocess.run`"):
        configure(module)
    unloaded = f"fixture_{uuid.uuid4().hex}"
    (tmp_path / f"{unloaded}.py").write_text(
        "import os\n\ndef home() -> str | None:\n    return os.getenv('HOME')\n"
    )
    configure(load(monitor_source(f"from {unloaded} import home\nhome()")))
    assert unloaded not in sys.modules


def test_a_wrapping_decorator_under_monitor_is_checked(load: Load) -> None:
    source = (
        HEADER
        + """
import functools
import os

def logged(factory):
    @functools.wraps(factory)
    def wrapper():
        os.system("ls")
        return factory()

    return wrapper

@monitor
@logged
def watched() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.0)

    return check
"""
    )
    with pytest.raises(PortabilityError, match="`os.system`"):
        configure(load(source))


def test_a_function_of_unknown_module_fails(load: Load) -> None:
    setup = "namespace = {}\nexec('def helper():\\n    return 1', namespace)\nhelper = namespace['helper']"
    with pytest.raises(PortabilityError, match="its module is unknown"):
        configure(load(monitor_source("helper()", setup)))


def test_an_object_that_cannot_be_inspected_fails(load: Load) -> None:
    setup = """
class Proxy:
    @property
    def __class__(self):
        raise RuntimeError("unbound")

PROXY = Proxy()
"""
    with pytest.raises(PortabilityError, match="could not be inspected"):
        configure(load(monitor_source("PROXY.value", setup)))


def test_an_installed_package_can_use_its_own_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"mymons_{uuid.uuid4().hex}"
    package = tmp_path / "site-packages" / name
    package.mkdir(parents=True)
    source = monitor_source(
        "_score(step.call.function)",
        "def _score(text: str) -> float:\n    return float(len(text))\n",
    )
    (package / "__init__.py").write_text(source)
    prepend_path(monkeypatch, str(tmp_path / "site-packages"))
    try:
        configure(importlib.import_module(name))
    finally:
        sys.modules.pop(name, None)


def test_helpers_that_call_each_other_are_checked_once(load: Load) -> None:
    setup = """
def ping(n: int) -> int:
    return pong(n - 1) if n else 0

def pong(n: int) -> int:
    return ping(n - 1) if n else 0
"""
    configure(load(monitor_source("ping(3)", setup)))


def test_an_unreferenced_task_beside_the_monitor_passes(load: Load) -> None:
    setup = """
import json
from inspect_ai import Task, task
from inspect_ai.model import get_model

@task
def unreferenced() -> Task:
    get_model()
    return Task()
"""
    configure(load(monitor_source("json.dumps({})", setup)))


def test_class_methods_are_followed(load: Load) -> None:
    clean = """
class Rules:
    LIMIT = 3

    @staticmethod
    def clean(text: str) -> str:
        return text.strip()
"""
    configure(load(monitor_source("Rules.clean('a')\nRules.LIMIT", clean)))
    setup = """
import os

class Rules:
    LIMIT = 3

    @staticmethod
    def clean(text: str) -> str:
        return text.strip()

    def home(self) -> str | None:
        return os.getenv("HOME")
"""
    with pytest.raises(PortabilityError, match=r"in Rules\.home: `os\.getenv`"):
        configure(load(monitor_source("Rules().home()", setup)))


def test_classes_for_store_as_and_pydantic_pass(load: Load) -> None:
    setup = """
from inspect_ai.scorer import Reference
from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field

class Count(StoreModel):
    n: int = 0

class Verdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)

CITE = Reference(type="message", id="m0")
"""
    body = """
context.store_as(Count).n += 1
Verdict.model_validate_json('{"score": 0.5}')
Reference(type="message", id="m1")
Observation.score(0.5, references=[CITE])
"""
    configure(load(monitor_source(body, setup)))


def test_local_scopes_pass(load: Load) -> None:
    body = """
def double(x: int) -> int:
    return x * 2
square = lambda x: x * x
values = [double(i) for i in range(3)]
pairs = {k: square(v) for k, v in zip("ab", values)}
"""
    configure(load(monitor_source(body)))


def test_closures_over_factory_parameters_pass(load: Load) -> None:
    source = (
        HEADER
        + """
@monitor
def watched(client: object = None, limit: float = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        assert client is not None
        return Observation.score(limit)

    return check
"""
    )
    module = load(source)
    configure_with = cast(Callable[..., object], module.watched)
    configure_with(client=sys.modules["subprocess"])


def test_a_default_argument_is_checked(load: Load) -> None:
    source = (
        HEADER
        + """
@monitor
def watched() -> Monitor:
    async def check(context: Context, step: BeforeToolCall, read=open) -> Observation:
        return Observation.score(0.0)

    return check
"""
    )
    with pytest.raises(PortabilityError, match="`open`"):
        configure(load(source))


def test_the_factory_body_is_checked(load: Load) -> None:
    source = (
        HEADER
        + """
import os

@monitor
def watched() -> Monitor:
    home = os.getenv("HOME")

    async def check(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.0, home)

    return check
"""
    )
    with pytest.raises(PortabilityError, match=r"in watched: `os\.getenv`"):
        configure(load(source))


def test_portable_false_skips_the_check(load: Load) -> None:
    module = load(monitor_source("open('f')", decorator="(portable=False)"))
    configure(module)
    assert registry_info(module.watched).metadata["portable"] is False


def test_portable_is_recorded_in_the_registry(load: Load) -> None:
    module = load(monitor_source("pass"))
    assert registry_info(module.watched).metadata["portable"] is True


def test_portable_must_be_a_bool(load: Load) -> None:
    with pytest.raises(TypeError, match="portable must be True or False"):
        load(monitor_source("pass", decorator="(portable=1)"))


def test_a_protocol_is_checked(load: Load) -> None:
    source = """
from inspect_sentinel import Context, Decision, Protocol, Step, protocol

@protocol
def decides_portably() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        open("f")
        return None

    return decide
"""
    with pytest.raises(PortabilityError, match="protocol decides_portably"):
        configure(load(source), "decides_portably")


def test_without_source_the_check_is_skipped() -> None:
    namespace: dict[str, Any] = {}
    code = compile(monitor_source("open('f')"), "<string>", "exec")
    exec(code, namespace)
    namespace["watched"]()


def test_a_jupyter_cell_is_checked() -> None:
    interactiveshell = pytest.importorskip("IPython.core.interactiveshell")
    shell = interactiveshell.InteractiveShell.instance()
    try:
        shell.run_cell(monitor_source("open('f')"))
        with pytest.raises(PortabilityError, match="`open`"):
            shell.run_cell("watched()").raise_error()
    finally:
        interactiveshell.InteractiveShell.clear_instance()


def test_the_check_runs_once_per_factory(load: Load) -> None:
    module = load(monitor_source("pass"))
    configure(module)
    path = Path(cast(str, module.__file__))
    path.write_text(path.read_text().replace("        pass", "        open('f')"))
    linecache.clearcache()
    configure(module)


def test_every_violation_is_reported_together(load: Load) -> None:
    body = """
from os import getenv
open('f')
os.environ['HOME']
os.getenv('A') or os.getenv('B')
"""
    module = load(monitor_source(body, "import os"))
    with pytest.raises(PortabilityError) as raised:
        configure(module)
    message = str(raised.value)
    lines = [line for line in message.splitlines() if line.startswith("- ")]
    assert len(lines) == 4
    assert f"{module.__file__}:" in lines[0]
    assert "watched.<locals>.check" in lines[0]
    assert "monitor watched is portable" in message
    assert "context.host.generate()" in message
    assert "@monitor(portable=False)" in message
    assert "ask for it" not in message


@monitor
def plain() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.0)

    return check


@pytest.mark.parametrize(
    "configure_shipped",
    [
        lambda: observe_only(plain()),
        lambda: concurrent([plain()]),
        lambda: sequential([plain()]),
        lambda: threshold(plain(), reject_at=0.5),
        lambda: human(stages=["tool_call"]),
    ],
)
def test_shipped_protocols_pass(configure_shipped: Callable[[], object]) -> None:
    configure_shipped()


@pytest.fixture
def examples(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    prepend_path(monkeypatch, str(EXAMPLES))
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if Path(getattr(sys.modules[name], "__file__", "") or "").parent == EXAMPLES:
            del sys.modules[name]


@pytest.mark.parametrize(
    "example,task",
    [
        ("no_network", "network_rule"),
        ("trajectory", "trajectory"),
        ("escalate_to_human", "escalate_to_human"),
        ("llm_suspicion", "llm_suspicion"),
        ("nested", "nested"),
    ],
)
@pytest.mark.usefixtures("examples")
def test_examples(example: str, task: str) -> None:
    getattr(importlib.import_module(example), task)()


FLUSH_LEFT_CHECK = '''
        async def check(context: Context, step: BeforeToolCall) -> Observation:
            prompt = """
flush-left template
"""
            os.getenv(prompt)
            return Observation.score(0.0)

        return check
'''


@pytest.mark.parametrize(
    "source",
    [
        f"""
def make() -> object:
    @monitor(name=NAME)
    def made() -> Monitor:{FLUSH_LEFT_CHECK}
    return made

watched = make()
""",
        f"""
class Factories:
    @staticmethod
    @monitor(name=NAME)
    def made() -> Monitor:{FLUSH_LEFT_CHECK}

watched = Factories.made
""",
    ],
)
def test_a_nested_factory_with_a_flush_left_string_is_checked(
    load: Load, source: str
) -> None:
    header = f"import os\n{HEADER}NAME = 'made_{uuid.uuid4().hex}'\n"
    with pytest.raises(PortabilityError, match="`os.getenv`"):
        configure(load(header + source))


@pytest.mark.parametrize(
    "setup,body",
    [
        (
            'class Reader:\n    def home(self) -> object:\n        note = """\nflush-left\n"""\n        return os.getenv(note)\n\nHOME = Reader().home',
            "HOME()",
        ),
        (
            "TABLE = {\n    'home': lambda: os.getenv('HOME'),\n}\nHOME = TABLE['home']",
            "HOME()",
        ),
        (
            "def keep(f: object) -> object:\n    return f\nCALL = keep(\n    lambda: os.getenv('HOME'))",
            "CALL",
        ),
    ],
)
def test_bound_methods_and_lambdas_are_located(
    load: Load, setup: str, body: str
) -> None:
    with pytest.raises(PortabilityError, match="`os.getenv`"):
        configure(load(monitor_source(body, f"import os\n{setup}")))


def test_only_the_lambda_on_its_line_is_checked(load: Load) -> None:
    setup = "import os\nPAIR = (os.getcwd, lambda: 1)\nCALL = PAIR[1]"
    configure(load(monitor_source("CALL()", setup)))


@pytest.mark.parametrize(
    "setup",
    [
        "from dataclasses import dataclass, field\n@dataclass\nclass Helper:\n    home: object = field(default_factory=os.getcwd)",
        "from typing import NamedTuple\nclass Helper(NamedTuple):\n    home: object = os.getcwd",
    ],
)
def test_a_class_with_generated_methods_outside_sys_modules_is_checked(
    tmp_path: Path, setup: str
) -> None:
    path = tmp_path / "task_file.py"
    path.write_text(monitor_source("Helper()", f"import os\n{setup}"))
    with pytest.raises(PortabilityError, match="`os.getcwd`"):
        configure(load_task_file(path))


def load_task_file(path: Path) -> ModuleType:
    # how `inspect eval` loads a task file: named by its path, not in sys.modules
    loader = importlib.machinery.SourceFileLoader(path.as_posix(), path.as_posix())
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "rewritten,reason",
    [
        (
            "import os\n\n\n\ndef other() -> None:\n    pass\n",
            "source changed since import",
        ),
        ("def helper(:\n", "unparseable source"),
    ],
)
def test_a_helper_whose_source_changed_could_not_be_checked(
    load: Load, rewritten: str, reason: str
) -> None:
    helpers = load("import os\n\ndef helper() -> None:\n    os.getenv('HOME')\n")
    module = load(monitor_source("helper()", f"from {helpers.__name__} import helper"))
    Path(cast(str, helpers.__file__)).write_text(rewritten)
    with pytest.raises(
        PortabilityError, match=rf"`helper`.*could not be checked \({reason}\)"
    ):
        configure(module)


def test_a_helper_without_source_could_not_be_checked(load: Load) -> None:
    setup = "exec('def helper():\\n    return 1', globals())"
    with pytest.raises(
        PortabilityError, match=r"`helper`.*could not be checked \(no source\)"
    ):
        configure(load(monitor_source("helper()", setup)))


def test_a_class_that_raises_on_lookup_is_still_checked(load: Load) -> None:
    setup = """
import os

class Strict(type):
    def __getattr__(cls, name: str) -> object:
        raise RuntimeError(name)

class Helper(metaclass=Strict):
    def home(self) -> object:
        return os.getenv("HOME")
"""
    with pytest.raises(PortabilityError, match="`os.getenv`"):
        configure(load(monitor_source("Helper().home()", setup)))


def test_deeply_nested_code_could_not_be_checked(load: Load) -> None:
    setup = f"def helper() -> int:\n    return {'-' * 600}1"
    with pytest.raises(PortabilityError, match=r"`helper`.*could not be checked"):
        configure(load(monitor_source("helper()", setup)))


def test_type_parameters_bind_their_names(load: Load) -> None:
    if sys.version_info < (3, 12):
        pytest.skip("type parameter syntax is new in Python 3.12")
    setup = """
import os as T

def generic[T](value: T) -> list[T]:
    return [T]

class Box[T]:
    def kind(self) -> object:
        return T

type Pair[T] = tuple[T, T]
"""
    configure(load(monitor_source("generic(1)\nBox().kind()\nPair", setup)))


def test_a_monitor_configured_inside_a_protocol_is_not_followed(load: Load) -> None:
    source = """
from inspect_sentinel import Context, Decision, Monitor, Observation, Protocol, Step, BeforeToolCall, monitor, protocol, run_monitors

@monitor(name=f"{NAME}_child", portable=False)
def reads_files() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        open("f")
        return Observation.score(0.0)

    return check

@protocol(name=NAME)
def watched() -> Protocol:
    child = reads_files()

    async def decide(context: Context, step: Step) -> Decision | None:
        await run_monitors(context, step, [child])
        return None

    return decide
"""
    configure(load(f"NAME = 'outer_{uuid.uuid4().hex}'\n{source}"))


def test_a_namespace_package_is_the_authors(
    load: Load, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"nspkg_{uuid.uuid4().hex}"
    (tmp_path / name).mkdir()
    (tmp_path / name / "helpers.py").write_text("def clean(text):\n    return text\n")
    module = load(monitor_source(f"import {name}"))
    configure(module)
    assert name not in sys.modules


def test_a_method_reached_through_its_class_is_reported_once(load: Load) -> None:
    setup = "import os\n\nclass Rules:\n    @staticmethod\n    def home() -> object:\n        return os.getenv('HOME')\n"
    with pytest.raises(PortabilityError) as raised:
        configure(load(monitor_source("Rules.home()\nRules().home()", setup)))
    assert str(raised.value).count("`os.getenv`") == 1


def test_a_config_file_entry_raises_portability_error(
    load: Load, tmp_path: Path
) -> None:
    name = f"cfg_portable_{uuid.uuid4().hex}"
    load(monitor_source("open('f')", decorator=f"(name={name!r})"))
    config = tmp_path / "sentinel.yaml"
    config.write_text(f"sentinel:\n  name: {name}\n")
    with pytest.raises(
        PortabilityError, match=rf"sentinel: monitor {name} is portable"
    ):
        sentinel_from_config(str(config))


def test_a_notebook_rechecks_after_a_helper_changes() -> None:
    interactiveshell = pytest.importorskip("IPython.core.interactiveshell")
    shell = interactiveshell.InteractiveShell.instance()
    try:
        shell.run_cell("def helper():\n    return 1\n")
        shell.run_cell(monitor_source("helper()"))
        shell.run_cell("watched()").raise_error()
        shell.run_cell("def helper():\n    return open('f')\n")
        with pytest.raises(PortabilityError, match="`open`"):
            shell.run_cell("watched()").raise_error()
    finally:
        interactiveshell.InteractiveShell.clear_instance()


def test_a_large_module_is_checked_quickly(load: Load) -> None:
    helpers = "".join(
        f'''
class Rule{i}:
    limit = {i}

    def apply(self, text: str) -> float:
        prompt = """
flush-left template {i}
"""
        return float(len(json.dumps([text, prompt])) > self.limit)


def helper{i}(text: str) -> float:
    return Rule{i}().apply(text)
'''
        for i in range(155)
    )
    calls = " + ".join(f"helper{i}('a')" for i in range(155))
    source = monitor_source(calls, f"import json\n{helpers}")
    assert len(source.splitlines()) >= 2000
    module = load(source)
    start = time.perf_counter()
    configure(module)
    assert time.perf_counter() - start < 1.0
