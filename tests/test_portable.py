import importlib
import importlib.util
import linecache
import sys
import textwrap
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

Load = Callable[..., ModuleType]

EXAMPLES = Path(__file__).parent.parent / "examples"
CORE = importlib.util.find_spec("inspect_ai.core") is not None

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
    ],
)
def test_allowed_references_pass(load: Load, setup: str, body: str) -> None:
    configure(load(monitor_source(body, setup)))


@pytest.mark.parametrize(
    "setup,body,reference",
    [
        ("import os", "os.environ.get('HOME')", "os.environ"),
        ("import os", "os.getenv('HOME')", "os.getenv"),
        ("import os", "os.system('ls')", "os.system"),
        ("import os", "os.sep", "os.sep"),
        ("import sys", "sys.modules", "sys.modules"),
        ("import sys", "sys", "sys"),
        ("import subprocess", "subprocess.run(['ls'])", "subprocess.run"),
        ("import socket", "socket.create_connection(('x', 1))", "socket"),
        ("import pathlib", "pathlib.Path('f').read_text()", "pathlib.Path"),
        ("import pickle", "pickle.loads(b'')", "pickle.loads"),
        ("import io", "io.StringIO()", "io.StringIO"),
        (
            "import asyncio",
            "asyncio.create_subprocess_exec('ls')",
            "asyncio.create_subprocess_exec",
        ),
        ("reader = open", "reader('f')", "reader"),
        ("import builtins", "builtins.open('f')", "builtins.open"),
        ("import importlib", "importlib.import_module('os')", "importlib"),
        ("", "open('f')", "open"),
        ("", "input()", "input"),
        ("", "eval('1')", "eval"),
        ("", "exec('1')", "exec"),
        ("", "compile('1', 'f', 'eval')", "compile"),
        ("", "__import__('os')", "__import__"),
        ("", "breakpoint()", "breakpoint"),
        ("from inspect_ai.model import get_model", "get_model()", "get_model"),
        ("import inspect_ai.model", "inspect_ai.model.get_model()", "get_model"),
        ("import os as o", "o.environ['HOME']", "o.environ"),
        ("from os import environ as e", "e.get('HOME')", "e"),
        ("from os import *", "getenv('HOME')", "getenv"),
        ("import functools, os", "functools.partial(os.system, 'ls')", "os.system"),
        ("", "import subprocess", "import subprocess"),
        ("", "from os import environ", "from os import environ"),
        ("import os", "(lambda: os.getenv('HOME'))()", "os.getenv"),
        ("import os", "[os.getenv(k) for k in ('A', 'B')]", "os.getenv"),
        (
            "import os",
            "def inner() -> str | None:\n    return os.getenv('HOME')\ninner()",
            "os.getenv",
        ),
        ("", "import os.path\nos.system('ls')", "os.system"),
        ("", "import posixpath\nposixpath.os.system('ls')", "posixpath.os.system"),
        (
            "from inspect_sentinel._runner import run_sentinel",
            "run_sentinel",
            "run_sentinel",
        ),
        (
            "import functools, os\n@functools.cache\ndef helper() -> None:\n    os.system('ls')",
            "helper()",
            "os.system",
        ),
    ],
)
def test_disallowed_references_fail(
    load: Load, setup: str, body: str, reference: str
) -> None:
    with pytest.raises(PortabilityError, match=f"`{reference}`"):
        configure(load(monitor_source(body, setup)))


def test_a_standard_library_module_off_the_list_names_the_list(load: Load) -> None:
    module = load(monitor_source("pathlib.Path('f')", "import pathlib"))
    with pytest.raises(PortabilityError, match="not on the portable standard-library"):
        configure(module)


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


def test_inspect_ai_outside_core_names_the_reason(load: Load) -> None:
    module = load(
        monitor_source("get_model()", "from inspect_ai.model import get_model")
    )
    with pytest.raises(PortabilityError, match="portable only through"):
        configure(module)


def test_inspect_ai_core_passes(load: Load) -> None:
    pytest.importorskip("inspect_ai.core")
    setup = "from inspect_ai.core import ChatMessageUser\nfrom inspect_ai.model import ChatMessageAssistant"
    body = "ChatMessageUser(content='a')\nChatMessageAssistant(content='b')"
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
import subprocess

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
    code = compile(monitor_source("open('f')"), "<notebook>", "exec")
    exec(code, namespace)
    namespace["watched"]()


def test_the_check_runs_once_per_factory(load: Load) -> None:
    module = load(monitor_source("pass"))
    configure(module)
    path = Path(cast(str, module.__file__))
    path.write_text(path.read_text().replace("        pass", "        open('f')"))
    linecache.clearcache()
    configure(module)


def test_every_violation_is_reported_together(load: Load) -> None:
    body = """
import subprocess
open('f')
os.environ['HOME']
"""
    module = load(monitor_source(body, "import os"))
    with pytest.raises(PortabilityError) as raised:
        configure(module)
    message = str(raised.value)
    lines = [line for line in message.splitlines() if line.startswith("- ")]
    assert len(lines) == 3
    assert f"{module.__file__}:" in lines[0]
    assert "watched.<locals>.check" in lines[0]
    assert "monitor watched is portable" in message
    assert "context.host.generate()" in message
    assert "@monitor(portable=False)" in message


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


# until inspect_ai.core exists, llm_suspicion's GenerateConfig is in inspect_ai.model
@pytest.mark.parametrize(
    "example,task,portable",
    [
        ("no_network", "network_rule", True),
        ("trajectory", "trajectory", True),
        ("escalate_to_human", "escalate_to_human", True),
        ("llm_suspicion", "llm_suspicion", CORE),
        ("nested", "nested", CORE),
    ],
)
@pytest.mark.usefixtures("examples")
def test_examples(example: str, task: str, portable: bool) -> None:
    configure_task = getattr(importlib.import_module(example), task)
    if portable:
        configure_task()
    else:
        with pytest.raises(PortabilityError, match="`VERDICT`"):
            configure_task()
