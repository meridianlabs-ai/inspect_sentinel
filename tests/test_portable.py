import ast
import importlib
import importlib.util
import inspect
import linecache
import re
import shutil
import sys
import textwrap
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, cast

import pytest
from inspect_ai._util.registry import is_registry_object, registry_info

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
CORPUS = Path(__file__).parent / "portable_corpus"
DOCS = Path(__file__).parent.parent / "docs"

ENVIRONMENT = "a portable function has no environment variables"
PROCESSES = "a portable function cannot start processes or threads"
NETWORK = "a portable function has no network access"
INSPECT = "inspect_ai is portable only through `inspect_ai.core`"

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


def prepend_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(sys, "path", [path, *sys.path])
    importlib.invalidate_caches()


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


def monitor_source(
    body: str, setup: str = "", decorator: str = "", header: str = HEADER
) -> str:
    indented = textwrap.indent(textwrap.dedent(body).strip(), " " * 8)
    return f"{header}{setup}\n{MONITOR.format(decorator=decorator, body=indented)}"


def configure(module: ModuleType, name: str = "watched") -> object:
    return cast(Callable[[], object], getattr(module, name))()


def refused(load: Load, body: str, setup: str = "") -> str:
    with pytest.raises(PortabilityError) as raised:
        configure(load(monitor_source(body, setup)))
    return str(raised.value)


def write_distribution(site: Path, name: str, files: dict[str, str]) -> None:
    for file, text in files.items():
        (site / file).parent.mkdir(parents=True, exist_ok=True)
        (site / file).write_text(text)
    info = site / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    )
    record = [*files, f"{info.name}/METADATA", f"{info.name}/RECORD"]
    (info / "RECORD").write_text("".join(f"{f},,\n" for f in record))


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "site-packages"
    site.mkdir()
    prepend_path(monkeypatch, str(site))
    return site


# What is refused


@pytest.mark.parametrize(
    "setup,body,reference,resolved",
    [
        (
            "from inspect_ai.model import get_model",
            "get_model()",
            "get_model",
            "function `get_model` from `inspect_ai.model._model`",
        ),
        (
            "import inspect_ai.util",
            "inspect_ai.util.sandbox()",
            "inspect_ai.util.sandbox",
            "function `sandbox` from `inspect_ai.util._sandbox.context`",
        ),
        (
            "from inspect_ai.util import store",
            "store()",
            "store",
            "function `store` from `inspect_ai.core._store`",
        ),
    ],
)
def test_inspect_ai_outside_core_fails(
    load: Load, setup: str, body: str, reference: str, resolved: str
) -> None:
    message = refused(load, body, setup)
    assert f"`{reference}` is {resolved}, and {INSPECT}" in message


@pytest.mark.parametrize(
    "setup,body,reference",
    [
        ("import os", "os.environ['KEY']", "os.environ"),
        ("import os", "os.environ.get('KEY')", "os.environ"),
        ("import os", "os.environb", "os.environb"),
        ("import os", "os.getenv('KEY')", "os.getenv"),
        ("import os", "os.putenv('KEY', 'a')", "os.putenv"),
        ("import os", "os.unsetenv('KEY')", "os.unsetenv"),
        ("from os import environ", "environ['KEY']", "environ"),
        ("from os import getenv as env", "env('KEY')", "env"),
        ("import os\nlookup = os.environ.get", "lookup('KEY')", "lookup"),
    ],
)
def test_environment_variables_fail(
    load: Load, setup: str, body: str, reference: str
) -> None:
    message = refused(load, body, setup)
    assert f"`{reference}` is " in message
    assert ENVIRONMENT in message


@pytest.mark.parametrize(
    "setup,body,reference",
    [
        ("import subprocess", "subprocess.run(['ls'])", "subprocess.run"),
        ("from subprocess import run", "run(['ls'])", "run"),
        ("import os", "os.system('ls')", "os.system"),
        ("import os", "os.execv('ls', [])", "os.execv"),
        ("import os", "os.fork()", "os.fork"),
        ("import os", "os.spawnl(0, 'ls')", "os.spawnl"),
        ("import os", "os.popen('ls')", "os.popen"),
        ("import os", "os.kill(1, 9)", "os.kill"),
        ("import threading", "threading.Thread()", "threading.Thread"),
        ("from threading import Thread", "Thread()", "Thread"),
        ("import threading", "threading.Timer(1, len)", "threading.Timer"),
        (
            "import _thread",
            "_thread.start_new_thread(len, ())",
            "_thread.start_new_thread",
        ),
        ("import pty", "pty.spawn('ls')", "pty.spawn"),
        ("import pty", "pty.fork()", "pty.fork"),
        ("import multiprocessing", "multiprocessing.Pool()", "multiprocessing.Pool"),
        (
            "import concurrent.futures",
            "concurrent.futures.ThreadPoolExecutor()",
            "concurrent.futures.ThreadPoolExecutor",
        ),
        (
            "from concurrent.futures import ProcessPoolExecutor",
            "ProcessPoolExecutor()",
            "ProcessPoolExecutor",
        ),
        ("import asyncio", "asyncio.to_thread(len, 'a')", "asyncio.to_thread"),
        (
            "import asyncio",
            "asyncio.create_subprocess_exec('ls')",
            "asyncio.create_subprocess_exec",
        ),
        ("", "import concurrent.interpreters", "import concurrent.interpreters"),
        ("", "import _interpreters", "import _interpreters"),
        (
            "",
            "from concurrent.futures import InterpreterPoolExecutor",
            "from concurrent.futures import InterpreterPoolExecutor",
        ),
        (
            "import concurrent.futures",
            "concurrent.futures.InterpreterPoolExecutor",
            "concurrent.futures.InterpreterPoolExecutor",
        ),
        (
            "",
            "import concurrent.futures.interpreter",
            "import concurrent.futures.interpreter",
        ),
        ("import anyio", "anyio.run_process(['ls'])", "anyio.run_process"),
        ("import anyio", "anyio.open_process(['ls'])", "anyio.open_process"),
        ("from anyio import run_process", "run_process(['ls'])", "run_process"),
        (
            "import anyio.to_thread",
            "anyio.to_thread.run_sync(len, 'a')",
            "anyio.to_thread.run_sync",
        ),
        (
            "",
            "import anyio\nanyio.to_thread.run_sync(len, 'a')",
            "anyio.to_thread.run_sync",
        ),
        (
            "import anyio.from_thread",
            "anyio.from_thread.run(len)",
            "anyio.from_thread.run",
        ),
        (
            "from anyio import to_process",
            "to_process.run_sync(len)",
            "to_process.run_sync",
        ),
    ],
)
def test_processes_and_threads_fail(
    load: Load, setup: str, body: str, reference: str
) -> None:
    message = refused(load, body, setup)
    assert f"`{reference}` is " in message
    assert PROCESSES in message


@pytest.mark.parametrize(
    "setup,body,reference",
    [
        ("import socket", "socket.socket()", "socket.socket"),
        (
            "import socket",
            "socket.create_connection(('a', 1))",
            "socket.create_connection",
        ),
        ("from socket import AF_INET", "AF_INET", "AF_INET"),
        ("import ssl", "ssl.create_default_context()", "ssl.create_default_context"),
        ("import ssl", "ssl.PROTOCOL_TLS_CLIENT", "ssl.PROTOCOL_TLS_CLIENT"),
        (
            "import http.client",
            "http.client.HTTPSConnection('a')",
            "http.client.HTTPSConnection",
        ),
        (
            "import urllib.request",
            "urllib.request.urlopen('https://a')",
            "urllib.request.urlopen",
        ),
        (
            "import asyncio",
            "asyncio.open_connection('a', 1)",
            "asyncio.open_connection",
        ),
        ("import asyncio", "asyncio.start_server(len)", "asyncio.start_server"),
        ("import httpx", "httpx.AsyncClient()", "httpx.AsyncClient"),
        ("", "import nntplib", "import nntplib"),
        ("", "import telnetlib", "import telnetlib"),
        ("import anyio", "anyio.connect_tcp('a', 1)", "anyio.connect_tcp"),
        ("import anyio", "anyio.connect_unix('a')", "anyio.connect_unix"),
        ("import anyio", "anyio.create_tcp_listener()", "anyio.create_tcp_listener"),
        (
            "import anyio",
            "anyio.create_unix_listener('a')",
            "anyio.create_unix_listener",
        ),
        ("import anyio", "anyio.create_udp_socket()", "anyio.create_udp_socket"),
        (
            "import anyio",
            "anyio.create_connected_udp_socket('a', 1)",
            "anyio.create_connected_udp_socket",
        ),
        ("from anyio import getaddrinfo", "getaddrinfo('a', 1)", "getaddrinfo"),
    ],
)
def test_direct_networking_fails(
    load: Load, setup: str, body: str, reference: str
) -> None:
    message = refused(load, body, setup)
    assert f"`{reference}` is " in message
    assert NETWORK in message


@pytest.mark.parametrize(
    "extension", ["_speedups.cpython-312-x86_64-linux-gnu.so", "_speedups.dll"]
)
def test_a_package_with_compiled_code_fails(
    load: Load, site: Path, extension: str
) -> None:
    name = f"fakec_{uuid.uuid4().hex}"
    write_distribution(
        site,
        name,
        {
            f"{name}/__init__.py": "LIMIT = 3\ndef fast(): return 1\n",
            f"{name}/{extension}": "",
        },
    )
    message = refused(load, f"{name}.fast()\n{name}.LIMIT", f"import {name}")
    reason = f"and `{name}` contains compiled extension modules, which a portable function cannot load"
    assert f"`{name}.fast` is function `fast` from `{name}`, {reason}" in message
    assert f"`{name}.LIMIT` is a value in `{name}`, {reason}" in message


def test_a_pure_python_package_passes(load: Load, site: Path) -> None:
    name = f"fakepure_{uuid.uuid4().hex}"
    write_distribution(site, name, {f"{name}/__init__.py": "def get(): return 1\n"})
    configure(load(monitor_source(f"{name}.get()", f"import {name}")))


def test_pydantic_core_passes(load: Load) -> None:
    configure(
        load(
            monitor_source(
                "to_json({})\nValidationError",
                "from pydantic_core import ValidationError, to_json",
            )
        )
    )


# What passes


@pytest.mark.parametrize(
    "setup,body",
    [
        (
            "from inspect_ai.core import ChatMessageUser, ToolCall",
            "ChatMessageUser(content='a')\nToolCall",
        ),
        ("import inspect_ai.core", "inspect_ai.core.ModelOutput"),
        (
            "from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, StopReason",
            "ChatMessage\nChatMessageUser\nGenerateConfig()\nStopReason",
        ),
        (
            "import inspect_ai.model",
            "inspect_ai.model.ModelOutput.from_content('m', 'a')",
        ),
        ("from inspect_ai.scorer import Reference", "Reference"),
        ("from inspect_ai.util import StoreModel", "StoreModel"),
        ("from inspect_ai.tool import ToolCall", "ToolCall"),
    ],
)
def test_inspect_ai_core_and_its_aliases_pass(
    load: Load, setup: str, body: str
) -> None:
    configure(load(monitor_source(body, setup)))


@pytest.mark.parametrize(
    "setup,body",
    [
        (
            "import os",
            "os.path.join('a', 'b')\nos.path.exists('a')\nos.sep\nos.getcwd()",
        ),
        ("from os import path, sep", "path.join('a', sep)"),
        ("import sys", "sys.argv\nsys.version_info"),
        ("", "open('f')\nexec('1')\neval('1')\ncompile('1', 'f', 'eval')"),
        ("import pathlib", "pathlib.Path('f').read_text()"),
        ("import io, pickle", "io.open('f')\npickle.loads(b'')"),
        ("import time", "time.sleep(1)\ntime.monotonic()"),
        ("import logging", "logging.basicConfig()\nlogging.getLogger('a').info('a')"),
        ("import json, re, math", "json.dumps({})\nre.compile('a')\nmath.sqrt(2)"),
        ("import asyncio", "asyncio.Lock()\nasyncio.sleep(0)\nasyncio.wait_for"),
        ("import concurrent.futures", "concurrent.futures.Future"),
        (
            "import threading, _thread",
            "threading.Lock()\nthreading.RLock()\nthreading.Event()\nthreading.Condition()"
            "\nthreading.Semaphore()\nthreading.local()\n_thread.allocate_lock()",
        ),
        (
            "import anyio",
            "anyio.run\nanyio.sleep(0)\nanyio.create_task_group()\nanyio.CancelScope()"
            "\nanyio.fail_after(1)\nanyio.Lock()\nanyio.Event()"
            "\nanyio.create_memory_object_stream()\nanyio.to_thread",
        ),
        ("import asyncio", "asyncio.run\nasyncio.TaskGroup\nasyncio.Event()"),
        ("from pydantic import BaseModel", "BaseModel"),
        ("import http", "http.HTTPStatus.OK"),
    ],
)
def test_what_is_not_checked_passes(load: Load, setup: str, body: str) -> None:
    configure(load(monitor_source(body, setup)))


def test_an_unreferenced_task_beside_the_monitor_passes(load: Load) -> None:
    setup = """
import json
import subprocess
from inspect_ai import Task, task
from inspect_ai.model import get_model
from inspect_ai.util import sandbox

@task
def unreferenced() -> Task:
    get_model()
    sandbox()
    subprocess.run(["ls"])
    return Task()
"""
    configure(load(monitor_source("json.dumps({})", setup)))


@pytest.mark.parametrize(
    "setup,body",
    [
        (
            "import os\n\ndef home() -> object:\n    return os.getenv('HOME')\n",
            "home()",
        ),
        (
            "import os\n\nclass Settings:\n    def key(self) -> object:\n        return os.getenv('A')\n",
            "Settings().key()\nSettings.key",
        ),
        (
            "import functools, os\ngetter = functools.partial(os.getenv, 'A')",
            "getter()",
        ),
        (
            "import os, types\nholder = types.SimpleNamespace(env=os.environ)",
            "holder.env",
        ),
    ],
)
def test_what_a_reference_leads_to_is_not_followed(
    load: Load, setup: str, body: str
) -> None:
    configure(load(monitor_source(body, setup)))


def test_annotations_are_not_checked(load: Load) -> None:
    body = """
    def inner(model: Model | None = None) -> Model | None:
        return model
    held: Model | None = inner()
    """
    configure(load(monitor_source(body, "from inspect_ai.model import Model")))


# Scoping


@pytest.mark.parametrize(
    "body",
    [
        "os = step\nos.environ",
        "for subprocess in []:\n    pass\nsubprocess.run",
        "def inner(getenv: object) -> None:\n    pass\ngetenv('KEY')",
        "def inner() -> None:\n    socket = 1\nsocket.socket",
        "[socket for socket in []]\nsocket.socket",
        "try:\n    pass\nexcept Exception as os:\n    pass\nos.environ",
    ],
)
def test_a_name_assigned_in_the_factory_is_local_throughout(
    load: Load, body: str
) -> None:
    setup = "import os, socket, subprocess\nfrom os import getenv"
    configure(load(monitor_source(body, setup)))


def test_a_factory_parameter_shadowing_a_module_passes(load: Load) -> None:
    source = f"""{HEADER}import os

@monitor
def watched(os: str = "a") -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        os.environ
        return Observation.score(0.0)

    return check
"""
    configure(load(source))


def test_a_closure_cell_is_resolved(load: Load) -> None:
    source = f"""{HEADER}
def make() -> object:
    import subprocess as sp

    @monitor
    def watched() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation:
            sp.run(["ls"])
            return Observation.score(0.0)

        return check

    return watched
"""
    factory = cast(Callable[[], Callable[[], object]], load(source).make)()
    with pytest.raises(
        PortabilityError, match=r"`sp\.run` is function `run` from `subprocess`"
    ):
        factory()


def test_class_bases_and_keywords_are_checked_where_the_class_is(
    load: Load,
) -> None:
    body = """
    class Local(subprocess.Popen, flag=os.getenv):
        def run(self) -> None:
            os.environ
    """
    message = refused(load, body, "import os, subprocess")
    assert "in watched.<locals>.check: `subprocess.Popen`" in message
    assert "in watched.<locals>.check: `os.getenv`" in message
    assert (
        "in watched.<locals>.check.<locals>.Local.<locals>.run: `os.environ`" in message
    )


def test_the_factory_body_and_nested_functions_are_checked(load: Load) -> None:
    source = f"""{HEADER}import os
import subprocess

@monitor
def watched() -> Monitor:
    key = os.environ["KEY"]
    run = lambda: subprocess.run(["ls"])

    def helper() -> object:
        return os.getenv("A")

    async def check(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.0, key)

    return check
"""
    with pytest.raises(PortabilityError) as raised:
        configure(load(source))
    message = str(raised.value)
    assert ":7 in watched: `os.environ`" in message
    assert ":8 in watched.<locals>.<lambda>: `subprocess.run`" in message
    assert ":11 in watched.<locals>.helper: `os.getenv`" in message


# Imports inside the factory


@pytest.mark.parametrize(
    "body,reference",
    [
        ("import subprocess", "import subprocess"),
        ("import json, multiprocessing", "import multiprocessing"),
        ("from os import getenv, environ", "from os import getenv"),
        ("from os import getenv, environ", "from os import environ"),
        ("from os import environ", "from os import environ"),
        (
            "from inspect_ai.model import get_model",
            "from inspect_ai.model import get_model",
        ),
        ("import xmlrpc.client", "import xmlrpc.client"),
    ],
)
def test_an_import_inside_the_factory_is_judged(
    load: Load, body: str, reference: str
) -> None:
    assert f"`{reference}` is " in refused(load, body)


@pytest.mark.parametrize(
    "body",
    [
        "import json\njson.dumps({})",
        "from inspect_ai.core import ToolCall",
        "from inspect_ai.model import ChatMessageUser",
        "import os\nos.path.join('a', 'b')",
    ],
)
def test_an_allowed_import_inside_the_factory_passes(load: Load, body: str) -> None:
    configure(load(monitor_source(body)))


@pytest.mark.parametrize(
    "body,reference,reason",
    [
        ("import os\nos.environ.get('KEY')", "os.environ", ENVIRONMENT),
        ("import os as o\no.getenv('KEY')", "o.getenv", ENVIRONMENT),
        ("from os import environ as env\nenv['KEY']", "env", ENVIRONMENT),
        (
            "import concurrent.futures\nconcurrent.futures.ThreadPoolExecutor()",
            "concurrent.futures.ThreadPoolExecutor",
            PROCESSES,
        ),
        ("import anyio\nanyio.connect_tcp('a', 1)", "anyio.connect_tcp", NETWORK),
    ],
)
def test_a_name_imported_inside_the_factory_resolves_to_its_module(
    load: Load, body: str, reference: str, reason: str
) -> None:
    message = refused(load, body)
    assert f"`{reference}` is " in message
    assert reason in message


def test_a_name_imported_and_assigned_in_the_factory_is_local(load: Load) -> None:
    configure(load(monitor_source("import os\nos = step\nos.environ")))


def test_a_non_module_in_sys_modules_is_judged_by_name(
    load: Load, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"fake_mod_{uuid.uuid4().hex}"
    monkeypatch.setitem(sys.modules, name, object())
    configure(load(monitor_source(f"from {name} import x\nx")))


def test_the_check_imports_nothing(load: Load, site: Path) -> None:
    name = f"fakec_{uuid.uuid4().hex}"
    write_distribution(site, name, {f"{name}/__init__.py": "", f"{name}/_c.so": ""})
    unloaded = f"unloaded_{uuid.uuid4().hex}"
    (site / f"{unloaded}.py").write_text("")
    body = f"import {unloaded}\n{unloaded}.y\nfrom {unloaded} import y\nimport {name}\n{name}.x"
    module = load(monitor_source(body))
    before = set(sys.modules)
    with pytest.raises(PortabilityError, match=f"`import {name}`"):
        configure(module)
    assert set(sys.modules) == before


def test_a_lazy_module_is_not_loaded_by_the_check(load: Load, site: Path) -> None:
    name = f"fakec_{uuid.uuid4().hex}"
    write_distribution(
        site,
        name,
        {f"{name}/__init__.py": "raise RuntimeError('loaded')\n", f"{name}/_c.so": ""},
    )
    setup = f"""
import importlib.util
spec = importlib.util.find_spec({name!r})
spec.loader = importlib.util.LazyLoader(spec.loader)
lazy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lazy)
"""
    message = refused(load, "lazy\nlazy.fast()", setup)
    assert f"`lazy` is module `{name}`" in message
    assert f"`lazy.fast` is `{name}.fast` (not loaded)" in message


# The error


def test_every_violation_is_reported_together(load: Load) -> None:
    body = """
    os.getenv("A")
    subprocess.run(["ls"])
    get_model()
    """
    setup = "import os, subprocess\nfrom inspect_ai.model import get_model"
    module = load(monitor_source(body, setup))
    with pytest.raises(PortabilityError) as raised:
        configure(module)
    file = module.__file__
    lines = str(raised.value).splitlines()
    assert (
        lines[0]
        == "monitor watched is portable, but its code references what a portable monitor cannot use:"
    )
    assert lines[1:4] == [
        f"- {file}:8 in watched.<locals>.check: `os.getenv` is function `getenv` from `os`, and {ENVIRONMENT}; take configuration as factory parameters",
        f"- {file}:9 in watched.<locals>.check: `subprocess.run` is function `run` from `subprocess`, and {PROCESSES}",
        f"- {file}:10 in watched.<locals>.check: `get_model` is function `get_model` from `inspect_ai.model._model`, and {INSPECT}",
    ]
    fixes = lines[4]
    for fix in (
        "context.host.generate()",
        "context.host.ask_human()",
        "context.store_as()",
        "factory parameters",
        "move the reference",
        "@monitor(portable=False)",
    ):
        assert fix in fixes


@pytest.mark.parametrize(
    "setup,body,described",
    [
        (
            "import os\nlookup = os.environ.get",
            "lookup('KEY')",
            f"`lookup` is method `get` of `os.environ`, and {ENVIRONMENT}",
        ),
        (
            "import os",
            "os.environ",
            f"`os.environ` is a value in `os`, and {ENVIRONMENT}",
        ),
        (
            "import ssl",
            "ssl.PROTOCOL_TLS_CLIENT",
            f"`ssl.PROTOCOL_TLS_CLIENT` is a value in `ssl`, and {NETWORK}",
        ),
        (
            "",
            "import telnetlib",
            f"`import telnetlib` is `telnetlib` (not loaded), and {NETWORK}",
        ),
        ("import socket", "socket", f"`socket` is module `socket`, and {NETWORK}"),
    ],
)
def test_what_a_reference_resolved_to_is_described(
    load: Load, setup: str, body: str, described: str
) -> None:
    assert described in refused(load, body, setup)


def test_a_protocol_is_checked(load: Load) -> None:
    source = """
import os
from inspect_sentinel import Context, Decision, Protocol, Step, protocol

@protocol
def decides_portably() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        os.getenv("A")
        return None

    return decide
"""
    with pytest.raises(
        PortabilityError, match="protocol decides_portably is portable"
    ) as raised:
        configure(load(source), "decides_portably")
    assert "@protocol(portable=False)" in str(raised.value)


def test_a_portability_error_is_a_type_error() -> None:
    assert issubclass(PortabilityError, TypeError)


# Opting out and when the check runs


def test_portable_is_recorded_in_the_registry(load: Load) -> None:
    module = load(monitor_source("pass"))
    assert registry_info(module.watched).metadata["portable"] is True


def test_portable_false_skips_the_check(load: Load) -> None:
    module = load(
        monitor_source("os.getenv('A')", "import os", decorator="(portable=False)")
    )
    configure(module)
    assert registry_info(module.watched).metadata["portable"] is False


def test_portable_must_be_a_bool(load: Load) -> None:
    with pytest.raises(TypeError, match="portable must be True or False"):
        load(monitor_source("pass", decorator="(portable=1)"))


def test_importing_checks_nothing(load: Load) -> None:
    load(monitor_source("os.getenv('A')", "import os"))


def test_without_source_the_check_is_skipped() -> None:
    namespace: dict[str, Any] = {}
    code = compile(monitor_source("os.getenv('A')", "import os"), "<string>", "exec")
    exec(code, namespace)
    namespace["watched"]()


def test_a_factory_whose_file_is_gone_is_skipped(load: Load) -> None:
    module = load(monitor_source("os.getenv('A')", "import os"))
    Path(cast(str, module.__file__)).unlink()
    linecache.clearcache()
    configure(module)


def test_a_factory_moved_in_its_edited_file_is_still_checked(load: Load) -> None:
    module = load(monitor_source("os.getenv('A')", "import os"))
    path = Path(cast(str, module.__file__))
    path.write_text("# edited\n\n" + path.read_text())
    linecache.clearcache()
    with pytest.raises(
        PortabilityError, match=":9 in watched.<locals>.check: `os.getenv`"
    ):
        configure(module)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695")
def test_type_aliases_and_bounds_are_not_checked(load: Load) -> None:
    body = """
    type Alias = Model

    def pick[T: Model](value: T) -> T:
        return value
    """
    configure(load(monitor_source(body, "from inspect_ai.model import Model")))


def test_a_lazy_proxy_is_judged_by_its_type(load: Load) -> None:
    setup = """
class Lazy:
    @property
    def __class__(self) -> type:
        raise RuntimeError("not configured")

settings = Lazy()
"""
    configure(load(monitor_source("settings", setup)))


def test_an_unreadable_record_is_skipped(load: Load, site: Path) -> None:
    info = site / "broken-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: broken\nVersion: 1.0\n"
    )
    (info / "RECORD").write_bytes(b"caf\xe9/x.so,,\n")
    configure(load(monitor_source("json.dumps({})", "import json")))


@pytest.mark.parametrize(
    "replacement,reason",
    [
        ("def (:\n", "its source could not be parsed"),
        (
            "x = 1\n" * 20 + "def watched():\n    pass\ndef watched():\n    pass\n",
            "its source could not be located",
        ),
    ],
)
def test_a_factory_whose_source_cannot_be_used_could_not_be_checked(
    load: Load, replacement: str, reason: str
) -> None:
    module = load(monitor_source("pass"))
    file = cast(str, module.__file__)
    Path(file).write_text(replacement)
    linecache.clearcache()
    with pytest.raises(PortabilityError) as raised:
        configure(module)
    assert str(raised.value) == (
        f"monitor watched is portable, but its factory `watched` ({file}:4) could not be checked: {reason}. Fix the source, or declare it with `@monitor(portable=False)`."
    )


def test_a_jupyter_cell_is_checked() -> None:
    interactiveshell = pytest.importorskip("IPython.core.interactiveshell")
    shell = interactiveshell.InteractiveShell.instance()
    try:
        shell.run_cell(monitor_source("os.getenv('A')", "import os"))
        with pytest.raises(PortabilityError, match="`os.getenv`"):
            shell.run_cell("watched()").raise_error()
    finally:
        interactiveshell.InteractiveShell.clear_instance()


def test_a_config_file_entry_raises_portability_error(
    load: Load, tmp_path: Path
) -> None:
    name = f"cfg_portable_{uuid.uuid4().hex}"
    load(monitor_source("os.getenv('A')", "import os", decorator=f"(name={name!r})"))
    config = tmp_path / "sentinel.yaml"
    config.write_text(f"sentinel:\n  name: {name}\n")
    with pytest.raises(
        PortabilityError, match=rf"sentinel: monitor {name} is portable"
    ):
        sentinel_from_config(str(config))


# Realistic monitors, shipped protocols and examples


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
def modules_from(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Path], None]]:
    before = set(sys.modules)
    roots: list[Path] = []

    def modules_from(root: Path) -> None:
        roots.append(root)
        prepend_path(monkeypatch, str(root))

    yield modules_from
    for name in set(sys.modules) - before:
        if Path(getattr(sys.modules[name], "__file__", "") or "").parent in roots:
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
def test_examples(
    modules_from: Callable[[Path], None], example: str, task: str
) -> None:
    modules_from(EXAMPLES)
    getattr(importlib.import_module(example), task)()


class DocBlock(NamedTuple):
    id: str
    earlier: list[str]
    source: str


_FENCE = re.compile(
    r"^```\s*(?:python|\{\.python[^}]*\})\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL
)

_DOC_BLOCKS_NOT_RUNNABLE = {
    "docs/portability.qmd:1": "a `...` sketch of `portable=False`, which the check skips",
}


def factory_names(source: str) -> list[str]:
    return [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(name := d.func if isinstance(d, ast.Call) else d, ast.Name)
            and name.id in ("monitor", "protocol")
            for d in node.decorator_list
        )
    ]


def doc_blocks() -> list[DocBlock]:
    found: list[DocBlock] = []
    for page in sorted(DOCS.rglob("*.qmd")):
        if any(part.startswith("_") for part in page.relative_to(DOCS).parts):
            continue
        blocks = _FENCE.findall(page.read_text())
        for index, block in enumerate(blocks):
            if factory_names(block):
                id = f"{page.relative_to(DOCS.parent).as_posix()}:{index + 1}"
                found.append(DocBlock(id, blocks[:index], block))
    return found


def import_doc_block(load: Load, block: DocBlock) -> ModuleType | None:
    for source in (block.source, "\n".join([*block.earlier, block.source])):
        try:
            return load(source)
        except Exception:
            continue
    return None


def needs_arguments(factory: Callable[..., object]) -> bool:
    return any(
        p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in inspect.signature(factory).parameters.values()
    )


@pytest.mark.parametrize("block", doc_blocks(), ids=lambda block: block.id)
def test_doc_snippets_pass(
    load: Load, modules_from: Callable[[Path], None], block: DocBlock
) -> None:
    modules_from(EXAMPLES)
    module = import_doc_block(load, block)
    if block.id in _DOC_BLOCKS_NOT_RUNNABLE:
        assert module is None, (
            f"{block.id} now runs; remove it from _DOC_BLOCKS_NOT_RUNNABLE"
        )
        pytest.skip(_DOC_BLOCKS_NOT_RUNNABLE[block.id])
    assert module is not None, (
        f"{block.id} does not import, alone or after the page's earlier blocks"
    )
    factories = [
        factory
        for factory in (getattr(module, name) for name in factory_names(block.source))
        if is_registry_object(factory, "monitor")
        or is_registry_object(factory, "protocol")
    ]
    callable_now = [f for f in factories if not needs_arguments(f)]
    if not callable_now:
        pytest.skip(f"{block.id}: every factory needs arguments")
    for factory in callable_now:
        factory()


def test_doc_exclusions_name_doc_blocks() -> None:
    assert set(_DOC_BLOCKS_NOT_RUNNABLE) <= {block.id for block in doc_blocks()}


def test_realistic_monitors_and_protocols_pass(
    modules_from: Callable[[Path], None], tmp_path: Path
) -> None:
    for data in CORPUS.glob("*.py.txt"):
        shutil.copy(data, tmp_path / data.name.removesuffix(".txt"))
    modules_from(tmp_path)
    a = importlib.import_module("corp_pass_a")
    b = importlib.import_module("corp_pass_b")
    compose = importlib.import_module("corp_compose")
    factories = cast(dict[str, Callable[..., object]], {**vars(a), **vars(b)})
    for name in (
        "llm_judge llm_retry llm_classify rule_words secrets_scan tally policy grab_bag "
        "window wrapped ensemble asyncio_ensemble agen trajectory cached matcher "
        "budget keyword boxed serial_llm"
    ).split():
        factories[name]()
    rules, tally = factories["rule_words"], factories["tally"]
    factories["escalating"](rules())
    factories["two_stage"]([rules(), tally()])
    factories["top_k"]([rules(), factories["escalating"](tally())])
    cast(Callable[[], object], compose.layered)()
