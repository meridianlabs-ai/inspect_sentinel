# The portable check runs when a portable factory is called. It parses the
# factory's file, finds the factory, and resolves the names its code references
# one level (an import in the factory, closure cell, global or builtin, then
# attributes through modules and names not loaded), raising for each that
# resolves to something a sandboxed guest cannot do. It never imports anything.
from __future__ import annotations

import ast
import functools
import importlib.metadata
import inspect
import linecache
import os
import sys
import types
from collections.abc import Callable, Iterable, Iterator
from typing import NamedTuple, TypeGuard, cast

import inspect_core

_CORE = "inspect_core"

_ENVIRONMENT = "and a portable function has no environment variables; take configuration as factory parameters"
_PROCESSES = "and a portable function cannot start processes or threads"
_NETWORK = "and a portable function has no network access; call a model with `context.host.generate()`"
_INSPECT = f"and inspect_ai is not portable; use `{_CORE}`"
_CORE_PRIVATE = f"and only the names `{_CORE}` exports are portable"

_OS_ENVIRONMENT = ("environ", "environb", "getenv", "getenvb", "putenv", "unsetenv")
_OS_PROCESSES = ("system", "popen", "kill", "killpg", "startfile")
_OS_PROCESS_PREFIXES = ("exec", "fork", "spawn", "posix_spawn")

# Modules are refused with everything in them, other names exactly. A WASM
# guest has no threads (componentize-py: "can't start new thread"), so only
# what starts a thread or process is refused; locks and events pass, and
# concurrency in a guest is async. concurrent.futures and anyio load these
# names lazily, and anyio rewrites their `__module__`, so they are named too.
_PROCESS_NAMES = """
subprocess _posixsubprocess multiprocessing _multiprocessing threading.Thread
threading.Timer _thread.start_new_thread _thread.start_new
concurrent.futures.thread concurrent.futures.process
concurrent.futures.interpreter concurrent.futures.ThreadPoolExecutor
concurrent.futures.ProcessPoolExecutor concurrent.futures.InterpreterPoolExecutor
concurrent.interpreters _interpreters pty.spawn pty.fork asyncio.subprocess
asyncio.threads anyio.to_thread.run_sync anyio.from_thread anyio.run_process
anyio.open_process anyio.to_process anyio.to_interpreter
""".split()
_NETWORK_NAMES = """
socket _socket ssl _ssl socketserver http.client http.server urllib.request
ftplib smtplib imaplib poplib nntplib telnetlib xmlrpc asyncio.streams httpx
httpcore requests urllib3 anyio.connect_tcp anyio.connect_unix
anyio.create_tcp_listener anyio.create_unix_listener anyio.create_udp_socket
anyio.create_connected_udp_socket anyio.create_unix_datagram_socket
anyio.create_connected_unix_datagram_socket anyio.getaddrinfo anyio.getnameinfo
""".split()
_REFUSED = {
    **{
        f"os.{name}": (_ENVIRONMENT if name in _OS_ENVIRONMENT else _PROCESSES)
        for name in dir(os)
        if name in _OS_ENVIRONMENT
        or name in _OS_PROCESSES
        or name.startswith(_OS_PROCESS_PREFIXES)
    },
    **dict.fromkeys(_PROCESS_NAMES, _PROCESSES),
    **dict.fromkeys(_NETWORK_NAMES, _NETWORK),
}

# built for WASI by sentinel
_WASI_BUILT = frozenset({"pydantic_core"})
_EXTENSIONS = (".so", ".pyd", ".dylib", ".dll")

_FIXES = "A portable {kind} affects the outside world only through `context`: call a model with `context.host.generate()`, ask a person with `context.host.ask_human()`, and keep state with `context.store_as()`. Take configuration as factory parameters. Otherwise move the reference out of the {kind}'s code, or, if the {kind} only runs in an eval, declare it with `@{kind}(portable=False)`."


class PortabilityError(TypeError):
    """A portable monitor or protocol references something a portable function cannot use.

    Raised when the factory of a `@monitor` or `@protocol` declared with `portable=True` (the default) is called, before it runs. The message lists each reference in the factory's code that a portable function cannot use: the file and line, the function, the name, what it resolved to and why. A factory whose source exists but cannot be parsed or located is reported as not checked. It subclasses `TypeError`, like the factory's other configuration checks.
    """


class _Violation(NamedTuple):
    line: int
    function: str
    reference: str
    resolved: str
    reason: str


class _Result(NamedTuple):
    reference: str
    resolved: str
    reason: str


class _Bindings(NamedTuple):
    local: frozenset[str]
    # bound only by imports in the factory: name -> (module, attribute or None)
    imported: dict[str, tuple[str, str | None]]


def check_portable(factory: Callable[..., object], kind: str, name: str) -> None:
    func = inspect.unwrap(factory)
    if not isinstance(func, types.FunctionType):
        return
    code = func.__code__
    file = code.co_filename
    # every exec() of a string shares `<string>`, so its lines are not this code's
    lines = [] if file == "<string>" else linecache.getlines(file, func.__globals__)
    if not lines:
        return
    node = _locate("".join(lines), code)
    if isinstance(node, str):
        raise PortabilityError(
            f"{kind} {name} is portable, but its factory `{func.__qualname__}` ({file}:{code.co_firstlineno}) could not be checked: {node}. Fix the source, or declare it with `@{kind}(portable=False)`."
        )
    violations = _check(node, func)
    if violations:
        listed = "\n".join(
            f"- {file}:{v.line} in {v.function}: `{v.reference}` is {v.resolved}, {v.reason}"
            for v in violations
        )
        raise PortabilityError(
            f"{kind} {name} is portable, but its code references what a portable {kind} cannot use:\n{listed}\n{_FIXES.format(kind=kind)}"
        )


_MISSING = object()
_Function = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_Scope = (*_Function, ast.ClassDef)


def _locate(text: str, code: types.CodeType) -> ast.AST | str:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return "its source could not be parsed"
    named = [
        node
        for node in ast.walk(tree)
        if isinstance(node, _Function) and _name(node) == code.co_name
    ]
    for node in named:
        decorators = getattr(node, "decorator_list", [])
        first = min([node.lineno, *(d.lineno for d in decorators)])
        if code.co_firstlineno in (first, node.lineno):
            return node
    # the file was edited since import, but its one definition of that name is
    # still the factory as far as the check can tell
    if len(named) == 1:
        return named[0]
    return "its source could not be located"


def _name(node: ast.AST) -> str:
    return "<lambda>" if isinstance(node, ast.Lambda) else getattr(node, "name", "")


def _check(factory: ast.AST, func: types.FunctionType) -> list[_Violation]:
    bindings = _bindings(factory)
    cells = {
        name: _contents(cell)
        for name, cell in zip(
            func.__code__.co_freevars, func.__closure__ or (), strict=True
        )
    }
    found: dict[tuple[int, str], _Violation] = {}
    for node, chain, where in _references(factory, func.__qualname__):
        results: Iterable[_Result | None]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            results = _judge_import(node)
        elif chain[0] in bindings.local:
            continue
        else:
            results = [_resolve(chain, func.__globals__, cells, bindings.imported)]
        line = getattr(node, "lineno", 0)
        for result in results:
            if result is not None:
                found.setdefault(
                    (line, result.reference), _Violation(line, where, *result)
                )
    return sorted(found.values())


def _contents(cell: types.CellType) -> object:
    try:
        return cell.cell_contents
    except ValueError:
        return _MISSING


def _bindings(factory: ast.AST) -> _Bindings:
    names: set[str] = set()
    imports: dict[str, set[tuple[str, str | None]]] = {}
    for node in ast.walk(factory):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name if alias.asname else alias.name.partition(".")[0]
                bound = alias.asname or module
                imports.setdefault(bound, set()).add((module, None))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if node.level == 0 and node.module:
                    imports.setdefault(bound, set()).add((node.module, alias.name))
                else:
                    names.add(bound)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, _Scope) and node is not factory:
            names.add(_name(node))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        else:
            for field in ("name", "rest"):
                # except, match and type-parameter targets
                value = getattr(node, field, None)
                if isinstance(value, str) and not isinstance(node, ast.alias):
                    names.add(value)
    imported = {
        bound: next(iter(targets))
        for bound, targets in imports.items()
        if len(targets) == 1 and bound not in names
    }
    return _Bindings(frozenset(names | (imports.keys() - imported.keys())), imported)


def _references(
    factory: ast.AST, qualname: str
) -> Iterator[tuple[ast.AST, list[str], str]]:
    body: object = getattr(factory, "body", [])
    stack = [(n, qualname) for n in _nodes(body)]
    while stack:
        node, where = stack.pop()
        chain = _chain(node)
        if chain is not None:
            yield node, chain, where
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, [], where
            continue
        inner = f"{where}.<locals>.{_name(node)}" if isinstance(node, _Scope) else where
        for field, value in ast.iter_fields(node):
            if field in ("annotation", "returns", "type_params") or (
                field == "value" and type(node).__name__ == "TypeAlias"
            ):
                continue
            # decorators, defaults and class bases run where the definition is
            outer = field in ("decorator_list", "args", "bases", "keywords")
            stack.extend((child, where if outer else inner) for child in _nodes(value))


def _nodes(value: object) -> list[ast.AST]:
    items: list[object] = (
        list(cast(list[object], value)) if isinstance(value, list) else [value]
    )
    return [item for item in items if isinstance(item, ast.AST)]


def _chain(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return [node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        inner = _chain(node.value)
        return None if inner is None else [*inner, node.attr]
    return None


# a name not loaded, judged by the name alone
class _Unloaded(str):
    pass


def _resolve(
    chain: list[str],
    module_globals: dict[str, object],
    cells: dict[str, object],
    imported: dict[str, tuple[str, str | None]],
) -> _Result | None:
    first = chain[0]
    read_from: str | None = None
    if first in imported:
        module, attribute = imported[first]
        if attribute is None:
            value = _loaded(module)
        else:
            value, read_from = _attribute(sys.modules.get(module), module, attribute)
    elif cells.get(first, _MISSING) is not _MISSING:
        value = cells[first]
    elif first in module_globals:
        value = module_globals[first]
    else:
        # a builtin, or a name defined later
        return None
    reference, reason = first, _judge(value, read_from)
    for name in chain[1:]:
        if _is_module(value):
            module_name = _module_name(value)
        elif type(value) is _Unloaded:
            module_name = str(value)
        else:
            break
        reference = f"{reference}.{name}"
        value, read_from = _attribute(value, module_name, name)
        reason = _judge(value, read_from)
    return (
        None
        if reason is None
        else _Result(reference, _describe(value, read_from), reason)
    )


def _judge_import(node: ast.Import | ast.ImportFrom) -> Iterator[_Result]:
    for alias in node.names:
        read_from: str | None = None
        if isinstance(node, ast.Import):
            reference = f"import {alias.name}"
            value = _loaded(alias.name)
        elif node.level == 0 and node.module:
            reference = f"from {node.module} import {alias.name}"
            module = sys.modules.get(node.module)
            value, read_from = _attribute(module, node.module, alias.name)
        else:
            return
        reason = _judge(value, read_from)
        if reason is not None:
            yield _Result(reference, _describe(value, read_from), reason)


def _loaded(module: str) -> object:
    value = sys.modules.get(module)
    return value if _is_module(value) else _Unloaded(module)


# the value and the module it was read from, or a placeholder if not loaded
def _attribute(
    module: object, module_name: str, name: str
) -> tuple[object, str | None]:
    namespace = _namespace(module) if _is_module(module) else {}
    value = namespace.get(name, _MISSING)
    if value is _MISSING:
        return _Unloaded(f"{module_name}.{name}"), None
    return value, module_name


# a module's own namespace, read without running a lazy module's loader
def _namespace(module: types.ModuleType) -> dict[str, object]:
    return cast(dict[str, object], object.__getattribute__(module, "__dict__"))


def _module_name(module: types.ModuleType) -> str:
    name = _namespace(module).get("__name__")
    return name if isinstance(name, str) else ""


def _judge(value: object, read_from: str | None = None) -> str | None:
    if type(value) is _Unloaded:
        return _judge_name(value)
    if _allowed(value):
        return None
    refused = _refused_objects()
    hit = refused.get(id(value)) or refused.get(id(_owner(value)))
    if hit is not None:
        return hit.reason
    if _is_module(value):
        return _judge_name(_module_name(value))
    if _is_code(value):
        return _judge_name(_module_of(value))
    # data is judged by the module it is read from, or else by its type
    if read_from is not None:
        return _judge_name(read_from)
    return None if _allowed(type(value)) else _judge_name(_module_of(type(value)))


class _Refused(NamedTuple):
    name: str
    reason: str


# the loaded objects among the refused names, by id
def _refused_objects() -> dict[int, _Refused]:
    found: dict[int, _Refused] = {}
    for name, reason in _REFUSED.items():
        module_name, _, attribute = name.rpartition(".")
        module = sys.modules.get(module_name)
        if _is_module(module):
            value = _namespace(module).get(attribute, _MISSING)
            if value is not _MISSING and not _is_module(value):
                found[id(value)] = _Refused(name, reason)
    return found


def _owner(value: object) -> object:
    return value.__self__ if type(value) is types.MethodType else None


_ROUTINES = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.ClassMethodDescriptorType,
)


# type(), not isinstance(), so a lazy proxy's `__class__` is never consulted
def _is_class(value: object) -> bool:
    return issubclass(type(value), type)


def _is_module(value: object) -> TypeGuard[types.ModuleType]:
    return issubclass(type(value), types.ModuleType)


def _is_code(value: object) -> bool:
    return _is_class(value) or issubclass(type(value), _ROUTINES)


def _allowed(value: object) -> bool:
    return id(value) in _core_values()


@functools.cache
def _core_values() -> frozenset[int]:
    return frozenset(
        id(value) for name, value in vars(inspect_core).items() if name[:1] != "_"
    )


def _module_of(value: object) -> str:
    module = getattr(value, "__module__", None)
    if isinstance(module, str):
        return module
    owner = getattr(value, "__objclass__", getattr(value, "__self__", None))
    if _is_module(owner):
        return _module_name(owner)
    if _is_class(owner):
        return str(getattr(owner, "__module__", ""))
    return ""


def _judge_name(name: str) -> str | None:
    for refused, reason in _REFUSED.items():
        if name == refused or name.startswith(f"{refused}."):
            return reason
    top = name.partition(".")[0]
    if top == "inspect_ai":
        return _INSPECT
    if top == _CORE:
        return None if name == _CORE else _CORE_PRIVATE
    if top in sys.stdlib_module_names or top in _WASI_BUILT:
        return None
    if top and top in _compiled_packages(tuple(sys.path)):
        return f"and `{top}` contains compiled extension modules, which a portable function cannot load"
    return None


# packages_distributions() reads only top_level.txt on 3.10, and parsing every
# RECORD through `files` is slow, so the top-level names come from the RECORD
# text; keyed by sys.path
@functools.cache
def _compiled_packages(path: tuple[str, ...]) -> frozenset[str]:
    compiled: dict[str, list[bool]] = {}
    for distribution in importlib.metadata.distributions():
        try:
            record = distribution.read_text("RECORD")
        except (OSError, UnicodeDecodeError):
            continue
        files = (
            [line.split(",")[0] for line in record.splitlines()]
            if record is not None
            else [str(file) for file in distribution.files or []]
        )
        has = any(file.endswith(_EXTENSIONS) for file in files)
        for top in {file.split("/")[0].partition(".")[0] for file in files}:
            compiled.setdefault(top, []).append(has)
    # a namespace package shared by several distributions counts only if all are
    return frozenset(top for top, flags in compiled.items() if all(flags))


def _describe(value: object, read_from: str | None = None) -> str:
    if type(value) is _Unloaded:
        return f"`{value}` (not loaded)"
    if _is_module(value):
        return f"module `{_module_name(value)}`"
    refused = _refused_objects()
    owner = refused.get(id(_owner(value)))
    if owner is not None and id(value) not in refused:
        return f"method `{getattr(value, '__name__', '?')}` of `{owner.name}`"
    if _is_code(value):
        label = getattr(value, "__qualname__", getattr(value, "__name__", "?"))
        kind = "class" if _is_class(value) else "function"
        return f"{kind} `{label}` from `{_module_of(value)}`"
    if read_from is not None:
        return f"a value in `{read_from}`"
    return f"a `{type(value).__qualname__}` from `{_module_of(type(value))}`"
