# The portable check runs when a portable factory is called. It parses the
# factory's file, finds the factory, and resolves the names its code references
# one level (global, closure cell or builtin, then attributes through modules),
# raising for each that resolves to something a sandboxed guest cannot do. It
# never imports anything.
from __future__ import annotations

import ast
import functools
import importlib.metadata
import inspect
import linecache
import os
import sys
import types
from collections.abc import Callable, Iterator
from typing import NamedTuple, TypeGuard, cast

import inspect_ai.core
from inspect_ai.util import StoreModel

_CORE = "inspect_ai.core"

_ENVIRONMENT = "and a portable function has no environment variables; take configuration as factory parameters"
_PROCESSES = "and a portable function cannot start processes or threads"
_NETWORK = "and a portable function has no network access; call a model with `context.host.generate()`"
_INSPECT = f"and inspect_ai is portable only through `{_CORE}`"

_OS_ENVIRONMENT = ("environ", "environb", "getenv", "getenvb", "putenv", "unsetenv")
_OS_PROCESSES = ("system", "popen", "kill", "killpg", "startfile")
_OS_PROCESS_PREFIXES = ("exec", "fork", "spawn", "posix_spawn")

_OS_REFUSED: dict[int, str] = {
    id(getattr(os, name)): (_ENVIRONMENT if name in _OS_ENVIRONMENT else _PROCESSES)
    for name in dir(os)
    if name in _OS_ENVIRONMENT
    or name in _OS_PROCESSES
    or name.startswith(_OS_PROCESS_PREFIXES)
}

# a module is refused with the modules it contains; the pool executors are
# named too, since concurrent.futures loads them on first use
_PROCESS_MODULES = """
subprocess _posixsubprocess threading _thread multiprocessing _multiprocessing
concurrent.futures.thread concurrent.futures.process asyncio.subprocess
asyncio.threads concurrent.futures.ThreadPoolExecutor
concurrent.futures.ProcessPoolExecutor
""".split()
_NETWORK_MODULES = """
socket _socket ssl _ssl socketserver http.client http.server urllib.request
ftplib smtplib imaplib poplib xmlrpc asyncio.streams httpx httpcore requests
urllib3
""".split()
_REFUSED_MODULES = {
    **dict.fromkeys(_PROCESS_MODULES, _PROCESSES),
    **dict.fromkeys(_NETWORK_MODULES, _NETWORK),
}

# built for WASI by sentinel
_WASI_BUILT = frozenset({"pydantic_core"})
_EXTENSIONS = (".so", ".pyd", ".dylib")

_FIXES = "A portable {kind} affects the outside world only through `context`: call a model with `context.host.generate()`, ask a person with `context.host.ask_human()`, and keep state with `context.store_as()`. Take configuration as factory parameters. Otherwise move the reference out of the {kind}'s code, or, if the {kind} only runs in an eval, declare it with `@{kind}(portable=False)`."


class PortabilityError(TypeError):
    """A portable monitor or protocol references something a portable function cannot use.

    Raised when the factory of a `@monitor` or `@protocol` declared with `portable=True` (the default) is called, before it runs. The message lists each reference in the factory's code that a portable function cannot use: the file and line, the function, the name, what it resolved to and why. A factory whose source exists but cannot be parsed or located is reported as not checked. A `TypeError`, as the factory's other configuration-time checks of the functions it returns are.
    """


class _Violation(NamedTuple):
    line: int
    function: str
    reference: str
    resolved: str
    reason: str


def check_portable(factory: Callable[..., object], kind: str, name: str) -> None:
    func = inspect.unwrap(factory)
    code = getattr(func, "__code__", None)
    if not isinstance(code, types.CodeType) or not isinstance(func, types.FunctionType):
        return
    file = code.co_filename
    # every exec() of a string shares `<string>`, so its lines are not this code's
    lines = [] if file == "<string>" else linecache.getlines(file, func.__globals__)
    if not lines:
        return
    node = _locate("".join(lines), code)
    qualname = func.__qualname__
    violations = (
        _check(node, func)
        if not isinstance(node, str)
        else [
            _Violation(
                code.co_firstlineno,
                qualname,
                qualname,
                f"{kind} factory `{qualname}`",
                f"and could not be checked ({node})",
            )
        ]
    )
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
    local = _assigned(factory)
    cells = {
        name: _contents(cell)
        for name, cell in zip(
            func.__code__.co_freevars, func.__closure__ or (), strict=True
        )
    }
    found: dict[tuple[int, str], _Violation] = {}
    for node, chain, where in _references(factory, func.__qualname__):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result = _judge_import(node)
        elif chain[0] in local:
            continue
        else:
            result = _resolve(chain, func.__globals__, cells)
        if result is not None:
            reference, resolved, reason = result
            line = getattr(node, "lineno", 0)
            found.setdefault(
                (line, reference), _Violation(line, where, reference, resolved, reason)
            )
    return sorted(found.values())


def _contents(cell: types.CellType) -> object:
    try:
        return cell.cell_contents
    except ValueError:
        return _MISSING


def _assigned(factory: ast.AST) -> frozenset[str]:
    names: set[str] = set()
    for node in ast.walk(factory):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (*_Function, ast.ClassDef)) and node is not factory:
            names.add(_name(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).partition(".")[0] for a in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        else:
            for field in ("name", "rest"):
                # except, match and type-parameter targets
                value = getattr(node, field, None)
                if isinstance(value, str) and not isinstance(node, ast.alias):
                    names.add(value)
    return frozenset(names)


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
        inner = where
        if isinstance(node, _Function):
            inner = f"{where}.<locals>.{_name(node)}"
        elif isinstance(node, ast.ClassDef):
            inner = f"{where}.<locals>.{node.name}"
        for field, value in ast.iter_fields(node):
            if field in ("annotation", "returns", "type_params") or (
                field == "value" and type(node).__name__ == "TypeAlias"
            ):
                continue
            # decorators and defaults run where the definition is
            scope = where if field in ("decorator_list", "args") else inner
            stack.extend((child, scope) for child in _nodes(value))


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


# a module attribute named but not loaded, judged as a module name
class _Unloaded(str):
    pass


_Result = tuple[str, str, str]


def _resolve(
    chain: list[str], module_globals: dict[str, object], cells: dict[str, object]
) -> _Result | None:
    first = chain[0]
    if cells.get(first, _MISSING) is not _MISSING:
        value = cells[first]
    elif first in module_globals:
        value = module_globals[first]
    else:
        # a builtin, or a name defined later
        return None
    reference, reason = first, _judge(value)
    for name in chain[1:]:
        if not _is_module(value):
            break
        reference = f"{reference}.{name}"
        value, reason = _attribute(value, value.__name__, name)
    return None if reason is None else (reference, _describe(value), reason)


def _judge_import(node: ast.Import | ast.ImportFrom) -> _Result | None:
    for alias in node.names:
        if isinstance(node, ast.Import):
            reference = f"import {alias.name}"
            value, reason = _Unloaded(alias.name), _judge_module(alias.name)
        elif node.level == 0 and node.module:
            reference = f"from {node.module} import {alias.name}"
            module = sys.modules.get(node.module)
            value, reason = _attribute(module, node.module, alias.name)
        else:
            return None
        if reason is not None:
            return reference, _describe(value), reason
    return None


def _attribute(
    module: object, module_name: str, name: str
) -> tuple[object, str | None]:
    namespace = vars(module) if _is_module(module) else {}
    value = namespace.get(name, _MISSING)
    if value is _MISSING:
        unloaded = _Unloaded(f"{module_name}.{name}")
        return unloaded, _judge_module(unloaded)
    return value, _judge(value, module_name)


def _judge(value: object, read_from: str | None = None) -> str | None:
    if _allowed(value):
        return None
    owner = value.__self__ if type(value) is types.MethodType else value
    if id(value) in _OS_REFUSED or id(owner) in _OS_REFUSED:
        return _OS_REFUSED.get(id(value)) or _OS_REFUSED[id(owner)]
    if _is_module(value):
        return _judge_module(value.__name__)
    if _is_code(value):
        return _judge_module(_module_of(value))
    # data is judged by the module it is read from, or else by its type
    if read_from is not None:
        return _judge_module(read_from)
    return None if _allowed(type(value)) else _judge_module(_module_of(type(value)))


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
    return value is StoreModel or id(value) in _core_values()


@functools.cache
def _core_values() -> frozenset[int]:
    return frozenset(
        id(value) for name, value in vars(inspect_ai.core).items() if name[:1] != "_"
    )


def _module_of(value: object) -> str:
    module = getattr(value, "__module__", None)
    if isinstance(module, str):
        return module
    owner = getattr(value, "__objclass__", getattr(value, "__self__", None))
    if _is_module(owner):
        return owner.__name__
    if _is_class(owner):
        return str(getattr(owner, "__module__", ""))
    return ""


def _judge_module(module: str) -> str | None:
    for refused, reason in _REFUSED_MODULES.items():
        if module == refused or module.startswith(f"{refused}."):
            return reason
    top = module.partition(".")[0]
    if top == "inspect_ai":
        return None if module == _CORE or module.startswith(f"{_CORE}.") else _INSPECT
    if top in sys.stdlib_module_names or top in _WASI_BUILT:
        return None
    if top and top in _compiled_packages(tuple(sys.path)):
        return f"and `{top}` contains compiled extension modules, which a portable function cannot load"
    return None


# packages_distributions() reads only top_level.txt on 3.10, and parsing every
# RECORD through `files` is slow, so the top-level names come from the RECORD text
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


def _describe(value: object) -> str:
    if type(value) is _Unloaded:
        return f"module `{value}`"
    if _is_module(value):
        return f"module `{value.__name__}`"
    if _is_code(value):
        label = getattr(value, "__qualname__", getattr(value, "__name__", "?"))
        kind = "class" if _is_class(value) else "function"
        return f"{kind} `{label}` from `{_module_of(value)}`"
    return f"a `{type(value).__qualname__}` from `{_module_of(type(value))}`"
