# The portable check runs when a portable factory is called, before it runs.
# It parses the factory's file once (cached on the file's text), locates the
# factory and the functions it returns, and resolves each non-local name and
# attribute chain they load to the real object, judging it by its defining
# module against the allowlists. Functions and classes of the author's own
# modules are followed the same way. Every violation is collected into one
# PortabilityError, and the check never imports anything.
from __future__ import annotations

import ast
import builtins
import calendar
import codecs
import contextlib
import functools
import importlib.util
import inspect
import io
import linecache
import logging
import posixpath
import sys
import time
import types
import typing
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import inspect_ai.core
import typing_extensions
from inspect_ai._util.registry import is_registry_object, registry_info
from inspect_ai.scorer import Reference
from inspect_ai.util import StoreModel

# Allowlists and per-object refusals

# mirrors ALLOWED_THIRD_PARTY in inspect_ai's tests/core/test_core_imports.py,
# plus anyio, which sentinel depends on and protocols use for concurrency
_ALLOWED_PACKAGES = frozenset(
    {
        "pydantic",
        "pydantic_core",
        "typing_extensions",
        "shortuuid",
        "anyio",
        "inspect_sentinel",
    }
)
_CORE = "inspect_ai.core"

# Standard-library modules that only compute and that CPython provides on
# WASI, matched exactly against the module an object is defined in (so the C
# modules behind public ones are listed too).
_PORTABLE_STDLIB = frozenset(
    """
    builtins abc _abc annotationlib array asyncio asyncio.base_futures
    asyncio.coroutines asyncio.events asyncio.exceptions asyncio.futures
    asyncio.locks asyncio.queues asyncio.taskgroups asyncio.tasks
    asyncio.timeouts _asyncio base64 binascii bisect _bisect calendar cmath
    codecs _codecs collections collections.abc _collections colorsys contextlib
    contextvars _contextvars copy csv _csv dataclasses datetime decimal _decimal
    _pydecimal difflib enum errno fnmatch fractions functools _functools
    genericpath graphlib hashlib _hashlib _blake2 _md5 _sha1 _sha2 _sha3 heapq
    _heapq hmac html html.entities html.parser ipaddress _markupbase itertools
    json json.decoder json.encoder _json keyword logging math ntpath numbers
    operator _operator posixpath pprint random _random re reprlib secrets shlex
    statistics string string.templatelib struct _struct textwrap time tomllib
    tomllib._parser tomllib._re tomllib._types traceback types typing
    unicodedata urllib.parse uuid warnings _warnings _py_warnings weakref
    _weakref _weakrefset zlib
    """.split()
)

# objects in allowed modules whose effects a portable function cannot have
_REFUSED = {
    id(value): value
    for module, name in (
        (logging, "FileHandler"),
        (logging, "basicConfig"),
        (contextlib, "chdir"),
        (uuid, "uuid1"),
        (uuid, "getnode"),
        (time, "tzset"),
        (calendar, "main"),
        (codecs, "open"),
    )
    for value in (cast(object, getattr(module, name, None)),)
    if value is not None
}


def _path_functions() -> frozenset[int]:
    # some of `os.path` is implemented in `posix`
    posix = sys.modules.get("posix")
    found = [*vars(posixpath).values()]
    if posix is not None:
        found += [v for n, v in vars(posix).items() if n.startswith("_path_")]
    return frozenset(id(v) for v in found if inspect.isroutine(v))


_PATH_FUNCTIONS = _path_functions()

_BANNED_BUILTINS = (
    "open",
    "input",
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
)

# allowed by identity, as classes and instances: inspect_sentinel's API takes
# StoreModel and Reference until they move into inspect_ai.core
_ALLOWED_TYPES: tuple[type, ...] = (StoreModel, Reference, io.StringIO, io.BytesIO)

# data read off a module that is otherwise not allowed: (module, attribute)
_ALLOWED_DATA = frozenset({("io", "SEEK_SET"), ("io", "SEEK_CUR"), ("io", "SEEK_END")})


@functools.cache
def _core_values() -> frozenset[int]:
    # by identity, so re-exports elsewhere in inspect_ai pass, aliases included;
    # plain constants are left out, since equal ones can be the same object
    return frozenset(
        id(value)
        for name, value in vars(inspect_ai.core).items()
        if not name.startswith("_")
        and not isinstance(value, (str, bytes, int, float, tuple, type(None)))
    )


def _types(*candidates: object) -> tuple[type, ...]:
    return tuple(c for c in candidates if isinstance(c, type))


_TYPE_ALIASES = _types(
    getattr(typing, "TypeAliasType", None),
    getattr(typing_extensions, "TypeAliasType", None),
)
_GENERIC_ALIASES = _types(
    getattr(typing, "_BaseGenericAlias", None), types.GenericAlias, types.UnionType
)

# Error and entry point

_FIXES = "A portable {kind} affects the outside world only through `context`: call a model with `context.host.generate()`, ask a person with `context.host.ask_human()`, and keep state with `context.store_as()`. Otherwise move the reference out of the {kind}'s code, or, if the {kind} only runs in an eval, declare it with `@{kind}(portable=False)`."

_NO_SOURCE = "no source"
_UNPARSEABLE = "unparseable source"
_UNREADABLE = "unreadable source"
_CHANGED = "source changed since import"
_TOO_DEEP = "too deeply nested"


class PortabilityError(TypeError):
    """A portable monitor or protocol references something a portable function cannot use.

    Raised when the factory of a `@monitor` or `@protocol` declared with `portable=True` (the default) is called, before it runs. The message lists every disallowed reference in the factory's code and in the functions and classes of the author's own modules that it reaches, and any of that code that could not be checked: the file and line, the function, the name and what it resolved to. A `TypeError`, as the factory's other configuration-time checks of the functions it returns are.
    """


def check_portable(factory: Callable[..., object], kind: str, name: str) -> None:
    own = str(getattr(factory, "__module__", "")).partition(".")[0]
    violations = _Checker(own).check(factory)
    if violations:
        lines = "\n".join(f"- {v.describe()}" for v in violations)
        raise PortabilityError(
            f"{kind} {name} is portable, but its code references what a portable {kind} cannot use:\n{lines}\n{_FIXES.format(kind=kind)}"
        )


class _Violation(NamedTuple):
    file: str
    line: int
    function: str
    reference: str
    resolved: str
    reason: str
    via: tuple[str, ...]

    def describe(self) -> str:
        via = f" (reached from {' -> '.join(self.via)})" if self.via else ""
        return f"{self.file}:{self.line} in {self.function}: `{self.reference}` is {self.resolved}, {self.reason}{via}"


# Judging objects


class _Kind(Enum):
    ALLOWED = "allowed"
    AUTHOR = "author"
    DISALLOWED = "disallowed"


class _Verdict(NamedTuple):
    kind: _Kind
    reason: str = ""


_ALLOWED = _Verdict(_Kind.ALLOWED)


def _classify(module: str, own: str) -> _Verdict:
    if not module:
        return _Verdict(_Kind.DISALLOWED, "and its module is unknown")
    top = module.partition(".")[0]
    if top == "inspect_ai":
        if module == _CORE or module.startswith(f"{_CORE}."):
            return _ALLOWED
        return _Verdict(
            _Kind.DISALLOWED, f"and inspect_ai is portable only through `{_CORE}`"
        )
    if top in _ALLOWED_PACKAGES or module in _PORTABLE_STDLIB:
        return _ALLOWED
    if top in sys.stdlib_module_names and top != own:
        return _Verdict(
            _Kind.DISALLOWED,
            f"and `{module}` is not on the portable standard-library list",
        )
    if top == own or _is_author(module):
        return _Verdict(_Kind.AUTHOR)
    return _Verdict(_Kind.DISALLOWED, f"and `{top}` is not a portable dependency")


def _is_author(module: str) -> bool:
    file = getattr(sys.modules.get(module), "__file__", None)
    if isinstance(file, str):
        locations = [file]
    else:
        try:
            spec = importlib.util.find_spec(module.partition(".")[0])
        except RecursionError:
            raise
        except Exception:
            return False
        if spec is None:
            return False
        # a namespace package has no origin, only its directories
        locations = (
            [spec.origin]
            if spec.origin
            else list(spec.submodule_search_locations or [])
        )
    return any(
        "site-packages" not in Path(location).parts
        and "dist-packages" not in Path(location).parts
        for location in locations
    )


def _module_of(value: object) -> str:
    if inspect.ismodule(value):
        return value.__name__
    if inspect.isclass(value) or inspect.isroutine(value):
        module = getattr(value, "__module__", None)
        if isinstance(module, str):
            return module
        # a method of a builtin type, or a builtin bound to an instance
        owner = getattr(value, "__objclass__", getattr(value, "__self__", None))
        if owner is None or inspect.isroutine(owner):
            return ""
        return _module_of(owner)
    return type(value).__module__


def _describe(value: object) -> str:
    if inspect.ismodule(value):
        return f"module `{value.__name__}`"
    module = _module_of(value)
    if inspect.isclass(value) or inspect.isroutine(value):
        label = getattr(value, "__qualname__", getattr(value, "__name__", "?"))
        kind = "class" if inspect.isclass(value) else "function"
        return f"{kind} `{label}` from `{module}`"
    return f"a `{type(value).__qualname__}` from `{module}`"


def _verdict_of(value: object, own: str) -> _Verdict:
    if id(value) in _REFUSED:
        return _Verdict(
            _Kind.DISALLOWED, "which has effects a portable function cannot have"
        )
    if id(value) in _PATH_FUNCTIONS or id(value) in _core_values():
        return _ALLOWED
    if type(value) in _ALLOWED_TYPES or any(value is t for t in _ALLOWED_TYPES):
        return _ALLOWED
    return _classify(_module_of(value), own)


def _is_module(value: object) -> bool:
    # type(), not isinstance(), so a proxy's raising `__class__` is not consulted
    return issubclass(type(value), types.ModuleType)


@functools.cache
def _public_sentinel() -> frozenset[int]:
    package = sys.modules["inspect_sentinel"]
    names = getattr(package, "__all__", ())
    return frozenset({id(package), *(id(getattr(package, n, None)) for n in names)})


def _is_sentinel_factory(value: object) -> bool:
    return (
        inspect.isfunction(value)
        and is_registry_object(value)
        and registry_info(value).type in ("monitor", "protocol")
    )


# A type alias is judged by the types it names, and a `functools.partial` by
# the function it calls and the functions and classes it passes.
def _leaves(value: object) -> list[object]:
    leaves: list[object] = []
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        parts = _parts(item)
        if parts is None:
            leaves.append(item)
        else:
            pending.extend(reversed(parts))
    return leaves


def _parts(value: object) -> list[object] | None:
    kind = type(value)
    if id(value) in _core_values():
        return None
    if issubclass(kind, functools.partial):
        partial = cast("functools.partial[object]", value)
        passed = [*partial.args, *partial.keywords.values()]
        return [
            partial.func,
            *(
                p
                for p in passed
                if inspect.isroutine(p)
                or inspect.isclass(p)
                or issubclass(type(p), functools.partial)
            ),
        ]
    if issubclass(kind, _TYPE_ALIASES):
        return [cast(Any, value).__value__]
    if not issubclass(kind, _GENERIC_ALIASES):
        return None
    parts: list[object] = []
    origin = typing.get_origin(value)
    if origin is not None:
        parts.append(origin)
    for arg in typing.get_args(value):
        # `Callable[[int], str]` holds its parameters in a list
        if isinstance(arg, (list, tuple)):
            parts.extend(cast(Sequence[object], arg))
        else:
            parts.append(arg)
    return parts


# Attribute resolution


# resolves through modules and classes only; an instance is judged by its type
def _attribute(value: object, name: str) -> object:
    if _is_module(value):
        namespace = vars(value)
        if name in namespace:
            return cast(object, namespace[name])
        submodule = sys.modules.get(f"{cast(types.ModuleType, value).__name__}.{name}")
        if submodule is None:
            raise AttributeError(name)
        return submodule
    if not inspect.isclass(value):
        raise AttributeError(name)
    found = inspect.getattr_static(value, name)
    if isinstance(found, (staticmethod, classmethod)):
        return cast(object, cast(Any, found).__func__)
    return found


def _unwrap(value: object) -> object:
    try:
        return inspect.unwrap(cast(Any, value))
    except RecursionError:
        raise
    except Exception:
        return value


# A wrapper the author wrote is checked as it is, and its closure reaches what
# it wraps; a library's wrapper, such as `functools.cache` or
# `contextlib.contextmanager`, is unwrapped to the author's function.
def _callable(value: object, own: str) -> object:
    for _ in range(100):
        if inspect.isclass(value) or inspect.ismethod(value):
            break
        if inspect.isfunction(value):
            defined_in = value.__globals__.get("__name__")
            if isinstance(defined_in, str) and (
                _classify(defined_in, own).kind is _Kind.AUTHOR
            ):
                break
        try:
            wrapped = cast(object, getattr(value, "__wrapped__", None))
        except RecursionError:
            raise
        except Exception:
            break
        if wrapped is None:
            break
        value = wrapped
    return value


# Locating source: the file cache

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_FUNCTION_SCOPES = (*_FUNCTIONS, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (*_FUNCTION_SCOPES, ast.ClassDef, *_COMPREHENSIONS)
_TYPE_ALIAS: tuple[type[ast.AST], ...] = tuple(
    t for t in (getattr(ast, "TypeAlias", None),) if t is not None
)


class _Source(NamedTuple):
    text: str
    # first line (decorators included) -> functions and lambdas starting there
    definitions: dict[int, list[ast.AST]]
    classes: dict[str, list[ast.ClassDef]]
    # module-level `from M import N`: bound name -> (level, M, N)
    from_imports: dict[str, tuple[int, str, str]]
    # id of a located node -> the identifiers written in it
    names: dict[int, frozenset[str]]


_SOURCES_KEPT = 32
_sources: OrderedDict[str, _Source] = OrderedDict()


def _source(filename: str, module_globals: dict[str, object]) -> _Source | str:
    # shared by every exec() of a string, so never this code's source
    if filename == "<string>":
        return _NO_SOURCE
    try:
        linecache.checkcache(filename)
        lines = linecache.getlines(filename, module_globals)
    except RecursionError:
        raise
    except Exception:
        return _NO_SOURCE
    if not lines:
        return _NO_SOURCE
    text = "".join(lines)
    cached = _sources.get(filename)
    if cached is not None and cached.text == text:
        _sources.move_to_end(filename)
        return cached
    try:
        tree = ast.parse(text)
    except RecursionError:
        raise
    except Exception:
        return _UNPARSEABLE
    source = _index(text, tree)
    _sources[filename] = source
    _sources.move_to_end(filename)
    while len(_sources) > _SOURCES_KEPT:
        _sources.popitem(last=False)
    return source


def _index(text: str, tree: ast.Module) -> _Source:
    source = _Source(text, {}, {}, {}, {})
    for node in ast.walk(tree):
        if isinstance(node, _FUNCTION_SCOPES):
            source.definitions.setdefault(_first_line(node), []).append(node)
        elif isinstance(node, ast.ClassDef):
            source.classes.setdefault(node.name, []).append(node)
    for node in _walk_scope(tree.body):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                source.from_imports.setdefault(
                    alias.asname or alias.name,
                    (node.level, node.module or "", alias.name),
                )
    return source


def _first_line(node: ast.AST) -> int:
    if isinstance(node, _DEFINITIONS):
        return min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return cast(int, getattr(node, "lineno", 0))


# A located node whose running code loads a name the source does not have is
# not the code that runs: the file changed after it was imported.
def _stale(
    source: _Source, nodes: Sequence[ast.AST], codes: Iterable[types.CodeType]
) -> bool:
    written: set[str] = set()
    for node in nodes:
        names = source.names.get(id(node))
        if names is None:
            names = source.names[id(node)] = _written_names(node)
        written |= names
    pending = list(codes)
    while pending:
        code = pending.pop()
        pending.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
        for name in code.co_names:
            if (
                name in written
                or not name.isidentifier()
                or (name.startswith("__") and name.endswith("__"))
            ):
                continue
            # a private name in a class is compiled mangled, as `_Class__name`
            if any(
                name.startswith("__", i) and name[i:] in written
                for i in range(1, len(name))
            ):
                continue
            return True
    return False


def _written_names(node: ast.AST) -> frozenset[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.alias):
            names.update(sub.name.split("."))
            if sub.asname:
                names.add(sub.asname)
        elif isinstance(sub, ast.ImportFrom) and sub.module:
            names.update(sub.module.split("."))
        elif isinstance(sub, (ast.Global, ast.Nonlocal)):
            names.update(sub.names)
        elif isinstance(sub, ast.MatchClass):
            names.update(sub.kwd_attrs)
        elif isinstance(sub, ast.MatchMapping) and sub.rest:
            names.add(sub.rest)
        elif isinstance(sub, ast.arg):
            names.add(sub.arg)
        elif isinstance(sub, ast.keyword) and sub.arg:
            names.add(sub.arg)
        else:
            # definitions, exception handlers, match captures, type parameters
            name = getattr(sub, "name", None)
            if isinstance(name, str):
                names.add(name)
    return frozenset(names)


# Locating source: functions and lambdas


class _Target(NamedTuple):
    value: object
    file: str
    roots: tuple[ast.AST, ...]
    source: _Source
    namespace: dict[str, object]
    closure: dict[str, object]
    via: tuple[str, ...]


def _locate(
    value: object, via: tuple[str, ...], hint: _Target | None
) -> _Target | str | None:
    try:
        if inspect.isfunction(value):
            return _function_target(value, via)
        if inspect.isclass(value):
            return _class_target(value, via, hint)
    except RecursionError:
        raise
    except Exception:
        return _UNREADABLE
    return None


def _function_target(func: types.FunctionType, via: tuple[str, ...]) -> _Target | str:
    code = func.__code__
    source = _source(code.co_filename, func.__globals__)
    if isinstance(source, str):
        return source
    nodes = _function_nodes(source, code)
    if not nodes or _stale(source, nodes, [code]):
        return _CHANGED
    return _Target(
        func,
        code.co_filename,
        tuple(nodes),
        source,
        func.__globals__,
        _cells(func),
        via,
    )


# A lambda is matched by its line alone, so every lambda starting on that line
# is checked.
def _function_nodes(source: _Source, code: types.CodeType) -> list[ast.AST]:
    candidates = source.definitions.get(code.co_firstlineno, [])
    if code.co_name == "<lambda>":
        return [node for node in candidates if isinstance(node, ast.Lambda)]
    functions: list[ast.AST] = [
        node
        for node in candidates
        if isinstance(node, _FUNCTIONS) and node.name == code.co_name
    ]
    return functions[:1]


def _cells(func: types.FunctionType) -> dict[str, object]:
    closure: dict[str, object] = {}
    for name, cell in zip(
        func.__code__.co_freevars, func.__closure__ or (), strict=True
    ):
        try:
            closure[name] = cell.cell_contents
        except ValueError:
            continue
    return closure


# Locating source: classes


def _class_target(
    cls: type, via: tuple[str, ...], hint: _Target | None
) -> _Target | str | None:
    if _generic_origin(cls) is not None:
        return None
    module = sys.modules.get(cls.__module__)
    module_file = getattr(module, "__file__", None)
    methods = _methods(cls)
    methods.sort(key=lambda f: f.__code__.co_filename != module_file)
    namespace: dict[str, object] | None = vars(module) if module is not None else None
    certain = bool(methods) or isinstance(module_file, str)
    if methods:
        file = methods[0].__code__.co_filename
        namespace = namespace or methods[0].__globals__
    elif isinstance(module_file, str):
        file = module_file
    elif hint is not None and hint.namespace.get("__name__") == cls.__module__:
        # defined beside the code that references it, outside sys.modules
        file, namespace = hint.file, hint.namespace
    else:
        return None
    source = _source(file, namespace or {})
    if isinstance(source, str):
        return source
    codes = [m.__code__ for m in methods if m.__code__.co_filename == file]
    nodes = _class_nodes(source, cls, codes)
    if not nodes:
        defined = bool(methods) or isinstance(vars(cls).get("__firstlineno__"), int)
        return _CHANGED if certain and defined else None
    if _stale(source, nodes, codes):
        return _CHANGED
    closure: dict[str, object] = {}
    for method in methods:
        closure.update(_cells(method))
    return _Target(cls, file, tuple(nodes), source, namespace or {}, closure, via)


# The class statement named like the class: on 3.13+ the one starting at
# `__firstlineno__`; otherwise the innermost one holding its methods, or,
# without methods, every one of that name.
def _class_nodes(
    source: _Source, cls: type, codes: Sequence[types.CodeType]
) -> list[ast.ClassDef]:
    candidates = source.classes.get(cls.__name__, [])
    first = vars(cls).get("__firstlineno__")
    if isinstance(first, int):
        return [node for node in candidates if _first_line(node) == first][:1]
    if not codes:
        return candidates
    lines = [code.co_firstlineno for code in codes]
    holding = [
        node
        for node in candidates
        if all(_first_line(node) <= line <= (node.end_lineno or 0) for line in lines)
    ]
    return [max(holding, key=_first_line)] if holding else []


def _generic_origin(cls: type) -> object:
    # a parametrized pydantic generic, such as `Box[int]`, has no statement
    metadata = vars(cls).get("__pydantic_generic_metadata__")
    if isinstance(metadata, dict):
        return cast(dict[str, object], metadata).get("origin")
    return None


def _methods(cls: type) -> list[types.FunctionType]:
    found: list[types.FunctionType] = []
    for value in vars(cls).values():
        if isinstance(value, (staticmethod, classmethod)):
            value = cast(object, cast(Any, value).__func__)
        functions = (
            (value.fget, value.fset, value.fdel)
            if isinstance(value, property)
            else (value,)
        )
        for function in functions:
            function = _unwrap(function)
            if (
                inspect.isfunction(function)
                and function.__qualname__.startswith(f"{cls.__qualname__}.")
                and function.__module__ == cls.__module__
                and function.__code__.co_filename != "<string>"
            ):
                found.append(function)
    return found


def _is_pseudo_file(filename: str) -> bool:
    # `<stdin>`, `<string>`, `<console>`: code typed or exec()'d, not a file
    return filename.startswith("<") and filename.endswith(">")


# Checker: follows the author's own functions and classes


class _Checker:
    def __init__(self, own: str) -> None:
        self.own = own
        self.seen: dict[int, object] = {}
        self.violations: dict[tuple[str, int, str], _Violation] = {}
        self._pending: list[_Target] = []

    def check(self, factory: Callable[..., object]) -> tuple[_Violation, ...]:
        value = _callable(factory, self.own)
        self.seen[id(value)] = value
        try:
            located = _locate(value, (), None)
        except RecursionError:
            located = _TOO_DEEP
        if isinstance(located, str):
            code = getattr(value, "__code__", None)
            file = str(getattr(code, "co_filename", "<unknown>"))
            if located == _NO_SOURCE and _is_pseudo_file(file):
                return ()
            line = int(getattr(code, "co_firstlineno", 1))
            self._unchecked(value, file, line, located, ())
        elif located is not None:
            self._pending.append(located)
        while self._pending:
            target = self._pending.pop(0)
            try:
                _Visitor(self, target).run()
            except RecursionError:
                line = getattr(target.roots[0], "lineno", 1)
                self._unchecked(target.value, target.file, line, _TOO_DEEP, target.via)
        return tuple(self.violations.values())

    def _unchecked(
        self, value: object, file: str, line: int, reason: str, via: tuple[str, ...]
    ) -> None:
        name = str(getattr(value, "__qualname__", "?"))
        self.add(
            _Violation(
                file,
                line,
                name,
                name,
                _describe(value),
                f"and could not be checked ({reason})",
                via,
            )
        )

    def follow(
        self,
        value: object,
        via: tuple[str, ...],
        unchecked: Callable[[str], None],
        hint: _Target,
    ) -> None:
        if _is_module(value):
            return
        value = _callable(value, self.own)
        if inspect.ismethod(value):
            value = _callable(value.__func__, self.own)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            value = type(value)
        if id(value) in self.seen:
            return
        self.seen[id(value)] = value
        try:
            located = _locate(value, via, hint)
        except RecursionError:
            located = _TOO_DEEP
        if located is None and inspect.isclass(value):
            # a class made without a statement: check the author code it is made of
            for part in self._made_of(value):
                self.follow(part, via, unchecked, hint)
        elif isinstance(located, str):
            unchecked(f"and could not be checked ({located})")
        elif located is not None:
            self._pending.append(located)

    def _made_of(self, cls: type) -> list[object]:
        parts = [_generic_origin(cls), *cls.__bases__]
        return [
            part
            for part in parts
            if inspect.isclass(part)
            and _verdict_of(part, self.own).kind is _Kind.AUTHOR
        ]

    def add(self, violation: _Violation) -> None:
        key = (violation.file, violation.line, violation.reference)
        self.violations.setdefault(key, violation)


# Visitor: judges every reference in the located code


class _Scope(NamedTuple):
    bound: frozenset[str]
    declared_global: frozenset[str]
    is_class: bool
    qualname: str
    imports: dict[str, object]


class _Global(NamedTuple):
    value: object
    # the module it was read from by `from M import N`, and N
    owner: object
    name: str


class _Visitor:
    def __init__(self, checker: _Checker, target: _Target) -> None:
        self._checker = checker
        self._target = target
        module = target.namespace.get("__name__")
        self._in_sentinel = isinstance(module, str) and (
            module == "inspect_sentinel" or module.startswith("inspect_sentinel.")
        )
        self._qualname = str(
            getattr(
                target.value, "__qualname__", getattr(target.value, "__name__", "?")
            )
        )
        # iterative, so deep expressions do not reach the recursion limit
        self._pending: list[tuple[ast.AST, list[_Scope], bool]] = []

    def run(self) -> None:
        self._push(self._target.roots, [], root=True)
        while self._pending:
            node, scopes, root = self._pending.pop()
            if root or isinstance(node, _SCOPES):
                self._visit_scope(node, scopes, root)
            else:
                self._visit(node, scopes)

    # pushed in reverse, so they are visited in source order
    def _push(
        self, nodes: Iterable[ast.AST], scopes: list[_Scope], root: bool = False
    ) -> None:
        self._pending.extend((node, scopes, root) for node in reversed(list(nodes)))

    def _visit(self, node: ast.AST, scopes: list[_Scope]) -> None:
        if isinstance(node, ast.Attribute):
            chain = _chain(node)
            if chain is None:
                self._push(ast.iter_child_nodes(node), scopes)
            else:
                self._reference(node, chain[0], chain[1:], scopes)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._reference(node, node.id, [], scopes)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._import(node, scopes, report=True)
        elif _is_type_checking(node):
            self._push(cast(ast.If, node).orelse, scopes)
        elif isinstance(node, ast.AnnAssign):
            parts: list[ast.AST] = []
            if not scopes or scopes[-1].is_class:
                parts.append(node.annotation)
            if node.value is not None:
                parts.append(node.value)
            self._push(parts, scopes)
        elif isinstance(node, _TYPE_ALIAS):
            self._push(
                [cast(ast.AST, cast(Any, node).value)],
                self._type_scope(node, scopes, self._where(scopes)),
            )
        else:
            self._push(ast.iter_child_nodes(node), scopes)

    def _visit_scope(self, node: ast.AST, scopes: list[_Scope], root: bool) -> None:
        outer = self._where(scopes)
        qualname = outer
        groups: list[tuple[list[ast.AST], list[_Scope]]] = []
        if isinstance(node, _DEFINITIONS):
            groups.append((list(node.decorator_list), scopes))
            if not root:
                qualname = _nested(outer, node.name, scopes)
        elif isinstance(node, ast.Lambda) and not root:
            qualname = _nested(outer, "<lambda>", scopes)
        if isinstance(node, _FUNCTION_SCOPES):
            groups.append((list(_defaults(node.args)), scopes))
        scopes = self._type_scope(node, scopes, qualname)
        if isinstance(node, ast.ClassDef):
            groups.append(([*node.bases, *node.keywords], scopes))
        bound, declared_global = _bound(node)
        scope = _Scope(
            bound, declared_global, isinstance(node, ast.ClassDef), qualname, {}
        )
        inner = [*scopes, scope]
        if isinstance(node, _DEFINITIONS):
            for statement in _imports(node.body):
                self._import(statement, inner, report=False)
            groups.append((list(node.body), inner))
        elif isinstance(node, ast.Lambda):
            groups.append(([node.body], inner))
        elif isinstance(node, _COMPREHENSIONS):
            # the first iterable is evaluated in the enclosing scope
            first = node.generators[0]
            groups.append(([first.iter], scopes))
            rest = [c for c in ast.iter_child_nodes(node) if c is not first]
            groups.append(([*rest, first.target, *first.ifs], inner))
        for nodes, group_scopes in reversed(groups):
            self._push(nodes, group_scopes)

    def _type_scope(
        self, node: ast.AST, scopes: list[_Scope], qualname: str
    ) -> list[_Scope]:
        params = cast(list[ast.AST], getattr(node, "type_params", None) or [])
        if not params:
            return scopes
        names = frozenset(str(cast(Any, p).name) for p in params)
        return [*scopes, _Scope(names, frozenset(), False, qualname, {})]

    def _lookup(
        self, name: str, scopes: Sequence[_Scope]
    ) -> tuple[Literal["imported", "local", "global"], object]:
        for index, scope in enumerate(reversed(scopes)):
            if name in scope.declared_global:
                break
            # a class body's names are not visible inside its methods
            if scope.is_class and index > 0:
                continue
            if name in scope.imports:
                return "imported", scope.imports[name]
            if name in scope.bound:
                return "local", None
        return "global", None

    def _resolve(self, name: str) -> _Global | None:
        if name in self._target.closure:
            return _Global(self._target.closure[name], None, name)
        namespace = self._target.namespace
        if name in namespace:
            value = namespace[name]
            owner, attribute = self._from_module(name, value)
            return _Global(value, owner, attribute)
        if hasattr(builtins, name):
            return _Global(getattr(builtins, name), None, name)
        return None

    # the module a global came from by `from M import N`, so data read off a
    # disallowed module is refused as it is through `M.N`
    def _from_module(self, name: str, value: object) -> tuple[object, str]:
        imported = self._target.source.from_imports.get(name)
        if imported is None:
            return None, name
        level, module, attribute = imported
        loaded = sys.modules.get(self._absolute(level, module) or "")
        if loaded is None:
            return None, name
        try:
            found = cast(object, getattr(loaded, attribute, None))
        except RecursionError:
            raise
        except Exception:
            # a module `__getattr__` that raises: judged by the module's name
            return loaded, attribute
        return (loaded, attribute) if found is value else (None, name)

    def _absolute(self, level: int, module: str) -> str | None:
        if not level:
            return module
        package = self._target.namespace.get("__package__")
        try:
            return importlib.util.resolve_name(
                "." * level + module, package if isinstance(package, str) else None
            )
        except (ImportError, ValueError):
            return None

    def _reference(
        self, node: ast.AST, root: str, attributes: list[str], scopes: list[_Scope]
    ) -> None:
        binding, value = self._lookup(root, scopes)
        if binding == "local":
            return
        owner: object = None
        name = root
        if binding == "global":
            found = self._resolve(root)
            if found is None:
                return
            value, owner, name = found
            if root in _BANNED_BUILTINS and value is getattr(builtins, root):
                self._violate(
                    node,
                    root,
                    "a builtin",
                    "which a portable function cannot call",
                    scopes,
                )
                return
        values = [_Global(value, owner, name)]
        for attribute in attributes:
            try:
                values.append(_Global(_attribute(value, attribute), value, attribute))
            except RecursionError:
                raise
            except Exception:
                break
            value = values[-1].value
        for index, (value, owner, name) in enumerate(values):
            # a module is a namespace on the way to what the chain reaches
            if _is_module(value) and index + 1 < len(values):
                continue
            reference = ".".join([root, *attributes[:index]])
            if not self._judge(node, reference, value, scopes, owner, name):
                return

    def _judge(
        self,
        node: ast.AST,
        reference: str,
        value: object,
        scopes: list[_Scope],
        owner: object,
        name: str,
    ) -> bool:
        try:
            leaves = _leaves(value)
        except RecursionError:
            raise
        except Exception:
            leaves = [value]
        for leaf in leaves:
            if leaf is not value:
                owner = None
            if not self._judge_one(node, reference, leaf, scopes, owner, name):
                return False
        return True

    def _judge_one(
        self,
        node: ast.AST,
        reference: str,
        value: object,
        scopes: list[_Scope],
        owner: object,
        name: str,
    ) -> bool:
        try:
            verdict, described = self._verdict(value, owner, name)
            covered = verdict.kind is _Kind.AUTHOR and (
                # checked on its own when configured
                _is_sentinel_factory(value)
                # checked with its class
                or (inspect.isclass(owner) and id(owner) in self._checker.seen)
            )
        except RecursionError:
            raise
        except Exception:
            verdict = _Verdict(_Kind.DISALLOWED, "which could not be inspected")
            described = f"a `{type(value).__qualname__}`"
            covered = False
        if verdict.kind is _Kind.DISALLOWED:
            self._violate(node, reference, described, verdict.reason, scopes)
            return False
        if verdict.kind is _Kind.AUTHOR and not covered:
            self._checker.follow(
                value,
                (*self._target.via, self._where(scopes)),
                lambda reason: self._violate(
                    node, reference, described, reason, scopes
                ),
                self._target,
            )
        return True

    def _verdict(self, value: object, owner: object, name: str) -> tuple[_Verdict, str]:
        own = self._checker.own
        described = _describe(value)
        if id(value) not in _REFUSED and inspect.isclass(owner):
            # an attribute reached through an allowed class is judged by the class
            if self._verdict(owner, None, "")[0].kind is _Kind.ALLOWED:
                return _ALLOWED, described
        verdict = _verdict_of(value, own)
        if _is_module(owner) and id(value) not in _core_values():
            module = cast(types.ModuleType, owner).__name__
            stricter = _classify(module, own)
            if (module, name) in _ALLOWED_DATA:
                return _ALLOWED, described
            if not _module_of(value):
                # a builtin without a module, such as `codecs.strict_errors`
                verdict = stricter
            elif stricter.kind is _Kind.DISALLOWED:
                if verdict.kind is _Kind.DISALLOWED:
                    verdict = _Verdict(_Kind.DISALLOWED, stricter.reason)
                elif not (
                    _is_module(value)
                    or inspect.isclass(value)
                    or inspect.isroutine(value)
                ):
                    # data read off a disallowed module counts as that module
                    verdict = stricter
                    described = f"a `{type(value).__qualname__}` in module `{module}`"
        if (
            verdict.kind is _Kind.ALLOWED
            and not self._in_sentinel
            and _module_of(value).partition(".")[0] == "inspect_sentinel"
            and id(value) not in _public_sentinel()
            and id(type(value)) not in _public_sentinel()
        ):
            verdict = _Verdict(
                _Kind.DISALLOWED,
                "and only `inspect_sentinel`'s public API is portable",
            )
        return verdict, described

    def _import(
        self, node: ast.Import | ast.ImportFrom, scopes: list[_Scope], report: bool
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                loaded = sys.modules.get(alias.name)
                if loaded is None:
                    if report:
                        self._unloaded(node, f"import {alias.name}", alias.name, scopes)
                    continue
                # bound without judging; references through it are judged
                bound = alias.asname or alias.name.partition(".")[0]
                value = loaded if alias.asname else sys.modules.get(bound)
                if value is not None:
                    scopes[-1].imports[bound] = value
            return
        module = self._absolute(node.level, node.module or "")
        if module is None:
            return
        loaded = sys.modules.get(module)
        for alias in node.names:
            statement = f"from {module} import {alias.name}"
            if loaded is None:
                if report:
                    self._unloaded(node, statement, module, scopes)
                continue
            try:
                value = _attribute(loaded, alias.name)
            except RecursionError:
                raise
            except Exception:
                submodule = f"{module}.{alias.name}"
                if report and _exists(submodule):
                    self._unloaded(node, statement, submodule, scopes)
                continue
            scopes[-1].imports[alias.asname or alias.name] = value
            if report:
                self._judge(node, statement, value, scopes, loaded, alias.name)

    # judged by name only: the check never imports anything
    def _unloaded(
        self, node: ast.AST, statement: str, module: str, scopes: list[_Scope]
    ) -> None:
        verdict = _classify(module, self._checker.own)
        if verdict.kind is _Kind.DISALLOWED:
            self._violate(node, statement, f"module `{module}`", verdict.reason, scopes)

    def _violate(
        self,
        node: ast.AST,
        reference: str,
        resolved: str,
        reason: str,
        scopes: list[_Scope],
    ) -> None:
        self._checker.add(
            _Violation(
                file=self._target.file,
                line=getattr(node, "lineno", 1),
                function=self._where(scopes),
                reference=reference,
                resolved=resolved,
                reason=reason,
                via=self._target.via,
            )
        )

    def _where(self, scopes: Sequence[_Scope]) -> str:
        return scopes[-1].qualname if scopes else self._qualname


# Scope helpers


def _exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except RecursionError:
        raise
    except Exception:
        return False


def _nested(qualname: str, name: str, scopes: Sequence[_Scope]) -> str:
    if scopes and scopes[-1].is_class:
        return f"{qualname}.{name}"
    if scopes:
        return f"{qualname}.<locals>.{name}"
    return name


def _defaults(arguments: ast.arguments) -> list[ast.expr]:
    return [*arguments.defaults, *(d for d in arguments.kw_defaults if d is not None)]


def _chain(node: ast.AST) -> list[str] | None:
    attributes: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        attributes.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return [current.id, *reversed(attributes)]


def _is_type_checking(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If) and (_chain(node.test) or [""])[-1] == "TYPE_CHECKING"
    )


def _bound(scope: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    names: set[str] = set()
    declared_global: set[str] = set()
    body: list[ast.AST] = []
    if isinstance(scope, _FUNCTION_SCOPES):
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ):
            if argument is not None:
                names.add(argument.arg)
        body = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
    elif isinstance(scope, ast.ClassDef):
        body = list(scope.body)
    elif isinstance(scope, _COMPREHENSIONS):
        body = [generator.target for generator in scope.generators]
    for node in _walk_scope(body):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, _DEFINITIONS):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.partition(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, _COMPREHENSIONS):
            # an assignment expression in a comprehension binds in this scope
            names.update(
                sub.target.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.NamedExpr)
            )
    return frozenset(names - declared_global), frozenset(declared_global)


def _walk_scope(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    pending = list(nodes)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, _SCOPES):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _imports(body: Iterable[ast.AST]) -> Iterator[ast.Import | ast.ImportFrom]:
    pending = list(reversed(list(body)))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif _is_type_checking(node):
            pending.extend(reversed(cast(ast.If, node).orelse))
        elif not isinstance(node, (ast.expr, *_SCOPES)):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))
