from __future__ import annotations

import ast
import builtins
import genericpath
import importlib
import importlib.util
import inspect
import linecache
import ntpath
import posixpath
import sys
import textwrap
import tokenize
import types
import weakref
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, cast

from inspect_ai.scorer import Reference
from inspect_ai.util import StoreModel

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

# Standard-library modules a portable function may use: pure computation that
# CPython provides on WASI. Matched exactly against the module an object is
# defined in, so the C modules behind public ones are listed too. Left out
# with effects or a filesystem dependency: os (except os.path), sys, io,
# pathlib, pickle, subprocess, socket, threading, zoneinfo (reads the tz
# database), gzip (gzip.open; use zlib), logging.handlers, and asyncio's
# streams, subprocesses and threads. `time` is allowed though `time.sleep`
# blocks the runtime (use `anyio.sleep`).
_PORTABLE_STDLIB = frozenset(
    """
    builtins abc _abc annotationlib array asyncio asyncio.base_futures
    asyncio.coroutines asyncio.events asyncio.exceptions asyncio.futures
    asyncio.locks asyncio.queues asyncio.taskgroups asyncio.tasks
    asyncio.timeouts _asyncio base64 binascii bisect _bisect calendar cmath
    collections collections.abc _collections contextlib contextvars _contextvars copy
    dataclasses datetime decimal _decimal _pydecimal difflib enum fnmatch
    fractions functools _functools genericpath hashlib _hashlib _blake2 _md5
    _sha1 _sha2 _sha3 heapq _heapq hmac html html.entities html.parser _markupbase
    itertools json json.decoder json.encoder _json keyword logging math ntpath
    numbers operator _operator posixpath pprint random _random re reprlib secrets shlex
    statistics string string.templatelib struct _struct textwrap time types typing unicodedata
    urllib.parse uuid warnings _warnings _py_warnings weakref _weakref
    _weakrefset zlib
    """.split()
)

# some of `os.path` is implemented in `posix`
_PATH_FUNCTIONS = {
    id(value): value
    for module in (posixpath, ntpath, genericpath)
    for value in vars(module).values()
    if inspect.isroutine(value)
}

_BANNED_BUILTINS = (
    "open",
    "input",
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
)

# inspect_sentinel's API takes these (`Context.store_as()` a StoreModel
# subclass, a report's `references`); allowed by identity until they move into
# inspect_ai.core
_ALLOWED_TYPES: tuple[type, ...] = (StoreModel, Reference)

_FIXES = "A portable {kind} affects the outside world only through `context`: call a model with `context.host.generate()`, ask a person with `context.host.ask_human()`, and keep state with `context.store_as()`. Otherwise move the reference out of the {kind}'s code, or, if the {kind} only runs in an eval, declare it with `@{kind}(portable=False)`."


class PortabilityError(TypeError):
    """A portable monitor or protocol references something a portable function cannot use.

    Raised when the factory of a `@monitor` or `@protocol` declared with `portable=True` (the default) is called, before it runs. The message lists every disallowed reference in the factory's code and in the functions and classes of the author's own modules that it reaches: the file and line, the function, the name and what it resolved to. A `TypeError`, as the factory's other configuration-time checks of the functions it returns are.
    """


def check_portable(factory: Callable[..., object], kind: str, name: str) -> None:
    violations = _checked.get(factory)
    if violations is None:
        own = str(getattr(factory, "__module__", "")).partition(".")[0]
        violations = _Checker(own).check(factory)
        _checked[factory] = violations
    if violations:
        lines = "\n".join(f"- {v.describe()}" for v in violations)
        raise PortabilityError(
            f"{kind} {name} is portable, but its code references what a portable {kind} cannot use:\n{lines}\n{_FIXES.format(kind=kind)}"
        )


_checked: weakref.WeakKeyDictionary[Callable[..., object], tuple[_Violation, ...]] = (
    weakref.WeakKeyDictionary()
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


class _Kind(Enum):
    ALLOWED = "allowed"
    AUTHOR = "author"
    DISALLOWED = "disallowed"


class _Verdict(NamedTuple):
    kind: _Kind
    reason: str = ""


_ALLOWED = _Verdict(_Kind.ALLOWED)


def _classify(module: str, own: str = "") -> _Verdict:
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
            f"and `{module}` is not on the portable standard-library list (use `portable=False`, or ask for it to be added)",
        )
    if top == own or _is_author(module):
        return _Verdict(_Kind.AUTHOR)
    return _Verdict(_Kind.DISALLOWED, f"and `{top}` is not a portable dependency")


def _is_author(module: str) -> bool:
    found = sys.modules.get(module)
    file = getattr(found, "__file__", None)
    if not isinstance(file, str):
        try:
            spec = importlib.util.find_spec(module.partition(".")[0])
        except (ImportError, ValueError):
            return False
        file = spec.origin if spec is not None else None
    if not file:
        return False
    parts = Path(file).parts
    return "site-packages" not in parts and "dist-packages" not in parts


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


def _judge(value: object, own: str = "") -> _Verdict:
    if id(value) in _PATH_FUNCTIONS:
        return _ALLOWED
    if type(value) in _ALLOWED_TYPES or any(value is t for t in _ALLOWED_TYPES):
        return _ALLOWED
    return _classify(_module_of(value), own)


def _is_module(value: object) -> bool:
    # type(), not isinstance(), so a proxy's raising `__class__` is not consulted
    return issubclass(type(value), types.ModuleType)


def _public_sentinel() -> dict[int, object]:
    package = sys.modules["inspect_sentinel"]
    public: dict[int, object] = {id(package): package}
    for name in getattr(package, "__all__", ()):
        value = getattr(package, name, None)
        public[id(value)] = value
    return public


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


_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (*_FUNCTION_SCOPES, ast.ClassDef, *_COMPREHENSIONS)


class _Scope(NamedTuple):
    node: ast.AST
    bound: frozenset[str]
    qualname: str


class _Target(NamedTuple):
    value: object
    file: str
    first_line: int
    tree: ast.Module
    namespace: dict[str, object]
    closure: dict[str, object]
    via: tuple[str, ...]


class _Checker:
    def __init__(self, own: str) -> None:
        self.own = own
        self._seen: dict[int, object] = {}
        self._pending: list[_Target] = []
        self._violations: list[_Violation] = []

    def check(self, factory: Callable[..., object]) -> tuple[_Violation, ...]:
        self._enqueue(factory, ())
        while self._pending:
            _Visitor(self, self._pending.pop(0)).run()
        return tuple(self._violations)

    def report(self, violation: _Violation) -> None:
        self._violations.append(violation)

    def follow(self, value: object, via: tuple[str, ...]) -> None:
        if _is_module(value):
            return
        value = _unwrap(value)
        if inspect.ismethod(value):
            value = value.__func__
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            value = type(value)
        self._enqueue(value, via)

    def _enqueue(self, value: object, via: tuple[str, ...]) -> None:
        value = _unwrap(value)
        if id(value) in self._seen:
            return
        self._seen[id(value)] = value
        target = _target(value, via)
        if target is not None:
            self._pending.append(target)


def _unwrap(value: object) -> object:
    try:
        return inspect.unwrap(cast(Any, value))
    except Exception:
        return value


def _target(value: object, via: tuple[str, ...]) -> _Target | None:
    try:
        lines, first_line = inspect.getsourcelines(cast(Any, value))
        file = inspect.getsourcefile(cast(Any, value)) or "<unknown>"
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, ValueError, SyntaxError, tokenize.TokenError):
        return _class_target(value, via) if inspect.isclass(value) else None
    if inspect.isfunction(value):
        namespace: dict[str, object] = value.__globals__
        closure: dict[str, object] = {}
        for name, cell in zip(
            value.__code__.co_freevars, value.__closure__ or (), strict=True
        ):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                continue
    else:
        module = sys.modules.get(getattr(value, "__module__", ""))
        namespace = vars(module) if module is not None else {}
        closure = {}
    return _Target(value, file, max(first_line, 1), tree, namespace, closure, via)


# a class whose module is not in `sys.modules` (a task file loaded by
# `inspect eval`, a notebook cell): find it through one of its methods
def _class_target(cls: type, via: tuple[str, ...]) -> _Target | None:
    methods = [
        cast(Any, f).__func__ if isinstance(f, (staticmethod, classmethod)) else f
        for f in vars(cls).values()
    ]
    method = next((f for f in methods if inspect.isfunction(f)), None)
    if method is None:
        return None
    code = method.__code__
    try:
        tree = ast.parse("".join(linecache.getlines(code.co_filename)))
    except (SyntaxError, ValueError):
        return None
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == cls.__name__
        and node.lineno <= code.co_firstlineno <= (node.end_lineno or 0)
    ]
    if not found:
        return None
    node = max(found, key=lambda n: n.lineno)
    body = ast.Module(body=[node], type_ignores=[])
    return _Target(cls, code.co_filename, 1, body, method.__globals__, {}, via)


class _Visitor:
    def __init__(self, checker: _Checker, target: _Target) -> None:
        self._checker = checker
        self._target = target
        self._imports: dict[tuple[int, str], object] = {}
        module = target.namespace.get("__name__")
        self._in_sentinel = isinstance(module, str) and (
            module == "inspect_sentinel" or module.startswith("inspect_sentinel.")
        )
        self._qualname = str(
            getattr(
                target.value, "__qualname__", getattr(target.value, "__name__", "?")
            )
        )

    def run(self) -> None:
        body = self._target.tree.body
        root = body[0] if len(body) == 1 else None
        if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._visit_scope(root, [], self._qualname, root=True)
        else:
            self._visit_all(body, [])

    def _visit_all(self, nodes: Iterable[ast.AST], scopes: list[_Scope]) -> None:
        for node in nodes:
            self._visit(node, scopes)

    def _visit(self, node: ast.AST, scopes: list[_Scope]) -> None:
        if isinstance(node, _SCOPES):
            qualname = scopes[-1].qualname if scopes else self._qualname
            self._visit_scope(node, scopes, qualname, root=False)
        elif isinstance(node, ast.Attribute):
            chain = _chain(node)
            if chain is None:
                self._visit_all(ast.iter_child_nodes(node), scopes)
            else:
                self._reference(node, chain[0], chain[1:], scopes)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._reference(node, node.id, [], scopes)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._import(node, scopes)
        elif (
            isinstance(node, ast.If)
            and (_chain(node.test) or [""])[-1] == "TYPE_CHECKING"
        ):
            self._visit_all(node.orelse, scopes)
        elif isinstance(node, ast.AnnAssign):
            if not (scopes and isinstance(scopes[-1].node, _FUNCTION_SCOPES)):
                self._visit(node.annotation, scopes)
            if node.value is not None:
                self._visit(node.value, scopes)
        else:
            self._visit_all(ast.iter_child_nodes(node), scopes)

    def _visit_scope(
        self, node: ast.AST, scopes: list[_Scope], qualname: str, root: bool
    ) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._visit_all(node.decorator_list, scopes)
            if not root:
                qualname = _nested(qualname, node.name, scopes)
        if isinstance(node, _FUNCTION_SCOPES):
            self._visit_all(_defaults(node.args), scopes)
        if isinstance(node, ast.Lambda) and not root:
            qualname = _nested(qualname, "<lambda>", scopes)
        if isinstance(node, ast.ClassDef):
            self._visit_all([*node.bases, *node.keywords], scopes)
        inner = [*scopes, _Scope(node, _bound(node), qualname)]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._visit_all(node.body, inner)
        elif isinstance(node, ast.Lambda):
            self._visit(node.body, inner)
        else:
            self._visit_all(ast.iter_child_nodes(node), inner)

    def _is_local(self, name: str, scopes: Sequence[_Scope]) -> bool:
        for index, scope in enumerate(reversed(scopes)):
            # a class body's names are not visible inside its methods
            if isinstance(scope.node, ast.ClassDef) and index > 0:
                continue
            if name in scope.bound:
                return True
        return False

    def _resolve(self, name: str) -> tuple[bool, object]:
        if name in self._target.closure:
            return True, self._target.closure[name]
        if name in self._target.namespace:
            return True, self._target.namespace[name]
        if hasattr(builtins, name):
            return True, getattr(builtins, name)
        return False, None

    def _imported(self, name: str, scopes: Sequence[_Scope]) -> tuple[bool, object]:
        for scope in reversed(scopes):
            key = (id(scope.node), name)
            if key in self._imports:
                return True, self._imports[key]
        return False, None

    def _reference(
        self, node: ast.AST, root: str, attributes: list[str], scopes: list[_Scope]
    ) -> None:
        imported, value = self._imported(root, scopes)
        if not imported and self._is_local(root, scopes):
            return
        if not imported:
            found, value = self._resolve(root)
            if not found:
                return
            if root in _BANNED_BUILTINS and value is getattr(builtins, root):
                self._violate(
                    node,
                    root,
                    "a builtin",
                    "which a portable function cannot call",
                    scopes,
                )
                return
        values = [value]
        for attribute in attributes:
            try:
                value = _attribute(value, attribute)
            except Exception:
                break
            values.append(value)
        for index, value in enumerate(values):
            # a module is a namespace on the way to what the chain reaches
            if _is_module(value) and index + 1 < len(values):
                continue
            owner = values[index - 1] if index > 0 else None
            reference = ".".join([root, *attributes[:index]])
            if not self._judge(node, reference, value, scopes, owner):
                return

    def _judge(
        self,
        node: ast.AST,
        reference: str,
        value: object,
        scopes: list[_Scope],
        owner: object = None,
    ) -> bool:
        try:
            verdict, described = self._verdict(value, owner)
        except Exception:
            verdict = _Verdict(_Kind.DISALLOWED, "which could not be inspected")
            described = f"a `{type(value).__qualname__}`"
        if verdict.kind is _Kind.DISALLOWED:
            self._violate(node, reference, described, verdict.reason, scopes)
            return False
        if verdict.kind is _Kind.AUTHOR:
            self._checker.follow(value, (*self._target.via, self._where(scopes)))
        return True

    def _verdict(self, value: object, owner: object) -> tuple[_Verdict, str]:
        own = self._checker.own
        verdict = _judge(value, own)
        described = _describe(value)
        if not _module_of(value) and inspect.isclass(owner):
            verdict = _judge(owner, own)
        # data read off a disallowed module counts as that module (`os.environ`)
        if (
            verdict.kind is not _Kind.DISALLOWED
            and _is_module(owner)
            and not (_is_module(value) or inspect.isclass(value))
            and not inspect.isroutine(value)
        ):
            name = cast(types.ModuleType, owner).__name__
            stricter = _classify(name, own)
            if stricter.kind is _Kind.DISALLOWED:
                verdict = stricter
                described = f"a `{type(value).__qualname__}` in module `{name}`"
        if (
            verdict.kind is _Kind.ALLOWED
            and not self._in_sentinel
            and _module_of(value).partition(".")[0] == "inspect_sentinel"
        ):
            public = _public_sentinel()
            if not (
                id(value) in public
                or id(type(value)) in public
                or (inspect.isclass(owner) and id(owner) in public)
            ):
                verdict = _Verdict(
                    _Kind.DISALLOWED,
                    "and only `inspect_sentinel`'s public API is portable",
                )
        return verdict, described

    def _import(self, node: ast.Import | ast.ImportFrom, scopes: list[_Scope]) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.partition(".")[0]
                value = self._import_module(node, alias.name, None, scopes)
                if value is not None and alias.asname is None:
                    value = sys.modules.get(bound)
                self._bind(bound, value, scopes)
            return
        module = node.module or ""
        if node.level:
            package = self._target.namespace.get("__package__")
            try:
                module = importlib.util.resolve_name(
                    "." * node.level + module,
                    package if isinstance(package, str) else None,
                )
            except (ImportError, ValueError):
                return
        for alias in node.names:
            value = self._import_module(node, module, alias.name, scopes)
            self._bind(alias.asname or alias.name, value, scopes)

    def _bind(self, name: str, value: object, scopes: Sequence[_Scope]) -> None:
        if value is not None and scopes:
            self._imports[(id(scopes[-1].node), name)] = value

    def _import_module(
        self, node: ast.AST, module: str, name: str | None, scopes: list[_Scope]
    ) -> object:
        statement = (
            f"import {module}" if name is None else f"from {module} import {name}"
        )
        loaded = sys.modules.get(module)
        if loaded is None:
            # judged by name only: the check never imports anything
            verdict = _classify(module, self._checker.own)
            if verdict.kind is _Kind.DISALLOWED:
                self._violate(
                    node, statement, f"module `{module}`", verdict.reason, scopes
                )
            return None
        value: object = loaded
        if name is not None:
            try:
                value = _attribute(loaded, name)
            except AttributeError:
                pass
        self._judge(node, statement, value, scopes, loaded)
        return value

    def _violate(
        self,
        node: ast.AST,
        reference: str,
        resolved: str,
        reason: str,
        scopes: list[_Scope],
    ) -> None:
        line = self._target.first_line + getattr(node, "lineno", 1) - 1
        self._checker.report(
            _Violation(
                file=self._target.file,
                line=line,
                function=self._where(scopes),
                reference=reference,
                resolved=resolved,
                reason=reason,
                via=self._target.via,
            )
        )

    def _where(self, scopes: Sequence[_Scope]) -> str:
        return scopes[-1].qualname if scopes else self._qualname


def _nested(qualname: str, name: str, scopes: Sequence[_Scope]) -> str:
    if scopes and isinstance(scopes[-1].node, ast.ClassDef):
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


def _bound(scope: ast.AST) -> frozenset[str]:
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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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
    return frozenset(names - declared_global)


def _walk_scope(nodes: Iterable[ast.AST]) -> Iterable[ast.AST]:
    pending = list(nodes)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, _SCOPES):
            continue
        pending.extend(ast.iter_child_nodes(node))
