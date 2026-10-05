from __future__ import annotations

import ast
import builtins
import functools
import importlib.util
import inspect
import sys
import textwrap
import weakref
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, cast

from inspect_ai.scorer import Reference
from inspect_ai.util import StoreModel

# mirrors ALLOWED_THIRD_PARTY in inspect_ai's tests/core/test_core_imports.py
_CORE_DEPENDENCIES = frozenset(
    {"pydantic", "pydantic_core", "typing_extensions", "shortuuid"}
)
_CORE = "inspect_ai.core"
_ALLOWED_PACKAGES = _CORE_DEPENDENCIES | {"inspect_sentinel"}

_EFFECTFUL = frozenset(
    {
        "socket",
        "ssl",
        "http.client",
        "urllib.request",
        "subprocess",
        "multiprocessing",
        "threading",
        "signal",
        "ctypes",
        "sqlite3",
        "shutil",
        "importlib",
        "os",
        "sys",
        # the C modules the ones above are built on
        "posix",
        "nt",
        "_socket",
        "_ssl",
        "_posixsubprocess",
        "_winapi",
        "_multiprocessing",
        "_thread",
        "_signal",
        "_ctypes",
        "_sqlite3",
        "_imp",
    }
)

_BANNED_BUILTINS = (
    "open",
    "input",
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
)

# inspect_sentinel's API takes these and they are not in inspect_ai.core:
# `Context.store_as()` a StoreModel subclass, and a report's `references`
_ALLOWED_TYPES: tuple[type, ...] = (StoreModel, Reference)

_FIXES = "A portable {kind} affects the outside world only through `context`: call a model with `context.host.generate()`, ask a person with `context.host.ask_human()`, and keep state with `context.store_as()`. Otherwise move the reference out of the {kind}'s code, or, if the {kind} only runs in an eval, declare it with `@{kind}(portable=False)`."


class PortabilityError(TypeError):
    """A portable monitor or protocol references something a portable function cannot use.

    Raised when the factory of a `@monitor` or `@protocol` declared with `portable=True` (the default) is called, before it runs. The message lists every disallowed reference in the factory's code and in the functions and classes of the author's own modules that it reaches: the file and line, the function, the name and what it resolved to. A `TypeError`, as the factory's other configuration-time checks of the functions it returns are.
    """


def check_portable(factory: Callable[..., object], kind: str, name: str) -> None:
    violations = _checked.get(factory)
    if violations is None:
        violations = _Checker().check(factory)
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


def _classify(module: str) -> _Verdict:
    top = module.partition(".")[0]
    for effectful in _EFFECTFUL:
        if module == effectful or module.startswith(f"{effectful}."):
            return _Verdict(
                _Kind.DISALLOWED,
                f"and `{effectful}` is a standard-library module with effects",
            )
    if top == "inspect_ai":
        if module == _CORE or module.startswith(f"{_CORE}."):
            return _ALLOWED
        return _Verdict(
            _Kind.DISALLOWED, f"and inspect_ai is portable only through `{_CORE}`"
        )
    if top in _ALLOWED_PACKAGES or top in sys.stdlib_module_names:
        return _ALLOWED
    if _is_author(module):
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
        owner = getattr(value, "__objclass__", None)
        if inspect.isclass(owner):
            return owner.__module__
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


def _judge(value: object) -> _Verdict:
    if type(value) in _ALLOWED_TYPES or any(value is t for t in _ALLOWED_TYPES):
        return _ALLOWED
    if any(value is getattr(builtins, name) for name in _BANNED_BUILTINS):
        return _Verdict(_Kind.DISALLOWED, "a builtin a portable function cannot call")
    if value is builtins:
        return _Verdict(_Kind.DISALLOWED, "which reaches the builtins by name")
    if isinstance(value, functools.partial):
        return _judge(value.func)
    return _classify(_module_of(value))


def _unwrap_attribute(value: object) -> object:
    if isinstance(value, (staticmethod, classmethod)):
        return cast(object, cast(Any, value).__func__)
    if isinstance(value, property) and value.fget is not None:
        return value.fget
    return value


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
    def __init__(self) -> None:
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
        if inspect.ismodule(value):
            return
        if inspect.ismethod(value):
            value = value.__func__
        if isinstance(value, functools.partial):
            value = value.func
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            value = type(value)
        self._enqueue(value, via)

    def _enqueue(self, value: object, via: tuple[str, ...]) -> None:
        if inspect.isfunction(value):
            value = inspect.unwrap(value)
        if id(value) in self._seen:
            return
        self._seen[id(value)] = value
        target = _target(value, via)
        if target is not None:
            self._pending.append(target)


def _target(value: object, via: tuple[str, ...]) -> _Target | None:
    try:
        lines, first_line = inspect.getsourcelines(cast(Any, value))
        file = inspect.getsourcefile(cast(Any, value)) or "<unknown>"
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError):
        return None
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


class _Visitor:
    def __init__(self, checker: _Checker, target: _Target) -> None:
        self._checker = checker
        self._target = target
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
                self._visit(node.value, scopes)
            else:
                self._reference(node, chain[0], chain[1:], scopes)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._reference(node, node.id, [], scopes)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._import(node, scopes)
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
            # the factory's own decorators include the `@monitor` or `@protocol`
            if not (root and not self._target.via):
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

    def _reference(
        self, node: ast.AST, root: str, attributes: list[str], scopes: list[_Scope]
    ) -> None:
        if self._is_local(root, scopes):
            return
        if root == "__builtins__":
            self._violate(
                node, root, "the builtins", "which reaches the builtins by name", scopes
            )
            return
        found, value = self._resolve(root)
        if not found:
            return
        values = [value]
        for attribute in attributes:
            try:
                value = _unwrap_attribute(inspect.getattr_static(value, attribute))
            except AttributeError:
                break
            values.append(value)
        for index, value in enumerate(values):
            # a module is a namespace on the way to what the chain reaches
            if inspect.ismodule(value) and index + 1 < len(values):
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
        verdict = _judge(value)
        described = _describe(value)
        # data has no module of its own, so a module's data is judged as the module
        if verdict.kind is not _Kind.DISALLOWED and inspect.ismodule(owner):
            if not (inspect.isclass(value) or inspect.isroutine(value)):
                verdict = _classify(owner.__name__)
                described = (
                    f"a `{type(value).__qualname__}` in module `{owner.__name__}`"
                )
        if verdict.kind is _Kind.DISALLOWED:
            self._violate(node, reference, described, verdict.reason, scopes)
            return False
        if verdict.kind is _Kind.AUTHOR:
            self._checker.follow(value, (*self._target.via, self._where(scopes)))
        return True

    def _import(self, node: ast.Import | ast.ImportFrom, scopes: list[_Scope]) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self._import_module(node, alias.name, None, scopes)
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
            self._import_module(node, module, alias.name, scopes)

    def _import_module(
        self, node: ast.AST, module: str, name: str | None, scopes: list[_Scope]
    ) -> None:
        statement = (
            f"import {module}" if name is None else f"from {module} import {name}"
        )
        loaded = sys.modules.get(module)
        if loaded is not None:
            value: object = loaded
            if name is not None and name != "*":
                try:
                    value = _unwrap_attribute(inspect.getattr_static(loaded, name))
                except AttributeError:
                    pass
            self._judge(node, statement, value, scopes, loaded)
            return
        verdict = _classify(module)
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


def _chain(node: ast.Attribute) -> list[str] | None:
    attributes: list[str] = []
    current: ast.expr = node
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
            names.update(a.asname or a.name for a in node.names if a.name != "*")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, ast.Global):
            declared_global.update(node.names)
    return frozenset(names - declared_global)


def _walk_scope(nodes: Iterable[ast.AST]) -> Iterable[ast.AST]:
    pending = list(nodes)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, _SCOPES):
            continue
        pending.extend(ast.iter_child_nodes(node))
