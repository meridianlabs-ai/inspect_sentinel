from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

# the prefix of the name a module loaded from its file gets
_FROM_FILE = "_sentinel_source_"


def source_file(module: str) -> str | None:
    file = getattr(sys.modules.get(module), "__file__", None)
    return file_reference(file) if file else None


def file_reference(file: str) -> str | None:
    # a file as a configuration or scan records it: relative to the working
    # directory where it is under it; None for one that isn't a file, such as
    # the `<stdin>` of code typed into a REPL
    return _relative(file) if Path(file).is_file() else None


def importable_by_name(module: str) -> bool:
    # whether another process can import a loaded module by its name: not a
    # script run as `__main__`, a module loaded from its file under a name of
    # its own, or a file inspect_ai loaded under its path, which is not in
    # `sys.modules`
    loaded = sys.modules.get(module)
    spec = getattr(loaded, "__spec__", None)
    return (
        module != "__main__"
        and not module.startswith(_FROM_FILE)
        and spec is not None
        and spec.name == module
    )


def import_source(module: str, file: str | None) -> ModuleType | None:
    # The module a recorded source names, imported if this process lacks it:
    # by name, else from its file under a name of its own. A script, recorded
    # as `__main__`, is this process's `__main__` (or `__mp_main__`, as a
    # worker started by spawning runs it) when that is its file, and is
    # otherwise loaded from the file, once. None for a `__main__` with no file.
    path = _absolute(file) if file is not None else None
    if module == "__main__":
        for alias in ("__main__", "__mp_main__"):
            loaded = sys.modules.get(alias)
            main = getattr(loaded, "__file__", None)
            if main is not None and path is not None and Path(main).resolve() == path:
                return loaded
        return _import_file(path) if path is not None else None
    if module in sys.modules:
        return sys.modules[module]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as ex:
        # only the module itself missing from the path, not one it imports
        if path is None or ex.name is None or not module.startswith(ex.name):
            raise
        return _import_file(path)


def _import_file(path: Path) -> ModuleType:
    # under a name of its own, so it never stands in for a module of the
    # same name, nor runs twice
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    name = f"{_FROM_FILE}{path.stem}_{digest}"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    if not path.is_file():
        raise ValueError(
            f"Cannot import {str(path)!r}, which defines a monitor, protocol or function that a configuration or scan names: the file does not exist."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {str(path)!r}.")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _relative(file: str) -> str:
    path = Path(file).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _absolute(file: str) -> Path:
    return (Path.cwd() / file).resolve()


_SOURCE_REFERENCE = "__inspect_sentinel_source_reference__"

_Converter = Callable[[object], object]


def source_reference(converter: _Converter) -> _Converter:
    # marks a `recorded_as()` converter whose recorded value is a source
    # reference, "module:function" where the module may be a `.py` file's
    # path, so `config_relative_to()` records the file relative to a directory
    setattr(converter, _SOURCE_REFERENCE, True)
    return converter


def is_source_reference(converter: object) -> bool:
    return getattr(converter, _SOURCE_REFERENCE, False) is True


def reference_origin(given: object, recorded: object) -> Path | None:
    # the file a recorded "<file>.py:function" reference names, resolved as
    # `import_source()` resolves it in the working directory, so a symlink
    # followed by `..` names the file the import found. None for a module's
    # name, and for a file given by its absolute path, which stays as given
    if not isinstance(recorded, str):
        return None
    module, _, qualname = recorded.rpartition(":")
    if not module.endswith(".py") or not qualname:
        return None
    if isinstance(given, str) and Path(module).is_absolute():
        return None
    return _absolute(module)


def relative_reference(recorded: str, origin: Path, directory: str) -> str:
    # a reference to the file `origin` relative to `directory`, or absolute
    # where there is no relative path to it (another drive)
    _, _, qualname = recorded.rpartition(":")
    try:
        file = os.path.relpath(origin, Path(directory).resolve())
    except ValueError:
        file = str(origin)
    return f"{Path(file).as_posix()}:{qualname}"
