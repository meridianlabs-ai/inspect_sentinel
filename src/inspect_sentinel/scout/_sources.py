from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

from inspect_ai._util.registry import registry_lookup
from inspect_ai.log import SentinelConfig, SentinelEntry

# a bare name in a configuration also finds a factory in this package, which
# registers its own factories when it is imported
_PACKAGE = "inspect_sentinel"
_KINDS = ("monitor", "protocol")


def factory_sources(config: SentinelConfig) -> list[dict[str, str | None]]:
    # Where the factories a configuration names are defined, so another
    # process, such as a resumed scan or a worker, can import them before
    # rebuilding it: each module's name and its file, relative to the working
    # directory where it is under it, for a module that does not import by
    # name, such as a script run as `__main__`. A module with no file, typed
    # into a REPL, notebook or `python -c`, has none.
    sources: dict[str, str | None] = {}
    for name in _entry_names(config):
        for factory in _factories(name):
            module = getattr(factory, "__module__", None)
            if module is None or module.split(".")[0] == _PACKAGE:
                continue
            sources[module] = source_file(module)
    return [{"module": m, "file": f} for m, f in sorted(sources.items())]


def import_factory_sources(
    config: SentinelConfig, sources: Sequence[Mapping[str, str | None]]
) -> None:
    # Nothing is imported where every factory is registered, such as the
    # process that recorded the configuration, so an author's script or
    # notebook never runs again there.
    if not _unregistered(config):
        return
    for source in sources:
        module, file = source.get("module"), source.get("file")
        if module is not None:
            import_source(module, file)
    missing = _unregistered(config)
    if missing:
        raise ValueError(
            f"Cannot rebuild the sentinel in this process: {', '.join(repr(m) for m in missing)} is not a registered monitor or protocol, and importing the modules that defined its factories did not register it. A factory defined in a REPL, notebook or `python -c` can be rebuilt only in the process that defined it; define it in a module or script."
        )


def source_file(module: str) -> str | None:
    file = getattr(sys.modules.get(module), "__file__", None)
    return _relative(file) if file and Path(file).is_file() else None


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
    name = f"_sentinel_source_{path.stem}_{digest}"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    if not path.is_file():
        raise ValueError(
            f"Cannot import {str(path)!r}, which defines a monitor, protocol or callback the scan records: the file does not exist."
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


def _unregistered(config: SentinelConfig) -> list[str]:
    return sorted({n for n in _entry_names(config) if not _factories(n)})


def _factories(name: str) -> list[object]:
    # the factories a configured name finds, as building the configuration
    # looks them up
    exact = _lookup(name)
    if exact or "/" in name:
        return exact
    return _lookup(f"{_PACKAGE}/{name}")


def _lookup(name: str) -> list[object]:
    found = (registry_lookup(kind, name) for kind in _KINDS)
    return [factory for factory in found if factory is not None]


def _entry_names(config: SentinelConfig) -> Iterator[str]:
    # every monitor and protocol a configuration names: its entries, the
    # layers nested under them, and the instances recorded in their params
    layer = config.root
    entries: Sequence[SentinelEntry] = (
        [layer]
        if isinstance(layer, SentinelEntry)
        else list(layer.values())
        if isinstance(layer, Mapping)
        else layer
    )
    for entry in entries:
        yield entry.name
        yield from _instance_names(entry.params)
        for nested in entry.nested.values():
            yield from _entry_names(nested)


def _instance_names(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        name, params = mapping.get("name"), mapping.get("params")
        if mapping.get("type") in _KINDS and isinstance(name, str):
            # a monitor or protocol recorded as a registry dict
            yield name
            yield from _instance_names(params)
            return
        for item in mapping.values():
            yield from _instance_names(item)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            yield from _instance_names(item)
