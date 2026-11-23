from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

from inspect_ai._util.registry import registry_lookup
from inspect_ai.log import SentinelConfig, SentinelEntry

from .._sources import import_source, source_file

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
