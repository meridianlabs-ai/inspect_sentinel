from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

from inspect_ai.core import SentinelConfig, SentinelEntry
from inspect_ai.core._registry import (
    RegistryDict,
    RegistryInfo,
    RegistryType,
    create_registry_object,
    has_registry_params,
    is_registry_dict,
    is_registry_object,
    registry_info,
    registry_lookup,
    registry_value,
)

from ._context import validate_instance_name
from ._decorators import ENTRY_FIELDS, VERSION
from ._portable import PortabilityError
from ._types import Sentinel, Sentinels

logger = logging.getLogger(__name__)

PACKAGE = "inspect_sentinel"

_warned_versions: set[tuple[str, int, int]] = set()


class _Factory(NamedTuple):
    kind: RegistryType
    name: str
    factory: object


def sentinel_from_config(
    config: str | SentinelConfig | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> Sentinels:
    """Build the monitors and protocols a configuration describes.

    Each entry is constructed through the registry with its `params` and its nested entries, which are built first. An entry whose recorded `version` differs from the installed factory's is built anyway, with a warning naming both versions; its `meta` is ignored. One entry, or a bare registered name, builds one instance, so it resolves as the root itself; a list or mapping builds a list or mapping. The result is not resolved; pass it to `resolve_sentinel`.

    Args:
        config: A registered monitor or protocol name, or the configuration itself: one entry, a list of entries, or a mapping of instance names to entries. The host reads configuration files and passes their `sentinel` value.

    Raises:
        ValueError: If the configuration is invalid; the message names the entry, as in `sentinel.attempt.children[1]`.
        TypeError: If a factory rejects its arguments, with the entry named the same way.
        PortabilityError: If a portable monitor or protocol references what a portable function cannot use, with the entry named the same way.
    """
    if isinstance(config, str):
        return _from_string(config)
    if isinstance(config, SentinelConfig):
        return _build_layer(config.model_dump(), "sentinel")
    return _build_layer(config, "sentinel")


def _from_string(config: str) -> Sentinels:
    if _find_all(config):
        return _build_entry({"name": config}, "sentinel")
    raise ValueError(
        f"{config!r} is neither a config file nor a registered monitor or protocol."
    )


def _build_layer(layer: object, path: str) -> Sentinels:
    if isinstance(layer, Mapping):
        entries = cast(Mapping[object, object], layer)
        if isinstance(entries.get("name"), str):
            return _build_entry(entries, path)
        built: dict[str, Sentinel] = {}
        for key, entry in entries.items():
            try:
                name = validate_instance_name(key)
            except ValueError as ex:
                raise ValueError(f"{path}: {ex}") from ex
            built[name] = _build_entry(entry, f"{path}.{name}")
        return built
    if isinstance(layer, Sequence) and not isinstance(layer, str):
        items = cast(Sequence[object], layer)
        return [_build_entry(entry, f"{path}[{i}]") for i, entry in enumerate(items)]
    raise ValueError(
        f"{path} must be a list or a mapping of entries, not {type(layer).__name__}."
    )


def _build_entry(entry: object, path: str) -> Sentinel:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{path} must be a mapping with a 'name', not {entry!r}.")
    fields = cast(Mapping[str, object], entry)
    name = fields.get("name")
    if not isinstance(name, str):
        raise ValueError(
            f"{path}: an entry needs a 'name' naming a monitor or protocol."
        )
    params = fields.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"{path}.params must be a mapping of arguments.")
    args = dict(cast(Mapping[str, object], params))
    found = _find(name, path)
    _check_version(found, fields.get(VERSION), path)
    signature = inspect.signature(cast(Any, found.factory)).parameters
    accepted = [key for key, p in signature.items() if p.kind is not p.VAR_KEYWORD]
    any_key = len(accepted) < len(signature)
    for key, value in fields.items():
        if key in ENTRY_FIELDS:
            continue
        if key not in accepted and not any_key:
            raise ValueError(
                f"{path}: {key!r} is not a parameter of {name}; an entry takes 'name', 'params', and nested entries under one of {name}'s parameters {accepted}."
            )
        if key in args:
            raise ValueError(f"{path}: {key!r} is given both nested and in params.")
        args[key] = _build_layer(value, f"{path}.{key}")
    try:
        instance = create_registry_object(found.kind, found.name, args)
    except (TypeError, ValueError) as ex:
        error = next(
            (kind for kind in (PortabilityError, TypeError) if isinstance(ex, kind)),
            ValueError,
        )
        raise error(f"{path}: {ex}") from ex
    return cast(Sentinel, instance)


def _check_version(found: _Factory, recorded: object, path: str) -> None:
    if recorded is None:
        return
    if not isinstance(recorded, int) or isinstance(recorded, bool):
        raise ValueError(f"{path}.version must be an integer, not {recorded!r}.")
    installed = _version(registry_info(found.factory))
    key = (found.name, recorded, installed)
    if recorded != installed and key not in _warned_versions:
        _warned_versions.add(key)
        logger.warning(
            "%s was recorded at version %d but version %d is installed; building it anyway.",
            found.name,
            recorded,
            installed,
        )


def _version(info: RegistryInfo) -> int:
    return cast(int, info.metadata.get(VERSION, 0))


def _find(name: str, path: str) -> _Factory:
    found = _find_all(name)
    if not found:
        raise ValueError(f"{path}: {name!r} is not a registered monitor or protocol.")
    if len(found) > 1:
        candidates = ", ".join(f"{f.kind} {f.name}" for f in found)
        raise ValueError(f"{path}: {name!r} is ambiguous; it names {candidates}.")
    return found[0]


def _find_all(name: str) -> list[_Factory]:
    exact = _lookup(name)
    if exact or "/" in name:
        return exact
    return _lookup(f"{PACKAGE}/{name}")


def _lookup(name: str) -> list[_Factory]:
    found: list[_Factory] = []
    for kind in cast(list[RegistryType], ["monitor", "protocol"]):
        factory = registry_lookup(kind, name)
        if factory is not None:
            found.append(_Factory(kind, registry_info(factory).name, factory))
    return found


def config_from_sentinel(sentinels: Sentinels) -> SentinelConfig:
    """Record constructed monitors and protocols as the configuration that rebuilds them.

    The inverse of `sentinel_from_config`, for the eval log and retry: each instance becomes an entry with its registry name, the params it was created with, and its factory's version unless that is 0, and a param holding monitors or protocols becomes nested entries. A package monitor or protocol is recorded by its bare name when that finds it unambiguously. A lone instance is recorded as a lone entry, so it rebuilds as the root it was.

    Args:
        sentinels: One monitor or protocol, or a sequence or mapping of instance names to them, as `Task(sentinel=)` accepts.

    Raises:
        TypeError: If a value is not a configured monitor or protocol.
    """
    if is_registry_object(sentinels):
        return SentinelConfig(_instance_entry(sentinels))
    if isinstance(sentinels, Mapping):
        mapping = cast(Mapping[str, object], sentinels)
        return SentinelConfig(
            {name: _instance_entry(child) for name, child in mapping.items()}
        )
    items = cast(Sequence[object], sentinels)
    return SentinelConfig([_instance_entry(child) for child in items])


def _instance_entry(instance: object) -> SentinelEntry:
    recorded = _registry_dict(instance)
    return _config_entry(recorded, _version(registry_info(instance)))


def _registry_dict(instance: object) -> RegistryDict:
    if not has_registry_params(instance) or registry_info(instance).type not in (
        "monitor",
        "protocol",
    ):
        raise TypeError(
            f"{getattr(instance, '__name__', instance)!r} is not a configured monitor or protocol."
        )
    return cast(RegistryDict, registry_value(instance))


def _config_layer(
    layer: Sequence[RegistryDict] | Mapping[str, RegistryDict],
) -> SentinelConfig:
    if isinstance(layer, Mapping):
        return SentinelConfig({name: _nested_entry(d) for name, d in layer.items()})
    return SentinelConfig([_nested_entry(d) for d in layer])


def _nested_entry(recorded: RegistryDict) -> SentinelEntry:
    # registry params hold nested instances as dicts, so the version comes from
    # the factory they name
    factory = registry_lookup(recorded["type"], recorded["name"])
    version = 0 if factory is None else _version(registry_info(factory))
    return _config_entry(recorded, version)


def _config_entry(recorded: RegistryDict, version: int) -> SentinelEntry:
    params: dict[str, Any] = {}
    nested: dict[str, SentinelConfig] = {}
    for key, value in recorded["params"].items():
        layer = None if key in ENTRY_FIELDS else _sentinel_layer(value)
        if layer is None:
            params[key] = value
        else:
            nested[key] = _config_layer(layer)
    fields: dict[str, Any] = {"name": _config_name(recorded["name"]), "params": params}
    if version != 0:
        fields[VERSION] = version
    return SentinelEntry.model_validate({**fields, **nested})


def _sentinel_layer(
    value: object,
) -> Sequence[RegistryDict] | Mapping[str, RegistryDict] | None:
    if isinstance(value, list) and value:
        items = cast(list[object], value)
        if all(_is_sentinel_dict(item) for item in items):
            return cast(list[RegistryDict], items)
    elif isinstance(value, dict) and value:
        entries = cast(dict[str, object], value)
        if not is_registry_dict(entries) and all(
            _is_sentinel_dict(item) for item in entries.values()
        ):
            return cast(dict[str, RegistryDict], entries)
    return None


def _is_sentinel_dict(value: object) -> bool:
    return is_registry_dict(value) and value["type"] in ("monitor", "protocol")


def _config_name(name: str) -> str:
    prefix = f"{PACKAGE}/"
    if name.startswith(prefix):
        bare = name[len(prefix) :]
        found = _find_all(bare)
        if len(found) == 1 and found[0].name == name:
            return bare
    return name
