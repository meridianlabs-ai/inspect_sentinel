"""Stand-in for inspect_ai._util.registry: the subset inspect_sentinel uses.

The real module imports entry points, package metadata and (lazily) most of
inspect_ai. Objects carry their info on attributes, as in the real module.
"""

import inspect
from typing import Any, Callable, Literal, cast

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

RegistryType = Literal["monitor", "protocol", "solver", "scorer", "tool", "model", "approver"]
REGISTRY_INFO = "__registry_info__"
REGISTRY_PARAMS = "__registry_params__"

_registry: dict[str, object] = {}


class RegistryDict(TypedDict):
    type: str
    name: str
    params: dict[str, Any]


class RegistryInfo(BaseModel):
    type: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def registry_name(o: object, name: str) -> str:
    package = getattr(o, "__module__", "").split(".")[0]
    return f"{package}/{name}" if package == "inspect_sentinel" else name


def registry_unqualified_name(o: object) -> str:
    info = o if isinstance(o, RegistryInfo) else registry_info(o)
    return info.name.split("/")[-1]


def registry_add(o: object, info: RegistryInfo) -> None:
    setattr(o, REGISTRY_INFO, info)
    _registry[f"{info.type}:{info.name}"] = o


def registry_has(type: str, name: str) -> bool:
    return f"{type}:{name}" in _registry


def registry_lookup(type: str, name: str) -> object | None:
    return _registry.get(f"{type}:{name}") or _registry.get(f"{type}:inspect_sentinel/{name}")


def registry_info(o: object) -> RegistryInfo:
    info = getattr(o, REGISTRY_INFO, None)
    if info is None:
        raise ValueError(f"{o!r} is not a registry object")
    return info


def is_registry_object(o: object, type: str | None = None) -> bool:
    info = getattr(o, REGISTRY_INFO, None)
    return info is not None and (type is None or info.type == type)


def registry_tag(type_or_factory: Callable[..., Any], o: object, info: RegistryInfo, *args: Any, **kwargs: Any) -> None:
    bound = inspect.signature(type_or_factory).bind_partial(*args, **kwargs)
    setattr(o, REGISTRY_INFO, info)
    setattr(o, REGISTRY_PARAMS, dict(bound.arguments))


def has_registry_params(o: object) -> bool:
    return hasattr(o, REGISTRY_PARAMS)


def registry_value(o: object) -> dict[str, Any]:
    return {"type": registry_info(o).type, "name": registry_info(o).name, "params": getattr(o, REGISTRY_PARAMS, {})}


def is_registry_dict(o: object) -> bool:
    return isinstance(o, dict) and "type" in o and "name" in o


def create_registry_object(type: str, name: str, kwargs: dict[str, Any]) -> object:
    factory = registry_lookup(type, name)
    if factory is None:
        raise ValueError(f"{type} {name!r} not found")
    return cast(Callable[..., object], factory)(**kwargs)
