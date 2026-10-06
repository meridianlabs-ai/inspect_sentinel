"""Stand-in for inspect_ai.util._store: a dict, without change events."""

from contextvars import ContextVar
from typing import Any


class Store:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        if default is not None and key not in self._data:
            self._data[key] = default
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self) -> Any:
        return self._data.keys()

    def values(self) -> Any:
        return self._data.values()

    def items(self) -> Any:
        return self._data.items()

    def __contains__(self, key: object) -> bool:
        return key in self._data


_store: ContextVar[Store] = ContextVar("store")


def store() -> Store:
    try:
        return _store.get()
    except LookupError:
        s = Store()
        _store.set(s)
        return s
