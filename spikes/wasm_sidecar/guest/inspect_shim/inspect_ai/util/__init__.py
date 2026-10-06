"""Stand-ins for Store, StoreModel, LimitExceededError and resource."""

from typing import Any

from ._store import Store, store
from ._store_model import StoreModel


class LimitExceededError(Exception):
    """Stand-in: the runner only catches it."""


def resource(
    resource: str, type: str = "auto", fs_options: dict[str, Any] | None = None
) -> str:
    """Stand-in: configuration loading is out of scope in the guest."""
    return resource


__all__ = ["LimitExceededError", "Store", "StoreModel", "resource", "store"]
