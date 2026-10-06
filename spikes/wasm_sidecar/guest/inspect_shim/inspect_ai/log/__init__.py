"""Stand-in: config serialization (`_config.py`) is out of scope in the guest."""

from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel


class SentinelEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    params: dict[str, Any] = {}


class SentinelConfig(RootModel[Any]):
    pass
