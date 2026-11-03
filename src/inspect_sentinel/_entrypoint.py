from typing import Any

from inspect_ai._util.registry import RegistryInfo, registry_add, registry_has

from ._protocols import concurrent, human, observe_only, sequential, threshold

__all__ = ["concurrent", "human", "observe_only", "sequential", "threshold"]


# A scan that resumes, or runs in another process, finds the Scout scanners by
# their registry names, which load this module whether or not Scout is loaded
# yet. So this module registers stand-ins under those names that import the
# adapter only when Scout calls one, and an eval, which never does, never
# loads Scout. Importing the adapter registers the scanners themselves in
# their place.
def _replay(**params: Any) -> Any:
    from .scout._scanner import sentinel_replay

    return sentinel_replay(**params)


def _read(**params: Any) -> Any:
    from .scout._reader import as_reader

    return as_reader(**params)


for _name, _factory in (("replay", _replay), ("read", _read)):
    if not registry_has("scanner", f"inspect_sentinel/{_name}"):
        registry_add(
            _factory, RegistryInfo(type="scanner", name=f"inspect_sentinel/{_name}")
        )
