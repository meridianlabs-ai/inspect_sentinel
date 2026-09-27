from __future__ import annotations

import logging
from typing import TypeAlias, cast

from inspect_ai._util.registry import is_registry_object, registry_info

from ._monitor import Children, ControlProtocol, Monitor, Monitors
from ._protocols import concurrent, observe
from ._runner import named_children

logger = logging.getLogger(__name__)

SentinelSpec: TypeAlias = Monitor | ControlProtocol | Children
"""What `Task(sentinel=)` accepts: one monitor or protocol, or a sequence or mapping of them."""


def compile_sentinel(spec: SentinelSpec) -> ControlProtocol:
    """Turn a sentinel configuration into the one protocol that owns the layer's decision.

    Every configuration is wrapped, so a lone child compiles exactly as a list of one does and records the same paths. A monitor, or a sequence or mapping of monitors only, compiles to `observe()` and logs a warning, since nothing is configured to act on the scores; anything containing a protocol, a lone protocol included, compiles to `concurrent()`. Every top-level configuration is therefore one of the two, and the dispatcher invokes it as the root, so the top-level children's paths are bare.

    Args:
        spec: One monitor or protocol, or a sequence or mapping of instance names to them.
    """
    single = is_registry_object(spec)
    children: Children = (
        [cast(Monitor | ControlProtocol, spec)] if single else cast(Children, spec)
    )
    named = named_children(children, None)
    if not named:
        raise ValueError(
            "A sentinel configuration needs at least one monitor or protocol."
        )
    if any(registry_info(child).type == "protocol" for _, child in named):
        return concurrent(children)
    for name, _ in named:
        logger.warning(
            "%s is a monitor and nothing is configured to act on it; wrap it in a protocol such as threshold() to act, or observe() to say that recording is intended.",
            name,
        )
    return observe(cast(Monitors, children))
