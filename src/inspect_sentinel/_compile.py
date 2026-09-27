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

    A lone protocol is returned as it is, since a lone protocol is already `concurrent()` of one and wrapping it would push its children a path segment deeper. A monitor, or a sequence or mapping of monitors only, compiles to `observe()` and logs a warning, since nothing is configured to act on the scores; anything else containing a protocol compiles to `concurrent()`. Every top-level configuration is therefore a protocol.

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
    if single and registry_info(spec).type == "protocol":
        return cast(ControlProtocol, spec)
    if any(registry_info(child).type == "protocol" for _, child in named):
        return concurrent(children)
    for name, _ in named:
        logger.warning(
            "%s is a monitor and nothing is configured to act on it; wrap it in a protocol such as threshold() to act, or observe() to say that recording is intended.",
            name,
        )
    return observe(cast(Monitors, children))
