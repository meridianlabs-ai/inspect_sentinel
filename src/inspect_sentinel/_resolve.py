from __future__ import annotations

import logging
from typing import TypeAlias, cast

from inspect_ai._util.registry import is_registry_object, registry_info

from ._monitor import (
    Child,
    Children,
    Group,
    Monitor,
    MonitorGroup,
    Monitors,
    Protocol,
    ProtocolGroup,
)
from ._protocols import concurrent, observe
from ._runner import named_children

logger = logging.getLogger(__name__)

Sentinels: TypeAlias = Monitor | MonitorGroup | Protocol | ProtocolGroup | Children
"""What `Task(sentinel=)` accepts: one monitor, protocol or group, or a sequence or mapping of them."""


def resolve_sentinel(spec: Sentinels) -> Protocol:
    """Turn a sentinel configuration into the one protocol that owns the layer's decision.

    A lone protocol is the root itself, so `threshold(suspicion(), ...)` records `threshold` at the empty path and its monitor at `suspicion`. A lone `ProtocolGroup`, or a sequence or mapping containing a protocol, resolves to `concurrent()`: the root returns the step's one outcome, and combining the decisions of several functions is `concurrent`'s job. A monitor, or a sequence or mapping of monitors only, resolves to `observe()` and logs a warning, since nothing is configured to act on the scores. The dispatcher invokes the result as the root, so the root's children's paths are bare.

    Args:
        spec: One monitor or protocol, or a sequence or mapping of instance names to them.
    """
    single = is_registry_object(spec)
    children: Children = [cast(Child, spec)] if single else cast(Children, spec)
    named = named_children(children, None)
    if not named:
        raise ValueError(
            "A sentinel configuration needs at least one monitor or protocol."
        )
    if (
        single
        and not isinstance(spec, Group)
        and registry_info(spec).type == "protocol"
    ):
        return cast(Protocol, spec)
    if any(registry_info(child).type == "protocol" for _, child in named):
        return concurrent(children)
    for name, _ in named:
        logger.warning(
            "%s is a monitor and nothing is configured to act on it; wrap it in a protocol such as threshold() to act, or observe() to say that recording is intended.",
            name,
        )
    return observe(cast(Monitors, children))
