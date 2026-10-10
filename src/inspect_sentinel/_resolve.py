from __future__ import annotations

from typing import cast

from inspect_ai._util.registry import is_registry_object, registry_info

from ._escalation import warn_unhandled
from ._protocols import concurrent
from ._types import (
    Group,
    Protocol,
    Sentinel,
    Sentinels,
)
from ._validate import named_children


def resolve_sentinel(spec: Sentinels) -> Protocol:
    """Turn a sentinel configuration into the one protocol that owns the layer's decision.

    A lone protocol is the root itself, so `threshold(suspicion(), ...)` records `threshold` at the empty path and its monitor at `suspicion`. A lone `ProtocolGroup`, or a sequence or mapping containing a protocol, resolves to `concurrent()`: the root returns the step's one outcome, and combining the decisions of several functions is `concurrent`'s job; monitors beside the protocols are recorded and nothing acts on them. The host invokes the result as the root, so the root's children's paths are bare.

    Logs a warning, once per process for each configuration it describes, when the root can return `escalate` with nothing to handle it, which the host treats as an unhandled escalation and ends the sample. Shipped protocols declare whether they can escalate and which escalations they handle: `human()` and `handle_escalation()` handle the escalations before them in a `sequential()`, and `threshold()` and `observe_only()` never escalate. A protocol that is not shipped is assumed to be able to escalate at every stage it watches.

    Args:
        spec: One protocol, or a sequence or mapping of instance names to monitors and protocols, at least one of them a protocol.

    Raises:
        ValueError: If `spec` holds no protocol: a monitor, a `MonitorGroup`, or a sequence or mapping of only monitors. Wrap monitors in a protocol such as `threshold()` to act on them, or `observe_only()` to record them without acting.
    """
    single = is_registry_object(spec)
    children: Sentinels = [cast(Sentinel, spec)] if single else spec
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
        root = cast(Protocol, spec)
    elif any(registry_info(child).type == "protocol" for _, child in named):
        root = concurrent(children)
    else:
        raise ValueError(
            f"A sentinel needs a protocol to decide each step, but it was given only monitors: {', '.join(name for name, _ in named)}. "
            "Wrap them in threshold() to act on their scores, or in observe_only() to record them without acting."
        )
    warn_unhandled(root)
    return root
