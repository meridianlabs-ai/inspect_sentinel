"""The contract inspect_ai depends on; it changes in step with inspect_ai.

Everything here is re-exported from the module that defines it. inspect_ai imports from this module only, so refactors inside the package have one file to keep stable.

The host implements `HostNative`, passes it with its `Recorder` and the sample store in a `HostContext`, which builds the top layer's `Context` with a `Host` around it, and invokes the resolved root through `run_sentinel`, which records the root's decision at the empty path like any layer's; a final decision is recorded there, when it takes effect, and returned as the step's outcome, so the host neither catches `Final` nor checks a decision's shape. A host whose recorder is also a `TimedRecorder` is told when each function starts and ends, to time its records.
"""

from ._config import sentinel_config, sentinel_from_config
from ._context import validate_instance_name
from ._decorators import step_types, watched_stages
from ._host import HostContext, HostNative, Recorder, TimedRecorder
from ._human_decisions import kept_human_decision, supersedes
from ._resolve import resolve_sentinel
from ._runner import run_sentinel
from ._sources import source_fallback
from ._types import Sentinels

# the name inspect_ai records the eval log's configuration through, until it
# takes `inspect_sentinel.config.sentinel_config()`
config_from_sentinel = sentinel_config

__all__ = [
    "Recorder",
    "HostContext",
    "HostNative",
    "config_from_sentinel",
    "kept_human_decision",
    "Sentinels",
    "sentinel_from_config",
    "validate_instance_name",
    "resolve_sentinel",
    "run_sentinel",
    "source_fallback",
    "step_types",
    "TimedRecorder",
    "supersedes",
    "watched_stages",
]
