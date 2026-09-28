# The surface inspect_ai's dispatcher imports; not part of the author-facing API.
#
# Everything here is re-exported from the module that defines it. inspect_ai
# imports from this module only, so refactors inside the package have one file
# to keep stable.
#
# The dispatcher invokes the compiled root through `run_root`, which records
# the root's decision at the empty path like any layer's; a final decision is
# recorded there, when it takes effect, and returned as the step's outcome, so
# the dispatcher neither catches `Final` nor checks a decision's shape.

from ._compile import SentinelSpec, compile_sentinel
from ._context import Recorder, RunnerContext, check_instance_name
from ._monitor import step_types
from ._runner import run_root

__all__ = [
    "Recorder",
    "RunnerContext",
    "SentinelSpec",
    "check_instance_name",
    "compile_sentinel",
    "run_root",
    "step_types",
]
