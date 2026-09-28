# The surface inspect_ai's dispatcher imports; not part of the author-facing API.
#
# Everything here is re-exported from the module that defines it. inspect_ai
# imports from this module only, so refactors inside the package have one file
# to keep stable.
#
# The dispatcher awaits the compiled root protocol inside
# `try: ... except Final as ex: decision = ex.decision`; a normal return is the
# root's own decision.

from ._check import check_decision_shape
from ._compile import SentinelSpec, compile_sentinel
from ._context import Recorder, RunnerContext, check_instance_name
from ._final import Final
from ._monitor import step_types

__all__ = [
    "Final",
    "Recorder",
    "RunnerContext",
    "SentinelSpec",
    "check_decision_shape",
    "check_instance_name",
    "compile_sentinel",
    "step_types",
]
