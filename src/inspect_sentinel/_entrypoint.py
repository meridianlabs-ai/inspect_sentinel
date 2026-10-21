from importlib.util import find_spec

from ._protocols import concurrent, human, observe_only, sequential, threshold

# a scan that resumes, or runs in another process, finds the Scout scanners
# and the fidelity probe by their registry names, which load this module
if find_spec("inspect_scout") is not None:
    from .scout import as_reader as as_reader
    from .scout import fidelity_probe as fidelity_probe
    from .scout._scanner import sentinel_replay as sentinel_replay

    __all__ = [
        "as_reader",
        "concurrent",
        "fidelity_probe",
        "human",
        "observe_only",
        "sentinel_replay",
        "sequential",
        "threshold",
    ]
else:
    __all__ = ["concurrent", "human", "observe_only", "sequential", "threshold"]
