import sys

from ._protocols import concurrent, human, observe_only, sequential, threshold

# A scan that resumes, or runs in another process, finds the Scout scanners
# and the fidelity probe by their registry names, which load this module. A
# process scanning has imported Scout already; any other, such as an eval,
# does not load it.
if "inspect_scout" in sys.modules:
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
