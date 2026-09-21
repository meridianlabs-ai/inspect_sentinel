"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = ["__version__"]
