"""Replay and read monitors and protocols over stored transcripts with Inspect Scout.

`as_scanner()` replays a monitor or protocol over each transcript a scan reads, with one result per step; `as_reader()` reads what one reported in the runs it was attached to. Their results join on transcript, step id and stage, and `fidelity_probe()` checks that replay hands a monitor what it was handed live. Needs the `scout` extra: `pip install inspect-sentinel[scout]`.
"""

from ._probe import fidelity_probe
from ._reader import as_reader
from ._scanner import Stage, as_scanner
from ._select import StepSelector, StepsIn, human_stand_in, step_selector, steps_in

__all__ = [
    "Stage",
    "StepSelector",
    "StepsIn",
    "as_reader",
    "as_scanner",
    "fidelity_probe",
    "human_stand_in",
    "step_selector",
    "steps_in",
]
