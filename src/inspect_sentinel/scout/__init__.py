"""Replay and read monitors and protocols over stored transcripts with Inspect Scout.

`as_scanner()` replays a monitor or protocol over each transcript a scan reads, with one result per step; `as_reader()` reads what one reported in the runs it was attached to. Their results join on transcript, step id and stage, and `replay_check()` with `replay_check_differences()` checks that replay hands a monitor what it was handed live. Needs the `scout` extra: `pip install inspect-sentinel[scout]`.
"""

from ._probe import replay_check, replay_check_differences
from ._reader import as_reader
from ._scanner import Stage, as_scanner
from ._select import StepSelector, StepTable, human_stand_in, step_selector

__all__ = [
    "Stage",
    "StepSelector",
    "StepTable",
    "as_reader",
    "as_scanner",
    "human_stand_in",
    "replay_check",
    "replay_check_differences",
    "step_selector",
]
