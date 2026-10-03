from __future__ import annotations

from typing import NamedTuple, NoReturn

from ._context import Context
from ._report import Decision, Reported
from ._step import Step


class Origin(NamedTuple):
    context: Context
    step: Step
    reported: Reported[Decision]


class Final(BaseException):
    """Raised by `decide_final()` and carried through the runner to `run_root`, which records its decision, the one time it is recorded, and returns it; each layer it passes is recorded as bypassed as it is passed. A `BaseException`, so a protocol's own `except Exception` cannot swallow it."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.action)
        self.decision = decision
        self.origin: Origin | None = None


def decide_final(decision: Decision) -> NoReturn:
    """End the step with this decision.

    Nothing above the calling protocol runs; siblings still in flight are cancelled and recorded as such. First call wins if two race.

    Args:
        decision: The outcome for this step.
    """
    raise Final(decision)
