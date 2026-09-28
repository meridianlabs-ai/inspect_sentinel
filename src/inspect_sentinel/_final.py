from __future__ import annotations

from typing import NoReturn

from ._report import Decision


class Final(BaseException):
    """Raised by `final()` and carried out of the runner to the dispatcher. A `BaseException`, so a protocol's own `except Exception` cannot swallow it."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.action)
        self.decision = decision
        self.origin_path: str | None = None


def final(decision: Decision) -> NoReturn:
    """End the step with this decision.

    Nothing above the calling protocol runs; siblings still in flight are cancelled and recorded as such. First call wins if two race.

    Args:
        decision: The outcome for this step.
    """
    raise Final(decision)
