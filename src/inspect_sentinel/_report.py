from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

from inspect_ai.core import Reference, ToolCall
from inspect_ai.event import SentinelAction, SentinelSuspicion
from pydantic import BaseModel, Field

Suspicion: TypeAlias = SentinelSuspicion
"""How suspicious a step is: one finite score, or a non-empty dict scoring several dimensions independently."""

Action: TypeAlias = SentinelAction
"""What a protocol can decide about the step it examined."""

PRECEDENCE: dict[Action, int] = {
    "terminate": 5,
    "reject": 4,
    "modify": 3,
    "escalate": 2,
    "continue": 1,
}


class Observation(BaseModel):
    """Evidence for a protocol to calibrate and threshold. What a monitor returns."""

    suspicion: Suspicion
    """How suspicious. Required: an observation with no score says nothing."""

    explanation: str | None = Field(default=None)
    """Why, for a reader of the transcript."""

    references: list[Reference] = Field(default_factory=list[Reference])
    """Messages and events in the transcript the report points at, which Inspect View links from cites such as `[M22]` in `explanation`."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Author-supplied structured context, recorded verbatim."""

    @classmethod
    def score(
        cls,
        suspicion: Suspicion,
        explanation: str | None = None,
        *,
        references: Sequence[Reference] = (),
    ) -> Observation:
        """An observation that sets `suspicion` to the given score."""
        return cls(
            suspicion=suspicion, explanation=explanation, references=list(references)
        )

    @classmethod
    def flag(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Observation:
        """A detector's observation: suspicion `1.0`, for a protocol to act on or not."""
        return cls(suspicion=1.0, explanation=explanation, references=list(references))


class Decision(BaseModel):
    """What should happen at a step. What a protocol returns; advisory to any protocol wrapping it unless passed to `decide_final()`."""

    action: Action
    """What should happen at this step. Required."""

    audit: bool = Field(default=False)
    """Request that oversight budget be spent on this step. Recorded for offline analysis and shown in the viewer as a flag; nothing acts on it yet."""

    modified: ToolCall | None = Field(default=None)
    """Replacement for the step's subject, for a `modify` decision."""

    explanation: str | None = Field(default=None)
    """Why, for the log. The agent never sees it."""

    references: list[Reference] = Field(default_factory=list[Reference])
    """Messages and events in the transcript the report points at, which Inspect View links from cites such as `[M22]` in `explanation`."""

    message: str | None = Field(default=None)
    """Text the agent reads when the step is rejected. The agent sees nothing else from a decision."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Author-supplied structured context, recorded verbatim."""

    @classmethod
    def proceed(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Decision:
        """Let the step proceed unchanged: a decision with action `continue`."""
        return cls(
            action="continue", explanation=explanation, references=list(references)
        )

    @classmethod
    def reject(
        cls,
        explanation: str | None = None,
        *,
        message: str | None = None,
        references: Sequence[Reference] = (),
    ) -> Decision:
        """Reject the step, telling the agent it was rejected: with `message` if given, else with the host's default text."""
        return cls(
            action="reject",
            explanation=explanation,
            message=message,
            references=list(references),
        )

    @classmethod
    def terminate(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Decision:
        """End the sample."""
        return cls(
            action="terminate", explanation=explanation, references=list(references)
        )

    @classmethod
    def escalate(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Decision:
        """Decline to decide and pass the step to the next decider."""
        return cls(
            action="escalate", explanation=explanation, references=list(references)
        )


Report: TypeAlias = Observation | Decision
"""What the transcript records."""

R_co = TypeVar("R_co", bound=Report, covariant=True)


@dataclass(frozen=True)
class Reported(Generic[R_co]):
    """A report with the identity of the configured instance that produced it."""

    name: str
    """Instance name: the mapping key, or the registry name without its package prefix."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`."""

    report: R_co
    """The observation or decision this instance produced."""

    function: str
    """The `__name__` of the function that produced the report, which tells apart the functions of an instance whose factory returned several."""


@dataclass(frozen=True)
class Failed:
    """A monitor function that raised instead of reporting, with the identity of its configured instance."""

    name: str
    """Instance name, as in `Reported.name`."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`."""

    function: str
    """The `__name__` of the function that raised."""

    error: Exception
    """What it raised."""
