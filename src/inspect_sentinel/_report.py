from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from inspect_ai.tool import ToolCall
from pydantic import BaseModel, Field

Score: TypeAlias = Annotated[float, Field(allow_inf_nan=False)]
"""A finite suspicion score."""

Suspicion: TypeAlias = Score | Annotated[dict[str, Score], Field(min_length=1)]
"""How suspicious a step is: one score, or a non-empty dict scoring several dimensions independently."""

Action: TypeAlias = Literal["continue", "modify", "reject", "terminate", "escalate"]
"""What a protocol can decide about the step it examined."""


class Observation(BaseModel):
    """Evidence for a protocol to calibrate and threshold. What a monitor returns."""

    suspicion: Suspicion
    """How suspicious. Required: an observation with no score says nothing."""

    explanation: str | None = Field(default=None)
    """Why, for a reader of the transcript."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Author-supplied structured context, recorded verbatim."""

    @classmethod
    def score(cls, suspicion: Suspicion, explanation: str | None = None) -> Observation:
        """An observation with the given suspicion."""
        return cls(suspicion=suspicion, explanation=explanation)

    @classmethod
    def flag(cls, explanation: str | None = None) -> Observation:
        """A detector's observation: suspicion `1.0`, for a protocol to act on or not."""
        return cls(suspicion=1.0, explanation=explanation)


class Decision(BaseModel):
    """What should happen at a step. What a protocol returns; advisory to any protocol wrapping it."""

    action: Action
    """What should happen at this step. Required."""

    audit: bool = Field(default=False)
    """Request that oversight budget be spent on this step."""

    authoritative: bool = Field(default=False)
    """A decision no protocol may weaken. Set by a human decider."""

    modified: ToolCall | None = Field(default=None)
    """Replacement for the step's subject, for a `modify` decision."""

    explanation: str | None = Field(default=None)
    """Why. Operator-facing by default."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Author-supplied structured context, recorded verbatim."""

    @classmethod
    def clear(cls, explanation: str | None = None) -> Decision:
        """Let the step proceed unchanged."""
        return cls(action="continue", explanation=explanation)

    @classmethod
    def reject(cls, explanation: str | None = None) -> Decision:
        """Reject the step, telling the agent it was rejected."""
        return cls(action="reject", explanation=explanation)

    @classmethod
    def terminate(cls, explanation: str | None = None) -> Decision:
        """End the sample."""
        return cls(action="terminate", explanation=explanation)

    @classmethod
    def escalate(cls, explanation: str | None = None) -> Decision:
        """Decline to decide and pass the step to the next decider."""
        return cls(action="escalate", explanation=explanation)


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
