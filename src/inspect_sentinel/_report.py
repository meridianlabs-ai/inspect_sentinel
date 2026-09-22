"""What monitors and protocols return, and the record the runner attaches identity to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from inspect_ai.tool import ToolCall
from pydantic import BaseModel, Field

Suspicion = float | dict[str, float]
"""How suspicious a step is. A dict scores several dimensions independently."""

Action = Literal["continue", "modify", "reject", "terminate", "escalate"]
"""What a protocol can decide about the step it examined."""


class Observation(BaseModel):
    """Evidence for a protocol to calibrate and threshold. What a monitor returns."""

    suspicion: Suspicion
    """How suspicious. Required: an observation with no score says nothing."""

    explanation: str | None = Field(default=None)
    """Why, for a reader of the transcript."""

    metadata: dict[str, Any] | None = Field(default=None)

    @classmethod
    def score(
        cls,
        suspicion: Suspicion,
        explanation: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """An observation with the given suspicion."""
        return cls(suspicion=suspicion, explanation=explanation, metadata=metadata)

    @classmethod
    def flag(cls, explanation: str | None = None) -> Observation:
        """A detector's observation: suspicion `1.0`, for a protocol to act on or not."""
        return cls(suspicion=1.0, explanation=explanation)


class Decision(BaseModel):
    """What should happen at a step. What a protocol returns; advisory to any protocol wrapping it."""

    action: Action

    audit: bool = Field(default=False)
    """Request that oversight budget be spent on this step."""

    authoritative: bool = Field(default=False)
    """A decision no protocol may weaken. Set by a human decider."""

    modified: ToolCall | None = Field(default=None)
    """Replacement for the step's subject, for a `modify` decision."""

    explanation: str | None = Field(default=None)
    """Why. Operator-facing by default."""

    metadata: dict[str, Any] | None = Field(default=None)

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


Report = Observation | Decision
"""What the transcript records."""

R = TypeVar("R", bound=Report)


@dataclass(frozen=True)
class Reported(Generic[R]):
    """A report with the identity of the configured instance that produced it."""

    name: str
    """Instance name: the mapping key, or the registry name."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`."""

    report: R
