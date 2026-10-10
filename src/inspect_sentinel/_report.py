from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

from inspect_ai.core import Reference, SentinelAction, SentinelSuspicion, ToolCall
from pydantic import BaseModel, Field, ModelWrapValidatorHandler, model_validator

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


D = TypeVar("D", bound="Decision")


class Decision(BaseModel):
    """What should happen at a step. What a protocol returns; advisory to any protocol wrapping it unless passed to `decide_final()`.

    Each action has its own class, `Proceed`, `Reject`, `Terminate`, `Modify` and `Escalate`, which the constructors (`Decision.reject(...)` and the others) return, so a protocol function can declare what it decides with a union such as `-> Reject | None`. `Decision(action=...)`, and validating a `Decision` from a dict or JSON, builds the class of the given action. Annotating a function `-> Decision` means it may decide anything.
    """

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

    def __new__(cls: type[D], /, *args: Any, **data: Any) -> D:
        # Decision(action=...) builds the class of its action
        target: type[Decision] = cls
        action = data.get("action")
        if cls is Decision and isinstance(action, str) and action in DECISION_CLASSES:
            target = DECISION_CLASSES[action]
        return cast(D, super().__new__(target))  # pyright: ignore[reportArgumentType]

    @model_validator(mode="wrap")
    @classmethod
    def _as_action_class(
        cls, data: Any, handler: ModelWrapValidatorHandler[Decision]
    ) -> Decision:
        # validating a Decision from a dict or JSON builds the class of its action
        if cls is Decision and isinstance(data, dict):
            action = cast(dict[str, Any], data).get("action")
            if isinstance(action, str) and action in DECISION_CLASSES:
                return DECISION_CLASSES[action].model_validate(data)
        return handler(data)

    @classmethod
    def proceed(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Proceed:
        """Let the step proceed unchanged: a `Proceed`, whose action is `continue`."""
        return Proceed(explanation=explanation, references=list(references))

    @classmethod
    def reject(
        cls,
        explanation: str | None = None,
        *,
        message: str | None = None,
        references: Sequence[Reference] = (),
    ) -> Reject:
        """Reject the step, telling the agent it was rejected: with `message` if given, else with the host's default text."""
        return Reject(
            explanation=explanation, message=message, references=list(references)
        )

    @classmethod
    def terminate(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Terminate:
        """End the sample."""
        return Terminate(explanation=explanation, references=list(references))

    @classmethod
    def escalate(
        cls, explanation: str | None = None, *, references: Sequence[Reference] = ()
    ) -> Escalate:
        """Decline to decide and pass the step to the next decider."""
        return Escalate(explanation=explanation, references=list(references))


class Proceed(Decision):
    """A decision to let the step proceed unchanged, action `continue`. Build one with `Decision.proceed()`."""

    action: Literal["continue"] = "continue"  # pyright: ignore[reportIncompatibleVariableOverride]
    """Always `continue`."""


class Reject(Decision):
    """A decision to reject the step, telling the agent with `message`. Build one with `Decision.reject()`. Legal only before a tool call."""

    action: Literal["reject"] = "reject"  # pyright: ignore[reportIncompatibleVariableOverride]
    """Always `reject`."""


class Terminate(Decision):
    """A decision to end the sample. Build one with `Decision.terminate()`."""

    action: Literal["terminate"] = "terminate"  # pyright: ignore[reportIncompatibleVariableOverride]
    """Always `terminate`."""


class Modify(Decision):
    """A decision to replace the step's subject with `modified`, which a `modify` must set. Legal only before a tool call, where it may change only the call's arguments."""

    action: Literal["modify"] = "modify"  # pyright: ignore[reportIncompatibleVariableOverride]
    """Always `modify`."""


class Escalate(Decision):
    """A decision not to decide, passing the step to the next decider in a `sequential()`. Build one with `Decision.escalate()`. One that reaches the root with nothing to handle it ends the sample as an unhandled escalation."""

    action: Literal["escalate"] = "escalate"  # pyright: ignore[reportIncompatibleVariableOverride]
    """Always `escalate`."""


DECISION_CLASSES: dict[Action, type[Decision]] = {
    "continue": Proceed,
    "reject": Reject,
    "terminate": Terminate,
    "modify": Modify,
    "escalate": Escalate,
}


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
