from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

from inspect_ai.core import Reference, SentinelAction, SentinelSuspicion, ToolCall
from pydantic import BaseModel, Field, PrivateAttr

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


class _Origin:
    # the step and invocation that returned a decision first. Every copy,
    # shallow or deep, shares it, so a decision passed up, as is or rewritten
    # with model_copy, still says who returned it. It is not part of the
    # decision's value: every origin compares equal, and pickling drops it.
    __slots__ = ("step", "node")

    def __init__(self, step: object = None, node: object = None) -> None:
        self.step = step
        self.node = node

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Origin)

    def __hash__(self) -> int:
        return 0

    def __deepcopy__(self, memo: dict[int, object]) -> _Origin:
        return self

    def __reduce__(self) -> tuple[type[_Origin], tuple[()]]:
        return (_Origin, ())


class Decision(BaseModel, frozen=True):
    """What should happen at a step. What a protocol returns; advisory to any protocol wrapping it unless passed to `decide_final()`.

    A monitor makes an `Observation`; a protocol makes a `Decision`. A `Decision` is either a `Verdict`, one of `Proceed`, `Reject`, `Terminate` and `Modify`, or an `Escalate`: `Decision = Verdict | Escalate`. `Verdict` is a type alias, for annotations and `isinstance()`, not a class to build.

    The base of one class per action, `Proceed`, `Reject`, `Terminate`, `Modify` and `Escalate`, and not built itself: build a decision with one of those classes, or with `Decision.proceed()`, `Decision.reject()`, `Decision.terminate()` or `Decision.escalate()`, which return them (a modify is built as `Modify(modified=...)`). A protocol function declares what it decides with a union of the classes, such as `-> Reject | Proceed | None`; annotating it `-> Decision` means it may decide anything. Decisions are frozen; derive one from another with `model_copy(update=...)`.
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

    def model_post_init(self, context: Any, /) -> None:
        if type(self) is Decision:
            raise TypeError(
                "Decision is the base of the decision classes and is not built itself; build a Proceed, Reject, Terminate, Modify or Escalate, for example with Decision.reject()."
            )

    _origin: _Origin = PrivateAttr(default_factory=_Origin)

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


class Proceed(Decision, frozen=True):
    """A decision to let the step proceed unchanged, action `continue`. Build one with `Decision.proceed()`."""

    action: Literal["continue"] = "continue"
    """Always `continue`."""


class Reject(Decision, frozen=True):
    """A decision to reject the step, telling the agent with `message`. Build one with `Decision.reject()`. Legal only before a tool call."""

    action: Literal["reject"] = "reject"
    """Always `reject`."""


class Terminate(Decision, frozen=True):
    """A decision to end the sample. Build one with `Decision.terminate()`."""

    action: Literal["terminate"] = "terminate"
    """Always `terminate`."""


class Modify(Decision, frozen=True):
    """A decision to replace the step's subject with `modified`, as in `Modify(modified=replace(step.call, arguments={...}))`. Legal only before a tool call, where it may change only the call's arguments."""

    action: Literal["modify"] = "modify"
    """Always `modify`."""

    # required here though optional on the base, which every decision carries
    # since hosts read it from any decision
    modified: ToolCall = Field()
    """The replacement for the step's subject. Required."""


class Escalate(Decision, frozen=True):
    """A decision not to decide, passing the step to the next decider in a `sequential()`. Build one with `Decision.escalate()`. One that reaches the root with nothing to handle it ends the sample as an unhandled escalation."""

    action: Literal["escalate"] = "escalate"
    """Always `escalate`."""


Verdict: TypeAlias = Proceed | Reject | Terminate | Modify
"""Every decision but `Escalate`: what a protocol that always decides returns, and what `run_sentinel` returns for a step. `Decision = Verdict | Escalate`. A type alias, for annotations and `isinstance()`, not a class to build."""

DECISION_CLASSES: dict[Action, type[Decision]] = {
    "continue": Proceed,
    "reject": Reject,
    "terminate": Terminate,
    "modify": Modify,
    "escalate": Escalate,
}


# read by name: the attribute is private to the class, the stamp to the runner
_ORIGIN = "_origin"
_D = TypeVar("_D", bound=Decision)


def origin(decision: Decision) -> tuple[object, object]:
    stamp = cast(_Origin, getattr(decision, _ORIGIN))
    return stamp.step, stamp.node


def with_origin(decision: _D, step: object, node: object) -> _D:
    copy = decision.model_copy()
    setattr(copy, _ORIGIN, _Origin(step, node))
    return copy


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
