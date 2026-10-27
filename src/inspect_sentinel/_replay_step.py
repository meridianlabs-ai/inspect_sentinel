from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from ._host import HumanAnswer
from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step


@dataclass(frozen=True)
class ReplayStep:
    """A step rebuilt from a stored transcript, for replaying a sentinel over it: the step a monitor is given, with what identifies it in the transcript and what was recorded at it.

    Replay builds them in the order the steps ran live. The sample's `Context` is the same for every step of a transcript, so the replay holds it rather than each step. Only `step` reaches monitors and protocols; the other fields are the replay's own.

    Raises:
        ValueError: If `stage` is not the stage of `step`, or `segment` names a conversation other than `step.conversation`.
    """

    id: str
    """The step id, as `SentinelEvent.step_id` records it live: the tool call id at a tool stage, and the generate's id (the `uuid` of its first `ModelEvent`) at both generate stages. Unique within a transcript for a stage, so a label made against it serves replay and the live record alike."""

    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]
    """The stage of `step`, as `SentinelEvent.stage` records it: `model_input` for a `BeforeGenerate`, `model_output` for an `AfterGenerate`, `tool_call` for a `BeforeToolCall` and `tool_result` for an `AfterToolCall`."""

    step: Step
    """The step, as the live host would have built it, less what the transcript does not hold: for a step from a log, `AfterToolCall.output` is `None`."""

    segment: tuple[str, int]
    """Where the step falls in its conversation: the conversation (`step.conversation`), and how many of that conversation's compactions came before the step. Steps with the same segment ran between the same two compactions."""

    recorded_answers: Mapping[str, HumanAnswer] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The answers a person gave through `human()` at this step when it ran, keyed by the path of the `human()` instance that asked, for the replay host to give again when the same instance asks. Empty when none were recorded. It is the replay's own record and never part of `step`, so monitors and protocols do not see it. Read-only: a copy of the mapping given."""

    def __post_init__(self) -> None:
        expected = _STAGES[type(self.step)]
        if self.stage != expected:
            raise ValueError(
                f"ReplayStep {self.id!r} has stage {self.stage!r}, but its step is a {type(self.step).__name__}, whose stage is {expected!r}."
            )
        if self.segment[0] != self.step.conversation:
            raise ValueError(
                f"ReplayStep {self.id!r} has segment conversation {self.segment[0]!r}, but its step is in conversation {self.step.conversation!r}."
            )
        object.__setattr__(
            self, "recorded_answers", MappingProxyType(dict(self.recorded_answers))
        )


_STAGES: dict[type, str] = {
    BeforeGenerate: "model_input",
    AfterGenerate: "model_output",
    BeforeToolCall: "tool_call",
    AfterToolCall: "tool_result",
}
