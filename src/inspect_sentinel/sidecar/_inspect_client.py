from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from inspect_ai.model import ChatMessageAssistant, ModelUsage
from inspect_ai.tool import ToolCall
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .._context import InspectSample
from .._report import Decision, Failed, Reported
from ._host import ModelUsed, Record, Stage
from ._wire import UnreadableError

INSPECT_CONTEXT = "x-inspect-context"
"""The header an Inspect eval names the sample of each model request in."""


class _Header(BaseModel):
    model_config = ConfigDict(strict=True)

    version: Literal[1]
    eval_id: str
    task: str
    sample_id: str | int
    epoch: int
    sample_uuid: str
    attempt: str
    attempt_number: int
    model: str | None = None


def inspect_sample(headers: Mapping[str, str]) -> InspectSample | None:
    """The sample an Inspect eval says a request is from, or None if the request has no `x-inspect-context`.

    Args:
        headers: The request's headers, by lowercase name.

    Raises:
        UnreadableError: If the header is there and isn't what Inspect writes, so the request's run can't be told.
    """
    value = headers.get(INSPECT_CONTEXT)
    if value is None:
        return None
    try:
        header = _Header.model_validate_json(value)
    except ValidationError:
        raise UnreadableError(
            f"The request's {INSPECT_CONTEXT} header isn't one this sidecar can read."
        ) from None
    return InspectSample(**header.model_dump(exclude={"version"}))


@dataclass
class Judged:
    """One step judged during an exchange, for the eval's judgment of it."""

    stage: Stage
    call: str | None
    """The id of the tool call, at a tool stage."""
    decision: Decision | None
    """The step's final decision, or None if no protocol decided."""
    records: Sequence[Record]
    """What was recorded while the step was judged."""
    models: Sequence[ModelUsed] = ()
    """The models the step's monitors and protocols called."""


@dataclass(frozen=True)
class Withheld:
    """A turn the client was refused, or handed with calls taken out or changed, which the sidecar adds to or puts back in the later requests of its conversation."""

    message: ChatMessageAssistant
    """The model's turn, as it proposed it."""
    told: Mapping[str, str]
    """What the model is told happened to each call taken out, by call id."""
    redacted: frozenset[str] = frozenset()
    """The calls a decision acted on, by id, whose arguments an eval is never handed: an eval holds each with its id and function only, and the sidecar puts the model's own call back in each later request."""
    ran: Mapping[str, ToolCall] = field(default_factory=dict[str, ToolCall])
    """The changed calls, as they ran, by id: what an eval holds in place of the model's own."""


@dataclass
class Exchange:
    """What was decided during one exchange, for the eval's judgment of it."""

    judged: list[Judged] = field(default_factory=list[Judged])
    """The steps judged, in order."""
    withheld: Withheld | None = None
    """The turn the client was refused, if it was."""
    proposed: Withheld | None = None
    """The turn as the model proposed it, if the client was handed it with calls taken out."""
    message: str | None = None
    """What the model is told of the rejection, if the turn was refused for one."""
    ended: bool = False
    """Whether the reply was refused because the run ended, by a sentinel's decision while it was judged or before, rather than for a rejection."""
    replaced: bool = False
    """Whether the request or reply was refused unjudged because it is of an attempt at the sample that another replaced, or of another attempt with the same number: not a sentinel's decision, and nobody waits for it."""
    usage: ModelUsage | None = None
    """What the provider billed for the model's reply, for a client that is refused it."""


class SentinelModel(BaseModel):
    """A model a monitor or protocol called while a step was judged."""

    role: str | None = Field(
        default=None,
        description="The model role the call asked for, as Inspect names roles. Left out where the call named its model without a role.",
    )
    model: str = Field(description="The model the call went to, as Inspect names it.")


class SentinelStep(BaseModel):
    """One step judged during an exchange."""

    stage: Stage = Field(
        description="Where the step was judged, as Inspect's sentinel events name it."
    )
    call: str | None = Field(
        default=None, description="The id of the tool call, at a tool stage."
    )
    decision: dict[str, Any] = Field(
        description="The step's final decision, as the core package writes a `Decision`."
    )
    recorded: list[dict[str, Any]] | None = Field(
        default=None,
        description="What each monitor and protocol recorded for the step. Left out of a trimmed field.",
    )
    models: list[SentinelModel] | None = Field(
        default=None,
        description="The models the step's monitors and protocols called, each once, in the order first called. The sidecar takes a role's model from its own configuration, not from the eval's. Left out where none was called, and of a trimmed field.",
    )


class WithheldTurn(BaseModel):
    """The model's turn, which the client was refused or handed with calls taken out or changed."""

    message: dict[str, Any] = Field(
        description="The model's message, as Inspect writes a `ChatMessageAssistant`, every call in it included, each redacted call with its id and function only, and each changed call as it ran."
    )
    results: dict[str, str] = Field(
        description="What the model is told happened to each call taken out, by call id. The sidecar adds the turn and these results to each later request of the conversation that doesn't already hold them."
    )
    redacted: list[str] = Field(
        default_factory=list[str],
        description="The calls a decision acted on, by id, whose arguments an eval is never handed. Each is in `message` with no arguments and no view. An eval holds it that way, and the sidecar puts the model's own call back in each later request of the conversation.",
    )


class Judgment(BaseModel):
    """What a sentinel decided during one exchange, for an Inspect eval, which asks for it with `request_result`."""

    version: Literal[1] = Field(default=1, description="The version of the judgment.")
    outcome: Literal["proceed", "reject", "terminate"] = Field(
        description="What the exchange came to: `proceed` when the reply goes on, `reject` when the model's turn was refused and the run goes on, and `terminate` when the run has ended, in this exchange or before it."
    )
    trimmed: bool | None = Field(
        default=None,
        description="True when the sidecar was told not to send results, so each step holds only what Inspect acts on: its stage, its call, and its decision's action, message and modified call.",
    )
    steps: list[SentinelStep] = Field(
        description="Every step judged during the exchange, in the order they were judged. A step that ended the run is the last. Empty when the run had already ended."
    )
    withheld: WithheldTurn | None = Field(
        default=None,
        description="The model's turn, when the client was refused it because a call in it was rejected, or because a sentinel ended the run at the reply or at a call in it.",
    )
    proposed: WithheldTurn | None = Field(
        default=None,
        description="The model's turn, when the client was handed it with calls taken out because they were rejected, or changed. Each call taken out is redacted, each changed call is as it ran, and `results` holds what the model is told of each call taken out. The sidecar puts those calls and results back in each later request of the conversation that doesn't already hold them.",
    )
    usage: dict[str, Any] | None = Field(
        default=None,
        description="What the provider billed for the model's reply, as Inspect writes `ModelUsage`, when the client was refused a reply the model gave. A refusal carries no usage of its own, so the client counts it from here.",
    )


def judgment(
    exchange: Exchange,
    outcome: Literal["proceed", "reject", "terminate"],
    *,
    trimmed: bool,
) -> Judgment:
    """What was decided during an exchange, for the eval.

    Args:
        exchange: What was decided during the exchange.
        outcome: What the exchange came to.
        trimmed: Whether to keep only what Inspect acts on, leaving out explanations, references, metadata and records.
    """
    steps: list[SentinelStep] = []
    for step in exchange.judged:
        decision = step.decision if step.decision is not None else Decision.proceed()
        steps.append(
            SentinelStep(
                stage=step.stage,
                call=step.call,
                decision=decision.model_dump(
                    mode="json",
                    exclude_defaults=True,
                    include={"action", "message", "modified"} if trimmed else None,
                ),
                recorded=None if trimmed else [record_json(r) for r in step.records],
                models=None
                if trimmed or not step.models
                else [SentinelModel(role=m.role, model=m.model) for m in step.models],
            )
        )
    withheld, proposed = exchange.withheld, exchange.proposed
    return Judgment(
        outcome=outcome,
        trimmed=True if trimmed else None,
        steps=steps,
        withheld=withheld_turn(withheld) if withheld is not None else None,
        proposed=withheld_turn(proposed) if proposed is not None else None,
        # a reply that goes on carries its own usage
        usage=(
            exchange.usage.model_dump(mode="json", exclude_none=True)
            if exchange.usage is not None and outcome != "proceed"
            else None
        ),
    )


def withheld_turn(withheld: Withheld) -> WithheldTurn:
    """A turn as an eval is handed it: each redacted call with its id and function only, and each changed call as it ran."""
    calls = [
        redacted(call)
        if call.id in withheld.redacted
        else withheld.ran.get(call.id, call)
        for call in withheld.message.tool_calls or []
    ]
    message = withheld.message.model_copy(
        update={"tool_calls": calls} if withheld.message.tool_calls else {}
    )
    return WithheldTurn(
        message=message.model_dump(mode="json", exclude_none=True),
        results=dict(withheld.told),
        redacted=[call.id for call in calls if call.id in withheld.redacted],
    )


def redacted(call: ToolCall) -> ToolCall:
    """A call as an eval holds one a decision acted on: its id and function, with no arguments and nothing made from them."""
    return replace(call, arguments={}, parse_error=None, view=None)


def record_json(record: Record) -> dict[str, Any]:
    """A record, as the endpoint for evals writes it."""
    written: dict[str, Any] = {
        "kind": record.kind,
        "path": record.path,
        "factory": record.factory,
    }
    detail = record.detail
    if isinstance(detail, Reported):
        written["name"] = detail.name
        written["function"] = detail.function
        written["report"] = detail.report.model_dump(mode="json", exclude_defaults=True)
    elif isinstance(detail, Failed):
        written["name"] = detail.name
        written["function"] = detail.function
        written["error"] = f"{type(detail.error).__name__}: {detail.error}"
    else:
        written["name"] = detail
    if record.call is not None:
        written["call"] = record.call
    return written
