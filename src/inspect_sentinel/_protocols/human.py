from __future__ import annotations

from collections.abc import Sequence

from .._context import Context, HumanAnswer
from .._decorators import protocol
from .._final import decide_final
from .._report import Decision
from .._step import AfterToolCall, BeforeToolCall, Step
from .._types import Protocol, ProtocolGroup

_LEGAL = {
    "tool_call": ("approve", "reject", "terminate", "modify"),
    "tool_result": ("approve", "terminate"),
}
_DEFAULTS = {
    "tool_call": ("approve", "reject", "terminate"),
    "tool_result": ("approve", "terminate"),
}


@protocol
def human(
    stages: Sequence[str],
    choices: Sequence[str] | None = None,
    final: bool = True,
) -> Protocol | ProtocolGroup:
    """Ask a person to decide, every time the step reaches this protocol.

    The person is asked through `Host.ask_human`, which is handed the step and shows it with `step.escalations`, so at the end of a `sequential()` the person sees who escalated and why. On its own, `human(stages=["tool_call"])` asks about every call. The answer maps to a decision: `approve` to `continue`, `reject` to `reject` with the person's reason as both the `message` the agent reads and the `explanation`, `terminate` to `terminate` explained by the reason, and `modify` to a `modify` with the person's replacement call.

    A `human()` directly beside other children of a `concurrent()` is a configuration error: the person would be asked about every call while the others decide in parallel. Put it at the end of a `sequential()` instead.

    Args:
        stages: The stages to ask at, as `SentinelEvent.stage` records them: `"tool_call"` (before a tool call) and/or `"tool_result"` (after it, with the result the model is about to receive).
        choices: What the person may pick, from `approve`, `reject`, `terminate` and `modify`. Defaults to `approve`, `reject` and `terminate` before a call and `approve` and `terminate` after it, where `reject` and `modify` are not legal.
        final: End the step with the person's decision through `decide_final()`, so no layer above may weaken it. `False` returns it as an ordinary decision, making the person one vote in a panel.
    """
    valid = ", ".join(repr(stage) for stage in _LEGAL)
    if isinstance(stages, str) or not stages:
        raise ValueError(f"human needs a list of stages, from {valid}; got {stages!r}.")
    for stage in stages:
        if stage not in _LEGAL:
            raise ValueError(
                f"human cannot ask at stage {stage!r}; the stages it supports are {valid}."
            )
    if len(set(stages)) != len(stages):
        raise ValueError(f"human was given a stage twice: {list(stages)!r}.")
    offered = {stage: _choices(stage, choices) for stage in stages}

    async def ask(context: Context, step: Step, stage: str) -> Decision:
        answer = await context.host.ask_human(step, offered[stage])
        decision = _decision(answer, offered[stage])
        if final:
            decide_final(decision)
        return decision

    async def tool_call(context: Context, step: BeforeToolCall) -> Decision | None:
        return await ask(context, step, "tool_call")

    async def tool_result(context: Context, step: AfterToolCall) -> Decision | None:
        return await ask(context, step, "tool_result")

    functions: dict[str, Protocol] = {
        "tool_call": tool_call,
        "tool_result": tool_result,
    }
    if len(stages) == 1:
        return functions[stages[0]]
    return ProtocolGroup(*(functions[stage] for stage in stages))


def _choices(stage: str, choices: Sequence[str] | None) -> tuple[str, ...]:
    if choices is None:
        return _DEFAULTS[stage]
    if isinstance(choices, str) or not choices:
        raise ValueError(
            f"human needs at least one choice, from {', '.join(repr(c) for c in _LEGAL['tool_call'])}; got {choices!r}."
        )
    for choice in choices:
        if choice not in _LEGAL["tool_call"]:
            raise ValueError(
                f"human cannot offer {choice!r}; the choices are {', '.join(repr(c) for c in _LEGAL['tool_call'])}."
            )
        if choice not in _LEGAL[stage]:
            raise ValueError(
                f"human cannot offer {choice!r} after a tool call, since the call has already run; the choices there are {', '.join(repr(c) for c in _LEGAL[stage])}."
            )
    return tuple(choices)


def _decision(answer: HumanAnswer, choices: tuple[str, ...]) -> Decision:
    if answer.decision not in choices:
        raise ValueError(
            f"The host answered {answer.decision!r}, which is not one of the choices offered: {list(choices)!r}."
        )
    if answer.modified is not None and answer.decision != "modify":
        raise ValueError(
            f"The host answered {answer.decision!r} with a modified call; only a 'modify' carries one."
        )
    reason = answer.reason
    match answer.decision:
        case "approve":
            return Decision.proceed()
        case "reject":
            return Decision.reject(reason, message=reason)
        case "terminate":
            return Decision.terminate(reason)
        case _:
            return Decision(action="modify", modified=answer.modified)
