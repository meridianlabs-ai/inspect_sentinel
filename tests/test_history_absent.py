from dataclasses import replace

import pytest
from inspect_ai.model import ChatMessage

from inspect_sentinel import HumanAnswer
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor
from inspect_sentinel._integration import Sentinels, resolve_sentinel
from inspect_sentinel._protocols import (
    concurrent,
    handle_escalation,
    human,
    observe_only,
    sequential,
    threshold,
)
from inspect_sentinel._report import Action, Observation
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Monitor, MonitorGroup
from tests._fakes import (
    FakeHost,
    ListRecorder,
    after_generate_step,
    after_step,
    before_generate_step,
    before_step,
    host_context,
)


def _evidence(history: list[ChatMessage] | None) -> str:
    # the None-safe pattern: say the history is absent rather than judge an empty one
    if history is None:
        return "history unavailable"
    return f"{len(history)} messages of history"


@monitor
def ha_every_stage() -> MonitorGroup:
    async def tool_call(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.9, _evidence(step.history))

    async def tool_result(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.9, _evidence(step.history))

    async def model_input(context: Context, step: BeforeGenerate) -> Observation | None:
        return Observation.score(0.9, _evidence(step.history))

    async def model_output(context: Context, step: AfterGenerate) -> Observation | None:
        return Observation.score(0.9, _evidence(step.history))

    return MonitorGroup(tool_call, tool_result, model_input, model_output)


@monitor
def ha_before_call() -> Monitor:
    async def tool_call(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.9, _evidence(step.history))

    return tool_call


_STEPS: list[Step] = [
    replace(before_step(), history=None),
    replace(after_step(), history=None),
    replace(before_generate_step(), history=None),
    replace(after_generate_step(), history=None),
]


def _escalate_then_human() -> Sentinels:
    return sequential(
        [
            threshold(ha_every_stage(), escalate_at=0.5),
            human(stages=["tool_call", "tool_result"]),
            handle_escalation("terminate"),
        ]
    )


def _escalate_then_handle() -> Sentinels:
    return sequential(
        [threshold(ha_every_stage(), escalate_at=0.5), handle_escalation("proceed")]
    )


def _concurrent_reject() -> Sentinels:
    return concurrent(
        [threshold(ha_before_call(), reject_at=0.5), observe_only(ha_every_stage())]
    )


# each built-in protocol at a step whose host could not provide history:
# (sentinel, step, the decision, whether a person was asked)
_TABLE: list[tuple[Sentinels, Step, Action | None, bool]] = [
    (_escalate_then_human(), _STEPS[0], "proceed", True),
    (_escalate_then_human(), _STEPS[1], "proceed", True),
    (_escalate_then_human(), _STEPS[2], "terminate", False),
    (_escalate_then_human(), _STEPS[3], "terminate", False),
    *((_escalate_then_handle(), step, "proceed", False) for step in _STEPS),
    (_concurrent_reject(), _STEPS[0], "reject", False),
    *((observe_only(ha_every_stage()), step, None, False) for step in _STEPS),
]


@pytest.mark.parametrize(("sentinel", "step", "action", "asked"), _TABLE)
@pytest.mark.anyio
async def test_builtins_run_without_history(
    sentinel: Sentinels, step: Step, action: Action | None, asked: bool
) -> None:
    host = FakeHost(*([HumanAnswer(decision="approve")] if asked else []))
    recorder = ListRecorder()
    decision = await run_sentinel(
        resolve_sentinel(sentinel), host_context(recorder=recorder, host=host), step
    )
    assert (decision.action if decision else None) == action
    assert [a.step.history for a in host.asked] == ([None] if asked else [])
    assert recorder.failures == []
    observed = [
        r.reported.report
        for r in recorder.records
        if isinstance(r.reported.report, Observation)
    ]
    assert observed
    assert all(o.explanation == "history unavailable" for o in observed)
