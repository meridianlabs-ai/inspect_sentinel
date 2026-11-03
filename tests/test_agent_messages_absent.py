from collections.abc import Sequence
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
from inspect_sentinel._types import MonitorGroup
from tests._fakes import (
    FakeHost,
    ListRecorder,
    after_generate_step,
    after_step,
    before_generate_step,
    before_step,
    host_context,
)


def _evidence(history: Sequence[ChatMessage] | None) -> str:
    # the None-safe pattern: say the history is absent rather than judge an empty one
    if history is None:
        return "history unavailable"
    return f"{len(history)} messages of history"


def _every_stage(value: float) -> MonitorGroup:
    async def tool_call(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value, _evidence(step.agent_messages))

    async def tool_result(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(value, _evidence(step.agent_messages))

    async def model_input(context: Context, step: BeforeGenerate) -> Observation | None:
        return Observation.score(value, _evidence(step.agent_messages))

    async def model_output(context: Context, step: AfterGenerate) -> Observation | None:
        return Observation.score(value, _evidence(step.agent_messages))

    return MonitorGroup(tool_call, tool_result, model_input, model_output)


@monitor
def ha_suspicious() -> MonitorGroup:
    return _every_stage(0.9)


@monitor
def ha_calm() -> MonitorGroup:
    return _every_stage(0.1)


def _every_builtin() -> Sentinels:
    return sequential(
        [
            concurrent(
                [
                    threshold(ha_suspicious(), escalate_at=0.5),
                    observe_only(ha_calm()),
                ]
            ),
            human(stages=["tool_call", "tool_result"]),
            handle_escalation("terminate"),
        ]
    )


# every built-in protocol at each stage, on a step whose host could not provide
# history: the escalation reaches human() at the tool stages and
# handle_escalation() at the generate stages
@pytest.mark.parametrize(
    ("step", "action", "asked"),
    [
        (replace(before_step(), agent_messages=None), "proceed", True),
        (replace(after_step(), agent_messages=None), "proceed", True),
        (replace(before_generate_step(), agent_messages=None), "terminate", False),
        (replace(after_generate_step(), agent_messages=None), "terminate", False),
    ],
    ids=["tool_call", "tool_result", "model_input", "model_output"],
)
@pytest.mark.anyio
async def test_builtins_run_without_agent_messages(
    step: Step, action: Action, asked: bool
) -> None:
    host = FakeHost(*([HumanAnswer(decision="approve")] if asked else []))
    recorder = ListRecorder()
    decision = await run_sentinel(
        resolve_sentinel(_every_builtin()),
        host_context(recorder=recorder, host=host),
        step,
    )
    assert decision is not None and decision.action == action
    assert [a.step.agent_messages for a in host.asked] == ([None] if asked else [])
    assert recorder.failures == []
    observed = [
        r.reported.report
        for r in recorder.records
        if isinstance(r.reported.report, Observation)
    ]
    assert len(observed) == 2
    assert all(o.explanation == "history unavailable" for o in observed)
