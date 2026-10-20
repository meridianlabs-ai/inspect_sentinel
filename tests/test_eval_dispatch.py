import logging
import re
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai._sentinel._config import resolve_sentinel_root, resolve_sentinel_spec
from inspect_ai._sentinel._context import init_sentinel
from inspect_ai.dataset import Sample
from inspect_ai.event import (
    Event,
    ModelEvent,
    SampleLimitEvent,
    SentinelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.log import EvalLog
from inspect_ai.log._condense import resolve_sample_attachments
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    get_model,
)
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.scorer import Reference
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, ToolCall, tool
from inspect_ai.util import StoreModel

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    Monitor,
    MonitorGroup,
    Observation,
    Protocol,
    ProtocolGroup,
    concurrent,
    decide_final,
    handle_escalation,
    monitor,
    observe_only,
    protocol,
    sequential,
    threshold,
)


@tool
def addition() -> Tool:
    async def execute(x: int, y: int) -> str:
        """Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.
        """
        return str(x + y)

    return execute


@protocol
def d3_terminate() -> Protocol:
    async def stop(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.terminate("too risky")

    return stop


@protocol
def d3_escalate() -> Protocol:
    async def unsure(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.escalate("not sure")

    return unsure


@protocol
def d3_escalate_after() -> Protocol:
    async def unsure(
        context: Context,
        step: AfterToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.escalate(f"unsure about {step.result.text}")

    return unsure


@protocol
def d3_continue() -> Protocol:
    async def fine(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.proceed()

    return fine


@protocol
def d3_final() -> Protocol:
    async def veto(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        decide_final(Decision.reject(message="vetoed"))
        return None

    return veto


@protocol
def d3_final_escalate() -> Protocol:
    async def unsure(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        decide_final(Decision.escalate("not sure"))
        return None

    return unsure


@monitor
def d3_suspicion(score: float = 0.25) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(score, "looked")

    return check


_CITE = Reference(type="message", id="msg_2", cite="[M2]")


@monitor
def d3_cites() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.3, "see [M2]", references=[_CITE])

    return check


@protocol
def d3_cites_reject() -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.reject("see [M2]", references=[_CITE])

    return decide


class Trajectory(StoreModel):
    calls: int = 0


@monitor
def d3_trajectory() -> MonitorGroup:
    async def count(context: Context, step: BeforeToolCall) -> Observation | None:
        context.store_as(Trajectory).calls += 1
        return None

    async def report(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(float(context.store_as(Trajectory).calls))

    return MonitorGroup(count, report)


@monitor
def d3_scores() -> Monitor:
    async def score(context: Context, step: BeforeToolCall) -> Observation | None:
        return await context.host.score(step, "Is this call safe?")

    return score


@monitor
def d3_raises() -> Monitor:
    async def boom(context: Context, step: BeforeToolCall) -> Observation | None:
        raise RuntimeError("monitor exploded")

    return boom


def agent_model(turns: int = 1) -> Any:
    outputs = [
        ModelOutput.for_tool_call(
            "mockllm/model",
            tool_name="addition",
            tool_arguments={"x": 1, "y": turn + 1},
        )
        for turn in range(turns)
    ]
    outputs.append(ModelOutput.from_content("mockllm/model", content="done"))
    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


def run(
    sentinel: Any,
    turns: int = 1,
    task_description: str | None = None,
    sample_description: str | None = None,
    **kwargs: Any,
) -> EvalLog:
    task = Task(
        dataset=[
            Sample(
                input="What is 1 + 1?",
                target="2",
                metadata={"s": 1},
                description=sample_description,
            )
        ],
        solver=[use_tools(addition()), generate()],
        metadata={"t": 1},
        sentinel=sentinel,
        description=task_description,
    )
    return eval(task, model=agent_model(turns), **kwargs)[0]


def sentinel_events(log: EvalLog) -> list[SentinelEvent]:
    assert log.samples
    return [e for e in log.samples[0].events if isinstance(e, SentinelEvent)]


def tool_messages(log: EvalLog) -> list[ChatMessageTool]:
    assert log.samples
    return [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]


def summary(events: list[SentinelEvent]) -> list[tuple[Any, ...]]:
    return [(e.factory, e.path, e.function, e.kind, e.status, e.action) for e in events]


def test_observe_only_records_observations_without_effect() -> None:
    log = run(observe_only([d3_suspicion(score=0.4)]))
    assert log.status == "success", log.error

    [message] = tool_messages(log)
    assert message.error is None
    assert message.text == "2"
    [event] = sentinel_events(log)
    assert (
        event.factory,
        event.path,
        event.function,
        event.kind,
        event.status,
    ) == ("d3_suspicion", "d3_suspicion", "check", "observation", "reported")
    assert event.suspicion == 0.4
    assert event.explanation == "looked"
    assert event.action is None


def test_references_are_recorded_from_each_report() -> None:
    log = run({"watch": d3_cites(), "block": d3_cites_reject()})
    assert log.status == "success", log.error
    events = {e.path: e for e in sentinel_events(log)}
    assert {path: e.references for path, e in events.items()} == {
        "watch": [_CITE],
        "block": [_CITE],
        "": [_CITE],
    }


def test_final_from_a_nested_protocol() -> None:
    log = run(concurrent({"inner": d3_final()}))
    assert log.status == "success", log.error

    [message] = tool_messages(log)
    assert message.error is not None
    assert message.error.message == "vetoed"
    assert summary(sentinel_events(log)) == [
        ("inspect_sentinel/concurrent", "", None, "decision", "bypassed", None),
        ("d3_final", "inner", "veto", "decision", "reported", "reject"),
    ]
    assert all(e.references == [] for e in sentinel_events(log))


def assert_unhandled_escalation(
    log: EvalLog, stage: str, source: str, recorded_at: str = ""
) -> None:
    assert log.status == "success", log.error
    assert log.samples
    sample = log.samples[0]
    assert sample.error is None
    assert sample.limit is not None
    assert sample.limit.type == "operator"
    reason = sample.limit.reason
    assert reason is not None
    assert reason.startswith(f"unhandled escalation from {source}: ")
    assert "handle_escalation(" in reason
    [limit_event] = [e for e in sample.events if isinstance(e, SampleLimitEvent)]
    assert limit_event.message == reason
    assert (recorded_at, stage, "reported", "escalate") in [
        (e.path, e.stage, e.status, e.action) for e in sentinel_events(log)
    ]


def test_an_unhandled_escalate_before_the_call_terminates() -> None:
    log = run({"unsure": d3_escalate(), "fine": d3_continue()}, turns=2)
    assert_unhandled_escalation(log, "tool_call", source="unsure")
    assert tool_messages(log) == []
    assert log.samples
    tool_events = [e for e in log.samples[0].events if isinstance(e, ToolEvent)]
    assert [e.failed for e in tool_events] == [True]


def test_an_unhandled_escalate_after_the_call_terminates() -> None:
    log = run(d3_escalate_after(), turns=2)
    assert_unhandled_escalation(log, "tool_result", source="d3_escalate_after")
    assert log.samples
    tool_events = [e for e in log.samples[0].events if isinstance(e, ToolEvent)]
    assert [(e.result, e.failed) for e in tool_events] == [("2", True)]


def test_an_unhandled_final_escalate_terminates() -> None:
    log = run(sequential({"veto": d3_final_escalate(), "fine": d3_continue()}))
    assert_unhandled_escalation(log, "tool_call", source="veto", recorded_at="veto")
    [root] = [e for e in sentinel_events(log) if e.path == ""]
    assert (root.status, root.action) == ("bypassed", None)


def test_an_unhandled_escalate_is_told_apart_from_a_terminate() -> None:
    terminated = run(d3_terminate())
    escalated = run(d3_escalate())
    assert terminated.samples and escalated.samples
    assert terminated.samples[0].limit is not None
    assert terminated.samples[0].limit.reason == "too risky"
    assert [e.action for e in sentinel_events(terminated)] == ["terminate"]
    assert [e.action for e in sentinel_events(escalated)] == ["escalate"]
    assert_unhandled_escalation(escalated, "tool_call", source="d3_escalate")


def test_an_escalate_handled_by_a_terminating_link_uses_its_reason() -> None:
    log = run(sequential([d3_escalate(), d3_terminate()]))
    assert log.samples
    sample = log.samples[0]
    assert sample.limit is not None
    assert sample.limit.type == "operator"
    assert sample.limit.reason == "too risky"
    root = [e for e in sentinel_events(log) if e.path == ""]
    assert [e.action for e in root] == ["terminate"]


def test_an_escalate_handled_by_a_continuing_link_proceeds() -> None:
    log = run(sequential([d3_escalate(), d3_continue()]), turns=2)
    assert log.status == "success", log.error
    assert log.samples and log.samples[0].limit is None
    assert [m.text for m in tool_messages(log)] == ["2", "3"]
    root = [e for e in sentinel_events(log) if e.path == ""]
    assert [e.action for e in root] == ["proceed", "proceed"]


def test_multi_function_monitor_shares_state_across_calls() -> None:
    log = run(observe_only([d3_trajectory()]), turns=2)
    assert log.status == "success", log.error

    events = sentinel_events(log)
    assert [(e.function, e.stage, e.suspicion) for e in events] == [
        ("report", "tool_result", 1.0),
        ("report", "tool_result", 2.0),
    ]
    assert log.samples
    assert log.samples[0].store["Trajectory:d3_trajectory:calls"] == 2


def test_host_score_asks_the_monitor_role_with_the_task() -> None:
    monitor_model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                "mockllm/model",
                tool_name="answer",
                tool_arguments={"explanation": "very", "suspicion": 1.5},
            ),
            ModelOutput.for_tool_call(
                "mockllm/model",
                tool_name="answer",
                tool_arguments={"explanation": "just arithmetic", "suspicion": 0.25},
            ),
        ],
        memoize=False,
    )
    log = run(observe_only([d3_scores()]), model_roles={"monitor": monitor_model})
    assert log.status == "success", log.error

    [event] = sentinel_events(log)
    assert (event.suspicion, event.explanation) == (0.25, "just arithmetic")
    assert log.samples
    [asked, corrected] = [
        e
        for e in resolve_sample_attachments(log.samples[0], "full").events
        if isinstance(e, ModelEvent) and e.role == "monitor"
    ]
    assert [tool.name for tool in asked.tools] == ["answer"]
    assert isinstance(corrected.input[-1], ChatMessageTool)
    assert corrected.input[-1].error is not None
    prompt = asked.input[0].text
    # Scout 0.6.0 and earlier write no nonce on the tool call line
    assert re.search(r"^Tool Call( \([0-9a-f]{8}\))?: addition$", prompt, re.MULTILINE)


def sentinel_span_ids(events: Sequence[Event]) -> list[str]:
    return [
        e.id for e in events if isinstance(e, SpanBeginEvent) and e.type == "sentinel"
    ]


def test_a_raising_monitor_under_observe_only_is_recorded_and_the_sample_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        log = run(observe_only([d3_raises(), d3_suspicion()]))
    assert log.status == "success", log.error
    [message] = tool_messages(log)
    assert message.text == "2"
    failed, observed = sentinel_events(log)
    assert (failed.path, failed.function, failed.kind, failed.status) == (
        "d3_raises",
        "boom",
        "observation",
        "error",
    )
    assert failed.error == "RuntimeError: monitor exploded"
    assert failed.suspicion is None
    assert observed.status == "reported"
    assert any(
        "Monitor 'd3_raises'" in r.getMessage() and "monitor exploded" in r.getMessage()
        for r in caplog.records
    )


def test_a_raising_monitor_under_threshold_fails_the_sample() -> None:
    log = run(threshold(d3_raises(), reject_at=0.5))
    assert log.status == "error"
    assert log.samples
    error = log.samples[0].error
    assert error is not None
    assert "MonitorFailedError" in error.traceback
    assert "'d3_raises' (function 'boom'): RuntimeError: monitor exploded" in (
        error.message
    )
    assert tool_messages(log) == []
    [failed] = sentinel_events(log)
    assert (failed.path, failed.status, failed.error) == (
        "d3_raises",
        "error",
        "RuntimeError: monitor exploded",
    )


@contextmanager
def active(sentinel: Any) -> Generator[None]:
    init_sentinel(resolve_sentinel_root(resolve_sentinel_spec(sentinel)))
    try:
        yield
    finally:
        init_sentinel(None)


def addition_call(id: str = "call") -> ToolCall:
    return ToolCall(id=id, function="addition", arguments={"x": 1, "y": 1})


def transcript_tool_events() -> list[ToolEvent]:
    return [e for e in transcript().events if isinstance(e, ToolEvent)]


@monitor
def d3_waiting(started: Any, cleaned_up: Any, after: bool = False) -> MonitorGroup:
    async def wait() -> None:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            cleaned_up.set()

    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        if not after:
            await wait()
        return None

    async def later(context: Context, step: AfterToolCall) -> Observation | None:
        if after:
            await wait()
        return None

    return MonitorGroup(before, later)


@pytest.mark.anyio
async def test_sample_cancellation_during_the_sentinel_propagates() -> None:
    init_transcript(Transcript())
    started = anyio.Event()
    cleaned_up = anyio.Event()

    with active(observe_only([d3_waiting(started, cleaned_up)])):
        with anyio.CancelScope() as scope:

            async def cancel_sample() -> None:
                await started.wait()
                scope.cancel()

            async with anyio.create_task_group() as tg:
                tg.start_soon(cancel_sample)
                await execute_tools(
                    [ChatMessageAssistant(content=[], tool_calls=[addition_call()])],
                    [addition()],
                )

    assert scope.cancelled_caught
    assert cleaned_up.is_set()
    events = [e for e in transcript().events if isinstance(e, SentinelEvent)]
    assert [(e.path, e.kind, e.status) for e in events] == [
        ("d3_waiting", "observation", "cancelled"),
        ("", "decision", "cancelled"),
    ]
    assert transcript_tool_events() == []
    [span_id] = sentinel_span_ids(transcript().events)
    assert {e.span_id for e in events} == {span_id}
    assert any(
        isinstance(e, SpanEndEvent) and e.id == span_id for e in transcript().events
    )


def test_sample_time_limit_during_the_sentinel_ends_the_sample() -> None:
    started = anyio.Event()
    cleaned_up = anyio.Event()
    log = run(observe_only([d3_waiting(started, cleaned_up)]), time_limit=1)
    assert log.status == "success", log.error
    assert log.samples
    sample = log.samples[0]
    assert sample.limit is not None
    assert sample.limit.type == "time"
    assert cleaned_up.is_set()
    assert [(e.path, e.kind, e.status) for e in sentinel_events(log)] == [
        ("d3_waiting", "observation", "cancelled"),
        ("", "decision", "cancelled"),
    ]
    assert tool_messages(log) == []


@protocol
def d3_generate_decides(
    action: Any, after: bool, any_output: bool = False
) -> ProtocolGroup:
    async def before_generate(
        context: Context,
        step: BeforeGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return None if after else Decision(action=action, explanation="no generate")

    async def after_generate(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        if after and (any_output or step.output.message.tool_calls):
            return Decision(action=action, explanation="no tools")
        return None

    return ProtocolGroup(before_generate, after_generate)


@pytest.mark.parametrize("after", [False, True])
def test_reject_at_a_generate_stage_fails_the_sample(after: bool) -> None:
    log = run(d3_generate_decides("reject", after))
    assert log.status == "error"
    assert log.samples
    error = log.samples[0].error
    assert error is not None
    assert "only 'proceed', 'escalate' and 'terminate' are supported" in error.message


@pytest.mark.parametrize("after", [False, True])
def test_an_unhandled_escalate_at_a_generate_stage_ends_the_sample(
    after: bool,
) -> None:
    log = run(d3_generate_decides("escalate", after))
    assert log.status == "success", log.error
    assert log.samples
    limit = log.samples[0].limit
    assert limit is not None and limit.type == "operator"
    assert limit.reason is not None
    assert limit.reason.startswith("unhandled escalation from")
    assert not [e for e in log.samples[0].events if isinstance(e, ToolEvent)]
    stage = "model_output" if after else "model_input"
    assert (stage, "escalate") in {
        (e.stage, e.action) for e in sentinel_events(log) if e.path == ""
    }


@pytest.mark.parametrize("after", [False, True])
def test_a_handled_escalate_at_a_generate_stage_proceeds(after: bool) -> None:
    log = run(
        sequential(
            [d3_generate_decides("escalate", after), handle_escalation("proceed")]
        )
    )
    assert log.status == "success", log.error
    assert log.samples
    assert log.samples[0].limit is None
    assert [m.text for m in tool_messages(log)] == ["2"]


@protocol
def d3_returns_a_configured_root() -> Protocol:
    return sequential(
        [threshold(d3_suspicion(0.9), escalate_at=0.5), handle_escalation("proceed")]
    )


@pytest.mark.parametrize(
    "configure",
    [
        lambda: sequential(
            [
                threshold(d3_suspicion(0.9), escalate_at=0.5),
                handle_escalation("proceed"),
            ]
        ),
        lambda: d3_returns_a_configured_root(),
    ],
)
def test_a_tool_only_sentinel_opens_no_generate_stage(configure: Any) -> None:
    log = run(configure())
    assert log.status == "success", log.error
    assert log.samples
    assert {e.stage for e in sentinel_events(log)} == {"tool_call"}
    assert len(sentinel_span_ids(log.samples[0].events)) == 1


@monitor
def d3_generate_raises() -> Monitor:
    async def check(context: Context, step: BeforeGenerate) -> Observation | None:
        raise ValueError("monitor broke")

    return check


def test_a_failing_generate_monitor_is_recorded_and_the_sample_continues() -> None:
    log = run(observe_only([d3_generate_raises()]))
    assert log.status == "success", log.error
    failures = [e for e in sentinel_events(log) if e.status == "error"]
    assert failures
    for failed in failures:
        assert (failed.stage, failed.path, failed.function) == (
            "model_input",
            "d3_generate_raises",
            "check",
        )
        assert failed.error == "ValueError: monitor broke"
        assert failed.step_id
