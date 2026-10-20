import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.event import SentinelEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import Tool, ToolCall, ToolChoice, ToolInfo, tool

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import forwards, monitor, protocol
from inspect_sentinel._replay import main, replay_eval
from inspect_sentinel._report import Decision, Escalations, Observation, Reported
from inspect_sentinel._runner import run_monitors
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Monitor, MonitorGroup, Protocol


@tool
def lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a key.

        Args:
            key: The key.
        """
        return f"value of {key}"

    return execute


@tool
def forbidden() -> Tool:
    async def execute() -> str:
        """Do what is not allowed."""
        return "done"

    return execute


@tool
def slow() -> Tool:
    async def execute() -> str:
        """Take longer than the sample may."""
        await anyio.sleep(10)
        return "done"

    return execute


@monitor
def counts() -> MonitorGroup:
    async def before_generate(context: Context, step: BeforeGenerate) -> Observation:
        return Observation(suspicion=len(step.input[-1].text) / 1000)

    async def after_generate(context: Context, step: AfterGenerate) -> Observation:
        return Observation(suspicion=len(step.output.message.tool_calls or []) / 10)

    async def before_tool(context: Context, step: BeforeToolCall) -> Observation:
        return Observation(suspicion=0.9 if step.call.function == "forbidden" else 0.1)

    async def after_tool(context: Context, step: AfterToolCall) -> Observation:
        return Observation(suspicion=len(step.result.text) / 100)

    return MonitorGroup(before_generate, after_generate, before_tool, after_tool)


@protocol
def no_forbidden(
    monitors: Monitor | MonitorGroup, action: Literal["reject", "terminate"]
) -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        await run_monitors(monitors, context, step)
        if isinstance(step, BeforeToolCall):
            if step.call.function == "forbidden":
                return (
                    Decision.reject("forbidden", message="Not allowed.")
                    if action == "reject"
                    else Decision.terminate("forbidden")
                )
            return Decision.proceed()
        return None

    return forwards(decide)


def agent_turn(
    input: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    config: GenerateConfig,
) -> ModelOutput:
    turn = sum(isinstance(message, ChatMessageAssistant) for message in input)
    calls = (
        [
            ToolCall(id="call-1", function="lookup", arguments={"key": "a"}),
            ToolCall(id="call-2", function="forbidden", arguments={}),
            ToolCall(id="call-3", function="missing", arguments={}),
        ]
        if turn == 0
        else [ToolCall(id="call-4", function="submit", arguments={"answer": "a"})]
    )
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=f"turn {turn}", tool_calls=calls),
                stop_reason="tool_calls",
            )
        ],
    )


@protocol
def end_at_lookup_result(monitors: Monitor | MonitorGroup) -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        await run_monitors(monitors, context, step)
        if isinstance(step, AfterToolCall) and step.call.function == "lookup":
            return Decision.terminate("looked up")
        return None

    return forwards(decide)


def run_eval(sentinel: Protocol, log_dir: Path) -> str:
    task = Task(
        dataset=[Sample(id="a", input="first " * 100), Sample(id="b", input="second")],
        solver=react(tools=[lookup(), forbidden()]),
        sentinel=sentinel,
        epochs=2,
    )
    [result] = eval(
        task,
        model=get_model("mockllm/model", custom_outputs=agent_turn),
        log_dir=str(log_dir),
        display="none",
    )
    assert result.status == "success"
    assert result.location
    return result.location


@pytest.fixture(scope="module")
def log(tmp_path_factory: pytest.TempPathFactory) -> str:
    return run_eval(no_forbidden(counts(), "reject"), tmp_path_factory.mktemp("logs"))


def replayed_lines(capsys: pytest.CaptureFixture[str]) -> list[tuple[Any, ...]]:
    return [
        (
            str(line["sample_id"]),
            line["epoch"],
            line["stage"],
            line["step_id"],
            record["path"],
            "decision" if "action" in record["report"] else "observation",
            record["report"].get("suspicion"),
            record["report"].get("action"),
        )
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        for record in line["records"]
    ]


def recorded(log: str, samples: set[tuple[str, int]]) -> list[tuple[Any, ...]]:
    # an observation's suspicion, or a decision's action, per record
    return [
        (
            str(sample.id),
            sample.epoch,
            event.stage,
            event.step_id,
            event.path,
            event.kind,
            event.suspicion,
            event.action,
        )
        for sample in read_eval_log(log).samples or []
        if (str(sample.id), sample.epoch) in samples
        for event in sample.events
        if isinstance(event, SentinelEvent)
    ]


@pytest.mark.parametrize(
    ("given", "selected"),
    [
        ([], {("a", 1), ("a", 2), ("b", 1), ("b", 2)}),
        (["b"], {("b", 1), ("b", 2)}),
        (["a:2"], {("a", 2)}),
    ],
)
def test_replay_reports_what_the_live_run_recorded(
    log: str,
    given: list[str],
    selected: set[tuple[str, int]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([log, *(arg for sample in given for arg in ("--sample", sample))])
    replayed = replayed_lines(capsys)
    assert replayed == recorded(log, selected)
    assert {stage for _, _, stage, *_ in replayed} == {
        "model_input",
        "model_output",
        "tool_call",
        "tool_result",
    }


@pytest.mark.parametrize(
    ("sentinel", "last"),
    [
        (lambda: no_forbidden(counts(), "terminate"), ("tool_call", "call-2")),
        (lambda: end_at_lookup_result(counts()), ("tool_result", "call-1")),
    ],
)
def test_a_call_the_sample_ended_at_has_a_tool_result_only_if_it_ran(
    sentinel: Callable[[], Protocol],
    last: tuple[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = run_eval(sentinel(), tmp_path)
    main([log, "--sample", "a:1"])
    replayed = replayed_lines(capsys)
    assert replayed == recorded(log, {("a", 1)})
    assert replayed[-1][2:4] == last


def test_a_call_a_time_limit_cut_off_has_no_tool_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def call_slow(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return ModelOutput.for_tool_call(
            "mockllm/model", "slow", {}, tool_call_id="slow"
        )

    [result] = eval(
        Task(
            dataset=[Sample(id="a", input="go")],
            solver=react(tools=[slow()]),
            sentinel=no_forbidden(counts(), "reject"),
            time_limit=1,
        ),
        model=get_model("mockllm/model", custom_outputs=call_slow),
        log_dir=str(tmp_path),
        display="none",
    )
    assert result.location
    main([result.location])
    replayed = replayed_lines(capsys)
    assert replayed == recorded(result.location, {("a", 1)})
    assert replayed[-1][2:4] == ("tool_call", "slow")


def test_a_sample_the_log_lacks_is_an_error(log: str) -> None:
    with pytest.raises(SystemExit):
        main([log, "--sample", "c"])


@pytest.mark.anyio
async def test_replay_eval_yields_the_steps_of_a_sample(log: str) -> None:
    replayed = [
        (
            str(step.sample_id),
            step.epoch,
            step.stage,
            step.step_id,
            record.path,
            "decision" if isinstance(report, Decision) else "observation",
            report.suspicion if isinstance(report, Observation) else None,
            report.action if isinstance(report, Decision) else None,
        )
        async for step in replay_eval(log, samples=[("a", 1)])
        for record in step.records
        if isinstance(record.detail, Reported)
        for report in [record.detail.report]
    ]
    # read_eval_log refuses to run under trio
    assert replayed == await anyio.to_thread.run_sync(recorded, log, {("a", 1)})
