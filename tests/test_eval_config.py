from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval, eval_retry, task
from inspect_ai._sentinel._context import active_sentinel
from inspect_ai._util.registry import registry_info
from inspect_ai.dataset import Sample
from inspect_ai.event import SentinelEvent
from inspect_ai.log import EvalLog, SentinelConfig, SentinelEntry, read_eval_log
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, tool

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    Monitor,
    Observation,
    Protocol,
    monitor,
    observe_only,
    protocol,
    threshold,
)


@monitor
def d2_suspicion(score: float = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(score)

    return check


@protocol
def d2_rule(reason: str = "no") -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.reject(reason)

    return decide


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


@solver
def record_sentinel() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        root = active_sentinel()
        state.metadata["sentinel"] = (
            registry_info(root).name if root is not None else None
        )
        return state

    return solve


def sentinel_task(sentinel: Any = None) -> Task:
    return Task(
        dataset=[Sample(input="x", target="y")],
        solver=record_sentinel(),
        sentinel=sentinel,
    )


def active_root(log: EvalLog) -> str | None:
    assert log.status == "success", log.error
    assert log.samples
    return log.samples[0].metadata["sentinel"]


RULE_CONFIG = [{"name": "d2_rule", "params": {"reason": "no"}}]


def config_data(log: EvalLog) -> Any:
    assert log.eval.config.sentinel is not None
    return log.eval.config.sentinel.model_dump()


def test_observe_only_records_monitors_with_nothing_acting() -> None:
    log = eval(
        sentinel_task(observe_only({"watch": d2_suspicion(score=0.2)})),
        model="mockllm/model",
    )[0]
    assert config_data(log) == {
        "name": "observe_only",
        "params": {},
        "monitors": {"watch": {"name": "d2_suspicion", "params": {"score": 0.2}}},
    }
    assert active_root(log) == "inspect_sentinel/observe_only"


@pytest.mark.parametrize(
    "sentinel",
    [
        lambda: d2_suspicion(),
        lambda: {"watch": d2_suspicion()},
        lambda: [{"name": "d2_suspicion"}],
        lambda: "d2_suspicion",
    ],
)
def test_monitors_alone_fail_at_task_construction(
    sentinel: Callable[[], Any],
) -> None:
    with pytest.raises(ValueError, match=r"only monitors: .*observe_only\(\)"):
        sentinel_task(sentinel())


def test_eval_sentinel_overrides_the_task() -> None:
    log = eval(
        sentinel_task(d2_rule(reason="task")),
        model="mockllm/model",
        sentinel=[d2_suspicion(score=0.7), d2_rule(reason="eval")],
    )[0]
    assert config_data(log) == [
        {"name": "d2_suspicion", "params": {"score": 0.7}},
        {"name": "d2_rule", "params": {"reason": "eval"}},
    ]
    assert active_root(log) == "inspect_sentinel/concurrent"


def test_config_file_and_registered_name(tmp_path: Path) -> None:
    config = tmp_path / "sentinel.yaml"
    config.write_text(
        "sentinel:\n  block:\n    name: d2_rule\n    params:\n      reason: file\n"
    )
    log = eval(sentinel_task(), model="mockllm/model", sentinel=str(config))[0]
    assert config_data(log) == {
        "block": {"name": "d2_rule", "params": {"reason": "file"}}
    }

    log = eval(sentinel_task(), model="mockllm/model", sentinel="d2_rule")[0]
    assert config_data(log) == {"name": "d2_rule", "params": {}}
    assert active_root(log) == "d2_rule"


@monitor(version=3)
def d2_versioned(score: float = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(score)

    return check


def test_version_round_trips_through_the_log(tmp_path: Path) -> None:
    log = eval(
        sentinel_task([d2_versioned(score=0.1), d2_rule(reason="no")]),
        model="mockllm/model",
        log_dir=str(tmp_path),
    )[0]
    expected = [
        {"name": "d2_versioned", "params": {"score": 0.1}, "version": 3},
        *RULE_CONFIG,
    ]
    assert config_data(log) == expected
    read = read_eval_log(log.location)
    assert read.eval.config.sentinel == log.eval.config.sentinel
    assert config_data(read) == expected


@solver
def fail_once(marker: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not Path(marker).exists():
            Path(marker).touch()
            raise RuntimeError("first attempt fails")
        return state

    return solve


@solver
def call_addition() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.append(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id="c1", function="addition", arguments={"x": 1, "y": 1})
                ],
            )
        )
        result = await execute_tools(state.messages, [addition()])
        state.messages.extend(result.messages)
        return state

    return solve


@task
def d2_lone_root_task(marker: str) -> Task:
    return Task(
        dataset=[Sample(input="x", target="y")],
        solver=[call_addition(), fail_once(marker)],
        sentinel=threshold(d2_suspicion(score=0.9), reject_at=0.5),
    )


def sentinel_paths(log: EvalLog) -> list[tuple[str, str]]:
    assert log.samples
    return [
        (e.factory, e.path)
        for e in log.samples[0].events
        if isinstance(e, SentinelEvent)
    ]


def test_eval_retry_records_a_lone_roots_paths_as_the_first_run_did(
    tmp_path: Path,
) -> None:
    log = eval(
        d2_lone_root_task(str(tmp_path / "marker")),
        model="mockllm/model",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "error"
    recorded = log.eval.config.sentinel
    assert isinstance(recorded, SentinelConfig)
    assert isinstance(recorded.root, SentinelEntry)
    assert recorded.root.name == "threshold"
    retried = eval_retry(log, log_dir=str(tmp_path))[0]
    assert retried.status == "success", retried.error
    expected = [("d2_suspicion", "d2_suspicion"), ("inspect_sentinel/threshold", "")]
    assert sentinel_paths(log) == expected
    assert sentinel_paths(retried) == expected


ran = False


@solver
def mark_ran() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        global ran
        ran = True
        return state

    return solve


@pytest.mark.parametrize(
    "sentinel,error,match",
    [
        ([{"name": "d2_missing"}], ValueError, "not a registered monitor or protocol"),
        ([{"name": "d2_rule", "params": {"bogus": 1}}], TypeError, "sentinel\\[0\\]"),
        ("no_such_file.yaml", ValueError, "neither a config file"),
        ([], ValueError, "at least one monitor or protocol"),
        (42, TypeError, "sentinel must be"),
        ("d2_suspicion", ValueError, "observe_only"),
        ([{"name": "d2_suspicion"}], ValueError, "observe_only"),
    ],
)
def test_bad_config_fails_at_eval_start(
    sentinel: Any, error: type[Exception], match: str, tmp_path: Path
) -> None:
    global ran
    ran = False
    with pytest.raises(error, match=match):
        eval(
            Task(dataset=[Sample(input="x")], solver=mark_ran()),
            model="mockllm/model",
            sentinel=sentinel,
            log_dir=str(tmp_path),
        )
    assert not ran
    assert not list(tmp_path.glob("*.eval"))
