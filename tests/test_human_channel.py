import functools
import importlib.metadata
import importlib.util
import subprocess
import sys
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from inspect_ai import eval, eval_retry, task_with
from inspect_ai.log import SentinelConfig
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store
from packaging.version import Version

from inspect_sentinel import (
    Context,
    Decision,
    Escalations,
    HumanAnswer,
    HumanChannel,
    HumanQuestion,
    Monitor,
    Observation,
    Protocol,
    ProxyContext,
    concurrent,
    human,
    monitor,
    protocol,
    sequential,
    threshold,
)
from inspect_sentinel._host import HostContext
from inspect_sentinel._human_decisions import kept_human_decision
from inspect_sentinel._integration import (
    Sentinels,
    config_relative_to,
    resolve_sentinel,
)
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._step import BeforeToolCall, Step
from inspect_sentinel._types import Sentinel
from inspect_sentinel.config import sentinel_config, sentinel_from_config
from tests._fakes import (
    FakeHost,
    ListRecorder,
    after_generate_step,
    after_step,
    before_generate_step,
    before_step,
    eval_context,
)


class Desk:
    """A team's own review tool, for a test: the questions it was asked, and the answers it gives in turn."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.questions: list[HumanQuestion] = []
        self.started = anyio.Event()
        self.withdrawn = False


_desk: ContextVar[Desk] = ContextVar("desk")


@contextmanager
def desk(*answers: object) -> Generator[Desk]:
    token = _desk.set(Desk(*answers))
    try:
        yield _desk.get()
    finally:
        _desk.reset(token)


async def ask_desk(question: HumanQuestion) -> HumanAnswer:
    asked = _desk.get()
    asked.questions.append(question)
    answer = asked.answers.pop(0)
    if isinstance(answer, Exception):
        raise answer
    return cast(HumanAnswer, answer)


async def wait_at_desk(question: HumanQuestion) -> HumanAnswer:
    asked = _desk.get()
    asked.started.set()
    try:
        await anyio.sleep_forever()
    finally:
        # a cancelled question is withdrawn from the team's tool
        asked.withdrawn = True
    raise AssertionError("unreachable")


def not_async(question: HumanQuestion) -> HumanAnswer:
    return HumanAnswer("approve")


def desk_for(room: str) -> HumanChannel:
    async def ask(question: HumanQuestion) -> HumanAnswer:
        return HumanAnswer("approve", f"approved in {room}")

    return ask


@monitor
def loud() -> Monitor:
    async def score(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.9, "posts credentials to an external host")

    return score


@monitor
def quiet() -> Monitor:
    async def score(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.2, "lists a directory")

    return score


@protocol
def stops_once_asked() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        await _desk.get().started.wait()
        return Decision.terminate("stopped while the person was asked")

    return decide


PROXY = ProxyContext(
    provider="anthropic", model="claude", user=None, headers={}, inspect=None
)


async def _run(
    spec: Sentinels,
    step: Step | None = None,
    host: FakeHost | None = None,
    proxy: ProxyContext | None = None,
) -> tuple[Decision | None, FakeHost, ListRecorder]:
    host = host or FakeHost()
    recorder = ListRecorder()
    context = HostContext(host=host, recorder=recorder, store=Store(), proxy=proxy)
    decision = await run_sentinel(
        resolve_sentinel(spec), context, step or before_step()
    )
    return decision, host, recorder


@pytest.mark.parametrize(
    ("stage", "step", "choices"),
    [
        ("tool_call", before_step(), ("approve", "reject", "terminate")),
        ("tool_result", after_step(), ("approve", "terminate")),
        ("model_input", before_generate_step(), ("approve", "terminate")),
        ("model_output", after_generate_step(), ("approve", "terminate")),
    ],
)
@pytest.mark.anyio
async def test_the_channel_is_asked_at_each_stage_in_place_of_the_host(
    stage: str, step: Step, choices: tuple[str, ...]
) -> None:
    with desk(HumanAnswer("approve")) as asked:
        decision, host, _ = await _run(
            human(stages=[stage], channel=ask_desk), step=step
        )

    assert decision == Decision.proceed()
    assert host.asked == []
    [question] = asked.questions
    assert (question.stage, question.step, question.choices) == (stage, step, choices)
    assert question.escalations == ()


@pytest.mark.anyio
async def test_the_channel_gets_every_escalations_full_report_and_the_sample() -> None:
    sentinel = sequential(
        [
            threshold([loud(), quiet()], escalate_at=0.5),
            human(["tool_call"], channel=ask_desk),
        ]
    )
    with desk(HumanAnswer("approve")) as asked:
        await _run(sentinel, host=FakeHost(eval=eval_context()), proxy=PROXY)

    [question] = asked.questions
    [escalation] = question.escalations
    assert (escalation.name, escalation.path) == ("threshold", "threshold")
    assert escalation.report.action == "escalate"
    assert escalation.report.explanation == (
        "suspicion 0.90 from loud: posts credentials to an external host"
    )
    assert escalation.report.metadata == {
        "observations": [
            {
                "monitor": "loud",
                "suspicion": 0.9,
                "explanation": "posts credentials to an external host",
            },
            {"monitor": "quiet", "suspicion": 0.2, "explanation": "lists a directory"},
        ]
    }
    assert question.eval == eval_context()
    assert question.proxy == PROXY


MODIFIED = ToolCall(id="call_1", function="bash", arguments={"cmd": "ls -a"})


@pytest.mark.parametrize(
    "answer",
    [
        HumanAnswer("approve"),
        HumanAnswer("approve", reason="the staging host is expected"),
        HumanAnswer("reject", reason="use the staging host"),
        HumanAnswer("reject"),
        HumanAnswer("terminate", reason="exfiltration"),
        # terminate, the answer when nobody can be reached, is taken unoffered
        HumanAnswer("terminate", reason="nobody answered within 30 minutes"),
    ],
)
@pytest.mark.anyio
async def test_a_channels_answer_is_decided_and_recorded_as_the_hosts_is(
    answer: HumanAnswer,
) -> None:
    choices = ["approve", "reject"] if answer.decision != "terminate" else ["approve"]
    from_host, _, host_recorder = await _run(
        human(["tool_call"], choices=choices), host=FakeHost(answer)
    )
    with desk(answer):
        from_channel, _, channel_recorder = await _run(
            human(["tool_call"], choices=choices, channel=ask_desk)
        )

    assert from_channel == from_host
    [by_host] = host_recorder.records
    [by_channel] = channel_recorder.records
    assert by_channel.factory == by_host.factory == "inspect_sentinel/human"
    assert by_channel.reported == by_host.reported
    # what later steps see in human_decisions
    kept = [
        kept_human_decision(r.factory, r.step, "step_1", r.reported)
        for r in (by_host, by_channel)
    ]
    assert kept[0] is not None and kept[0] == kept[1]


@pytest.mark.parametrize(
    ("answer", "match"),
    [
        (
            HumanAnswer("reject"),
            "channel tests.test_human_channel:ask_desk answered 'reject', which is not one of the choices offered",
        ),
        (HumanAnswer("approved"), "answered 'approved', which is not one of"),
        (HumanAnswer("modify", modified=MODIFIED), "answered 'modify'"),
        (HumanAnswer("approve", modified=MODIFIED), "only a 'modify' carries one"),
    ],
)
@pytest.mark.anyio
async def test_an_answer_the_channel_was_not_offered_fails_the_step(
    answer: HumanAnswer, match: str
) -> None:
    with desk(answer), pytest.raises(ValueError, match=match):
        await _run(human(["tool_result"], channel=ask_desk), step=after_step())


@pytest.mark.anyio
async def test_a_channel_that_returns_something_else_fails_the_step() -> None:
    with (
        desk("approve"),
        pytest.raises(TypeError, match="returned str, not a HumanAnswer"),
    ):
        await _run(human(["tool_call"], channel=ask_desk))


@pytest.mark.anyio
async def test_a_channel_that_raises_fails_the_step_with_its_own_error() -> None:
    with desk(ConnectionError("review service unreachable")):
        with pytest.raises(ConnectionError, match="unreachable") as raised:
            await _run(human(["tool_call"], channel=ask_desk))

    assert raised.value.__notes__ == [
        "The channel tests.test_human_channel:ask_desk raised while asking about a tool_call step, so the step does not proceed."
    ]


@pytest.mark.anyio
async def test_a_question_is_withdrawn_when_the_step_is_cancelled_meanwhile() -> None:
    tree = concurrent(
        {
            "person": human(["tool_call"], channel=wait_at_desk),
            "stop": stops_once_asked(),
        }
    )
    with desk() as asked:
        decision, _, recorder = await _run(tree)

    assert decision is not None and decision.action == "terminate"
    assert asked.withdrawn
    assert ("person", "person") in recorder.cancellations
    assert [
        r.reported.path for r in recorder.records if r.factory.endswith("human")
    ] == []


CHANNELS_NOT_FOUND_AGAIN: list[tuple[Any, type[Exception], str]] = [
    ("no_colon", ValueError, "'module:function'"),
    ("tests.test_human_channel:missing", ValueError, "does not define 'missing'"),
    ("missing/review.py:ask", ValueError, "the file does not exist"),
    (desk_for("#reviews"), TypeError, "defined at module level"),
    (not_async, TypeError, "not_async is not async"),
    (Desk(), TypeError, "defined at module level"),
]


@pytest.mark.parametrize(("channel", "error", "match"), CHANNELS_NOT_FOUND_AGAIN)
def test_a_channel_must_be_found_again_by_name(
    channel: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        human(["tool_call"], channel=channel)


def test_a_channel_cannot_be_offered_modify_yet() -> None:
    with pytest.raises(ValueError, match="'modify' through the channel"):
        human(["tool_call"], choices=["approve", "modify"], channel=ask_desk)


@pytest.mark.anyio
async def test_a_channel_is_recorded_by_name_and_rebuilt_from_it() -> None:
    config = sentinel_config(human(["tool_call"], channel=ask_desk))
    assert config == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {
                "stages": ["tool_call"],
                "channel": "tests.test_human_channel:ask_desk",
            },
        }
    )

    rebuilt = sentinel_from_config(config)
    assert sentinel_config(rebuilt) == config
    with desk(HumanAnswer("reject", reason="no")) as asked:
        decision, _, _ = await _run(rebuilt)

    assert decision == Decision.reject("no", message="no")
    assert len(asked.questions) == 1


CHANNEL_FILE = """\
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspect_sentinel import HumanAnswer

# each time the file runs
with open(Path(__file__).with_suffix(".runs"), "a") as runs:
    runs.write("ran\\n")


@dataclass
class Settings:
    queue: str = "reviews"


async def ask(question):
    return HumanAnswer("terminate", reason=f"asked in {Settings().queue} at {question.stage}")
"""


# a directory with a colon in its name, as every absolute Windows path has
@pytest.mark.parametrize("directory", ["", "c:"])
@pytest.mark.anyio
async def test_a_configuration_names_a_channel_in_a_python_file(
    tmp_path: Path, directory: str
) -> None:
    (tmp_path / directory).mkdir(exist_ok=True)
    file = (tmp_path / directory / "review.py").resolve()
    file.write_text(CHANNEL_FILE)
    raw: Any = {
        "name": "human",
        "params": {"stages": ["tool_call"], "channel": f"{file}:ask"},
    }

    configured = sentinel_from_config(raw)
    decision, _, _ = await _run(configured)
    again = sentinel_from_config(raw)

    assert decision == Decision.terminate("asked in reviews at tool_call")
    assert sentinel_config(configured) == SentinelConfig.model_validate(raw)
    # the function the file defines is recorded by the file, as it was named
    assert sentinel_config(again) == SentinelConfig.model_validate(raw)
    assert file.with_suffix(".runs").read_text() == "ran\n"


def _answer_file(reason: str) -> str:
    return f"from inspect_sentinel import HumanAnswer\n\n\nasync def ask(question):\n    return HumanAnswer('terminate', {reason!r})\n"


def _load_file(file: Path) -> Any:
    # a file loaded by its path and not under a name another process imports,
    # as inspect_ai loads a task file, so a function in it is recorded by file
    spec = importlib.util.spec_from_file_location(f"_loaded_{file.stem}", file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _human_config(channel: str) -> SentinelConfig:
    return SentinelConfig.model_validate(
        {"name": "human", "params": {"stages": ["tool_call"], "channel": channel}}
    )


@pytest.mark.parametrize(
    ("launch", "logged"),
    [
        ("project", "tasks/review.py:ask"),
        # a sibling of the directory the channel was made in
        ("project/runs", "../tasks/review.py:ask"),
        # below the directory the channel was made in
        ("project/tasks/runs", "../review.py:ask"),
    ],
)
@pytest.mark.parametrize("given", ["function", "name"])
@pytest.mark.anyio
async def test_a_channel_file_is_recorded_relative_to_another_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch: str,
    logged: str,
    given: str,
) -> None:
    tasks, launch_dir = tmp_path / "project" / "tasks", tmp_path / launch
    tasks.mkdir(parents=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    (tasks / "review.py").write_text(_answer_file("beside the task file"))
    monkeypatch.chdir(tasks)
    channel = (
        _load_file(tasks / "review.py").ask if given == "function" else "review.py:ask"
    )
    sentinel = human(["tool_call"], channel=channel)

    config = config_relative_to(sentinel, str(launch_dir))
    monkeypatch.chdir(launch_dir)
    decision, _, _ = await _run(sentinel_from_config(config))

    # recorded where it was made, as before
    assert sentinel_config(sentinel) == _human_config("review.py:ask")
    assert config == _human_config(logged)
    assert decision == Decision.terminate("beside the task file")


def test_a_channel_named_in_a_parent_directory_is_recorded_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, shared = tmp_path / "tasks", tmp_path / "shared"
    tasks.mkdir()
    shared.mkdir()
    (shared / "review.py").write_text(_answer_file("shared"))
    monkeypatch.chdir(tasks)

    sentinel = human(["tool_call"], channel="../shared/review.py:ask")

    assert config_relative_to(sentinel, str(tmp_path)) == _human_config(
        "shared/review.py:ask"
    )


@pytest.mark.anyio
async def test_a_function_from_a_file_outside_where_it_was_made_is_recorded_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, shared = tmp_path / "tasks", tmp_path / "shared"
    tasks.mkdir()
    shared.mkdir()
    (shared / "review.py").write_text(_answer_file("shared"))
    monkeypatch.chdir(tasks)
    sentinel = human(["tool_call"], channel=_load_file(shared / "review.py").ask)

    config = config_relative_to(sentinel, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    decision, _, _ = await _run(sentinel_from_config(config))

    # outside the directory it was made in, the file is recorded by its
    # absolute path, but relative to a directory that has a relative path to it
    shared_file = (shared / "review.py").resolve().as_posix()
    assert sentinel_config(sentinel) == _human_config(f"{shared_file}:ask")
    assert config == _human_config("shared/review.py:ask")
    assert decision == Decision.terminate("shared")


@pytest.mark.parametrize("beside_link", [False, True])
@pytest.mark.anyio
async def test_a_channel_named_through_a_symlink_is_recorded_as_the_file_it_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beside_link: bool
) -> None:
    # `link/../review.py` names the link target's parent's file; read as text,
    # without the link, it is `tasks/review.py`
    tasks, shared = tmp_path / "tasks", tmp_path / "shared"
    tasks.mkdir()
    (shared / "nested").mkdir(parents=True)
    (tasks / "link").symlink_to(shared / "nested", target_is_directory=True)
    (shared / "review.py").write_text(_answer_file("through the link"))
    if beside_link:
        (tasks / "review.py").write_text(_answer_file("beside the link"))
    monkeypatch.chdir(tasks)
    sentinel = human(["tool_call"], channel="link/../review.py:ask")

    config = config_relative_to(sentinel, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    decision, _, _ = await _run(sentinel_from_config(config))

    assert config == _human_config("shared/review.py:ask")
    assert decision == Decision.terminate("through the link")


def test_nested_and_named_channels_are_recorded_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for name in ("a", "b", "c"):
        (tasks / f"{name}.py").write_text(_answer_file(name))
    monkeypatch.chdir(tasks)
    second = human(["tool_call"], channel="b.py:ask")
    sentinels: dict[str, Sentinel] = {
        "first": sequential([loud(), human(["tool_call"], channel="a.py:ask")]),
        "second": second,
        # a lone nested instance, which the configuration records in params
        "third": sequential(human(["tool_call"], channel="c.py:ask")),
    }

    named = config_relative_to(sentinels, str(tmp_path)).model_dump()
    listed = config_relative_to([second], str(tmp_path)).model_dump()

    assert named["first"]["children"][1]["params"]["channel"] == "tasks/a.py:ask"
    assert named["second"]["params"]["channel"] == "tasks/b.py:ask"
    assert named["third"]["params"]["children"]["params"]["channel"] == "tasks/c.py:ask"
    assert listed[0]["params"]["channel"] == "tasks/b.py:ask"


@protocol
def reads_policy(policy_file: str, notes: str = "notes.py:x") -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        return None

    return decide


def test_other_values_are_recorded_as_sentinel_config_records_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "review.py").write_text(_answer_file("task"))
    absolute = (tasks / "review.py").resolve().as_posix()
    monkeypatch.chdir(tasks)
    unchanged = [
        # a module's name, and a file given by its absolute path
        human(["tool_call"], channel="tests.test_human_channel:ask_desk"),
        human(["tool_call"], channel=f"{absolute}:ask"),
        # a param the factory does not mark as a source reference, even one
        # that reads as a file or a reference
        reads_policy(policy_file="policy.txt", notes="notes.py:x"),
    ]

    for sentinel in unchanged:
        assert config_relative_to(sentinel, str(tmp_path)) == sentinel_config(sentinel)
    # relative to the directory it was made in, as recorded
    made_here = human(["tool_call"], channel="review.py:ask")
    assert config_relative_to(made_here, str(tasks)) == sentinel_config(made_here)
    # rebuilt from a configuration relative to the launch directory, there
    monkeypatch.chdir(tmp_path)
    rebuilt = sentinel_from_config(_human_config("tasks/review.py:ask"))
    assert config_relative_to(rebuilt, str(tmp_path)) == _human_config(
        "tasks/review.py:ask"
    )


def test_a_file_with_no_relative_path_is_recorded_by_its_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # as on Windows, for a file on another drive
    import inspect_sentinel._sources as sources

    def no_relative_path(path: object, start: object = None) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    (tmp_path / "review.py").write_text(_answer_file("task"))
    monkeypatch.chdir(tmp_path)
    sentinel = human(["tool_call"], channel="review.py:ask")
    monkeypatch.setattr(sources.os.path, "relpath", no_relative_path)

    config = config_relative_to(sentinel, "/elsewhere")

    file = (tmp_path / "review.py").resolve().as_posix()
    assert config == _human_config(f"{file}:ask")


def retried(
    ask: Callable[[HumanQuestion], Awaitable[HumanAnswer]],
) -> Callable[[HumanQuestion], Awaitable[HumanAnswer]]:
    # a decorator defined in another file than the channel it wraps, as a
    # library's retry decorator is
    @functools.wraps(ask)
    async def again(question: HumanQuestion) -> HumanAnswer:
        return await ask(question)

    return again


DECORATED_TASK_FILE = """\
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from inspect_sentinel import HumanAnswer, human
from tests.test_human_channel import retried


@retried
async def ask(question):
    return HumanAnswer("approve", "approved by the decorated channel")


@task
def decorated():
    return Task(
        dataset=[Sample(input="hello")],
        solver=generate(),
        sentinel=human(["model_input"], channel=ask),
    )
"""


def test_a_decorated_channel_in_a_task_file_is_found_again_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("decorated_task.py").write_text(DECORATED_TASK_FILE)

    log = eval("decorated_task.py", model="mockllm/model", log_dir="logs")[0]
    retried_log = eval_retry(log, log_dir="logs")[0]

    assert log.status == "success", log.error
    assert log.eval.config.sentinel == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {"stages": ["model_input"], "channel": "decorated_task.py:ask"},
        }
    )
    assert retried_log.status == "success", retried_log.error


# a task file in a subdirectory whose channel is the task file's own function,
# or a file beside it named by its relative path; the sample fails once, so
# the retry runs it and asks the channel
SUBDIRECTORY_TASK_FILE = """\
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, solver

from inspect_sentinel import HumanAnswer, human

FAILED_ONCE = Path(__file__).parent / "failed_once"


async def ask(question):
    return HumanAnswer("approve", "approved by the task file's channel")


@solver
def fail_once():
    async def solve(state, generate):
        if not FAILED_ONCE.exists():
            FAILED_ONCE.touch()
            raise RuntimeError("the first attempt fails")
        return state

    return solve


@task
def plain():
    return Task(
        dataset=[Sample(input="hello")],
        solver=[fail_once(), generate()],
        sentinel=human(["model_input"], channel=CHANNEL),
    )
"""

BESIDE_FILE = """\
from inspect_sentinel import HumanAnswer


async def ask(question):
    return HumanAnswer("approve", "approved by the channel beside the task file")
"""


def _inspect_ai_version() -> Version:
    return Version(importlib.metadata.version("inspect-ai"))


@pytest.mark.skipif(
    _inspect_ai_version() < Version("0.3.293"),
    reason="before inspect-ai 0.3.293, the log records a task's own channel file relative to the task file's directory, and eval_retry() rebuilds it where the eval was launched, so it is not found",
)
@pytest.mark.parametrize(
    ("channel", "file", "reason"),
    [
        ("ask", "plain_task.py", "approved by the task file's channel"),
        (
            '"review.py:ask"',
            "review.py",
            "approved by the channel beside the task file",
        ),
    ],
)
@pytest.mark.parametrize(
    ("launch", "tasks_from_launch"),
    [
        ("project", "tasks"),
        # outside the project, so the log records the task file by its
        # absolute path
        ("run", "../project/tasks"),
    ],
)
def test_a_channel_named_relative_to_a_task_file_in_a_subdirectory_is_found_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    file: str,
    reason: str,
    launch: str,
    tasks_from_launch: str,
) -> None:
    tasks = tmp_path / "project" / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "plain_task.py").write_text(
        SUBDIRECTORY_TASK_FILE.replace("CHANNEL", channel)
    )
    (tasks / "review.py").write_text(BESIDE_FILE)
    (tmp_path / launch).mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path / launch)
    task = (
        "tasks/plain_task.py" if launch == "project" else str(tasks / "plain_task.py")
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path / "logs"))[0]
    retried_log = eval_retry(log, log_dir=str(tmp_path / "logs"))[0]

    assert log.status == "error"
    # relative to the directory the eval was launched from, as the task file is
    assert log.eval.config.sentinel == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {
                "stages": ["model_input"],
                "channel": f"{tasks_from_launch}/{file}:ask",
            },
        }
    )
    assert retried_log.status == "success", retried_log.error
    assert retried_log.eval.config.sentinel == log.eval.config.sentinel
    assert retried_log.samples is not None
    assert reason in retried_log.samples[0].model_dump_json()


_GATE = pytest.mark.skipif(
    _inspect_ai_version() < Version("0.3.293"),
    reason="before inspect-ai 0.3.293, the log records a task's own channel file relative to where it was made, and eval_retry() rebuilds it where the eval was launched",
)

ROOT_FILE = BESIDE_FILE.replace(
    "approved by the channel beside the task file", "approved by the root's channel"
)


def _import_task_file(file: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    # a task file imported as a module, registering its tasks, as a project
    # imports its own task factories
    spec = importlib.util.spec_from_file_location(name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@_GATE
@pytest.mark.parametrize("called_as", ["object", "name"])
def test_an_imported_task_factorys_channel_is_found_where_it_was_made_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, called_as: str
) -> None:
    # the factory runs in the launch directory, so its relative channel is
    # the root's file, not the one beside the task file
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "imported_task.py").write_text(
        SUBDIRECTORY_TASK_FILE.replace("CHANNEL", '"review.py:ask"').replace(
            "def plain", "def imported_task"
        )
    )
    (tasks / "review.py").write_text(BESIDE_FILE)
    (tmp_path / "review.py").write_text(ROOT_FILE)
    module = _import_task_file(tasks / "imported_task.py", "imported_task", monkeypatch)
    monkeypatch.chdir(tmp_path)
    task = module.imported_task() if called_as == "object" else "imported_task"

    log = eval(task, model="mockllm/model", log_dir="logs")[0]
    retried_log = eval_retry(log, log_dir="logs")[0]

    assert log.status == "error"
    assert log.eval.config.sentinel == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {"stages": ["model_input"], "channel": "review.py:ask"},
        }
    )
    assert retried_log.status == "success", retried_log.error
    assert retried_log.samples is not None
    assert "approved by the root's channel" in retried_log.samples[0].model_dump_json()


@_GATE
def test_a_channel_made_elsewhere_and_attached_to_a_task_is_found_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, elsewhere = tmp_path / "tasks", tmp_path / "elsewhere"
    tasks.mkdir()
    elsewhere.mkdir()
    (tasks / "attached_task.py").write_text(
        PLAIN_SUBDIRECTORY_TASK_FILE.replace("def plain", "def attached_task")
    )
    (tmp_path / "review.py").write_text(ROOT_FILE)
    (elsewhere / "review.py").write_text(
        BESIDE_FILE.replace(
            "approved by the channel beside the task file", "approved elsewhere"
        )
    )
    module = _import_task_file(tasks / "attached_task.py", "attached_task", monkeypatch)
    monkeypatch.chdir(elsewhere)
    sentinel = human(["model_input"], channel="review.py:ask")
    monkeypatch.chdir(tmp_path)

    log = eval(
        task_with(module.attached_task(), sentinel=sentinel),
        model="mockllm/model",
        log_dir="logs",
    )[0]
    retried_log = eval_retry(log, log_dir="logs")[0]

    assert log.status == "error"
    assert log.eval.config.sentinel == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {"stages": ["model_input"], "channel": "elsewhere/review.py:ask"},
        }
    )
    assert retried_log.status == "success", retried_log.error
    assert retried_log.samples is not None
    assert "approved elsewhere" in retried_log.samples[0].model_dump_json()


@_GATE
@pytest.mark.parametrize("beside_link", [False, True])
def test_a_task_files_channel_named_through_a_symlink_is_found_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beside_link: bool
) -> None:
    tasks, shared = tmp_path / "tasks", tmp_path / "shared"
    tasks.mkdir()
    (shared / "nested").mkdir(parents=True)
    (tasks / "link").symlink_to(shared / "nested", target_is_directory=True)
    (tasks / "plain_task.py").write_text(
        SUBDIRECTORY_TASK_FILE.replace("CHANNEL", '"link/../review.py:ask"')
    )
    (shared / "review.py").write_text(
        BESIDE_FILE.replace(
            "approved by the channel beside the task file", "approved through the link"
        )
    )
    if beside_link:
        (tasks / "review.py").write_text(BESIDE_FILE)
    monkeypatch.chdir(tmp_path)

    log = eval("tasks/plain_task.py", model="mockllm/model", log_dir="logs")[0]
    retried_log = eval_retry(log, log_dir="logs")[0]

    assert log.status == "error"
    assert log.eval.config.sentinel == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {"stages": ["model_input"], "channel": "shared/review.py:ask"},
        }
    )
    assert retried_log.status == "success", retried_log.error
    assert retried_log.samples is not None
    assert "approved through the link" in retried_log.samples[0].model_dump_json()


# the same task file with no sentinel of its own
PLAIN_SUBDIRECTORY_TASK_FILE = SUBDIRECTORY_TASK_FILE.replace(
    '        sentinel=human(["model_input"], channel=CHANNEL),\n', ""
)


# an eval's own sentinel is made where the eval runs, so its relative files
# are relative to there, even when the task file's directory has one of the
# same name
@pytest.mark.parametrize("beside_task_file", [False, True])
def test_an_evals_own_channel_file_is_found_where_the_eval_ran_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beside_task_file: bool
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "plain_task.py").write_text(PLAIN_SUBDIRECTORY_TASK_FILE)
    (tmp_path / "review.py").write_text(
        BESIDE_FILE.replace(
            "approved by the channel beside the task file", "approved beside the eval"
        )
    )
    if beside_task_file:
        (tasks / "review.py").write_text(
            BESIDE_FILE.replace(
                "approved by the channel beside the task file",
                "approved beside the task file",
            )
        )
    monkeypatch.chdir(tmp_path)

    log = eval(
        "tasks/plain_task.py",
        model="mockllm/model",
        sentinel=human(["model_input"], channel="review.py:ask"),
        log_dir="logs",
    )[0]
    retried_log = eval_retry(log, log_dir="logs")[0]

    assert log.status == "error"
    assert retried_log.status == "success", retried_log.error
    assert retried_log.samples is not None
    assert "approved beside the eval" in retried_log.samples[0].model_dump_json()
    assert log.eval.config.sentinel == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {"stages": ["model_input"], "channel": "review.py:ask"},
        }
    )


SCRIPT = """\
from inspect_sentinel import HumanAnswer, human
from inspect_sentinel.config import sentinel_config


async def ask(question):
    return HumanAnswer("approve", "approved by the script's channel")


if __name__ == "__main__":
    print(sentinel_config(human(["tool_call"], channel=ask)).model_dump_json())
"""


@pytest.mark.anyio
async def test_a_scripts_channel_is_rebuilt_in_another_process(tmp_path: Path) -> None:
    script = (tmp_path / "run_eval.py").resolve()
    script.write_text(SCRIPT)

    recorded = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    ).stdout
    config = SentinelConfig.model_validate_json(recorded)
    decision, _, _ = await _run(sentinel_from_config(config))

    assert (
        config.root
        == SentinelConfig.model_validate(
            {
                "name": "human",
                "params": {"stages": ["tool_call"], "channel": f"{script}:ask"},
            }
        ).root
    )
    assert decision == Decision.proceed("approved by the script's channel")


def test_a_channel_typed_into_a_repl_is_refused() -> None:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile("async def ask(question):\n    pass\n", "<stdin>", "exec"), namespace)

    with pytest.raises(TypeError, match="REPL, notebook or `python -c`"):
        human(["tool_call"], channel=namespace["ask"])
