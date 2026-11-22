from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from inspect_ai.log import SentinelConfig
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store

from inspect_sentinel import (
    Context,
    Decision,
    Escalations,
    HumanAnswer,
    HumanQuestion,
    HumanSurface,
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
from inspect_sentinel._integration import Sentinels, resolve_sentinel
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._step import BeforeToolCall, Step
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


def desk_for(channel: str) -> HumanSurface:
    async def ask(question: HumanQuestion) -> HumanAnswer:
        return HumanAnswer("approve", f"approved in {channel}")

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
async def test_the_surface_is_asked_at_each_stage_in_place_of_the_host(
    stage: str, step: Step, choices: tuple[str, ...]
) -> None:
    with desk(HumanAnswer("approve")) as asked:
        decision, host, _ = await _run(
            human(stages=[stage], surface=ask_desk), step=step
        )

    assert decision == Decision.proceed()
    assert host.asked == []
    [question] = asked.questions
    assert (question.stage, question.step, question.choices) == (stage, step, choices)
    assert question.escalations == ()


@pytest.mark.anyio
async def test_the_surface_gets_every_escalations_full_report_and_the_sample() -> None:
    sentinel = sequential(
        [
            threshold([loud(), quiet()], escalate_at=0.5),
            human(["tool_call"], surface=ask_desk),
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
async def test_a_surfaces_answer_is_decided_and_recorded_as_the_hosts_is(
    answer: HumanAnswer,
) -> None:
    choices = ["approve", "reject"] if answer.decision != "terminate" else ["approve"]
    from_host, _, host_recorder = await _run(
        human(["tool_call"], choices=choices), host=FakeHost(answer)
    )
    with desk(answer):
        from_surface, _, surface_recorder = await _run(
            human(["tool_call"], choices=choices, surface=ask_desk)
        )

    assert from_surface == from_host
    [by_host] = host_recorder.records
    [by_surface] = surface_recorder.records
    assert by_surface.factory == by_host.factory == "inspect_sentinel/human"
    assert by_surface.reported == by_host.reported
    # what later steps see in human_decisions
    kept = [
        kept_human_decision(r.factory, r.step, "step_1", r.reported)
        for r in (by_host, by_surface)
    ]
    assert kept[0] is not None and kept[0] == kept[1]


@pytest.mark.parametrize(
    ("answer", "match"),
    [
        (
            HumanAnswer("reject"),
            "surface tests.test_human_surface:ask_desk answered 'reject', which is not one of the choices offered",
        ),
        (HumanAnswer("approved"), "answered 'approved', which is not one of"),
        (HumanAnswer("modify", modified=MODIFIED), "answered 'modify'"),
        (HumanAnswer("approve", modified=MODIFIED), "only a 'modify' carries one"),
    ],
)
@pytest.mark.anyio
async def test_an_answer_the_surface_was_not_offered_fails_the_step(
    answer: HumanAnswer, match: str
) -> None:
    with desk(answer), pytest.raises(ValueError, match=match):
        await _run(human(["tool_result"], surface=ask_desk), step=after_step())


@pytest.mark.anyio
async def test_a_surface_that_returns_something_else_fails_the_step() -> None:
    with (
        desk("approve"),
        pytest.raises(TypeError, match="returned str, not a HumanAnswer"),
    ):
        await _run(human(["tool_call"], surface=ask_desk))


@pytest.mark.anyio
async def test_a_surface_that_raises_fails_the_step_with_its_own_error() -> None:
    with desk(ConnectionError("review service unreachable")):
        with pytest.raises(ConnectionError, match="unreachable") as raised:
            await _run(human(["tool_call"], surface=ask_desk))

    assert raised.value.__notes__ == [
        "The surface tests.test_human_surface:ask_desk raised while asking about a tool_call step, so the step does not proceed."
    ]


@pytest.mark.anyio
async def test_a_question_is_withdrawn_when_the_step_is_cancelled_meanwhile() -> None:
    tree = concurrent(
        {
            "person": human(["tool_call"], surface=wait_at_desk),
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


SURFACES_NOT_FOUND_AGAIN: list[tuple[Any, type[Exception], str]] = [
    ("no_colon", ValueError, "'module:function'"),
    ("tests.test_human_surface:missing", ValueError, "does not define 'missing'"),
    ("missing/review.py:ask", ValueError, "no file 'missing/review.py'"),
    (desk_for("#reviews"), TypeError, "defined at module level"),
    (not_async, TypeError, "not_async is not async"),
    (Desk(), TypeError, "defined at module level"),
]


@pytest.mark.parametrize(("surface", "error", "match"), SURFACES_NOT_FOUND_AGAIN)
def test_a_surface_must_be_found_again_by_name(
    surface: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        human(["tool_call"], surface=surface)


def test_a_surface_cannot_be_offered_modify_yet() -> None:
    with pytest.raises(ValueError, match="'modify' through the surface"):
        human(["tool_call"], choices=["approve", "modify"], surface=ask_desk)


@pytest.mark.anyio
async def test_a_surface_is_recorded_by_name_and_rebuilt_from_it() -> None:
    config = sentinel_config(human(["tool_call"], surface=ask_desk))
    assert config == SentinelConfig.model_validate(
        {
            "name": "human",
            "params": {
                "stages": ["tool_call"],
                "surface": "tests.test_human_surface:ask_desk",
            },
        }
    )

    rebuilt = sentinel_from_config(config)
    assert sentinel_config(rebuilt) == config
    with desk(HumanAnswer("reject", reason="no")) as asked:
        decision, _, _ = await _run(rebuilt)

    assert decision == Decision.reject("no", message="no")
    assert len(asked.questions) == 1


@pytest.mark.anyio
async def test_a_configuration_names_a_surface_in_a_python_file(tmp_path: Path) -> None:
    file = tmp_path / "review.py"
    file.write_text(
        "from inspect_sentinel import HumanAnswer\n\n"
        "async def ask(question):\n"
        "    return HumanAnswer('terminate', reason=f'asked at {question.stage}')\n"
    )
    raw: Any = {
        "name": "human",
        "params": {"stages": ["tool_call"], "surface": f"{file}:ask"},
    }

    configured = sentinel_from_config(raw)
    decision, _, _ = await _run(configured)

    assert decision == Decision.terminate("asked at tool_call")
    assert sentinel_config(configured) == SentinelConfig.model_validate(raw)
