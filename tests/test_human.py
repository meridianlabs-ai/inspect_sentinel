from collections.abc import Sequence
from typing import Any

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.log import SentinelConfig
from inspect_ai.tool import ToolCall

from inspect_sentinel import HumanAnswer, human, sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol, step_types
from inspect_sentinel._integration import (
    Sentinels,
    config_from_sentinel,
    resolve_sentinel,
    sentinel_from_config,
)
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Action, Decision, Observation
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol, ProtocolGroup
from tests._fakes import FakeHost, ListRecorder, after_step, before_step, host_context


@monitor
def hum_score() -> Monitor:
    async def score(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation(suspicion=0.1)

    return score


@protocol
def hum_rule(action: Action = "escalate") -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision(action=action, explanation=f"rule says {action}")

    return decide


async def _root(
    spec: Sentinels,
    *answers: HumanAnswer,
    step: Step | None = None,
) -> tuple[Decision | None, FakeHost, ListRecorder]:
    host = FakeHost(*answers)
    recorder = ListRecorder()
    context = host_context(recorder=recorder, host=host)
    decision = await run_sentinel(
        resolve_sentinel(spec), context, step or before_step()
    )
    return decision, host, recorder


@pytest.mark.parametrize(
    "stages",
    [[], ["model_input"], ["model_output"], ["tool_call", "bogus"], "tool_call"],
)
def test_stages_must_name_tool_stages(stages: Sequence[str]) -> None:
    with pytest.raises(ValueError, match="'tool_call'.*'tool_result'"):
        human(stages=stages)


def test_one_function_per_chosen_stage() -> None:
    before = human(stages=["tool_call"])
    after = human(stages=["tool_result"])
    both = human(stages=["tool_call", "tool_result"])
    assert not isinstance(before, ProtocolGroup)
    assert step_types(before) == {BeforeToolCall}
    assert step_types(after) == {AfterToolCall}
    assert isinstance(both, ProtocolGroup)
    assert step_types(both) == {BeforeToolCall, AfterToolCall}


@pytest.mark.parametrize(
    ("stage", "step", "expected"),
    [
        ("tool_call", before_step(), ("approve", "reject", "terminate")),
        ("tool_result", after_step(), ("approve", "terminate")),
    ],
)
@pytest.mark.anyio
async def test_default_choices_are_what_is_legal_at_the_stage(
    stage: str, step: Step, expected: tuple[str, ...]
) -> None:
    _, host, _ = await _root(human(stages=[stage]), HumanAnswer("approve"), step=step)
    assert host.asked[0].choices == expected
    assert host.asked[0].step == step


@pytest.mark.anyio
async def test_custom_choices_are_offered() -> None:
    _, host, _ = await _root(
        human(stages=["tool_call"], choices=["approve", "modify"]),
        HumanAnswer("approve"),
    )
    assert host.asked[0].choices == ("approve", "modify")


@pytest.mark.parametrize(
    ("stages", "choices", "match"),
    [
        (["tool_call"], [], "at least one"),
        (["tool_call"], ["approve", "escalate"], "'escalate'"),
        (["tool_result"], ["approve", "reject"], "'reject'.*after a tool call"),
        (["tool_call", "tool_result"], ["approve", "modify"], "'modify'"),
    ],
)
def test_invalid_choices_are_a_configuration_error(
    stages: list[str], choices: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        human(stages=stages, choices=choices)


MODIFIED = ToolCall(id="call_1", function="bash", arguments={"cmd": "ls -a"})


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (HumanAnswer("approve"), Decision.proceed()),
        (
            HumanAnswer("reject", reason="use the staging host"),
            Decision.reject("use the staging host", message="use the staging host"),
        ),
        (HumanAnswer("reject"), Decision.reject()),
        (HumanAnswer("reject", reason=""), Decision.reject()),
        (
            HumanAnswer("terminate", reason="exfiltration"),
            Decision.terminate("exfiltration"),
        ),
        (
            HumanAnswer("modify", modified=MODIFIED),
            Decision(action="modify", modified=MODIFIED),
        ),
    ],
)
@pytest.mark.anyio
async def test_the_answer_becomes_the_decision(
    answer: HumanAnswer, expected: Decision
) -> None:
    choices = ["approve", "reject", "terminate", "modify"]
    decision, _, _ = await _root(human(stages=["tool_call"], choices=choices), answer)
    assert decision == expected


@pytest.mark.parametrize(
    ("answer", "match"),
    [
        (HumanAnswer("modify", modified=MODIFIED), "'modify'.*not one of"),
        (HumanAnswer("approve", modified=MODIFIED), "modified"),
    ],
)
@pytest.mark.anyio
async def test_an_answer_the_person_was_not_offered_is_an_error(
    answer: HumanAnswer, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        await _root(human(stages=["tool_call"]), answer)


@pytest.mark.anyio
async def test_a_host_may_end_the_prompt_with_terminate() -> None:
    decision, _, _ = await _root(
        human(stages=["tool_call"], choices=["approve"]),
        HumanAnswer("terminate", reason="dismissed"),
    )
    assert decision == Decision.terminate("dismissed")


@pytest.mark.anyio
async def test_an_unoffered_reject_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="'reject'.*not one of"):
        await _root(
            human(stages=["tool_call"], choices=["approve"]), HumanAnswer("reject")
        )


@pytest.mark.anyio
async def test_the_answer_ends_only_its_chain() -> None:
    tree = concurrent(
        {
            "chain": sequential([hum_rule(), human(stages=["tool_call"])]),
            "peer": hum_rule("continue"),
        }
    )
    decision, _, recorder = await _root(tree, HumanAnswer("reject", reason="no"))
    assert decision is not None and decision.action == "reject"
    assert decision.message == "no"
    assert recorder.bypassed_layers == []
    assert decision.explanation == "no (chain: reject; peer: continue)"


@pytest.mark.anyio
async def test_a_peer_reject_outranks_the_persons_approve() -> None:
    tree = concurrent(
        {
            "chain": sequential([hum_rule(), human(stages=["tool_call"])]),
            "guard": hum_rule("reject"),
        }
    )
    decision, host, _ = await _root(tree, HumanAnswer("approve"))
    assert len(host.asked) == 1
    assert decision is not None and decision.action == "reject"


@pytest.mark.anyio
async def test_the_person_sees_who_escalated() -> None:
    _, host, _ = await _root(
        sequential([hum_rule(), human(stages=["tool_call"])]), HumanAnswer("approve")
    )
    [asked] = host.asked
    assert [(e.name, e.report.explanation) for e in asked.step.escalations] == [
        ("hum_rule", "rule says escalate")
    ]


@pytest.mark.anyio
async def test_the_person_is_not_asked_when_an_earlier_link_decides() -> None:
    decision, host, _ = await _root(
        sequential([hum_rule("reject"), human(stages=["tool_call"])])
    )
    assert decision is not None and decision.action == "reject"
    assert host.asked == []


@pytest.mark.anyio
async def test_a_lone_human_asks_about_every_call() -> None:
    instance = resolve_sentinel(human(stages=["tool_call"]))
    host = FakeHost(HumanAnswer("approve"), HumanAnswer("approve"))
    for _ in range(2):
        await run_sentinel(instance, host_context(host=host), before_step())
    assert len(host.asked) == 2


@pytest.mark.anyio
async def test_after_a_call_the_person_sees_the_result() -> None:
    decision, host, _ = await _root(
        human(stages=["tool_call", "tool_result"]),
        HumanAnswer("terminate", reason="leaked a key"),
        step=after_step(),
    )
    assert decision == Decision.terminate("leaked a key")
    [asked] = host.asked
    assert isinstance(asked.step, AfterToolCall)
    assert asked.step.result.text == "out"


@pytest.mark.parametrize(
    "spec",
    [
        [hum_rule("continue"), human(stages=["tool_call"])],
        {"rule": hum_rule("continue"), "person": human(stages=["tool_call"])},
        [hum_score(), human(stages=["tool_call"])],
    ],
)
@pytest.mark.anyio
async def test_a_human_beside_others_in_concurrent_asks_every_call(
    spec: Sentinels,
) -> None:
    decision, host, _ = await _root(spec, HumanAnswer("reject"))
    assert len(host.asked) == 1
    assert decision is not None and decision.action == "reject"


def test_human_is_registered_under_the_package_name() -> None:
    assert registry_info(human(stages=["tool_call"])).name == "inspect_sentinel/human"


RAW: Any = {
    "name": "sequential",
    "children": [
        {"name": "hum_rule"},
        {"name": "human", "params": {"stages": ["tool_call"]}},
    ],
}


@pytest.mark.anyio
async def test_a_human_round_trips_through_config() -> None:
    config = SentinelConfig.model_validate(RAW)
    configured = sentinel_from_config(RAW)
    assert config_from_sentinel(configured) == config
    decision, host, _ = await _root(configured, HumanAnswer("reject", reason="no"))
    assert decision is not None and decision.message == "no"
    assert len(host.asked) == 1
