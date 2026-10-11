from typing import Any

import pytest
from inspect_ai.scorer import Reference

from inspect_sentinel import HumanAnswer, handle_escalation, human, sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._integration import Sentinels, resolve_sentinel
from inspect_sentinel._protocols import threshold
from inspect_sentinel._report import Action, Decision, Observation, Report, Reported
from inspect_sentinel._runner import run_sentinel
from inspect_sentinel._step import BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol
from tests._fakes import FakeHost, ListRecorder, after_step, before_step, host_context


@protocol
def he_rule(
    action: Action | None = "escalate", explanation: str | None = "unsure"
) -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        if action is None:
            return None
        cite = f"[{explanation}]"
        return Decision(
            action=action,
            explanation=explanation,
            references=[Reference(type="message", id=cite, cite=cite)],
        )

    return decide


@monitor
def he_score(value: float) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value, "scored")

    return check


async def _run(
    spec: Sentinels, step: Step, *answers: HumanAnswer
) -> tuple[Decision | None, list[Reported[Report]], FakeHost]:
    host = FakeHost(*answers)
    recorder = ListRecorder()
    decision = await run_sentinel(
        resolve_sentinel(spec), host_context(recorder=recorder, host=host), step
    )
    return decision, [r.reported for r in recorder.records], host


def _handled(records: list[Reported[Report]]) -> list[Reported[Report]]:
    return [r for r in records if r.name == "handle_escalation"]


@pytest.mark.parametrize("action", ["reject", "escalate", "modify", ""])
def test_the_action_is_terminate_or_continue(action: Any) -> None:
    with pytest.raises(ValueError, match="'terminate' or 'continue'"):
        handle_escalation(action)


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_step(), after_step()])
@pytest.mark.parametrize(
    "spec",
    [
        lambda: handle_escalation("terminate"),
        lambda: sequential([he_rule(None), handle_escalation("terminate")]),
    ],
    ids=["at_the_root", "after_an_abstaining_link"],
)
async def test_it_abstains_with_nothing_to_handle(spec: Any, step: Step) -> None:
    decision, records, _ = await _run(spec(), step)
    assert decision is None
    assert not _handled(records)


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_step(), after_step()])
@pytest.mark.parametrize("action", ["terminate", "continue"])
async def test_it_answers_the_escalation_before_it(action: Any, step: Step) -> None:
    decision, records, _ = await _run(
        sequential([he_rule("escalate", "unsure"), handle_escalation(action)]), step
    )
    assert decision is not None
    assert decision.action == action
    assert decision.explanation == "escalated by he_rule: unsure"
    assert [r.cite for r in decision.references] == ["[unsure]"]
    [handled] = _handled(records)
    assert handled.report == decision
    stage = "tool_call" if isinstance(step, BeforeToolCall) else "tool_result"
    assert handled.function == stage


@pytest.mark.anyio
async def test_its_explanation_names_each_escalation() -> None:
    decision, _, _ = await _run(
        sequential(
            {
                "first": he_rule("escalate", "deletes files"),
                "second": he_rule("escalate", None),
                "third": he_rule("escalate", "suspicion 0.6"),
                "policy": handle_escalation("continue"),
            }
        ),
        before_step(),
    )
    assert decision is not None
    assert decision.action == "continue"
    assert (
        decision.explanation
        == "escalated by first: deletes files; second; third: suspicion 0.6"
    )
    assert [r.cite for r in decision.references] == [
        "[deletes files]",
        "[None]",
        "[suspicion 0.6]",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(("score", "expected"), [(0.9, "reject"), (0.1, "continue")])
async def test_after_threshold_the_threshold_decides(
    score: float, expected: str
) -> None:
    decision, records, _ = await _run(
        sequential(
            [threshold(he_score(score), reject_at=0.5), handle_escalation("terminate")]
        ),
        before_step(),
    )
    assert decision is not None
    assert decision.action == expected
    assert not _handled(records)


@pytest.mark.anyio
async def test_after_human_it_handles_the_stages_the_person_is_not_asked_at() -> None:
    spec = sequential(
        [
            he_rule("escalate", "unsure"),
            human(stages=["tool_call"]),
            handle_escalation("terminate"),
        ]
    )
    before, before_records, host = await _run(
        spec, before_step(), HumanAnswer(decision="approve")
    )
    assert before is not None
    assert before.action == "continue"
    assert len(host.asked) == 1
    assert not _handled(before_records)

    after, after_records, host = await _run(spec, after_step())
    assert after is not None
    assert after.action == "terminate"
    assert after.explanation == "escalated by he_rule: unsure"
    assert not host.asked
    assert len(_handled(after_records)) == 1
