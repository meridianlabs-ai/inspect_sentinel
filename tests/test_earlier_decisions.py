from collections.abc import Sequence
from dataclasses import replace

import anyio
import pytest
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.model._model_output import ChatCompletionChoice
from inspect_ai.tool import ToolCall

from inspect_sentinel import (
    EarlierDecision,
    HumanAnswer,
    decide_final,
    human,
    sequential,
)
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._integration import (
    Sentinels,
    earlier_decision,
    resolve_sentinel,
    run_sentinel,
)
from inspect_sentinel._protocols import concurrent, handle_escalation, threshold
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._step import AfterGenerate, BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol
from tests._fakes import (
    FakeHost,
    ListRecorder,
    after_generate_step,
    after_step,
    before_step,
    host_context,
)

_REASONING = "the call posts credentials to an unknown host"


@monitor
def loud() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.93, _REASONING)

    return check


@monitor
def loud_turn() -> Monitor:
    async def check(context: Context, step: AfterGenerate) -> Observation | None:
        return Observation.score(0.93, _REASONING)

    return check


@protocol
def escalates() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.escalate("unsure")

    return decide


async def _kept(
    spec: Sentinels, step: Step, *answers: HumanAnswer
) -> list[EarlierDecision]:
    # what a host keeps for later steps from every report the runner recorded
    recorder = ListRecorder()
    context = host_context(recorder=recorder, host=FakeHost(*answers))
    await run_sentinel(resolve_sentinel(spec), context, step)
    kept = (
        earlier_decision(r.factory, r.step, "step_1", r.reported)
        for r in recorder.records
    )
    return [record for record in kept if record is not None]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("answer", "action", "explanation", "message"),
    [
        (HumanAnswer("approve"), "proceed", None, None),
        (
            HumanAnswer("reject", reason="not that host"),
            "reject",
            "not that host",
            "not that host",
        ),
        (HumanAnswer("terminate", reason="stop"), "terminate", "stop", None),
    ],
)
async def test_only_the_persons_decision_is_kept(
    answer: HumanAnswer, action: str, explanation: str | None, message: str | None
) -> None:
    spec = sequential([threshold(loud(), escalate_at=0.5), human(stages=["tool_call"])])
    [kept] = await _kept(spec, before_step(), answer)
    assert (kept.stage, kept.step_id, kept.call_ids) == (
        "tool_call",
        "step_1",
        ("call_1",),
    )
    assert (kept.decided.name, kept.decided.path) == ("human", "human")
    report = kept.decided.report
    assert (report.action, report.explanation, report.message) == (
        action,
        explanation,
        message,
    )


@pytest.mark.anyio
async def test_no_monitor_score_reaches_the_kept_decision() -> None:
    # threshold's escalation carries every score in its metadata; the person's
    # decision carries only their answer
    spec = sequential([threshold(loud(), escalate_at=0.5), human(stages=["tool_call"])])
    host = FakeHost(HumanAnswer("approve", reason="fine"))
    await run_sentinel(resolve_sentinel(spec), host_context(host=host), before_step())
    assert _REASONING in repr(host.asked[0].step.escalations)
    [kept] = await _kept(spec, before_step(), HumanAnswer("approve", reason="fine"))
    report = kept.decided.report
    assert report.metadata is None
    assert report.references == []
    assert _REASONING not in repr(kept)
    assert "0.93" not in repr(kept)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "spec",
    [
        sequential([escalates(), handle_escalation("proceed")]),
        threshold(loud(), escalate_at=0.95, reject_at=0.99),
        concurrent([escalates(), handle_escalation("terminate")]),
    ],
    ids=["handle_escalation", "threshold", "unhandled_escalation"],
)
async def test_other_protocols_decisions_are_not_kept(spec: Sentinels) -> None:
    assert await _kept(spec, before_step()) == []


@pytest.mark.anyio
async def test_a_decision_after_a_call_is_kept_for_that_call() -> None:
    [kept] = await _kept(
        human(stages=["tool_result"]), after_step(), HumanAnswer("approve")
    )
    assert (kept.stage, kept.call_ids) == ("tool_result", ("call_1",))


def _turn(*ids: str) -> AfterGenerate:
    message = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id=id, function="bash", arguments={}) for id in ids],
    )
    output = ModelOutput(model="m", choices=[ChatCompletionChoice(message=message)])
    return replace(after_generate_step(), output=output)


@pytest.mark.anyio
@pytest.mark.parametrize("ids", [("a", "b"), ()])
async def test_a_turn_decision_covers_each_of_its_calls(ids: tuple[str, ...]) -> None:
    spec = sequential(
        [threshold(loud_turn(), escalate_at=0.5), human(stages=["model_output"])]
    )
    [kept] = await _kept(spec, _turn(*ids), HumanAnswer("approve"))
    assert (kept.stage, kept.call_ids) == ("model_output", ids)
    assert _REASONING not in repr(kept)


@pytest.mark.anyio
async def test_a_superseded_decision_is_the_report_a_host_kept() -> None:
    # a host drops a kept decision when the runner supersedes it, matching the
    # report it kept
    started, answered = anyio.Event(), anyio.Event()

    class Host(FakeHost):
        async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
            await started.wait()
            answered.set()
            return HumanAnswer("terminate", reason="stop")

    @protocol
    def final_after_the_person() -> Protocol:
        async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
            with anyio.CancelScope(shield=True):
                started.set()
                await answered.wait()
                decide_final(Decision.proceed())
            return None

        return decide

    recorder = ListRecorder()
    context = host_context(recorder=recorder, host=Host())
    spec = concurrent([human(stages=["tool_call"]), final_after_the_person()])
    with anyio.fail_after(5):
        decision = await run_sentinel(resolve_sentinel(spec), context, before_step())

    assert decision is not None and decision.action == "proceed"
    kept = [
        record
        for r in recorder.records
        if (record := earlier_decision(r.factory, r.step, "call_1", r.reported))
    ]
    [superseded] = recorder.supersessions
    assert [k.decided.report for k in kept] == [superseded.reported.report]
    assert kept[0].decided.report is superseded.reported.report
