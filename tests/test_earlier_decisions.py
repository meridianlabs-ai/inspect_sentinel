from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

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
    run_protocols,
    sequential,
)
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._integration import (
    Sentinels,
    earlier_decision,
    resolve_sentinel,
    run_sentinel,
    supersedes,
)
from inspect_sentinel._protocols import concurrent, handle_escalation, threshold
from inspect_sentinel._report import Decision, Observation, Reported, origin
from inspect_sentinel._step import AfterGenerate, BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol
from inspect_sentinel._unhandled import Invocation
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
    return _settled(recorder)


def _settled(recorder: ListRecorder) -> list[EarlierDecision]:
    # what a host publishes once the step settles: the kept decisions, less the
    # ones a supersession withdrew
    found = (
        earlier_decision(r.factory, r.step, "step_1", r.reported)
        for r in recorder.records
    )
    kept = [record for record in found if record is not None]
    for s in recorder.supersessions:
        withdrawn = cast(Reported[Decision], s.reported)
        kept = [k for k in kept if not supersedes(s.context.path, withdrawn, k)]
    return kept


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


class Answering(FakeHost):
    """Answers each prompt with the given answers in turn, and says when it has."""

    def __init__(self, *answers: HumanAnswer) -> None:
        super().__init__(*answers)
        self.answered = anyio.Event()

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        answer = await super().ask_human(step, choices)
        self.answered.set()
        return answer


@protocol
def final_after(event: Any) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        # outlives a sibling's terminate, which cancels it
        with anyio.CancelScope(shield=True):
            await event.wait()
            decide_final(Decision.proceed())
        return None

    return decide


@protocol
def forwards(child: Any, done: Any) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        strongest = (await run_protocols(child, context, step)).strongest()
        done.set()
        return strongest.report if strongest is not None else None

    return decide


@protocol
def proceeds() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.proceed()

    return decide


@protocol
def terminates_after(event: Any) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await event.wait()
        return Decision.terminate("peer")

    return decide


@protocol
def rejects() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("peer")

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("case", "answer", "kept_actions"),
    [
        # the person's terminate, superseded where it was made
        ("direct", "terminate", []),
        # passed up as a copy carrying the vote, then superseded above
        ("forwarded_copy", "terminate", []),
        # a peer's terminate passed up and superseded: the person's approve stands
        ("peer_superseded", "approve", ["proceed"]),
        # an ordinary vote outranks the person's approve; nothing is superseded
        ("outvoted", "approve", ["proceed"]),
    ],
)
async def test_only_a_supersession_of_the_persons_decision_withdraws_it(
    case: str, answer: str, kept_actions: list[str]
) -> None:
    host = Answering(HumanAnswer(answer, reason="because"))
    done = anyio.Event()
    person = human(stages=["tool_call"])
    match case:
        case "direct":
            spec: Sentinels = concurrent([person, final_after(host.answered)])
        case "forwarded_copy":
            inner = concurrent({"first_vote": proceeds(), "person": person})
            spec = concurrent([forwards(inner, done), final_after(done)])
        case "peer_superseded":
            inner = concurrent(
                {"person": person, "peer": terminates_after(host.answered)}
            )
            spec = concurrent([forwards(inner, done), final_after(done)])
        case _:
            spec = concurrent([person, rejects()])
    recorder = ListRecorder()
    with anyio.fail_after(5):
        await run_sentinel(
            resolve_sentinel(spec),
            host_context(recorder=recorder, host=host),
            before_step(),
        )

    if case == "forwarded_copy":
        # the decision superseded is a copy of the person's, carrying the vote
        [superseded] = recorder.supersessions
        [mine] = [r for r in recorder.records if r.reported.name == "person"]
        assert superseded.reported.report is not mine.reported.report
        assert "first_vote: proceed" in (superseded.reported.report.explanation or "")
    assert bool(recorder.supersessions) == (case != "outvoted")
    assert [k.decided.report.action for k in _settled(recorder)] == kept_actions


@pytest.mark.anyio
async def test_a_kept_decision_carries_none_of_its_steps_provenance() -> None:
    spec = sequential([threshold(loud(), escalate_at=0.5), human(stages=["tool_call"])])
    recorder = ListRecorder()
    host = FakeHost(HumanAnswer("approve"))
    await run_sentinel(
        resolve_sentinel(spec),
        host_context(recorder=recorder, host=host),
        before_step(),
    )
    [mine] = [r.reported for r in recorder.records if r.reported.name == "human"]
    # the person's own decision reaches threshold's scores through the runner's
    # record of the step
    assert isinstance(mine.report, Decision)
    assert _REASONING in str(_reachable_metadata(mine.report))
    [kept] = _settled(recorder)
    assert _reachable_metadata(kept.decided.report) == []
    assert kept.decided.report == mine.report


def _reachable_metadata(decision: Decision) -> list[object]:
    step, _ = origin(decision)
    nodes = cast(list[Invocation], step) if step is not None else []
    return [
        node.decided.report.metadata
        for node in nodes
        if node.decided is not None and node.decided.report.metadata
    ]
