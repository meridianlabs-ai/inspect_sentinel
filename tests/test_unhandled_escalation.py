from collections.abc import Callable

import pytest
from inspect_ai.scorer import Reference

from inspect_sentinel import handle_escalation, human, sequential, threshold
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._final import decide_final
from inspect_sentinel._integration import Sentinels, resolve_sentinel, run_sentinel
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._step import BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol
from tests._fakes import ListRecorder, after_step, before_step, host_context


@monitor
def ue_score() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    return check


@protocol
def ue_escalates(explanation: str | None = "unsure") -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        cite = f"[{explanation}]"
        return Decision.escalate(
            explanation, references=[Reference(type="message", id=cite, cite=cite)]
        )

    return decide


@protocol
def ue_final() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        decide_final(Decision.escalate("final"))

    return decide


def _limit() -> Protocol:
    return threshold(ue_score(), reject_at=0.5)


GUIDANCE = "To say what an escalation means, end the configuration with human() or handle_escalation('terminate' | 'continue') in a sequential()."

CONVERTED: list[tuple[str, Callable[[], Sentinels], Step, str, list[str]]] = [
    (
        "a lone rule",
        lambda: ue_escalates(),
        before_step(),
        "ue_escalates: unsure",
        ["[unsure]"],
    ),
    (
        "every link of a chain",
        lambda: sequential({"a": ue_escalates("first"), "b": ue_escalates(None)}),
        before_step(),
        "a: first; b",
        ["[first]", "[None]"],
    ),
    (
        "a nested link beside a threshold",
        lambda: {"attempt": sequential({"triage": ue_escalates()}), "llm": _limit()},
        before_step(),
        "attempt/triage: unsure",
        ["[unsure]"],
    ),
    (
        "not an escalation a nested chain handled",
        lambda: {
            "gate": sequential(
                {"x": ue_escalates("handled"), "h": handle_escalation("continue")}
            ),
            "rule": ue_escalates(),
        },
        before_step(),
        "rule: unsure",
        ["[unsure]"],
    ),
    (
        "a stage the person is not asked at",
        lambda: sequential({"rule": ue_escalates(), "h": human(stages=["tool_call"])}),
        after_step(),
        "rule: unsure",
        ["[unsure]"],
    ),
    (
        "a final escalate",
        lambda: {"f": ue_final(), "llm": _limit()},
        before_step(),
        "f: final",
        [],
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("spec", "step", "named", "cites"),
    [(s, st, n, c) for _, s, st, n, c in CONVERTED],
    ids=[n for n, *_ in CONVERTED],
)
async def test_an_escalate_at_the_root_is_returned_as_a_terminate_naming_where_it_came_from(
    spec: Callable[[], Sentinels], step: Step, named: str, cites: list[str]
) -> None:
    recorder = ListRecorder()
    decision = await run_sentinel(
        resolve_sentinel(spec()), host_context(recorder=recorder), step
    )
    assert decision is not None
    assert decision.action == "terminate"
    assert decision.explanation == f"unhandled escalation from {named}. {GUIDANCE}"
    assert [r.cite for r in decision.references] == cites
    # what is recorded is the escalate, which tells this apart from a
    # protocol's own terminate
    actions = [
        r.reported.report.action
        for r in recorder.records
        if isinstance(r.reported.report, Decision)
    ]
    assert "escalate" in actions and "terminate" not in actions
