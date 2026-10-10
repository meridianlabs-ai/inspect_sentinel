import logging
from collections.abc import Callable

import pytest
from inspect_ai.scorer import Reference

from inspect_sentinel import (
    _escalation,
    concurrent,
    handle_escalation,
    human,
    observe_only,
    sequential,
    threshold,
)
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._final import decide_final
from inspect_sentinel._integration import Sentinels, resolve_sentinel, run_sentinel
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol, ProtocolGroup
from tests._fakes import ListRecorder, after_step, before_step, host_context


@monitor
def ue_score() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    return check


@protocol
def ue_before() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.proceed()

    return decide


@protocol
def ue_any() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision.proceed()

    return decide


@protocol
def ue_group() -> ProtocolGroup:
    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.proceed()

    return ProtocolGroup(after)


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


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    # the warning is once per process for each message
    monkeypatch.setattr(_escalation, "_warned", set[str]())


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _limit() -> Protocol:
    return threshold(ue_score(), reject_at=0.5)


HANDLED: list[tuple[str, Callable[[], Sentinels]]] = [
    ("threshold", _limit),
    ("observe_only", lambda: observe_only(ue_score())),
    ("human", lambda: human(stages=["tool_call", "tool_result"])),
    ("handle_escalation", lambda: handle_escalation("terminate")),
    ("rule then human", lambda: sequential([ue_before(), human(stages=["tool_call"])])),
    (
        "rule then handle_escalation",
        lambda: sequential([ue_any(), handle_escalation("continue")]),
    ),
    (
        "rule then human then handle_escalation",
        lambda: sequential(
            [ue_any(), human(stages=["tool_call"]), handle_escalation("terminate")]
        ),
    ),
    (
        "a handled chain beside a threshold",
        lambda: {
            "gate": sequential([ue_any(), handle_escalation("terminate")]),
            "llm": _limit(),
        },
    ),
    (
        "rule then a concurrent that handles",
        lambda: sequential(
            [ue_before(), concurrent([human(stages=["tool_call"]), _limit()])]
        ),
    ),
]


@pytest.mark.parametrize("spec", [s for _, s in HANDLED], ids=[n for n, _ in HANDLED])
def test_a_configuration_that_handles_its_escalations_does_not_warn(
    spec: Callable[[], Sentinels], caplog: pytest.LogCaptureFixture
) -> None:
    resolve_sentinel(spec())
    assert not _warnings(caplog)


UNHANDLED: list[tuple[str, Callable[[], Sentinels], str]] = [
    ("a lone custom rule", ue_before, "ue_before (tool_call)"),
    (
        "a custom group",
        lambda: [ue_group(), _limit()],
        "ue_group (tool_result)",
    ),
    (
        "a rule beside a threshold",
        lambda: {"rule": ue_any(), "llm": _limit()},
        "rule (tool_call, tool_result)",
    ),
    (
        "a chain ending in a threshold",
        lambda: {"attempt": sequential([ue_before(), _limit()])},
        "attempt/ue_before (tool_call)",
    ),
    (
        "a human that asks at one stage",
        lambda: sequential([ue_any(), human(stages=["tool_call"])]),
        "ue_any (tool_result)",
    ),
    (
        "a rule after the handler",
        lambda: sequential([handle_escalation("terminate"), ue_before()]),
        "ue_before (tool_call)",
    ),
    (
        "a human beside the rule",
        lambda: [ue_before(), human(stages=["tool_call"])],
        "ue_before (tool_call)",
    ),
    (
        "a concurrent whose peer escalates",
        lambda: sequential(
            {
                "rule": ue_before(),
                "last": concurrent([human(stages=["tool_call"]), ue_before()]),
            }
        ),
        "last/ue_before (tool_call), rule (tool_call)",
    ),
]


@pytest.mark.parametrize(
    ("spec", "sources"),
    [(s, w) for _, s, w in UNHANDLED],
    ids=[n for n, _, _ in UNHANDLED],
)
def test_a_root_that_can_escalate_unhandled_warns_naming_where(
    spec: Callable[[], Sentinels], sources: str, caplog: pytest.LogCaptureFixture
) -> None:
    resolve_sentinel(spec())
    [warning] = _warnings(caplog)
    assert f"ends the sample as an unhandled escalation: {sources}." in warning
    assert "handle_escalation('terminate' | 'continue')" in warning


def test_the_warning_is_logged_once_for_a_configuration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolve_sentinel(ue_before())
    resolve_sentinel(ue_before())
    resolve_sentinel(ue_any())
    assert len(_warnings(caplog)) == 2


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
