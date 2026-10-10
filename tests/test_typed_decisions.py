from collections.abc import Callable
from typing import Any

import pytest
from inspect_ai.tool import ToolCall
from pydantic import BaseModel

from inspect_sentinel import (
    Escalate,
    Modify,
    Proceed,
    Reject,
    Terminate,
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
from inspect_sentinel._report import Action, Decision, Observation
from inspect_sentinel._step import BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol, ProtocolGroup
from tests._fakes import ListRecorder, after_step, before_step, host_context

CALL = ToolCall(id="call_1", function="bash", arguments={"cmd": "ls"})

CLASSES: list[tuple[Action, type[Decision], dict[str, Any]]] = [
    ("continue", Proceed, {}),
    ("reject", Reject, {}),
    ("terminate", Terminate, {}),
    ("modify", Modify, {"modified": CALL}),
    ("escalate", Escalate, {}),
]


@pytest.mark.parametrize(
    ("action", "cls", "fields"), CLASSES, ids=[a for a, _, _ in CLASSES]
)
def test_a_decision_is_built_as_its_actions_class(
    action: Action, cls: type[Decision], fields: dict[str, Any]
) -> None:
    class Holder(BaseModel):
        decision: Decision

    built = Decision(action=action, explanation="why", **fields)
    dumped = built.model_dump()
    assert type(built) is cls
    assert dumped["action"] == action
    assert type(Decision.model_validate(dumped)) is cls
    assert type(Decision.model_validate_json(built.model_dump_json())) is cls
    assert type(Holder.model_validate({"decision": dumped}).decision) is cls
    assert built == cls(explanation="why", **fields)


def test_the_constructors_return_their_classes() -> None:
    assert type(Decision.proceed()) is Proceed
    assert type(Decision.reject()) is Reject
    assert type(Decision.terminate()) is Terminate
    assert type(Decision.escalate()) is Escalate


def test_a_class_holds_only_its_action() -> None:
    with pytest.raises(ValueError, match="'reject'"):
        Reject(action="continue")  # pyright: ignore[reportArgumentType]


@monitor
def td_score(value: float = 0.1) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value)

    return check


@protocol
def td_escalates() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Escalate | Proceed:
        return Decision.escalate("unsure")

    return decide


@protocol
def td_escalates_anywhere() -> Protocol:
    async def decide(context: Context, step: Step) -> Escalate | Proceed | None:
        return Decision.escalate("unsure")

    return decide


@protocol
def td_may_escalate() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.escalate("unsure")

    return decide


@protocol
def td_terminates() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Terminate:
        return Decision.terminate("stop")

    return decide


@protocol
def td_group() -> ProtocolGroup:
    async def first(context: Context, step: BeforeToolCall) -> Reject | Proceed:
        return Decision.proceed()

    async def second(context: Context, step: BeforeToolCall) -> Escalate | None:
        return None

    return ProtocolGroup(first, second)


@protocol
def td_forwards(child: Protocol) -> Protocol:
    # a composition of its own, judged by its annotation alone
    async def decide(context: Context, step: Step) -> Decision | None:
        return None

    return decide


def _limit() -> Protocol:
    return threshold(td_score(), reject_at=0.5)


UNHANDLED: list[tuple[str, Callable[[], Sentinels], str]] = [
    ("a lone rule", td_escalates, "td_escalates (tool_call)"),
    (
        "a chain ending in a threshold, which can abstain",
        lambda: sequential({"triage": td_escalates(), "llm": _limit()}),
        "triage (tool_call)",
    ),
    (
        "a nested chain",
        lambda: {"attempt": sequential({"triage": td_escalates_anywhere()})},
        "attempt/triage (tool_call, tool_result)",
    ),
    (
        "a stage the person is not asked at",
        lambda: sequential([td_escalates_anywhere(), human(stages=["tool_call"])]),
        "td_escalates_anywhere (tool_result)",
    ),
    (
        "beside a person, whose approve an escalate outranks",
        lambda: [td_escalates(), human(stages=["tool_call"])],
        "td_escalates (tool_call)",
    ),
    ("a function of a group", lambda: [td_group(), _limit()], "td_group (tool_call)"),
]


@pytest.mark.parametrize(
    ("spec", "sources"),
    [(s, w) for _, s, w in UNHANDLED],
    ids=[n for n, _, _ in UNHANDLED],
)
def test_an_annotated_escalate_with_nothing_to_handle_it_is_a_configuration_error(
    spec: Callable[[], Sentinels], sources: str
) -> None:
    with pytest.raises(ValueError) as raised:
        resolve_sentinel(spec())
    message = str(raised.value)
    assert f"which would end the sample: {sources}." in message
    assert "handle_escalation('terminate' | 'continue')" in message


HANDLED: list[tuple[str, Callable[[], Sentinels]]] = [
    ("threshold", _limit),
    ("observe_only", lambda: observe_only(td_score())),
    ("human", lambda: human(stages=["tool_call", "tool_result"])),
    ("handle_escalation", lambda: handle_escalation("terminate")),
    (
        "a rule then a person",
        lambda: sequential([td_escalates(), human(stages=["tool_call"])]),
    ),
    (
        "a rule then handle_escalation",
        lambda: sequential([td_escalates_anywhere(), handle_escalation("continue")]),
    ),
    (
        "a rule after a link that always decides, so never reached",
        lambda: sequential([handle_escalation("continue"), td_escalates()]),
    ),
    (
        "beside a peer that always outranks it",
        lambda: [td_escalates(), td_terminates()],
    ),
    (
        "a rule then a concurrent that always decides",
        lambda: sequential(
            [td_escalates(), concurrent([human(stages=["tool_call"]), _limit()])]
        ),
    ),
    ("an unannotated rule, which the run-time terminate covers", td_may_escalate),
    (
        "an annotated rule inside a composition of one's own",
        lambda: td_forwards(td_escalates()),
    ),
]


@pytest.mark.parametrize("spec", [s for _, s in HANDLED], ids=[n for n, _ in HANDLED])
def test_a_configuration_that_handles_its_annotated_escalates_builds(
    spec: Callable[[], Sentinels],
) -> None:
    resolve_sentinel(spec())


@protocol
def td_returns(action: Action | None) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Reject | Proceed:
        return None if action is None else Decision(action=action)  # pyright: ignore[reportReturnType]

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "returned"),
    [("terminate", r"a Terminate \('terminate'\)"), (None, "None, abstaining")],
)
async def test_returning_outside_the_annotation_fails_the_step(
    action: Action | None, returned: str
) -> None:
    with pytest.raises(
        TypeError,
        match=f"protocol 'td_returns' returned {returned}, which its return annotation does not allow; it is annotated to return Proceed | Reject.",
    ):
        await run_sentinel(td_returns(action), host_context(), before_step())


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["reject", "continue"])
async def test_returning_inside_the_annotation_decides(action: Action) -> None:
    decision = await run_sentinel(td_returns(action), host_context(), before_step())
    assert decision is not None and decision.action == action


@protocol
def td_finalizes() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Proceed | None:
        decide_final(Decision.reject("final"))

    return decide


@pytest.mark.anyio
async def test_a_final_decision_is_not_held_to_the_return_annotation() -> None:
    recorder = ListRecorder()
    decision = await run_sentinel(
        td_finalizes(), host_context(recorder=recorder), before_step()
    )
    assert decision == Decision.reject("final")


def test_postponed_annotations_are_resolved() -> None:
    @protocol
    def td_postponed() -> Protocol:
        async def decide(context: Context, step: BeforeToolCall) -> "Reject | None":
            return None

        return decide

    resolve_sentinel(td_postponed())


@pytest.mark.parametrize(
    "annotation", [Observation, int, Reject | str, Observation | None]
)
def test_a_protocol_annotated_to_return_something_else_is_rejected(
    annotation: Any,
) -> None:
    @protocol
    def td_wrong() -> Protocol:
        async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
            return None

        decide.__annotations__["return"] = annotation
        return decide

    with pytest.raises(
        TypeError, match="Proceed, Reject, Terminate, Modify and Escalate"
    ):
        td_wrong()


@pytest.mark.anyio
async def test_a_protocol_annotated_none_never_decides() -> None:
    @protocol
    def td_silent() -> Protocol:
        async def decide(context: Context, step: Step) -> None:
            return None

        return decide

    recorder = ListRecorder()
    assert (
        await run_sentinel(td_silent(), host_context(recorder=recorder), after_step())
        is None
    )
    assert not recorder.records


@protocol
def td_modifies() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Modify:
        return Modify(modified=CALL)

    return decide


@pytest.mark.anyio
async def test_a_contested_modify_becomes_a_reject() -> None:
    decision = await run_sentinel(
        concurrent({"edit": td_modifies(), "stop": td_terminates()}),
        host_context(),
        before_step(),
    )
    assert type(decision) is Terminate
    decision = await run_sentinel(
        concurrent({"edit": td_modifies(), "ok": td_returns("continue")}),
        host_context(),
        before_step(),
    )
    assert type(decision) is Reject
    assert decision.modified is None
