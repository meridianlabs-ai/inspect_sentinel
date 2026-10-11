from collections.abc import Callable
from typing import Any, cast

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
from tests._fakes import ListRecorder, after_step, before_step, decision, host_context

CALL = ToolCall(id="call_1", function="bash", arguments={"cmd": "ls"})

CLASSES: list[tuple[Action, type[Decision], Decision]] = [
    ("continue", Proceed, Decision.proceed("why")),
    ("reject", Reject, Decision.reject("why")),
    ("terminate", Terminate, Decision.terminate("why")),
    ("modify", Modify, Modify(modified=CALL, explanation="why")),
    ("escalate", Escalate, Decision.escalate("why")),
]


@pytest.mark.parametrize(
    ("action", "cls", "built"), CLASSES, ids=[a for a, _, _ in CLASSES]
)
def test_each_action_has_a_class_that_logs_its_action(
    action: Action, cls: type[Decision], built: Decision
) -> None:
    class Holder(BaseModel):
        decision: Proceed | Reject | Terminate | Modify | Escalate

    dumped = built.model_dump()
    assert type(built) is cls
    assert dumped["action"] == action
    assert cls.model_validate_json(built.model_dump_json()) == built
    assert type(Holder.model_validate({"decision": dumped}).decision) is cls


def test_the_base_class_is_not_built() -> None:
    with pytest.raises(
        TypeError, match="Proceed, Reject, Terminate, Modify or Escalate"
    ):
        Decision(action="reject")


def test_a_decision_is_frozen() -> None:
    decision = Decision.reject("why")
    with pytest.raises(ValueError, match="frozen"):
        decision.action = "continue"  # pyright: ignore[reportAttributeAccessIssue]
    assert decision.model_copy(update={"explanation": "how"}).explanation == "how"


def test_a_modify_needs_its_replacement() -> None:
    with pytest.raises(ValueError, match="modified"):
        Modify()  # pyright: ignore[reportCallIssue]


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


@protocol
def td_any_action() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall
    ) -> Proceed | Reject | Terminate | Modify | Escalate:
        return Decision.escalate("unsure")

    return decide


@protocol
def td_any_action_or_none() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall
    ) -> Proceed | Reject | Terminate | Modify | Escalate | None:
        return Decision.escalate("unsure")

    return decide


@protocol
def td_rejects() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Reject:
        return Decision.reject("no")

    return decide


@protocol
def td_maybe_proceeds() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Proceed | None:
        return Decision.proceed()

    return decide


@protocol
def td_grouped(mode: str) -> ProtocolGroup:
    # a group whose one function is a composition another factory configured
    inner = {"inner": td_escalates()}
    return ProtocolGroup(
        sequential(inner) if mode == "sequential" else concurrent(inner)
    )


@protocol
def td_regrouped() -> ProtocolGroup:
    # a factory of one's own returning another factory's group
    return td_grouped("sequential")


@protocol
def td_named_chain() -> Protocol:
    # a composition registered under its own name, checked through its children
    return sequential({"triage": td_escalates(), "llm": _limit()})


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
    ("a chain a factory of one's own returns", td_named_chain, "triage (tool_call)"),
    (
        "a chain that is a function of a group",
        lambda: td_grouped("sequential"),
        "td_grouped/inner (tool_call)",
    ),
    (
        "a concurrent that is a function of a group",
        lambda: td_grouped("concurrent"),
        "td_grouped/inner (tool_call)",
    ),
    (
        "a group a factory of one's own returns",
        td_regrouped,
        "td_regrouped/inner (tool_call)",
    ),
    ("every decision class named", td_any_action, "td_any_action (tool_call)"),
    (
        "every decision class named, with None",
        td_any_action_or_none,
        "td_any_action_or_none (tool_call)",
    ),
    (
        "beside a chain that may proceed before it rejects",
        lambda: {
            "e": td_escalates(),
            "r": sequential([td_maybe_proceeds(), td_rejects()]),
        },
        "e (tool_call)",
    ),
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
        "a rule then handle_escalation as then",
        lambda: sequential(
            [td_escalates_anywhere()], then=handle_escalation("terminate")
        ),
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
        "beside a chain that always rejects",
        lambda: {"e": td_escalates(), "r": sequential([td_rejects()])},
    ),
    (
        "a grouped chain before handle_escalation",
        lambda: sequential([td_grouped("sequential"), handle_escalation("continue")]),
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
        return None if action is None else decision(action)  # pyright: ignore[reportReturnType]

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


@protocol
def td_copies(route: str) -> Protocol:
    # decisions derived with model_copy, whose updates are not validated
    async def decide(context: Context, step: BeforeToolCall) -> Modify | Reject:
        if route == "drop_replacement":
            return Modify(modified=step.call).model_copy(update={"modified": None})
        if route == "change_action":
            return cast(
                Reject, Decision.proceed().model_copy(update={"action": "reject"})
            )
        return Modify(modified=step.call).model_copy(update={"explanation": "same"})

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("route", "match"),
    [
        (
            "drop_replacement",
            "`modified` must be set exactly when the action is `modify`",
        ),
        (
            "change_action",
            "A Proceed cannot have action 'reject'; build a Reject instead",
        ),
    ],
)
async def test_a_copied_decision_is_checked_at_the_boundary(
    route: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        await run_sentinel(td_copies(route), host_context(), before_step())


@pytest.mark.anyio
async def test_a_valid_copy_decides() -> None:
    decision = await run_sentinel(td_copies("keep"), host_context(), before_step())
    assert type(decision) is Modify
    assert decision.explanation == "same"


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
    # nobody decided, so the step proceeds
    decision = await run_sentinel(
        td_silent(), host_context(recorder=recorder), after_step()
    )
    assert decision == Decision.proceed()
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
