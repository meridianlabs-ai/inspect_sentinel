from collections.abc import Callable
from typing import Literal

import anyio
import pytest
from inspect_ai.scorer import Reference
from inspect_ai.util import StoreModel

from inspect_sentinel import (
    concurrent,
    handle_escalation,
    human,
    sequential,
    threshold,
)
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._final import decide_final
from inspect_sentinel._integration import Sentinels, resolve_sentinel, run_sentinel
from inspect_sentinel._report import Decision, Observation, Reported
from inspect_sentinel._runner import run_protocols
from inspect_sentinel._step import BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol, ProtocolGroup
from tests._fakes import ListRecorder, after_step, before_step, host_context


@monitor
def ue_score() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    return check


@protocol
def ue_escalates(explanation: str | None = "unsure") -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        cite = f"[{explanation}]"
        return Decision.escalate(
            explanation, references=[Reference(type="message", id=cite, cite=cite)]
        )

    return decide


@protocol
def ue_final() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        decide_final(Decision.escalate("final"))

    return decide


def _limit() -> Protocol | ProtocolGroup:
    return threshold(ue_score(), reject_at=0.5)


GUIDANCE = "To say what an escalation means, end the configuration with human() or handle_escalation('terminate' | 'proceed') in a sequential()."

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
                {"x": ue_escalates("handled"), "h": handle_escalation("proceed")}
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


def _child_ref() -> Reference:
    return Reference(type="message", id="c", cite="[child]")


@protocol
def ue_child(reason: str = "child reason") -> Protocol:
    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        return Decision.escalate(reason, references=[_child_ref()])

    return run


@protocol
def ue_group(forwards: bool) -> ProtocolGroup:
    # two functions at one path: one runs a child and handles or forwards its
    # escalation, the other escalates on its own
    configured = ue_child()

    async def nested(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        decisions = await run_protocols({"child": configured}, context, step)
        return decisions[0].report if forwards else Decision.proceed("child handled")

    async def independent(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        return Decision.escalate(
            "independent reason",
            references=[Reference(type="message", id="o", cite="[own]")],
        )

    return ProtocolGroup(nested, independent)


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_step(), after_step()])
@pytest.mark.parametrize(
    ("forwards", "named", "cites"),
    [
        (False, "group: independent reason", ["[own]"]),
        (
            True,
            "group/child: child reason; group: independent reason",
            ["[child]", "[own]"],
        ),
    ],
    ids=["handled", "forwarded"],
)
async def test_the_functions_of_a_group_are_told_apart(
    forwards: bool, named: str, cites: list[str], step: Step
) -> None:
    decision = await run_sentinel(
        resolve_sentinel({"group": ue_group(forwards)}), host_context(), step
    )
    assert decision is not None
    assert decision.explanation == f"unhandled escalation from {named}. {GUIDANCE}"
    assert [r.cite for r in decision.references] == cites


Returns = Literal["forwarded", "relabelled", "rewritten", "equal", "carried", "own"]


@protocol
def ue_consults(returns: Returns) -> Protocol:
    # consults an escalating child, then passes its escalation up, as is or as
    # a copy, or escalates with a decision of its own
    configured = ue_child()

    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        decisions = await run_protocols({"child": configured}, context, step)
        child = decisions[0].report
        if returns == "forwarded":
            return child
        if returns == "relabelled":
            return child.model_copy(update={"metadata": {"by": "parent"}})
        if returns == "rewritten":
            return child.model_copy(update={"explanation": "child reason, seen"})
        if returns == "equal":
            return Decision.escalate("child reason", references=[_child_ref()])
        own = Reference(type="message", id="p", cite="[parent]")
        carried = child.references if returns == "carried" else []
        return Decision.escalate("parent reason", references=[*carried, own])

    return run


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_step(), after_step()])
@pytest.mark.parametrize(
    ("returns", "named", "cites"),
    [
        ("forwarded", "parent/child: child reason", ["[child]"]),
        ("relabelled", "parent/child: child reason", ["[child]"]),
        ("rewritten", "parent/child: child reason", ["[child]"]),
        (
            "equal",
            "parent/child: child reason; parent: child reason",
            ["[child]"],
        ),
        (
            "carried",
            "parent/child: child reason; parent: parent reason",
            ["[child]", "[parent]"],
        ),
        (
            "own",
            "parent/child: child reason; parent: parent reason",
            ["[child]", "[parent]"],
        ),
    ],
)
async def test_a_parent_is_named_unless_it_passed_its_childs_escalation_up(
    returns: Returns, named: str, cites: list[str], step: Step
) -> None:
    decision = await run_sentinel(
        resolve_sentinel({"parent": ue_consults(returns)}), host_context(), step
    )
    assert decision is not None
    assert decision.explanation == f"unhandled escalation from {named}. {GUIDANCE}"
    assert [r.cite for r in decision.references] == cites


@protocol
def ue_above(forwards: bool) -> Protocol:
    # above a protocol that escalates on its own after its child did: passes
    # that up, or escalates with a decision equal to the grandchild's
    configured = ue_consults("own")

    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        decisions = await run_protocols({"middle": configured}, context, step)
        if forwards:
            return decisions[0].report
        return Decision.escalate("child reason", references=[_child_ref()])

    return run


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_step(), after_step()])
@pytest.mark.parametrize(
    ("forwards", "named"),
    [
        (True, "top/middle/child: child reason; top/middle: parent reason"),
        (
            False,
            "top/middle/child: child reason; top/middle: parent reason; top: child reason",
        ),
    ],
    ids=["forwarded", "equal-to-grandchild"],
)
async def test_an_escalation_equal_to_a_grandchilds_is_named(
    forwards: bool, named: str, step: Step
) -> None:
    decision = await run_sentinel(
        resolve_sentinel({"top": ue_above(forwards)}), host_context(), step
    )
    assert decision is not None
    assert decision.explanation == f"unhandled escalation from {named}. {GUIDANCE}"
    assert [r.cite for r in decision.references] == ["[child]", "[parent]"]


@protocol
def ue_over_votes(forwards: bool) -> Protocol:
    # above a concurrent vote whose escalation it rewrites with its summary
    votes = concurrent({"rule": ue_escalates(), "llm": _limit()})

    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        decisions = await run_protocols({"votes": votes}, context, step)
        return decisions[0].report if forwards else Decision.escalate("own reason")

    return run


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("forwards", "named"),
    [
        (True, "parent/votes/rule: unsure"),
        (False, "parent/votes/rule: unsure; parent: own reason"),
    ],
    ids=["forwarded", "own"],
)
async def test_a_concurrent_vote_summary_is_not_named(
    forwards: bool, named: str
) -> None:
    decision = await run_sentinel(
        resolve_sentinel({"parent": ue_over_votes(forwards)}),
        host_context(),
        before_step(),
    )
    assert decision is not None
    assert decision.explanation == f"unhandled escalation from {named}. {GUIDANCE}"


class Calls(StoreModel):
    count: int = 0


@protocol
def ue_counts() -> Protocol:
    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        calls = context.store_as(Calls)
        calls.count += 1
        return Decision.escalate(f"call {calls.count}")

    return run


@protocol
def ue_twice() -> Protocol:
    configured = ue_counts()

    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        await run_protocols({"again": configured}, context, step)
        decisions = await run_protocols({"again": configured}, context, step)
        return decisions[0].report

    return run


@pytest.mark.anyio
async def test_each_invocation_at_one_path_is_named() -> None:
    decision = await run_sentinel(
        resolve_sentinel({"twice": ue_twice()}), host_context(), before_step()
    )
    assert decision is not None
    assert decision.explanation == (
        f"unhandled escalation from twice/again: call 1; twice/again: call 2. {GUIDANCE}"
    )


@protocol
def ue_inner() -> Protocol:
    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        return Decision.escalate(
            "inner reason",
            references=[Reference(type="message", id="i", cite="[inner]")],
        )

    return run


@protocol
def ue_outer(results: list[Decision | None], spawn: bool) -> Protocol:
    # runs another sentinel for its step, awaited or from a task it starts,
    # then escalates on its own
    async def run(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        async def inner() -> None:
            results.append(
                await run_sentinel(
                    resolve_sentinel({"inner": ue_inner()}), host_context(), step
                )
            )

        if spawn:
            async with anyio.create_task_group() as tg:
                tg.start_soon(inner)
        else:
            await inner()
        return Decision.escalate("outer reason")

    return run


@pytest.mark.anyio
@pytest.mark.parametrize("step", [before_step(), after_step()])
@pytest.mark.parametrize("spawn", [False, True], ids=["awaited", "spawned"])
async def test_a_nested_step_names_its_own_escalations(spawn: bool, step: Step) -> None:
    results: list[Decision | None] = []
    outer = await run_sentinel(
        resolve_sentinel({"outer": ue_outer(results, spawn)}), host_context(), step
    )
    [inner] = results
    assert inner is not None
    assert (
        inner.explanation
        == f"unhandled escalation from inner: inner reason. {GUIDANCE}"
    )
    assert [r.cite for r in inner.references] == ["[inner]"]
    assert outer is not None
    assert (
        outer.explanation
        == f"unhandled escalation from outer: outer reason. {GUIDANCE}"
    )
