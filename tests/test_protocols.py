from collections.abc import Callable
from typing import Any, cast

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.tool import ToolCall

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import (
    monitor,
    protocol,
    step_types,
)
from inspect_sentinel._final import decide_final
from inspect_sentinel._protocols import concurrent, observe_only, threshold
from inspect_sentinel._report import Action, Decision, Observation, Suspicion
from inspect_sentinel._runner import (
    run_protocols,
    run_sentinel,
)
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import (
    Monitor,
    Protocol,
    ProtocolGroup,
    Protocols,
)
from tests._fakes import (
    ListRecorder,
    before_step,
    host_context,
    in_step,
)


@monitor
def graded(value: Suspicion = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value)

    return check


@monitor
def silent() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    return check


@monitor
def afterwards() -> Monitor:
    async def check(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.9)

    return check


@protocol
def says(action: Action = "continue", explanation: str | None = None) -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(action=action, explanation=explanation)

    return decide


@protocol
def rewrites(explanation: str | None = None) -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return Decision(
            action="modify",
            modified=ToolCall(
                id="call_1", function="bash", arguments={"cmd": "echo hi"}
            ),
            explanation=explanation,
        )

    return decide


@protocol
def wrapper(child: Protocol) -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        await run_protocols(child, context, step)
        return Decision.proceed()

    return decide


async def _run(
    instance: Protocol, step: Step, recorder: ListRecorder
) -> Decision | None:
    with in_step(recorder=recorder) as context:
        decisions = await run_protocols(instance, context, step)
    return decisions[0].report if decisions else None


@pytest.mark.anyio
async def test_observe_records_every_observation_and_never_acts() -> None:
    recorder = ListRecorder()
    decision = await _run(
        observe_only({"low": graded(0.2), "high": graded(0.8)}), before_step(), recorder
    )
    assert decision is None
    assert sorted(r.reported.name for r in recorder.records) == ["high", "low"]


@pytest.mark.anyio
async def test_observe_records_nothing_for_a_monitor_of_another_stage() -> None:
    recorder = ListRecorder()
    assert await _run(observe_only([afterwards()]), before_step(), recorder) is None
    assert recorder.records == []


@pytest.mark.anyio
async def test_concurrent_over_monitors_only_does_not_decide() -> None:
    recorder = ListRecorder()
    assert await _run(concurrent([graded(0.3)]), before_step(), recorder) is None
    assert [r.reported.name for r in recorder.records] == ["graded"]


@pytest.mark.anyio
async def test_concurrent_passes_all_escalate_up() -> None:
    decision = await _run(
        concurrent({"a": says("escalate", "unsure"), "b": says("escalate")}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "escalate"
    assert decision.explanation == "unsure (a: escalate; b: escalate)"


@pytest.mark.anyio
async def test_nested_concurrent_passes_escalate_up() -> None:
    inner = concurrent({"unsure": says("escalate"), "fine": says("continue")})
    decision = await _run(
        concurrent({"inner": inner, "approver": says("continue")}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "escalate"


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["continue", "reject"], "reject"),
        (["reject", "terminate"], "terminate"),
        (["escalate", "continue"], "escalate"),
        (["continue", "escalate"], "escalate"),
        (["escalate", "reject"], "reject"),
    ],
)
@pytest.mark.anyio
async def test_concurrent_takes_the_strongest_decision(
    actions: list[Action], expected: Action
) -> None:
    children = {f"p{index}": says(action) for index, action in enumerate(actions)}
    decision = await _run(concurrent(children), before_step(), ListRecorder())
    assert decision is not None and decision.action == expected


@pytest.mark.anyio
async def test_concurrent_honours_a_lone_modify() -> None:
    decision = await _run(concurrent([rewrites()]), before_step(), ListRecorder())
    assert decision is not None and decision.action == "modify"
    assert decision.modified is not None


@pytest.mark.anyio
async def test_concurrent_turns_a_contested_modify_into_a_reject() -> None:
    decision = await _run(
        concurrent({"rewriter": rewrites("safer"), "approver": says("continue")}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation == (
        "rewriter modified the call but 1 other vote(s) also decided: safer"
        " (rewriter: modify; approver: continue)"
    )


@pytest.mark.parametrize(
    ("children", "expected"),
    [
        (
            {"a": says("continue"), "b": says("reject", "too risky")},
            "too risky (a: continue; b: reject)",
        ),
        ({"a": says("continue"), "b": says("reject")}, "a: continue; b: reject"),
        (
            {"a": says("reject", "sure"), "b": says("escalate")},
            "sure (a: reject; b: escalate)",
        ),
        (
            {"a": says("continue"), "b": says("escalate", "ask a person")},
            "ask a person (a: continue; b: escalate)",
        ),
    ],
)
@pytest.mark.anyio
async def test_concurrent_explains_a_contested_decision_by_its_voters(
    children: Protocols, expected: str
) -> None:
    decision = await _run(concurrent(children), before_step(), ListRecorder())
    assert decision is not None and decision.explanation == expected


@pytest.mark.parametrize(
    ("children", "expected"),
    [
        ({"a": says("continue"), "b": says("continue")}, None),
        ({"a": says("continue", "looks fine"), "b": says("continue")}, "looks fine"),
    ],
)
@pytest.mark.anyio
async def test_concurrent_lists_no_votes_when_all_continue(
    children: Protocols, expected: str | None
) -> None:
    decision = await _run(concurrent(children), before_step(), ListRecorder())
    assert decision is not None and decision.action == "continue"
    assert decision.explanation == expected


@protocol
def panel(children: Protocols) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        votes = await run_protocols(children, context, step)
        rejects = sum(v.report.action == "reject" for v in votes)
        decide_final(
            Decision.reject() if rejects * 2 > len(votes) else Decision.proceed()
        )

    return decide


@pytest.mark.parametrize(
    ("votes", "expected"),
    [
        (["reject", "reject", "continue"], "reject"),
        (["reject", "continue", "continue"], "continue"),
    ],
)
@pytest.mark.anyio
async def test_a_panel_of_advisory_votes_has_the_last_word(
    votes: list[Action], expected: Action
) -> None:
    recorder = ListRecorder()
    children = {f"v{index}": says(vote) for index, vote in enumerate(votes)}
    decision = await run_sentinel(
        wrapper(panel(children)), host_context(recorder=recorder), before_step()
    )
    assert decision is not None and decision.action == expected
    assert sorted(r.reported.path for r in recorder.records) == [
        "panel",
        "panel/v0",
        "panel/v1",
        "panel/v2",
    ]
    assert recorder.bypassed_layers == [("", "wrapper")]


@pytest.mark.anyio
async def test_concurrent_honours_a_modify_the_others_only_escalated_against() -> None:
    decision = await _run(
        concurrent({"rewriter": rewrites(), "abstainer": says("escalate")}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "modify"


@pytest.mark.anyio
async def test_concurrent_turns_two_modifies_into_a_reject() -> None:
    decision = await _run(
        concurrent({"a": rewrites(), "b": rewrites()}), before_step(), ListRecorder()
    )
    assert decision is not None and decision.action == "reject"


@pytest.mark.anyio
async def test_concurrent_records_monitors_and_protocols_alike() -> None:
    recorder = ListRecorder()
    await _run(
        concurrent({"m": graded(0.4), "p": says("continue")}), before_step(), recorder
    )
    assert sorted(r.reported.path for r in recorder.records) == [
        "concurrent",
        "concurrent/m",
        "concurrent/p",
    ]


@monitor
def suspicion_monitor(model: str | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.9, "posts credentials to an external host")

    return check


@pytest.mark.parametrize(
    ("configure", "expected"),
    [
        (lambda: observe_only(suspicion_monitor()), None),
        (lambda: threshold(suspicion_monitor(), reject_at=0.8), "reject"),
        (lambda: concurrent(suspicion_monitor()), None),
        (lambda: concurrent(says("terminate")), "terminate"),
    ],
)
@pytest.mark.anyio
async def test_a_shipped_protocol_takes_a_single_instance(
    configure: Callable[[], Protocol], expected: Action | None
) -> None:
    recorder = ListRecorder()
    decision = await _run(configure(), before_step(), recorder)
    assert (decision.action if decision else None) == expected
    assert recorder.records[0].reported.name in ("suspicion_monitor", "says")


@pytest.mark.parametrize(
    ("configure", "error", "match"),
    [
        (lambda: observe_only(cast(Any, says())), TypeError, "monitor"),
        (lambda: concurrent(cast(Any, graded)), TypeError, "call it"),
    ],
)
def test_a_single_instance_is_validated_when_it_is_configured(
    configure: Callable[[], Protocol], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        configure()


def test_the_shipped_protocols_register_under_the_package() -> None:
    assert [registry_info(f).name for f in (observe_only, concurrent, threshold)] == [
        "inspect_sentinel/observe_only",
        "inspect_sentinel/concurrent",
        "inspect_sentinel/threshold",
    ]
    assert {registry_info(f).type for f in (observe_only, concurrent, threshold)} == {
        "protocol"
    }


def test_step_types_come_from_the_annotations() -> None:
    both = frozenset({BeforeToolCall, AfterToolCall})
    assert step_types(concurrent([graded()])) == both
    assert step_types(observe_only([graded()])) == both


def test_observe_rejects_a_protocol_when_it_is_configured() -> None:
    with pytest.raises(TypeError, match="monitor"):
        observe_only(cast(Any, [says()]))


def test_concurrent_rejects_duplicate_names_when_it_is_configured() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        concurrent([graded(), graded()])


def test_a_shipped_protocol_rejects_an_uncalled_factory() -> None:
    with pytest.raises(TypeError, match="call it"):
        observe_only(cast(Any, [graded]))


@pytest.mark.parametrize(
    "configure",
    [
        lambda: observe_only([]),
        lambda: concurrent({}),
        lambda: threshold([], reject_at=0.5),
    ],
)
def test_a_shipped_protocol_needs_at_least_one_child(
    configure: Callable[[], Protocol],
) -> None:
    with pytest.raises(ValueError, match="needs at least one child"):
        configure()


@protocol
def rules() -> ProtocolGroup:
    async def lenient(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.proceed()

    async def strict(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("strict")

    return ProtocolGroup(lenient, strict)


@pytest.mark.anyio
async def test_concurrent_votes_a_groups_members_like_separate_protocols() -> None:
    recorder = ListRecorder()
    decision = await _run(concurrent([rules()]), before_step(), recorder)
    assert decision is not None and decision.action == "reject"
    assert [
        (r.reported.path, r.reported.function)
        for r in recorder.records
        if r.reported.name == "rules"
    ] == [("concurrent/rules", "lenient"), ("concurrent/rules", "strict")]


def _edits(first: Action, second: Action) -> ProtocolGroup:
    modified = ToolCall(id="call_1", function="bash", arguments={"cmd": "echo hi"})

    @protocol
    def edits() -> ProtocolGroup:
        async def rewrite(context: Context, step: BeforeToolCall) -> Decision | None:
            return Decision(
                action=first, modified=modified if first == "modify" else None
            )

        async def audit(context: Context, step: BeforeToolCall) -> Decision | None:
            return Decision(
                action=second, modified=modified if second == "modify" else None
            )

        return ProtocolGroup(rewrite, audit)

    return edits()


@pytest.mark.parametrize(
    ("children", "action", "explanation"),
    [
        (
            lambda: [_edits("modify", "continue")],
            "modify",
            "edits.rewrite: modify; edits.audit: continue",
        ),
        (
            lambda: [_edits("modify", "modify")],
            "reject",
            "edits.rewrite modified the call but 1 other vote(s) also decided"
            " (edits.rewrite: modify; edits.audit: modify)",
        ),
        (
            lambda: {"e": _edits("modify", "continue"), "ok": says("continue")},
            "reject",
            "e.rewrite modified the call but 1 other vote(s) also decided"
            " (e.rewrite: modify; e.audit: continue; ok: continue)",
        ),
    ],
)
@pytest.mark.anyio
async def test_concurrent_counts_a_groups_votes_by_instance(
    children: Callable[[], Protocols], action: Action, explanation: str
) -> None:
    decision = await _run(concurrent(children()), before_step(), ListRecorder())
    assert decision is not None
    assert (decision.action, decision.explanation) == (action, explanation)
