import math
from collections.abc import Callable
from typing import Any, cast

import pytest
from inspect_ai.core import Reference, ToolCall
from inspect_ai.core._registry import registry_info

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
    MonitorGroup,
    Protocol,
    ProtocolGroup,
    Protocols,
)
from tests._fakes import (
    ListRecorder,
    after_step,
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


@pytest.mark.parametrize(
    ("value", "expected"), [(0.2, "continue"), (0.5, "reject"), (0.7, "reject")]
)
@pytest.mark.anyio
async def test_threshold_rejects_at_or_above_its_constant(
    value: float, expected: Action
) -> None:
    decision = await _run(
        threshold([graded(value)], reject_at=0.5), before_step(), ListRecorder()
    )
    assert decision is not None and decision.action == expected


@pytest.mark.anyio
async def test_threshold_terminates_at_or_above_its_second_constant() -> None:
    decision = await _run(
        threshold([graded(0.95)], reject_at=0.5, terminate_at=0.9),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "terminate"


@pytest.mark.anyio
async def test_threshold_abstains_when_nobody_observed() -> None:
    decision = await _run(
        threshold([silent()], reject_at=0.5), before_step(), ListRecorder()
    )
    assert decision is None


@pytest.mark.anyio
async def test_threshold_does_not_run_after_a_tool_call() -> None:
    recorder = ListRecorder()
    decision = await _run(
        threshold([graded(0.9)], reject_at=0.5), after_step(), recorder
    )
    assert decision is None
    assert recorder.records == []


@pytest.mark.anyio
async def test_threshold_thresholds_the_highest_dimension() -> None:
    decision = await _run(
        threshold([graded({"x": 0.1, "y": 0.8})], reject_at=0.5),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation == "suspicion 0.80 from graded"


@monitor
def explained(value: float, reason: str) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value, reason)

    return check


_MONITORS = {
    "quiet": explained(0.2, "looks routine"),
    "loud": explained(0.9, "posts credentials to an external host"),
}


@pytest.mark.anyio
async def test_threshold_explains_a_reject_in_the_log_but_not_to_the_agent() -> None:
    decision = await _run(
        threshold(_MONITORS, reject_at=0.5), before_step(), ListRecorder()
    )
    assert decision is not None and decision.action == "reject"
    assert (
        decision.explanation
        == "suspicion 0.90 from loud: posts credentials to an external host"
    )
    assert decision.message is None


@pytest.mark.anyio
async def test_threshold_explains_a_termination_with_the_monitors_reasoning() -> None:
    decision = await _run(
        threshold(_MONITORS, reject_at=0.5, terminate_at=0.8),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "terminate"
    assert (
        decision.explanation
        == "suspicion 0.90 from loud: posts credentials to an external host"
    )
    assert decision.message is None


@pytest.mark.anyio
async def test_threshold_names_the_first_monitor_on_a_tie() -> None:
    decision = await _run(
        threshold(
            {"a": explained(0.9, "first"), "b": explained(0.9, "second")},
            reject_at=0.5,
            terminate_at=0.8,
        ),
        before_step(),
        ListRecorder(),
    )
    assert (
        decision is not None and decision.explanation == "suspicion 0.90 from a: first"
    )


@pytest.mark.anyio
async def test_threshold_explains_nothing_when_it_clears() -> None:
    decision = await _run(
        threshold([explained(0.1, "fine")], reject_at=0.5),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "continue"
    assert decision.explanation is None


@monitor
def cited(
    value: Suspicion, reason: str | None = None, cites: tuple[str, ...] = ()
) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(
            value,
            reason,
            references=[Reference(type="message", id=f"m{c}", cite=c) for c in cites],
        )

    return check


_STRUCTURED: dict[str, Monitor] = {
    "audit": cited({"sabotage": 0.72, "exfil": 0.3}, "edits the tests", ("[M2]",)),
    "egress": cited(0.4, "curls a host", ("[M3]",)),
}


@pytest.mark.parametrize(
    ("monitors", "reject_at", "terminate_at"),
    [
        (_STRUCTURED, 0.5, None),
        (_STRUCTURED, 0.5, 0.7),
        (_STRUCTURED, 0.8, None),
        ({"a": explained(0.9, "first"), "b": explained(0.9, "second")}, 0.5, 0.8),
        (_MONITORS, 0.5, None),
    ],
)
@pytest.mark.anyio
async def test_threshold_a_float_is_shorthand_for_a_star_key(
    monitors: dict[str, Monitor], reject_at: float, terminate_at: float | None
) -> None:
    shorthand = await _run(
        threshold(monitors, reject_at=reject_at, terminate_at=terminate_at),
        before_step(),
        ListRecorder(),
    )
    starred = await _run(
        threshold(
            monitors,
            reject_at={"*": reject_at},
            terminate_at=None if terminate_at is None else {"*": terminate_at},
        ),
        before_step(),
        ListRecorder(),
    )
    assert shorthand == starred


@pytest.mark.parametrize(
    ("reject_at", "action", "explanation"),
    [
        ({"audit.sabotage": 0.6}, "reject", "sabotage 0.72 ≥ 0.60 from audit"),
        ({"sabotage": 0.6}, "reject", "sabotage 0.72 ≥ 0.60 from audit"),
        ({"audit.sabotage": 0.8, "sabotage": 0.6}, "continue", None),
        ({"sabotage": 0.8, "*": 0.1}, "reject", "suspicion 0.40 from egress"),
        ({"audit.exfil": 0.2, "exfil": 0.9}, "reject", "exfil 0.30 ≥ 0.20 from audit"),
        ({"exfil": 0.9}, "continue", None),
        ({"egress": 0.4}, "reject", "suspicion 0.40 from egress"),
    ],
)
@pytest.mark.anyio
async def test_threshold_resolves_a_qualified_then_a_bare_then_a_star_key(
    reject_at: dict[str, float], action: Action, explanation: str | None
) -> None:
    decision = await _run(
        threshold(_STRUCTURED, reject_at=reject_at), before_step(), ListRecorder()
    )
    assert decision is not None and decision.action == action
    reason = {"audit": "edits the tests", "egress": "curls a host"}
    expected = (
        None
        if explanation is None
        else f"{explanation}: {reason[explanation.rsplit(' ', 1)[1]]}"
    )
    assert decision.explanation == expected
    assert decision.message is None


@pytest.mark.anyio
async def test_threshold_a_star_crossing_is_explained_as_a_float_is() -> None:
    decision = await _run(
        threshold(_STRUCTURED, reject_at={"egress": 0.9, "*": 0.5}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None
    assert decision.explanation == "suspicion 0.72 from audit: edits the tests"


@pytest.mark.anyio
async def test_threshold_names_the_highest_crossing_dimension_across_monitors() -> None:
    decision = await _run(
        threshold(
            {
                "a": cited({"sabotage": 0.7}, "first"),
                "b": cited({"sabotage": 0.9}, "second"),
                "c": cited({"sabotage": 0.9}, "third"),
            },
            reject_at={"sabotage": 0.6},
        ),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation == "sabotage 0.90 ≥ 0.60 from b: second"


@pytest.mark.anyio
async def test_threshold_carries_the_named_monitors_references() -> None:
    decision = await _run(
        threshold(_STRUCTURED, reject_at={"sabotage": 0.6}),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None
    assert decision.references == [Reference(type="message", id="m[M2]", cite="[M2]")]


@pytest.mark.parametrize(
    ("reject_at", "terminate_at", "action"),
    [
        ({"sabotage": 0.6}, {"sabotage": 0.7}, "terminate"),
        ({"sabotage": 0.6}, {"audit.sabotage": 0.8}, "reject"),
        ({"egress": 0.3}, {"sabotage": 0.7}, "terminate"),
        (0.3, {"sabotage": 0.9}, "reject"),
        ({"sabotage": 0.6}, 0.7, "terminate"),
    ],
)
@pytest.mark.anyio
async def test_threshold_terminates_before_it_rejects_per_dimension(
    reject_at: float | dict[str, float],
    terminate_at: float | dict[str, float],
    action: Action,
) -> None:
    decision = await _run(
        threshold(_STRUCTURED, reject_at=reject_at, terminate_at=terminate_at),
        before_step(),
        ListRecorder(),
    )
    assert decision is not None and decision.action == action


@pytest.mark.parametrize(
    ("reject_at", "terminate_at", "message"),
    [
        ({"sabotage": math.inf}, None, "finite"),
        ({"sabotage": 0.5}, {"*": math.nan}, "finite"),
        ({"sabotage": cast(Any, "high")}, None, "finite"),
        ({"sabotage": cast(Any, True)}, None, "finite"),
        ({}, None, "at least one"),
        ({cast(Any, 3): 0.5}, None, "key"),
        ({"": 0.5}, None, "key"),
        ({"sab*": 0.5}, None, "key"),
        ({"audit.": 0.5}, None, "key"),
        ({"nobody.sabotage": 0.5}, None, "names no monitor"),
        (
            {"sabotage": 0.8},
            {"sabotage": 0.5},
            "terminate_at must be above reject_at",
        ),
        (
            {"audit.sabotage": 0.8},
            {"sabotage": 0.8},
            "terminate_at must be above reject_at",
        ),
        ({"sabotage": 0.8}, 0.5, "terminate_at must be above reject_at"),
        (0.8, {"egress": 0.7}, "terminate_at must be above reject_at"),
    ],
)
def test_threshold_validates_its_mappings_when_it_is_configured(
    reject_at: float | dict[str, float],
    terminate_at: float | dict[str, float] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        threshold(_STRUCTURED, reject_at=reject_at, terminate_at=terminate_at)


def test_threshold_allows_terminate_below_an_unrelated_reject() -> None:
    threshold(_STRUCTURED, reject_at={"sabotage": 0.8}, terminate_at={"exfil": 0.5})


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
        (lambda: threshold(cast(Any, says()), reject_at=0.5), TypeError, "monitor"),
        (lambda: threshold(afterwards(), reject_at=0.5), TypeError, "afterwards"),
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
    assert step_types(threshold([graded()], reject_at=0.5)) == frozenset(
        {BeforeToolCall}
    )
    assert step_types(concurrent([graded()])) == both
    assert step_types(observe_only([graded()])) == both


def test_observe_rejects_a_protocol_when_it_is_configured() -> None:
    with pytest.raises(TypeError, match="monitor"):
        observe_only(cast(Any, [says()]))


def test_threshold_rejects_a_protocol_when_it_is_configured() -> None:
    with pytest.raises(TypeError, match="monitor"):
        threshold(cast(Any, [says()]), reject_at=0.5)


def test_concurrent_rejects_duplicate_names_when_it_is_configured() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        concurrent([graded(), graded()])


def test_a_shipped_protocol_rejects_an_uncalled_factory() -> None:
    with pytest.raises(TypeError, match="call it"):
        observe_only(cast(Any, [graded]))


@pytest.mark.parametrize(
    ("reject_at", "terminate_at", "message"),
    [
        (math.inf, None, "finite"),
        (math.nan, None, "finite"),
        (0.5, math.inf, "finite"),
        (0.5, 0.4, "terminate_at must be above reject_at"),
        (0.5, 0.5, "terminate_at must be above reject_at"),
    ],
)
def test_thresholds_constants_must_be_finite_and_ordered(
    reject_at: float, terminate_at: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        threshold([graded()], reject_at=reject_at, terminate_at=terminate_at)


def test_threshold_rejects_a_monitor_that_never_watches_a_tool_call() -> None:
    with pytest.raises(TypeError, match="afterwards"):
        threshold([afterwards()], reject_at=0.5)


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


@monitor
def two_before(value: float = 0.7) -> MonitorGroup:
    async def low(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    async def high(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(value)

    return MonitorGroup(low, high)


@monitor
def before_and_after() -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    async def after(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(1.0)

    return MonitorGroup(before, after)


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


@pytest.mark.anyio
async def test_threshold_reads_every_member_of_a_group() -> None:
    recorder = ListRecorder()
    decision = await _run(
        threshold([two_before(0.7)], reject_at=0.5), before_step(), recorder
    )
    assert decision is not None and decision.action == "reject"


def test_threshold_rejects_a_group_that_never_watches_a_tool_call() -> None:
    @monitor
    def later() -> MonitorGroup:
        async def one(context: Context, step: AfterToolCall) -> Observation | None:
            return None

        async def two(context: Context, step: AfterToolCall) -> Observation | None:
            return None

        return MonitorGroup(one, two)

    with pytest.raises(TypeError, match="later"):
        threshold([later()], reject_at=0.5)


def test_threshold_rejects_a_group_with_a_member_that_never_watches_a_tool_call() -> (
    None
):
    with pytest.raises(TypeError, match="'before_and_after' \\(function 'after'\\)"):
        threshold([before_and_after()], reject_at=0.5)


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
