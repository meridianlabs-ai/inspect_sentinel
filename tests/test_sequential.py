from typing import Any, NamedTuple, cast

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.log import SentinelConfig
from inspect_ai.util import Store, StoreModel

from inspect_sentinel import sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._final import decide_final
from inspect_sentinel._integration import (
    config_from_sentinel,
    resolve_sentinel,
    sentinel_from_config,
)
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Action, Decision, Observation, Reported
from inspect_sentinel._runner import run_protocols, run_sentinel
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import Monitor, Protocol, ProtocolGroup, Sentinels
from tests._fakes import ListRecorder, after_step, before_step, host_context


class Seen(StoreModel):
    escalations: list[tuple[str, ...]] = []


class PairSaw(StoreModel):
    first: int | None = None
    second: int | None = None


def _seen(store: Store) -> dict[str, list[tuple[str, ...]]]:
    paths = [key.split(":")[1] for key in store.keys() if key.startswith("Seen:")]
    return {path: Seen(store=store, instance=path).escalations for path in paths}


@protocol
def seq_link(action: Action = "proceed", explanation: str | None = None) -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        seen = context.store_as(Seen)
        seen.escalations = [
            *seen.escalations,
            tuple(e.name for e in escalations),
        ]
        return Decision(action=action, explanation=explanation)

    return decide


@protocol
def seq_abstains() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        return None

    return decide


@protocol
def seq_after_only() -> Protocol:
    async def decide(
        context: Context,
        step: AfterToolCall,
        escalations: tuple[Reported[Decision], ...],
    ) -> Decision | None:
        return Decision.terminate("after")

    return decide


@protocol
def seq_pair() -> ProtocolGroup:
    async def first(
        context: Context,
        step: BeforeToolCall,
        escalations: tuple[Reported[Decision], ...],
    ) -> Decision | None:
        context.store_as(PairSaw).first = len(escalations)
        return Decision.escalate("first")

    async def second(
        context: Context,
        step: BeforeToolCall,
        escalations: tuple[Reported[Decision], ...],
    ) -> Decision | None:
        context.store_as(PairSaw).second = len(escalations)
        return Decision.escalate("second")

    return ProtocolGroup(first, second)


@protocol
def seq_mixed(first_action: Action, second_action: Action) -> ProtocolGroup:
    async def first(
        context: Context,
        step: BeforeToolCall,
        escalations: tuple[Reported[Decision], ...],
    ) -> Decision | None:
        return Decision(action=first_action, explanation="first")

    async def second(
        context: Context,
        step: BeforeToolCall,
        escalations: tuple[Reported[Decision], ...],
    ) -> Decision | None:
        return Decision(action=second_action, explanation="second")

    return ProtocolGroup(first, second)


@protocol
def seq_final() -> Protocol:
    async def decide(
        context: Context,
        step: BeforeToolCall,
        escalations: tuple[Reported[Decision], ...],
    ) -> Decision | None:
        decide_final(Decision.reject("final"))

    return decide


@protocol
def seq_forward(child: Protocol) -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: tuple[Reported[Decision], ...]
    ) -> Decision | None:
        decisions = await run_protocols(child, context, step, escalations=escalations)
        strongest = decisions.strongest()
        return None if strongest is None else strongest.report

    return decide


@monitor
def seq_watch() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.4)

    return check


class _Rooted(NamedTuple):
    decision: Decision | None
    recorder: ListRecorder
    store: Store


async def _root(
    instance: Protocol, step: Step | None = None, recorder: ListRecorder | None = None
) -> _Rooted:
    recorder = recorder or ListRecorder()
    store = Store()
    context = host_context(recorder=recorder, store=store)
    decision = await run_sentinel(instance, context, step or before_step())
    return _Rooted(decision, recorder, store)


def _decided(rooted: _Rooted) -> Decision | None:
    # the chain's own decision, as recorded at the root; run_sentinel returns a
    # terminate in place of an escalate there
    recorded = [
        r.reported.report for r in rooted.recorder.records if not r.reported.path
    ]
    assert len(recorded) <= 1
    return cast(Decision, recorded[0]) if recorded else None


def _paths(recorder: ListRecorder) -> list[str]:
    return [r.reported.path for r in recorder.records]


@pytest.mark.anyio
async def test_the_first_real_decision_ends_the_chain() -> None:
    decision, recorder, _ = await _root(
        sequential(
            {
                "a": seq_link("escalate"),
                "b": seq_link("reject", "no"),
                "c": seq_link("terminate"),
            }
        )
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation == "no"
    assert _paths(recorder) == ["a", "b", ""]


@pytest.mark.anyio
async def test_escalations_are_handed_forward_link_by_link() -> None:
    _, _, store = await _root(
        sequential(
            {
                "a": seq_link("escalate"),
                "b": seq_link("escalate"),
                "c": seq_link("proceed"),
            }
        )
    )
    assert _seen(store) == {
        "a": [()],
        "b": [("a",)],
        "c": [("a", "b")],
    }


@pytest.mark.anyio
async def test_a_chain_does_not_inherit_the_incoming_escalations() -> None:
    _, _, store = await _root(
        sequential(
            {
                "first": seq_link("escalate"),
                "inner": sequential({"x": seq_link("escalate"), "y": seq_link()}),
            }
        )
    )
    assert _seen(store) == {
        "first": [()],
        "inner/x": [()],
        "inner/y": [("x",)],
    }


@pytest.mark.anyio
async def test_a_concurrent_link_hands_its_children_no_escalations() -> None:
    _, _, store = await _root(
        sequential(
            {
                "first": seq_link("escalate"),
                "both": concurrent({"x": seq_link()}),
            }
        )
    )
    assert _seen(store) == {"first": [()], "both/x": [()]}


@pytest.mark.anyio
async def test_a_protocol_can_hand_its_escalations_on() -> None:
    _, _, store = await _root(
        sequential(
            {
                "first": seq_link("escalate"),
                "outer": seq_forward(seq_link()),
            }
        )
    )
    assert _seen(store) == {"first": [()], "outer/seq_link": [("first",)]}


@pytest.mark.anyio
async def test_a_monitor_after_an_escalation_is_handed_the_hosts_step() -> None:
    seen: list[Step] = []

    @monitor
    def seq_step_watch() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            seen.append(step)
            return None

        return check

    step = before_step()
    await _root(
        sequential(
            {
                "first": seq_link("escalate"),
                "watch": seq_step_watch(),
                "last": seq_link(),
            }
        ),
        step,
    )
    [watched] = seen
    assert watched is step


@pytest.mark.anyio
async def test_monitors_are_recorded_and_fall_through() -> None:
    decision, recorder, _ = await _root(
        sequential({"watch": seq_watch(), "rule": seq_link("reject")})
    )
    assert decision is not None and decision.action == "reject"
    assert _paths(recorder) == ["watch", "rule", ""]


@pytest.mark.anyio
async def test_a_link_that_does_not_watch_the_stage_is_skipped() -> None:
    decision, recorder, _ = await _root(
        sequential({"later": seq_after_only(), "rule": seq_link("reject")})
    )
    assert decision is not None and decision.action == "reject"
    assert _paths(recorder) == ["rule", ""]


@pytest.mark.anyio
async def test_a_link_at_its_own_stage_decides() -> None:
    decision, _, _ = await _root(
        sequential({"later": seq_after_only(), "rule": seq_link()}), after_step()
    )
    assert decision is not None and decision.action == "terminate"


@pytest.mark.parametrize(
    ("children", "expected"),
    [
        (
            {"a": seq_link("escalate", "a?"), "b": seq_link("escalate", "b?")},
            ("escalate", "b?"),
        ),
        ({"watch": seq_watch()}, ("proceed", None)),
        ({"quiet": seq_abstains(), "later": seq_after_only()}, None),
    ],
)
@pytest.mark.anyio
async def test_what_a_chain_returns_when_no_link_decides(
    children: Sentinels, expected: tuple[Action, str | None] | None
) -> None:
    decision = _decided(await _root(sequential(children)))
    if expected is None:
        assert decision is None
    else:
        assert decision is not None
        assert (decision.action, decision.explanation) == expected


@pytest.mark.anyio
async def test_a_group_link_runs_in_one_call_and_is_handed_the_escalations_before_its_own() -> (
    None
):
    decision, recorder, store = await _root(
        sequential({"pair": seq_pair(), "next": seq_link()})
    )
    assert decision is not None and decision.action == "proceed"
    saw = PairSaw(store=store, instance="pair")
    assert (saw.first, saw.second) == (0, 0)
    assert _seen(store) == {"next": [("pair", "pair")]}
    assert [r.reported.function for r in recorder.records[:2]] == ["first", "second"]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("proceed", "terminate", ("terminate", "second")),
        ("proceed", "reject", ("reject", "second")),
        ("reject", "proceed", ("reject", "first")),
        ("escalate", "proceed", ("escalate", "first")),
    ],
)
@pytest.mark.anyio
async def test_a_group_link_decides_by_its_strongest_decision(
    first: Action, second: Action, expected: tuple[Action, str]
) -> None:
    decision = _decided(await _root(sequential({"pair": seq_mixed(first, second)})))
    assert decision is not None
    assert (decision.action, decision.explanation) == expected


@pytest.mark.anyio
async def test_a_group_link_that_escalates_hands_forward() -> None:
    decision, _, store = await _root(
        sequential({"pair": seq_mixed("escalate", "proceed"), "next": seq_link()})
    )
    assert decision is not None and decision.action == "proceed"
    # only the escalating function's decision is handed forward
    assert _seen(store) == {"next": [("pair",)]}


@pytest.mark.anyio
async def test_decide_final_from_a_link_bypasses_the_layers_above() -> None:
    decision, recorder, _ = await _root(
        concurrent(
            {
                "chain": sequential({"rule": seq_final(), "after": seq_link()}),
                "peer": seq_link(),
            }
        )
    )
    assert decision is not None and decision.action == "reject"
    assert decision.explanation == "final"
    assert recorder.bypassed_layers == [("chain", "chain"), ("", "concurrent")]
    assert _paths(recorder) == ["peer", "chain/rule"]


@pytest.mark.parametrize("children", [[], {}])
def test_an_empty_chain_is_a_configuration_error(children: Sentinels) -> None:
    with pytest.raises(ValueError, match="sequential needs at least one child"):
        sequential(children)


def test_sequential_is_registered_under_the_package_name() -> None:
    assert registry_info(sequential([seq_link()])).name == "inspect_sentinel/sequential"


RAW: Any = [
    {
        "name": "sequential",
        "children": {
            "network_review": {"name": "seq_link", "params": {"action": "escalate"}},
            "watch": {"name": "seq_watch"},
            "last": {"name": "seq_link", "params": {"action": "reject"}},
        },
    },
    {"name": "seq_watch"},
]


def test_a_chain_round_trips_through_config() -> None:
    config = SentinelConfig.model_validate(RAW)
    assert config_from_sentinel(sentinel_from_config(RAW)) == config


@pytest.mark.anyio
async def test_a_configured_chain_records_its_links_under_its_path() -> None:
    recorder = ListRecorder()
    decision = await run_sentinel(
        resolve_sentinel(sentinel_from_config(RAW)),
        host_context(recorder=recorder),
        before_step(),
    )
    assert decision is not None and decision.action == "reject"
    assert sorted(_paths(recorder)) == [
        "",
        "seq_watch",
        "sequential",
        "sequential/last",
        "sequential/network_review",
        "sequential/watch",
    ]
