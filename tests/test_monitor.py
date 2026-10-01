import inspect
from collections.abc import Callable
from typing import Any, cast

import pytest
from inspect_ai._util.registry import (
    RegistryType,
    create_registry_object,
    registry_create,
    registry_info,
    registry_lookup,
    registry_params,
)

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import (
    monitor,
    protocol,
    step_types,
)
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._runner import (
    run_monitors,
    run_root,
)
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import (
    Monitor,
    MonitorGroup,
    Protocol,
    ProtocolGroup,
)
from tests._fakes import before_step, runner_context


@monitor
def before_monitor(threshold_hint: float = 0.5) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(threshold_hint)

    return check


@monitor
def after_monitor() -> Monitor:
    async def check(context: Context, step: AfterToolCall) -> Observation | None:
        return None

    return check


@protocol
def any_stage_protocol() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return None

    return decide


@protocol
def no_curl() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if "curl" in str(step.call.arguments):
            return Decision.reject("network access is not allowed")
        return None

    return decide


@pytest.mark.parametrize(
    "factory", [before_monitor, after_monitor, any_stage_protocol, no_curl]
)
def test_stage_annotated_functions_are_accepted(
    factory: Callable[[], Monitor | Protocol],
) -> None:
    assert inspect.iscoroutinefunction(factory())


def test_monitor_may_annotate_a_bare_observation_return() -> None:
    @monitor
    def always_flags() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation:
            return Observation.score(0.1)

        return check

    assert inspect.iscoroutinefunction(always_flags())


def test_monitor_registers_under_the_monitor_type() -> None:
    assert registry_lookup("monitor", "before_monitor") is before_monitor
    assert registry_info(before_monitor).type == "monitor"
    assert registry_info(before_monitor).metadata["params"] == ["threshold_hint"]


def test_protocol_registers_under_the_protocol_type() -> None:
    assert registry_lookup("protocol", "no_curl") is no_curl
    assert registry_info(no_curl).type == "protocol"


def test_registry_object_is_created_by_name_with_params() -> None:
    instance = create_registry_object(
        "monitor", "before_monitor", {"threshold_hint": 0.9}
    )
    assert registry_info(instance).type == "monitor"
    assert registry_info(instance).name == "before_monitor"
    assert registry_params(instance) == {"threshold_hint": 0.9}


def test_registry_create_is_not_a_construction_path_for_sentinels() -> None:
    """registry_create instantiates only factories whose return annotation's class name equals the registry type; Monitor and Protocol are union aliases with no name, so it hands back the factory. Construction from config goes through create_registry_object, which always instantiates."""
    kind: RegistryType = "monitor"
    create = cast(Callable[..., Any], registry_create)
    assert create(kind, "before_monitor") is before_monitor


def test_instance_is_tagged_with_registry_info() -> None:
    instance = no_curl()
    assert registry_info(instance).name == "no_curl"
    assert registry_params(instance) == {}


def test_instances_have_independent_registry_metadata() -> None:
    a = before_monitor(threshold_hint=0.1)
    b = before_monitor(threshold_hint=0.2)
    registry_info(a).metadata["extra"] = True
    assert "extra" not in registry_info(b).metadata
    assert "extra" not in registry_info(before_monitor).metadata


def test_instances_have_independent_registry_info() -> None:
    a = before_monitor(threshold_hint=0.1)
    b = before_monitor(threshold_hint=0.2)
    assert registry_params(a) == {"threshold_hint": 0.1}
    assert registry_params(b) == {"threshold_hint": 0.2}
    assert registry_info(a) is not registry_info(b)


def test_factory_keeps_its_name_and_signature() -> None:
    assert before_monitor.__name__ == "before_monitor"
    assert list(inspect.signature(before_monitor).parameters) == ["threshold_hint"]


def test_monitor_returning_decisions_is_rejected_at_configuration() -> None:
    async def decides_not_observes(
        context: Context, step: BeforeToolCall
    ) -> Decision | None:
        return None

    def factory_returning() -> Any:
        return decides_not_observes

    wrong = monitor(cast(Callable[[], Monitor], factory_returning))
    with pytest.raises(TypeError, match="Observation"):
        wrong()


def test_protocol_returning_observations_is_rejected_at_configuration() -> None:
    async def observes_not_decides(
        context: Context, step: BeforeToolCall
    ) -> Observation | None:
        return None

    def protocol_returning() -> Any:
        return observes_not_decides

    wrong = protocol(cast(Callable[[], Protocol], protocol_returning))
    with pytest.raises(TypeError, match="Decision"):
        wrong()


def test_unannotated_step_parameter_is_rejected_at_configuration() -> None:
    async def unannotated_step(context: Context, step: Any) -> Observation | None:
        return None

    unannotated_step.__annotations__.pop("step")

    def factory_returning() -> Any:
        return unannotated_step

    wrong = monitor(cast(Callable[[], Monitor], factory_returning))
    with pytest.raises(TypeError, match="annotat"):
        wrong()


def test_unresolvable_annotation_names_the_fix() -> None:
    async def unresolvable_step(context: Context, step: Any) -> Observation | None:
        return None

    unresolvable_step.__annotations__["step"] = "Missing"

    def factory_returning() -> Any:
        return unresolvable_step

    wrong = monitor(cast(Callable[[], Monitor], factory_returning))
    with pytest.raises(TypeError, match="Missing"):
        wrong()


def test_non_async_function_is_rejected_at_configuration() -> None:
    def not_async(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    def factory_returning() -> Any:
        return not_async

    wrong = monitor(cast(Callable[[], Monitor], factory_returning))
    with pytest.raises(TypeError, match="async"):
        wrong()


def test_positional_only_step_is_accepted() -> None:
    @monitor
    def positional_only() -> Monitor:
        async def check(
            context: Context, step: BeforeToolCall, /
        ) -> Observation | None:
            return None

        return check

    assert inspect.iscoroutinefunction(positional_only())


def test_keyword_only_step_is_rejected_at_configuration() -> None:
    async def keyword_only(
        context: Context, *, step: BeforeToolCall
    ) -> Observation | None:
        return None

    def factory() -> Any:
        return keyword_only

    wrong = monitor(cast(Callable[[], Monitor], factory))
    with pytest.raises(TypeError, match="positional"):
        wrong()


def test_monitor_annotating_step_is_rejected_at_configuration() -> None:
    async def every_stage(context: Context, step: Step) -> Observation | None:
        return None

    def factory() -> Any:
        return every_stage

    wrong = monitor(cast(Callable[[], Monitor], factory))
    with pytest.raises(TypeError, match="one stage"):
        wrong()


def test_step_types_are_recorded_on_the_instance() -> None:
    assert step_types(before_monitor()) == frozenset({BeforeToolCall})
    assert step_types(after_monitor()) == frozenset({AfterToolCall})
    assert step_types(any_stage_protocol()) == frozenset(
        {BeforeToolCall, AfterToolCall}
    )


def test_step_types_requires_a_decorated_instance() -> None:
    async def undecorated(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    with pytest.raises(TypeError, match="@monitor or @protocol"):
        step_types(undecorated)


def test_step_types_names_an_uncalled_factory() -> None:
    with pytest.raises(TypeError, match="call it"):
        step_types(cast(Any, before_monitor))


def test_bound_method_is_rejected_at_configuration() -> None:
    class Detector:
        async def observe(
            self, context: Context, step: BeforeToolCall
        ) -> Observation | None:
            return None

    def factory() -> Any:
        return Detector().observe

    wrong = monitor(cast(Callable[[], Monitor], factory))
    with pytest.raises(TypeError, match="plain async function"):
        wrong()


@monitor
def paired() -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    async def after(context: Context, step: AfterToolCall) -> Observation | None:
        return None

    return MonitorGroup(before, after)


def test_a_group_is_one_registered_instance() -> None:
    group = paired()
    assert registry_info(group).name == "paired"
    assert registry_info(group).type == "monitor"
    assert step_types(group) == frozenset({BeforeToolCall, AfterToolCall})


async def _before(context: Context, step: BeforeToolCall) -> Observation | None:
    return None


async def _after(context: Context, step: AfterToolCall) -> Observation | None:
    return None


async def _unannotated(context: Context, step: Any) -> Observation | None:
    return None


_unannotated.__annotations__.pop("step")


async def _decide(context: Context, step: Step) -> Decision | None:
    return None


@pytest.mark.parametrize(
    ("functions", "message"),
    [([], "at least one function"), ([_before, _before], "'_before'")],
)
def test_an_invalid_group_is_rejected_when_built(
    functions: list[Monitor], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MonitorGroup(*functions)


@pytest.mark.parametrize(
    ("returned", "error", "message"),
    [
        (lambda: MonitorGroup(_before, _unannotated), TypeError, "annotat"),
        (lambda: [_before, _after], TypeError, r"return MonitorGroup\(\.\.\.\)"),
        (lambda: (_before, _after), TypeError, r"return MonitorGroup\(\.\.\.\)"),
        (lambda: ProtocolGroup(_decide), TypeError, "not a ProtocolGroup"),
    ],
)
def test_a_factory_returning_the_wrong_thing_is_rejected_at_configuration(
    returned: Callable[[], object], error: type[Exception], message: str
) -> None:
    wrong = monitor(cast(Callable[[], Monitor], returned))
    with pytest.raises(error, match=message):
        wrong()


def test_a_monitor_group_from_a_protocol_factory_is_rejected() -> None:
    def returns_monitor_group() -> MonitorGroup:
        return MonitorGroup(_before)

    wrong = protocol(cast(Callable[[], Protocol], returns_monitor_group))
    with pytest.raises(TypeError, match="not a MonitorGroup"):
        wrong()


@pytest.mark.anyio
async def test_a_group_cannot_be_called() -> None:
    with pytest.raises(TypeError, match="not callable"):
        await cast(Any, paired())(runner_context(), before_step())


@monitor(name="renamed_monitor", version=3)
def configured_monitor() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.1)

    return check


@protocol()
def bare_call_protocol() -> ProtocolGroup:
    async def first(context: Context, step: BeforeToolCall) -> Decision | None:
        return None

    async def second(context: Context, step: AfterToolCall) -> Decision | None:
        return None

    return ProtocolGroup(first, second)


def test_decorator_arguments_set_the_name_and_version() -> None:
    instance = configured_monitor()
    assert registry_info(instance).name == "renamed_monitor"
    assert registry_info(instance).metadata["version"] == 3
    assert registry_lookup("monitor", "renamed_monitor") is configured_monitor
    assert registry_lookup("monitor", "configured_monitor") is None
    assert registry_info(before_monitor()).metadata["version"] == 0
    group = bare_call_protocol()
    assert registry_info(group).name == "bare_call_protocol"
    assert step_types(group) == frozenset({BeforeToolCall, AfterToolCall})


@pytest.mark.anyio
async def test_a_configured_function_keeps_its_identity() -> None:
    check = cast(Callable[..., Any], configured_monitor())
    assert check.__name__ == "check"
    assert list(inspect.signature(check).parameters) == ["context", "step"]
    assert inspect.iscoroutinefunction(check)
    assert await check(runner_context(), before_step()) == Observation.score(0.1)


@monitor
def watched() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.4)

    return check


@protocol
def calls_directly(child: Monitor) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await cast(Callable[..., Any], child)(context, step)
        return None

    return decide


@protocol
def calls_through_runner(child: Monitor) -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        observations = await run_monitors(child, context, step)
        return Decision.reject() if observations.max_suspicion() else None

    return decide


@pytest.mark.anyio
async def test_a_direct_call_during_a_run_is_rejected() -> None:
    root = concurrent(calls_directly(watched()))
    with pytest.raises(RuntimeError, match="run_monitors/run_protocols/run_children"):
        await run_root(root, runner_context(), before_step())


@pytest.mark.anyio
async def test_a_call_through_the_runner_during_a_run_is_allowed() -> None:
    root = concurrent(calls_through_runner(watched()))
    decision = await run_root(root, runner_context(), before_step())
    assert decision is not None and decision.action == "reject"


@pytest.mark.anyio
async def test_a_direct_call_outside_a_run_is_allowed() -> None:
    decide = cast(Callable[..., Any], calls_directly(watched()))
    assert await decide(runner_context(), before_step()) is None


def test_a_monitor_and_a_protocol_cannot_share_a_name() -> None:
    def factory() -> Protocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            return None

        return decide

    factory.__name__ = "before_monitor"
    with pytest.raises(
        ValueError, match="'before_monitor' is already registered as a monitor"
    ):
        protocol(factory)
    assert registry_lookup("protocol", "before_monitor") is None


def test_a_factory_may_be_registered_again_as_the_same_kind() -> None:
    def factory() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            return None

        return check

    factory.__name__ = "registered_twice"
    monitor(factory)
    again = monitor(factory)
    assert registry_lookup("monitor", "registered_twice") is again
