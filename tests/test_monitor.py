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
from inspect_sentinel._monitor import (
    ControlProtocol,
    Monitor,
    monitor,
    protocol,
    stages,
)
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step


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


@monitor
def any_stage_monitor() -> Monitor:
    async def check(context: Context, step: Step) -> Observation | None:
        return None

    return check


@protocol
def no_curl() -> ControlProtocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if "curl" in str(step.call.arguments):
            return Decision.reject("network access is not allowed")
        return None

    return decide


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (before_monitor, frozenset({"tool_call"})),
        (after_monitor, frozenset({"tool_result"})),
        (any_stage_monitor, frozenset({"tool_call", "tool_result"})),
        (no_curl, frozenset({"tool_call"})),
    ],
)
def test_stage_is_inferred_from_the_second_parameter(
    factory: Callable[[], Monitor | ControlProtocol], expected: frozenset[str]
) -> None:
    assert stages(factory()) == expected


def test_monitor_may_annotate_a_bare_observation_return() -> None:
    @monitor
    def always_flags() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation:
            return Observation.score(0.1)

        return check

    assert stages(always_flags()) == frozenset({"tool_call"})


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
    """registry_create instantiates only factories whose return annotation's class name equals the registry type; Monitor and ControlProtocol are union aliases with no name, so it hands back the factory. Construction from config goes through create_registry_object, which always instantiates."""
    kind: RegistryType = "monitor"
    create = cast(Callable[..., Any], registry_create)
    assert create(kind, "before_monitor") is before_monitor


def test_instance_is_tagged_with_registry_info() -> None:
    instance = no_curl()
    assert registry_info(instance).name == "no_curl"
    assert registry_params(instance) == {}


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

    wrong = monitor(cast(Any, factory_returning))
    with pytest.raises(TypeError, match="Observation"):
        wrong()


def test_protocol_returning_observations_is_rejected_at_configuration() -> None:
    async def observes_not_decides(
        context: Context, step: BeforeToolCall
    ) -> Observation | None:
        return None

    def factory_returning() -> Any:
        return observes_not_decides

    wrong = protocol(cast(Any, factory_returning))
    with pytest.raises(TypeError, match="Decision"):
        wrong()


def test_unannotated_step_parameter_is_rejected_at_configuration() -> None:
    async def unannotated_step(context: Context, step: Any) -> Observation | None:
        return None

    unannotated_step.__annotations__.pop("step")

    def factory_returning() -> Any:
        return unannotated_step

    wrong = monitor(cast(Any, factory_returning))
    with pytest.raises(TypeError, match="annotat"):
        wrong()


def test_unresolvable_annotation_names_the_fix() -> None:
    async def unresolvable_step(context: Context, step: Any) -> Observation | None:
        return None

    unresolvable_step.__annotations__["step"] = "Missing"

    def factory_returning() -> Any:
        return unresolvable_step

    wrong = monitor(cast(Any, factory_returning))
    with pytest.raises(TypeError, match="Missing"):
        wrong()


def test_non_async_function_is_rejected_at_configuration() -> None:
    def not_async(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    def factory_returning() -> Any:
        return not_async

    wrong = monitor(cast(Any, factory_returning))
    with pytest.raises(TypeError, match="async"):
        wrong()
