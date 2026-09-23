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


def test_registry_create_returns_the_factory_for_alias_return_types() -> None:
    kind: RegistryType = "monitor"
    create = cast(Callable[..., Any], registry_create)
    assert create(kind, "before_monitor") is before_monitor


def test_instance_is_tagged_with_registry_info() -> None:
    instance = no_curl()
    assert registry_info(instance).name == "no_curl"
    assert registry_params(instance) == {}


def test_monitor_returning_decisions_is_rejected_at_configuration() -> None:
    async def check(context: Context, step: BeforeToolCall) -> Decision | None:
        return None

    wrong = monitor(cast(Any, lambda: check))
    with pytest.raises(TypeError, match="Observation"):
        wrong()


def test_protocol_returning_observations_is_rejected_at_configuration() -> None:
    async def decide(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    wrong = protocol(cast(Any, lambda: decide))
    with pytest.raises(TypeError, match="Decision"):
        wrong()


def test_unannotated_step_parameter_is_rejected_at_configuration() -> None:
    async def check(context: Context, step: Any) -> Observation | None:
        return None

    check.__annotations__.pop("step")
    wrong = monitor(cast(Any, lambda: check))
    with pytest.raises(TypeError, match="annotat"):
        wrong()


def test_non_async_function_is_rejected_at_configuration() -> None:
    def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return None

    wrong = monitor(cast(Any, lambda: check))
    with pytest.raises(TypeError, match="async"):
        wrong()
