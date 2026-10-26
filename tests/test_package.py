import anyio
import pytest

import inspect_sentinel
import inspect_sentinel._integration
import inspect_sentinel.config


def test_version_is_exposed() -> None:
    assert isinstance(inspect_sentinel.__version__, str)
    assert inspect_sentinel.__version__


@pytest.mark.anyio
async def test_async_tests_run() -> None:
    await anyio.sleep(0)


def test_author_facing_names_are_exported() -> None:
    expected = {
        "BeforeToolCall",
        "AfterToolCall",
        "Truncation",
        "Step",
        "ReplayStep",
        "Observation",
        "Decision",
        "Suspicion",
        "Action",
        "Context",
        "Host",
        "HumanAnswer",
        "Report",
        "Reported",
        "Sentinel",
        "Sentinels",
        "Protocol",
        "Monitor",
        "Monitors",
        "MonitorGroup",
        "ProtocolGroup",
        "Protocols",
        "monitor",
        "protocol",
        "Decisions",
        "Observations",
        "Reports",
        "run_children",
        "run_monitors",
        "run_protocols",
        "observe_only",
        "concurrent",
        "sequential",
        "human",
        "handle_escalation",
        "threshold",
        "decide_final",
        "PortabilityError",
        "Escalations",
        "HumanStandIn",
    }
    assert expected <= set(inspect_sentinel.__all__)
    for name in expected:
        assert hasattr(inspect_sentinel, name)


def test_the_replay_runner_is_internal() -> None:
    # replay's public API is the Scout adapter's (decided 2026-10-26)
    for name in (
        "replay_monitor",
        "ReplayReport",
        "steps_from_messages",
        "HumanFallback",
        "HumanPolicy",
        "HumanSource",
        "Split",
    ):
        assert name not in inspect_sentinel.__all__
        assert not hasattr(inspect_sentinel, name)


def test_integration_names_are_not_exported() -> None:
    for name in (
        "HostContext",
        "HostNative",
        "Recorder",
        "step_types",
        "Stage",
        "Final",
        "validate_decision_shape",
        "resolve_sentinel",
        "SentinelConfig",
        "SentinelEntry",
        "sentinel_from_config",
        "sentinel_config",
        "config_from_sentinel",
        "watched_stages",
        "validate_instance_name",
        "PRECEDENCE",
        "named_children",
        "run_sentinel",
    ):
        assert name not in inspect_sentinel.__all__


def test_single_child_runners_are_gone() -> None:
    for name in ("run_monitor", "run_protocol"):
        assert name not in inspect_sentinel.__all__
        assert not hasattr(inspect_sentinel, name)


def test_integration_module_exports_the_host_surface() -> None:
    assert set(inspect_sentinel._integration.__all__) == {
        "Recorder",
        "HostContext",
        "Sentinels",
        "sentinel_from_config",
        "config_from_sentinel",
        "kept_human_decision",
        "validate_instance_name",
        "resolve_sentinel",
        "step_types",
        "supersedes",
        "watched_stages",
        "run_sentinel",
        "HostNative",
    }


def test_the_config_module_exports_the_configuration_round_trip() -> None:
    assert set(inspect_sentinel.config.__all__) == {
        "sentinel_config",
        "sentinel_from_config",
        "watched_stages",
    }
    # inspect_ai's contract keeps its name for the same function
    assert (
        inspect_sentinel._integration.config_from_sentinel
        is inspect_sentinel.config.sentinel_config
    )


def test_wire_types_come_from_inspect_ai() -> None:
    from inspect_ai.event import SentinelAction, SentinelSuspicion

    assert inspect_sentinel.Action is SentinelAction
    assert inspect_sentinel.Suspicion is SentinelSuspicion
