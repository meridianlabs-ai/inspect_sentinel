import anyio
import pytest

import inspect_sentinel
import inspect_sentinel._integration


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
        "Step",
        "Stage",
        "Observation",
        "Decision",
        "Suspicion",
        "Action",
        "Context",
        "Host",
        "Report",
        "Reported",
        "Children",
        "ControlProtocol",
        "Monitor",
        "Monitors",
        "Protocols",
        "monitor",
        "protocol",
        "Decisions",
        "Observations",
        "Reports",
        "run_children",
        "run_monitor",
        "run_monitors",
        "run_protocol",
        "run_protocols",
        "observe",
        "concurrent",
        "threshold",
    }
    assert expected <= set(inspect_sentinel.__all__)
    for name in expected:
        assert hasattr(inspect_sentinel, name)


def test_integration_names_are_not_exported() -> None:
    for name in (
        "RunnerContext",
        "Recorder",
        "step_types",
        "validate_decision",
        "compile_sentinel",
        "SentinelSpec",
        "check_instance_name",
        "PRECEDENCE",
        "named_children",
    ):
        assert name not in inspect_sentinel.__all__


def test_integration_module_exports_the_dispatcher_surface() -> None:
    assert set(inspect_sentinel._integration.__all__) == {
        "Recorder",
        "RunnerContext",
        "SentinelSpec",
        "check_instance_name",
        "compile_sentinel",
        "step_types",
        "validate_decision",
    }
