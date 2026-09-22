import anyio
import pytest

import inspect_sentinel


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
    }
    assert expected <= set(inspect_sentinel.__all__)
    for name in expected:
        assert hasattr(inspect_sentinel, name)


def test_integration_names_are_not_exported() -> None:
    assert "RunnerContext" not in inspect_sentinel.__all__
    assert "Recorder" not in inspect_sentinel.__all__
