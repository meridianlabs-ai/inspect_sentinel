import pytest


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Run every anyio-marked test on both backends. The runner's cancellation paths differ between them, so trio is a dev dependency."""
    return str(request.param)
