import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run `@pytest.mark.anyio` tests on asyncio only.

    Without this fixture anyio parametrizes every async test over asyncio and trio, and trio is not a dependency of this project.
    """
    return "asyncio"
