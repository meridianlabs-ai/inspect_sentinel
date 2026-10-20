import os

import pytest

# The Envoy shim's tests run docker, which forks, while gRPC's threads run.
# gRPC's handlers for a fork are for a child that goes on using gRPC, which
# docker doesn't, and they log at every fork. Set before gRPC is imported.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Run every anyio-marked test on both backends. The runner's cancellation paths differ between them, so trio is a dev dependency."""
    return str(request.param)
