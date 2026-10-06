# Import-order constraints: the ssl stub before anything imports anyio, and the
# loop patches before any task runs.
import loop_patch  # noqa: F401
import wasi_shims  # noqa: F401

# isort: split
import json

import wit_world
from m1.guest_host import AsyncHost
from m1.monitor import tool_call_monitor
from m3.models import BeforeToolCall, Report
from pydantic import ValidationError
from wit_world.imports import host_async


class WitWorld(wit_world.WitWorld):
    async def run_monitor(self, step: str) -> str:
        try:
            parsed = BeforeToolCall.model_validate_json(step)
        except ValidationError as ex:
            return json.dumps(
                {"error": "invalid step", "detail": ex.errors(include_url=False)}
            )
        raw = await tool_call_monitor(parsed.model_dump(), AsyncHost(host_async))
        return Report.model_validate(raw).model_dump_json()
