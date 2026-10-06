import json

import probes
import wit_world
from m1.guest_host import BlockingHost, run_blocking
from m1.monitor import tool_call_monitor
from wit_world.imports import host


class WitWorld(wit_world.WitWorld):
    def run_monitor(self, step: str) -> str:
        data = json.loads(step)
        if data.get("probe"):
            return json.dumps(probes.run(host))
        return json.dumps(run_blocking(tool_call_monitor(data, BlockingHost(host))))
