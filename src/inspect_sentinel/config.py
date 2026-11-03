"""A sentinel as configuration, and the stages it watches.

`sentinel_config()` records constructed monitors and protocols as the configuration an eval log records, `sentinel_from_config()` builds them again from it, and `watched_stages()` says which steps a monitor or protocol is run at. Tools that run a sentinel somewhere other than an eval, such as a scan that replays it over logs, use these to carry it between processes and to tell which steps it reads.
"""

from ._config import sentinel_config, sentinel_from_config
from ._decorators import watched_stages

__all__ = ["sentinel_config", "sentinel_from_config", "watched_stages"]
