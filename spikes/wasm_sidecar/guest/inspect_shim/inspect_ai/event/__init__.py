"""Stand-in for inspect_ai.event.

The two type aliases sentinel's reports use, copied from
inspect_ai/event/_sentinel.py (feature/sentinel). The module itself imports
the event base classes, which import much of inspect_ai.
"""

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, FiniteFloat

SentinelAction: TypeAlias = Literal[
    "continue", "modify", "reject", "terminate", "escalate"
]
SentinelSuspicion: TypeAlias = (
    FiniteFloat | Annotated[dict[str, FiniteFloat], Field(min_length=1)]
)
