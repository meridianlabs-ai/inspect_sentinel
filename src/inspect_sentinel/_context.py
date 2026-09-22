"""What a monitor or protocol is given besides the step, and the interfaces its effects go through."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.tool import ToolInfo
from inspect_ai.util import Store, StoreModel

from ._report import Report, Reported
from ._step import Step

SMT = TypeVar("SMT", bound=StoreModel)


class Host(Protocol):
    """What a monitor or protocol may do to the outside world. Author-facing."""

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        """Generate with `model`, a model name or a model role; the default role is `monitor`."""
        ...


class Recorder(Protocol):
    """Where the runner records every report. Runner-facing; authors never call it."""

    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None:
        """Record one child's report for the step it examined."""
        ...


@dataclass(frozen=True)
class Context:
    """What a monitor or protocol is given. Constant for a sample; nothing in it came from the agent."""

    task: str | None
    """Task name, or the deployed agent's identifier."""

    task_description: str | None
    """What agents in this setting are asked to do."""

    sample_id: str | int | None
    """Dataset sample identifier."""

    epoch: int | None
    """Epoch of the sample, when the task runs several."""

    sample_description: str | None
    """What is distinctive about this instance."""

    input: str | list[ChatMessage]
    """The prompt this agent was given."""

    metadata: dict[str, Any]
    """Task metadata merged with sample metadata."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer."""

    store: Store
    """Per-sample state. Use `store_as()` for a typed, namespaced view."""

    host: Host
    """Inference and other effects on the outside world."""

    target: Target | None = None
    """The expected answer. Absent unless the monitor asked for it."""

    def store_as(self, model_cls: type[SMT]) -> SMT:
        """Typed view of this instance's state, namespaced by `path`."""
        return model_cls(store=self.store, instance=self.path or None)


@dataclass(frozen=True, kw_only=True)
class RunnerContext(Context):
    """A `Context` plus what the runner needs. Built by the dispatcher, never by authors."""

    recorder: Recorder
    """Where the runner records reports."""

    def child(self, name: str) -> RunnerContext:
        """The context for a child of this layer, with `path` extended by `name`."""
        return replace(self, path=f"{self.path}/{name}" if self.path else name)
