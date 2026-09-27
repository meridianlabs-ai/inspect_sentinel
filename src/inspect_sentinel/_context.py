from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.tool import ToolInfo
from inspect_ai.util import Store, StoreModel

from ._report import Decision, Report, Reported
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
        """Generate with the host's model.

        Args:
            input: A prompt string or a list of chat messages.
            model: A model name or a model role. The default role is `monitor`.
            tools: Tool definitions to offer the model.
            config: Generation configuration.
        """
        ...


@runtime_checkable
class Recorder(Protocol):
    """Where the runner records every report. Runner-facing; authors never call it."""

    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None:
        """Record one child's report for the step it examined.

        Args:
            context: The child's context; its `path` identifies the instance.
            step: The step the child examined.
            reported: The child's report with its instance identity.
        """
        ...

    def cancelled(self, context: Context, step: Step, name: str) -> None:
        """Record that a child was cancelled before it reported: a sibling decided `terminate`, a sibling raised, or the layer was cancelled from above.

        The cause is not recorded here; a `terminate` or an exception in the same layer says which it was.

        Args:
            context: The child's context; its `path` identifies the instance.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...


@dataclass(frozen=True, kw_only=True)
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
        """Typed view of this instance's state, namespaced by `path`.

        Args:
            model_cls: The `StoreModel` subclass to read and write through.
        """
        return model_cls(store=self.store, instance=self.path)


@dataclass(frozen=True, kw_only=True)
class RunnerContext(Context):
    """A `Context` plus what the runner needs."""

    recorder: Recorder
    """Where the runner records reports and cancellations."""

    decisions: list[Reported[Decision]] = field(
        default_factory=list[Reported[Decision]]
    )
    """Decisions recorded by this layer's children, for the binding floor. Runner-owned; the dispatcher builds a fresh RunnerContext per step, so nothing accumulates across steps."""

    def __post_init__(self) -> None:
        # a runtime check for hosts that are not type-checked against Recorder
        if not isinstance(cast(object, self.recorder), Recorder):
            raise TypeError(
                f"Recorder {type(self.recorder).__name__} must implement record() and cancelled()."
            )

    def child(self, name: str) -> RunnerContext:
        """The context for a child of this layer.

        Args:
            name: The child's instance name, appended to this layer's `path`. Must be non-empty and must not contain `/`.
        """
        check_instance_name(name)
        return replace(
            self, path=f"{self.path}/{name}" if self.path else name, decisions=[]
        )


def check_instance_name(name: object) -> str:
    """Return `name` if it is a string that can be a path segment, else raise `ValueError`.

    Args:
        name: A candidate instance name; configuration may hand over a non-string key.
    """
    if not isinstance(name, str) or name == "" or "/" in name:
        raise ValueError(
            f"Instance name {name!r} must be a non-empty string without '/'."
        )
    return name
