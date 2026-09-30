from __future__ import annotations

from dataclasses import dataclass, replace
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

    def record(
        self, context: RunnerContext, step: Step, reported: Reported[Report]
    ) -> None:
        """Record one child's report for the step it examined.

        Args:
            context: The child's context; its `path` identifies the instance and its `factory` the code that ran.
            step: The step the child examined.
            reported: The child's report with its instance identity.
        """
        ...

    def cancelled(self, context: RunnerContext, step: Step, name: str) -> None:
        """Record that a child was cancelled before it finished: a sibling decided `terminate` or called `decide_final()`, a sibling raised, or the layer was cancelled from above. Any reports it already made, from the earlier functions of an instance whose factory returned several, stand.

        The cause is not recorded here; a `terminate`, a `decide_final()` or an exception in the same layer says which it was.

        Args:
            context: The child's context; its `path` identifies the instance and its `factory` the code that ran.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...

    def bypassed(self, context: RunnerContext, step: Step, name: str) -> None:
        """Record that a final decision from a descendant passed this layer without running its decision logic. The step ended with that decision unless a later superseded record says otherwise.

        The final decision itself is recorded through `record` for the protocol that made it when it takes effect, at the root, after the `bypassed` record of each layer it passed.

        Args:
            context: The child's context; its `path` identifies the instance and its `factory` the code that ran.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...

    def superseded(
        self, context: RunnerContext, step: Step, reported: Reported[Decision]
    ) -> None:
        """Record a final decision that lost a race to another final decision in the same layer, or that an exception in the same layer outranked; it did not take effect.

        Also records a `terminate`, already recorded through `record`, that a sibling's `decide_final()` outran in a runner-managed group (`run_children`, `run_protocols`): the step ended with the final decision instead. A protocol that fans out with its own task group gets no such record; its children's decisions stay as recorded.

        Args:
            context: The deciding protocol's context; its `path` identifies the instance and its `factory` the code that ran.
            step: The step the protocol was examining.
            reported: The decision that did not take effect, with its instance identity.
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

    factory: str = ""
    """Registry name of the factory of the instance at `path`, package prefix included, e.g. `inspect_sentinel/concurrent`. Empty in the context the dispatcher builds; `run_root` sets the root's."""

    def __post_init__(self) -> None:
        # a runtime check for hosts that are not type-checked against Recorder
        if not isinstance(cast(object, self.recorder), Recorder):
            raise TypeError(
                f"Recorder {type(self.recorder).__name__} must implement record(), cancelled(), bypassed() and superseded()."
            )

    def child(self, name: str, factory: str) -> RunnerContext:
        """The context for a child of this layer.

        Args:
            name: The child's instance name, appended to this layer's `path`. Must be non-empty and must not contain `/`.
            factory: The child's registry name, as `registry_info(child).name` gives it.
        """
        validate_instance_name(name)
        return replace(
            self,
            path=f"{self.path}/{name}" if self.path else name,
            factory=factory,
        )


def validate_instance_name(name: object) -> str:
    """Return `name` if it is a string that can be a path segment, else raise `ValueError`.

    Args:
        name: A candidate instance name; configuration may hand over a non-string key.
    """
    if not isinstance(name, str) or name == "" or "/" in name:
        raise ValueError(
            f"Instance name {name!r} must be a non-empty string without '/'."
        )
    return name
