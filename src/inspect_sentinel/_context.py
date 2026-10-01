from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar, cast, runtime_checkable
from typing import Protocol as TypingProtocol

from inspect_ai.model import ChatMessage, GenerateConfig, Model, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.tool import ToolCall, ToolInfo
from inspect_ai.util import Store, StoreModel

from ._report import Decision, Report, Reported
from ._step import Step

SMT = TypeVar("SMT", bound=StoreModel)


@dataclass(frozen=True)
class HumanAnswer:
    """What a person answered when `Host.ask_human` asked them about a step."""

    decision: str
    """The choice the person picked, one of the `choices` they were offered: `approve`, `reject`, `terminate` or `modify`."""

    reason: str | None = None
    """The reason the person typed, if any."""

    modified: ToolCall | None = None
    """The replacement call, when the person chose `modify` at a tool stage."""


class Host(TypingProtocol):
    """What a monitor or protocol may do to the outside world. Author-facing.

    Inference and asking a person are the only effects for now. Outbound HTTP through named endpoints (`fetch`) is planned. Meanwhile a monitor running in-process may call inspect_ai APIs directly, such as `sandbox()` or `logging`, at the cost of portability to a proxy, which a future `portable=False` will declare.
    """

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | Model | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        """Generate with a model the host resolves.

        The host resolves the model as inspect's `get_model(model=, role=)` does: `role`'s model if that role is configured for the task or eval, otherwise `model`, otherwise the agent's model, with a warning once per eval naming the role to set. With neither `model` nor `role`, the role is `monitor`.

        Args:
            input: A prompt string or a list of chat messages.
            model: A model name or instance, used when `role` is not configured.
            role: A model role, e.g. `trusted`. Defaults to `monitor` when `model` is None.
            tools: Tool definitions to offer the model.
            config: Generation configuration.
        """
        ...

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        """Ask a person to decide about a step, and wait for the answer.

        The host is handed the step itself and renders it by its type: before a tool call the call, its view, the conversation and `step.escalations` (who escalated and why); after a call the same with the result the model is about to receive. A host raises a clear error for a stage it cannot render. It notifies the person and marks the sample as awaiting a person while it waits. `human()` calls this; most protocols never need to.

        Args:
            step: The step to decide about.
            choices: What the person may pick, from `approve`, `reject`, `terminate` and `modify`, in the order to present them.
        """
        ...


@runtime_checkable
class Recorder(TypingProtocol):
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
    """What agents in this setting are asked to do. None until inspect_ai has `Task(description=)`."""

    sample_id: str | int | None
    """Dataset sample identifier."""

    epoch: int | None
    """Epoch of the sample, when the task runs several."""

    sample_description: str | None
    """What is distinctive about this instance. None until inspect_ai has `Sample(description=)`."""

    input: str | list[ChatMessage]
    """The sample's input: the assignment the agent was given. Judge a step against `step.input`, which is exactly what the model was sent; use this for what the agent was asked to do."""

    metadata: dict[str, Any]
    """Task metadata merged with sample metadata."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer."""

    store: Store
    """The whole sample store, the agent's state included; not namespaced. Use `store_as()` for this instance's own state."""

    host: Host
    """Inference through the host's models, and asking a person. See `Host`."""

    target: Target | None = None
    """The expected answer. None until monitors and protocols can opt in with `target=True`."""

    @property
    def input_text(self) -> str:
        """The sample's `input` as one string: `input` itself if a string, else its messages' text joined with newlines."""
        if isinstance(self.input, str):
            return self.input
        return "\n".join(message.text for message in self.input)

    def store_as(self, model_cls: type[SMT]) -> SMT:
        """Typed view of this instance's state, namespaced by `path`.

        Renaming the instance (its mapping key) or wrapping it in another layer changes `path`, so its state moves with it.

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
