from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, runtime_checkable
from typing import Protocol as TypingProtocol

from inspect_ai.model import ChatMessage, GenerateConfig, Model, ModelOutput
from inspect_ai.tool import ToolCall, ToolInfo

from ._report import Decision, Failed, Report, Reported
from ._step import Step

if TYPE_CHECKING:
    from ._context import Context


@dataclass(frozen=True)
class HumanAnswer:
    """What a person answered when `Host.ask_human` asked them about a step."""

    decision: str
    """The choice the person picked, one of the `choices` they were offered (`approve`, `reject`, `terminate` or `modify`), or, when the person ended the prompt without one of them, `reject` if it was offered and `terminate` otherwise."""

    reason: str | None = None
    """The reason the surface gives for the answer, if any: the person's typed reason, or fixed text such as "Human operator rejected the tool call."."""

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

        The host is handed the step itself and renders it by its type: before a tool call the call, its view, the conversation and `step.escalations` (who escalated and why); after a call the same with the result the model is about to receive. A host raises a clear error for a stage it cannot render. When the person ends the prompt without one of the `choices`, the host answers `reject` if it was offered and `terminate` otherwise, so the step fails closed; `human()` accepts that `terminate` even when it was not offered. It notifies the person and marks the sample as awaiting a person while it waits. `human()` calls this; most protocols never need to.

        Args:
            step: The step to decide about.
            choices: What the person may pick, from `approve`, `reject`, `terminate` and `modify`, in the order to present them.
        """
        ...


@runtime_checkable
class Recorder(TypingProtocol):
    """Where the runner records every report. Runner-facing; authors never call it."""

    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        """Record one child's report for the step it examined.

        Args:
            context: The child's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the child examined.
            reported: The child's report with its instance identity.
        """
        ...

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        """Record that a monitor function raised instead of reporting. The step goes on; a protocol that reads the layer's observations fails unless it checks `Observations.failed` first.

        Args:
            context: The monitor's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the monitor was examining.
            failed: The failure with its instance identity.
        """
        ...

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        """Record that a child was cancelled before it finished: a sibling decided `terminate` or called `decide_final()`, a sibling raised, or the layer was cancelled from above. Any reports it already made, from the earlier functions of an instance whose factory returned several, stand.

        The cause is not recorded here; a `terminate`, a `decide_final()` or an exception in the same layer says which it was.

        Args:
            context: The child's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        """Record that a final decision from a descendant passed this layer without running its decision logic. The step ended with that decision unless a later superseded record says otherwise.

        The final decision itself is recorded through `record` for the protocol that made it when it takes effect, at the root, after the `bypassed` record of each layer it passed.

        Args:
            context: The child's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        """Record a final decision that lost a race to another final decision in the same layer, or that an exception in the same layer outranked; it did not take effect.

        Also records a `terminate`, already recorded through `record`, that a sibling's `decide_final()` outran in a runner-managed group (`run_children`, `run_protocols`): the step ended with the final decision instead. A protocol that fans out with its own task group gets no such record; its children's decisions stay as recorded.

        Args:
            context: The deciding protocol's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the protocol was examining.
            reported: The decision that did not take effect, with its instance identity.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class HostContext:
    """What the host passes to `run_sentinel`: the top layer's `Context`, and where the runner records reports.

    The host builds the `Context` with an empty `path` and the sample store as `_store`; monitors and protocols reach the store only through `store_as()`.
    """

    context: Context
    """The top layer's context, as monitors and protocols will read it."""

    recorder: Recorder
    """Where the runner records reports, failures and cancellations."""

    def __post_init__(self) -> None:
        # a runtime check for hosts that are not type-checked against Recorder
        if not isinstance(cast(object, self.recorder), Recorder):
            raise TypeError(
                f"Recorder {type(self.recorder).__name__} must implement record(), failed(), cancelled(), bypassed() and superseded()."
            )


@dataclass(frozen=True)
class RunState:
    recorder: Recorder
    factory: str


def run_state(context: Context) -> RunState:
    # getattr: `_run` is hidden from authors, not from the runner
    run: RunState | None = getattr(context, "_run", None)
    if run is None:
        raise TypeError(
            "The runner needs a Context it built and passed to this monitor or protocol; a Context constructed elsewhere cannot record reports."
        )
    return run
