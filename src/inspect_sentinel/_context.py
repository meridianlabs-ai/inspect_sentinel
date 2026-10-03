from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from inspect_ai.model import ChatMessage
from inspect_ai.scorer import Target
from inspect_ai.util import StoreModel

from ._host import Host, active_step

SMT = TypeVar("SMT", bound=StoreModel)


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

    sample_input: str | list[ChatMessage]
    """The sample's input: the assignment the agent was given. Judge a step against `step.input`, which is exactly what the model was sent; use this for what the agent was asked to do."""

    metadata: dict[str, Any]
    """Task metadata merged with sample metadata."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer."""

    host: Host
    """Inference through the host's models, and asking a person. See `Host`."""

    target: Target | None = None
    """The expected answer. None until monitors and protocols can opt in with `target=True`."""

    @property
    def sample_input_text(self) -> str:
        """`sample_input` as one string: `sample_input` itself if a string, else its messages' text joined with newlines."""
        if isinstance(self.sample_input, str):
            return self.sample_input
        return "\n".join(message.text for message in self.sample_input)

    def store_as(self, model_cls: type[SMT]) -> SMT:
        """Typed view of this instance's state, namespaced by `path`.

        Renaming the instance (its mapping key) or wrapping it in another layer changes `path`, so its state moves with it. The store is the one the host supplied for the running step, so call this while the monitor or protocol runs; outside a step it raises `RuntimeError`.

        Args:
            model_cls: The `StoreModel` subclass to read and write through.
        """
        return model_cls(store=active_step("store_as()").store, instance=self.path)


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
