from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import Store, StoreModel
from pydantic import ValidationError

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Monitor,
    Observation,
    monitor,
    run_monitors,
)
from inspect_sentinel._host import HostContext
from tests._fakes import before_step, layer_context


class Trajectory(StoreModel):
    calls: int = 0


def test_store_as_namespaces_by_path() -> None:
    store = Store()
    context = layer_context("attempt/judge", store=store)
    context.store_as(Trajectory).calls = 3
    assert context.store_as(Trajectory).calls == 3
    assert replace(context, path="escape").store_as(Trajectory).calls == 0
    assert Trajectory(store=store, instance="attempt/judge").calls == 3


def test_root_store_does_not_share_the_ambient_namespace() -> None:
    store = Store()
    layer_context("", store=store).store_as(Trajectory).calls = 5
    assert Trajectory(store=store).calls == 0


def test_the_context_exposes_no_store_and_cannot_be_reassigned() -> None:
    context = layer_context()
    assert not hasattr(context, "store")
    with pytest.raises(FrozenInstanceError):
        cast(Any, context).path = "elsewhere"


@pytest.mark.anyio
async def test_a_child_is_given_a_plain_context_under_its_layer() -> None:
    seen: list[Context] = []

    @monitor
    def sees_context() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            seen.append(context)
            return None

        return check

    parent = layer_context("attempt")
    parent.store_as(Trajectory).calls = 2
    await run_monitors({"judge": sees_context()}, parent, before_step())
    [child] = seen
    assert type(child) is Context
    assert child.path == "attempt/judge"
    assert child.host is parent.host
    assert replace(child, path="attempt").store_as(Trajectory).calls == 2


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ("list the files", "list the files"),
        (
            [
                ChatMessageSystem(content="be careful"),
                ChatMessageUser(content="list the files"),
            ],
            "be careful\nlist the files",
        ),
    ],
)
def test_sample_input_text_joins_the_messages(
    input: str | list[ChatMessage], expected: str
) -> None:
    assert replace(layer_context(), sample_input=input).sample_input_text == expected


def test_target_is_absent_by_default() -> None:
    assert layer_context().target is None


def test_root_store_validates_writes() -> None:
    bad: Any = "two"
    root = layer_context("")
    with pytest.raises(ValidationError):
        root.store_as(Trajectory).calls = bad


def test_host_context_requires_a_complete_recorder() -> None:
    class RecordOnly:
        def record(self, context: object, step: object, reported: object) -> None: ...

    with pytest.raises(TypeError, match="cancelled"):
        HostContext(context=layer_context(), recorder=cast(Any, RecordOnly()))
