from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import Store, StoreModel
from pydantic import ValidationError

from inspect_sentinel._context import HostContext
from tests._fakes import host_context


class Trajectory(StoreModel):
    calls: int = 0


def test_store_as_namespaces_by_path() -> None:
    store = Store()
    context = host_context("attempt/judge", store=store)
    context.store_as(Trajectory).calls = 3
    assert context.store_as(Trajectory).calls == 3
    assert replace(context, path="escape").store_as(Trajectory).calls == 0
    assert Trajectory(store=store, instance="attempt/judge").calls == 3


def test_root_store_does_not_share_the_ambient_namespace() -> None:
    store = Store()
    host_context("", store=store).store_as(Trajectory).calls = 5
    assert Trajectory(store=store).calls == 0


def test_the_context_exposes_no_store_and_cannot_be_reassigned() -> None:
    context = host_context()
    assert not hasattr(context, "store")
    with pytest.raises(FrozenInstanceError):
        cast(Any, context).path = "elsewhere"


@pytest.mark.parametrize(
    ("parent", "name", "expected"),
    [("", "attempt", "attempt"), ("attempt", "judge", "attempt/judge")],
)
def test_child_composes_path(parent: str, name: str, expected: str) -> None:
    child = host_context(parent).child(name, "acme/judge")
    assert isinstance(child, HostContext)
    assert (child.path, child.factory) == (expected, "acme/judge")


def test_child_shares_store_host_and_recorder() -> None:
    parent = host_context("")
    parent.store_as(Trajectory).calls = 2
    child = replace(parent.child("x", "x"), path="")
    assert child.store_as(Trajectory).calls == 2
    assert child.host is parent.host
    assert child.recorder is parent.recorder


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
    assert replace(host_context(), sample_input=input).sample_input_text == expected


def test_target_is_absent_by_default() -> None:
    assert host_context().target is None


def test_root_store_validates_writes() -> None:
    bad: Any = "two"
    root = host_context("")
    with pytest.raises(ValidationError):
        root.store_as(Trajectory).calls = bad


@pytest.mark.parametrize("name", ["", "a/b"])
def test_child_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        host_context().child(name, "x")


def test_host_context_requires_a_complete_recorder() -> None:
    class RecordOnly:
        def record(self, context: object, step: object, reported: object) -> None: ...

    parent = host_context()
    with pytest.raises(TypeError, match="cancelled"):
        HostContext(
            task=None,
            task_description=None,
            sample_id=None,
            epoch=None,
            sample_description=None,
            sample_input="p",
            metadata={},
            path="",
            _store=Store(),
            host=parent.host,
            recorder=cast(Any, RecordOnly()),
        )
