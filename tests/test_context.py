from dataclasses import replace
from typing import Any, cast

import pytest
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import StoreModel
from pydantic import ValidationError

from inspect_sentinel._context import RunnerContext
from tests._fakes import runner_context


class Trajectory(StoreModel):
    calls: int = 0


def test_store_as_namespaces_by_path() -> None:
    context = runner_context("attempt/judge")
    context.store_as(Trajectory).calls = 3
    assert context.store_as(Trajectory).calls == 3
    assert replace(context, path="escape").store_as(Trajectory).calls == 0
    assert Trajectory(store=context.store, instance="attempt/judge").calls == 3


def test_root_store_does_not_share_the_ambient_namespace() -> None:
    root = runner_context("")
    root.store_as(Trajectory).calls = 5
    assert Trajectory(store=root.store).calls == 0


@pytest.mark.parametrize(
    ("parent", "name", "expected"),
    [("", "attempt", "attempt"), ("attempt", "judge", "attempt/judge")],
)
def test_child_composes_path(parent: str, name: str, expected: str) -> None:
    child = runner_context(parent).child(name, "acme/judge")
    assert isinstance(child, RunnerContext)
    assert (child.path, child.factory) == (expected, "acme/judge")


def test_child_shares_store_host_and_recorder() -> None:
    parent = runner_context("")
    child = parent.child("x", "x")
    assert child.store is parent.store
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
def test_input_text_joins_the_messages(
    input: str | list[ChatMessage], expected: str
) -> None:
    assert replace(runner_context(), input=input).input_text == expected


def test_target_is_absent_by_default() -> None:
    assert runner_context().target is None


def test_root_store_validates_writes() -> None:
    bad: Any = "two"
    root = runner_context("")
    with pytest.raises(ValidationError):
        root.store_as(Trajectory).calls = bad


@pytest.mark.parametrize("name", ["", "a/b"])
def test_child_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        runner_context().child(name, "x")


def test_runner_context_requires_a_complete_recorder() -> None:
    class RecordOnly:
        def record(self, context: object, step: object, reported: object) -> None: ...

    parent = runner_context()
    with pytest.raises(TypeError, match="cancelled"):
        RunnerContext(
            task=None,
            task_description=None,
            sample_id=None,
            epoch=None,
            sample_description=None,
            input="p",
            metadata={},
            path="",
            store=parent.store,
            host=parent.host,
            recorder=cast(Any, RecordOnly()),
        )
