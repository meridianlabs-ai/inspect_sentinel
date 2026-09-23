from dataclasses import replace
from typing import Any, NamedTuple

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolInfo
from inspect_ai.util import Store, StoreModel
from pydantic import ValidationError

from inspect_sentinel._context import Context, RunnerContext
from inspect_sentinel._report import Report, Reported
from inspect_sentinel._step import Step


class FakeHost:
    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        return ModelOutput.from_content(model="fake", content="ok")


class Recorded(NamedTuple):
    context: Context
    step: Step
    reported: Reported[Report]


class ListRecorder:
    def __init__(self) -> None:
        self.records: list[Recorded] = []

    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None:
        self.records.append(Recorded(context, step, reported))


class Trajectory(StoreModel):
    calls: int = 0


def _context(path: str = "") -> RunnerContext:
    return RunnerContext(
        task="t",
        task_description=None,
        sample_id=1,
        epoch=1,
        sample_description=None,
        input="prompt",
        metadata={},
        path=path,
        store=Store(),
        host=FakeHost(),
        recorder=ListRecorder(),
    )


def test_store_as_namespaces_by_path() -> None:
    context = _context("attempt/judge")
    context.store_as(Trajectory).calls = 3
    assert context.store_as(Trajectory).calls == 3
    assert replace(context, path="escape").store_as(Trajectory).calls == 0
    assert Trajectory(store=context.store, instance="attempt/judge").calls == 3


def test_root_store_does_not_share_the_ambient_namespace() -> None:
    root = _context("")
    root.store_as(Trajectory).calls = 5
    assert Trajectory(store=root.store).calls == 0


@pytest.mark.parametrize(
    ("parent", "name", "expected"),
    [("", "attempt", "attempt"), ("attempt", "judge", "attempt/judge")],
)
def test_child_composes_path(parent: str, name: str, expected: str) -> None:
    child = _context(parent).child(name)
    assert isinstance(child, RunnerContext)
    assert child.path == expected


def test_child_shares_store_host_and_recorder() -> None:
    parent = _context("")
    child = parent.child("x")
    assert child.store is parent.store
    assert child.host is parent.host
    assert child.recorder is parent.recorder


def test_target_is_absent_by_default() -> None:
    assert _context().target is None


def test_root_store_validates_writes() -> None:
    bad: Any = "two"
    root = _context("")
    with pytest.raises(ValidationError):
        root.store_as(Trajectory).calls = bad
