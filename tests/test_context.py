from dataclasses import FrozenInstanceError, replace
from typing import Any, cast, get_type_hints

import pytest
from inspect_ai.core import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    Store,
    StoreModel,
)
from pydantic import ValidationError

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    EvalContext,
    Host,
    Monitor,
    Observation,
    monitor,
    observe_only,
    run_monitors,
)
from inspect_sentinel._host import HostContext
from inspect_sentinel._runner import run_sentinel
from tests._fakes import (
    ListRecorder,
    before_step,
    eval_context,
    host_context,
    in_step,
)


class Trajectory(StoreModel):
    calls: int = 0


def test_store_as_namespaces_by_path() -> None:
    store = Store()
    with in_step("attempt/judge", store=store) as context:
        context.store_as(Trajectory).calls = 3
        assert context.store_as(Trajectory).calls == 3
        assert replace(context, path="escape").store_as(Trajectory).calls == 0
        assert Trajectory(store=store, instance="attempt/judge").calls == 3


def test_store_as_outside_a_step_is_an_error() -> None:
    with pytest.raises(RuntimeError, match=r"store_as\(\) works only .* outside"):
        host_context().context.store_as(Trajectory)


def test_root_store_does_not_share_the_ambient_namespace() -> None:
    store = Store()
    with in_step("", store=store) as context:
        context.store_as(Trajectory).calls = 5
    assert Trajectory(store=store).calls == 0


def test_the_contexts_annotations_resolve_at_runtime() -> None:
    assert get_type_hints(Context)["host"] is Host


def test_the_context_exposes_no_store_and_cannot_be_reassigned() -> None:
    context = host_context().context
    assert not hasattr(context, "store")
    with pytest.raises(FrozenInstanceError):
        cast(Any, context).path = "elsewhere"
    with pytest.raises(FrozenInstanceError):
        cast(Any, context).eval = None
    with pytest.raises(FrozenInstanceError):
        cast(Any, context.eval).task = "other"


@pytest.mark.anyio
async def test_a_child_is_given_a_plain_context_under_its_layer() -> None:
    seen: list[Context] = []

    @monitor
    def sees_context() -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            seen.append(context)
            return None

        return check

    with in_step("attempt") as parent:
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
    assert replace(eval_context(), sample_input=input).sample_input_text == expected


def test_target_is_absent_by_default() -> None:
    assert eval_context().target is None


@monitor
def reads_eval() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        context.store_as(Trajectory).calls += 1
        if context.eval is None:
            return Observation.score(0.0, "outside an eval")
        return Observation.score(
            0.5,
            f"{context.eval.task}/{context.eval.sample_id}: {context.eval.sample_input_text}",
        )

    return check


@pytest.mark.parametrize(
    ("eval_context_", "explanation"),
    [(eval_context(), "t/1: prompt"), (None, "outside an eval")],
)
@pytest.mark.anyio
async def test_a_monitor_runs_with_and_without_an_eval(
    eval_context_: EvalContext | None, explanation: str
) -> None:
    store = Store()
    recorder = ListRecorder()
    built = host_context(recorder=recorder, store=store)
    host = replace(built, context=replace(built.context, eval=eval_context_))
    await run_sentinel(observe_only([reads_eval()]), host, before_step())
    [recorded] = recorder.records
    assert recorded.context.eval == eval_context_
    assert recorded.reported.report.explanation == explanation
    assert Trajectory(store=store, instance=recorded.context.path).calls == 1


def test_root_store_validates_writes() -> None:
    bad: Any = "two"
    with in_step("") as root:
        with pytest.raises(ValidationError):
            root.store_as(Trajectory).calls = bad


def test_host_context_requires_a_complete_recorder() -> None:
    class RecordOnly:
        def record(self, context: object, step: object, reported: object) -> None: ...

    with pytest.raises(TypeError, match="cancelled"):
        HostContext(
            context=host_context().context,
            recorder=cast(Any, RecordOnly()),
            store=Store(),
        )
