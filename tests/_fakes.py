from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, NamedTuple, TypeVar, overload

from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallError, ToolCallView, ToolChoice, ToolInfo
from inspect_ai.util import Store

from inspect_sentinel._context import Context, EvalContext
from inspect_sentinel._host import (
    HostContext,
    HumanAnswer,
    enter_layer,
    running_step,
)
from inspect_sentinel._report import Decision, Escalations, Failed, Report, Reported
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)

T = TypeVar("T")


class View(Sequence[T]):
    # a read-only view of a host's list, as a step may hold one, counting the
    # items read through it
    def __init__(self, items: Sequence[T]) -> None:
        self._items = list(items)
        self.reads = 0

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        item = self._items[index]
        self.reads += 1
        return item


# the forms a step's sequence fields can take
SEQUENCES: list[Any] = [list, tuple, View]


class Asked(NamedTuple):
    step: Step
    escalations: Escalations
    choices: tuple[str, ...]


class Offered(NamedTuple):
    tools: list[ToolInfo]
    tool_choice: ToolChoice | None
    config: GenerateConfig | None


class FakeHost:
    def __init__(
        self,
        *answers: HumanAnswer,
        completion: str = "ok",
        outputs: Sequence[ModelOutput] = (),
        eval: EvalContext | None = None,
    ) -> None:
        self.eval = eval
        self.answers = list(answers)
        self.asked: list[Asked] = []
        self.completion = completion
        self.outputs = list(outputs)
        self.models: list[tuple[str | None, str | None]] = []
        self.inputs: list[list[ChatMessage]] = []
        self.offered: list[Offered] = []

    async def ask_human(
        self,
        step: Step,
        escalations: Escalations,
        choices: Sequence[str],
    ) -> HumanAnswer:
        self.asked.append(Asked(step, escalations, tuple(choices)))
        return self.answers.pop(0)

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        self.models.append((model, role))
        self.offered.append(Offered(tools or [], tool_choice, config))
        self.inputs.append(
            [ChatMessageUser(content=input)] if isinstance(input, str) else input
        )
        return (
            self.outputs.pop(0)
            if self.outputs
            else ModelOutput.from_content(model="fake", content=self.completion)
        )


class Recorded(NamedTuple):
    context: Context
    factory: str
    step: Step
    reported: Reported[Report]


class Instance(NamedTuple):
    path: str
    factory: str


class ListRecorder:
    def __init__(self) -> None:
        self.records: list[Recorded] = []
        self.cancellations: list[tuple[str, str]] = []
        self.cancelled_instances: list[Instance] = []
        self.bypassed_layers: list[tuple[str, str]] = []
        self.bypassed_instances: list[Instance] = []
        self.supersessions: list[Recorded] = []
        self.failures: list[Failed] = []
        self.failed_instances: list[Instance] = []

    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        self.records.append(Recorded(context, factory, step, reported))

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        self.failures.append(failed)
        self.failed_instances.append(Instance(context.path, factory))

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        self.cancellations.append((context.path, name))
        self.cancelled_instances.append(Instance(context.path, factory))

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        self.bypassed_layers.append((context.path, name))
        self.bypassed_instances.append(Instance(context.path, factory))

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        self.supersessions.append(Recorded(context, factory, step, reported))


def eval_context() -> EvalContext:
    return EvalContext(
        task="t",
        task_description=None,
        sample_id=1,
        epoch=1,
        sample_description=None,
        sample_input="prompt",
        metadata={},
    )


def host_context(
    recorder: ListRecorder | None = None,
    host: FakeHost | None = None,
    store: Store | None = None,
) -> HostContext:
    return HostContext(
        host=host or FakeHost(),
        recorder=recorder or ListRecorder(),
        store=store or Store(),
    )


@contextmanager
def in_step(
    path: str = "",
    recorder: ListRecorder | None = None,
    host: FakeHost | None = None,
    store: Store | None = None,
    factory: str = "",
) -> Generator[Context]:
    built = host_context(recorder, host, store)
    with running_step(built.recorder, built.store):
        enter_layer(path, factory)
        yield replace(built.context, path=path)


def before_step() -> BeforeToolCall:
    return before_tool_call("bash", cmd="ls")


def after_step() -> AfterToolCall:
    return after_tool_call("bash", "out", cmd="ls")


def before_tool_call(function: str, /, **arguments: Any) -> BeforeToolCall:
    return BeforeToolCall(
        conversation="conversation",
        message="",
        message_id="msg_1",
        call=ToolCall(id="call_1", function=function, arguments=arguments),
        view=ToolCallView(),
        input=[],
        agent_messages=[],
        history=[],
    )


def after_tool_call(
    function: str,
    /,
    result: str = "",
    *,
    error: ToolCallError | None = None,
    **arguments: Any,
) -> AfterToolCall:
    before = before_tool_call(function, **arguments)
    return AfterToolCall(
        conversation=before.conversation,
        message=before.message,
        message_id=before.message_id,
        call=before.call,
        result=ChatMessageTool(
            content=result,
            tool_call_id=before.call.id,
            function=function,
            error=error,
        ),
        output=result,
        view=before.view,
        input=before.input,
        agent_messages=before.agent_messages,
        history=before.history,
    )


def before_generate_step() -> BeforeGenerate:
    return BeforeGenerate(
        model="mockllm/model",
        conversation="c",
        input=[ChatMessageUser(content="go")],
        agent_messages=[ChatMessageUser(content="go")],
        history=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
    )


def after_generate_step() -> AfterGenerate:
    return AfterGenerate(
        model="mockllm/model",
        conversation="c",
        input=[ChatMessageUser(content="go")],
        agent_messages=[ChatMessageUser(content="go")],
        history=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content(model="mockllm/model", content="done"),
    )
