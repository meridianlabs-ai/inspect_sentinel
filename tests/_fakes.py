from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallError, ToolCallView, ToolInfo
from inspect_ai.util import Store

from inspect_sentinel._context import Context, EvalContext
from inspect_sentinel._host import HostContext, HumanAnswer, enter_layer, running_step
from inspect_sentinel._report import Decision, Failed, Report, Reported
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step


class Asked(NamedTuple):
    step: Step
    choices: tuple[str, ...]


class FakeHost:
    def __init__(self, *answers: HumanAnswer, completion: str = "ok") -> None:
        self.answers = list(answers)
        self.asked: list[Asked] = []
        self.completion = completion
        self.models: list[tuple[str | None, str | None]] = []

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        self.asked.append(Asked(step, tuple(choices)))
        return self.answers.pop(0)

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        self.models.append((model, role))
        return ModelOutput.from_content(model="fake", content=self.completion)


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
    path: str = "",
    recorder: ListRecorder | None = None,
    host: FakeHost | None = None,
    store: Store | None = None,
) -> HostContext:
    return HostContext(
        context=Context(path=path, host=host or FakeHost(), eval=eval_context()),
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
    built = host_context(path, recorder, host, store)
    with running_step(built.recorder, built.store):
        enter_layer(path, factory)
        yield built.context


def before_step() -> BeforeToolCall:
    return before_tool_call("bash", cmd="ls")


def after_step() -> AfterToolCall:
    return after_tool_call("bash", "out", cmd="ls")


def before_tool_call(function: str, /, **arguments: Any) -> BeforeToolCall:
    return BeforeToolCall(
        conversation="conversation",
        message="",
        call=ToolCall(id="call_1", function=function, arguments=arguments),
        view=ToolCallView(),
        input=[],
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
        history=before.history,
    )
