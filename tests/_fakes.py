from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple

from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallView, ToolInfo
from inspect_ai.util import Store

from inspect_sentinel._context import Context
from inspect_sentinel._host import HostContext, HumanAnswer, RunState
from inspect_sentinel._report import Decision, Failed, Report, Reported
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step


class Asked(NamedTuple):
    step: Step
    choices: tuple[str, ...]


class FakeHost:
    def __init__(self, *answers: HumanAnswer) -> None:
        self.answers = list(answers)
        self.asked: list[Asked] = []

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        self.asked.append(Asked(step, tuple(choices)))
        return self.answers.pop(0)

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | Model | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        return ModelOutput.from_content(model="fake", content="ok")


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


def host_context(
    path: str = "",
    recorder: ListRecorder | None = None,
    host: FakeHost | None = None,
    store: Store | None = None,
) -> HostContext:
    return HostContext(
        context=Context(
            task="t",
            task_description=None,
            sample_id=1,
            epoch=1,
            sample_description=None,
            sample_input="prompt",
            metadata={},
            path=path,
            _store=store or Store(),
            host=host or FakeHost(),
        ),
        recorder=recorder or ListRecorder(),
    )


def layer_context(
    path: str = "",
    recorder: ListRecorder | None = None,
    host: FakeHost | None = None,
    store: Store | None = None,
    factory: str = "",
) -> Context:
    built = host_context(path, recorder, host, store)
    return replace(built.context, _run=RunState(built.recorder, factory))


def before_step() -> BeforeToolCall:
    return BeforeToolCall(
        conversation="c",
        message="",
        call=ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}),
        view=ToolCallView(),
        input=[ChatMessageUser(content="go")],
        history=[ChatMessageUser(content="go")],
    )


def after_step() -> AfterToolCall:
    return AfterToolCall(
        conversation="c",
        message="",
        call=ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}),
        result=ChatMessageTool(content="out", tool_call_id="c1"),
        output="out",
        view=ToolCallView(),
        input=[ChatMessageUser(content="go")],
        history=[ChatMessageUser(content="go")],
    )
