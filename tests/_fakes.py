from collections.abc import Sequence
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

from inspect_sentinel._context import HumanAnswer, RunnerContext
from inspect_sentinel._report import Decision, Report, Reported
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
    context: RunnerContext
    step: Step
    reported: Reported[Report]


class ListRecorder:
    def __init__(self) -> None:
        self.records: list[Recorded] = []
        self.cancellations: list[tuple[str, str]] = []
        self.cancelled_contexts: list[RunnerContext] = []
        self.bypassed_layers: list[tuple[str, str]] = []
        self.bypassed_contexts: list[RunnerContext] = []
        self.supersessions: list[Recorded] = []

    def record(
        self, context: RunnerContext, step: Step, reported: Reported[Report]
    ) -> None:
        self.records.append(Recorded(context, step, reported))

    def cancelled(self, context: RunnerContext, step: Step, name: str) -> None:
        self.cancellations.append((context.path, name))
        self.cancelled_contexts.append(context)

    def bypassed(self, context: RunnerContext, step: Step, name: str) -> None:
        self.bypassed_layers.append((context.path, name))
        self.bypassed_contexts.append(context)

    def superseded(
        self, context: RunnerContext, step: Step, reported: Reported[Decision]
    ) -> None:
        self.supersessions.append(Recorded(context, step, reported))


def runner_context(
    path: str = "", recorder: ListRecorder | None = None, host: FakeHost | None = None
) -> RunnerContext:
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
        host=host or FakeHost(),
        recorder=recorder or ListRecorder(),
    )


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
