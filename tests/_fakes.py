from typing import NamedTuple

from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallView, ToolInfo
from inspect_ai.util import Store

from inspect_sentinel._context import Context, RunnerContext
from inspect_sentinel._report import Report, Reported
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step


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
        self.cancellations: list[tuple[str, str]] = []
        self.bypassed_layers: list[tuple[str, str]] = []

    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None:
        self.records.append(Recorded(context, step, reported))

    def cancelled(self, context: Context, step: Step, name: str) -> None:
        self.cancellations.append((context.path, name))

    def bypassed(self, context: Context, step: Step, name: str) -> None:
        self.bypassed_layers.append((context.path, name))


def runner_context(
    path: str = "", recorder: ListRecorder | None = None
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
        host=FakeHost(),
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
