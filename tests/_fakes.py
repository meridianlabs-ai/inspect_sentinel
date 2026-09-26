from typing import NamedTuple

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolInfo
from inspect_ai.util import Store

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
        self.cancellations: list[tuple[str, str]] = []

    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None:
        self.records.append(Recorded(context, step, reported))

    def cancelled(self, context: Context, step: Step, name: str) -> None:
        self.cancellations.append((context.path, name))


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
