# Temporary: runs a sentinel over the samples of an eval log, until the replay
# of design/sentinel-development.md (workstream 10) replaces it. Not exported;
# delete it then.

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

import anyio
from inspect_ai.event import ModelEvent, SpanBeginEvent, ToolEvent
from inspect_ai.log import (
    EvalLog,
    EvalSample,
    read_eval_log,
    read_eval_log_sample,
    read_eval_log_sample_summaries,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    Content,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import (
    ToolCall,
    ToolCallView,
    ToolChoice,
    ToolInfo,
)
from inspect_ai.tool._tool_call import default_tool_call_viewer
from inspect_ai.util import Store

from ._config import sentinel_from_config
from ._context import EvalContext
from ._decorators import watched_stages
from ._host import HostContext, HumanAnswer
from ._report import Decision, Escalations
from ._resolve import resolve_sentinel
from ._runner import run_sentinel
from ._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
    Truncation,
)
from ._types import Sentinels
from .sidecar._host import ListRecorder, Record, SidecarHost, Stage, stage
from .sidecar._inspect_client import record_json


@dataclass(frozen=True)
class ReplayedStep:
    """What the sentinel decided at one step of a replayed sample."""

    sample_id: str | int
    """The sample's id."""

    epoch: int
    """The sample's epoch."""

    stage: Stage
    """The stage of the step."""

    step_id: str
    """The step's id, as the live run gives `SentinelEvent.step_id`."""

    decision: Decision | None
    """The root's decision, or None if it abstained."""

    records: list[Record]
    """Everything the runner recorded at the step, in order."""


async def replay_eval(
    log: str,
    sentinel: Sentinels | None = None,
    *,
    samples: Sequence[tuple[str | int, int]] | None = None,
    model_roles: Mapping[str, str] | None = None,
) -> list[ReplayedStep]:
    """Run a sentinel over the steps of samples in an eval log, as if it had watched them live.

    Temporary, until the replay `design/sentinel-development.md` describes replaces it.

    Steps are rebuilt from the sample's events: each agent turn's `ModelEvent` gives a `BeforeGenerate` and, if it returned output, an `AfterGenerate`; each `ToolEvent` gives a `BeforeToolCall` and, if the call ran, an `AfterToolCall`. Events inside a sentinel's own span are skipped. `agent_messages`, `history` and `human_decisions` are None. Only the stages the sentinel watches are run, in the order the log recorded them, with one store per sample.

    A decision changes nothing: the replay goes on with what the agent did. Monitors' model calls are made for real. A protocol that asks a person raises, since nobody can be asked.

    Args:
        log: The eval log file.
        sentinel: The sentinel to run. Defaults to the one the eval ran with.
        samples: The `(id, epoch)` of each sample to replay. Defaults to every sample.
        model_roles: The model each role generates with, e.g. `{"monitor": "anthropic/claude-haiku-4-5"}`.

    Raises:
        ValueError: If `sentinel` is None and the eval ran without one.
    """
    header = read_eval_log(log, header_only=True)
    if sentinel is None:
        if header.eval.config.sentinel is None:
            raise ValueError(f"{log} ran without a sentinel; pass one to replay.")
        sentinel = sentinel_from_config(header.eval.config.sentinel)
    root = resolve_sentinel(sentinel)
    watched = watched_stages(root)
    models = SidecarHost(model_roles)
    selected = (
        samples
        if samples is not None
        else [
            (summary.id, summary.epoch)
            for summary in read_eval_log_sample_summaries(log)
        ]
    )
    replayed: list[ReplayedStep] = []
    for id, epoch in selected:
        sample = read_eval_log_sample(log, id, epoch)
        host = _ReplayHost(models, _eval_context(header, sample))
        store = Store()
        for step_id, step in _steps(sample):
            if type(step) not in watched:
                continue
            recorder = ListRecorder()
            decision = await run_sentinel(
                root, HostContext(host=host, recorder=recorder, store=store), step
            )
            replayed.append(
                ReplayedStep(
                    sample.id,
                    sample.epoch,
                    stage(step),
                    step_id,
                    decision,
                    recorder.records,
                )
            )
    return replayed


class _ReplayHost:
    def __init__(self, models: SidecarHost, eval: EvalContext) -> None:
        self._models = models
        self._eval = eval

    @property
    def eval(self) -> EvalContext | None:
        return self._eval

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
        return await self._models.generate(
            input,
            model=model,
            role=role,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )

    async def ask_human(
        self, step: Step, escalations: Escalations, choices: Sequence[str]
    ) -> HumanAnswer:
        raise NotImplementedError("A replay cannot ask a person.")


def _eval_context(header: EvalLog, sample: EvalSample) -> EvalContext:
    return EvalContext(
        task=header.eval.task,
        task_description=header.eval.task_description,
        sample_id=sample.id,
        epoch=sample.epoch,
        sample_description=None,
        sample_input=sample.input,
        metadata={**(header.eval.metadata or {}), **sample.metadata},
    )


class _Producer(NamedTuple):
    event: ModelEvent
    message: ChatMessageAssistant
    call: ToolCall


def _steps(sample: EvalSample) -> Iterator[tuple[str, Step]]:
    spans = {e.id: e for e in sample.events if isinstance(e, SpanBeginEvent)}

    def ancestry(span_id: str | None) -> list[SpanBeginEvent]:
        # innermost first
        found: list[SpanBeginEvent] = []
        while span_id is not None and span_id in spans:
            found.append(spans[span_id])
            span_id = spans[span_id].parent_id
        return found

    def kind(span: SpanBeginEvent) -> str:
        return span.type or span.name

    # the model call and assistant message that proposed each tool call
    producers: dict[str, _Producer] = {}
    inputs = Counter[str]()
    for event in sample.events:
        if not isinstance(event, ModelEvent | ToolEvent):
            continue
        spanned = ancestry(event.span_id)
        if any(kind(span) == "sentinel" for span in spanned):
            continue
        agent = next((s.id for s in spanned if kind(s) == "agent"), None)
        conversation = agent if agent is not None else sample.uuid or ""
        if isinstance(event, ModelEvent):
            message = event.output.message if event.output.choices else None
            if message is not None:
                producers.update(
                    (call.id, _Producer(event, message, call))
                    for call in message.tool_calls or []
                )
            # a turn is an agent's when it was made while solving, outside a tool
            scope = next(
                (kind(s) for s in spanned if kind(s) in ("agent", "tool")), None
            )
            if scope == "tool" or not any(kind(s) == "solvers" for s in spanned):
                continue
            last = (event.input[-1].id if event.input else None) or ""
            inputs[last] += 1
            before = BeforeGenerate(
                model=event.model,
                conversation=conversation,
                input=event.input,
                agent_messages=None,
                tools=event.tools,
                tool_choice=event.tool_choice,
                config=event.config,
            )
            yield (last if inputs[last] == 1 else f"{last}:{inputs[last]}"), before
            if message is not None and event.error is None:
                yield (
                    message.id or "",
                    AfterGenerate(
                        model=event.model,
                        conversation=conversation,
                        input=event.input,
                        agent_messages=None,
                        tools=event.tools,
                        tool_choice=event.tool_choice,
                        config=event.config,
                        output=event.output,
                    ),
                )
        else:
            producer = producers.get(event.id)
            ran = ToolCall(
                id=event.id, function=event.function, arguments=event.arguments
            )
            proposed, text, input, tools = (
                (ran, "", None, None)
                if producer is None
                else (
                    producer.call,
                    producer.message.text,
                    producer.event.input,
                    producer.event.tools,
                )
            )
            yield (
                event.id,
                BeforeToolCall(
                    conversation=conversation,
                    message=text,
                    call=proposed,
                    view=_view(proposed, event),
                    input=input,
                    agent_messages=None,
                    tools=tools,
                ),
            )
            # a call rejected or unparsed never ran, so it has no result
            if event.error is not None and event.error.type in ("approval", "parsing"):
                continue
            yield (
                event.id,
                AfterToolCall(
                    conversation=conversation,
                    message=text,
                    call=ran,
                    result=ChatMessageTool(
                        id=event.message_id,
                        content=cast(str | list[Content], event.result),
                        tool_call_id=event.id,
                        function=event.function,
                        error=event.error,
                    ),
                    output=event.result,
                    view=_view(ran, event),
                    input=input,
                    agent_messages=None,
                    tools=tools,
                    truncated=Truncation(*event.truncated)
                    if event.truncated
                    else False,
                ),
            )


def _view(call: ToolCall, event: ToolEvent) -> ToolCallView:
    return (
        ToolCallView(call=event.view) if event.view else default_tool_call_viewer(call)
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Replay a sentinel over samples of an eval log, writing one JSON line per step.

    Args:
        argv: The arguments, without the program's name. The process's own if not given.
    """
    parser = argparse.ArgumentParser(
        prog="python -m inspect_sentinel._replay",
        description="Run a sentinel over the samples of an eval log, as if it had watched them live. Temporary.",
    )
    parser.add_argument("log", help="The eval log file.")
    parser.add_argument(
        "--sentinel",
        help="A YAML or JSON configuration file, or the registered name of a monitor or protocol. Defaults to the sentinel the eval ran with.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="ID[:EPOCH]",
        help="A sample to replay, every epoch of it unless one is given. May be repeated. Defaults to every sample.",
    )
    parser.add_argument(
        "--model-role",
        action="append",
        default=[],
        metavar="ROLE=MODEL",
        help="The model a role generates with, e.g. monitor=anthropic/claude-haiku-4-5. May be repeated.",
    )
    arguments = parser.parse_args(argv)
    roles: dict[str, str] = {}
    for role in arguments.model_role:
        name, sep, model = str(role).partition("=")
        if not sep or not name or not model:
            parser.error(f"--model-role takes ROLE=MODEL, not {role!r}.")
        roles[name] = model
    summaries = read_eval_log_sample_summaries(arguments.log)
    selected = [
        (summary.id, summary.epoch)
        for summary in summaries
        if not arguments.sample
        or any(_selects(given, summary.id, summary.epoch) for given in arguments.sample)
    ]
    if arguments.sample and not selected:
        parser.error(f"{arguments.log} has none of the samples {arguments.sample}.")
    replayed = anyio.run(
        lambda: replay_eval(
            arguments.log,
            sentinel_from_config(arguments.sentinel) if arguments.sentinel else None,
            samples=selected,
            model_roles=roles,
        )
    )
    for step in replayed:
        print(
            json.dumps(
                {
                    "sample_id": step.sample_id,
                    "epoch": step.epoch,
                    "stage": step.stage,
                    "step_id": step.step_id,
                    "decision": step.decision.model_dump(
                        mode="json", exclude_defaults=True
                    )
                    if step.decision is not None
                    else None,
                    "records": [record_json(record) for record in step.records],
                }
            )
        )


def _selects(given: str, id: str | int, epoch: int) -> bool:
    sample, sep, at = given.rpartition(":")
    if not sep or not at.isdigit():
        return given == str(id)
    return sample == str(id) and int(at) == epoch


if __name__ == "__main__":
    main()
