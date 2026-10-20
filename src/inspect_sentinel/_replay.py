# Temporary: runs a sentinel over the samples of an eval log, until the replay
# of design/sentinel-development.md (workstream 10) replaces it. Not exported;
# delete it then.

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

import anyio
from inspect_ai.event import (
    ApprovalEvent,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    ToolEvent,
)
from inspect_ai.log import (
    EvalLog,
    EvalSample,
    read_eval_log_async,
    read_eval_log_sample_async,
    read_eval_log_sample_summaries,
    read_eval_log_sample_summaries_async,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    Content,
    GenerateConfig,
    ModelConfig,
    ModelOutput,
    get_model,
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
from .sidecar._host import ListRecorder, Record, Stage, stage
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
) -> AsyncIterator[ReplayedStep]:
    """Run a sentinel over the steps of samples in an eval log, as if it had watched them live, yielding each step as it is judged.

    Temporary, until the replay `design/sentinel-development.md` describes replaces it.

    Steps are rebuilt from the sample's events, as inspect_ai builds them live. Each agent turn gives a `BeforeGenerate` from its first `ModelEvent`, so attempts a provider error made inspect_ai retry are one step, and an `AfterGenerate` from the attempt that returned output. Each `ToolEvent` gives a `BeforeToolCall` of the call as the sentinel would have seen it, after an approver's modification, unless the call never reached the sentinel (it did not parse, named no tool the model was offered, or an approver rejected it), and an `AfterToolCall` if the call ran to a result. Only model and tool calls made while the solvers ran count, none inside a sentinel's own span, and no model call inside a tool's. Only the stages the sentinel watches are run, in the order of the events they come from, with one store per sample.

    What the log does not hold is missing or approximate. `agent_messages`, `history` and `human_decisions` are None. `AfterToolCall.output` is the result as the model saw it, after truncation, not what the tool returned. A model call that inspect_ai keeps out of the sentinel without a span, such as a compaction summary or a model-based approver, is taken for an agent turn. `context.eval.metadata` holds the eval's metadata as well as the task's, since the log records them merged. A `BeforeGenerate` that made no model call, because the sentinel ended the sample there or a limit stopped the call, has no `ModelEvent` and is not replayed. A tool call's two steps come out together where its `ToolEvent` is, so its `AfterToolCall` comes before any steps of a sub-agent the call ran, and parallel calls' steps do not interleave. A parallel call cancelled because a sibling raised is taken to have reached the sentinel.

    A decision changes nothing: the replay goes on with what the agent did. Monitors' model calls are made for real, with the model of their role (from `model_roles`, then the eval's model roles), or else the eval's own model. A protocol that asks a person raises, since nobody can be asked.

    Args:
        log: The eval log file.
        sentinel: The sentinel to run. Defaults to the one the eval ran with.
        samples: The `(id, epoch)` of each sample to replay. Defaults to every sample.
        model_roles: The model each role generates with, e.g. `{"monitor": "anthropic/claude-haiku-4-5"}`, over the eval's own model roles.

    Raises:
        ValueError: If `sentinel` is None and the eval ran without one.
    """
    header = await read_eval_log_async(log, header_only=True)
    if sentinel is None:
        if header.eval.config.sentinel is None:
            raise ValueError(f"{log} ran without a sentinel; pass one to replay.")
        sentinel = sentinel_from_config(header.eval.config.sentinel)
    root = resolve_sentinel(sentinel)
    watched = watched_stages(root)
    roles = {
        **{
            role: config if isinstance(config, ModelConfig) else config[0]
            for role, config in (header.eval.model_roles or {}).items()
        },
        **{
            role: ModelConfig(model=model)
            for role, model in (model_roles or {}).items()
        },
    }
    own = ModelConfig(
        model=header.eval.model,
        config=header.eval.model_generate_config,
        base_url=header.eval.model_base_url,
        args=header.eval.model_args,
    )
    selected = (
        samples
        if samples is not None
        else [
            (summary.id, summary.epoch)
            for summary in await read_eval_log_sample_summaries_async(log)
        ]
    )
    for sample_id, epoch in selected:
        sample = await read_eval_log_sample_async(
            log, sample_id, epoch, resolve_attachments=True
        )
        host = _ReplayHost(roles, own, _eval_context(header, sample))
        store = Store()
        for step_id, step in _steps(sample):
            if type(step) not in watched:
                continue
            recorder = ListRecorder()
            decision = await run_sentinel(
                root, HostContext(host=host, recorder=recorder, store=store), step
            )
            yield ReplayedStep(
                sample.id,
                sample.epoch,
                stage(step),
                step_id,
                decision,
                recorder.records,
            )


class _ReplayHost:
    def __init__(
        self, roles: Mapping[str, ModelConfig], own: ModelConfig, eval: EvalContext
    ) -> None:
        self._roles = roles
        self._own = own
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
        # as inspect_ai's host: a model alone is that model; otherwise the
        # role's model, then the model given, then the eval's own
        found = (
            ModelConfig(model=model)
            if model is not None and role is None
            else self._roles.get(
                role or "monitor",
                ModelConfig(model=model) if model is not None else self._own,
            )
        )
        resolved = get_model(
            found.model, config=found.config, base_url=found.base_url, **found.args
        )
        return await resolved.generate(
            input,
            tools=tools or [],
            tool_choice=tool_choice,
            config=config or GenerateConfig(),
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
        sample_description=sample.description,
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

    # an approver decides before the sentinel: a call it rejected or ended the
    # sample at never reached the sentinel, and one it modified reached it
    # modified
    approvals = [e for e in sample.events if isinstance(e, ApprovalEvent)]
    refused = {e.call.id for e in approvals if e.decision in ("reject", "terminate")}
    modified = {e.call.id: e.modified for e in approvals if e.modified is not None}
    # the calls the sentinel ended the sample at before they ran
    terminated = {
        e.step_id
        for e in sample.events
        if isinstance(e, SentinelEvent)
        and e.path == ""
        and e.stage == "tool_call"
        and e.action == "terminate"
        and e.status == "reported"
    }
    # the model call and assistant message that proposed each tool call
    producers: dict[str, _Producer] = {}
    # each conversation's last turn whose attempt failed, by its last input
    failed: dict[str, str] = {}
    inputs = Counter[str]()
    for event in sample.events:
        if not isinstance(event, ModelEvent | ToolEvent):
            continue
        spanned = ancestry(event.span_id)
        kinds = [kind(span) for span in spanned]
        if "sentinel" in kinds or "solvers" not in kinds:
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
            # a model call inside a tool is not an agent turn
            if next((k for k in kinds if k in ("agent", "tool")), None) == "tool":
                continue
            last = (event.input[-1].id if event.input else None) or ""
            retried = failed.pop(conversation, None) == last
            if event.error is not None:
                failed[conversation] = last
            if not retried:
                inputs[last] += 1
                yield (
                    last if inputs[last] == 1 else f"{last}:{inputs[last]}",
                    BeforeGenerate(
                        model=event.model,
                        conversation=conversation,
                        input=event.input,
                        agent_messages=None,
                        tools=event.tools,
                        tool_choice=event.tool_choice,
                        config=event.config,
                    ),
                )
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
                    modified.get(event.id, producer.call),
                    producer.message.text,
                    producer.event.input,
                    producer.event.tools,
                )
            )
            # a call halted before it ran is cancelled without failing
            reached = (
                proposed.parse_error is None
                and event.id not in refused
                and not (
                    event.error is not None
                    and event.error.type == "cancelled"
                    and not event.failed
                )
                and (tools is None or any(t.name == proposed.function for t in tools))
            )
            if not reached:
                continue
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
            # a call the sentinel rejected or ended the sample at, that was
            # cancelled, or that a time limit cut off, has no result, and neither has one that raised, which
            # failed the call and errored the sample; a call the sentinel ended
            # the sample at once it ran failed too, but it had a result
            if (
                (
                    event.error is not None
                    and event.error.type in ("approval", "parsing", "cancelled")
                )
                or event.id in terminated
                or event.pending
                or (event.failed and sample.error is not None)
            ):
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
    # the event's view is of the call that ran, which an approver or the
    # sentinel may have modified
    return (
        ToolCallView(call=event.view)
        if event.view and call.arguments == event.arguments
        else default_tool_call_viewer(call)
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Replay a sentinel over samples of an eval log, writing one JSON line per step as it is judged.

    Args:
        argv: The arguments, without the program's name. The process's own if not given.
    """
    parser = argparse.ArgumentParser(
        prog="python -m inspect_sentinel._replay",
        description="Run a sentinel over the samples of an eval log, as if it had watched them live. Temporary. The sentinel's monitors and protocols must be registered by an installed package; for ones a task file defines, import it and call replay_eval() instead.",
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
        help="A sample to replay: its id for every epoch of it, or its id and an epoch. May be repeated. Defaults to every sample.",
    )
    parser.add_argument(
        "--model-role",
        action="append",
        default=[],
        metavar="ROLE=MODEL",
        help="The model a role generates with, e.g. monitor=anthropic/claude-haiku-4-5, over the eval's own model roles. May be repeated.",
    )
    arguments = parser.parse_args(argv)
    roles: dict[str, str] = {}
    for role in arguments.model_role:
        name, sep, model = str(role).partition("=")
        if not sep or not name or not model:
            parser.error(f"--model-role takes ROLE=MODEL, not {role!r}.")
        roles[name] = model
    summaries = read_eval_log_sample_summaries(arguments.log)
    missing = [
        given
        for given in arguments.sample
        if not any(_selects(given, s.id, s.epoch) for s in summaries)
    ]
    if missing:
        parser.error(f"{arguments.log} has no sample {', '.join(missing)}.")
    selected = [
        (summary.id, summary.epoch)
        for summary in summaries
        if not arguments.sample
        or any(_selects(given, summary.id, summary.epoch) for given in arguments.sample)
    ]

    async def replay() -> None:
        async for step in replay_eval(
            arguments.log,
            sentinel_from_config(arguments.sentinel) if arguments.sentinel else None,
            samples=selected,
            model_roles=roles,
        ):
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
                ),
                flush=True,
            )

    anyio.run(replay)


def _selects(given: str, sample_id: str | int, epoch: int) -> bool:
    return given in (str(sample_id), f"{sample_id}:{epoch}")


if __name__ == "__main__":
    main()
