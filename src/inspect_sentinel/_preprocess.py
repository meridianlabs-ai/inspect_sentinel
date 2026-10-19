# What a monitor shows a model about a step, set once and applied by
# `Host.score` (or by `preprocess_step` before `step_as_str` in a custom
# prompt). Inside a monitor, which is given its context and the step:
#
#   async def check(context: Context, step: BeforeToolCall) -> Observation | None:
#       return await context.host.score(step, DIMENSIONS, preprocessor=StepPreprocessor(
#           input_window=LastTurns(6),                      # the last 6 turns the model was sent
#           omit_reasoning=True,                            # without the model's reasoning
#           transform=withhold_flagged,                     # anything else, on the thinned step
#       ))

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from inspect import signature
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentToolUse,
)
from inspect_ai.scorer import Reference
from inspect_ai.util import StoreModel
from pydantic import Field
from typing_extensions import TypeVar

from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step
from ._unhandled import current_invocation

if TYPE_CHECKING:
    from ._context import Context

# a preprocessor with no transform is for any step
S = TypeVar("S", bound=Step, default=Step)

ExtractReferences: TypeAlias = Callable[[str], list[Reference]]

# a list a window cuts, which keys the marks `NewSinceLastReport` keeps
_Source: TypeAlias = Literal["input", "history", "agent_messages"]


@dataclass(frozen=True)
class AllMessages:
    """Keep all of a conversation."""


@dataclass(frozen=True)
class LastTurns:
    """Keep the last `n` turns of a conversation: everything from the `n`th assistant message from the end. A conversation with no more than `n` assistant messages is kept whole. `n=0` keeps nothing."""

    n: int
    """How many turns to keep."""

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"LastTurns needs n >= 0, got {self.n}.")


@dataclass(frozen=True)
class LastMessages:
    """Keep the last `n` messages of a conversation.

    The cut can fall between a tool call and its result, so the kept messages can start with a result whose call is not shown, and the monitor model sees output without the command that produced it. Prefer `LastTurns`, which keeps each call with its result, unless the count of messages matters more than that.
    """

    n: int
    """How many messages to keep."""

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"LastMessages needs n >= 0, got {self.n}.")


@dataclass(frozen=True)
class NewSinceLastReport:
    """Keep the messages after the last one this monitor reported on, so an incremental monitor does not re-read its conversations.

    The last message shown is recorded by id in the monitor's store (by `context.path`), separately for each conversation, for each list it windows, and for each function of the monitor. It is recorded only when the function returns or decides final and the runner accepts what it returned, so a function that raises or is cancelled is shown those messages again at the next step. A conversation whose recorded message is gone, as after a compaction, is kept whole. Preprocess with it from the monitor or protocol function the runner is running; elsewhere preprocessing raises `RuntimeError`.
    """

    context: Context
    """The monitor's context, whose store holds what was shown."""


Window: TypeAlias = AllMessages | LastTurns | LastMessages | NewSinceLastReport
"""How much of one conversation, `input`, `history` or `agent_messages`, to show."""


@dataclass(frozen=True, kw_only=True)
class StepPreprocessor(Generic[S]):
    """What a monitor shows a model about a step.

    `preprocess_step()` applies it, in this order: the windows to `input`, `history` and `agent_messages`, then the omissions to every message the step holds, then `transform`, which sees the result.
    """

    input_window: Window | None = field(default_factory=AllMessages)
    """How much of `input` to show: all of it by default; None for none."""

    history_window: Window | None = None
    """How much of `history`, every message the agent has sent its model, to show: none by default."""

    agent_messages_window: Window | None = None
    """How much of `agent_messages`, the agent loop's own list, to show: none by default."""

    omit_system: bool = True
    """Leave system messages out."""

    omit_reasoning: bool = False
    """Leave the model's reasoning out."""

    omit_tool_usage: bool = False
    """Leave out a provider's own tool use (such as web search) recorded in a message."""

    transform: Callable[[S], Awaitable[S]] | None = None
    """Any further change, applied last to the thinned step. It returns a step of the same type. The messages it is given are the host's own objects: copy one before changing it."""


async def preprocess_step(
    step: S, preprocessor: StepPreprocessor[S] | None = None
) -> S:
    """`step`, thinned as `preprocessor` says (`StepPreprocessor()` when None).

    Each window cuts its list as it is. A list with no window becomes `[]`; a list the host could not provide stays None. The omissions apply to `input`, `history`, `agent_messages`, a tool call's result and a generate's reply.

    Args:
        step: The step a monitor was given.
        preprocessor: What to show; `StepPreprocessor()` when None.

    Raises:
        RuntimeError: If a window is `NewSinceLastReport` and this is called outside a monitor or protocol function the runner is running.
    """
    p = preprocessor or StepPreprocessor()

    def thin(
        source: _Source,
        messages: list[ChatMessage] | None,
        window: Window | None,
    ) -> list[ChatMessage] | None:
        if messages is None:
            return None
        if window is None:
            return []
        return _omitted(_cut(step, source, messages, window), p)

    thinned = replace(
        step,
        input=thin("input", step.input, p.input_window),
        history=thin("history", step.history, p.history_window),
        agent_messages=thin(
            "agent_messages", step.agent_messages, p.agent_messages_window
        ),
    )
    if isinstance(thinned, AfterToolCall):
        [result] = _omitted([thinned.result], p)
        thinned = replace(thinned, result=result)
    if isinstance(thinned, AfterGenerate):
        output = thinned.output.model_copy(
            update={
                "choices": [
                    choice.model_copy(
                        update={"message": _omitted([choice.message], p)[0]}
                    )
                    for choice in thinned.output.choices
                ]
            }
        )
        thinned = replace(thinned, output=output)
    return await p.transform(thinned) if p.transform else thinned


async def step_as_str(
    step: Step, *, nonce: str | None = None
) -> tuple[str, ExtractReferences]:
    """Render `step` for a monitor's prompt, numbering its messages `[M1]`, `[M2]` and so on, and return the text with a function that turns those cites in a model's answer into references.

    Each part is a block between `[BEGIN ...]` and `[END ...]` lines, left out when it is empty: `INPUT`, `HISTORY` (`step.history`, every message the agent has sent its model), `AGENT MESSAGES` (`step.agent_messages`, the agent loop's own list), then the step itself in a block marked `[BEGIN AGENT STEP - ...]` and `[END AGENT STEP - ...]`. The step block is `TOOL CALL` (the assistant's text and the call) before a tool call, `TOOL CALL AND RESULT` after one, `MODEL REQUEST` before a generate, and `MODEL REPLY` after one. Each of the last two holds the request: its `INPUT` block, then `TOOLS` (the tools offered, as JSON) and a one-line `[TOOL CHOICE] "auto"`, and a reply's adds a `REPLY` block, so a generate has no `INPUT` block before `HISTORY`. A list the host could not provide (None) is named in its block. With a `nonce`, every `[BEGIN ...]` and `[END ...]` line, and the `[TOOL CHOICE]` line, carries it in parentheses, as `[BEGIN INPUT (3f9a1c7e)]`, and so does every message header where the installed Scout takes a nonce. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it. Needs `inspect_scout` installed; it is not a dependency of this package yet.

    Args:
        step: The step to render, usually from `preprocess_step()`.
        nonce: A code the agent cannot guess, such as `secrets.token_hex(4)`, for every block marker and message header; none by default. Scout needs at least 8 characters.
    """
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import MessagesPreprocessor, message_numbering

    # the preprocessor has already decided what is left out
    numbered, extract_refs = message_numbering(
        MessagesPreprocessor(exclude_system=False),
        **({} if nonce is None else scout_nonce(nonce)),  # pyright: ignore[reportArgumentType]
    )

    async def text(messages: list[ChatMessage] | None, unknown: str) -> str:
        if messages is None:
            return unknown
        return await numbered(messages) if messages else ""

    mark = "" if nonce is None else f" ({nonce})"

    def blocks(*parts: tuple[str, str]) -> str:
        return "\n\n".join(
            f"[BEGIN {name}{mark}]\n{body}\n[END {name}{mark}]"
            for name, body in parts
            if body
        )

    # rendered in this order, so messages are numbered in it too
    unknown_input = "The model input for this call is not available."
    # a generate's input is part of its step block
    generate = isinstance(step, BeforeGenerate | AfterGenerate)
    input = "" if generate else await text(step.input, unknown_input)
    history = await text(
        step.history, "The messages the agent has sent are not available."
    )
    agent_messages = await text(
        step.agent_messages, "The agent's own message list is not available."
    )
    if isinstance(step, BeforeGenerate | AfterGenerate):
        # the tools follow the input, so the input stays a prefix the monitor
        # model's prompt cache can reuse
        own = "\n\n".join(
            [
                blocks(
                    ("INPUT", await text(step.input, unknown_input)),
                    ("TOOLS", _tools(step)),
                ),
                # one line, so it needs no end marker
                f"[TOOL CHOICE{mark}] {_tool_choice(step)}",
                *(
                    [blocks(("REPLY", await text(_reply(step), "")))]
                    if isinstance(step, AfterGenerate)
                    else []
                ),
            ]
        )
    else:
        own = await text(_call(step), "")
    return (
        blocks(
            ("INPUT", input),
            ("HISTORY", history),
            ("AGENT MESSAGES", agent_messages),
            (f"AGENT STEP - {_step_block(step)}", own),
        ),
        extract_refs,
    )


def scout_nonce(nonce: str) -> dict[str, str]:
    # Scout 0.5.5 and earlier take no nonce; until this package requires a
    # later Scout, it is passed only to one that takes it. Scout added it to
    # messages_as_str and message_numbering together, so one check serves both
    from inspect_scout import messages_as_str

    return {"nonce": nonce} if "nonce" in signature(messages_as_str).parameters else {}


class _Marks(StoreModel):
    # "<conversation>/<source>/<function>" -> id of the last message shown
    marks: dict[str, str] = Field(default_factory=dict)


def _cut(
    step: Step,
    source: _Source,
    messages: list[ChatMessage],
    window: Window,
) -> list[ChatMessage]:
    match window:
        case AllMessages():
            return messages
        case LastTurns(n):
            starts = [i for i, m in enumerate(messages) if m.role == "assistant"]
            return (
                []
                if n == 0
                else messages
                if len(starts) <= n
                else messages[starts[-n] :]
            )
        case LastMessages(n):
            return messages[len(messages) - min(n, len(messages)) :]
        case NewSinceLastReport(context):
            return _after_mark(context, step, source, messages)


def _after_mark(
    context: Context,
    step: Step,
    source: _Source,
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    node = current_invocation()
    if node is None:
        raise RuntimeError(
            "NewSinceLastReport works only in a monitor or protocol function the runner is running, since it records what the function reported on."
        )
    store = context.store_as(_Marks)
    key = f"{step.conversation}/{source}/{node.function}"
    seen = store.marks.get(key)
    ids = [m.id for m in messages]
    last = next((m.id for m in reversed(messages) if m.id is not None), None)
    if last is not None:
        node.on_success.append(
            lambda: setattr(store, "marks", {**store.marks, key: last})
        )
    # from the last copy: `history` holds each edited version of a message
    # under its one id
    return (
        messages[len(ids) - ids[::-1].index(seen) :]
        if seen is not None and seen in ids
        else messages
    )


def _omitted(messages: list[ChatMessage], p: StepPreprocessor[S]) -> list[ChatMessage]:
    removed = tuple(
        kind
        for omit, kind in (
            (p.omit_reasoning, ContentReasoning),
            (p.omit_tool_usage, ContentToolUse),
        )
        if omit
    )
    return [
        message
        if isinstance(message.content, str) or not removed
        else message.model_copy(
            update={
                "content": [c for c in message.content if not isinstance(c, removed)]
            }
        )
        for message in messages
        if not (p.omit_system and message.role == "system")
    ]


def _step_block(step: Step) -> str:
    # named for what it holds, so the model can tell whether a call has run
    if isinstance(step, BeforeToolCall):
        return "TOOL CALL"
    if isinstance(step, AfterToolCall):
        return "TOOL CALL AND RESULT"
    if isinstance(step, BeforeGenerate):
        return "MODEL REQUEST"
    return "MODEL REPLY"


def _tools(step: BeforeGenerate | AfterGenerate) -> str:
    return json.dumps([t.model_dump(exclude_none=True) for t in step.tools], indent=2)


def _tool_choice(step: BeforeGenerate | AfterGenerate) -> str:
    choice = step.tool_choice
    return json.dumps(choice if isinstance(choice, str) else {"name": choice.name})


def _call(step: BeforeToolCall | AfterToolCall) -> list[ChatMessage]:
    # the message holds only this call, under the id of the one that made it,
    # so a cite of it resolves to that message
    call = ChatMessageAssistant(
        id=step.message_id, content=step.message, tool_calls=[step.call]
    )
    return [call, step.result] if isinstance(step, AfterToolCall) else [call]


def _reply(step: AfterGenerate) -> list[ChatMessage]:
    # a provider can return no choices: show that as an empty reply
    return [
        step.output.message if step.output.choices else ChatMessageAssistant(content="")
    ]
