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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentToolUse,
)
from inspect_ai.scorer import Reference
from inspect_ai.tool import ToolChoice, ToolInfo
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

    The last message shown is recorded by id and position in the monitor's store (by `context.path`), separately for each conversation, for each list it windows, and for each function of the monitor. It is recorded only when the function returns or decides final and the runner accepts what it returned, so a function that raises or is cancelled is shown those messages again at the next step. `history`, which only grows, is resumed where it was without reading what was shown before; `input` and `agent_messages` can be rewritten, as by a compaction, so for them the whole list is read to find the message shown last, a cost linear in the list at each step. An edited version of a message, which `history` holds again under the same id, is new. A conversation whose recorded message is gone, as after a compaction, is kept whole. Preprocess with it from the monitor or protocol function the runner is running; elsewhere preprocessing raises `RuntimeError`.
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
    """How much of `input` to show: all of it by default; None for none.

    Before a generate, `input` is the request being judged, so a window that leaves none of it (None, `LastTurns(0)`, `LastMessages(0)`, or `NewSinceLastReport` when nothing is new) shows the model only the tools offered. At a tool stage the call is judged against only the tools of the request that produced it.
    """

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
        messages: Sequence[ChatMessage] | None,
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

    Everything is in one block, marked `[BEGIN AGENT STEP - ...]` and `[END AGENT STEP - ...]` and named for the stage: `TOOL CALL REQUEST` before a tool call, `TOOL CALL RESULT` after one, `MODEL REQUEST` before a generate, and `MODEL REPLY` after one. Inside it are blocks between `[BEGIN ...]` and `[END ...]` lines, each left out when it is empty. At every stage: `HISTORY` (`step.history`, every message the agent has sent its model), `AGENT MESSAGES` (`step.agent_messages`, the agent loop's own list), then the model request, as `INPUT`, `TOOLS` (the tools offered, as JSON) and a one-line `[TOOL CHOICE] "auto"`; at a tool stage, the request that produced the call. Then after a generate, `REPLY`; at a tool stage, `CALL` (the assistant's text and the call) and, after the call, `RESULT`. A list the host could not provide (None) is named in its block; at a tool stage whose input is not known, the tools and tool choice are left out. With a `nonce`, every `[BEGIN ...]` and `[END ...]` line, and the `[TOOL CHOICE]` line, carries it in parentheses, as `[BEGIN INPUT (3f9a1c7e)]`, and so does every message header. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it.

    Args:
        step: The step to render, usually from `preprocess_step()`.
        nonce: A code the agent cannot guess, such as `secrets.token_hex(4)`, for every block marker and message header; none by default. Scout needs at least 8 characters.
    """
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import MessagesPreprocessor, message_numbering

    # the preprocessor has already decided what is left out
    numbered, extract_refs = message_numbering(
        MessagesPreprocessor(exclude_system=False),
        nonce=nonce,
    )

    async def text(messages: Sequence[ChatMessage] | None, unknown: str) -> str:
        if messages is None:
            return unknown
        if not messages:
            return ""
        # Scout renders a list; a view becomes one only here, where every
        # message is rendered anyway
        return await numbered(
            messages if isinstance(messages, list) else list(messages)
        )

    mark = "" if nonce is None else f" ({nonce})"

    def blocks(*parts: tuple[str, str]) -> str:
        return "\n\n".join(
            f"[BEGIN {name}{mark}]\n{body}\n[END {name}{mark}]"
            for name, body in parts
            if body
        )

    # a request whose input is not known shows none of its parts, even when a
    # transform kept them
    tools, choice = (
        (step.tools, step.tool_choice) if step.input is not None else (None, None)
    )
    # rendered in this order, so messages are numbered in it too
    parts = [
        blocks(
            (
                "HISTORY",
                await text(
                    step.history, "The messages the agent has sent are not available."
                ),
            ),
            (
                "AGENT MESSAGES",
                await text(
                    step.agent_messages,
                    "The agent's own message list is not available.",
                ),
            ),
        ),
        # the tools follow the input, so the input stays a prefix the monitor
        # model's prompt cache can reuse
        blocks(
            (
                "INPUT",
                await text(
                    step.input, "The model input for this call is not available."
                ),
            ),
            ("TOOLS", _tools(tools)),
        ),
        # one line, so it needs no end marker
        "" if choice is None else f"[TOOL CHOICE{mark}] {_tool_choice(choice)}",
    ]
    if isinstance(step, AfterGenerate):
        parts.append(blocks(("REPLY", await text(_reply(step), ""))))
    elif isinstance(step, BeforeToolCall | AfterToolCall):
        parts.append(blocks(("CALL", await text([_call(step)], ""))))
        if isinstance(step, AfterToolCall):
            parts.append(blocks(("RESULT", await text([step.result], ""))))
    return (
        blocks(
            (
                f"AGENT STEP - {_step_block(step)}",
                "\n\n".join(part for part in parts if part),
            )
        ),
        extract_refs,
    )


def _mark_key(context: Context, step: Step, source: _Source, function: str) -> str:
    # one store entry per conversation, list and function, so a report reads
    # and writes only its own: (id of the last message shown, how many copies
    # of that id the list held then, its position + 1)
    return f"NewSinceLastReport:{context.path}:{step.conversation}/{source}/{function}"


def _cut(
    step: Step,
    source: _Source,
    messages: Sequence[ChatMessage],
    window: Window,
) -> Sequence[ChatMessage]:
    # a window reads the list it is given as it is and slices off what it
    # keeps; `AllMessages` keeps it whole, and the others read from where
    # what they keep starts, except for `NewSinceLastReport` over `input` or
    # `agent_messages`, which a scaffold can rewrite, so it scans them
    match window:
        case AllMessages():
            return messages
        case LastTurns(n):
            return messages[_turns_start(messages, n) :]
        case LastMessages(n):
            return messages[len(messages) - min(n, len(messages)) :]
        case NewSinceLastReport(context):
            return _after_mark(context, step, source, messages)


def _turns_start(messages: Sequence[ChatMessage], n: int) -> int:
    # from the end, so a short window reads little more than the turns it
    # keeps; a list of no more than n turns is kept whole
    if n == 0:
        return len(messages)
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            if n == 0:
                return start
            n -= 1
            start = i
    return 0


def _after_mark(
    context: Context,
    step: Step,
    source: _Source,
    messages: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    node = current_invocation()
    if node is None:
        raise RuntimeError(
            "NewSinceLastReport works only in a monitor or protocol function the runner is running, since it records what the function reported on."
        )
    # `_host` imports this module, through `Host.score`
    from ._host import active_step

    store = active_step("NewSinceLastReport").store
    key = _mark_key(context, step, source, node.function)
    seen = cast("tuple[str, int, int] | None", store.get(key))
    start, grew = _resumed(source, messages, seen)
    last = _last_with_id(messages)
    if last is not None:
        index, id = last
        if grew and seen is not None:
            # count the copies in what is new, missing only an earlier copy of
            # a new last id, which `history` resumes past by position anyway
            copies = (seen[1] if id == seen[0] else 0) + sum(
                1 for i in range(start, index + 1) if messages[i].id == id
            )
        else:
            copies = sum(1 for m in messages if m.id == id)
        mark = (id, copies, index + 1)
        node.on_success.append(lambda: store.set(key, mark))
    return messages[start:]


def _resumed(
    source: _Source,
    messages: Sequence[ChatMessage],
    seen: tuple[str, int, int] | None,
) -> tuple[int, bool]:
    # where to show from, and whether the list only grew since
    if seen is None:
        return 0, False
    id, copies, length = seen
    # only `history` is append-only, so only there does a list that holds the
    # message shown last where it was hold everything shown before it
    if (
        source == "history"
        and 0 < length <= len(messages)
        and messages[length - 1].id == id
    ):
        return length, True
    # from the copy shown last: `history` holds each edited version of a
    # message under its one id, so a later version is new; a list that no
    # longer holds that copy, as after a compaction, is shown whole
    found = [i for i, m in enumerate(messages) if m.id == id]
    return (found[copies - 1] + 1 if len(found) >= copies else 0), False


def _last_with_id(messages: Sequence[ChatMessage]) -> tuple[int, str] | None:
    for i in range(len(messages) - 1, -1, -1):
        id = messages[i].id
        if id is not None:
            return i, id
    return None


def _omitted(
    messages: Sequence[ChatMessage], p: StepPreprocessor[S]
) -> list[ChatMessage]:
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
        return "TOOL CALL REQUEST"
    if isinstance(step, AfterToolCall):
        return "TOOL CALL RESULT"
    if isinstance(step, BeforeGenerate):
        return "MODEL REQUEST"
    return "MODEL REPLY"


def _tools(tools: Sequence[ToolInfo] | None) -> str:
    # none offered, or not known, leaves the block out
    return (
        json.dumps([t.model_dump(exclude_none=True) for t in tools], indent=2)
        if tools
        else ""
    )


def _tool_choice(choice: ToolChoice) -> str:
    return json.dumps(choice if isinstance(choice, str) else {"name": choice.name})


def _call(step: BeforeToolCall | AfterToolCall) -> ChatMessage:
    # the message holds only this call, under the id of the one that made it,
    # so a cite of it resolves to that message
    return ChatMessageAssistant(
        id=step.message_id, content=step.message, tool_calls=[step.call]
    )


def _reply(step: AfterGenerate) -> list[ChatMessage]:
    # a provider can return no choices: show that as an empty reply
    return [
        step.output.message if step.output.choices else ChatMessageAssistant(content="")
    ]
