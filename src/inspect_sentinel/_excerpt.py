from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, TypeVar

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallView
from inspect_ai.util import StoreModel
from pydantic import Field

from ._context import Context
from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step


@dataclass(frozen=True)
class BeforeToolCallExcerpt:
    """The parts of a `BeforeToolCall` step a monitor shows a model, cut down by the windowing functions."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage] | None
    """What the model was sent for the generate that produced the call, or None when the step has none (see `BeforeToolCall.input`)."""

    history: list[ChatMessage]
    """The scaffold's conversation before the assistant message that holds the call."""

    message: str
    """Assistant text alongside the call."""

    call: ToolCall
    """The call as it will execute."""

    view: ToolCallView
    """Rendering of the tool call and its context."""


@dataclass(frozen=True)
class AfterToolCallExcerpt:
    """The parts of an `AfterToolCall` step a monitor shows a model, cut down by the windowing functions."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage] | None
    """What the model was sent for the generate that produced the call, or None when the step has none (see `AfterToolCall.input`)."""

    history: list[ChatMessage]
    """The scaffold's conversation before the assistant message that holds the call."""

    message: str
    """Assistant text alongside the call."""

    call: ToolCall
    """The call as executed."""

    result: ChatMessageTool
    """What the model will see, after truncation."""

    view: ToolCallView
    """Rendering of the tool call and its context."""


@dataclass(frozen=True)
class BeforeGenerateExcerpt:
    """The parts of a `BeforeGenerate` step a monitor shows a model, cut down by the windowing functions."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage]
    """What the model is sent, up to and including its last reply."""

    history: list[ChatMessage]
    """The conversation as the caller passed it, up to and including the model's last reply."""

    new: list[ChatMessage]
    """The messages the model is about to answer: those after its last reply in the request, such as tool results or a user message. Empty when the request ends with an assistant message."""


@dataclass(frozen=True)
class AfterGenerateExcerpt:
    """The parts of an `AfterGenerate` step a monitor shows a model, cut down by the windowing functions."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: list[ChatMessage]
    """What the model was sent."""

    history: list[ChatMessage]
    """The conversation as the caller passed it."""

    reply: ChatMessageAssistant
    """What the model returned, empty when the output has no choices."""


Excerpt: TypeAlias = (
    BeforeToolCallExcerpt
    | AfterToolCallExcerpt
    | BeforeGenerateExcerpt
    | AfterGenerateExcerpt
)
"""The union of the excerpt types, one per stage."""

E = TypeVar(
    "E",
    BeforeToolCallExcerpt,
    AfterToolCallExcerpt,
    BeforeGenerateExcerpt,
    AfterGenerateExcerpt,
)

MessagesAsStr: TypeAlias = Callable[[list[ChatMessage]], Awaitable[str]]


def excerpt(step: Step) -> Excerpt:
    """The parts of `step` a monitor shows a model, all of them kept.

    Each conversation (`input` and `history`) is cut before the step's own part so that part is not shown twice: before the assistant message that holds the call at a tool stage, and after the model's last reply before a generate. Narrow the result with the windowing functions, such as `last_turns()`, and render it with `step_as_str()`.

    Args:
        step: The step to excerpt.
    """
    if isinstance(step, BeforeToolCall):
        return BeforeToolCallExcerpt(
            conversation=step.conversation,
            input=step.input,
            history=_before_call(step.history, step.call),
            message=step.message,
            call=step.call,
            view=step.view,
        )
    if isinstance(step, AfterToolCall):
        return AfterToolCallExcerpt(
            conversation=step.conversation,
            input=step.input,
            history=_before_call(step.history, step.call),
            message=step.message,
            call=step.call,
            result=step.result,
            view=step.view,
        )
    if isinstance(step, BeforeGenerate):
        replied = _last_reply(step.input) + 1
        return BeforeGenerateExcerpt(
            conversation=step.conversation,
            input=step.input[:replied],
            history=step.history[: _last_reply(step.history) + 1],
            new=step.input[replied:],
        )
    assert isinstance(step, AfterGenerate)
    return AfterGenerateExcerpt(
        conversation=step.conversation,
        input=step.input,
        history=step.history,
        # a provider can return no choices: show that as an empty reply
        reply=step.output.message
        if step.output.choices
        else ChatMessageAssistant(content=""),
    )


def drop_input(excerpt: E) -> E:
    """`excerpt` without the model's input: an empty `input`, or None where the step had none.

    Args:
        excerpt: The excerpt to narrow.
    """
    return replace(excerpt, input=None if excerpt.input is None else [])


def drop_history(excerpt: E) -> E:
    """`excerpt` with an empty `history`.

    Args:
        excerpt: The excerpt to narrow.
    """
    return replace(excerpt, history=[])


def last_turns(excerpt: E, n: int) -> E:
    """`excerpt` with `input` and `history` each cut to their last `n` turns.

    A turn starts at an assistant message and runs to the next one, so a call is never separated from its result. Earlier messages, including the system message and the task, are left out unless a conversation has no more than `n` turns, in which case all of it is kept.

    Args:
        excerpt: The excerpt to narrow.
        n: How many turns to keep.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError(f"last_turns() needs n >= 0, got {n}.")

    def cut(messages: list[ChatMessage]) -> list[ChatMessage]:
        starts = [i for i, m in enumerate(messages) if m.role == "assistant"]
        return (
            [] if n == 0 else messages if len(starts) <= n else messages[starts[-n] :]
        )

    return _window(excerpt, cut)


def last_messages(excerpt: E, n: int) -> E:
    """`excerpt` with `input` and `history` each cut to their last `n` messages, fewer when the cut would start on a tool result, so a result is never shown without its call.

    Args:
        excerpt: The excerpt to narrow.
        n: How many messages to keep at most.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError(f"last_messages() needs n >= 0, got {n}.")

    def cut(messages: list[ChatMessage]) -> list[ChatMessage]:
        start = len(messages) - min(n, len(messages))
        return messages[
            next(
                (i for i in range(start, len(messages)) if messages[i].role != "tool"),
                len(messages),
            ) :
        ]

    return _window(excerpt, cut)


class _Marks(StoreModel):
    # (conversation, source) -> id of the last message shown
    marks: dict[str, str] = Field(default_factory=dict)


def new_since_last_report(context: Context, excerpt: E) -> E:
    """`excerpt` with `input` and `history` each cut to the messages after the last one this monitor was shown, so an incremental monitor never re-reads what it already judged.

    The last message of each is recorded in the monitor's store (by `context.path`), separately for each conversation. A conversation is kept whole the first time, and when its recorded message is gone, as after a compaction. Call it once per step, while the monitor runs.

    Args:
        context: The monitor's context.
        excerpt: The excerpt to narrow.
    """
    store = context.store_as(_Marks)
    marks = dict(store.marks)

    def cut(
        source: Literal["input", "history"],
    ) -> Callable[[list[ChatMessage]], list[ChatMessage]]:
        key = f"{excerpt.conversation}/{source}"

        def after_mark(messages: list[ChatMessage]) -> list[ChatMessage]:
            ids = [m.id for m in messages]
            seen = marks.get(key)
            if messages and messages[-1].id is not None:
                marks[key] = messages[-1].id
            return messages[ids.index(seen) + 1 :] if seen in ids else messages

        return after_mark

    narrowed = replace(
        excerpt,
        input=None if excerpt.input is None else cut("input")(excerpt.input),
        history=cut("history")(excerpt.history),
    )
    store.marks = marks
    return narrowed


async def step_as_str(
    excerpt: Excerpt, *, messages_as_str: MessagesAsStr | None = None
) -> str:
    """Render `excerpt` for a monitor's prompt.

    Each part is a block between `[BEGIN ...]` and `[END ...]` lines: `HISTORY` and `INPUT` when they have messages, then `AGENT STEP`, the step's own part: the assistant's message and the call before a tool call, those and the result after one, the messages the model is about to answer before a generate, and its reply after one. A tool stage whose model input is unknown says so in place of the `INPUT` block. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it.

    Args:
        excerpt: The excerpt to render.
        messages_as_str: Renders a list of messages; pass the first function of Scout's `message_numbering()` to number them so a model's answer can cite them. Scout's `messages_as_str` by default.
    """
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import messages_as_str as scout_messages_as_str

    render = messages_as_str or scout_messages_as_str
    input = (
        "The model input for this call is not available."
        if excerpt.input is None
        else await render(excerpt.input)
    )
    blocks = [
        ("HISTORY", await render(excerpt.history)),
        ("INPUT", input),
        ("AGENT STEP", await render(_step_part(excerpt))),
    ]
    return "\n\n".join(
        f"[BEGIN {name}]\n{text}\n[END {name}]" for name, text in blocks if text
    )


def _step_part(excerpt: Excerpt) -> list[ChatMessage]:
    if isinstance(excerpt, BeforeToolCallExcerpt | AfterToolCallExcerpt):
        call = ChatMessageAssistant(content=excerpt.message, tool_calls=[excerpt.call])
        return (
            [call, excerpt.result]
            if isinstance(excerpt, AfterToolCallExcerpt)
            else [call]
        )
    if isinstance(excerpt, BeforeGenerateExcerpt):
        return excerpt.new
    return [excerpt.reply]


def _window(excerpt: E, cut: Callable[[list[ChatMessage]], list[ChatMessage]]) -> E:
    return replace(
        excerpt,
        input=None if excerpt.input is None else cut(excerpt.input),
        history=cut(excerpt.history),
    )


def _last_reply(messages: Sequence[ChatMessage]) -> int:
    return max((i for i, m in enumerate(messages) if m.role == "assistant"), default=-1)


def _before_call(history: list[ChatMessage], call: ToolCall) -> list[ChatMessage]:
    index = next(
        (i for i in range(len(history) - 1, -1, -1) if _holds(history[i], call)),
        None,
    )
    return history if index is None else history[:index]


def _holds(message: ChatMessage, call: ToolCall) -> bool:
    return isinstance(message, ChatMessageAssistant) and any(
        c.id == call.id for c in message.tool_calls or []
    )
