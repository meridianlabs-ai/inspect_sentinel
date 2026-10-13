from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeAlias, TypeVar

from inspect_ai.model import ChatMessage, ChatMessageAssistant
from inspect_ai.tool import ToolCall
from inspect_ai.util import StoreModel
from pydantic import Field

from ._context import Context
from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step

S = TypeVar("S", bound=Step)

MessagesAsStr: TypeAlias = Callable[[list[ChatMessage]], Awaitable[str]]


@dataclass(frozen=True)
class Excerpt(Generic[S]):
    """What a monitor shows a model about a step: the step, and the conversations to show with it, cut down by the windowing functions.

    `step_as_str()` renders the conversations and the step's own part: the assistant's message and the call before a tool call, those and the result after one, the messages after the model's last reply before a generate, and the reply after one. Nothing else of the step is rendered, such as its escalations, tools, config, `view` or untruncated tool output.
    """

    step: S
    """The step, whole. Its `input` and `history` are not what is shown; the excerpt's are."""

    input: list[ChatMessage] | None
    """The part of `step.input` to show, None when the step has none (a tool step no model call produced)."""

    history: list[ChatMessage]
    """The part of `step.history` to show."""


def excerpt(step: S) -> Excerpt[S]:
    """An excerpt of `step` that shows all of its conversations.

    Each conversation is cut before the step's own part, so that part is not shown twice: `history` before the assistant message that holds the call at a tool stage, and `input` and `history` after the model's last reply before a generate. Narrow the result with the windowing functions, such as `last_turns()`, and render it with `step_as_str()`.

    Args:
        step: The step to excerpt.
    """
    if isinstance(step, BeforeToolCall | AfterToolCall):
        return Excerpt(step, step.input, _before_call(step.history, step.call))
    if isinstance(step, BeforeGenerate):
        return Excerpt(
            step,
            step.input[: _last_reply(step.input) + 1],
            step.history[: _last_reply(step.history) + 1],
        )
    return Excerpt(step, step.input, step.history)


def drop_input(excerpt: Excerpt[S]) -> Excerpt[S]:
    """`excerpt` without the model's input: an empty `input`, or None where the step had none.

    Args:
        excerpt: The excerpt to narrow.
    """
    return replace(excerpt, input=None if excerpt.input is None else [])


def drop_history(excerpt: Excerpt[S]) -> Excerpt[S]:
    """`excerpt` with an empty `history`.

    Args:
        excerpt: The excerpt to narrow.
    """
    return replace(excerpt, history=[])


def last_turns(excerpt: Excerpt[S], n: int) -> Excerpt[S]:
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


def last_messages(excerpt: Excerpt[S], n: int) -> Excerpt[S]:
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
    # "<conversation>/<input|history>" -> id of the last message shown
    marks: dict[str, str] = Field(default_factory=dict)


def new_since_last_report(context: Context, excerpt: Excerpt[S]) -> Excerpt[S]:
    """`excerpt` with `input` and `history` each cut to the messages after the last one this monitor was shown in them, so an incremental monitor does not re-read its conversations.

    The last message of each is recorded by id in the monitor's store (by `context.path`), separately for each conversation and shared across stages. Only `input` and `history` are tracked, so the step's own part, such as the last step's call, can be shown again in a later step's conversation. A conversation is kept whole the first time, and when its recorded message is gone: after a compaction, or where a host gives messages new ids on each step. Call it once per step, while the monitor runs.

    Args:
        context: The monitor's context.
        excerpt: The excerpt to narrow.
    """
    store = context.store_as(_Marks)
    marks = dict(store.marks)

    def after_mark(
        source: Literal["input", "history"], messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        key = f"{excerpt.step.conversation}/{source}"
        seen = marks.get(key)
        ids = [m.id for m in messages]
        if messages and messages[-1].id is not None:
            marks[key] = messages[-1].id
        return (
            messages[ids.index(seen) + 1 :]
            if seen is not None and seen in ids
            else messages
        )

    narrowed = replace(
        excerpt,
        input=None if excerpt.input is None else after_mark("input", excerpt.input),
        history=after_mark("history", excerpt.history),
    )
    store.marks = marks
    return narrowed


async def step_as_str(
    excerpt: Excerpt[S], *, messages_as_str: MessagesAsStr | None = None
) -> str:
    """Render `excerpt` for a monitor's prompt.

    Each part is a block between `[BEGIN ...]` and `[END ...]` lines, left out when it has nothing to show: `HISTORY`, `INPUT`, then `AGENT STEP`, the step's own part (see `Excerpt`). When a tool step has no model input, the `INPUT` block says so. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it.

    Args:
        excerpt: The excerpt to render.
        messages_as_str: Renders a list of messages; pass the first function of Scout's `message_numbering()` to number them so a model's answer can cite them. Scout's `messages_as_str` by default.
    """
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import messages_as_str as scout_messages_as_str

    render = messages_as_str or scout_messages_as_str
    blocks = [
        ("HISTORY", await render(excerpt.history)),
        (
            "INPUT",
            "The model input for this call is not available."
            if excerpt.input is None
            else await render(excerpt.input),
        ),
        ("AGENT STEP", await render(_step_part(excerpt.step))),
    ]
    return "\n\n".join(
        f"[BEGIN {name}]\n{text}\n[END {name}]" for name, text in blocks if text
    )


def _step_part(step: Step) -> list[ChatMessage]:
    if isinstance(step, BeforeToolCall | AfterToolCall):
        # the agent's own message holds the call, and its id lets a model's
        # answer cite it; other calls it holds are not shown
        holder = next((m for m in reversed(step.history) if _holds(m, step.call)), None)
        call = ChatMessageAssistant(
            id=holder.id if holder else None,
            content=step.message,
            tool_calls=[step.call],
        )
        return [call, step.result] if isinstance(step, AfterToolCall) else [call]
    if isinstance(step, BeforeGenerate):
        return step.input[_last_reply(step.input) + 1 :]
    assert isinstance(step, AfterGenerate)
    # a provider can return no choices: show that as an empty reply
    return [
        step.output.message if step.output.choices else ChatMessageAssistant(content="")
    ]


def _window(
    excerpt: Excerpt[S], cut: Callable[[list[ChatMessage]], list[ChatMessage]]
) -> Excerpt[S]:
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
