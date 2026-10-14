# Common ways to choose what a monitor shows a model about a step. Each window
# takes a step or an excerpt and returns an excerpt, so they compose. Inside a
# monitor, which is given its context and the step:
#
#   async def check(context: Context, step: BeforeToolCall) -> Observation | None:
#       last_turns(step, 6)                             # the last 6 turns
#       last_messages(step, 20)                         # the last 20 messages
#       drop_history(last_turns(step, 6))               # the last 6 turns the model was sent
#       drop_input(step)                                # the scaffold's conversation only
#       drop_input(drop_history(step))                  # the step alone
#       drop_reasoning(last_turns(step, 6))             # without the model's reasoning
#       keep_system(step)                               # with the system prompt, left out by default
#       new_since_last_report(context, step)            # only what this monitor has not seen
#       last_turns(new_since_last_report(context, step), 6)  # that, at most 6 turns
#
# then render it for the prompt (a step on its own renders whole):
#
#       evidence = await step_as_str(last_turns(step, 6))

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeAlias, TypeVar

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentToolUse,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import StoreModel
from pydantic import Field

from ._context import Context
from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step
from ._unhandled import current_invocation

S = TypeVar("S", bound=Step)

# an excerpt is frozen, so an Excerpt[BeforeToolCall] is an Excerpt[Step]
S_co = TypeVar("S_co", bound=Step, covariant=True)

Exclusion: TypeAlias = Literal["system", "reasoning", "tool_usage"]

MessagesAsStr: TypeAlias = Callable[[list[ChatMessage]], Awaitable[str]]


@dataclass(frozen=True)
class Excerpt(Generic[S_co]):
    """What a monitor shows a model about a step: the step, and the conversations to show with it, cut down by the windowing functions, which return one.

    A step's conversations start whole, each cut before the step's own part so that part is not shown twice: `history` before the assistant message that holds the call at a tool stage, and before a generate `input` after the model's last reply and `history` before the messages that follow it, matched by id. Where the host gave those messages new ids in the request (inspect_ai does when a provider merges consecutive messages, or a tool rewrites its result for the model), `history` keeps them and they show twice.

    `step_as_str()` renders the conversations and the step's own part: the assistant's message and the call before a tool call, those and the result after one (other calls the same message holds, and their results, are not shown), the messages after the model's last reply before a generate (none when the request ends with an assistant message, such as a prefill, which then shows in `input`), and the reply after one. Nothing else of the step is rendered, such as its escalations, tools, config, `view` or untruncated tool output.
    """

    step: S_co
    """The step, whole. Its `input` and `history` are not what is shown; the excerpt's are."""

    input: list[ChatMessage] | None
    """The part of `step.input` to show, None when a tool step has none (no model call produced it, or the record of that call is gone; see `BeforeToolCall.input`)."""

    history: list[ChatMessage]
    """The part of `step.history` to show."""

    exclude: frozenset[Exclusion] = frozenset({"system"})
    """What `step_as_str()` leaves out of every block, the step's own part included: `system` messages, `reasoning`, and `tool_usage` (a provider's own tool use, such as web search, recorded in a message). System messages are left out by default, as Scout leaves them out; `keep_system()` keeps them, and `drop_reasoning()` and `drop_tool_usage()` add to it."""


def drop_input(step: S | Excerpt[S]) -> Excerpt[S]:
    """`step`, excerpted, without the model's input: an empty `input`, or None where the step had none, which `step_as_str()` still names as unknown.

    Args:
        step: A step, or an excerpt to narrow further.
    """
    cut = _excerpt(step)
    return replace(cut, input=None if cut.input is None else [])


def drop_history(step: S | Excerpt[S]) -> Excerpt[S]:
    """`step`, excerpted, with an empty `history`.

    Args:
        step: A step, or an excerpt to narrow further.
    """
    return replace(_excerpt(step), history=[])


def keep_system(step: S | Excerpt[S]) -> Excerpt[S]:
    """`step`, excerpted, with system messages shown when it is rendered; they are left out by default.

    Args:
        step: A step, or an excerpt to narrow further.
    """
    shown = _excerpt(step)
    return replace(shown, exclude=shown.exclude - {"system"})


def drop_reasoning(step: S | Excerpt[S]) -> Excerpt[S]:
    """`step`, excerpted, with the model's reasoning left out when it is rendered, in the step's own reply too.

    Args:
        step: A step, or an excerpt to narrow further.
    """
    return _leave_out(step, "reasoning")


def drop_tool_usage(step: S | Excerpt[S]) -> Excerpt[S]:
    """`step`, excerpted, with a provider's own tool use (such as web search, recorded in a message) left out when it is rendered.

    Args:
        step: A step, or an excerpt to narrow further.
    """
    return _leave_out(step, "tool_usage")


def last_turns(step: S | Excerpt[S], n: int) -> Excerpt[S]:
    """`step`, excerpted, with `input` and `history` each cut to their last `n` turns.

    A turn starts at an assistant message and runs to the next one, so a call in a conversation is never separated from its result, unless the conversation already starts mid-turn, as after `new_since_last_report()`. The step's own part is not windowed: before a generate it can be a result whose call was cut. Earlier messages, including the system message and the task, are left out unless a conversation has no more than `n` turns, in which case all of it is kept.

    Args:
        step: A step, or an excerpt to narrow further.
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

    return _window(step, cut)


def last_messages(step: S | Excerpt[S], n: int) -> Excerpt[S]:
    """`step`, excerpted, with `input` and `history` each cut to their last `n` messages, fewer when the cut would start on a tool result, so the cut never parts a result from its call. A conversation that already starts with a result, as after `new_since_last_report()`, keeps it. The step's own part is not windowed.

    Args:
        step: A step, or an excerpt to narrow further.
        n: How many messages to keep at most.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError(f"last_messages() needs n >= 0, got {n}.")

    def cut(messages: list[ChatMessage]) -> list[ChatMessage]:
        start = len(messages) - min(n, len(messages))
        # skip results only where this cut parted them from their call
        return messages[
            next(
                (
                    i
                    for i in range(start, len(messages))
                    if start == 0 or messages[i].role != "tool"
                ),
                len(messages),
            ) :
        ]

    return _window(step, cut)


class _Marks(StoreModel):
    # "<conversation>/<input|history>/<function>" -> id of the last message
    # shown
    marks: dict[str, str] = Field(default_factory=dict)


def new_since_last_report(context: Context, step: S | Excerpt[S]) -> Excerpt[S]:
    """`step`, excerpted, with `input` and `history` each cut to the messages after the last one this monitor reported on, so an incremental monitor does not re-read its conversations.

    The last message of each is recorded by id in the monitor's store (by `context.path`), separately for each conversation and each function of the monitor; a function that watches several stages shares them across those stages. It is recorded only when the function returns or decides final and the runner accepts what it returned; when it raises or is cancelled, the next step shows those messages again. Only `input` and `history` are tracked, so the step's own part, such as the last step's call, can be shown again in a later step's conversation. A conversation is kept whole the first time, and when its recorded message is gone: after a compaction (and a tool step's `history`, which is not compacted, may then be shown whole once after a generate recorded the compacted one), or where a host gives messages new ids on each step. inspect_ai does that to `input` when a provider merges consecutive messages (Anthropic and Bedrock merge user messages) and to earlier reasoning messages when `reasoning_history` is `none` or `last`, so for those, window `input` some other way or drop it. Call it once per step, from the monitor or protocol function the runner is running.

    Args:
        context: The monitor's context.
        step: A step, or an excerpt to narrow further.

    Raises:
        RuntimeError: If called outside a monitor or protocol function the runner is running.
    """
    node = current_invocation()
    if node is None:
        raise RuntimeError(
            "new_since_last_report() works only in a monitor or protocol function the runner is running, since it records what the function reported on."
        )
    cut = _excerpt(step)
    store = context.store_as(_Marks)
    marks: dict[str, str] = {}

    def after_mark(
        source: Literal["input", "history"], messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        key = f"{cut.step.conversation}/{source}/{node.function}"
        seen = store.marks.get(key)
        ids = [m.id for m in messages]
        last = next((m.id for m in reversed(messages) if m.id is not None), None)
        if last is not None:
            marks[key] = last
        return (
            messages[ids.index(seen) + 1 :]
            if seen is not None and seen in ids
            else messages
        )

    narrowed = replace(
        cut,
        input=None if cut.input is None else after_mark("input", cut.input),
        history=after_mark("history", cut.history),
    )
    node.on_success.append(lambda: setattr(store, "marks", {**store.marks, **marks}))
    return narrowed


async def step_as_str(
    step: S | Excerpt[S], *, messages_as_str: MessagesAsStr | None = None
) -> str:
    """Render `step`, or an excerpt of it, for a monitor's prompt.

    Each part is a block between `[BEGIN ...]` and `[END ...]` lines, left out when it has nothing to show: `HISTORY`, `INPUT`, then `AGENT STEP`, the step's own part (see `Excerpt`). The history is usually the same conversation as the input, so `HISTORY` is shown, whole, only when it holds a message the input does not, such as turns before a compaction; messages are matched by id, counting the ones a merged message was made from. When a tool step has no model input, the `INPUT` block says so, even after `drop_input()`. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it. Needs `inspect_scout` installed; it is not a dependency of this package yet.

    Args:
        step: A step, rendered whole, or an excerpt of one.
        messages_as_str: Renders a list of messages; pass the first function of Scout's `message_numbering()` to number them so a model's answer can cite them (made with `MessagesPreprocessor(exclude_system=False)`, so that the excerpt decides whether system messages show). By default Scout's `messages_as_str`, which leaves out only what the excerpt excludes.
    """
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import MessagesPreprocessor
    from inspect_scout import messages_as_str as scout_messages_as_str

    async def render_all(messages: list[ChatMessage]) -> str:
        # Scout leaves out system messages by default; the excerpt decides here
        return await scout_messages_as_str(
            messages, preprocessor=MessagesPreprocessor(exclude_system=False)
        )

    shown = _excerpt(step)
    render = messages_as_str or render_all

    async def text(messages: list[ChatMessage]) -> str:
        # a renderer may write an empty list as something, such as JSON's []
        return await render(messages) if messages else ""

    history = _without(shown.history, shown.exclude)
    input = None if shown.input is None else _without(shown.input, shown.exclude)
    blocks = [
        # usually the same conversation as the input, so shown only when it
        # holds a message the input does not, such as turns before a compaction
        ("HISTORY", await text(history) if _adds(history, input or []) else ""),
        (
            "INPUT",
            "The model input for this call is not available."
            if input is None
            else await text(input),
        ),
        ("AGENT STEP", await text(_without(_step_part(shown.step), shown.exclude))),
    ]
    return "\n\n".join(
        f"[BEGIN {name}]\n{text}\n[END {name}]" for name, text in blocks if text
    )


def _adds(history: list[ChatMessage], input: list[ChatMessage]) -> bool:
    # a message the model's preparation merged with another gets a new id,
    # and inspect_ai records the ids it was made from
    seen = {message.id for message in input} | {
        id
        for message in input
        for id in (message.metadata or {}).get("combined_from", [])
    }
    return any(message.id is None or message.id not in seen for message in history)


def _step_part(step: Step) -> list[ChatMessage]:
    if isinstance(step, BeforeToolCall | AfterToolCall):
        # the agent's own message holds the call, and its id lets a model's
        # answer cite it; when history lacks it, the message gets a new id and
        # a cite of it matches no message. Other calls it holds are not shown
        holder = next((m for m in reversed(step.history) if _holds(m, step.call)), None)
        call = ChatMessageAssistant(
            id=holder.id if holder else None,
            content=step.message,
            tool_calls=[step.call],
        )
        return [call, step.result] if isinstance(step, AfterToolCall) else [call]
    if isinstance(step, BeforeGenerate):
        return _new(step)
    assert isinstance(step, AfterGenerate)
    # a provider can return no choices: show that as an empty reply
    return [
        step.output.message if step.output.choices else ChatMessageAssistant(content="")
    ]


def _excerpt(step: S | Excerpt[S]) -> Excerpt[S]:
    if isinstance(step, Excerpt):
        return step
    if isinstance(step, BeforeToolCall | AfterToolCall):
        return Excerpt(step, step.input, _before_call(step.history, step.call))
    if isinstance(step, BeforeGenerate):
        new = {m.id for m in _new(step) if m.id is not None}
        return Excerpt(
            step,
            step.input[: _last(step.input, lambda m: m.role == "assistant") + 1],
            step.history[: _last(step.history, lambda m: m.id not in new) + 1],
        )
    return Excerpt(step, step.input, step.history)


def _leave_out(step: S | Excerpt[S], exclusion: Exclusion) -> Excerpt[S]:
    shown = _excerpt(step)
    return replace(shown, exclude=shown.exclude | {exclusion})


def _without(
    messages: list[ChatMessage], exclude: frozenset[Exclusion]
) -> list[ChatMessage]:
    removed = tuple(
        kind
        for name, kind in (
            ("reasoning", ContentReasoning),
            ("tool_usage", ContentToolUse),
        )
        if name in exclude
    )
    # a copy keeps the message's id, so a cite of it still resolves
    return [
        message
        if isinstance(message.content, str) or not removed
        else message.model_copy(
            update={
                "content": [c for c in message.content if not isinstance(c, removed)]
            }
        )
        for message in messages
        if not ("system" in exclude and message.role == "system")
    ]


def _window(
    step: S | Excerpt[S], cut: Callable[[list[ChatMessage]], list[ChatMessage]]
) -> Excerpt[S]:
    shown = _excerpt(step)
    return replace(
        shown,
        input=None if shown.input is None else cut(shown.input),
        history=cut(shown.history),
    )


def _new(step: BeforeGenerate) -> list[ChatMessage]:
    return step.input[_last(step.input, lambda m: m.role == "assistant") + 1 :]


def _last(messages: Sequence[ChatMessage], test: Callable[[ChatMessage], bool]) -> int:
    return max((i for i, m in enumerate(messages) if test(m)), default=-1)


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
