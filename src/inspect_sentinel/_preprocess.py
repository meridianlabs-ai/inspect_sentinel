# What a monitor shows a model about a step, set once and applied by
# `Host.score` (or by `preprocess_step` before `step_as_str` in a custom
# prompt). Inside a monitor, which is given its context and the step:
#
#   async def check(context: Context, step: BeforeToolCall) -> Observation | None:
#       return await context.host.score(step, DIMENSIONS, preprocessor=StepPreprocessor(
#           input_window=LastTurns(6),                      # the last 6 turns the model was sent,
#                                                           # in place of what is new since its last report
#           omit_reasoning=True,                            # without the model's reasoning
#           transform=withhold_flagged,                     # anything else, on the thinned step
#       ))

from __future__ import annotations

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
from inspect_ai.tool import ToolCall
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


@dataclass(frozen=True)
class LastTurns:
    """Keep the last `n` turns of a conversation.

    A turn starts at an assistant message and runs to the next one, so a call is never separated from its result, unless the conversation already starts mid-turn. Earlier messages, including the system message and the task, are left out unless the conversation has no more than `n` turns, in which case all of it is kept. `n=0` keeps nothing.
    """

    n: int
    """How many turns to keep."""

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"LastTurns needs n >= 0, got {self.n}.")


@dataclass(frozen=True)
class LastMessages:
    """Keep the last `n` messages of a conversation, fewer when the cut would start on a tool result, so the cut never parts a result from its call. A conversation that already starts with a result keeps it."""

    n: int
    """How many messages to keep at most."""

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"LastMessages needs n >= 0, got {self.n}.")


@dataclass(frozen=True)
class AllMessages:
    """Keep all of a conversation."""


@dataclass(frozen=True)
class NewSinceLastReport:
    """Keep the messages after the last one this monitor reported on, so an incremental monitor does not re-read its conversations.

    The last message shown is recorded by id in the monitor's store (by `context.path`), separately for each conversation, for `input` and `history`, and for each function of the monitor; a function that watches several stages shares them across those stages. It is recorded only when the function returns or decides final and the runner accepts what it returned; when it raises or is cancelled, or a descendant's final decision ends the step past it, the next step shows those messages again. A conversation is kept whole the first time, and when its recorded message is gone: after a compaction, or where a host gives messages new ids on each step (inspect_ai does that to `input` when a provider merges consecutive messages, and to earlier reasoning messages when `reasoning_history` is `none` or `last`). Preprocess with it once per step, from the monitor or protocol function the runner is running; elsewhere preprocessing raises `RuntimeError`.
    """

    context: Context
    """The monitor's context, whose store holds what was shown."""


Window: TypeAlias = AllMessages | LastTurns | LastMessages | NewSinceLastReport
"""How much of one conversation, `input` or `history`, to show."""

# what a conversation with no window is cut to
_NOTHING = LastTurns(0)


@dataclass(frozen=True, kw_only=True)
class StepPreprocessor(Generic[S]):
    """What a monitor shows a model about a step: how much of its conversations, which parts of their messages, and any change of the monitor's own.

    `preprocess_step()` applies it, in this order: the windows to `input` and `history`, then the omissions to every message the step holds, then `transform`, which sees the result. The windows cut only each conversation's part before the step's own part, so that part is never windowed and is not shown twice: at a tool stage, `history` is cut before the assistant message that holds the call and `input` is whole; before a generate, `input` is cut after the model's last reply, and `history` before the messages that follow it, matched by id; after a generate, neither is cut.
    """

    input_window: Window | None = field(default_factory=AllMessages)
    """How much of `input` to show: by default all of it, so each step's prompt starts with the last one's and the monitor model's prompt cache keeps working; `NewSinceLastReport(context)` for what is new since this monitor last reported; None for none. A tool step whose input is unknown says so whatever the window."""

    history_window: Window | None = None
    """How much of `history` to show: none by default; `AllMessages()` for all of it. Shown, it is usually the same conversation as `input`, so `step_as_str()` shows it only when it holds a message `input` does not, such as turns before a compaction; a history the host could not provide is named."""

    omit_system: bool = True
    """Leave system messages out, as Scout does by default."""

    omit_reasoning: bool = False
    """Leave the model's reasoning out, in the step's own reply too."""

    omit_tool_usage: bool = False
    """Leave out a provider's own tool use (such as web search) recorded in a message."""

    allow_placeholders: bool = False
    """Render content that cannot be shown as text (images, audio, video, documents, provider data, unknown types) as a placeholder such as `<image />`. When False, as by default, `step_as_str()` raises rather than show the model less than the agent saw."""

    transform: Callable[[S], Awaitable[S]] | None = None
    """Any further change, applied last to the thinned step: for example, withholding a message flagged for the monitor not to see. It returns a step of the same type."""


async def preprocess_step(
    step: S, preprocessor: StepPreprocessor[S] | None = None
) -> S:
    """`step`, thinned as `preprocessor` says, by default as `StepPreprocessor()` says: all of `input`, no history, no system messages.

    The step's own objects are not changed; the result is a copy with the same type.

    Args:
        step: The step a monitor was given.
        preprocessor: What to show; `StepPreprocessor()` when None.

    Raises:
        RuntimeError: If a window is `NewSinceLastReport` and this is called outside a monitor or protocol function the runner is running.
    """
    p = preprocessor or StepPreprocessor()
    # a conversation with no window keeps only the step's own part, so a
    # cite of the message that holds a call still resolves
    input = (
        None
        if step.input is None
        else _windowed(step, "input", p.input_window or _NOTHING)
    )
    # a history that is not shown is not named as unknown either
    history = (
        []
        if step.history is None and p.history_window is None
        else None
        if step.history is None
        else _windowed(step, "history", p.history_window or _NOTHING)
    )
    thinned = _with_output(
        replace(
            step,
            input=None if input is None else _omitted(input, p),
            history=None if history is None else _omitted(history, p),
        ),
        p,
    )
    return await p.transform(thinned) if p.transform else thinned


async def step_as_str(
    step: Step, *, nonce: str | None = None, allow_placeholders: bool = False
) -> tuple[str, ExtractReferences]:
    """Render `step` for a monitor's prompt, numbering its messages `[M1]`, `[M2]` and so on, and return the text with a function that turns those cites in a model's answer into references.

    Pass a step that `preprocess_step()` thinned, or a step, shown with all of its history. Each part is a block between `[BEGIN ...]` and `[END ...]` lines, left out when it has nothing to show: `INPUT`, `HISTORY`, then the step's own part, in a block marked `[AGENT STEP - BEGIN ...]` and `[AGENT STEP - END ...]` and named for the stage: `TOOL CALL` before a tool call, `TOOL CALL AND RESULT` after one, `MODEL REQUEST` (the messages the model is about to answer) before a generate, and `MODEL REPLY` after one. The step's own part is the assistant's message and the call before a tool call, those and the result after one (other calls the same message holds, and their results, are not shown), the messages after the model's last reply before a generate (none when the request ends with an assistant message, such as a prefill, which then shows in `input`), and the reply after one. `HISTORY`, usually the same conversation as `input`, is shown whole only when it holds a message `input` does not, such as turns before a compaction; messages are matched by id, counting the ones a merged message was made from. An unknown model input or history (`None`) is named in its block. With a `nonce`, every `[BEGIN ...]` and `[END ...]` line carries it in parentheses, as `[BEGIN INPUT (3f9a1c7e)]`, and so does every message header where the installed Scout takes a nonce. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it. Needs `inspect_scout` installed; it is not a dependency of this package yet.

    Args:
        step: The step to render, usually from `preprocess_step()`.
        nonce: A code the agent cannot guess, such as `secrets.token_hex(4)`, for every block marker and message header; none by default. Scout needs at least 8 characters.
        allow_placeholders: Render content that cannot be shown as text as a placeholder such as `<image />`; when False, raise instead.

    Raises:
        ValueError: If a message holds content that cannot be shown as text and `allow_placeholders` is False.
    """
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import MessagesPreprocessor, message_numbering

    # the preprocessor has already decided what is left out, system
    # messages included
    numbered, extract_refs = message_numbering(
        MessagesPreprocessor(exclude_system=False),
        **({} if nonce is None else scout_nonce(nonce)),  # pyright: ignore[reportArgumentType]
    )

    async def text(messages: list[ChatMessage]) -> str:
        if not allow_placeholders:
            _check_renderable(messages)
        return await numbered(messages) if messages else ""

    input = None if step.input is None else _before_own_part(step, "input")
    history = None if step.history is None else _before_own_part(step, "history")
    # rendered in this order, so messages are numbered in it too
    blocks = [
        (
            "",
            "INPUT",
            "The model input for this call is not available."
            if input is None
            else await text(input),
        ),
        (
            "",
            "HISTORY",
            "The agent's conversation history is not available."
            if history is None
            else await text(history)
            if history and _adds(history, input or [])
            else "",
        ),
        ("AGENT STEP - ", _step_block(step), await text(_step_part(step))),
    ]
    mark = "" if nonce is None else f" ({nonce})"
    return (
        "\n\n".join(
            f"[{prefix}BEGIN {name}{mark}]\n{body}\n[{prefix}END {name}{mark}]"
            for prefix, name, body in blocks
            if body
        ),
        extract_refs,
    )


def scout_nonce(nonce: str) -> dict[str, str]:
    # Scout 0.5.5 and earlier take no nonce; until this package requires a
    # later Scout, it is passed only to one that takes it. Scout added it to
    # messages_as_str and message_numbering together, so one check serves both
    from inspect_scout import messages_as_str

    return {"nonce": nonce} if "nonce" in signature(messages_as_str).parameters else {}


# what Scout renders as text; the rest it shows as a placeholder or drops.
# Stands in for Scout's own allow_placeholders option until it has one
_RENDERABLE = ("text", "reasoning", "tool_use")


def _check_renderable(messages: list[ChatMessage]) -> None:
    unrenderable = sorted(
        {
            content.type
            for message in messages
            if not isinstance(message.content, str)
            for content in message.content
            if content.type not in _RENDERABLE
        }
    )
    if unrenderable:
        raise ValueError(
            f"The step holds content that cannot be shown as text ({', '.join(unrenderable)}), so the monitor model would see less than the agent did. Pass allow_placeholders=True to show it as a placeholder."
        )


class _Marks(StoreModel):
    # "<conversation>/<input|history>/<function>" -> id of the last message
    # shown
    marks: dict[str, str] = Field(default_factory=dict)


def _windowed(
    step: Step, source: Literal["input", "history"], window: Window
) -> list[ChatMessage]:
    messages = getattr(step, source)
    before = _before_own_part(step, source)
    own = messages[len(before) :]
    return [*_cut(step, source, before, window), *own]


def _cut(
    step: Step,
    source: Literal["input", "history"],
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
        case NewSinceLastReport(context):
            return _after_mark(context, step, source, messages)


def _after_mark(
    context: Context,
    step: Step,
    source: Literal["input", "history"],
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
    return (
        messages[ids.index(seen) + 1 :]
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
        if not (p.omit_system and message.role == "system")
    ]


def _with_output(step: S, p: StepPreprocessor[S]) -> S:
    # after a generate the step's own part is the reply, which the
    # omissions reach too
    if not isinstance(step, AfterGenerate) or not step.output.choices:
        return step
    output = step.output.model_copy(
        update={
            "choices": [
                choice.model_copy(update={"message": _omitted([choice.message], p)[0]})
                for choice in step.output.choices
            ]
        }
    )
    return replace(step, output=output)


def _before_own_part(
    step: Step, source: Literal["input", "history"]
) -> list[ChatMessage]:
    messages: list[ChatMessage] = getattr(step, source)
    if isinstance(step, BeforeToolCall | AfterToolCall):
        return messages if source == "input" else _before_call(messages, step.call)
    if isinstance(step, BeforeGenerate):
        if source == "input":
            return messages[: _last(messages, lambda m: m.role == "assistant") + 1]
        new = {m.id for m in _new(step) if m.id is not None}
        return messages[: _last(messages, lambda m: m.id not in new) + 1]
    return messages


def _adds(history: list[ChatMessage], input: list[ChatMessage]) -> bool:
    # a message the model's preparation merged with another gets a new id,
    # and inspect_ai records the ids it was made from
    seen = {message.id for message in input} | {
        id
        for message in input
        for id in (message.metadata or {}).get("combined_from", [])
    }
    return any(message.id is None or message.id not in seen for message in history)


def _step_block(step: Step) -> str:
    # named for what it holds, so the model can tell whether a call has run
    if isinstance(step, BeforeToolCall):
        return "TOOL CALL"
    if isinstance(step, AfterToolCall):
        return "TOOL CALL AND RESULT"
    return "MODEL REQUEST" if isinstance(step, BeforeGenerate) else "MODEL REPLY"


def _step_part(step: Step) -> list[ChatMessage]:
    if isinstance(step, BeforeToolCall | AfterToolCall):
        # the agent's own message holds the call, and its id lets a model's
        # answer cite it; when history lacks it, the message gets a new id and
        # a cite of it matches no message. Other calls it holds are not shown
        holder = next(
            (m for m in reversed(step.history or []) if _holds(m, step.call)), None
        )
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


def _new(step: BeforeGenerate) -> list[ChatMessage]:
    return step.input[_last(step.input, lambda m: m.role == "assistant") + 1 :]


def _last(messages: list[ChatMessage], test: Callable[[ChatMessage], bool]) -> int:
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
