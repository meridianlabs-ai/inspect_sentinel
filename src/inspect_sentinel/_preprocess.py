# What a monitor shows a model about a step, set once and applied by
# `Host.score` (or by `preprocess_step` before `step_as_str` in a custom
# prompt). Inside a monitor, which is given its context and the step:
#
#   async def check(context: Context, step: BeforeToolCall) -> Observation | None:
#       return await context.host.score(step, DIMENSIONS, preprocessor=StepPreprocessor(
#           input_window=LastTurns(6),                      # the last 6 turns the model was sent
#           omit_reasoning=True,                            # without the model's reasoning
#           hide_messages=lambda m: (m.metadata or {}).get("red_team") is True,  # marked where they were
#           transform=paraphrased,                          # anything else, on the thinned step
#       ))

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    Content,
    ContentReasoning,
    ContentText,
    ContentToolUse,
    ModelOutput,
)
from inspect_ai.scorer import Reference
from inspect_ai.tool import ToolChoice, ToolInfo
from typing_extensions import TypeVar

from ._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    HiddenMessages,
    Step,
    Truncation,
)
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

    An incremental prompt chooses `input_window` as well: `StepPreprocessor` shows all of `input` by default, so `StepPreprocessor(history_window=NewSinceLastReport(context))` still sends the monitor model the whole request at every step. To judge what was already sent, at `AfterGenerate` or a tool stage, where `history` holds the request, window `history` and omit or bound `input`: `StepPreprocessor(input_window=None, history_window=NewSinceLastReport(context))`. At `BeforeGenerate`, `history` does not yet hold the request about to go out, so a monitor that judges that request keeps `input`: `StepPreprocessor(input_window=NewSinceLastReport(context))` shows only what is new in it. `history` only grows and is resumed where it was; over `input` or `agent_messages`, which can be rewritten, each step rereads the whole list to find what is new, though it shows only that.

    The last message shown is recorded by id and position in the monitor's store (by `context.path`), separately for each conversation, for each list it windows, and for each function of the monitor. It is recorded only when the function returns or decides final and the runner accepts what it returned, so a function that raises or is cancelled is shown those messages again at the next step. `history`, which only grows, is resumed where it was without reading what was shown before; `input` and `agent_messages` can be rewritten, as by a compaction, so for them the whole list is read to find the message shown last, a cost linear in the list at each step. An edited version of a message, which `history` holds again under the same id, is new. A conversation whose recorded message is gone, as after a compaction, is kept whole. Preprocess with it from the monitor or protocol function the runner is running; elsewhere preprocessing raises `RuntimeError`.
    """

    context: Context
    """The monitor's context, whose store holds what was shown."""


Window: TypeAlias = AllMessages | LastTurns | LastMessages | NewSinceLastReport
"""How much of one conversation, `input`, `history` or `agent_messages`, to show."""


@dataclass(frozen=True, kw_only=True)
class StepPreprocessor(Generic[S]):
    """What a monitor shows a model about a step.

    `preprocess_step()` applies it, in this order: `hide_messages` to every message the step holds except assistant messages, then the windows to `input`, `history` and `agent_messages`, then the omissions to every message the step holds (`omit_assistant_text` also leaves out an assistant message it empties), then `transform`, which sees the result. By default `input` is shown whole, while `history` and `agent_messages` are omitted, even when the host could not provide them, so a default prompt is the same live and in replay. A scaffold may keep hidden instructions out of its own list but send them in `input` and `history`. To omit the request lists and show the agent's own list, use `StepPreprocessor(input_window=None, history_window=None, agent_messages_window=AllMessages())`; this requires the conversation-owning scaffold to be the outermost `@agent` agent function. Other stage content remains: every stage renders the request's tools and tool choice while `input` is known (a tool stage's are the producing request's); `AfterGenerate` renders the reply, including reasoning by default before the scaffold has curated it; both tool stages render the assistant text (unless `omit_assistant_text`) and call; and `AfterToolCall` renders the result. Strict blindness also needs a stage-specific `transform` or a custom prompt that curates these surfaces; a transform can clear `tools` and `tool_choice`.
    """

    input_window: Window | None = field(default_factory=AllMessages)
    """How much of `input` to show: all of it by default; None for none.

    Before a generate, `input` is the request being judged, so a window that leaves none of it (None, `LastTurns(0)`, `LastMessages(0)`, or `NewSinceLastReport` when nothing is new) shows the model only the tools offered when the other message windows also leave none. At a tool stage the call is judged against only the tools of the request that produced it.
    """

    history_window: Window | None = None
    """How much of `history`, every message the agent has sent its model, to show: none by default."""

    agent_messages_window: Window | None = None
    """How much of `agent_messages`, the agent's own message list, to show: none by default."""

    omit_system: bool = True
    """Leave system messages out."""

    omit_reasoning: bool = False
    """Leave the model's reasoning out."""

    omit_tool_usage: bool = False
    """Leave out a provider's own tool use (such as web search) recorded in a message."""

    omit_assistant_text: bool = False
    """Leave out the text of assistant messages, the agent's model's own words, and keep their tool calls, so the monitor model judges what the agent did rather than what it says it is doing.

    It applies to every assistant message in `input`, `history` and `agent_messages`, to a tool stage's `message` (the text beside its call, which becomes `""`) and `assistant_message`, and to a generate's reply: every choice's message, the output's `completion`, which becomes the thinned first message's text (`""`), and the choices' `logprobs` and `prompt_logprobs`, which hold text as tokens and are left out. Reasoning follows `omit_reasoning` and a provider's own tool use follows `omit_tool_usage`, so either can be kept or left out with it; other content, such as an image, is kept. An assistant message in a list that is left with no tool calls and no content (or only empty reasoning) is left out of the list; the windows are applied first, so it still counts toward `LastTurns(n)` and `LastMessages(n)`. The reply and the call are always shown, empty if need be.

    Text the agent's model wrote can still reach the monitor model elsewhere: in tool call arguments (such as a `think()` call or a comment in a command), in compaction summaries (user messages), and in results returned by sub-agents or tools."""

    hide_messages: Callable[[ChatMessage], bool] | None = None
    """Hide the messages for which this returns True, always leaving a marker line where they were, so the monitor model can tell that something was hidden (unlike the `omit_` options, which leave content out silently).

    It is for a deployment that keeps messages it has flagged from the monitor, such as a red-team instruction the agent is given in a control eval: `hide_messages=lambda m: (m.metadata or {}).get("red_team") is True`. The marker counts the messages and says nothing else of them, as `[2 messages hidden (3f9a1c7e)]` with a nonce; consecutive hidden messages share one marker. It is asked only about the messages that come into the conversation, system, user and tool messages, and never about an assistant message: the agent's replies and tool calls, including a tool stage's call and a generate's reply, are always shown, so the agent cannot use it to hide what it does. It applies before the windows, so a hidden message does not count toward `LastTurns(n)` or `LastMessages(n)`, to `input`, `history` and `agent_messages`, and to an `AfterToolCall` step's result, which is then rendered as the marker after its call, still shown. It also applies to the task `Host.score` shows, the sample's input (a string input is tested as one user message). A system message `omit_system` leaves out is left out without a marker. `preprocess_step()` records what was hidden in the step's `hidden`.

    In the thinned lists, each run of hidden messages is a placeholder user message, with no id, whose text is the marker without a nonce; `step_as_str()` renders it as the marker, and a `transform` that keeps it keeps the marker where it is. A step thinned again keeps what was hidden, and `Host.score` hides from the task what any of the preprocessings applied to it would hide.

    Match on something the agent cannot write, such as metadata the eval sets on its own messages. A match on text also lets the agent get a tool result or a user message it wrote (such as a sub-agent's prompt) hidden, behind a genuine marker, by making it contain that text. It is given the host's own messages: it must not change them. With it set, every message of each windowed list is read at every step, even under `NewSinceLastReport`.
    """

    transform: Callable[[S], Awaitable[S]] | None = None
    """Any further change, applied last to the thinned step. It returns a step of the same type. The messages it is given are the host's own objects: copy one before changing it."""


async def preprocess_step(
    step: S, preprocessor: StepPreprocessor[S] | None = None
) -> S:
    """`step`, thinned as `preprocessor` says (`StepPreprocessor()` when None).

    Each window cuts its list as it is, less what `hide_messages` hides. A list with no window becomes `[]`, even one the host could not provide; a windowed list the host could not provide stays None. The omissions apply to `input`, `history`, `agent_messages`, a tool call's assistant text, assistant message and result, and a generate's reply. What `hide_messages` hid is recorded in the returned step's `hidden`, with what a step given already thinned had hidden; it is None when no `hide_messages` was applied.

    Args:
        step: The step a monitor was given.
        preprocessor: What to show; `StepPreprocessor()` when None.

    Raises:
        RuntimeError: If a window is `NewSinceLastReport` and this is called outside a monitor or protocol function the runner is running.
    """
    p = preprocessor or StepPreprocessor()
    hide = p.hide_messages
    # what a step given already thinned had hidden, which stays hidden; its
    # lists hold the placeholders that stand for its hidden messages
    before = step.hidden

    def thin(
        source: _Source,
        messages: Sequence[ChatMessage] | None,
        window: Window | None,
    ) -> list[ChatMessage] | None:
        # a list with no window is left out even when the host could not provide
        # it, so a default prompt is the same live and in replay
        if window is None:
            return []
        if messages is None:
            return None
        # hidden over the whole list before the window, so a hidden message
        # counts toward no window, and a result whose call is outside the
        # window is still hidden with its call
        if hide is not None or before is not None:
            messages = _placed(messages, hide, p.omit_system)
        return _omitted(_cut(step, source, messages, window), p)

    input = thin("input", step.input, p.input_window)
    history = thin("history", step.history, p.history_window)
    agent_messages = thin(
        "agent_messages", step.agent_messages, p.agent_messages_window
    )
    was = HiddenMessages() if before is None else before
    # the step's own call and reply are the agent's, so only its result can
    # be hidden
    result = hide is not None and isinstance(step, AfterToolCall) and hide(step.result)
    thinned = replace(
        step,
        input=input,
        history=history,
        agent_messages=agent_messages,
        hidden=None
        if hide is None and before is None
        else HiddenMessages(
            ids=tuple(dict.fromkeys([*was.ids, *_ids(input, history, agent_messages)])),
            result=result or was.result,
            filters=(*was.filters, *([] if hide is None else [hide])),
        ),
    )
    if isinstance(thinned, AfterToolCall):
        thinned = replace(thinned, result=_omitted_from(thinned.result, p))
    if isinstance(thinned, BeforeToolCall | AfterToolCall):
        if p.omit_assistant_text:
            thinned = replace(thinned, message="")
        if thinned.assistant_message is not None:
            made = _omitted_from(thinned.assistant_message, p)
            assert isinstance(made, ChatMessageAssistant)
            thinned = replace(thinned, assistant_message=made)
    if isinstance(thinned, AfterGenerate):
        thinned = replace(thinned, output=_omitted_output(thinned.output, p))
    return await p.transform(thinned) if p.transform is not None else thinned


async def step_as_str(
    step: Step, *, nonce: str | None = None
) -> tuple[str, ExtractReferences]:
    """Render `step` for a monitor's prompt, numbering its messages `[M1]`, `[M2]` and so on, and return the text with a function that turns those cites in a model's answer into references.

    Everything is in one block, marked `[BEGIN AGENT STEP - ...]` and `[END AGENT STEP - ...]` and named for the stage: `TOOL CALL REQUEST` before a tool call, `TOOL CALL RESULT` after one, `MODEL REQUEST` before a generate, and `MODEL REPLY` after one. Inside it are blocks between `[BEGIN ...]` and `[END ...]` lines, each left out when it is empty. At every stage: `HISTORY` (`step.history`, every message the agent has sent its model), `AGENT MESSAGES` (`step.agent_messages`, the outermost running agent function's own message list, which its scaffold may curate), then the model request, as `INPUT`, `TOOLS` (the tools offered, as JSON) and a one-line `[TOOL CHOICE] "auto"`; at a tool stage, the request that produced the call. Then after a generate, `REPLY`; at a tool stage, `CALL` (the assistant's text and the call) and, after the call, `RESULT`. When the host cut the tool's output before the agent's model saw it (`step.truncated` is a `Truncation`), `RESULT` opens with a line such as `[OUTPUT TRUNCATED] The tool's output was 48,210 bytes; the agent's model was shown at most its first 8,192 and last 8,192 bytes, with the middle left out.`; a result the host did not cut, or cannot tell about (`None`, as in a replay from messages), has no such line. Messages that `StepPreprocessor.hide_messages` hid are one marker line per run where they stood, such as `[2 messages hidden]`, which takes no message number; a hidden result of the step's own call is that marker in the RESULT block, after the call. A list the host could not provide (None) is named in its block (`preprocess_step()` leaves a list with no window empty, so only a windowed one is named); at a tool stage whose input is not known, the tools and tool choice are left out. With a `nonce`, every `[BEGIN ...]` and `[END ...]` line, the `[TOOL CHOICE]` and `[OUTPUT TRUNCATED]` lines and every hidden-messages marker carry it in parentheses, as `[BEGIN INPUT (3f9a1c7e)]`, and so does every message header. Everything rendered was written by or shown to the agent under observation, so treat it as untrusted in the prompt around it.

    Args:
        step: The step to render, usually from `preprocess_step()`.
        nonce: A code the agent cannot guess, such as `secrets.token_hex(4)`, for every block marker and message header, the `[TOOL CHOICE]` and `[OUTPUT TRUNCATED]` lines and every hidden-messages marker; none by default. Scout needs at least 8 characters.
    """
    return await render_step(step, nonce=nonce, cut={})


HeadCut: TypeAlias = Mapping[_Source, int]
"""How many head messages a `Host.score` cut drops from each shown list, keyed by list; only lists that lose messages are keyed."""

# the shown lists a head cut drops from, in cut order: HISTORY, then AGENT
# MESSAGES, then INPUT (`Host.score`, design/sentinel.md)
_CUT_ORDER: tuple[_Source, ...] = ("history", "agent_messages", "input")


def _shown(step: Step, source: _Source) -> list[ChatMessage]:
    # the messages a list shows, without the placeholders for hidden ones
    messages: Sequence[ChatMessage] | None = getattr(step, source)
    return [m for m in messages or [] if not isinstance(m, _Hidden)]


def _cuttable_in(step: Step, source: _Source) -> int:
    shown = len(_shown(step, source))
    # before a generate, the newest INPUT message is the request being judged,
    # so a cut keeps it: a step the model is not shown cannot be scored
    if source == "input" and isinstance(step, BeforeGenerate):
        return max(shown - 1, 0)
    return shown


def keeps_newest_input(step: Step) -> bool:
    """Whether a head cut keeps the newest INPUT message as part of the fixed part of the prompt: before a generate, when `input` shows a message."""
    return isinstance(step, BeforeGenerate) and bool(_shown(step, "input"))


def cuttable(step: Step) -> int:
    """How many messages a head cut could drop from the step's shown lists. Before a generate, the newest INPUT message is not cuttable."""
    return sum(_cuttable_in(step, source) for source in _CUT_ORDER)


def head_cut(step: Step, k: int) -> dict[_Source, int]:
    """How many head messages to drop from each shown list to drop `k` in all, HISTORY first, then AGENT MESSAGES, then INPUT. Before a generate, the newest INPUT message is never dropped. Only the lists that lose messages are keyed."""
    counts: dict[_Source, int] = {}
    for source in _CUT_ORDER:
        take = min(k, _cuttable_in(step, source))
        if take:
            counts[source] = take
            k -= take
    return counts


async def render_step(
    step: Step, *, nonce: str | None, cut: HeadCut
) -> tuple[str, ExtractReferences]:
    # `step_as_str` with the head of each shown list dropped as `cut` says, so
    # `Host.score` can fit a prompt to a model's context window
    #
    # Scout imports duckdb and pandas, so load it only when a monitor renders
    from inspect_scout import MessagesPreprocessor, message_numbering

    # the preprocessor has already decided what is left out
    numbered, extract_refs = message_numbering(
        MessagesPreprocessor(exclude_system=False),
        nonce=nonce,
    )

    mark = "" if nonce is None else f" ({nonce})"
    hidden = HiddenMessages() if step.hidden is None else step.hidden

    def hidden_marker(count: int) -> str:
        return f"{hidden_line(count, nonce)}\n"

    async def text(messages: Sequence[ChatMessage] | None, unknown: str) -> str:
        if messages is None:
            return unknown
        # placeholders are one marker line where their hidden messages stood
        # (two are adjacent once an omitted system message between them is
        # gone); the marker takes no message number, and the numbering runs
        # on across it, so each cite still resolves to the message it names
        parts: list[str] = []
        shown: list[ChatMessage] = []
        hidden = 0
        for message in [*messages, None]:
            if shown and (message is None or isinstance(message, _Hidden)):
                parts.append(await numbered(shown))
                shown = []
            if hidden and not isinstance(message, _Hidden):
                parts.append(hidden_marker(hidden))
                hidden = 0
            if isinstance(message, _Hidden):
                hidden += len(message.hidden_ids)
            elif message is not None:
                shown.append(message)
        return "\n".join(parts)

    async def listed(
        source: _Source, messages: Sequence[ChatMessage] | None, unknown: str
    ) -> str:
        # a head cut drops the oldest messages of a shown list and names the
        # count in one marker line before the kept ones; the marker takes no
        # message number, so the render renumbers the kept messages from the
        # top, and each cite still resolves to the kept message it names
        if messages is None:
            return unknown
        dropped = cut.get(source, 0)
        if not dropped:
            return await text(messages, unknown)
        # the placeholders among the dropped messages, and those just after
        # them, go with them
        messages = list(messages)
        start = 0
        while dropped and start < len(messages):
            dropped -= not isinstance(messages[start], _Hidden)
            start += 1
        while start < len(messages) and isinstance(messages[start], _Hidden):
            start += 1
        kept = await text(messages[start:], unknown)
        marker = (
            f"[{cut[source]} earlier messages left out to fit the context window{mark}]"
        )
        return f"{marker}\n{kept}" if kept else marker

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
                await listed(
                    "history",
                    step.history,
                    "The messages the agent has sent are not available.",
                ),
            ),
            (
                "AGENT MESSAGES",
                await listed(
                    "agent_messages",
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
                await listed(
                    "input",
                    step.input,
                    "The model input for this call is not available.",
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
            # a hidden result is its marker, after the call it answers
            result = (
                hidden_marker(1)
                if hidden.result
                else _truncation(step.truncated, mark) + await text([step.result], "")
            )
            parts.append(blocks(("RESULT", result)))
    return (
        blocks(
            (
                f"AGENT STEP - {_step_block(step)}",
                "\n\n".join(part for part in parts if part),
            )
        ),
        extract_refs,
    )


def hidden_line(count: int, nonce: str | None) -> str:
    # the marker for a run of hidden messages, in fixed wording that never
    # holds message text, with the nonce when there is one
    mark = "" if nonce is None else f" ({nonce})"
    return f"[{count} message{'' if count == 1 else 's'} hidden{mark}]"


def task_text(
    sample_input: str | Sequence[ChatMessage],
    filters: Sequence[Callable[[ChatMessage], bool]],
    nonce: str | None,
) -> str:
    # the sample's input as `Host.score` shows it: its messages' text joined
    # with newlines, each run of messages any filter matches a marker line in
    # its place. A string input is tested as the user message it becomes
    messages = (
        [ChatMessageUser(content=sample_input)]
        if isinstance(sample_input, str)
        else sample_input
    )

    def hide(message: ChatMessage) -> bool:
        return any(hides(message) for hides in filters)

    return "\n".join(
        hidden_line(len(m.hidden_ids), nonce) if isinstance(m, _Hidden) else m.text
        for m in _placed(messages, hide, False)
    )


def _mark_key(context: Context, step: Step, source: _Source, function: str) -> str:
    # one store entry per monitor, conversation, list and function, so a
    # report reads and writes only its own: (id of the last message shown, how
    # many copies of that id the list held then, its position + 1). The parts
    # are JSON-encoded, since a path or a conversation can hold any delimiter.
    return "NewSinceLastReport:" + json.dumps(
        [context.path, step.conversation, source, function]
    )


def _cut(
    step: Step,
    source: _Source,
    messages: Sequence[ChatMessage],
    window: Window,
) -> Sequence[ChatMessage]:
    # a window reads the list it is given as it is and slices off what it
    # keeps; `AllMessages` keeps it whole, and the others read from where
    # what they keep starts, except for `NewSinceLastReport` over `input` or
    # `agent_messages`, which a scaffold can rewrite, so it scans them. A
    # placeholder for hidden messages is not counted (it is not an
    # assistant message, so `LastTurns` passes it by)
    match window:
        case AllMessages():
            return messages
        case LastTurns(n):
            return messages[_turns_start(messages, n) :]
        case LastMessages(n):
            start = len(messages)
            while n and start:
                start -= 1
                n -= not isinstance(messages[start], _Hidden)
            return messages[start:]
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


class _Hidden(ChatMessageUser):
    # the placeholder for a run of hidden messages, where they stood in a
    # thinned list: windows skip it, and `render_step` renders it as the
    # nonce-marked marker. Its text is the marker without a nonce, for a
    # renderer that does not know it
    hidden_ids: tuple[str | None, ...]


def _placeholder(ids: Sequence[str | None]) -> _Hidden:
    placeholder = _Hidden(content=hidden_line(len(ids), None), hidden_ids=tuple(ids))
    # no id, so `NewSinceLastReport` never records it as the last message shown
    placeholder.id = None
    return placeholder


def _ids(*lists: list[ChatMessage] | None) -> list[str]:
    # the ids of the hidden messages the placeholders in the lists stand for
    return [
        id
        for messages in lists
        for m in messages or []
        if isinstance(m, _Hidden)
        for id in m.hidden_ids
        if id is not None
    ]


def _placed(
    messages: Sequence[ChatMessage],
    hide: Callable[[ChatMessage], bool] | None,
    omit_system: bool,
) -> list[ChatMessage]:
    # the messages with each run of hidden ones, and of placeholders for
    # some, replaced by one placeholder. `hide` is never asked about an
    # assistant message, which the agent's model wrote; a system message
    # `omit_system` leaves out is not hidden, so it leaves no marker
    placed: list[ChatMessage] = []
    run: list[str | None] = []
    for message in messages:
        if isinstance(message, _Hidden):
            run.extend(message.hidden_ids)
            continue
        if (
            hide is not None
            and not isinstance(message, ChatMessageAssistant)
            and not (omit_system and message.role == "system")
            and hide(message)
        ):
            run.append(message.id)
            continue
        if run:
            placed.append(_placeholder(run))
            run = []
        placed.append(message)
    if run:
        placed.append(_placeholder(run))
    return placed


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
    # the omissions over a list, which also leave out the system messages and,
    # under `omit_assistant_text`, an assistant message with nothing left to
    # show: no tool calls and no content
    kept: list[ChatMessage] = []
    for message in messages:
        if p.omit_system and message.role == "system":
            continue
        thinned = _omitted_from(message, p)
        if (
            p.omit_assistant_text
            and isinstance(thinned, ChatMessageAssistant)
            and not thinned.tool_calls
            and _nothing_in(thinned.content)
        ):
            continue
        kept.append(thinned)
    return kept


def _nothing_in(content: str | list[Content]) -> bool:
    # no content, or only reasoning parts that hold nothing: no reasoning, no
    # summary and not redacted
    return not content or all(
        isinstance(c, ContentReasoning)
        and not c.reasoning
        and not c.summary
        and not c.redacted
        for c in content
    )


def _omitted_output(output: ModelOutput, p: StepPreprocessor[S]) -> ModelOutput:
    # the reply is kept even when the omissions empty it, so the stage always
    # shows what the model returned. Without its text, nothing else on the
    # output may still carry it: `completion` is a stored copy of the first
    # choice's text, so it is taken from the thinned choice as inspect_ai
    # takes it, and logprobs hold the text as tokens that cannot be told
    # apart by content part, so they are left out
    text: dict[str, Any] = (
        {"logprobs": None, "prompt_logprobs": None} if p.omit_assistant_text else {}
    )
    choices = [
        choice.model_copy(update={"message": _omitted_from(choice.message, p), **text})
        for choice in output.choices
    ]
    update: dict[str, Any] = {"choices": choices}
    if p.omit_assistant_text:
        update["completion"] = choices[0].message.text if choices else ""
    return output.model_copy(update=update)


def _omitted_from(message: ChatMessage, p: StepPreprocessor[S]) -> ChatMessage:
    # one message with the omitted kinds of content taken out; a message with
    # nothing to take out is the host's own object
    text = p.omit_assistant_text and isinstance(message, ChatMessageAssistant)
    removed = tuple(
        kind
        for omit, kind in (
            (p.omit_reasoning, ContentReasoning),
            (p.omit_tool_usage, ContentToolUse),
            (text, ContentText),
        )
        if omit
    )
    if not removed:
        return message
    if isinstance(message.content, str):
        return message.model_copy(update={"content": ""}) if text else message
    return message.model_copy(
        update={"content": [c for c in message.content if not isinstance(c, removed)]}
    )


def _step_block(step: Step) -> str:
    # named for what it holds, so the model can tell whether a call has run
    if isinstance(step, BeforeToolCall):
        return "TOOL CALL REQUEST"
    if isinstance(step, AfterToolCall):
        return "TOOL CALL RESULT"
    if isinstance(step, BeforeGenerate):
        return "MODEL REQUEST"
    return "MODEL REPLY"


def _truncation(truncated: Truncation | Literal[False] | None, mark: str) -> str:
    # a line before the result when the host cut it. Inspect keeps the first
    # half of the limit and the last, joined with nothing between them, and
    # drops a character either cut splits, so the counts are upper bounds. A host
    # that cannot tell (None) gets no line: a replay from messages always
    # cannot, and a line there would change every such prompt
    if not truncated:
        return ""
    head = truncated.limit_bytes // 2
    tail = truncated.limit_bytes - head
    return (
        f"[OUTPUT TRUNCATED{mark}] The tool's output was "
        f"{truncated.original_bytes:,} bytes; the agent's model was shown at most "
        f"its first {head:,} and last {tail:,} bytes, with the middle left out.\n"
    )


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
