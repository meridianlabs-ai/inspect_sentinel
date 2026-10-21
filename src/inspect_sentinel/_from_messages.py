from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar, overload

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCallView

from ._replay_step import ReplayStep
from ._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
)

T = TypeVar("T")


class Cut(Sequence[T]):
    # the first `length` items of a list that only grows, shared rather than
    # copied, so a step holding one costs nothing however long the list is
    def __init__(self, items: list[T], length: int) -> None:
        self._items = items
        self._length = length

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, slice):
            return [self._items[i] for i in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("Cut index out of range")
        return self._items[index]

    def __iter__(self) -> Iterator[T]:
        for index in range(self._length):
            yield self._items[index]

    def __repr__(self) -> str:
        return f"Cut({list(self)!r})"


def steps_from_messages(
    messages: Sequence[ChatMessage], *, conversation: str = ""
) -> Iterator[ReplayStep]:
    """The steps a sentinel would have been handed for an agent's run, rebuilt from its messages alone.

    Every source holds a message list, so this works for any transcript, a proxy's request or a list written by hand for a monitor's unit test. It is the projection a proxy makes: each assistant message is a generate whose input is every message before it, each of its tool calls a `tool_call` step, and the tool message answering a call, the first after the assistant message with that call's id, its `tool_result` step. A call with no answering message has no `tool_result` step. Steps come in the order they ran: a generate's two stages, then each of its calls' two stages in the order the message lists them.

    What messages do not hold is absent:

    - At the generate stages, `tools` is empty, `tool_choice` is `"auto"` and `config` is the default; `model` is the assistant message's `model`, or empty. The output has no usage or time.
    - At the tool stages, `input`, `tools` and `tool_choice` are `None`, since a compaction makes the messages before a call differ from what the model was sent. `view` holds the call's own rendering where it has one, `output` is `None` and `truncated` is `None`.
    - `history` is the messages up to the generate's input (before it, at `model_input`), as if every message had been sent once. `agent_messages` is the messages before the first assistant message, then each assistant and tool message so far, as if the list were one agent's run: an assistant message joins after its `model_output` step, and its calls' results after its last call's steps. `human_decisions` is `None`.
    - Every step is in one segment, `(conversation, 0)`, since messages do not say where a compaction was.

    A generate's step id is its assistant message's id, or its position (`m3`) when it has none; a tool step's is its call's id.

    Args:
        messages: The conversation, oldest first.
        conversation: The id to give the conversation, `step.conversation` on every step.
    """
    sent: list[ChatMessage] = list(messages)
    ran: list[ChatMessage] = []
    answers = _answers(sent)
    segment = (conversation, 0)
    opened = False
    previous_input = 0
    for index, message in enumerate(sent):
        if not isinstance(message, ChatMessageAssistant):
            if not opened:
                ran.append(message)
            continue
        opened = True
        step_id = message.id or f"m{index}"
        before = BeforeGenerate(
            model=message.model or "",
            conversation=conversation,
            input=list(sent[:index]),
            agent_messages=Cut(ran, len(ran)),
            history=Cut(sent, previous_input),
            human_decisions=None,
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
        )
        yield ReplayStep(step_id, "model_input", before, segment)
        output = ModelOutput(
            model=message.model or "",
            choices=[
                ChatCompletionChoice(
                    message=message,
                    stop_reason="tool_calls" if message.tool_calls else "stop",
                )
            ],
        )
        yield ReplayStep(
            step_id,
            "model_output",
            AfterGenerate(
                model=before.model,
                conversation=conversation,
                input=before.input,
                agent_messages=before.agent_messages,
                history=Cut(sent, index),
                human_decisions=None,
                tools=before.tools,
                tool_choice=before.tool_choice,
                config=before.config,
                output=output,
            ),
            segment,
        )
        ran.append(message)
        history = Cut(sent, index)
        agent_messages = Cut(ran, len(ran))
        results: list[ChatMessageTool] = []
        for call in message.tool_calls or []:
            view = ToolCallView(call=call.view)
            yield ReplayStep(
                call.id,
                "tool_call",
                BeforeToolCall(
                    conversation=conversation,
                    message=message.text,
                    message_id=message.id or step_id,
                    call=call,
                    view=view,
                    input=None,
                    agent_messages=agent_messages,
                    history=history,
                    human_decisions=None,
                ),
                segment,
            )
            result = _answer(answers, call.id, index)
            if result is None:
                continue
            results.append(result)
            yield ReplayStep(
                call.id,
                "tool_result",
                AfterToolCall(
                    conversation=conversation,
                    message=message.text,
                    message_id=message.id or step_id,
                    call=call,
                    result=result,
                    output=None,
                    view=view,
                    input=None,
                    agent_messages=agent_messages,
                    history=history,
                    human_decisions=None,
                    truncated=None,
                ),
                segment,
            )
        ran.extend(results)
        previous_input = index


def _answers(
    messages: list[ChatMessage],
) -> dict[str, list[tuple[int, ChatMessageTool]]]:
    # each call id's tool messages by position, found in one pass
    found: dict[str, list[tuple[int, ChatMessageTool]]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, ChatMessageTool) and message.tool_call_id is not None:
            found.setdefault(message.tool_call_id, []).append((index, message))
    return found


def _answer(
    answers: dict[str, list[tuple[int, ChatMessageTool]]], call_id: str, after: int
) -> ChatMessageTool | None:
    # the first answer after the assistant message; a provider can reuse ids
    for index, message in answers.get(call_id, []):
        if index > after:
            return message
    return None
