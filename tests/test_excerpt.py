from collections.abc import Callable
from dataclasses import replace

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store
from inspect_scout import message_numbering

from inspect_sentinel import (
    AfterGenerateExcerpt,
    AfterToolCallExcerpt,
    BeforeGenerateExcerpt,
    BeforeToolCallExcerpt,
    Excerpt,
    drop_history,
    drop_input,
    excerpt,
    last_messages,
    last_turns,
    new_since_last_report,
    step_as_str,
)

from ._fakes import (
    after_generate_step,
    after_tool_call,
    before_generate_step,
    before_tool_call,
    in_step,
)


def call(id: str) -> ToolCall:
    return ToolCall(id=id, function="bash", arguments={"cmd": "ls"})


def user(text: str) -> ChatMessage:
    return ChatMessageUser(content=text, id=text)


def asks(text: str, *calls: ToolCall) -> ChatMessage:
    return ChatMessageAssistant(content=text, id=text, tool_calls=list(calls) or None)


def result(of: str, text: str) -> ChatMessage:
    return ChatMessageTool(content=text, id=text, tool_call_id=of, function="bash")


# two turns: task, then [reply 1 + call a + result a], then [reply 2]
CONVERSATION = [
    ChatMessageSystem(content="sys", id="sys"),
    user("task"),
    asks("reply 1", call("a")),
    result("a", "result a"),
    asks("reply 2"),
]


def texts(messages: list[ChatMessage] | None) -> list[str] | None:
    return None if messages is None else [m.text for m in messages]


def test_a_tool_steps_history_ends_before_the_message_holding_its_call() -> None:
    step = replace(
        before_tool_call("bash", cmd="ls"),
        history=[*CONVERSATION, asks("calls b", call("b")), user("later")],
        call=call("b"),
    )
    cut = excerpt(step)
    assert isinstance(cut, BeforeToolCallExcerpt)
    assert texts(cut.history) == texts(CONVERSATION)


def test_after_a_tool_call_the_excerpt_keeps_the_result() -> None:
    cut = excerpt(after_tool_call("bash", "out.txt", cmd="ls"))
    assert isinstance(cut, AfterToolCallExcerpt)
    assert cut.result.text == "out.txt"


@pytest.mark.parametrize(
    "input, kept, new",
    [
        (CONVERSATION[:4], ["sys", "task", "reply 1"], ["result a"]),
        (CONVERSATION[:2], [], ["sys", "task"]),
        (CONVERSATION, texts(CONVERSATION), []),
    ],
)
def test_before_a_generate_the_new_messages_follow_the_last_reply(
    input: list[ChatMessage], kept: list[str], new: list[str]
) -> None:
    step = replace(before_generate_step(), input=input, history=input)
    cut = excerpt(step)
    assert isinstance(cut, BeforeGenerateExcerpt)
    assert (texts(cut.input), texts(cut.history), texts(cut.new)) == (kept, kept, new)


@pytest.mark.parametrize(
    "output, reply",
    [
        (ModelOutput.from_content(model="m", content="done"), "done"),
        (ModelOutput(model="m", choices=[]), ""),
    ],
)
def test_after_a_generate_the_excerpt_keeps_the_reply(
    output: ModelOutput, reply: str
) -> None:
    cut = excerpt(replace(after_generate_step(), output=output))
    assert isinstance(cut, AfterGenerateExcerpt)
    assert cut.reply.text == reply


def windowed() -> BeforeToolCallExcerpt:
    cut = excerpt(
        replace(
            before_tool_call("bash", cmd="ls"), input=CONVERSATION, history=CONVERSATION
        )
    )
    assert isinstance(cut, BeforeToolCallExcerpt)
    return cut


@pytest.mark.parametrize(
    "window, n, kept",
    [
        (last_turns, 0, []),
        (last_turns, 1, ["reply 2"]),
        # no more than n turns: the system message and task stay
        (last_turns, 2, texts(CONVERSATION)),
        (last_messages, 1, ["reply 2"]),
        # never start on a tool result
        (last_messages, 2, ["reply 2"]),
        (last_messages, 3, ["reply 1", "result a", "reply 2"]),
        (last_messages, 9, texts(CONVERSATION)),
    ],
)
def test_windows_cut_input_and_history_alike(
    window: Callable[[BeforeToolCallExcerpt, int], BeforeToolCallExcerpt],
    n: int,
    kept: list[str],
) -> None:
    cut = window(windowed(), n)
    assert (texts(cut.input), texts(cut.history)) == (kept, kept)
    assert cut.call == windowed().call


@pytest.mark.parametrize("window", [last_turns, last_messages])
def test_a_negative_window_raises(
    window: Callable[[BeforeToolCallExcerpt, int], BeforeToolCallExcerpt],
) -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        window(windowed(), -1)


def test_drops_empty_a_source_and_keep_an_unknown_input_unknown() -> None:
    assert (
        texts(drop_input(windowed()).input),
        texts(drop_history(windowed()).history),
    ) == ([], [])
    unknown = replace(windowed(), input=None)
    assert drop_input(unknown).input is None
    assert texts(last_turns(unknown, 1).input) is None


def test_new_since_last_report_shows_each_message_once_per_conversation() -> None:
    def step(conversation: str, messages: list[ChatMessage]) -> BeforeToolCallExcerpt:
        return replace(
            windowed(), conversation=conversation, input=messages, history=messages
        )

    store = Store()
    shown: list[list[str] | None] = []
    for conversation, messages in [
        ("c", CONVERSATION[:3]),
        ("c", CONVERSATION),
        ("other", CONVERSATION[:2]),
        ("c", [user("compacted"), asks("after")]),
    ]:
        with in_step("judge", store=store) as context:
            shown.append(
                texts(
                    new_since_last_report(context, step(conversation, messages)).history
                )
            )
    assert shown == [
        ["sys", "task", "reply 1"],
        ["result a", "reply 2"],
        ["sys", "task"],
        ["compacted", "after"],
    ]


@pytest.mark.anyio
async def test_step_as_str_renders_each_kept_part_in_a_block() -> None:
    rendered = await step_as_str(windowed())
    assert rendered.startswith("[BEGIN HISTORY]\n")
    assert "[END HISTORY]\n\n[BEGIN INPUT]\n" in rendered
    assert rendered.endswith("[END AGENT STEP]")
    assert "cmd: ls" in rendered.split("[BEGIN AGENT STEP]")[1]


@pytest.mark.parametrize(
    "cut, absent, present",
    [
        (drop_history(windowed()), ["HISTORY"], ["INPUT"]),
        (
            replace(windowed(), input=None),
            [],
            ["The model input for this call is not available."],
        ),
        (excerpt(before_generate_step()), ["HISTORY", "INPUT"], ["go"]),
    ],
)
@pytest.mark.anyio
async def test_step_as_str_leaves_out_empty_parts(
    cut: Excerpt, absent: list[str], present: list[str]
) -> None:
    rendered = await step_as_str(cut)
    assert not any(f"[BEGIN {name}]" in rendered for name in absent)
    assert all(text in rendered for text in present)


@pytest.mark.anyio
async def test_step_as_str_numbers_messages_across_blocks() -> None:
    numbered, extract_refs = message_numbering()
    rendered = await step_as_str(drop_input(windowed()), messages_as_str=numbered)
    assert "[M3] " in rendered
    assert "[M5] " in rendered.split("[BEGIN AGENT STEP]")[1]
    assert [r.id for r in extract_refs("see [M2]")] == ["reply 1"]
