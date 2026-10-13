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
    BeforeToolCall,
    Excerpt,
    Step,
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
    return ToolCall(id=id, function="bash", arguments={"cmd": id})


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


def tool_step(
    input: list[ChatMessage] | None, history: list[ChatMessage]
) -> BeforeToolCall:
    return replace(before_tool_call("bash", cmd="ls"), input=input, history=history)


def windowed() -> Excerpt[BeforeToolCall]:
    return excerpt(tool_step(CONVERSATION, CONVERSATION))


def parallel_after_b() -> Step:
    # one message holds calls a and b; a has run, b is the step
    return replace(
        after_tool_call("bash", "result b", cmd="b"),
        call=call("b"),
        message="both",
        history=[
            user("task"),
            asks("both", call("a"), call("b")),
            result("a", "result a"),
        ],
    )


@pytest.mark.parametrize(
    "step, history",
    [
        (
            replace(
                tool_step(
                    [], [*CONVERSATION, asks("calls b", call("b")), user("later")]
                ),
                call=call("b"),
            ),
            texts(CONVERSATION),
        ),
        (parallel_after_b(), ["task"]),
        # the call is not in history: history is kept whole
        (tool_step([], CONVERSATION), texts(CONVERSATION)),
    ],
)
def test_a_tool_steps_history_ends_before_the_message_holding_its_call(
    step: Step, history: list[str]
) -> None:
    assert texts(excerpt(step).history) == history


REPLIED = ["sys", "task", "reply 1"]


@pytest.mark.parametrize(
    "input, history, kept_input, kept_history",
    [
        (CONVERSATION[:4], CONVERSATION[:4], REPLIED, REPLIED),
        (CONVERSATION[:2], CONVERSATION[:2], [], []),
        (CONVERSATION, CONVERSATION, texts(CONVERSATION), texts(CONVERSATION)),
        # each is cut at its own last reply
        (CONVERSATION[:4], CONVERSATION[1:2], REPLIED, []),
    ],
)
def test_before_a_generate_conversations_end_at_the_last_reply(
    input: list[ChatMessage],
    history: list[ChatMessage],
    kept_input: list[str],
    kept_history: list[str],
) -> None:
    cut = excerpt(replace(before_generate_step(), input=input, history=history))
    assert (texts(cut.input), texts(cut.history)) == (kept_input, kept_history)


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
    window: Callable[[Excerpt[BeforeToolCall], int], Excerpt[BeforeToolCall]],
    n: int,
    kept: list[str],
) -> None:
    cut = window(windowed(), n)
    assert (texts(cut.input), texts(cut.history)) == (kept, kept)
    assert cut.step == windowed().step


@pytest.mark.parametrize("window", [last_turns, last_messages])
def test_a_negative_window_raises(
    window: Callable[[Excerpt[BeforeToolCall], int], Excerpt[BeforeToolCall]],
) -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        window(windowed(), -1)


def test_drops_empty_a_source_and_unknown_input_stays_unknown() -> None:
    assert texts(drop_input(windowed()).input) == []
    assert texts(drop_history(windowed()).history) == []
    unknown = excerpt(tool_step(None, CONVERSATION))
    assert drop_input(unknown).input is None
    assert last_turns(unknown, 1).input is None


def test_new_since_last_report_shows_each_message_once_per_conversation() -> None:
    store = Store()
    shown: list[tuple[list[str] | None, list[str] | None]] = []
    for conversation, input, history in [
        ("c", CONVERSATION[:3], CONVERSATION[:3]),
        ("c", CONVERSATION, CONVERSATION),
        ("other", None, CONVERSATION[:2]),
        ("c", CONVERSATION, [user("compacted"), asks("after")]),
    ]:
        step = replace(tool_step(input, history), conversation=conversation)
        with in_step("judge", store=store) as context:
            cut = new_since_last_report(context, excerpt(step))
        shown.append((texts(cut.input), texts(cut.history)))
    assert shown == [
        (REPLIED, REPLIED),
        (["result a", "reply 2"], ["result a", "reply 2"]),
        (None, ["sys", "task"]),
        ([], ["compacted", "after"]),
    ]


def test_new_since_last_report_shows_messages_without_ids() -> None:
    unnamed = ChatMessageUser(content="no id")
    unnamed.id = None
    with in_step("judge", store=Store()) as context:
        cut = new_since_last_report(context, excerpt(tool_step(None, [unnamed])))
    assert texts(cut.history) == ["no id"]


@pytest.mark.parametrize(
    "cut, absent, shown",
    [
        (windowed(), [], ["cmd: ls"]),
        (drop_history(windowed()), ["HISTORY"], ["cmd: ls"]),
        (
            excerpt(after_tool_call("bash", "out.txt", cmd="ls")),
            ["HISTORY"],
            ["cmd: ls", "out.txt"],
        ),
        (excerpt(before_generate_step()), ["HISTORY", "INPUT"], ["go"]),
        (excerpt(after_generate_step()), [], ["ASSISTANT:\ndone"]),
        (
            excerpt(
                replace(
                    after_generate_step(), output=ModelOutput(model="m", choices=[])
                )
            ),
            [],
            ["ASSISTANT:"],
        ),
    ],
)
@pytest.mark.anyio
async def test_step_as_str_renders_each_kept_part_in_a_block(
    cut: Excerpt[Step], absent: list[str], shown: list[str]
) -> None:
    rendered = await step_as_str(cut)
    assert not any(f"[BEGIN {name}]" in rendered for name in absent)
    assert rendered.endswith("[END AGENT STEP]")
    step_block = rendered.split("[BEGIN AGENT STEP]")[1]
    assert all(text in step_block for text in shown)


@pytest.mark.anyio
async def test_step_as_str_says_when_the_model_input_is_unknown() -> None:
    rendered = await step_as_str(excerpt(tool_step(None, [])))
    assert (
        "[BEGIN INPUT]\nThe model input for this call is not available.\n[END INPUT]"
        in rendered
    )


@pytest.mark.anyio
async def test_numbered_citations_resolve_to_the_agents_messages() -> None:
    numbered, extract_refs = message_numbering()
    rendered = await step_as_str(excerpt(parallel_after_b()), messages_as_str=numbered)
    step_block = rendered.split("[BEGIN AGENT STEP]")[1]
    assert "cmd: b" in step_block and "cmd: a" not in step_block
    # history's task is [M1]; the step's message holding the call is [M2]
    assert [r.id for r in extract_refs("see [M1] and [M2]")] == ["task", "both"]
