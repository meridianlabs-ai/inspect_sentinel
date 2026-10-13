import json
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
    AfterGenerate,
    BeforeToolCall,
    Context,
    Excerpt,
    Step,
    drop_history,
    drop_input,
    last_messages,
    last_turns,
    new_since_last_report,
    step_as_str,
)
from inspect_sentinel._decorators import monitor
from inspect_sentinel._excerpt import _excerpt
from inspect_sentinel._report import Observation
from inspect_sentinel._runner import run_monitors
from inspect_sentinel._types import MonitorGroup

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


def unnamed(text: str) -> ChatMessage:
    message = ChatMessageUser(content=text)
    message.id = None
    return message


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


def windowed() -> BeforeToolCall:
    return tool_step(CONVERSATION, CONVERSATION)


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
    assert texts(_excerpt(step).history) == history


REPLIED = ["sys", "task", "reply 1"]

GO = [user("go")]


@pytest.mark.parametrize(
    "input, history, kept_input, kept_history",
    [
        (CONVERSATION[:4], CONVERSATION[:4], REPLIED, REPLIED),
        (CONVERSATION[:2], CONVERSATION[:2], [], []),
        (CONVERSATION, CONVERSATION, texts(CONVERSATION), texts(CONVERSATION)),
        # history loses only the step's own messages, by id
        (CONVERSATION[:4], CONVERSATION[1:2], REPLIED, ["task"]),
        ([user("prepared")], [user("original")], [], ["original"]),
        # messages without ids are never taken for the request's
        ([unnamed("prepared")], [unnamed("original")], [], ["original"]),
    ],
)
def test_before_a_generate_conversations_end_at_the_last_reply(
    input: list[ChatMessage],
    history: list[ChatMessage],
    kept_input: list[str],
    kept_history: list[str],
) -> None:
    cut = _excerpt(replace(before_generate_step(), input=input, history=history))
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
    window: Callable[
        [BeforeToolCall | Excerpt[BeforeToolCall], int], Excerpt[BeforeToolCall]
    ],
    n: int,
    kept: list[str],
) -> None:
    cut = window(windowed(), n)
    assert (texts(cut.input), texts(cut.history)) == (kept, kept)
    assert cut.step == windowed()
    # a window narrows an excerpt as well as a step
    assert window(drop_input(windowed()), n).history == cut.history


def test_last_messages_keeps_a_result_it_did_not_part_from_its_call() -> None:
    # as new_since_last_report leaves it when the call was already shown
    started = Excerpt(windowed(), None, CONVERSATION[3:])
    assert texts(last_messages(started, 9).history) == ["result a", "reply 2"]


@pytest.mark.parametrize("window", [last_turns, last_messages])
def test_a_negative_window_raises(
    window: Callable[
        [BeforeToolCall | Excerpt[BeforeToolCall], int], Excerpt[BeforeToolCall]
    ],
) -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        window(windowed(), -1)


def test_drops_empty_a_source_and_unknown_input_stays_unknown() -> None:
    assert texts(drop_input(windowed()).input) == []
    assert texts(drop_history(windowed()).history) == []
    unknown = _excerpt(tool_step(None, CONVERSATION))
    assert drop_input(unknown).input is None
    assert last_turns(unknown, 1).input is None


Shown = tuple[list[str] | None, list[str] | None]


def report(cut: Excerpt[Step]) -> Observation:
    return Observation.score(0.0, json.dumps([texts(cut.input), texts(cut.history)]))


@monitor
def incremental() -> MonitorGroup:
    async def calls(context: Context, step: BeforeToolCall) -> Observation | None:
        cut = new_since_last_report(context, step)
        if step.message == "fail":
            raise ValueError("model unavailable")
        return report(cut)

    async def also_calls(context: Context, step: BeforeToolCall) -> Observation | None:
        return report(new_since_last_report(context, step))

    async def replies(context: Context, step: AfterGenerate) -> Observation | None:
        return report(new_since_last_report(context, step))

    return MonitorGroup(calls, also_calls, replies)


async def shown(store: Store, steps: list[Step]) -> list[list[Shown]]:
    # what each function of the monitor was shown at each step
    seen: list[list[Shown]] = []
    for step in steps:
        with in_step(store=store) as context:
            observations = await run_monitors({"judge": incremental()}, context, step)
        seen.append(
            [
                tuple(json.loads(o.report.explanation or ""))
                for o in observations.succeeded
            ]
        )
    return seen


def shared(input: list[str] | None, history: list[str] | None) -> list[Shown]:
    # both tool-stage functions were shown the same
    return [(input, history)] * 2


@pytest.mark.anyio
async def test_new_since_last_report_shows_each_message_once_per_function() -> None:
    steps: list[Step] = [
        replace(tool_step(input, history), conversation=conversation)
        for conversation, input, history in [
            ("c", CONVERSATION[:3], CONVERSATION[:3]),
            ("c", CONVERSATION, CONVERSATION),
            ("other", None, CONVERSATION[:2]),
            ("c", CONVERSATION, [user("compacted"), asks("after")]),
        ]
    ]
    assert await shown(Store(), steps) == [
        shared(REPLIED, REPLIED),
        shared(["result a", "reply 2"], ["result a", "reply 2"]),
        shared(None, ["sys", "task"]),
        shared([], ["compacted", "after"]),
    ]


@pytest.mark.anyio
async def test_new_since_last_report_shows_again_what_a_failed_function_was_shown() -> (
    None
):
    steps: list[Step] = [
        replace(tool_step(None, CONVERSATION[:3]), message="fail"),
        tool_step(None, CONVERSATION),
    ]
    assert await shown(Store(), steps) == [
        [(None, REPLIED)],
        [(None, texts(CONVERSATION)), (None, ["result a", "reply 2"])],
    ]


@pytest.mark.anyio
async def test_new_since_last_report_records_the_last_message_with_an_id() -> None:
    histories: list[list[ChatMessage]] = [
        [unnamed("no id"), user("task"), unnamed("no id")],
        [unnamed("no id"), user("task"), unnamed("no id"), asks("later")],
        [unnamed("no id"), user("task"), unnamed("no id"), asks("later")],
    ]
    assert await shown(Store(), [tool_step(None, h) for h in histories]) == [
        shared(None, ["no id", "task", "no id"]),
        shared(None, ["no id", "later"]),
        shared(None, []),
    ]


@pytest.mark.anyio
async def test_new_since_last_report_records_per_function_so_per_stage() -> None:
    generated = replace(
        after_generate_step(),
        conversation="c",
        input=CONVERSATION[:2],
        history=CONVERSATION[:2],
    )
    called = replace(tool_step(None, CONVERSATION), conversation="c")
    assert await shown(Store(), [generated, called]) == [
        [(texts(CONVERSATION[:2]), texts(CONVERSATION[:2]))],
        shared(None, texts(CONVERSATION)),
    ]


def test_new_since_last_report_outside_the_runner_raises() -> None:
    with in_step() as context, pytest.raises(RuntimeError, match="runner"):
        new_since_last_report(context, windowed())


@pytest.mark.parametrize(
    "cut, absent, shown",
    [
        (windowed(), [], ["cmd: ls"]),
        (drop_history(windowed()), ["HISTORY"], ["cmd: ls"]),
        (
            _excerpt(after_tool_call("bash", "out.txt", cmd="ls")),
            ["HISTORY"],
            ["cmd: ls", "out.txt"],
        ),
        (
            _excerpt(replace(before_generate_step(), input=GO, history=GO)),
            ["HISTORY", "INPUT"],
            ["go"],
        ),
        (_excerpt(after_generate_step()), [], ["ASSISTANT:\ndone"]),
        (
            _excerpt(
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
async def test_a_request_ending_in_a_prefill_has_no_step_block() -> None:
    prefilled = [user("task"), asks("prefill")]
    rendered = await step_as_str(
        _excerpt(replace(before_generate_step(), input=prefilled, history=prefilled))
    )
    assert "[BEGIN AGENT STEP]" not in rendered
    assert "prefill" in rendered.split("[BEGIN INPUT]")[1]


@pytest.mark.anyio
async def test_step_as_str_says_when_the_model_input_is_unknown() -> None:
    rendered = await step_as_str(_excerpt(tool_step(None, [])))
    assert (
        "[BEGIN INPUT]\nThe model input for this call is not available.\n[END INPUT]"
        in rendered
    )


@pytest.mark.anyio
async def test_numbered_citations_resolve_to_the_agents_messages() -> None:
    numbered, extract_refs = message_numbering()
    rendered = await step_as_str(_excerpt(parallel_after_b()), messages_as_str=numbered)
    step_block = rendered.split("[BEGIN AGENT STEP]")[1]
    assert "cmd: b" in step_block and "cmd: a" not in step_block
    # history's task is [M1]; the step's message holding the call is [M2]
    assert [r.id for r in extract_refs("see [M1] and [M2]")] == ["task", "both"]
