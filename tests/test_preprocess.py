import json
from dataclasses import replace

import anyio
import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
    ContentToolUse,
    ModelOutput,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store

from inspect_sentinel import (
    AfterGenerate,
    AllMessages,
    BeforeToolCall,
    Context,
    LastMessages,
    LastTurns,
    NewSinceLastReport,
    Step,
    StepPreprocessor,
    Window,
    concurrent,
    decide_final,
    preprocess_step,
    step_as_str,
)
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._preprocess import _before_own_part, scout_nonce
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._runner import run_monitors, run_protocols, run_sentinel
from inspect_sentinel._types import Monitor, MonitorGroup, Protocol

from ._fakes import (
    ListRecorder,
    after_generate_step,
    after_tool_call,
    before_generate_step,
    before_tool_call,
    host_context,
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

# shows every message the windows keep, so a test sees the cut alone
ALL = StepPreprocessor(
    input_window=AllMessages(), history_window=AllMessages(), omit_system=False
)

# the whole input and no history: the default
WHOLE = StepPreprocessor()


def texts(messages: list[ChatMessage] | None) -> list[str] | None:
    return None if messages is None else [m.text for m in messages]


def tool_step(
    input: list[ChatMessage] | None, history: list[ChatMessage] | None
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


Shown = tuple[list[str] | None, list[str] | None]


def shown(step: Step) -> Shown:
    # each conversation as step_as_str shows it, before the step's own part
    return (
        None if step.input is None else texts(_before_own_part(step, "input")),
        None if step.history is None else texts(_before_own_part(step, "history")),
    )


async def render(
    step: Step,
    preprocessor: StepPreprocessor[Step] | None = None,
    *,
    nonce: str | None = None,
) -> str:
    p = preprocessor or WHOLE
    text, _ = await step_as_str(
        await preprocess_step(step, p),
        nonce=nonce,
        allow_placeholders=p.allow_placeholders,
    )
    return text


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
@pytest.mark.anyio
async def test_a_tool_steps_history_ends_before_the_message_holding_its_call(
    step: Step, history: list[str]
) -> None:
    assert shown(await preprocess_step(step, ALL))[1] == history


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
@pytest.mark.anyio
async def test_before_a_generate_conversations_end_at_the_last_reply(
    input: list[ChatMessage],
    history: list[ChatMessage],
    kept_input: list[str],
    kept_history: list[str],
) -> None:
    step = replace(before_generate_step(), input=input, history=history)
    assert shown(await preprocess_step(step, ALL)) == (kept_input, kept_history)


@pytest.mark.parametrize(
    "window, kept",
    [
        (LastTurns(0), []),
        (LastTurns(1), ["reply 2"]),
        # no more than n turns: the system message and task stay
        (LastTurns(2), texts(CONVERSATION)),
        (LastMessages(1), ["reply 2"]),
        # never start on a tool result
        (LastMessages(2), ["reply 2"]),
        (LastMessages(3), ["reply 1", "result a", "reply 2"]),
        (LastMessages(9), texts(CONVERSATION)),
    ],
)
@pytest.mark.anyio
async def test_windows_cut_input_and_history_each_as_asked(
    window: Window, kept: list[str]
) -> None:
    both = replace(ALL, input_window=window, history_window=window)
    assert shown(await preprocess_step(windowed(), both)) == (kept, kept)
    # each conversation has its own window
    input_only = replace(ALL, input_window=window)
    assert shown(await preprocess_step(windowed(), input_only)) == (
        kept,
        texts(CONVERSATION),
    )


@pytest.mark.parametrize("window", [LastTurns, LastMessages])
def test_a_negative_window_raises(window: type[LastTurns | LastMessages]) -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        window(-1)


UNKNOWN_INPUT = (
    "[BEGIN INPUT]\nThe model input for this call is not available.\n[END INPUT]"
)


@pytest.mark.parametrize(
    "preprocessor, input, says_unknown",
    [
        (StepPreprocessor(input_window=None), CONVERSATION, False),
        # an unknown input is still named, without a window or windowed
        (StepPreprocessor(input_window=None), None, True),
        (StepPreprocessor(input_window=LastTurns(1)), None, True),
    ],
)
@pytest.mark.anyio
async def test_no_input_window_leaves_it_out_and_an_unknown_input_stays_named(
    preprocessor: StepPreprocessor[Step],
    input: list[ChatMessage] | None,
    says_unknown: bool,
) -> None:
    rendered = await render(tool_step(input, CONVERSATION), preprocessor)
    assert (UNKNOWN_INPUT in rendered) == says_unknown
    assert ("[BEGIN INPUT]" in rendered) == says_unknown


@pytest.mark.anyio
async def test_leaving_out_a_requests_input_keeps_the_request() -> None:
    step = replace(
        before_generate_step(), input=[*CONVERSATION, user("go")], history=GO
    )
    rendered = await render(step, StepPreprocessor(input_window=None))
    assert "[BEGIN INPUT]" not in rendered
    assert "go" in rendered.split("[AGENT STEP - BEGIN MODEL REQUEST]")[1]


def new_since(context: Context) -> StepPreprocessor[Step]:
    return replace(
        ALL,
        input_window=NewSinceLastReport(context),
        history_window=NewSinceLastReport(context),
    )


def report(step: Step) -> Observation:
    return Observation.score(0.0, json.dumps(shown(step)))


@monitor
def incremental() -> MonitorGroup:
    async def calls(context: Context, step: BeforeToolCall) -> Observation | None:
        thinned = await preprocess_step(step, new_since(context))
        if step.message == "fail":
            raise ValueError("model unavailable")
        return report(thinned)

    async def also_calls(context: Context, step: BeforeToolCall) -> Observation | None:
        return report(await preprocess_step(step, new_since(context)))

    async def replies(context: Context, step: AfterGenerate) -> Observation | None:
        return report(await preprocess_step(step, new_since(context)))

    return MonitorGroup(calls, also_calls, replies)


async def seen_by(store: Store, steps: list[Step]) -> list[list[Shown]]:
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
    assert await seen_by(Store(), steps) == [
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
    assert await seen_by(Store(), steps) == [
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
    assert await seen_by(Store(), [tool_step(None, h) for h in histories]) == [
        shared(None, ["no id", "task", "no id"]),
        shared(None, ["no id", "later"]),
        shared(None, []),
    ]


@pytest.mark.anyio
async def test_new_since_last_report_records_per_function() -> None:
    generated = replace(
        after_generate_step(),
        conversation="c",
        input=CONVERSATION[:2],
        history=CONVERSATION[:2],
    )
    called = replace(tool_step(None, CONVERSATION), conversation="c")
    assert await seen_by(Store(), [generated, called]) == [
        [(texts(CONVERSATION[:2]), texts(CONVERSATION[:2]))],
        shared(None, texts(CONVERSATION)),
    ]


@pytest.mark.anyio
async def test_new_since_last_report_is_not_recorded_for_a_cancelled_function() -> None:
    started = anyio.Event()

    @monitor
    def watcher(stall: bool = False) -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            thinned = await preprocess_step(step, new_since(context))
            if stall:
                started.set()
                await anyio.sleep_forever()
            return report(thinned)

        return check

    @protocol
    def stops() -> Protocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            await started.wait()
            return Decision.terminate()

        return decide

    store, recorder = Store(), ListRecorder()
    with anyio.fail_after(5):
        await run_sentinel(
            concurrent({"watch": watcher(stall=True), "stops": stops()}),
            host_context(store=store),
            tool_step(None, CONVERSATION[:3]),
        )
        await run_sentinel(
            concurrent({"watch": watcher()}),
            host_context(store=store, recorder=recorder),
            tool_step(None, CONVERSATION),
        )
    [seen] = [r.reported.report.explanation for r in recorder.records]
    assert json.loads(seen or "") == [None, texts(CONVERSATION)]


@pytest.mark.parametrize("in_task", [False, True])
@pytest.mark.anyio
async def test_new_since_last_report_is_recorded_for_a_final_decision(
    in_task: bool,
) -> None:
    @protocol
    def decides() -> Protocol:
        async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
            thinned = await preprocess_step(step, new_since(context))
            final = Decision.proceed(json.dumps(shown(thinned)))
            if in_task:
                async with anyio.create_task_group() as group:
                    group.start_soon(_decide_final, final)
            decide_final(final)

        return decide

    store = Store()
    seen: list[str | None] = []
    for history in [CONVERSATION[:3], CONVERSATION]:
        decision = await run_sentinel(
            decides(), host_context(store=store), tool_step(None, history)
        )
        assert decision is not None
        seen.append(decision.explanation)
    assert [json.loads(e or "") for e in seen] == [
        [None, REPLIED],
        [None, ["result a", "reply 2"]],
    ]


async def _decide_final(decision: Decision) -> None:
    decide_final(decision)


@protocol
def ends_the_step() -> Protocol:
    async def end(context: Context, step: BeforeToolCall) -> Decision | None:
        decide_final(Decision.proceed())

    return end


@pytest.mark.anyio
async def test_new_since_last_report_is_not_recorded_when_a_childs_final_passes_by() -> (
    None
):
    seen: list[list[str]] = []

    @protocol
    def passed_by() -> Protocol:
        async def judge(context: Context, step: BeforeToolCall) -> Decision | None:
            thinned = await preprocess_step(step, new_since(context))
            seen.append(shown(thinned)[1] or [])
            await run_protocols(ends_the_step(), context, step)
            return Decision.proceed()

        return judge

    store = Store()
    for _ in range(2):
        await run_sentinel(
            passed_by(), host_context(store=store), tool_step(None, CONVERSATION)
        )
    # it never judged the history, so it is shown again
    assert seen == [texts(CONVERSATION)] * 2


@pytest.mark.anyio
async def test_new_since_last_report_outside_the_runner_raises() -> None:
    with in_step() as context, pytest.raises(RuntimeError, match="runner"):
        await preprocess_step(windowed(), new_since(context))


@pytest.mark.parametrize(
    "step, preprocessor, absent, block, contents",
    [
        (windowed(), None, [], "TOOL CALL", ["cmd: ls"]),
        # the same conversation as the input, so not shown even when kept
        (windowed(), ALL, ["HISTORY"], "TOOL CALL", ["cmd: ls"]),
        (
            after_tool_call("bash", "out.txt", cmd="ls"),
            None,
            ["HISTORY"],
            "TOOL CALL AND RESULT",
            ["cmd: ls", "out.txt"],
        ),
        (
            replace(before_generate_step(), input=GO, history=GO),
            None,
            ["HISTORY", "INPUT"],
            "MODEL REQUEST",
            ["go"],
        ),
        (after_generate_step(), None, [], "MODEL REPLY", ["ASSISTANT:\ndone"]),
        (
            replace(after_generate_step(), output=ModelOutput(model="m", choices=[])),
            None,
            [],
            "MODEL REPLY",
            ["ASSISTANT:"],
        ),
    ],
)
@pytest.mark.anyio
async def test_step_as_str_renders_each_kept_part_in_a_block(
    step: Step,
    preprocessor: StepPreprocessor[Step] | None,
    absent: list[str],
    block: str,
    contents: list[str],
) -> None:
    rendered = await render(step, preprocessor)
    assert not any(f"[BEGIN {name}]" in rendered for name in absent)
    # the step's own part comes last, in a block named for its stage
    assert rendered.endswith(f"[AGENT STEP - END {block}]")
    step_block = rendered.split(f"[AGENT STEP - BEGIN {block}]")[1]
    assert all(text in step_block for text in contents)


@pytest.mark.anyio
async def test_a_request_ending_in_a_prefill_has_no_step_block() -> None:
    prefilled = [user("task"), asks("prefill")]
    rendered = await render(
        replace(before_generate_step(), input=prefilled, history=prefilled)
    )
    assert "[AGENT STEP - BEGIN MODEL REQUEST]" not in rendered
    assert "[BEGIN HISTORY]" not in rendered
    assert "prefill" in rendered.split("[BEGIN INPUT]")[1]


def thinks(text: str, id: str) -> ChatMessageAssistant:
    return ChatMessageAssistant(
        id=id,
        content=[
            ContentReasoning(reasoning=f"thinking about {text}"),
            ContentToolUse(
                tool_type="web_search",
                id="w",
                name="search",
                arguments="{}",
                result=f"searched {text}",
            ),
            ContentText(text=text),
        ],
    )


def thoughtful() -> Step:
    # reasoning and a provider's tool use in history and in the reply itself
    conversation: list[ChatMessage] = [
        ChatMessageSystem(content="be helpful", id="sys"),
        user("task"),
        asks("first"),
        user("more"),
        thinks("earlier", "earlier"),
        user("go on"),
    ]
    return replace(
        after_generate_step(),
        input=conversation,
        history=conversation,
        output=ModelOutput(
            model="m",
            choices=[ChatCompletionChoice(message=thinks("reply", "reply"))],
        ),
    )


@pytest.mark.parametrize(
    "preprocessor, gone, kept",
    [
        # system messages are left out by default, as Scout leaves them out
        (WHOLE, ["be helpful"], ["thinking about", "searched"]),
        (
            replace(WHOLE, omit_system=False),
            [],
            ["be helpful", "thinking about", "searched"],
        ),
        (
            replace(WHOLE, omit_reasoning=True),
            ["be helpful", "thinking about"],
            ["searched"],
        ),
        (
            replace(WHOLE, omit_tool_usage=True),
            ["be helpful", "searched"],
            ["thinking about"],
        ),
    ],
)
@pytest.mark.anyio
async def test_omissions_leave_content_out_of_every_block(
    preprocessor: StepPreprocessor[Step], gone: list[str], kept: list[str]
) -> None:
    rendered = await render(thoughtful(), preprocessor)
    step_block = rendered.split("[AGENT STEP - BEGIN MODEL REPLY]")[1]
    assert not any(text in rendered for text in gone)
    assert all(text in rendered for text in kept)
    # the reply's own reasoning goes too, its text stays
    assert ("thinking about reply" in step_block) == (not preprocessor.omit_reasoning)
    assert "reply" in step_block


@pytest.mark.anyio
async def test_omissions_combine_with_windows_and_keep_cites() -> None:
    preprocessor = StepPreprocessor(input_window=LastTurns(1), omit_reasoning=True)
    rendered, extract_refs = await step_as_str(
        await preprocess_step(thoughtful(), preprocessor)
    )
    assert "be helpful" not in rendered and "thinking about" not in rendered
    assert [r.id for r in extract_refs("see [M1]")] == ["earlier"]


@pytest.mark.anyio
async def test_transform_runs_last_on_the_thinned_step() -> None:
    seen: list[Shown] = []

    async def withhold(step: Step) -> Step:
        seen.append(shown(step))
        assert step.input is not None
        return replace(step, input=[m for m in step.input if m.text != "go on"])

    preprocessor = StepPreprocessor(input_window=LastTurns(1), transform=withhold)
    rendered = await render(thoughtful(), preprocessor)
    # the transform saw the window's cut, and no history
    assert seen == [(["earlier", "go on"], [])]
    assert "go on" not in rendered and "searched earlier" in rendered


@pytest.mark.parametrize("allow_placeholders", [False, True])
@pytest.mark.anyio
async def test_content_without_text_needs_placeholders_allowed(
    allow_placeholders: bool,
) -> None:
    pictured = ChatMessageUser(
        content=[
            ContentText(text="look"),
            ContentImage(image="data:image/png;base64,"),
        ],
        id="pictured",
    )
    step = tool_step([pictured], [pictured])
    preprocessor = replace(WHOLE, allow_placeholders=allow_placeholders)
    if allow_placeholders:
        assert "<image />" in await render(step, preprocessor)
    else:
        with pytest.raises(ValueError, match="image"):
            await render(step, preprocessor)


def merged() -> ChatMessage:
    # what a provider's preparation makes of two consecutive user messages
    return ChatMessageUser(
        content="task\nmore", id="merged", metadata={"combined_from": ["task", "more"]}
    )


KEEP_HISTORY = replace(WHOLE, history_window=AllMessages())


@pytest.mark.parametrize(
    "preprocessor, input, history, shows_history",
    [
        # left out without a window
        (WHOLE, [user("summary")], CONVERSATION, False),
        (WHOLE, None, CONVERSATION, False),
        # kept: the same conversation is shown once
        (KEEP_HISTORY, CONVERSATION, CONVERSATION, False),
        # kept: turns before a compaction are only in history, so all of it shows
        (KEEP_HISTORY, [user("summary")], CONVERSATION, True),
        (KEEP_HISTORY, [merged()], [user("task"), user("more")], False),
        (KEEP_HISTORY, CONVERSATION, [unnamed("no id")], True),
        (replace(KEEP_HISTORY, input_window=None), CONVERSATION, CONVERSATION, True),
        (KEEP_HISTORY, None, CONVERSATION, True),
        # only hidden messages are extra
        (KEEP_HISTORY, CONVERSATION[1:], CONVERSATION, False),
        (ALL, CONVERSATION[1:], CONVERSATION, True),
    ],
)
@pytest.mark.anyio
async def test_history_is_kept_by_choice_and_shown_when_it_adds_to_the_input(
    preprocessor: StepPreprocessor[Step],
    input: list[ChatMessage] | None,
    history: list[ChatMessage],
    shows_history: bool,
) -> None:
    rendered = await render(tool_step(input, history), preprocessor)
    assert ("[BEGIN HISTORY]" in rendered) == shows_history
    if shows_history:
        # the whole history, messages the input also has included
        assert all(
            m.text in rendered.split("[END HISTORY]")[0]
            for m in history
            if m.role != "system" or not preprocessor.omit_system
        )


@pytest.mark.anyio
async def test_blocks_come_input_then_history_then_the_step_numbered_in_that_order() -> (
    None
):
    step = replace(
        tool_step([user("summary")], CONVERSATION), call=call("b"), message="calls b"
    )
    rendered, extract_refs = await step_as_str(
        await preprocess_step(step, KEEP_HISTORY)
    )
    starts = [
        rendered.index(marker)
        for marker in (
            "[BEGIN INPUT]",
            "[BEGIN HISTORY]",
            "[AGENT STEP - BEGIN TOOL CALL]",
        )
    ]
    assert starts == sorted(starts)
    assert [r.id for r in extract_refs("see [M1]")] == ["summary"]


UNKNOWN_HISTORY = (
    "[BEGIN HISTORY]\nThe agent's conversation history is not available.\n[END HISTORY]"
)

WINDOWS = StepPreprocessor(history_window=LastMessages(1), input_window=LastTurns(1))


# a host that cannot provide the history (a proxy, a replay) passes None; it
# is named only where a window asks for history
@pytest.mark.parametrize(
    "step, preprocessor, says_unknown",
    [
        (tool_step(CONVERSATION, None), WHOLE, False),
        (tool_step(CONVERSATION, None), KEEP_HISTORY, True),
        (tool_step(None, None), WHOLE, False),
        # windows keep it unknown rather than empty
        (tool_step(CONVERSATION, None), WINDOWS, True),
        (replace(before_generate_step(), history=None), KEEP_HISTORY, True),
        (replace(after_generate_step(), history=None), KEEP_HISTORY, True),
    ],
)
@pytest.mark.anyio
async def test_a_history_the_host_could_not_provide_is_named_where_it_would_show(
    step: Step, preprocessor: StepPreprocessor[Step], says_unknown: bool
) -> None:
    rendered = await render(step, preprocessor)
    assert (UNKNOWN_HISTORY in rendered) == says_unknown
    assert "[AGENT STEP - BEGIN" in rendered


@pytest.mark.parametrize("nonce", [None, "3f9a1c7e"])
@pytest.mark.anyio
async def test_a_nonce_marks_every_block_and_every_message_header(
    nonce: str | None,
) -> None:
    rendered = await render(tool_step(GO, CONVERSATION), KEEP_HISTORY, nonce=nonce)
    mark = "" if nonce is None else f" ({nonce})"
    assert [line for line in rendered.splitlines() if line.startswith("[B")] == [
        f"[BEGIN INPUT{mark}]",
        f"[BEGIN HISTORY{mark}]",
    ]
    assert [line for line in rendered.splitlines() if line.startswith("[E")] == [
        f"[END INPUT{mark}]",
        f"[END HISTORY{mark}]",
    ]
    assert f"[AGENT STEP - END TOOL CALL{mark}]" in rendered
    # Scout 0.5.5 and earlier write no nonce on message headers
    header = mark if scout_nonce("3f9a1c7e") else ""
    assert f"USER{header}:\ngo" in rendered


@pytest.mark.anyio
async def test_numbered_citations_resolve_to_the_agents_messages() -> None:
    rendered, extract_refs = await step_as_str(
        await preprocess_step(parallel_after_b(), KEEP_HISTORY)
    )
    step_block = rendered.split("[AGENT STEP - BEGIN TOOL CALL AND RESULT]")[1]
    assert "cmd: b" in step_block and "cmd: a" not in step_block
    # history's task is [M1]; the step's message holding the call is [M2]
    assert [r.id for r in extract_refs("see [M1] and [M2]")] == ["task", "both"]
