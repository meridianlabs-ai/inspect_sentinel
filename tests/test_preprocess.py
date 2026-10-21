import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import anyio
import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ContentToolUse,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolFunction, ToolInfo
from inspect_ai.util import Store

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
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
from inspect_sentinel._report import Decision, Escalations, Observation
from inspect_sentinel._runner import run_monitors, run_protocols, run_sentinel
from inspect_sentinel._types import Monitor, MonitorGroup, Protocol

from ._fakes import (
    SEQUENCES,
    ListRecorder,
    View,
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

REPLIED = ["sys", "task", "reply 1"]

# every list whole, nothing omitted, so a test sees the window alone
ALL = StepPreprocessor(
    input_window=AllMessages(),
    history_window=AllMessages(),
    agent_messages_window=AllMessages(),
    omit_system=False,
)

LISTS = ("input", "history", "agent_messages")


def texts(messages: Sequence[ChatMessage] | None) -> list[str] | None:
    return None if messages is None else [m.text for m in messages]


def tool_step(
    input: Sequence[ChatMessage] | None, agent_messages: Sequence[ChatMessage] | None
) -> BeforeToolCall:
    return replace(
        before_tool_call("bash", cmd="ls"), input=input, agent_messages=agent_messages
    )


def all_lists(messages: Sequence[ChatMessage] | None) -> BeforeToolCall:
    return replace(tool_step(messages, messages), history=messages)


BASH = ToolInfo(name="bash", description="Run a command.")

Shown = tuple[list[str] | None, list[str] | None]


def shown(step: Step) -> Shown:
    return texts(step.input), texts(step.agent_messages)


def every(step: Step) -> dict[str, list[str] | None]:
    return {name: texts(getattr(step, name)) for name in LISTS}


async def render(
    step: Step,
    preprocessor: StepPreprocessor | None = None,
    *,
    nonce: str | None = None,
) -> str:
    p = preprocessor or StepPreprocessor()
    text, _ = await step_as_str(
        await preprocess_step(step, p),
        nonce=nonce,
    )
    return text


@pytest.mark.parametrize(
    "window, kept",
    [
        (AllMessages(), texts(CONVERSATION)),
        (LastTurns(0), []),
        (LastTurns(1), ["reply 2"]),
        # no more than n turns: all of it
        (LastTurns(2), texts(CONVERSATION)),
        (LastMessages(0), []),
        (LastMessages(2), ["result a", "reply 2"]),
        (LastMessages(9), texts(CONVERSATION)),
    ],
)
@pytest.mark.parametrize("name", LISTS)
@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.anyio
async def test_each_window_cuts_its_list_as_it_is(
    window: Window,
    kept: list[str],
    name: str,
    sequence: Callable[[list[ChatMessage]], Sequence[ChatMessage]],
) -> None:
    step = all_lists(sequence(CONVERSATION))
    thinned = await preprocess_step(step, replace(ALL, **{f"{name}_window": window}))
    assert every(thinned) == {**every(step), name: kept}


@pytest.mark.parametrize("window", [LastTurns, LastMessages])
def test_a_negative_window_raises(window: type[LastTurns | LastMessages]) -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        window(-1)


# a long conversation of turns: a task, then a reply and its result each turn
LONG = [user("task")] + [
    message
    for turn in range(500)
    for message in (
        asks(f"reply {turn}", call(f"c{turn}")),
        result(f"c{turn}", f"r{turn}"),
    )
]


@pytest.mark.parametrize(
    "window, kept",
    [
        (LastMessages(2), ["reply 499", "r499"]),
        (LastTurns(2), ["reply 498", "r498", "reply 499", "r499"]),
    ],
)
@pytest.mark.anyio
async def test_a_window_reads_only_the_end_of_a_list_it_keeps_the_end_of(
    window: Window, kept: list[str]
) -> None:
    view = View(LONG)
    thinned = await preprocess_step(
        replace(tool_step(None, None), history=view),
        StepPreprocessor(input_window=None, history_window=window),
    )
    assert texts(thinned.history) == kept
    # a few reads at the end, not the whole conversation
    assert view.reads < 4 * len(kept)


@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.anyio
async def test_any_sequence_renders_as_a_list_does(
    sequence: Callable[[list[Any]], Sequence[Any]],
) -> None:
    def step(sequence: Callable[[list[Any]], Sequence[Any]]) -> AfterToolCall:
        return replace(
            after_tool_call("bash", "out", cmd="ls"),
            input=sequence(CONVERSATION),
            history=sequence(CONVERSATION),
            agent_messages=sequence(CONVERSATION),
            tools=sequence([BASH]),
            tool_choice="auto",
        )

    assert await render(step(sequence), ALL) == await render(step(list), ALL)
    # rendered without preprocessing too
    assert (await step_as_str(step(sequence)))[0] == (await step_as_str(step(list)))[0]


@pytest.mark.anyio
async def test_a_result_whose_whole_output_is_unknown_is_shown_by_its_result() -> None:
    known = after_tool_call("bash", "out", cmd="ls")
    unknown = replace(known, output=None)
    assert await preprocess_step(unknown) == replace(
        await preprocess_step(known), output=None
    )
    assert await render(unknown, ALL) == await render(known, ALL)


WHOLE = [m.text for m in CONVERSATION]


@pytest.mark.parametrize(
    "preprocessor, given, shown_after",
    [
        # by default the whole input and nothing else
        (
            StepPreprocessor(),
            CONVERSATION,
            {"input": WHOLE[1:], "history": [], "agent_messages": []},
        ),
        (
            replace(ALL, input_window=None),
            CONVERSATION,
            {"input": [], "history": WHOLE, "agent_messages": WHOLE},
        ),
        # a list the host could not provide stays None, windowed or not
        (StepPreprocessor(), None, dict.fromkeys(LISTS)),
        (ALL, None, dict.fromkeys(LISTS)),
    ],
)
@pytest.mark.anyio
async def test_a_list_with_no_window_is_empty_and_an_unknown_one_stays_none(
    preprocessor: StepPreprocessor,
    given: list[ChatMessage] | None,
    shown_after: dict[str, list[str] | None],
) -> None:
    step = all_lists(given)
    assert every(await preprocess_step(step, preprocessor)) == shown_after


UNKNOWN_INPUT = (
    "[BEGIN INPUT]\nThe model input for this call is not available.\n[END INPUT]"
)
UNKNOWN_HISTORY = (
    "[BEGIN HISTORY]\nThe messages the agent has sent are not available.\n[END HISTORY]"
)
UNKNOWN_AGENT_MESSAGES = "[BEGIN AGENT MESSAGES]\nThe agent's own message list is not available.\n[END AGENT MESSAGES]"


@pytest.mark.anyio
async def test_an_unknown_list_is_named_and_an_empty_one_has_no_block() -> None:
    rendered = await render(all_lists(None))
    assert all(
        block in rendered
        for block in (UNKNOWN_INPUT, UNKNOWN_HISTORY, UNKNOWN_AGENT_MESSAGES)
    )
    rendered = await render(all_lists([]), ALL)
    assert not any(
        f"[BEGIN {name}]" in rendered for name in ("INPUT", "HISTORY", "AGENT MESSAGES")
    )


def new_since(context: Context) -> StepPreprocessor[Step]:
    return replace(
        ALL,
        input_window=NewSinceLastReport(context),
        agent_messages_window=NewSinceLastReport(context),
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


def shared(input: list[str] | None, agent_messages: list[str] | None) -> list[Shown]:
    # both tool-stage functions were shown the same
    return [(input, agent_messages)] * 2


@pytest.mark.anyio
async def test_new_since_last_report_shows_each_message_once_per_function() -> None:
    steps: list[Step] = [
        replace(tool_step(input, agent_messages), conversation=conversation)
        for conversation, input, agent_messages in [
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


@monitor
def sent_since() -> MonitorGroup:
    async def calls(context: Context, step: BeforeToolCall) -> Observation | None:
        thinned = await preprocess_step(
            step,
            StepPreprocessor(
                input_window=None, history_window=NewSinceLastReport(context)
            ),
        )
        return Observation.score(0.0, json.dumps(texts(thinned.history)))

    return MonitorGroup(calls)


# `history` holds each version of an edited message under its one id
V1 = ChatMessageUser(content="v1", id="edited")
V2 = ChatMessageUser(content="v2", id="edited")
NEXT = user("next")


@pytest.mark.parametrize(
    "histories, expected",
    [
        (
            [[V1, NEXT], [V1, NEXT, V2], [V1, NEXT, V2]],
            [["v1", "next"], ["v2"], []],
        ),
        # the message shown last is edited later: what came between is new
        ([[V1], [V1, NEXT, V2]], [["v1"], ["next", "v2"]]),
    ],
)
@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.anyio
async def test_new_since_last_report_resumes_after_the_copy_shown_last(
    histories: list[list[ChatMessage]],
    expected: list[list[str]],
    sequence: Callable[[list[ChatMessage]], Sequence[ChatMessage]],
) -> None:
    store = Store()
    seen: list[list[str]] = []
    for history in histories:
        step = replace(tool_step(None, None), history=sequence(history))
        with in_step(store=store) as context:
            observations = await run_monitors({"judge": sent_since()}, context, step)
        [observation] = observations.succeeded
        seen.append(json.loads(observation.report.explanation or ""))
    assert seen == expected


@pytest.mark.anyio
async def test_new_since_last_report_reads_only_what_is_new_in_a_list_that_grew() -> (
    None
):
    store = Store()
    shown: list[list[str]] = []
    views = [View(LONG[:-2]), View(LONG)]
    for view in views:
        step = replace(tool_step(None, None), history=view)
        with in_step(store=store) as context:
            observations = await run_monitors({"judge": sent_since()}, context, step)
        [observation] = observations.succeeded
        shown.append(json.loads(observation.report.explanation or ""))
    assert shown[1] == ["reply 499", "r499"]
    # a few reads at the end, not the whole conversation again
    assert views[1].reads < 4 * len(shown[1])


@pytest.mark.anyio
async def test_new_since_last_report_records_per_function() -> None:
    generated = replace(
        after_generate_step(),
        conversation="c",
        input=CONVERSATION[:2],
        agent_messages=CONVERSATION[:2],
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
        async def decide(
            context: Context, step: Step, escalations: Escalations
        ) -> Decision | None:
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
        async def decide(
            context: Context, step: BeforeToolCall, escalations: Escalations
        ) -> Decision | None:
            thinned = await preprocess_step(step, new_since(context))
            final = Decision.proceed(json.dumps(shown(thinned)))
            if in_task:
                async with anyio.create_task_group() as group:
                    group.start_soon(_decide_final, final)
            decide_final(final)

        return decide

    store = Store()
    seen: list[str | None] = []
    for agent_messages in [CONVERSATION[:3], CONVERSATION]:
        decision = await run_sentinel(
            decides(), host_context(store=store), tool_step(None, agent_messages)
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
    async def end(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        decide_final(Decision.proceed())

    return end


@pytest.mark.anyio
async def test_new_since_last_report_is_not_recorded_when_a_childs_final_passes_by() -> (
    None
):
    seen: list[list[str]] = []

    @protocol
    def passed_by() -> Protocol:
        async def judge(
            context: Context, step: BeforeToolCall, escalations: Escalations
        ) -> Decision | None:
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
    # it never judged the agent's messages, so it is shown again
    assert seen == [texts(CONVERSATION)] * 2


@pytest.mark.anyio
async def test_new_since_last_report_outside_the_runner_raises() -> None:
    with in_step() as context, pytest.raises(RuntimeError, match="runner"):
        await preprocess_step(tool_step(CONVERSATION, CONVERSATION), new_since(context))


@pytest.mark.parametrize(
    "step, block, contents",
    [
        (tool_step(CONVERSATION, None), "TOOL CALL REQUEST", ["cmd: ls"]),
        (
            after_tool_call("bash", "out.txt", cmd="ls"),
            "TOOL CALL RESULT",
            ["cmd: ls", "out.txt"],
        ),
        (replace(before_generate_step(), input=[user("go")]), "MODEL REQUEST", ["go"]),
        (after_generate_step(), "MODEL REPLY", ["go", "ASSISTANT:\ndone"]),
        (
            replace(after_generate_step(), output=ModelOutput(model="m", choices=[])),
            "MODEL REPLY",
            ["go", "ASSISTANT:"],
        ),
    ],
)
@pytest.mark.anyio
async def test_everything_is_in_one_block_named_for_the_stage(
    step: Step, block: str, contents: list[str]
) -> None:
    rendered = await render(step, ALL)
    assert rendered.startswith(f"[BEGIN AGENT STEP - {block}]\n")
    assert rendered.endswith(f"\n[END AGENT STEP - {block}]")
    assert rendered.count("[BEGIN AGENT STEP") == 1
    assert all(text in rendered for text in contents)


@pytest.mark.anyio
async def test_a_reply_holds_its_request_then_the_reply() -> None:
    request = replace(
        before_generate_step(),
        input=CONVERSATION[1:],
        tools=[BASH],
    )
    reply = replace(after_generate_step(), input=request.input, tools=request.tools)
    asked, answered = await render(request), await render(reply)
    # the reply's block is the request's, with the reply added
    assert answered == asked.replace(
        "[BEGIN AGENT STEP - MODEL REQUEST]", "[BEGIN AGENT STEP - MODEL REPLY]"
    ).replace(
        "[END AGENT STEP - MODEL REQUEST]",
        "\n[BEGIN REPLY]\n[M5] ASSISTANT:\ndone\n\n[END REPLY]\n"
        "[END AGENT STEP - MODEL REPLY]",
    )


REQUEST = [user("go")]


@pytest.mark.parametrize(
    "step",
    [
        before_generate_step(),
        after_generate_step(),
        before_tool_call("bash", cmd="ls"),
        after_tool_call("bash", "out", cmd="ls"),
    ],
)
@pytest.mark.anyio
async def test_every_stage_shows_the_request_as_its_input_then_its_tools(
    step: Step,
) -> None:
    rendered = await render(
        replace(
            step, input=REQUEST, tools=[BASH], tool_choice=ToolFunction(name="bash")
        )
    )
    headers = [
        line
        for line in rendered.splitlines()
        if re.match(r"\[(BEGIN|END|M|TOOL CHOICE)", line)
    ]
    request = [
        "[BEGIN INPUT]",
        "[M1] USER:",
        "[END INPUT]",
        "[BEGIN TOOLS]",
        "[END TOOLS]",
        '[TOOL CHOICE] {"name": "bash"}',
    ]
    assert headers[1 : 1 + len(request)] == request
    assert json.loads(
        rendered.split("[BEGIN TOOLS]\n")[1].split("\n[END TOOLS]")[0]
    ) == [BASH.model_dump(exclude_none=True)]


@pytest.mark.parametrize(
    "step, left_out",
    [
        # no tools offered
        (replace(before_generate_step(), input=REQUEST), ["[BEGIN TOOLS]"]),
        # a tool stage whose request is not known
        (
            replace(
                before_tool_call("bash", cmd="ls"),
                input=None,
                tools=[BASH],
                tool_choice="auto",
            ),
            ["[BEGIN TOOLS]", "[TOOL CHOICE]"],
        ),
    ],
)
@pytest.mark.anyio
async def test_tools_not_offered_or_not_known_are_left_out(
    step: Step, left_out: list[str]
) -> None:
    rendered = await render(step)
    assert not any(text in rendered for text in left_out)


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


def thoughtful_reply() -> AfterGenerate:
    conversation: list[ChatMessage] = [
        ChatMessageSystem(content="be helpful", id="sys"),
        user("task"),
        thinks("earlier", "earlier"),
    ]
    return replace(
        after_generate_step(),
        input=conversation,
        agent_messages=conversation,
        output=ModelOutput(
            model="m", choices=[ChatCompletionChoice(message=thinks("reply", "reply"))]
        ),
    )


def thoughtful_result() -> Step:
    return replace(
        after_tool_call("bash", "", cmd="ls"),
        input=[user("task")],
        result=ChatMessageTool(
            content=[
                ContentReasoning(reasoning="thinking about result"),
                ContentText(text="result"),
            ],
            tool_call_id="call_1",
            function="bash",
        ),
    )


@pytest.mark.parametrize("step", [thoughtful_reply(), thoughtful_result()])
@pytest.mark.parametrize(
    "preprocessor, gone",
    [
        (StepPreprocessor(), ["be helpful"]),
        (StepPreprocessor(omit_system=False), []),
        (StepPreprocessor(omit_reasoning=True), ["be helpful", "thinking about"]),
        (StepPreprocessor(omit_tool_usage=True), ["be helpful", "searched"]),
    ],
)
@pytest.mark.anyio
async def test_omissions_reach_every_list_and_the_steps_own_messages(
    step: Step, preprocessor: StepPreprocessor, gone: list[str]
) -> None:
    rendered = await render(
        step, replace(preprocessor, agent_messages_window=AllMessages())
    )
    assert not any(text in rendered for text in gone)
    step_block = rendered.split("[BEGIN AGENT STEP - ")[1]
    if "thinking about" in gone:
        assert "thinking about" not in step_block


@pytest.mark.anyio
async def test_preprocessing_leaves_the_hosts_objects_unchanged() -> None:
    step = thoughtful_reply()
    before = step.output.model_dump()
    assert step.input is not None
    reasoning = list(step.input[-1].content)
    await preprocess_step(
        step, StepPreprocessor(omit_reasoning=True, omit_tool_usage=True)
    )
    assert step.output.model_dump() == before
    assert list(step.input[-1].content) == reasoning


@pytest.mark.anyio
async def test_transform_runs_last_on_the_thinned_step() -> None:
    seen: list[Shown] = []

    async def withhold(step: Step) -> Step:
        seen.append(shown(step))
        assert step.input is not None
        return replace(step, input=[m for m in step.input if m.text != "result a"])

    step = tool_step(CONVERSATION, CONVERSATION)
    rendered = await render(
        step, StepPreprocessor(input_window=LastMessages(3), transform=withhold)
    )
    assert seen == [(["reply 1", "result a", "reply 2"], [])]
    assert "result a" not in rendered and "reply 2" in rendered


@pytest.mark.parametrize("nonce", [None, "3f9a1c7e"])
@pytest.mark.anyio
async def test_a_nonce_marks_every_block_and_every_message_header(
    nonce: str | None,
) -> None:
    rendered = await render(
        replace(
            tool_step([user("go")], CONVERSATION),
            history=CONVERSATION,
            tools=[BASH],
            tool_choice="auto",
        ),
        StepPreprocessor(
            history_window=AllMessages(), agent_messages_window=AllMessages()
        ),
        nonce=nonce,
    )
    mark = "" if nonce is None else f" ({nonce})"
    assert [line for line in rendered.splitlines() if line.startswith("[B")] == [
        f"[BEGIN AGENT STEP - TOOL CALL REQUEST{mark}]",
        f"[BEGIN HISTORY{mark}]",
        f"[BEGIN AGENT MESSAGES{mark}]",
        f"[BEGIN INPUT{mark}]",
        f"[BEGIN TOOLS{mark}]",
        f"[BEGIN CALL{mark}]",
    ]
    assert f'[TOOL CHOICE{mark}] "auto"' in rendered.splitlines()
    assert rendered.endswith(f"[END AGENT STEP - TOOL CALL REQUEST{mark}]")
    assert f"USER{mark}:\ngo" in rendered


@pytest.mark.anyio
async def test_messages_are_numbered_in_block_order_and_cites_resolve() -> None:
    step = replace(tool_step([user("summary")], [user("task")]), history=[user("sent")])
    rendered, extract_refs = await step_as_str(await preprocess_step(step, ALL))
    starts = [
        rendered.index(marker)
        for marker in (
            "[BEGIN AGENT STEP - TOOL CALL REQUEST]",
            "[BEGIN HISTORY]",
            "[BEGIN AGENT MESSAGES]",
            "[BEGIN INPUT]",
            "[BEGIN CALL]",
        )
    ]
    assert starts == sorted(starts)
    assert [r.id for r in extract_refs("see [M1], [M2], [M3] and [M4]")] == [
        "sent",
        "task",
        "summary",
        "msg_1",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "step",
    [before_tool_call("bash", cmd="ls"), after_tool_call("bash", "out", cmd="ls")],
)
async def test_a_cite_of_the_call_resolves_to_the_message_that_made_it(
    step: BeforeToolCall | AfterToolCall,
) -> None:
    _, extract_refs = await step_as_str(
        replace(step, input=[], agent_messages=[], message_id="msg_A7")
    )
    assert [r.id for r in extract_refs("[M1]")] == ["msg_A7"]
