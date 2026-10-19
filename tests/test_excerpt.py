import json
from collections.abc import Callable
from dataclasses import replace
from functools import reduce
from itertools import permutations
from typing import cast

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
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store
from inspect_scout import MessagesPreprocessor, message_numbering

from inspect_sentinel import (
    AfterGenerate,
    BeforeToolCall,
    Context,
    Excerpt,
    Step,
    concurrent,
    decide_final,
    drop_input,
    drop_reasoning,
    drop_tool_usage,
    keep_agent_messages,
    keep_system,
    last_messages,
    last_turns,
    new_since_last_report,
    step_as_str,
)
from inspect_sentinel._decorators import monitor, protocol
from inspect_sentinel._excerpt import scout_nonce
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


def texts(messages: list[ChatMessage] | None) -> list[str] | None:
    return None if messages is None else [m.text for m in messages]


def tool_step(
    input: list[ChatMessage] | None, history: list[ChatMessage] | None
) -> BeforeToolCall:
    return replace(
        before_tool_call("bash", cmd="ls"), input=input, agent_messages=history
    )


def windowed() -> BeforeToolCall:
    return tool_step(CONVERSATION, CONVERSATION)


def parallel_after_b() -> Step:
    # one message holds calls a and b; a has run, b is the step
    return replace(
        after_tool_call("bash", "result b", cmd="b"),
        call=call("b"),
        message="both",
        agent_messages=[
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
    # an excerpt from any window shows the cut
    assert texts(keep_agent_messages(step).agent_messages) == history


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
    cut = keep_agent_messages(
        replace(before_generate_step(), input=input, agent_messages=history)
    )
    assert (texts(cut.input), texts(cut.agent_messages)) == (kept_input, kept_history)


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
    assert (texts(cut.input), texts(cut.agent_messages)) == (kept, kept)
    assert cut.step == windowed()
    # a window narrows an excerpt as well as a step
    assert window(drop_input(windowed()), n).agent_messages == cut.agent_messages


def test_last_messages_keeps_a_result_it_did_not_part_from_its_call() -> None:
    # as new_since_last_report leaves it when the call was already shown
    started = Excerpt(windowed(), None, CONVERSATION[3:])
    assert texts(last_messages(started, 9).agent_messages) == ["result a", "reply 2"]


@pytest.mark.parametrize("window", [last_turns, last_messages])
def test_a_negative_window_raises(
    window: Callable[
        [BeforeToolCall | Excerpt[BeforeToolCall], int], Excerpt[BeforeToolCall]
    ],
) -> None:
    with pytest.raises(ValueError, match="n >= 0"):
        window(windowed(), -1)


def one_turn(step: Step | Excerpt[Step]) -> Excerpt[Step]:
    return last_turns(step, 1)


UNKNOWN_INPUT = (
    "[BEGIN INPUT]\nThe model input for this call is not available.\n[END INPUT]"
)


@pytest.mark.parametrize(
    "excerpting, input, says_unknown",
    [
        (drop_input, CONVERSATION, False),
        # an unknown input is still named, dropped or windowed
        (drop_input, None, True),
        (one_turn, None, True),
    ],
)
@pytest.mark.anyio
async def test_drop_input_leaves_it_out_and_an_unknown_input_stays_named(
    excerpting: Callable[[Step], Excerpt[Step]],
    input: list[ChatMessage] | None,
    says_unknown: bool,
) -> None:
    rendered = await step_as_str(excerpting(tool_step(input, CONVERSATION)))
    assert (UNKNOWN_INPUT in rendered) == says_unknown
    assert ("[BEGIN INPUT]" in rendered) == says_unknown


Shown = tuple[list[str] | None, list[str] | None]


def report(cut: Excerpt[Step]) -> Observation:
    return Observation.score(
        0.0, json.dumps([texts(cut.input), texts(cut.agent_messages)])
    )


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
async def test_new_since_last_report_records_per_function() -> None:
    generated = replace(
        after_generate_step(),
        conversation="c",
        input=CONVERSATION[:2],
        agent_messages=CONVERSATION[:2],
    )
    called = replace(tool_step(None, CONVERSATION), conversation="c")
    assert await shown(Store(), [generated, called]) == [
        [(texts(CONVERSATION[:2]), texts(CONVERSATION[:2]))],
        shared(None, texts(CONVERSATION)),
    ]


@pytest.mark.anyio
async def test_new_since_last_report_is_not_recorded_for_a_cancelled_function() -> None:
    started = anyio.Event()

    @monitor
    def watcher(stall: bool = False) -> Monitor:
        async def check(context: Context, step: BeforeToolCall) -> Observation | None:
            cut = new_since_last_report(context, step)
            if stall:
                started.set()
                await anyio.sleep_forever()
            return report(cut)

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
            cut = new_since_last_report(context, step)
            final = Decision.proceed(
                json.dumps([texts(cut.input), texts(cut.agent_messages)])
            )
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
            cut = new_since_last_report(context, step)
            seen.append(texts(cut.agent_messages) or [])
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
async def test_step_as_str_leaves_out_empty_parts_whatever_the_renderer() -> None:
    async def as_json(messages: list[ChatMessage]) -> str:
        return json.dumps([m.text for m in messages])

    prefilled = [user("task"), asks("prefill")]
    step = replace(before_generate_step(), input=prefilled, agent_messages=prefilled)
    rendered = await step_as_str(step, messages_as_str=as_json)
    assert "[BEGIN HISTORY]" not in rendered
    assert "[AGENT STEP - BEGIN MODEL REQUEST]" not in rendered


def test_new_since_last_report_outside_the_runner_raises() -> None:
    with in_step() as context, pytest.raises(RuntimeError, match="runner"):
        new_since_last_report(context, windowed())


@pytest.mark.parametrize(
    "cut, absent, block, shown",
    [
        (windowed(), [], "TOOL CALL", ["cmd: ls"]),
        # the same conversation as the input, so not shown even when kept
        (keep_agent_messages(windowed()), ["HISTORY"], "TOOL CALL", ["cmd: ls"]),
        (
            after_tool_call("bash", "out.txt", cmd="ls"),
            ["HISTORY"],
            "TOOL CALL AND RESULT",
            ["cmd: ls", "out.txt"],
        ),
        (
            replace(before_generate_step(), input=GO, agent_messages=GO),
            ["HISTORY", "INPUT"],
            "MODEL REQUEST",
            ["go"],
        ),
        (after_generate_step(), [], "MODEL REPLY", ["ASSISTANT:\ndone"]),
        (
            replace(after_generate_step(), output=ModelOutput(model="m", choices=[])),
            [],
            "MODEL REPLY",
            ["ASSISTANT:"],
        ),
    ],
)
@pytest.mark.anyio
async def test_step_as_str_renders_each_kept_part_in_a_block(
    cut: Step | Excerpt[Step], absent: list[str], block: str, shown: list[str]
) -> None:
    rendered = await step_as_str(cut)
    assert not any(f"[BEGIN {name}]" in rendered for name in absent)
    # the step's own part comes last, in a block named for its stage
    assert rendered.endswith(f"[AGENT STEP - END {block}]")
    step_block = rendered.split(f"[AGENT STEP - BEGIN {block}]")[1]
    assert all(text in step_block for text in shown)


@pytest.mark.anyio
async def test_a_request_ending_in_a_prefill_has_no_step_block() -> None:
    prefilled = [user("task"), asks("prefill")]
    rendered = await step_as_str(
        replace(before_generate_step(), input=prefilled, agent_messages=prefilled)
    )
    assert "[AGENT STEP - BEGIN MODEL REQUEST]" not in rendered
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
        agent_messages=conversation,
        output=ModelOutput(
            model="m",
            choices=[ChatCompletionChoice(message=thinks("reply", "reply"))],
        ),
    )


@pytest.mark.parametrize(
    "drop, gone, kept",
    [
        # system messages are left out by default, as Scout leaves them out
        (None, ["be helpful"], ["thinking about", "searched"]),
        (keep_system, [], ["be helpful", "thinking about", "searched"]),
        (drop_reasoning, ["be helpful", "thinking about"], ["searched"]),
        (drop_tool_usage, ["be helpful", "searched"], ["thinking about"]),
    ],
)
@pytest.mark.anyio
async def test_drops_leave_content_out_of_every_block(
    drop: Callable[[Step], Excerpt[Step]] | None, gone: list[str], kept: list[str]
) -> None:
    step = thoughtful()
    rendered = await step_as_str(drop(step) if drop else step)
    step_block = rendered.split("[AGENT STEP - BEGIN MODEL REPLY]")[1]
    assert not any(text in rendered for text in gone)
    assert all(text in rendered for text in kept)
    # the reply's own reasoning goes too, its text stays
    assert ("thinking about reply" in step_block) == (drop is not drop_reasoning)
    assert "reply" in step_block


@pytest.mark.anyio
async def test_drops_compose_with_windows_and_keep_cites() -> None:
    numbered, extract_refs = message_numbering(
        MessagesPreprocessor(exclude_system=False)
    )
    rendered = await step_as_str(
        drop_reasoning(last_turns(thoughtful(), 1)),
        messages_as_str=numbered,
    )
    assert "be helpful" not in rendered and "thinking about" not in rendered
    assert [r.id for r in extract_refs("see [M1]")] == ["earlier"]


def two_turns(step: Step | Excerpt[Step]) -> Excerpt[Step]:
    # keeps the whole conversation, system message included
    return last_turns(step, 2)


def many_messages(step: Step | Excerpt[Step]) -> Excerpt[Step]:
    return last_messages(step, 50)


EXCERPTING: list[Callable[[Step | Excerpt[Step]], Excerpt[Step]]] = [
    keep_system,
    keep_agent_messages,
    drop_reasoning,
    drop_tool_usage,
    two_turns,
    drop_input,
]


@pytest.mark.parametrize(
    "order",
    list(permutations(EXCERPTING)),
    ids=lambda order: "-".join(f.__name__ for f in order),
)
@pytest.mark.anyio
async def test_a_chain_gives_the_same_excerpt_in_any_order(
    order: tuple[Callable[[Step | Excerpt[Step]], Excerpt[Step]], ...],
) -> None:
    chained = reduce(
        lambda shown, f: f(shown), order, cast(Step | Excerpt[Step], thoughtful())
    )
    assert isinstance(chained, Excerpt)
    assert chained.exclude == {"input", "reasoning", "tool_usage"}
    rendered = await step_as_str(chained)
    assert rendered == await step_as_str(
        drop_input(
            two_turns(
                drop_tool_usage(
                    drop_reasoning(keep_agent_messages(keep_system(thoughtful())))
                )
            )
        )
    )
    # the input is dropped, so the kept history shows, system message included
    assert "be helpful" in rendered.split("[END HISTORY]")[0]
    assert not any(
        text in rendered for text in ["thinking about", "searched", "[BEGIN INPUT]"]
    )


@pytest.mark.parametrize(
    "excerpting",
    [
        drop_input,
        keep_agent_messages,
        two_turns,
        many_messages,
        drop_reasoning,
        drop_tool_usage,
        keep_system,
    ],
)
def test_every_function_keeps_what_earlier_ones_excluded_or_kept(
    excerpting: Callable[[Step | Excerpt[Step]], Excerpt[Step]],
) -> None:
    earlier = drop_tool_usage(keep_system(thoughtful()))
    # a later function must not put back the default of leaving system out
    assert {"tool_usage"} <= excerpting(earlier).exclude
    assert "system" not in excerpting(earlier).exclude


def test_keep_system_twice_is_keep_system_once() -> None:
    assert keep_system(keep_system(thoughtful())).exclude == {"agent_messages"}


@monitor
def keeps_system(first: bool = True) -> Monitor:
    async def check(context: Context, step: AfterGenerate) -> Observation | None:
        shown = (
            new_since_last_report(context, keep_system(step))
            if first
            else keep_system(new_since_last_report(context, step))
        )
        return Observation.score(0.0, json.dumps(sorted(shown.exclude)))

    return check


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.anyio
async def test_keep_system_holds_before_or_after_new_since_last_report(
    first: bool,
) -> None:
    with in_step(store=Store()) as context:
        observations = await run_monitors(
            {"judge": keeps_system(first=first)}, context, thoughtful()
        )
    [seen] = [o.report.explanation for o in observations.succeeded]
    assert json.loads(seen or "") == ["agent_messages"]


def merged() -> ChatMessage:
    # what a provider's preparation makes of two consecutive user messages
    return ChatMessageUser(
        content="task\nmore", id="merged", metadata={"combined_from": ["task", "more"]}
    )


def history_only(step: Step | Excerpt[Step]) -> Excerpt[Step]:
    return keep_agent_messages(drop_input(step))


def with_system(step: Step | Excerpt[Step]) -> Excerpt[Step]:
    return keep_system(keep_agent_messages(step))


@pytest.mark.parametrize(
    "excerpting, input, history, shows_history",
    [
        # left out unless kept
        (None, [user("summary")], CONVERSATION, False),
        # in place of an unknown input, kept or not
        (None, None, CONVERSATION, True),
        # kept: the same conversation is shown once
        (keep_agent_messages, CONVERSATION, CONVERSATION, False),
        # kept: turns before a compaction are only in history, so all of it shows
        (keep_agent_messages, [user("summary")], CONVERSATION, True),
        (keep_agent_messages, [merged()], [user("task"), user("more")], False),
        (keep_agent_messages, CONVERSATION, [unnamed("no id")], True),
        (history_only, CONVERSATION, CONVERSATION, True),
        # a dropped input is not an unknown one
        (drop_input, CONVERSATION, CONVERSATION, False),
        # and an unknown input, dropped, leaves only the step
        (drop_input, None, CONVERSATION, False),
        (history_only, None, CONVERSATION, True),
        # only hidden messages are extra
        (keep_agent_messages, CONVERSATION[1:], CONVERSATION, False),
        (with_system, CONVERSATION[1:], CONVERSATION, True),
    ],
)
@pytest.mark.anyio
async def test_history_is_kept_by_choice_and_shown_when_it_adds_to_the_input(
    excerpting: Callable[[Step | Excerpt[Step]], Excerpt[Step]] | None,
    input: list[ChatMessage] | None,
    history: list[ChatMessage],
    shows_history: bool,
) -> None:
    step = tool_step(input, history)
    rendered = await step_as_str(excerpting(step) if excerpting else step)
    assert ("[BEGIN HISTORY]" in rendered) == shows_history
    if shows_history:
        # the whole history, messages the input also has included
        assert all(
            m.text in rendered.split("[END HISTORY]")[0]
            for m in history
            if m.role != "system" or excerpting is with_system
        )


@pytest.mark.anyio
async def test_blocks_come_input_then_history_then_the_step_numbered_in_that_order() -> (
    None
):
    numbered, extract_refs = message_numbering(
        MessagesPreprocessor(exclude_system=False)
    )
    step = replace(
        tool_step([user("summary")], CONVERSATION), call=call("b"), message="calls b"
    )
    rendered = await step_as_str(keep_agent_messages(step), messages_as_str=numbered)
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


def windows(step: Step) -> Excerpt[Step]:
    return last_messages(last_turns(keep_agent_messages(step), 1), 1)


UNKNOWN_HISTORY = (
    "[BEGIN HISTORY]\nThe agent's conversation history is not available.\n[END HISTORY]"
)


# a host that cannot provide the history (a proxy, a replay) passes None
@pytest.mark.parametrize(
    "step, excerpting, says_unknown",
    [
        (tool_step(CONVERSATION, None), None, False),
        (tool_step(CONVERSATION, None), keep_agent_messages, True),
        # in place of an unknown input
        (tool_step(None, None), None, True),
        # windows keep it unknown rather than empty
        (tool_step(CONVERSATION, None), windows, True),
        (
            replace(before_generate_step(), agent_messages=None),
            keep_agent_messages,
            True,
        ),
        (
            replace(after_generate_step(), agent_messages=None),
            keep_agent_messages,
            True,
        ),
    ],
)
@pytest.mark.anyio
async def test_a_history_the_host_could_not_provide_is_named_where_it_would_show(
    step: Step,
    excerpting: Callable[[Step], Excerpt[Step]] | None,
    says_unknown: bool,
) -> None:
    rendered = await step_as_str(excerpting(step) if excerpting else step)
    assert (UNKNOWN_HISTORY in rendered) == says_unknown
    assert "[AGENT STEP - BEGIN" in rendered


@pytest.mark.parametrize("nonce", [None, "3f9a1c7e"])
@pytest.mark.anyio
async def test_a_nonce_marks_every_block_and_the_default_renderers_headers(
    nonce: str | None,
) -> None:
    rendered = await step_as_str(
        keep_agent_messages(tool_step(GO, CONVERSATION)), nonce=nonce
    )
    mark = "" if nonce is None else f" ({nonce})"
    assert [line for line in rendered.splitlines() if line.startswith("[")] == [
        f"[BEGIN INPUT{mark}]",
        f"[END INPUT{mark}]",
        f"[BEGIN HISTORY{mark}]",
        f"[END HISTORY{mark}]",
        f"[AGENT STEP - BEGIN TOOL CALL{mark}]",
        f"[AGENT STEP - END TOOL CALL{mark}]",
    ]
    # Scout 0.5.5 and earlier write no nonce on message headers
    header = mark if scout_nonce("3f9a1c7e") else ""
    assert f"USER{header}:\ngo" in rendered


@pytest.mark.anyio
async def test_step_as_str_says_when_the_model_input_is_unknown() -> None:
    rendered = await step_as_str(tool_step(None, []))
    assert (
        "[BEGIN INPUT]\nThe model input for this call is not available.\n[END INPUT]"
        in rendered
    )


@pytest.mark.anyio
async def test_numbered_citations_resolve_to_the_agents_messages() -> None:
    numbered, extract_refs = message_numbering()
    rendered = await step_as_str(
        keep_agent_messages(parallel_after_b()), messages_as_str=numbered
    )
    step_block = rendered.split("[AGENT STEP - BEGIN TOOL CALL AND RESULT]")[1]
    assert "cmd: b" in step_block and "cmd: a" not in step_block
    # history's task is [M1]; the step's message holding the call is [M2]
    assert [r.id for r in extract_refs("see [M1] and [M2]")] == ["task", "both"]
