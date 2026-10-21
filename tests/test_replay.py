from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import anyio
import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import StoreModel

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    HumanAnswer,
    MonitorGroup,
    Observation,
    ReplayStep,
    concurrent,
    handle_escalation,
    human,
    monitor,
    protocol,
    replay_monitor,
    sequential,
    steps_from_messages,
    threshold,
)
from inspect_sentinel._from_messages import Cut
from tests._fakes import View


def call(call_id: str, cmd: str = "ls") -> ToolCall:
    return ToolCall(id=call_id, function="bash", arguments={"cmd": cmd})


def conversation(turns: int) -> list[ChatMessage]:
    # a system prompt and a task, then a turn of one tool call per index
    messages: list[ChatMessage] = [
        ChatMessageSystem(id="sys", content="You are an agent."),
        ChatMessageUser(id="task", content="Do the task."),
    ]
    for turn in range(turns):
        messages.append(
            ChatMessageAssistant(
                id=f"a{turn}", content=f"turn {turn}", tool_calls=[call(f"c{turn}")]
            )
        )
        messages.append(
            ChatMessageTool(id=f"r{turn}", content="out", tool_call_id=f"c{turn}")
        )
    return messages


def keys(steps: Sequence[ReplayStep]) -> list[tuple[str, str]]:
    return [(step.stage, step.id) for step in steps]


def ids(messages: Sequence[ChatMessage] | None) -> list[str | None]:
    assert messages is not None
    return [message.id for message in messages]


def test_messages_project_each_turn_into_its_steps_in_the_order_they_ran() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="task", content="Do it."),
        ChatMessageAssistant(id="a0", content="two", tool_calls=[call("x"), call("y")]),
        ChatMessageTool(id="rx", content="x out", tool_call_id="x"),
        ChatMessageTool(id="ry", content="y out", tool_call_id="y"),
        ChatMessageUser(id="nudge", content="continue"),
        ChatMessageAssistant(id="a1", content="one", tool_calls=[call("z")]),
    ]

    steps = list(steps_from_messages(messages, conversation="conv"))

    assert keys(steps) == [
        ("model_input", "a0"),
        ("model_output", "a0"),
        ("tool_call", "x"),
        ("tool_result", "x"),
        ("tool_call", "y"),
        ("tool_result", "y"),
        ("model_input", "a1"),
        ("model_output", "a1"),
        ("tool_call", "z"),
    ]
    by_key = {(s.stage, s.id): s.step for s in steps}
    second = by_key[("model_input", "a1")]
    assert isinstance(second, BeforeGenerate)
    assert ids(second.input) == ["task", "a0", "rx", "ry", "nudge"]
    assert ids(second.history) == ["task"]
    # later user messages are not the agent's own
    assert ids(second.agent_messages) == ["task", "a0", "rx", "ry"]
    output = by_key[("model_output", "a1")]
    assert isinstance(output, AfterGenerate)
    assert output.output.message.id == "a1"
    assert ids(output.history) == ["task", "a0", "rx", "ry", "nudge"]
    before_y = by_key[("tool_call", "y")]
    assert isinstance(before_y, BeforeToolCall)
    assert (before_y.input, before_y.tools, before_y.message_id) == (None, None, "a0")
    assert ids(before_y.agent_messages) == ["task", "a0"]
    after_y = by_key[("tool_result", "y")]
    assert isinstance(after_y, AfterToolCall)
    assert (after_y.result.id, after_y.output, after_y.truncated) == ("ry", None, None)
    assert all(s.segment == ("conv", 0) for s in steps)
    assert all(s.step.human_decisions is None for s in steps)


def test_the_messages_projection_reads_each_message_a_bounded_number_of_times() -> None:
    def reads(turns: int) -> int:
        messages = View(conversation(turns))
        for step in steps_from_messages(messages):
            # what a window monitor reads: the last few of each list
            for history in (step.step.history, step.step.agent_messages):
                assert history is not None
                list(history[-3:])
        return messages.reads

    assert reads(200) <= 2 * reads(100) + 10


def retained(steps: Sequence[ReplayStep]) -> int:
    # the entries the steps hold: a list they share counts once, and any
    # other sequence of messages, a copy, counts in full
    shared: dict[int, int] = {}
    copied = 0
    for replay_step in steps:
        for name in ("input", "history", "agent_messages"):
            value = getattr(replay_step.step, name)
            if isinstance(value, Cut):
                items: list[ChatMessage] = cast(Any, value)._items
                shared[id(items)] = len(items)
            elif value is not None:
                copied += len(value)
    return sum(shared.values()) + copied


def test_the_messages_projection_retains_entries_in_proportion_to_the_transcript() -> (
    None
):
    def kept(turns: int) -> int:
        return retained(list(steps_from_messages(conversation(turns))))

    assert kept(200) <= 2 * kept(100) + 10


@monitor
def counter() -> MonitorGroup:
    # how many steps this instance has seen in its store, at both tool stages
    class Seen(StoreModel):
        count: int = 0

    async def before(context: Context, step: BeforeToolCall) -> Observation:
        seen = context.store_as(Seen)
        seen.count += 1
        return Observation(suspicion=0.0, metadata={"seen": seen.count})

    async def after(context: Context, step: AfterToolCall) -> Observation:
        seen = context.store_as(Seen)
        seen.count += 1
        return Observation(suspicion=0.0, metadata={"seen": seen.count})

    return MonitorGroup(before, after)


def segmented(
    turns_per_segment: Sequence[int], conversations: int = 1
) -> list[ReplayStep]:
    # tool steps over conversations and compaction segments
    steps: list[ReplayStep] = []
    for conv in range(conversations):
        for segment, turns in enumerate(turns_per_segment):
            messages = conversation(turns)
            for step in steps_from_messages(messages, conversation=f"conv{conv}"):
                if step.stage in ("tool_call", "tool_result"):
                    steps.append(
                        replace(
                            step,
                            id=f"{conv}.{segment}.{step.id}",
                            segment=(f"conv{conv}", segment),
                        )
                    )
    return steps


@pytest.mark.parametrize(
    "split,max_unit_steps,expected",
    [
        # two conversations of two segments with 2 and 1 tool calls each
        ("compaction", None, [1, 2, 3, 4, 1, 2] * 2),
        ("conversation", None, [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
        (None, None, list(range(1, 13))),
        ("conversation", 4, [1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2]),
    ],
)
@pytest.mark.anyio
async def test_a_unit_keeps_its_store_from_step_to_step_and_the_next_starts_fresh(
    split: Any, max_unit_steps: int | None, expected: list[int]
) -> None:
    steps = segmented([2, 1], conversations=2)

    reports = await replay_monitor(
        counter(), steps, split=split, max_unit_steps=max_unit_steps
    )

    seen = [
        report.report.report.metadata["seen"]
        for report in reports
        if report.report is not None and report.report.report.metadata is not None
    ]
    assert seen == expected


@pytest.mark.anyio
async def test_units_run_concurrently() -> None:
    started = 0
    all_started = anyio.Event()

    @monitor
    def replay_waits() -> Any:
        async def before(context: Context, step: BeforeToolCall) -> Observation:
            nonlocal started
            started += 1
            if started == 3:
                all_started.set()
            with anyio.fail_after(5):
                await all_started.wait()
            return Observation.score(0.0)

        return before

    # a sample with two compactions: three units, one tool call each
    steps = [s for s in segmented([1, 1, 1]) if s.stage == "tool_call"]

    reports = await replay_monitor(replay_waits(), steps, split="compaction")

    assert started == 3
    assert all(report.report is not None for report in reports)


@pytest.mark.anyio
async def test_a_lone_monitor_reports_at_its_name_and_a_protocol_at_the_root() -> None:
    steps = [s for s in segmented([1]) if s.stage == "tool_call"]

    monitored = await replay_monitor(counter(), steps)
    protocol_run = await replay_monitor(threshold(flag(0.1), terminate_at=0.5), steps)

    assert (monitored[0].name, monitored[0].path) == ("counter", "counter")
    assert monitored[0].decision is None
    assert (protocol_run[0].name, protocol_run[0].path) == ("threshold", "")
    report = protocol_run[0].report
    assert report is not None and isinstance(report.report, Decision)
    assert report.report.action == "proceed"
    assert [r.path for r in protocol_run[0].reports] == ["flag_1", ""]


@pytest.mark.anyio
async def test_a_step_at_a_stage_nothing_watches_reports_nothing() -> None:
    steps = list(steps_from_messages(conversation(1)))

    reports = await replay_monitor(counter(), steps)

    assert [(r.step.stage, r.report is not None) for r in reports] == [
        ("model_input", False),
        ("model_output", False),
        ("tool_call", True),
        ("tool_result", True),
    ]


@pytest.mark.anyio
async def test_a_failed_monitor_is_recorded_and_the_replay_goes_on() -> None:
    @monitor
    def replay_broken() -> Any:
        async def before(context: Context, step: BeforeToolCall) -> Observation:
            raise RuntimeError("boom")

        return before

    steps = [s for s in segmented([2]) if s.stage == "tool_call"]

    reports = await replay_monitor(replay_broken(), steps)

    assert [len(r.failed) for r in reports] == [1, 1]
    assert all(r.report is None for r in reports)
    assert str(reports[0].failed[0].error) == "boom"


@pytest.mark.anyio
async def test_a_protocol_error_fails_the_replay() -> None:
    @protocol
    def replay_raises() -> Any:
        async def tool_call(
            context: Context, step: BeforeToolCall, escalations: Escalations
        ) -> Decision:
            raise ValueError("bad protocol")

        return tool_call

    steps = [s for s in segmented([1]) if s.stage == "tool_call"]

    with pytest.raises(ValueError, match="bad protocol"):
        await replay_monitor(replay_raises(), steps)


def flag(threshold_value: float = 1.0) -> Any:
    @monitor(name=f"flag_{int(threshold_value * 10)}")
    def flagger() -> Any:
        async def before(context: Context, step: BeforeToolCall) -> Observation:
            return Observation.score(threshold_value, "flagged")

        return before

    return flagger()


def escalated_to_human() -> Any:
    return sequential([threshold(flag(), escalate_at=0.5), human(stages=["tool_call"])])


@pytest.mark.parametrize(
    "answers,policy,action,source",
    [
        # no recorded answer: the fail-closed answer, not an unhandled escalation
        ({}, "recorded", "reject", "unanswered"),
        ({"human": HumanAnswer("approve", "fine")}, "recorded", "proceed", "recorded"),
        (
            {"human": HumanAnswer("terminate", "stop")},
            "recorded",
            "terminate",
            "recorded",
        ),
        # recorded answers are ignored on request
        (
            {"human": HumanAnswer("approve", "fine")},
            "fail_closed",
            "reject",
            "unanswered",
        ),
        # a recorded answer at another instance's path is not this one's
        ({"other": HumanAnswer("approve")}, "recorded", "reject", "unanswered"),
        # an answer the replayed human() does not offer counts as none
        ({"human": HumanAnswer("modify")}, "recorded", "reject", "unanswered"),
    ],
)
@pytest.mark.anyio
async def test_human_is_answered_by_the_recorded_answer_else_fails_closed(
    answers: dict[str, HumanAnswer], policy: Any, action: str, source: str
) -> None:
    step = next(s for s in segmented([1]) if s.stage == "tool_call")
    answered = ReplayStep(step.id, step.stage, step.step, step.segment, answers)

    [report] = await replay_monitor(escalated_to_human(), [answered], human=policy)

    assert report.decision is not None and report.decision.action == action
    assert report.human == source


@pytest.mark.anyio
async def test_a_stand_in_is_asked_only_where_no_answer_is_recorded() -> None:
    asked: list[tuple[str, tuple[str, ...], int]] = []

    async def stand_in(
        step: ReplayStep, escalations: Escalations, choices: Sequence[str]
    ) -> HumanAnswer:
        asked.append((step.id, tuple(choices), len(escalations)))
        return HumanAnswer("approve", "oracle says fine")

    first, second = [s for s in segmented([2]) if s.stage == "tool_call"]
    steps = [
        ReplayStep(
            first.id,
            first.stage,
            first.step,
            first.segment,
            {"human": HumanAnswer("reject", "no")},
        ),
        ReplayStep(second.id, second.stage, second.step, second.segment),
    ]

    reports = await replay_monitor(escalated_to_human(), steps, human=stand_in)

    assert asked == [(second.id, ("approve", "reject", "terminate"), 1)]
    assert [(r.decision.action if r.decision else None, r.human) for r in reports] == [
        ("reject", "recorded"),
        ("proceed", "simulated"),
    ]


@pytest.mark.anyio
async def test_an_escalation_nothing_handles_still_terminates_as_live() -> None:
    steps = [s for s in segmented([1]) if s.stage == "tool_call"]

    [report] = await replay_monitor(threshold(flag(), escalate_at=0.5), steps)

    assert report.decision is not None and report.decision.action == "terminate"
    root = report.report
    assert root is not None and isinstance(root.report, Decision)
    assert root.report.action == "escalate"
    assert report.human is None


@pytest.mark.anyio
async def test_a_handled_escalation_needs_no_person() -> None:
    steps = [s for s in segmented([1]) if s.stage == "tool_call"]
    sentinel = sequential(
        [threshold(flag(), escalate_at=0.5), handle_escalation("proceed")]
    )

    [report] = await replay_monitor(sentinel, steps)

    assert report.decision is not None and report.decision.action == "proceed"


@pytest.mark.anyio
async def test_two_steps_of_one_stage_with_one_id_are_refused() -> None:
    step = next(s for s in segmented([1]) if s.stage == "tool_call")

    with pytest.raises(ValueError, match="Two 'tool_call' steps have the id"):
        await replay_monitor(counter(), [step, step])


@pytest.mark.parametrize("kwargs", [{"max_units": 0}, {"max_unit_steps": 0}])
@pytest.mark.anyio
async def test_unit_limits_must_be_positive(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        await replay_monitor(counter(), [], **kwargs)


@pytest.mark.anyio
async def test_a_step_not_selected_is_not_run_and_its_state_is_not_kept() -> None:
    steps = segmented([3])
    chosen = {steps[2].id, steps[3].id}

    reports = await replay_monitor(
        counter(), steps, select=lambda step: step.id in chosen, split=None
    )

    assert [(r.step.stage, r.step.id) for r in reports] == [
        (steps[2].stage, steps[2].id),
        (steps[3].stage, steps[3].id),
    ]
    seen = [
        r.report.report.metadata["seen"]
        for r in reports
        if r.report is not None and r.report.report.metadata is not None
    ]
    assert seen == [1, 2]


@pytest.mark.anyio
async def test_a_recorded_terminate_is_kept_even_where_not_offered() -> None:
    # human() accepts the fail-closed terminate a host gives for a closed prompt
    step = next(s for s in segmented([1]) if s.stage == "tool_call")
    sentinel = sequential(
        [
            threshold(flag(), escalate_at=0.5),
            human(stages=["tool_call"], choices=["approve", "reject"]),
        ]
    )
    answered = ReplayStep(
        step.id,
        step.stage,
        step.step,
        step.segment,
        {"human": HumanAnswer("terminate")},
    )

    [report] = await replay_monitor(sentinel, [answered])

    assert report.decision is not None and report.decision.action == "terminate"
    assert report.human == "recorded"


async def approves_all(
    step: ReplayStep, escalations: Escalations, choices: Sequence[str]
) -> HumanAnswer:
    return HumanAnswer("approve")


@pytest.mark.parametrize(
    "answers,policy,source",
    [
        # one human() answered as recorded, the other got nobody
        ({"first": HumanAnswer("approve")}, "recorded", "unanswered"),
        # one answered as recorded, the other by the stand-in
        ({"first": HumanAnswer("approve")}, approves_all, "simulated"),
        (
            {"first": HumanAnswer("approve"), "second": HumanAnswer("approve")},
            "recorded",
            "recorded",
        ),
    ],
)
@pytest.mark.anyio
async def test_a_step_that_asked_two_people_reports_the_least_real_answer(
    answers: dict[str, HumanAnswer], policy: Any, source: str
) -> None:
    step = next(s for s in segmented([1]) if s.stage == "tool_call")
    sentinel = concurrent(
        {"first": human(stages=["tool_call"]), "second": human(stages=["tool_call"])}
    )
    answered = ReplayStep(step.id, step.stage, step.step, step.segment, answers)

    [report] = await replay_monitor(sentinel, [answered], human=policy)

    assert report.human == source
