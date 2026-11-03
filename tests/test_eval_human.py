from typing import Any, NamedTuple

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.approval._approval import Approval, ApprovalDecision
from inspect_ai.approval._human import acp as acp_module
from inspect_ai.approval._human import approver as approver_module
from inspect_ai.dataset import Sample
from inspect_ai.event import SentinelEvent
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolCallContent, ToolCallView, tool

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    HumanDecision,
    Monitor,
    MonitorGroup,
    Observation,
    Protocol,
    Reported,
    Step,
    concurrent,
    decide_final,
    human,
    monitor,
    observe_only,
    protocol,
    run_protocols,
    sequential,
    threshold,
)
from inspect_sentinel._report import origin


@tool
def addition() -> Tool:
    async def execute(x: int, y: int) -> str:
        """Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.
        """
        return str(x + y)

    return execute


def shout_viewer(call: ToolCall) -> ToolCallView:
    return ToolCallView(
        call=ToolCallContent(title="Shout", format="markdown", content="**{{text}}**")
    )


@tool(viewer=shout_viewer)
def shout() -> Tool:
    async def execute(text: str) -> str:
        """Shout some text.

        Args:
            text: Text to shout.
        """
        return text.upper()

    return execute


@protocol
def h_escalate_turn() -> Protocol:
    async def unsure(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        if step.output.message.tool_calls:
            return Decision.escalate("two calls together, {{x}}")
        return Decision.proceed()

    return unsure


@protocol
def h_escalate() -> Protocol:
    async def unsure(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return Decision.escalate("adds {{x}} suspiciously")

    return unsure


class Panel:
    def __init__(
        self, decision: ApprovalDecision, explanation: str | None = None
    ) -> None:
        self.decision: ApprovalDecision = decision
        self.explanation = explanation
        self.views: list[ToolCallView] = []
        self.choices: list[list[str]] = []
        self.messages: list[str] = []
        self.histories: list[object] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def acp(**kwargs: object) -> Approval | None:
            return None

        async def panel(
            message: str,
            call: ToolCall,
            view: ToolCallView,
            history: object,
            choices: list[str],
        ) -> Approval:
            self.messages.append(message)
            self.histories.append(history)
            self.views.append(view)
            self.choices.append(choices)
            return Approval(decision=self.decision, explanation=self.explanation)

        monkeypatch.setattr(acp_module, "request_human_approval_via_acp", acp)
        monkeypatch.setattr(approver_module, "panel_approval", panel)

    def content(self) -> str:
        [view] = self.views
        assert view.call is not None
        return view.call.content


def run(sentinel: Any, **kwargs: Any) -> EvalLog:
    outputs = [
        ModelOutput.for_tool_call(
            "mockllm/model",
            tool_name="addition",
            tool_arguments={"x": 1, "y": 1},
            content="Adding now.",
        ),
        ModelOutput.from_content("mockllm/model", content="done"),
    ]
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target="2")],
        solver=[use_tools(addition()), generate()],
        sentinel=sentinel,
    )
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    return eval(task, model=model, **kwargs)[0]


def tool_messages(log: EvalLog) -> list[ChatMessageTool]:
    assert log.samples
    return [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]


def escalated_to_human() -> Any:
    return sequential([h_escalate(), human(stages=["tool_call"])])


def test_an_escalation_asks_the_person_and_approve_runs_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = Panel("approve")
    panel.install(monkeypatch)

    log = run(escalated_to_human())

    assert log.status == "success", log.error
    assert panel.choices == [["approve", "reject", "terminate"]]
    assert panel.messages == ["Adding now."]
    content = panel.content()
    assert (
        "**Escalated by**\n\n```\n- h_escalate: adds { {x}} suspiciously\n```"
        in content
    )
    assert content.index("Escalated by") < content.index("addition(")
    [message] = tool_messages(log)
    assert message.error is None
    assert message.text == "2"


def test_after_a_call_the_person_sees_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = Panel("approve")
    panel.install(monkeypatch)

    log = run(human(stages=["tool_result"]))

    assert log.status == "success", log.error
    assert panel.choices == [["approve", "terminate"]]
    assert "**Result**\n\n```\n2\n```" in panel.content()


def step(
    after: bool = False, view: ToolCallView | None = None
) -> BeforeToolCall | AfterToolCall:
    call = ToolCall(id="c1", function="addition", arguments={"x": 1, "y": 1})
    view = view or ToolCallView()
    if after:
        result = ChatMessageTool(content="2", tool_call_id="c1", function="addition")
        return AfterToolCall("", "Adding.", "m1", call, result, "2", view, [], [])
    return BeforeToolCall("", "Adding.", "m1", call, view, [], [])


def escalation(explanation: str) -> Escalations:
    return (Reported("monitor", "monitor", Decision.escalate(explanation), "check"),)


ADD = ToolCall(id="c1", function="addition", arguments={"x": 1, "y": 1})


SHOUT = ToolCall(id="c2", function="shout", arguments={"text": "hello"})


def turn_output(*calls: ToolCall, content: str = "Adding and shouting.") -> ModelOutput:
    message = ChatMessageAssistant(
        content=content, model="mockllm/model", tool_calls=list(calls) or None
    )
    return ModelOutput(
        model="mockllm/model",
        choices=[ChatCompletionChoice(message=message, stop_reason="tool_calls")],
    )


def run_turn(sentinel: Any) -> EvalLog:
    outputs = [
        turn_output(ADD, SHOUT),
        ModelOutput.from_content("mockllm/model", content="done"),
    ]
    task = Task(
        dataset=[Sample(input="Add 1 and 1, and shout hello.")],
        solver=[use_tools(addition(), shout()), generate()],
        sentinel=sentinel,
    )
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    return eval(task, model=model)[0]


def turn_escalated_to_human() -> Any:
    return sequential([h_escalate_turn(), human(stages=["model_output"])])


def test_an_escalated_turn_shows_every_call_and_approve_runs_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = Panel("approve")
    panel.install(monkeypatch)

    log = run_turn(turn_escalated_to_human())

    assert log.status == "success", log.error
    assert panel.choices == [["approve", "terminate"]]
    assert panel.messages == ["Adding and shouting."]
    content = panel.content()
    escalated = (
        "**Escalated by**\n\n```\n- h_escalate_turn: two calls together, { {x}}\n```"
    )
    assert content.startswith(escalated)
    # each call as the transcript shows it: the default view, and the tool's
    # own viewer with its placeholders filled in
    assert content.index("addition(x=1, y=1)") < content.index("**hello**")
    assert [m.text for m in tool_messages(log)] == ["2", "HELLO"]


class Seen(NamedTuple):
    stage: str
    conversation: str
    subject: str
    decisions: tuple[HumanDecision, ...] | None


def _held(step: Step) -> tuple[HumanDecision, ...] | None:
    # whatever sequence the host hands the step, as it was at the step
    return None if step.human_decisions is None else tuple(step.human_decisions)


@monitor
def h_spy(seen: Any) -> MonitorGroup:
    async def tool_call(context: Context, step: BeforeToolCall) -> Observation | None:
        seen.append(Seen("tool_call", step.conversation, step.call.id, _held(step)))
        return None

    async def tool_result(context: Context, step: AfterToolCall) -> Observation | None:
        seen.append(Seen("tool_result", step.conversation, step.call.id, _held(step)))
        return None

    async def model_output(context: Context, step: AfterGenerate) -> Observation | None:
        seen.append(
            Seen(
                "model_output",
                step.conversation,
                step.output.completion,
                _held(step),
            )
        )
        return None

    return MonitorGroup(tool_call, tool_result, model_output)


def call_id(log: EvalLog, function: str = "addition") -> str:
    assert log.samples
    [found] = [
        call.id
        for m in log.samples[0].messages
        if isinstance(m, ChatMessageAssistant)
        for call in m.tool_calls or []
        if call.function == function
    ]
    return found


def kept(record: HumanDecision) -> tuple[Any, ...]:
    report = record.decided.report
    return (
        record.stage,
        record.step_id,
        record.call_ids,
        record.decided.name,
        report.action,
        report.explanation,
    )


@monitor
def h_loud() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.93, "posts credentials to an unknown host")

    return check


def test_no_monitor_score_is_seen_in_human_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Panel("approve", "fine").install(monkeypatch)
    seen: list[Seen] = []

    log = run(
        [
            sequential(
                [threshold(h_loud(), escalate_at=0.5), human(stages=["tool_call"])]
            ),
            observe_only([h_spy(seen)]),
        ]
    )

    assert log.status == "success", log.error
    assert log.samples
    # the escalation the person answered carried the score
    [escalation] = [
        e
        for e in log.samples[0].events
        if isinstance(e, SentinelEvent) and e.action == "escalate"
    ]
    assert escalation.metadata is not None
    assert "credentials" in str(escalation.metadata)
    [last] = [s for s in seen if s.subject == "done"]
    assert last.decisions is not None
    [record] = last.decisions
    assert record.decided.report.metadata is None
    assert record.decided.report.references == []
    assert "credentials" not in repr(record)
    assert "0.93" not in repr(record)
    # nor through the runner's record of the step it was made at
    step_nodes, _ = origin(record.decided.report)
    assert step_nodes is None


@protocol
def h_overrules() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        await run_protocols(human(stages=["tool_call"]), context, step)
        return Decision.proceed("carry on regardless")

    return decide


def test_a_terminate_a_parent_overruled_is_seen_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Panel("terminate", "stop").install(monkeypatch)
    seen: list[Seen] = []

    log = run([h_overrules(), observe_only([h_spy(seen)])])

    assert log.status == "success", log.error
    [last] = [s for s in seen if s.subject == "done"]
    assert last.decisions is not None
    assert [kept(r) for r in last.decisions] == [
        ("tool_call", call_id(log), (call_id(log),), "human", "terminate", "stop")
    ]


class Answered:
    """An event a test's panel sets once the person answers, made in the eval's loop."""

    def __init__(self) -> None:
        self._event: anyio.Event | None = None

    @property
    def event(self) -> anyio.Event:
        if self._event is None:
            self._event = anyio.Event()
        return self._event


@protocol
def h_final_after_the_person(answered: Any) -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        # outlives the person's terminate, which cancels its siblings
        with anyio.CancelScope(shield=True):
            await answered.event.wait()
            decide_final(Decision.proceed("carry on"))
        return None

    return decide


def test_a_superseded_decision_is_not_seen_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answered = Answered()

    async def acp(**kwargs: object) -> Approval | None:
        return None

    async def panel(
        message: str,
        call: ToolCall,
        view: ToolCallView,
        history: object,
        choices: list[str],
    ) -> Approval:
        answered.event.set()
        return Approval(decision="terminate", explanation="stop")

    monkeypatch.setattr(acp_module, "request_human_approval_via_acp", acp)
    monkeypatch.setattr(approver_module, "panel_approval", panel)
    seen: list[Seen] = []

    log = run(
        [
            human(stages=["tool_call"]),
            h_final_after_the_person(answered),
            observe_only([h_spy(seen)]),
        ]
    )

    assert log.status == "success", log.error
    assert log.samples
    # the person's terminate was recorded, then outrun by the final proceed
    assert [
        e.status
        for e in log.samples[0].events
        if isinstance(e, SentinelEvent) and e.factory == "inspect_sentinel/human"
    ] == ["reported", "superseded"]
    assert [m.text for m in tool_messages(log)] == ["2"]
    later = [s for s in seen if s.stage == "tool_result" or s.subject == "done"]
    assert [(s.stage, s.decisions) for s in later] == [
        ("tool_result", ()),
        ("model_output", ()),
    ]


@protocol
def h_proceeds() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed()

    return decide


@protocol
def h_forwards(child: Any, done: Any) -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        strongest = (await run_protocols(child, context, step)).strongest()
        done.event.set()
        return strongest.report if strongest is not None else None

    return decide


def test_a_superseded_copy_of_the_persons_decision_is_not_seen_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Panel("terminate", "stop").install(monkeypatch)
    done = Answered()
    seen: list[Seen] = []
    # the inner concurrent passes the person's terminate up as a copy carrying
    # the vote; a final proceed then supersedes it above
    inner = concurrent(
        {"first_vote": h_proceeds(), "person": human(stages=["tool_call"])}
    )

    log = run(
        [
            h_forwards(inner, done),
            h_final_after_the_person(done),
            observe_only([h_spy(seen)]),
        ]
    )

    assert log.status == "success", log.error
    assert log.samples
    [superseded] = [
        e
        for e in log.samples[0].events
        if isinstance(e, SentinelEvent) and e.status == "superseded"
    ]
    assert superseded.factory != "inspect_sentinel/human"
    assert "first_vote: proceed" in (superseded.explanation or "")
    [last] = [s for s in seen if s.subject == "done"]
    assert last.decisions == ()
