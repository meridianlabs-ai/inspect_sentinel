import ast
import importlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai._sentinel._integration import EvalInfo, steps_from_events
from inspect_ai.approval import (
    Approval,
    ApprovalPolicy,
    Approver,
    approver,
    human_approver,
)
from inspect_ai.approval._human import acp as acp_module
from inspect_ai.approval._human import approver as approver_module
from inspect_ai.dataset import Sample
from inspect_ai.event import (
    Event,
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    Timeline,
    TimelineEvent,
    TimelineSpan,
    ToolEvent,
    timeline_build,
)
from inspect_ai.log import EvalLog, condense_events, write_eval_log
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    execute_tools,
    get_model,
    model_origin,
)
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolCallView, ToolError, tool
from inspect_ai.util import StoreModel
from inspect_ai.util._span import span
from inspect_scout import (
    Result,
    Scanner,
    Transcript,
    scan,
    scan_results_df,
    scan_resume,
    scanner,
    transcripts_from,
)
from inspect_scout._transcript.timeline import (  # pyright: ignore[reportPrivateUsage]
    _walk_spans as scout_walk_spans,  # pyright: ignore[reportPrivateUsage]
)

from inspect_sentinel import (
    AfterGenerate,
    BeforeGenerate,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    HumanAnswer,
    MonitorGroup,
    Observation,
    ReplayStep,
    Sentinels,
    human,
    monitor,
    observe_only,
    protocol,
    sequential,
    threshold,
)
from inspect_sentinel._replay import replay_monitor
from inspect_sentinel.scout import (
    Stage,
    as_reader,
    as_scanner,
    human_stand_in,
    replay_check,
    replay_check_differences,
    step_selector,
)
from inspect_sentinel.scout._context import Contexts
from inspect_sentinel.scout._steps import transcript_steps
from tests._fakes import View

Row = dict[str, Any]
# pandas is untyped here; only the label table needs it
pandas: Any = importlib.import_module("pandas")

SCOUT = Path(__file__).parents[2] / "src" / "inspect_sentinel" / "scout"


@tool
def addition() -> Tool:
    async def execute(x: int, y: int) -> str:
        """Add two numbers; negative numbers are an error.

        Args:
            x: First number to add.
            y: Second number to add.
        """
        if x < 0:
            raise ToolError("negative numbers are not supported")
        return str(x + y)

    return execute


def call(x: int, call_id: str | None = None) -> ModelOutput:
    output = ModelOutput.for_tool_call("mockllm/model", "addition", {"x": x, "y": 1})
    if call_id is not None:
        message = output.message
        assert message.tool_calls is not None
        message.tool_calls[0].id = call_id
    return output


@solver
def with_helper() -> Solver:
    """A sub-agent that makes one tool call in a conversation of its own, before the agent's own turns."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with span("helper", type="agent"):
            messages: list[ChatMessage] = [ChatMessageUser(content="help me add")]
            output = await get_model().generate(messages, tools=[addition()])
            messages.append(output.message)
            await execute_tools(messages, [addition()])
        return state

    return solve


def run(
    log_dir: Path,
    turns: Sequence[Sequence[ModelOutput]],
    sentinel: Sentinels | None,
    samples: Sequence[Sample] | None = None,
    *,
    helpers: Sequence[ModelOutput] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvalLog:
    # one sample per list of tool-call outputs, each ending with a reply, after
    # its helper's tool call where there are helpers
    found = (
        list(samples)
        if samples is not None
        else [
            Sample(id=index, input=f"add, task {index}") for index in range(len(turns))
        ]
    )
    outputs = [
        output
        for index, calls in enumerate(turns)
        for output in [
            *([helpers[index]] if helpers is not None else []),
            *calls,
            ModelOutput.from_content("mockllm/model", "done"),
        ]
    ]
    [log] = eval(
        Task(
            dataset=found,
            solver=[
                use_tools(addition()),
                *([with_helper()] if helpers is not None else []),
                generate(),
            ],
            sentinel=sentinel,
            metadata=metadata,
        ),
        model=get_model("mockllm/model", custom_outputs=outputs),
        log_dir=str(log_dir),
        max_samples=1,
        display="none",
    )
    assert log.status == "success", log.error
    return log


def scanned(scans: Path, logs: Path, **scanners: Any) -> dict[str, list[Row]]:
    status = scan(
        scanners=scanners,
        transcripts=transcripts_from(str(logs)),
        scans=str(scans),
        display="none",
        max_processes=1,
    )
    return rows(status.location)


def rows(location: str) -> dict[str, list[Row]]:
    frames: dict[str, Any] = dict(scan_results_df(location).scanners)
    return {
        name: cast(list[Row], frame.to_dict("records"))
        for name, frame in frames.items()
    }


def keyed(found: list[Row], column: str) -> dict[tuple[str, str], Any]:
    return {
        (row["transcript_id"], row["metadata.step_id"]): row[column] for row in found
    }


def references(row: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    cited = row[f"{kind}_references"]
    return json.loads(cited) if isinstance(cited, str) else []


def test_importing_inspect_sentinel_does_not_load_scout() -> None:
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, inspect_sentinel; print('inspect_scout' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert loaded.stdout.strip() == "False"


def test_the_scout_adapter_imports_only_the_private_names_it_is_allowed() -> None:
    # inspect_ai's _integration is its designated boundary for sentinel, and
    # its registry lookup is the one private name besides (design call 1);
    # from sentinel, the replay runner and the messages projection, internal
    # since 2026-10-26 with the adapter as their only caller (design call 23)
    private: dict[str, set[str]] = {}
    sentinel: set[str] = set()
    for source in SCOUT.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [(node.module, {alias.name for alias in node.names})]
            elif isinstance(node, ast.Import):
                imported = [(alias.name, set[str]()) for alias in node.names]
            else:
                continue
            for module, names in imported:
                if module.split(".")[0] == "inspect_sentinel":
                    sentinel.add(module)
                if module != "__future__" and (
                    any(part.startswith("_") for part in module.split("."))
                    or any(name.startswith("_") for name in names)
                ):
                    private.setdefault(module, set()).update(names)
    assert sentinel == {
        "inspect_sentinel",
        "inspect_sentinel.config",
        "inspect_sentinel._replay",
        "inspect_sentinel._from_messages",
    }
    assert private == {
        "inspect_ai._sentinel._integration": {
            "EvalInfo",
            "MalformedLogError",
            "steps_from_events",
        },
        "inspect_ai._util.registry": {"registry_lookup"},
        "inspect_sentinel._replay": {"HumanFallback", "ReplayReport", "replay_monitor"},
        "inspect_sentinel._from_messages": {"steps_from_messages"},
    }


def test_the_scout_adapter_exports_the_replay_api() -> None:
    import inspect_sentinel.scout

    assert set(inspect_sentinel.scout.__all__) == {
        "Stage",
        "StepSelector",
        "StepTable",
        "as_reader",
        "as_scanner",
        "human_stand_in",
        "replay_check",
        "replay_check_differences",
        "step_selector",
    }


@pytest.mark.parametrize(
    "stage", ["model_input", "model_output", "tool_call", "tool_result"]
)
def test_replay_and_read_agree_through_scout(tmp_path: Path, stage: Stage) -> None:
    samples = [
        Sample(id=7, input="add, with an integer id", description="the first"),
        Sample(id="8", input="add, with a numeric string id"),
        Sample(
            id="listed", input=[ChatMessageUser(content="add, from a message list")]
        ),
    ]
    run(
        tmp_path / "logs",
        [[call(1), call(2)], [call(3)], [call(4)]],
        observe_only(replay_check()),
        samples,
        helpers=[call(5, f"sub{i}") for i in range(3)],
        metadata={"difficulty": "hard"},
    )

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        replay=as_scanner(replay_check(), stage=stage),
        read=as_reader("replay_check", stage=stage),
    )

    replayed = keyed(found["replay"], "metadata.metadata")
    read = keyed(found["read"], "metadata.metadata")
    assert replayed.keys() == read.keys()
    # each sample's turns and its helper's
    expected = {"model_input": 10, "model_output": 10, "tool_call": 7}
    assert len(read) == expected.get(stage, 7)
    for key, recorded in read.items():
        assert replayed[key]["eval"] == recorded["eval"], key
        differ = {
            name
            for name, digest in recorded["step"].items()
            if replayed[key]["step"].get(name) != digest
        }
        assert not differ, (key, differ)
    for kind in ("message", "event"):
        cited = {k: references(r, kind) for k, r in by_key(found["replay"]).items()}
        assert cited == {
            k: references(r, kind) for k, r in by_key(found["read"]).items()
        }
    assert all(
        references(r, "message") or references(r, "event") for r in found["read"]
    )


def by_key(found: list[Row]) -> dict[tuple[str, str], Row]:
    return {(row["transcript_id"], row["metadata.step_id"]): row for row in found}


@monitor
def by_argument() -> MonitorGroup:
    # suspicion from the call's own argument, so a replay is deterministic
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(min(1.0, int(step.call.arguments["x"]) / 10))

    return MonitorGroup(before)


def test_replay_monitor_gives_the_reports_the_scanner_records(tmp_path: Path) -> None:
    log = run(tmp_path / "logs", [[call(2), call(5)]], None)
    assert log.samples is not None
    sample = log.samples[0]

    found = scanned(tmp_path / "scans", tmp_path / "logs", s=as_scanner(by_argument()))

    # the eval's own sample holds its events expanded, with no pool
    steps = steps_from_events(
        sample.messages,
        sample.events,
        [],
        sample_uuid=sample.uuid or "",
        eval_info=EvalInfo.from_eval(log.eval),
    )
    reports = anyio.run(replay_monitor, by_argument(), steps)
    expected = {
        (sample.uuid, r.step.id): r.report.report.suspicion
        for r in reports
        if r.report is not None and isinstance(r.report.report, Observation)
    }
    assert keyed(found["s"], "value") == expected
    assert sorted(cast(float, v) for v in expected.values()) == [0.2, 0.5]


def test_a_transcript_reduces_to_its_highest_score_citing_the_step(
    tmp_path: Path,
) -> None:
    run(tmp_path / "logs", [[call(2), call(5), call(3)]], None)

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        t=as_scanner(by_argument(), results="transcript"),
    )

    [row] = found["t"]
    assert row["value"] == 0.5
    [set_by] = json.loads(row["metadata"])["set_by"]
    assert set_by["stage"] == "tool_call"


def test_failure_count_scanned_at_one_stage_keeps_its_live_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = str(Path(__file__).parents[2] / "examples")
    monkeypatch.setattr(sys, "path", [examples, *sys.path])
    failure_count: Any = importlib.import_module("trajectory").failure_count

    run(
        tmp_path / "logs",
        [[call(-1), call(-2), call(3), call(4)]],
        observe_only(failure_count()),
    )

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        replay=as_scanner(failure_count(), stage="tool_call"),
        read=as_reader("failure_count", stage="tool_call"),
    )

    # the before function's scores rise with failures its after function counted
    replayed, live = keyed(found["replay"], "value"), keyed(found["read"], "value")
    assert replayed == live
    assert sorted(live.values()) == [0.0, 0.2, 0.4, 0.4]


@protocol
def stop_second_generate() -> Any:
    async def model_input(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision:
        if len(step.input) > 2:
            return Decision.terminate("stop before the second generate")
        return Decision.proceed()

    return model_input


def test_the_reader_cites_each_step_and_a_step_with_no_event_cites_nothing(
    tmp_path: Path,
) -> None:
    run(tmp_path / "logs", [[call(1)]], stop_second_generate())

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        read=as_reader(stage="model_input"),
    )

    first, second = found["read"]
    # the generate the step preceded, found by position in a log whose step ids
    # are the last input message's
    assert first["answer"] == "proceed"
    assert len(references(first, "event")) == 1
    # the terminated generate left no model event
    assert second["answer"] == "terminate"
    assert not references(second, "event") and not references(second, "message")


@pytest.mark.parametrize(
    "stage,cites", [("model_output", "message"), ("tool_call", "message")]
)
def test_the_reader_cites_a_generates_output_and_a_calls_result(
    tmp_path: Path, stage: Stage, cites: str
) -> None:
    log = run(tmp_path / "logs", [[call(1)]], observe_only(replay_check()))
    assert log.samples is not None
    messages = {m.id for m in log.samples[0].messages}

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        read=as_reader("replay_check", stage=stage),
    )

    for row in found["read"]:
        cited = references(row, cites)
        assert len(cited) >= 1
        assert cited[0]["id"] in messages


def test_a_failed_monitor_makes_the_transcripts_result_an_error(
    tmp_path: Path,
) -> None:
    @monitor(name="always_fails")
    def always_fails() -> Any:
        async def before(context: Context, step: BeforeToolCall) -> Observation:
            raise RuntimeError("the monitor broke")

        return before

    run(tmp_path / "logs", [[call(1)]], None)

    status = scan(
        scanners={"s": as_scanner(always_fails())},
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )

    assert not status.complete
    [row] = rows(status.location)["s"]
    assert "the monitor broke" in row["scan_error"]


def test_an_escalation_with_no_recorded_answer_fails_closed(tmp_path: Path) -> None:
    run(tmp_path / "logs", [[call(9)]], None)
    sentinel = sequential(
        [threshold(by_argument(), escalate_at=0.5), human(stages=["tool_call"])]
    )

    found = scanned(tmp_path / "scans", tmp_path / "logs", s=as_scanner(sentinel))

    [row] = found["s"]
    assert (row["answer"], row["value"]) == ("reject", True)
    assert (row["metadata.human"], row["metadata.human_answer"]) == (
        "unanswered",
        "reject",
    )
    assert row["metadata.turns"] == "recorded"


def test_steps_are_selected_by_transcript_as_well_as_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # two transcripts share a tool call id; a label names the call in one
    log = run(tmp_path / "logs", [[call(1, "shared")], [call(2, "shared")]], None)
    assert log.samples is not None
    labelled = log.samples[1].uuid
    labels = pandas.DataFrame([{"transcript_id": labelled, "step_id": "shared"}])
    from inspect_sentinel.scout import _scanner

    built: list[str] = []
    rebuild = _scanner.transcript_steps

    def counted(transcript: Transcript, *args: Any) -> Any:
        built.append(transcript.transcript_id)
        return rebuild(transcript, *args)

    monkeypatch.setattr(_scanner, "transcript_steps", counted)

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        s=as_scanner(by_argument(), steps=labels),
    )

    assert keyed(found["s"], "value") == {(labelled, "shared"): 0.2}
    # a transcript the table does not list is not rebuilt
    assert built == [labelled]


@human_stand_in
async def approves(
    step: ReplayStep, escalations: Escalations, choices: Sequence[str]
) -> HumanAnswer:
    return HumanAnswer("approve", "the stand-in approves")


@monitor
def breaks_while(marker: str) -> MonitorGroup:
    # fails while the marker file exists, so a scan can be interrupted
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        if Path(marker).exists():
            raise RuntimeError("interrupted")
        return Observation.score(min(1.0, int(step.call.arguments["x"]) / 10))

    return MonitorGroup(before)


def test_a_resumed_scan_gives_the_rows_a_whole_scan_gives(tmp_path: Path) -> None:
    log = run(tmp_path / "logs", [[call(9, "a"), call(8, "b")], [call(7, "c")]], None)
    assert log.samples is not None
    labels = [(s.uuid or "", "a") for s in log.samples] + [
        (log.samples[1].uuid or "", "c")
    ]
    marker = tmp_path / "marker"

    def scanner() -> Any:
        sentinel = sequential(
            [
                threshold(breaks_while(str(marker)), escalate_at=0.5),
                human(stages=["tool_call"]),
            ]
        )
        return as_scanner(sentinel, steps=labels, human=approves)

    whole = scanned(tmp_path / "whole", tmp_path / "logs", s=scanner())
    marker.touch()
    first = scan(
        scanners={"s": scanner()},
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "resumed"),
        display="none",
        max_processes=1,
    )
    assert not first.complete
    marker.unlink()
    resumed = scan_resume(first.location, display="none")

    assert resumed.complete
    assert keyed(rows(resumed.location)["s"], "answer") == keyed(whole["s"], "answer")
    assert keyed(whole["s"], "answer") == {
        (log.samples[0].uuid, "a"): "proceed",
        (log.samples[1].uuid, "c"): "proceed",
    }
    assert {row["metadata.human"] for row in whole["s"]} == {"simulated"}


def anything(*args: Any) -> Any:
    return True


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"steps": anything}, "registered with @step_selector"),
        ({"human": anything}, "registered with @human_stand_in"),
    ],
)
def test_a_scan_takes_only_what_it_can_record(
    kwargs: dict[str, Any], error: str
) -> None:
    with pytest.raises(TypeError, match=error):
        as_scanner(by_argument(), **kwargs)


@pytest.mark.parametrize(
    "sentinel,kwargs,error",
    [
        (lambda: replay_check(), {"stage": "nowhere"}, "does not watch"),
        (lambda: [by_argument(), replay_check()], {}, "one monitor at a time"),
    ],
)
def test_a_scan_needs_one_subject_at_a_stage_it_watches(
    sentinel: Any, kwargs: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        as_scanner(sentinel(), **kwargs)


def test_a_selector_must_be_importable_to_register() -> None:
    def nested(transcript_id: str, step: ReplayStep) -> bool:
        return True

    with pytest.raises(TypeError, match="module level"):
        step_selector(nested)


@monitor
def conversations() -> MonitorGroup:
    # which conversation each generate is in
    async def model_input(context: Context, step: BeforeGenerate) -> Observation:
        return Observation(suspicion=0.0, metadata={"conversation": step.conversation})

    return MonitorGroup(model_input)


@solver
def two_branches() -> Solver:
    """Two attempts at the task in one conversation, each in a span of its own, as Petri's rollbacks are."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for branch in ("first", "second"):
            async with span(branch, type="trajectory"):
                state.output = await get_model().generate(
                    [ChatMessageUser(content=f"The {branch} attempt.")]
                )
        return state

    return solve


@pytest.mark.parametrize("origins", [True, False])
def test_a_petri_shaped_log_replays_one_conversation_per_branch(
    tmp_path: Path, origins: bool
) -> None:
    # Petri stores a "target" timeline whose branches are the rolled-back
    # attempts, which the span tree alone puts in one conversation
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=two_branches()),
        model="mockllm/model",
        log_dir=str(tmp_path / "run"),
        display="none",
    )
    if not origins:
        without_origins(log)
    assert log.samples is not None
    sample = log.samples[0]
    spans = {e.name: e.id for e in sample.events if isinstance(e, SpanBeginEvent)}
    first, second = [
        [
            TimelineEvent(event=e)
            for e in sample.events
            if isinstance(e, ModelEvent) and e.span_id == spans[branch]
        ]
        for branch in ("first", "second")
    ]
    sample.timelines = [
        Timeline(
            name="target",
            description="",
            root=TimelineSpan(
                id="target",
                name="target",
                content=list(first),
                branches=[
                    TimelineSpan(id="rollback", name="rollback", content=list(second))
                ],
            ),
        )
    ]
    write_eval_log(log, str(tmp_path / "logs" / "petri.eval"))

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        stored=as_scanner(conversations(), timeline="target"),
        tree=as_scanner(conversations()),
    )

    def conversation(row: Row) -> str:
        return row["metadata.metadata"]["conversation"]

    assert [conversation(r) for r in found["stored"]] == ["target", "rollback"]
    assert {r["metadata.turns"] for r in found["stored"]} == {"stored"}
    assert {conversation(r) for r in found["tree"]} == {sample.uuid}


@solver
def nested_agents() -> Solver:
    """A generate in an agent, and one in an agent inside it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with span("outer", type="agent"):
            await get_model().generate([ChatMessageUser(content="outer")])
            async with span("inner", type="agent"):
                state.output = await get_model().generate(
                    [ChatMessageUser(content="inner")]
                )
        return state

    return solve


def test_depth_and_labels_choose_a_sub_agents_steps(tmp_path: Path) -> None:
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=nested_agents()),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.samples is not None
    sample = log.samples[0]
    spans = {e.name: e.id for e in sample.events if isinstance(e, SpanBeginEvent)}
    inner = next(
        e.uuid
        for e in sample.events
        if isinstance(e, ModelEvent) and e.span_id == spans["inner"]
    )
    labels = [(sample.uuid or "", inner or "")]

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        top=as_scanner(conversations(), depth=1),
        all=as_scanner(conversations()),
        labelled=as_scanner(conversations(), steps=labels),
    )

    def seen(rows: list[Row]) -> list[str]:
        return [r["metadata.metadata"]["conversation"] for r in rows]

    assert seen(found["top"]) == [spans["outer"]]
    assert seen(found["all"]) == [spans["outer"], spans["inner"]]
    assert [r["metadata.depth"] for r in found["all"]] == [1, 2]
    assert seen(found["labelled"]) == [spans["inner"]]


def test_the_probe_reads_only_the_tip_of_each_conversation_list() -> None:
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall, ToolCallView

    from inspect_sentinel import HumanDecision, Reported

    decisions = View(
        [
            HumanDecision(
                stage="tool_call",
                step_id=f"c{i}",
                call_ids=(f"c{i}",),
                decided=Reported("human", "human", Decision.proceed(), "tool_call"),
            )
            for i in range(1000)
        ]
    )
    history = View([ChatMessageAssistant(content=f"turn {i}") for i in range(1000)])
    step = BeforeToolCall(
        conversation="c",
        message="",
        message_id="m",
        call=ToolCall(id="c", function="addition", arguments={"x": 1, "y": 1}),
        view=ToolCallView(),
        input=None,
        agent_messages=None,
        history=history,
        human_decisions=decisions,
    )
    replayed = ReplayStep("c", "tool_call", step, ("c", 0))

    anyio.run(replay_monitor, replay_check(), [replayed])

    assert decisions.reads <= 2 and history.reads <= 2


def test_a_scan_gives_rows_at_every_watched_stage_unless_one_is_named(
    tmp_path: Path,
) -> None:
    log = run(tmp_path / "logs", [[call(1)]], observe_only(replay_check()))
    assert log.samples is not None
    recorded = [e for e in log.samples[0].events if isinstance(e, SentinelEvent)]

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        every=as_scanner(replay_check()),
        named=as_scanner(replay_check(), stage="tool_result"),
        tools=as_scanner(by_argument()),
    )

    # two generates, the call's two steps
    assert sorted(r["metadata.stage"] for r in found["every"]) == [
        "model_input",
        "model_input",
        "model_output",
        "model_output",
        "tool_call",
        "tool_result",
    ]
    # a monitor that watches one stage gives rows at it only
    assert [r["metadata.stage"] for r in found["tools"]] == ["tool_call"]
    [row] = found["named"]
    assert (row["metadata.stage"], row["metadata.turns"]) == ("tool_result", "recorded")
    assert row["metadata.step_id"] in {e.step_id for e in recorded}
    assert row["value"] == 0.0


@scanner(messages="all")
def sample_identity() -> Scanner[Transcript]:
    # the sample a transcript's context is read from, as the replay scanner
    # reads it
    contexts = Contexts()

    async def scan(transcript: Transcript) -> Result:
        context = await contexts.eval_context(transcript)
        assert context is not None
        return Result(
            value=repr(context.sample_id),
            metadata={"epoch": context.epoch, "input": context.sample_input},
        )

    return scan


def test_a_sample_without_a_uuid_gets_its_context_by_its_typed_id(
    tmp_path: Path,
) -> None:
    log = run(tmp_path / "logs", [[call(1)], [call(2)]], None)
    assert log.samples is not None
    for sample in log.samples:
        sample.uuid = None
    write_eval_log(log, str(tmp_path / "old" / "old.eval"))

    found = scanned(tmp_path / "scans", tmp_path / "old", identity=sample_identity())

    read = {
        (row["value"], *json.loads(row["metadata"]).values())
        for row in found["identity"]
    }
    assert read == {(repr(s.id), s.epoch, s.input) for s in log.samples}


AUTHOR_MODULE = """
from pathlib import Path

from inspect_sentinel import BeforeToolCall, Context, MonitorGroup, Observation, monitor


@monitor
def authors_monitor(marker: str) -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        if Path(marker).exists():
            raise RuntimeError("interrupted")
        return Observation.score(min(1.0, int(step.call.arguments["x"]) / 10))

    return MonitorGroup(before)
"""


def test_a_scan_resumes_in_a_process_that_never_imported_the_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = run(tmp_path / "logs", [[call(9, "a"), call(4, "b")]], None)
    assert log.samples is not None
    authors = tmp_path / "authors"
    authors.mkdir()
    (authors / "author_monitors.py").write_text(AUTHOR_MODULE)
    monkeypatch.setattr(sys, "path", [str(authors), *sys.path])
    module: Any = importlib.import_module("author_monitors")
    marker = tmp_path / "marker"
    marker.touch()
    sentinel = sequential(
        [
            threshold(module.authors_monitor(str(marker)), escalate_at=0.5),
            human(stages=["tool_call"]),
        ]
    )
    first = scan(
        scanners={"s": as_scanner(sentinel, human=approves)},
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )
    assert not first.complete
    marker.unlink()

    # a fresh interpreter, whose path holds neither the author's module nor
    # anything this test imported
    resumed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlib.util import find_spec; "
            "from inspect_scout import scan_resume; "
            "print(find_spec('author_monitors') is None); "
            "print(scan_resume(sys.argv[1], display='none').complete)",
            first.location,
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[2],
    )

    assert resumed.returncode == 0, resumed.stderr[-3000:]
    # the module was not on the child's path; the scan imported it from its file
    assert resumed.stdout.split()[-2:] == ["True", "True"]
    uuid = log.samples[0].uuid
    found = rows(first.location)["s"]
    assert keyed(found, "value") == {(uuid, "a"): False, (uuid, "b"): False}
    assert keyed(found, "answer") == {(uuid, "a"): "proceed", (uuid, "b"): "proceed"}


@pytest.mark.parametrize(
    "form,label",
    [("bare", "breaks_while"), ("list", "breaks_while"), ("mapping", "risk")],
)
@pytest.mark.parametrize("fails", [False, True])
def test_a_lone_monitor_reports_and_fails_alike_bare_or_in_a_container(
    tmp_path: Path, form: str, label: str, fails: bool
) -> None:
    run(tmp_path / "logs", [[call(8)]], None)
    marker = tmp_path / "marker"
    if fails:
        marker.touch()
    instance = breaks_while(str(marker))
    sentinel: Sentinels = {
        "bare": instance,
        "list": [instance],
        "mapping": {"risk": instance},
    }[form]

    status = scan(
        scanners={"s": as_scanner(sentinel)},
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )

    [row] = rows(status.location)["s"]
    if fails:
        assert not status.complete
        assert "interrupted" in row["scan_error"]
    else:
        assert status.complete
        assert (row["value"], row["label"]) == (0.8, label)


def scan_rows(
    logs: Path, scans: Path, processes: int | None, **scanners: Any
) -> tuple[bool, dict[str, list[Row]]]:
    status = scan(
        scanners=scanners,
        transcripts=transcripts_from(str(logs)),
        scans=str(scans),
        display="none",
        max_processes=processes,
    )
    return status.complete, rows(status.location)


@pytest.mark.parametrize("processes", [2, None])
def test_a_scan_in_worker_processes_gives_the_rows_one_process_gives(
    tmp_path: Path, processes: int | None
) -> None:
    log = run(
        tmp_path / "logs",
        [[call(9, "a"), call(4, "b")], [call(8, "c")], [call(2, "d")]],
        None,
    )
    assert log.samples is not None
    labels = [(s.uuid or "", "a") for s in log.samples] + [
        (log.samples[1].uuid or "", "c"),
        (log.samples[2].uuid or "", "d"),
    ]

    def scanner() -> Any:
        sentinel = sequential(
            [threshold(by_argument(), escalate_at=0.5), human(stages=["tool_call"])]
        )
        return as_scanner(sentinel, steps=labels, human=approves)

    one, single = scan_rows(tmp_path / "logs", tmp_path / "one", 1, s=scanner())
    many, workers = scan_rows(
        tmp_path / "logs", tmp_path / "many", processes, s=scanner()
    )

    assert one and many
    for column in ("answer", "metadata.human"):
        assert keyed(workers["s"], column) == keyed(single["s"], column)
    assert len(single["s"]) == 3


def test_a_monitor_typed_into_python_c_scans(tmp_path: Path) -> None:
    run(tmp_path / "logs", [[call(3)]], None)
    code = f"""
from inspect_scout import scan, transcripts_from
from inspect_sentinel import BeforeToolCall, Context, MonitorGroup, Observation, monitor
from inspect_sentinel.scout import as_scanner

@monitor
def typed_in() -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.3)

    return MonitorGroup(before)

status = scan(
    scanners={{"s": as_scanner(typed_in())}},
    transcripts=transcripts_from({str(tmp_path / "logs")!r}),
    scans={str(tmp_path / "scans")!r},
    display="none",
    max_processes=1,
)
print(status.complete)
"""

    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path
    )

    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.split()[-1] == "True"


SCRIPT = """
import sys
from pathlib import Path

from inspect_scout import scan, transcripts_from
from inspect_sentinel import BeforeToolCall, Context, MonitorGroup, Observation, monitor
from inspect_sentinel.scout import as_scanner

# a side effect of running the script, which must happen once
with open(sys.argv[1], "a") as counter:
    counter.write("ran\\n")


@monitor
def scripted() -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.3)

    return MonitorGroup(before)


if __name__ == "__main__":
    status = scan(
        scanners={"s": as_scanner(scripted())},
        transcripts=transcripts_from(sys.argv[2]),
        scans=sys.argv[3],
        display="none",
        max_processes=1,
    )
    print(status.complete)
"""


def test_a_scripts_top_level_code_runs_once(tmp_path: Path) -> None:
    run(tmp_path / "logs", [[call(3)]], None)
    (tmp_path / "authoring.py").write_text(SCRIPT)
    counter = tmp_path / "counter"

    done = subprocess.run(
        [
            sys.executable,
            "authoring.py",
            str(counter),
            str(tmp_path / "logs"),
            str(tmp_path / "scans"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.split()[-1] == "True"
    assert counter.read_text().splitlines() == ["ran"]


def test_a_failure_a_protocol_went_on_without_fails_both_modes(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    marker.touch()

    def sentinel() -> Any:
        return sequential(
            [breaks_while(str(marker)), threshold(by_argument(), reject_at=0.9)]
        )

    run(tmp_path / "logs", [[call(2)]], sentinel())

    status = scan(
        scanners={
            "replay": as_scanner(sentinel()),
            "read": as_reader(stage="tool_call"),
        },
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )

    assert not status.complete
    found = rows(status.location)
    for name in ("replay", "read"):
        [row] = found[name]
        assert "interrupted" in row["scan_error"], name


@pytest.mark.parametrize(
    "code",
    [
        "from inspect_ai._util.entrypoints import ensure_entry_points; ensure_entry_points()",
        "from inspect_ai import Task, eval; from inspect_ai.dataset import Sample; "
        "eval(Task(dataset=[Sample(input='x')]), model='mockllm/model', display='none', log_dir=sys.argv[1])",
    ],
)
def test_loading_inspect_ai_entry_points_does_not_load_scout(
    tmp_path: Path, code: str
) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; {code}; print('inspect_scout' in sys.modules)",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.split()[-1] == "False"


@pytest.mark.parametrize("model_events", [True, False])
def test_a_log_from_before_origins_joins_read_mode_and_keeps_its_answers(
    tmp_path: Path, model_events: bool
) -> None:
    log = run(tmp_path / "run", [[call(9, "a")]], observe_only(replay_check()))
    assert log.samples is not None
    sample = log.samples[0]
    # as an inspect_ai of that time wrote it: no origins, and the generate
    # stages keyed by the last input message's id and the output message's;
    # without model events, it replays from its messages
    turns = {e.uuid: e for e in sample.events if isinstance(e, ModelEvent) and e.uuid}
    for event in sample.events:
        if isinstance(event, ModelEvent):
            event.origin = None
        elif isinstance(event, SentinelEvent) and event.step_id in turns:
            turn = turns[event.step_id]
            event.step_id = (
                turn.input[-1].id or ""
                if event.stage == "model_input"
                else turn.output.message.id or ""
            )
    # a person rejected the call through human(), as a sentinel recorded it:
    # in the sentinel span of the call's tool_call step
    at, step = next(
        (i, e)
        for i, e in enumerate(sample.events)
        if isinstance(e, SentinelEvent) and (e.stage, e.step_id) == ("tool_call", "a")
    )
    sample.events.insert(
        at + 1,
        SentinelEvent(
            span_id=step.span_id,
            factory="inspect_sentinel/human",
            path="human",
            function="tool_call",
            step_id="a",
            conversation=sample.uuid or "",
            stage="tool_call",
            kind="decision",
            status="reported",
            action="reject",
            explanation="not that",
        ),
    )
    if not model_events:
        sample.events = [e for e in sample.events if not isinstance(e, ModelEvent)]
    write_eval_log(log, str(tmp_path / "logs" / "old.eval"))
    escalating = sequential(
        [threshold(by_argument(), escalate_at=0.5), human(stages=["tool_call"])]
    )

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        replay=as_scanner(replay_check(), stage="model_input"),
        read=as_reader("replay_check", stage="model_input"),
        human=as_scanner(escalating),
    )

    assert (
        keyed(found["replay"], "value").keys() == keyed(found["read"], "value").keys()
    )
    for kind in ("message", "event"):
        cited = {k: references(r, kind) for k, r in by_key(found["replay"]).items()}
        assert cited == {
            k: references(r, kind) for k, r in by_key(found["read"]).items()
        }
    if model_events:
        # read mode cites the step's model event, which a transcript without
        # them does not have
        assert all(references(r, "event") for r in found["read"])
    [row] = found["human"]
    assert (row["answer"], row["metadata.human"], row["metadata.turns"]) == (
        "reject",
        "recorded",
        "inferred",
    )


@solver
def own_calls_and_an_agent() -> Solver:
    """The sample's own generate, then one in a top-level agent."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await get_model().generate([ChatMessageUser(content="mine")])
        async with span("worker", type="agent"):
            state.output = await get_model().generate(
                [ChatMessageUser(content="the worker's")]
            )
        return state

    return solve


def test_depth_counts_spans_that_make_model_calls_as_scout_does(tmp_path: Path) -> None:
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=own_calls_and_an_agent()),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.samples is not None
    sample = log.samples[0]
    worker = next(
        e.id
        for e in sample.events
        if isinstance(e, SpanBeginEvent) and e.name == "worker"
    )

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        one=as_scanner(conversations(), depth=1),
        two=as_scanner(conversations(), depth=2),
        all=as_scanner(conversations()),
    )

    def seen(found: list[Row]) -> list[str]:
        return [r["metadata.metadata"]["conversation"] for r in found]

    # Scout's timeline makes the worker its top span and leaves the sample's
    # lone call, which makes no tool calls, out of every span it scans, so
    # llm_scanner(depth=) never reaches it; a count of agent spans would
    assert seen(found["all"]) == [sample.uuid, worker]
    assert seen(found["one"]) == [worker]
    assert seen(found["two"]) == [worker]


@solver
def a_helper_with_a_prompt_of_its_own() -> Solver:
    """An agent's tool loop, and inside it a one-turn helper agent with a system prompt of its own."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with span("agent", type="agent"):
            messages: list[ChatMessage] = [
                ChatMessageSystem(content="You add numbers."),
                ChatMessageUser(content="add"),
            ]
            output = await get_model().generate(messages, tools=[addition()])
            messages.append(output.message)
            messages.extend((await execute_tools(messages, [addition()])).messages)
            async with span("helper", type="agent"):
                await get_model().generate(
                    [
                        ChatMessageSystem(content="You summarise."),
                        ChatMessageUser(content="summarise"),
                    ]
                )
            state.output = await get_model().generate(messages, tools=[addition()])
        return state

    return solve


def test_depth_reads_a_transcripts_model_inputs_as_scout_does(tmp_path: Path) -> None:
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=a_helper_with_a_prompt_of_its_own()),
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                call(2),
                ModelOutput.from_content("mockllm/model", "a summary"),
                ModelOutput.from_content("mockllm/model", "done"),
            ],
        ),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.samples is not None
    sample = log.samples[0]
    # no stored timeline, so depth= builds one from the transcript's events
    assert not sample.timelines
    spans = {e.name: e.id for e in sample.events if isinstance(e, SpanBeginEvent)}

    def walked(events: list[Event]) -> set[str]:
        return {s.id for s in scout_walk_spans(timeline_build(events).root, depth=2)}

    # Scout's timeline tells the helper is a utility agent by its own system
    # prompt, so llm_scanner(depth=) never reaches it; read without its model
    # inputs, as condensed events hold them, the helper would be reached
    assert walked(sample.events) == {spans["agent"]}
    assert walked(condense_events(sample.events)[0]) == {
        spans["agent"],
        spans["helper"],
    }

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        two=as_scanner(conversations(), depth=2),
        all=as_scanner(conversations()),
    )

    def seen(found: list[Row]) -> set[str]:
        return {r["metadata.metadata"]["conversation"] for r in found}

    # the same holds for a transcript handed to the scanner directly
    handed = Transcript(
        transcript_id=sample.uuid or "",
        messages=sample.messages,
        events=sample.events,
    )
    results = anyio.run(as_scanner(conversations(), depth=2), handed)
    assert isinstance(results, list)
    assert seen(found["all"]) == {spans["agent"], spans["helper"]}
    assert seen(found["two"]) == {spans["agent"]}
    assert {r.metadata["metadata"]["conversation"] for r in results if r.metadata} == {
        spans["agent"]
    }


def test_depth_without_timeline_is_the_span_trees_whatever_is_stored(
    tmp_path: Path,
) -> None:
    # the steps' conversations come from the span tree unless timeline= names
    # a stored timeline, so their depth does too, even where the log stores a
    # timeline that puts every model call in a branch, which no depth reaches
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=nested_agents()),
        model="mockllm/model",
        log_dir=str(tmp_path / "run"),
        display="none",
    )
    assert log.samples is not None
    sample = log.samples[0]
    calls: list[TimelineEvent | TimelineSpan] = [
        TimelineEvent(event=e) for e in sample.events if isinstance(e, ModelEvent)
    ]
    sample.timelines = [
        Timeline(
            name="target",
            description="",
            root=TimelineSpan(
                id="target",
                name="target",
                content=[],
                branches=[TimelineSpan(id="rollback", name="rollback", content=calls)],
            ),
        )
    ]
    write_eval_log(log, str(tmp_path / "logs" / "stored.eval"))

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        top=as_scanner(conversations(), depth=1),
        all=as_scanner(conversations()),
        stored=as_scanner(conversations(), timeline="target", depth=1),
    )

    assert [r["metadata.depth"] for r in found["all"]] == [1, 2]
    assert [r["metadata.depth"] for r in found["top"]] == [1]
    assert found["stored"] == []
    # a transcript handed to the scanner directly, with its stored timeline
    handed = Transcript(
        transcript_id=sample.uuid or "",
        messages=sample.messages,
        events=sample.events,
        timelines=sample.timelines,
    )
    results = anyio.run(as_scanner(conversations(), depth=1), handed)
    assert isinstance(results, list)
    assert [r.metadata["depth"] for r in results if r.metadata] == [1]


def test_depth_builds_one_timeline_per_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect_ai.event

    from inspect_sentinel.scout import _steps

    run(tmp_path / "logs", [[call(1)], [call(2)], [call(3)]], None)
    built = 0
    build = inspect_ai.event.timeline_build

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal built
        built += 1
        return build(*args, **kwargs)

    # Scout's own build imports it from inspect_ai.event when it runs
    monkeypatch.setattr(inspect_ai.event, "timeline_build", counted)
    monkeypatch.setattr(_steps, "timeline_build", counted)

    found = scanned(
        tmp_path / "scans", tmp_path / "logs", s=as_scanner(by_argument(), depth=1)
    )

    assert len(found["s"]) == 3
    assert built == 3


@tool
def long_output() -> Tool:
    async def execute(x: int) -> str:
        """Return a long output.

        Args:
            x: Any number.
        """
        return "result " * 120

    return execute


class Counted(StoreModel):
    steps: int = 0


@monitor
def counts_calls() -> MonitorGroup:
    # how many tool calls the monitor's store has seen, this one included
    async def tool_call(context: Context, step: BeforeToolCall) -> Observation:
        counted = context.store_as(Counted)
        counted.steps += 1
        return Observation(suspicion=0.0, metadata={"steps": counted.steps})

    return MonitorGroup(tool_call)


@step_selector
def after_four_compactions(transcript_id: str, step: ReplayStep) -> bool:
    return step.segment[1] >= 4


def test_state_carries_across_compactions_and_a_selector_picks_segments(
    tmp_path: Path,
) -> None:
    from inspect_ai.agent import react
    from inspect_ai.model import CompactionTrim

    outputs = [
        ModelOutput.for_tool_call("mockllm/model", "long_output", {"x": i})
        for i in range(6)
    ] + [ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"})]
    eval(
        Task(
            dataset=[Sample(input="go")],
            solver=react(
                tools=[long_output()], compaction=CompactionTrim(threshold=700)
            ),
        ),
        model=get_model("mockllm/model", custom_outputs=outputs),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        all=as_scanner(counts_calls()),
        latest=as_scanner(counts_calls(), steps=after_four_compactions),
    )

    def counts(rows: list[Row]) -> list[int]:
        return [r["metadata.metadata"]["steps"] for r in rows]

    # five segments: three calls before the first compaction, one after each;
    # one store for the whole sample, as live, so the count runs across them
    assert counts(found["all"]) == list(range(1, 8))
    # the steps not chosen are not run, so the store sees only the last
    assert counts(found["latest"]) == [1]


def test_a_protocol_reduces_to_its_strongest_decision(tmp_path: Path) -> None:
    run(tmp_path / "logs", [[call(2), call(9), call(3)]], None)

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        t=as_scanner(threshold(by_argument(), reject_at=0.5), results="transcript"),
    )

    [row] = found["t"]
    assert (row["answer"], row["value"]) == ("reject", True)


def test_a_label_names_one_transcripts_sub_agent_step_of_a_shared_call_id(
    tmp_path: Path,
) -> None:
    log = run(
        tmp_path / "logs",
        [[call(1)], [call(2)]],
        None,
        helpers=[call(5, "shared"), call(6, "shared")],
    )
    assert log.samples is not None
    labelled = log.samples[1].uuid

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        s=as_scanner(by_argument(), steps=[(labelled or "", "shared")]),
    )

    assert keyed(found["s"], "value") == {(labelled, "shared"): 0.6}


DRIVING_SCRIPT = """
import sys
from pathlib import Path

from inspect_scout import scan, transcripts_from
from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Escalations,
    HumanAnswer,
    MonitorGroup,
    Observation,
    ReplayStep,
    human,
    monitor,
    sequential,
    threshold,
)
from inspect_sentinel.scout import as_scanner, human_stand_in, step_selector


@monitor
def scripted_score(marker: str) -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        if Path(marker).exists():
            raise RuntimeError("interrupted")
        return Observation.score(min(1.0, int(step.call.arguments["x"]) / 10))

    return MonitorGroup(before)


@step_selector
def calls_named_a(transcript_id: str, step: ReplayStep) -> bool:
    return step.id.startswith("a")


@human_stand_in
async def scripted_person(
    step: ReplayStep, escalations: Escalations, choices: list[str]
) -> HumanAnswer:
    return HumanAnswer("reject", "the script's person")


if __name__ == "__main__":
    logs, scans, marker, processes = sys.argv[1:5]
    sentinel = sequential(
        [threshold(scripted_score(marker), escalate_at=0.5), human(stages=["tool_call"])]
    )
    status = scan(
        scanners={
            "s": as_scanner(sentinel, steps=calls_named_a, human=scripted_person)
        },
        transcripts=transcripts_from(logs),
        scans=scans,
        display="none",
        max_processes=int(processes),
    )
    print(status.complete, status.location)
"""


def drive(tmp_path: Path, *args: str, cwd: Path) -> list[str]:
    done = subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=cwd
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout.split()


def test_a_scripts_own_selector_and_stand_in_run_in_workers_and_resume(
    tmp_path: Path,
) -> None:
    log = run(
        tmp_path / "logs", [[call(9, "a1"), call(8, "b1")], [call(7, "a2")]], None
    )
    assert log.samples is not None
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "driving.py").write_text(DRIVING_SCRIPT)
    marker = tmp_path / "marker"
    expected = {
        (log.samples[0].uuid, "a1"): "reject",
        (log.samples[1].uuid, "a2"): "reject",
    }

    def scanned_by_script(scans: str, processes: int) -> list[str]:
        return drive(
            tmp_path,
            "driving.py",
            str(tmp_path / "logs"),
            str(tmp_path / scans),
            str(marker),
            str(processes),
            cwd=scripts,
        )

    # in two worker processes
    complete, location = scanned_by_script("workers", 2)
    assert complete == "True"
    found = rows(location)["s"]
    assert keyed(found, "answer") == expected
    assert {r["metadata.human"] for r in found} == {"simulated"}

    # interrupted, then resumed by an interpreter that never ran the script
    marker.touch()
    complete, location = scanned_by_script("resumed", 1)
    assert complete == "False"
    marker.unlink()
    [complete] = drive(
        tmp_path,
        "-c",
        "import sys; from inspect_scout import scan_resume; "
        "print(scan_resume(sys.argv[1], display='none').complete)",
        location,
        cwd=scripts,
    )
    assert complete == "True"
    assert keyed(rows(location)["s"], "answer") == expected


def test_a_stand_in_typed_into_python_c_is_refused() -> None:
    code = """
from inspect_sentinel import HumanAnswer, human
from inspect_sentinel.scout import as_scanner, human_stand_in

@human_stand_in
async def typed_in(step, escalations, choices):
    return HumanAnswer("approve")

as_scanner(human(stages=["tool_call"]), human=typed_in)
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert done.returncode != 0
    assert "notebook, REPL or `python -c`" in done.stderr


@monitor
def ties() -> MonitorGroup:
    # the same score at every step, as a benign transcript gives
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        return Observation.score(0.0)

    return MonitorGroup(before)


@monitor
def sparse() -> MonitorGroup:
    # a dimension of its own at every step
    async def before(context: Context, step: BeforeToolCall) -> Observation:
        return Observation(suspicion={f"d_{step.call.id}": 0.5})

    return MonitorGroup(before)


def messages(turns: int) -> list[ChatMessage]:
    found: list[ChatMessage] = [ChatMessageUser(id="task", content="add")]
    for turn in range(turns):
        made = call(turn, f"c{turn}").message
        made.id = f"a{turn}"
        found.append(made)
        found.append(
            ChatMessageTool(id=f"r{turn}", content="1", tool_call_id=f"c{turn}")
        )
    return found


def work(scanner: Any, turns: int) -> int:
    # every Python call and generator resume while the scanner reads one
    # transcript of messages: the whole pipeline's work, end to end
    from inspect_scout import Transcript

    transcript = Transcript(transcript_id="t", messages=messages(turns))
    calls = 0

    def count(frame: Any, event: str, arg: Any) -> None:
        nonlocal calls
        if event == "call":
            calls += 1

    async def scanned() -> None:
        sys.setprofile(count)
        try:
            await scanner(transcript)
        finally:
            sys.setprofile(None)

    anyio.run(scanned)
    return calls


@pytest.mark.parametrize(
    "sentinel",
    [
        lambda: ties(),
        lambda: sparse(),
        lambda: threshold(ties(), reject_at=0.9),
    ],
    ids=["tied scores", "sparse dimensions", "tied actions"],
)
def test_a_transcript_reduction_does_work_in_proportion_to_its_steps(
    sentinel: Any,
) -> None:
    scanner = as_scanner(sentinel(), results="transcript")
    # the first transcript rebuilds the sentinel, once per process
    work(scanner, 1)

    assert work(scanner, 400) <= 2.2 * work(scanner, 200)


def as_scout_reads(turns: int) -> Transcript:
    # a transcript as Scout's read hands it to a scanner: each model event's
    # input a list of the conversation's shared message objects
    found = messages(turns)
    events: list[Event] = []
    for turn in range(turns):
        made = found[1 + 2 * turn]
        assert isinstance(made, ChatMessageAssistant)
        events.append(
            ModelEvent(
                model="mockllm/model",
                input=found[: 1 + 2 * turn],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
                output=ModelOutput(
                    model="mockllm/model",
                    choices=[
                        ChatCompletionChoice(message=made, stop_reason="tool_calls")
                    ],
                ),
                origin="agent",
            )
        )
    return Transcript(transcript_id="t", messages=found, events=events)


def test_replay_pools_scouts_model_inputs_in_work_linear_in_their_entries() -> None:
    # Scout's read holds n turns' model inputs as about n²/2 list entries, and
    # pooling them reads each entry once: linear in the entries, so quadratic
    # in turns, though nothing is decoded or copied again (design/pr-series.md,
    # Deferred)
    def pooled(turns: int) -> tuple[int, int]:
        transcript = as_scout_reads(turns)
        entries = sum(
            len(e.input) for e in transcript.events if isinstance(e, ModelEvent)
        )
        calls = 0

        def count(frame: Any, event: str, arg: Any) -> None:
            nonlocal calls
            if event in ("call", "c_call"):
                calls += 1

        sys.setprofile(count)
        try:
            transcript_steps(
                transcript, EvalInfo(inspect_ai_version=None, sentinel=False), None
            )
        finally:
            sys.setprofile(None)
        return calls, entries

    (small, small_entries), (large, large_entries) = pooled(200), pooled(400)
    assert large_entries > 3.9 * small_entries
    assert large / large_entries <= 1.05 * small / small_entries


def test_transcripts_from_one_log_share_its_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_scout import Transcript

    from inspect_sentinel.scout import _context

    log = run(tmp_path / "logs", [[call(i)] for i in range(5)], None)
    assert log.samples is not None
    path = str(log.location)
    reads = {"header": 0, "summaries": 0}
    failures = [RuntimeError("the first header read fails")]
    from inspect_ai.log import (
        read_eval_log_async as header,
    )
    from inspect_ai.log import (
        read_eval_log_sample_summaries_async as summaries,
    )

    async def read_header(log: str, header_only: bool) -> Any:
        reads["header"] += 1
        await anyio.sleep(0)  # let the other transcripts reach the cache
        if failures:
            raise failures.pop()
        return await header(log, header_only=header_only)

    async def read_summaries(log: str) -> Any:
        reads["summaries"] += 1
        await anyio.sleep(0)
        return await summaries(log)

    monkeypatch.setattr(_context, "read_eval_log_async", read_header)
    monkeypatch.setattr(
        _context, "read_eval_log_sample_summaries_async", read_summaries
    )
    contexts = _context.Contexts()
    transcripts = [
        Transcript(transcript_id=s.uuid or "", source_type="eval_log", source_uri=path)
        for s in log.samples
    ]
    found: dict[str, Any] = {}

    async def read_all() -> None:
        async def one(transcript: Transcript) -> None:
            found[transcript.transcript_id] = await contexts.eval_context(transcript)

        async with anyio.create_task_group() as tg:
            for transcript in transcripts:
                tg.start_soon(one, transcript)

    # the failed read reaches its caller and leaves nothing cached
    with pytest.raises(BaseExceptionGroup):
        anyio.run(read_all)
    reads.update(header=0, summaries=0)
    anyio.run(read_all)

    assert reads == {"header": 1, "summaries": 1}
    assert [found[s.uuid or ""].sample_id for s in log.samples] == [0, 1, 2, 3, 4]


def test_the_probe_digests_a_samples_context_once_live_and_in_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_sentinel.scout import _probe

    forms = 0
    form = _probe._eval_form

    def counted(context: Any) -> Any:
        nonlocal forms
        forms += 1
        return form(context)

    monkeypatch.setattr(_probe, "_eval_form", counted)
    large: list[ChatMessage] = [
        ChatMessageUser(content=f"part {i}") for i in range(2000)
    ]
    samples = [Sample(id=i, input=large) for i in range(2)]

    # live: inspect_ai builds a new EvalContext at every step it checks
    run(
        tmp_path / "logs",
        [[call(1), call(2), call(3)], [call(4)]],
        observe_only(replay_check()),
        samples,
    )
    live = forms
    forms = 0
    # replayed: two transcripts at once, each one unit
    status = scan(
        scanners={"s": as_scanner(replay_check(), stage="tool_call")},
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
        max_transcripts=2,
    )

    assert status.complete
    assert (live, forms) == (2, 2)


def test_a_results_frame_selects_the_steps_it_holds(tmp_path: Path) -> None:
    run(tmp_path / "logs", [[call(9, "a"), call(2, "b")], [call(7, "c")]], None)
    first = scan(
        scanners={"s": as_scanner(by_argument())},
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "first"),
        display="none",
        max_processes=1,
    )
    frame: Any = scan_results_df(first.location).scanners["s"]
    # the documented flow: filter the results, then replay what it kept
    flagged = frame[frame["value"] >= 0.5]

    found = scanned(
        tmp_path / "again",
        tmp_path / "logs",
        s=as_scanner(by_argument(), steps=flagged),
    )

    assert {step for _, step in keyed(found["s"], "value")} == {"a", "c"}


def test_a_scan_resumes_after_an_eval_loaded_the_entry_points(
    tmp_path: Path,
) -> None:
    run(tmp_path / "logs", [[call(9, "a")]], observe_only(replay_check()))
    marker = tmp_path / "marker"
    marker.touch()
    first = scan(
        scanners={
            "replay": as_scanner(breaks_while(str(marker))),
            "read": as_reader("replay_check", stage="tool_call"),
        },
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )
    assert not first.complete
    marker.unlink()

    # inspect_ai loads sentinel's entry point, as an eval does, before the
    # process imports Scout to resume
    [loaded, complete] = drive(
        tmp_path,
        "-c",
        "import sys; from inspect_ai._util.entrypoints import ensure_entry_points; "
        "ensure_entry_points(); print('inspect_scout' in sys.modules); "
        "from inspect_scout import scan_resume; "
        "print(scan_resume(sys.argv[1], display='none').complete)",
        first.location,
        cwd=Path(__file__).parents[2],
    )

    assert (loaded, complete) == ("False", "True")
    found = rows(first.location)
    assert [r["value"] for r in found["replay"]] == [0.9]
    assert [r["value"] for r in found["read"]] == [0.0]


def test_a_stored_timeline_cites_a_sub_agents_tool_message(tmp_path: Path) -> None:
    log = run(tmp_path / "run", [[call(1, "main")]], None, helpers=[call(5, "helper")])
    assert log.samples is not None
    sample = log.samples[0]
    tools = {e.id: e for e in sample.events if isinstance(e, ToolEvent)}
    # the helper's tool message is in none of the sample's messages
    assert tools["helper"].message_id not in {m.id for m in sample.messages}
    # a stored timeline whose one conversation is none of the span tree's
    sample.timelines = [
        Timeline(
            name="target",
            description="",
            root=TimelineSpan(
                id="target-branch",
                name="target",
                content=[
                    TimelineEvent(event=e)
                    for e in sample.events
                    if isinstance(e, ModelEvent | ToolEvent)
                ],
            ),
        )
    ]
    write_eval_log(log, str(tmp_path / "logs" / "stored.eval"))

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        s=as_scanner(by_argument(), timeline="target"),
    )

    cited = {r["metadata.step_id"]: references(r, "message") for r in found["s"]}
    assert cited == {
        call_id: [{"type": "message", "id": tools[call_id].message_id}]
        for call_id in ("helper", "main")
    }


@monitor(portable=False)
def records_contexts(path: str) -> MonitorGroup:
    # the probe's digest of each context the host hands over, computed whole
    from inspect_sentinel.scout._probe import context_digest

    def write(context: Context) -> Observation:
        with open(path, "a") as recorded:
            recorded.write(context_digest(context.eval) + "\n")
        return Observation.score(0.0)

    async def model_input(context: Context, step: BeforeGenerate) -> Observation:
        return write(context)

    async def model_output(context: Context, step: AfterGenerate) -> Observation:
        return write(context)

    return MonitorGroup(model_input, model_output)


@solver
def changes_its_metadata() -> Solver:
    """Two generates, with the sample's metadata changed between them."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = await get_model().generate(state.messages)
        state.metadata["phase"] = "later"
        state.output = await get_model().generate(state.messages)
        return state

    return solve


def test_the_probe_digests_the_context_each_step_is_handed(tmp_path: Path) -> None:
    handed = tmp_path / "handed"
    [log] = eval(
        Task(
            dataset=[Sample(input="go", metadata={"phase": "initial"})],
            solver=changes_its_metadata(),
            sentinel=observe_only([replay_check(), records_contexts(str(handed))]),
        ),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.samples is not None

    recorded = [
        cast(dict[str, Any], e.metadata)["eval"]
        for e in log.samples[0].events
        if isinstance(e, SentinelEvent) and e.path == "replay_check"
    ]

    assert recorded == handed.read_text().split()
    assert len(set(recorded)) == 2


def test_replay_check_differences_lists_what_replay_did_not_hand_over(
    tmp_path: Path,
) -> None:
    # replay hands every step the sample's final metadata, so the first
    # generate's steps, which ran before the solver changed it, differ
    [log] = eval(
        Task(
            dataset=[Sample(input="go", metadata={"phase": "initial"})],
            solver=changes_its_metadata(),
            sentinel=observe_only(replay_check()),
        ),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    status = scan(
        scanners={
            "replayed": as_scanner(replay_check()),
            "recorded": as_reader("replay_check"),
        },
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )
    frames: Any = scan_results_df(status.location).scanners
    replayed, recorded = frames["replayed"], frames["recorded"]

    differences: Any = replay_check_differences(replayed, recorded)

    assert log.samples is not None
    uuid = log.samples[0].uuid
    first = cast(list[Row], replayed.to_dict("records"))[0]["metadata.step_id"]
    assert [
        (r["transcript_id"], r["step_id"], r["stage"], r["field"], r["reason"])
        for r in cast(list[Row], differences.to_dict("records"))
    ] == [
        (uuid, first, "model_input", "eval", "differs"),
        (uuid, first, "model_output", "eval", "differs"),
    ]
    assert all(differences["replayed"] != differences["recorded"])
    # a step one side lacks
    later = recorded[recorded["metadata.step_id"] != first]
    lacking: Any = replay_check_differences(replayed, later)
    assert set(zip(lacking["stage"], lacking["reason"], strict=True)) == {
        ("model_input", "only replayed"),
        ("model_output", "only replayed"),
    }
    assert replay_check_differences(replayed, replayed).empty


@monitor
def hands_over() -> MonitorGroup:
    # the probe's digest of the context the host hands over, computed whole
    from inspect_sentinel.scout._probe import context_digest

    async def model_input(context: Context, step: BeforeGenerate) -> Observation:
        return Observation(
            suspicion=0.0, metadata={"eval": context_digest(context.eval)}
        )

    return MonitorGroup(model_input)


@pytest.mark.anyio
async def test_the_probe_digests_a_context_a_sidecar_learns_late() -> None:
    from dataclasses import replace

    from inspect_sentinel.sidecar import Handler
    from inspect_sentinel.sidecar._registration import SampleFacts, TaskFacts
    from tests.sidecar._traffic import inspect_headers, request

    handler = Handler(observe_only([replay_check(), hands_over()]))
    task = TaskFacts(
        task="philadelphia",
        task_description="File the weather report.",
        metadata={},
        samples=[
            SampleFacts(
                sample_id=1,
                sample_description=None,
                sample_input="File today's report.",
                metadata={},
            )
        ],
    )

    async def digests(id: str) -> dict[str, Any]:
        await handler.request(replace(request(id), headers=inspect_headers()))
        records = await handler.records(id)
        assert records is not None
        return {
            r.path: cast(Any, r.detail).report.metadata["eval"]
            for r in records
            if r.kind == "record" and r.path in ("replay_check", "hands_over")
        }

    # the run's first request comes before its task is registered
    before = await digests("a")
    await handler.register_task("eval_789", task)
    after = await digests("b")

    assert before["replay_check"] == before["hands_over"]
    assert after["replay_check"] == after["hands_over"]
    assert before["hands_over"] != after["hands_over"]


# logs that do not record where each model call was made (R7 PR 6)


def without_origins(log: EvalLog) -> EvalLog:
    # as an inspect_ai from before ModelEvent.origin wrote it
    for sample in log.samples or []:
        for event in sample.events:
            if isinstance(event, ModelEvent):
                event.origin = None
    return log


@pytest.mark.parametrize(
    "stage", ["model_input", "model_output", "tool_call", "tool_result"]
)
def test_a_log_without_origins_replays_as_one_with_them(
    tmp_path: Path, stage: Stage
) -> None:
    log = run(
        tmp_path / "run",
        [[call(1), call(2)], [call(3)]],
        observe_only(replay_check()),
        helpers=[call(5, f"sub{i}") for i in range(2)],
    )
    write_eval_log(log, str(tmp_path / "recorded" / "log.eval"))
    write_eval_log(without_origins(log), str(tmp_path / "inferred" / "log.eval"))

    found = {
        name: scanned(
            tmp_path / f"scans-{name}",
            tmp_path / name,
            replay=as_scanner(replay_check(), stage=stage),
        )["replay"]
        for name in ("recorded", "inferred")
    }

    recorded = keyed(found["recorded"], "metadata.metadata")
    inferred = keyed(found["inferred"], "metadata.metadata")
    assert recorded and inferred == recorded
    assert {r["metadata.turns"] for r in found["recorded"]} == {"recorded"}
    assert {r["metadata.turns"] for r in found["inferred"]} == {"inferred"}


@monitor
def decisions_so_far() -> MonitorGroup:
    # the human() decisions a generate is handed
    async def model_input(context: Context, step: BeforeGenerate) -> Observation:
        return Observation(
            suspicion=0.0,
            metadata={
                "decisions": [
                    [d.decided.name, d.decided.report.action]
                    for d in step.human_decisions or []
                ]
            },
        )

    return MonitorGroup(model_input)


@approver(name="team_review")
def team_review() -> Approver:
    async def approve(
        message: str, call: ToolCall, view: ToolCallView, history: list[ChatMessage]
    ) -> Approval:
        return Approval(decision="reject", explanation="not ours")

    return approve


@pytest.mark.parametrize("person", ["human", "team_review"])
def test_only_the_built_in_human_approver_is_a_person_in_a_log_without_a_sentinel(
    tmp_path: Path, person: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def acp(**kwargs: object) -> Approval | None:
        return None

    async def panel(
        message: str,
        call: ToolCall,
        view: ToolCallView,
        history: object,
        choices: list[str],
    ) -> Approval:
        return Approval(decision="reject", explanation="not that")

    monkeypatch.setattr(acp_module, "request_human_approval_via_acp", acp)
    monkeypatch.setattr(approver_module, "panel_approval", panel)
    [log] = eval(
        Task(
            dataset=[Sample(input="add")],
            solver=[use_tools(addition()), generate()],
            approval=[
                ApprovalPolicy(
                    human_approver() if person == "human" else team_review(), "*"
                )
            ],
        ),
        model=get_model(
            "mockllm/model",
            custom_outputs=[call(1), ModelOutput.from_content("mockllm/model", "done")],
        ),
        log_dir=str(tmp_path / "run"),
        display="none",
    )
    assert log.status == "success", log.error
    write_eval_log(without_origins(log), str(tmp_path / "logs" / "log.eval"))

    found = scanned(
        tmp_path / "scans", tmp_path / "logs", replay=as_scanner(decisions_so_far())
    )

    last = found["replay"][-1]["metadata.metadata"]["decisions"]
    assert last == ([["human", "reject"]] if person == "human" else [])


def test_a_transcript_without_model_events_replays_from_its_messages(
    tmp_path: Path,
) -> None:
    log = run(tmp_path / "run", [[call(1), call(2)]], None)
    assert log.samples is not None
    sample = log.samples[0]
    sample.events = [e for e in sample.events if not isinstance(e, ModelEvent)]
    write_eval_log(log, str(tmp_path / "logs" / "log.eval"))

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        replay=as_scanner(conversations()),
    )

    generates = [m for m in sample.messages if isinstance(m, ChatMessageAssistant)]
    assert len(found["replay"]) == len(generates)
    assert {r["metadata.turns"] for r in found["replay"]} == {"inferred"}


@solver
def asks_a_helper_then_answers() -> Solver:
    """A helper's call outside the agent's turns, then the agent's own."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        with model_origin("helper"):
            await get_model().generate([ChatMessageUser(content="Summarise the task.")])
        return await generate(state)

    return solve


@pytest.mark.parametrize("helper_origin", [True, False])
def test_a_log_with_a_call_that_records_no_origin_has_inferred_turns(
    tmp_path: Path, helper_origin: bool
) -> None:
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=asks_a_helper_then_answers()),
        model="mockllm/model",
        log_dir=str(tmp_path / "run"),
        display="none",
    )
    assert log.samples is not None
    if not helper_origin:
        # the helper's call alone records no origin, so is inferred
        for event in log.samples[0].events:
            if isinstance(event, ModelEvent) and event.origin == "helper":
                event.origin = None
    write_eval_log(log, str(tmp_path / "logs" / "log.eval"))

    found = scanned(
        tmp_path / "scans", tmp_path / "logs", replay=as_scanner(conversations())
    )

    assert {r["metadata.turns"] for r in found["replay"]} == {
        "recorded" if helper_origin else "inferred"
    }
