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
from inspect_ai.dataset import Sample
from inspect_ai.event import (
    ModelEvent,
    SentinelEvent,
    SpanBeginEvent,
    Timeline,
    TimelineEvent,
    TimelineSpan,
)
from inspect_ai.log import EvalLog, write_eval_log
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util._span import span
from inspect_scout import scan, scan_results_df, scan_resume, transcripts_from

from inspect_sentinel import (
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
    replay_monitor,
    sequential,
    threshold,
)
from inspect_sentinel.scout import (
    Stage,
    as_reader,
    as_scanner,
    fidelity_probe,
    human_stand_in,
    step_selector,
    steps_in,
)
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


def run(
    log_dir: Path,
    turns: Sequence[Sequence[ModelOutput]],
    sentinel: Sentinels | None,
    samples: Sequence[Sample] | None = None,
) -> EvalLog:
    # one sample per list of tool-call outputs, each ending with a reply
    found = (
        list(samples)
        if samples is not None
        else [
            Sample(id=index, input=f"add, task {index}") for index in range(len(turns))
        ]
    )
    outputs = [
        output
        for calls in turns
        for output in [*calls, ModelOutput.from_content("mockllm/model", "done")]
    ]
    [log] = eval(
        Task(
            dataset=found,
            solver=[use_tools(addition()), generate()],
            sentinel=sentinel,
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


def test_the_scout_adapter_imports_only_public_sentinel_modules() -> None:
    # so it can move to inspect_scout unchanged; _integration is the contract
    # sentinel offers other packages, as inspect_ai uses it
    allowed = {"inspect_sentinel", "inspect_sentinel._integration"}
    for source in SCOUT.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.startswith("inspect_sentinel"):
                    assert node.module in allowed, (source.name, node.module)


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
        observe_only(fidelity_probe()),
        samples,
    )

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        replay=as_scanner(fidelity_probe(), stage=stage),
        read=as_reader("fidelity_probe", stage=stage),
    )

    replayed = keyed(found["replay"], "metadata.metadata")
    read = keyed(found["read"], "metadata.metadata")
    assert replayed.keys() == read.keys()
    assert (
        len(read)
        == {"model_input": 7, "model_output": 7, "tool_call": 4, "tool_result": 4}[
            stage
        ]
    )
    for key, recorded in read.items():
        assert replayed[key]["eval"] == recorded["eval"], key
        differ = {
            name
            for name, digest in recorded["step"].items()
            if replayed[key]["step"].get(name) != digest
        }
        assert not differ, (key, differ)


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
        t=as_scanner(by_argument(), unit="transcript"),
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
        replay=as_scanner(failure_count(), stage="tool_call", split=None),
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
        read=as_reader("", stage="model_input"),
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
    log = run(tmp_path / "logs", [[call(1)]], observe_only(fidelity_probe()))
    assert log.samples is not None
    messages = {m.id for m in log.samples[0].messages}

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        read=as_reader("fidelity_probe", stage=stage),
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
    assert row["metadata.human"] == "unanswered"
    assert row["metadata.turns"] == "recorded"


def test_steps_are_selected_by_transcript_as_well_as_step(tmp_path: Path) -> None:
    # two transcripts share a tool call id; a label names the call in one
    log = run(tmp_path / "logs", [[call(1, "shared")], [call(2, "shared")]], None)
    assert log.samples is not None
    labelled = log.samples[1].uuid
    labels = pandas.DataFrame([{"transcript_id": labelled, "step_id": "shared"}])

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        s=as_scanner(by_argument(), steps=steps_in(labels)),
    )

    assert keyed(found["s"], "value") == {(labelled, "shared"): 0.2}


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
    labels = [(s.uuid, "a") for s in log.samples] + [(log.samples[1].uuid, "c")]
    marker = tmp_path / "marker"

    def scanner() -> Any:
        sentinel = sequential(
            [
                threshold(breaks_while(str(marker)), escalate_at=0.5),
                human(stages=["tool_call"]),
            ]
        )
        return as_scanner(sentinel, steps=steps_in(labels), human=approves)

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
        (lambda: fidelity_probe(), {}, "pass stage="),
        (lambda: fidelity_probe(), {"stage": "nowhere"}, "does not watch"),
        (lambda: [by_argument(), fidelity_probe()], {}, "one monitor at a time"),
    ],
)
def test_a_scan_needs_one_subject_at_one_stage(
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


def test_a_petri_shaped_log_replays_one_conversation_per_branch(
    tmp_path: Path,
) -> None:
    # Petri stores a "target" timeline whose branches are the rolled-back
    # attempts, which the span tree alone puts in one conversation
    [log] = eval(
        Task(dataset=[Sample(input="go")], solver=two_branches()),
        model="mockllm/model",
        log_dir=str(tmp_path / "run"),
        display="none",
    )
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
    labels = [(sample.uuid, inner)]

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        top=as_scanner(conversations(), depth=1),
        all=as_scanner(conversations()),
        labelled=as_scanner(conversations(), steps=steps_in(labels)),
    )

    def seen(rows: list[Row]) -> list[str]:
        return [r["metadata.metadata"]["conversation"] for r in rows]

    assert seen(found["top"]) == [spans["outer"]]
    assert seen(found["all"]) == [spans["outer"], spans["inner"]]
    assert seen(found["labelled"]) == [spans["inner"]]


def test_the_probe_reads_only_the_tip_of_each_conversation_list() -> None:
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall, ToolCallView

    from inspect_sentinel import Host, HumanDecision, Reported
    from tests._fakes import FakeHost

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
        agent_messages=history,
        history=history,
        human_decisions=decisions,
    )
    probe = fidelity_probe()
    tool_call: Any = next(f for f in probe.functions if f.__name__ == "tool_call")

    anyio.run(tool_call, Context(path="", host=Host(FakeHost())), step)

    assert decisions.reads <= 2 and history.reads <= 4


def test_a_scanned_step_carries_its_stage_and_turns(tmp_path: Path) -> None:
    log = run(tmp_path / "logs", [[call(1)]], observe_only(fidelity_probe()))
    assert log.samples is not None
    recorded = [e for e in log.samples[0].events if isinstance(e, SentinelEvent)]

    found = scanned(
        tmp_path / "scans",
        tmp_path / "logs",
        s=as_scanner(fidelity_probe(), stage="tool_result"),
    )

    [row] = found["s"]
    assert (row["metadata.stage"], row["metadata.turns"]) == ("tool_result", "recorded")
    assert row["metadata.step_id"] in {e.step_id for e in recorded}
    assert row["value"] == 0.0


def test_a_sample_without_a_uuid_gets_its_context_by_its_typed_id(
    tmp_path: Path,
) -> None:
    from inspect_ai.analysis._dataframe.samples.extract import auto_sample_id
    from inspect_ai.log import write_eval_log
    from inspect_scout import Transcript

    from inspect_sentinel.scout._context import Contexts

    log = run(tmp_path / "logs", [[call(1)], [call(2)]], None)
    assert log.samples is not None
    for sample in log.samples:
        sample.uuid = None
    path = str(tmp_path / "old.json")
    write_eval_log(log, path, format="json")
    second = log.samples[1]
    transcript = Transcript(
        transcript_id=auto_sample_id(log.eval.eval_id, second),
        source_type="eval_log",
        source_uri=path,
    )

    context = anyio.run(Contexts().eval_context, transcript)

    assert context is not None
    assert (context.sample_id, context.epoch) == (1, 1)
    assert context.sample_input == second.input


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
