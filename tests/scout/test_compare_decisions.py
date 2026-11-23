import importlib
from pathlib import Path
from typing import Any, cast

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, tool
from inspect_scout import scan, scan_results_df, transcripts_from

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    ProtocolGroup,
    protocol,
)
from inspect_sentinel.scout import as_reader, as_scanner, compare_decisions

Row = dict[str, Any]
# pandas is untyped here
pandas: Any = importlib.import_module("pandas")

# a step's comparison: conversation, step id, the replayed and the recorded
# decision, whether they agree, and the status
Compared = tuple[str | None, str, Any, Any, bool, str]


def checked(
    step_id: str,
    decision: Any,
    *,
    conversation: str = "agent",
    turns: str | None = None,
    **metadata: Any,
) -> Row:
    # a results row at a tool call: a protocol's action, or a monitor's
    # suspicion, None where it abstained, as scan_results_df() gives them
    if isinstance(decision, str):
        held: Row = {"answer": decision, "value": decision != "proceed"}
    else:
        held = {"value": decision}
    return {
        "transcript_id": "sample",
        "metadata.conversation": conversation,
        "metadata.stage": "tool_call",
        "metadata.step_id": step_id,
        "metadata.turns": turns,
        "explanation": f"why {step_id}",
        **held,
        **{f"metadata.{k}": v for k, v in metadata.items()},
    }


def replayed(step_id: str, decision: Any, **metadata: Any) -> Row:
    return checked(step_id, decision, **({"turns": "recorded"} | metadata))


def recorded(step_id: str, decision: Any, **metadata: Any) -> Row:
    # a read row, which names the replayed step it joins
    return checked(step_id, decision, **({"replay_step_id": step_id} | metadata))


CASES: dict[str, tuple[list[Row], list[Row], list[Compared]]] = {
    "a protocol's actions, agreeing and not": (
        [replayed("s1", "proceed"), replayed("s2", "reject")],
        [recorded("s1", "proceed"), recorded("s2", "proceed")],
        [
            ("agent", "s1", "proceed", "proceed", True, "matched"),
            ("agent", "s2", "reject", "proceed", False, "matched"),
        ],
    ),
    # a score a frame holds as text is read as its value; a step the replay
    # abstained at, which the live monitor recorded nothing at either, is only
    # replayed
    "a monitor's scores": (
        [
            replayed("s1", 0.2),
            replayed("s2", {"files": 0.1}),
            replayed("s3", 0.5) | {"value": "0.5", "value_type": "number"},
            replayed("s4", None),
        ],
        [
            recorded("s1", 0.7),
            recorded("s2", {"files": 0.1}),
            recorded("s3", 0.5),
        ],
        [
            ("agent", "s1", 0.2, 0.7, False, "matched"),
            ("agent", "s2", {"files": 0.1}, {"files": 0.1}, True, "matched"),
            ("agent", "s3", 0.5, 0.5, True, "matched"),
            ("agent", "s4", None, None, False, "only replayed"),
        ],
    ),
    # an agent bridge's check is keyed apart from replay's steps and names
    # the step it checked; one that names none is only recorded, under its own
    # step id
    "a bridge's checks": (
        [replayed("m/a", "reject"), replayed("m/b", "proceed")],
        [
            recorded("e1", "proceed", replay_step_id="m/a"),
            recorded("e2", "proceed", replay_step_id=None),
        ],
        [
            ("agent", "e2", None, "proceed", False, "only recorded"),
            ("agent", "m/a", "reject", "proceed", False, "matched"),
            ("agent", "m/b", "proceed", None, False, "only replayed"),
        ],
    ),
    # a replay whose conversations are read mode's is matched by them
    "recorded turns": (
        [replayed("s1", "proceed", conversation="agent-a")],
        [recorded("s1", "proceed", conversation="agent-b")],
        [
            ("agent-a", "s1", "proceed", None, False, "only replayed"),
            ("agent-b", "s1", None, "proceed", False, "only recorded"),
        ],
    ),
    # a stored timeline's are not, so its rows are matched without them
    "stored turns": (
        [replayed("s1", "reject", conversation="rollback", turns="stored")],
        [recorded("s1", "proceed", conversation="sample")],
        [(None, "s1", "reject", "proceed", False, "matched")],
    ),
    # where that leaves two rows of one side with one key, no row of that key
    # is paired
    "rows that share a key": (
        [
            replayed("s1", "proceed", conversation="target", turns="stored"),
            replayed("s1", "proceed", conversation="rollback", turns="stored"),
        ],
        [recorded("s1", "proceed", conversation="agent-a")],
        [
            (None, "s1", "proceed", None, False, "only replayed"),
            (None, "s1", "proceed", None, False, "only replayed"),
            (None, "s1", None, "proceed", False, "only recorded"),
        ],
    ),
    # a decision that did not take effect is not the step's
    "a superseded decision": (
        [replayed("s1", "reject")],
        [recorded("s1", "escalate", status="superseded"), recorded("s1", "reject")],
        [("agent", "s1", "reject", "reject", True, "matched")],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_compare_decisions_puts_each_steps_two_decisions_side_by_side(
    case: str,
) -> None:
    replayed_rows, recorded_rows, expected = CASES[case]

    compared: Any = compare_decisions(
        pandas.DataFrame(replayed_rows), pandas.DataFrame(recorded_rows)
    )

    found = cast(list[Row], compared.to_dict("records"))
    assert [
        (
            r["conversation"],
            r["step_id"],
            r["replayed"],
            r["recorded"],
            r["agree"],
            r["status"],
        )
        for r in found
    ] == expected
    # each side's explanation, where it has the row
    assert all(
        (r["replayed_explanation"] is not None) == (r["status"] != "only recorded")
        and (r["recorded_explanation"] is not None) == (r["status"] != "only replayed")
        for r in found
    )


@pytest.mark.parametrize(
    "row, error",
    [
        (
            {"transcript_id": "sample", "scan_error": "MonitorFailedError: boom"},
            "are an error, so they cannot be compared: MonitorFailedError: boom",
        ),
        (
            {"transcript_id": "sample", "value": 0.5, "metadata.steps": 3},
            "is not a step's",
        ),
    ],
)
def test_compare_decisions_refuses_a_row_that_is_not_a_steps_decision(
    row: Row, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        compare_decisions(
            pandas.DataFrame([row]), pandas.DataFrame([recorded("s1", "proceed")])
        )


@tool
def read_file() -> Tool:
    async def execute(path: str) -> str:
        """Read a file.

        Args:
            path: The file.
        """
        return f"contents of {path}"

    return execute


@protocol
def refuses_to_read(path: str) -> ProtocolGroup:
    """Rejects a call to read the file `path`, and proceeds otherwise."""

    async def tool_call(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision:
        if step.call.arguments.get("path") == path:
            return Decision(action="reject", message="Not that file.")
        return Decision(action="proceed")

    return ProtocolGroup(tool_call)


def test_compare_decisions_shows_where_a_new_protocol_decides_otherwise(
    tmp_path: Path,
) -> None:
    # one call per turn, the second of which the live protocol rejects
    outputs = [
        ModelOutput.for_tool_call("mockllm/model", "read_file", {"path": p})
        for p in ("a", "b", "c")
    ]
    [log] = eval(
        Task(
            dataset=[Sample(input="Read the files.")],
            solver=[use_tools(read_file()), generate()],
            sentinel=refuses_to_read("b"),
        ),
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                *outputs,
                ModelOutput.from_content("mockllm/model", "done"),
            ],
        ),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error
    status = scan(
        scanners={
            "replayed": as_scanner(refuses_to_read("a"), stage="tool_call"),
            "recorded": as_reader(stage="tool_call"),
        },
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )
    frames: Any = scan_results_df(status.location).scanners

    compared: Any = compare_decisions(frames["replayed"], frames["recorded"])

    # each call's step, by the file it read
    paths = {
        str(call.id): str(call.arguments["path"])
        for output in outputs
        for call in output.message.tool_calls or []
    }
    assert sorted(
        (paths[r["step_id"]], r["replayed"], r["recorded"], r["agree"], r["status"])
        for r in cast(list[Row], compared.to_dict("records"))
    ) == [
        ("a", "reject", "proceed", False, "matched"),
        ("b", "proceed", "reject", False, "matched"),
        ("c", "proceed", "proceed", True, "matched"),
    ]
