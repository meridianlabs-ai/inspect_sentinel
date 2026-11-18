from itertools import count
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, agent, agent_bridge, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall
from inspect_scout import scan, scan_results_df, transcripts_from

from inspect_sentinel import (
    AfterToolCall,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    ProtocolGroup,
    observe_only,
    protocol,
)
from inspect_sentinel.scout import (
    as_reader,
    as_scanner,
    replay_check,
    replay_check_differences,
)

Row = dict[str, Any]
Client = Literal["openai", "anthropic"]
READ_FILE = "Read a file."
PATH = {"type": "object", "properties": {"path": {"type": "string"}}}

# Agents run through an agent bridge, as Claude Code is, run their own tools
# and the bridge checks each call and result live, in `sentinel_bridge` spans.
# Read mode keys those checks apart from any replayed step, and names the step
# each joins as `metadata.replay_step_id`: `<model call uuid>/<call id>`, which
# inspect_ai's replay names a bridged call's steps.


def calls(*ids: str) -> ModelOutput:
    # one turn whose response reads the file named by each call id
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(id=i, function="read_file", arguments={"path": i})
                        for i in ids
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


@agent(name="scaffold")
def scaffold(client: Client, retry: bool = False) -> Agent:
    """A tool loop over the `client` SDK that runs its tools itself, as Claude Code does.

    With `retry`, it sends its first request twice and uses the second response, as an agent that retries a request does.
    """

    async def execute(state: AgentState) -> AgentState:
        async with agent_bridge(state) as bridge:
            if client == "openai":
                await _openai_loop(state.messages[0].text, retry)
            else:
                await _anthropic_loop(state.messages[0].text)
            return bridge.state

    return execute


async def _openai_loop(task: str, retry: bool) -> None:
    from openai import AsyncOpenAI

    messages: list[Any] = [{"role": "user", "content": task}]
    tool = {"name": "read_file", "description": READ_FILE, "parameters": PATH}
    async with AsyncOpenAI(api_key="sk-test") as openai:
        for turn in count():
            # a retried request is sent again as is; the agent uses the last
            # response
            completions = [
                await openai.chat.completions.create(
                    model="inspect",
                    messages=messages,
                    tools=[{"type": "function", "function": tool}],  # type: ignore[list-item]
                )
                for _ in range(2 if retry and turn == 0 else 1)
            ]
            message = completions[-1].choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return
            # each result names its turn and call, so a check paired with
            # another turn's result shows it
            for call in message.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"t{turn}:{call.id}",
                    }
                )


async def _anthropic_loop(task: str) -> None:
    from anthropic import AsyncAnthropic

    messages: list[Any] = [{"role": "user", "content": task}]
    tool = {"name": "read_file", "description": READ_FILE, "input_schema": PATH}
    async with AsyncAnthropic(api_key="sk-test") as anthropic:
        while True:
            reply = await anthropic.messages.create(
                model="inspect",
                max_tokens=1024,
                messages=messages,
                tools=[tool],  # type: ignore[list-item]
            )
            uses = [block for block in reply.content if block.type == "tool_use"]
            messages.append(
                {
                    "role": "assistant",
                    "content": [b.model_dump(exclude_none=True) for b in reply.content],
                }
            )
            if not uses:
                return
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": u.id, "content": "x"}
                        for u in uses
                    ],
                }
            )


@protocol
def judge() -> ProtocolGroup:
    """Rejects a call to read the file `bad`, and proceeds otherwise, recording the call's arguments or the result it saw."""

    async def tool_call(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision:
        seen = {"args": step.call.arguments}
        if step.call.arguments.get("path") == "bad":
            return Decision(action="reject", message="Not that file.", metadata=seen)
        return Decision(action="proceed", metadata=seen)

    async def tool_result(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision:
        return Decision(action="proceed", metadata={"result": step.result.text})

    return ProtocolGroup(tool_call, tool_result)


def run(
    tmp_path: Path,
    turns: list[ModelOutput],
    client: Client,
    sentinel: Any,
    retry: bool = False,
) -> list[str]:
    # the eval's log, in tmp_path / "logs", and the uuids of the model calls
    # the bridge made, in order
    [log] = eval(
        Task(
            dataset=[Sample(input="Read the files.")],
            solver=as_solver(scaffold(client, retry)),
            sentinel=sentinel,
        ),
        model=get_model(
            "mockllm/model",
            custom_outputs=[*turns, ModelOutput.from_content("mockllm/model", "done")],
        ),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error
    assert log.samples is not None
    return [
        str(e.uuid)
        for e in log.samples[0].events
        if isinstance(e, ModelEvent) and e.requested_model is not None
    ]


def scanned(tmp_path: Path, **scanners: Any) -> dict[str, Any]:
    status = scan(
        scanners=scanners,
        transcripts=transcripts_from(str(tmp_path / "logs")),
        scans=str(tmp_path / "scans"),
        display="none",
        max_processes=1,
    )
    assert status.complete
    return dict(scan_results_df(status.location).scanners)


def records(frame: Any) -> list[Row]:
    return cast(list[Row], frame.to_dict("records"))


def joined(step: str | None, turns: list[str]) -> str | None:
    # "<turn>/<call id>", as the step replay names by that turn's model call
    if step is None:
        return None
    turn, call_id = step.split("/")
    return f"{turns[int(turn)]}/{call_id}"


# each case's turns, and the step each check joins, by stage and by the turn
# whose model call proposed the call ("1/c": the second turn's call "c"), with
# the decision the check recorded
CASES: dict[str, tuple[list[ModelOutput], list[tuple[str, str | None, str]]]] = {
    # a check of every call and result
    "plain": (
        [calls("a", "b"), calls("c")],
        [
            ("tool_call", "0/a", "proceed"),
            ("tool_call", "0/b", "proceed"),
            ("tool_result", "0/a", "proceed"),
            ("tool_result", "0/b", "proceed"),
            ("tool_call", "1/c", "proceed"),
            ("tool_result", "1/c", "proceed"),
        ],
    ),
    # the bridge checks no result of a response with a rejected call
    "rejection": (
        [calls("ok", "bad"), calls("c")],
        [
            ("tool_call", "0/ok", "proceed"),
            ("tool_call", "0/bad", "reject"),
            ("tool_call", "1/c", "proceed"),
            ("tool_result", "1/c", "proceed"),
        ],
    ),
    # once "a" is handed out twice, the bridge checks the first turn's result
    # again beside the second's, and cannot say which check is whose: both
    # name the second turn's step, so neither joins it
    "reused id": (
        [calls("a", "b"), calls("a")],
        [
            ("tool_call", "0/a", "proceed"),
            ("tool_call", "0/b", "proceed"),
            ("tool_result", "0/a", "proceed"),
            ("tool_result", "0/b", "proceed"),
            ("tool_call", "1/a", "proceed"),
            ("tool_result", None, "proceed"),
            ("tool_result", None, "proceed"),
        ],
    ),
    # replay builds no steps for calls sharing an id in one response
    "duplicate ids": (
        [calls("d", "d"), calls("c")],
        [
            ("tool_call", None, "proceed"),
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
            ("tool_call", "1/c", "proceed"),
            ("tool_result", "1/c", "proceed"),
        ],
    ),
    # the agent sends its first request twice and uses the second response,
    # then the model reuses "a": each result is the one the agent used, so
    # its check joins the latest proposal of the id, never the first response
    "retried request, reused id": (
        [calls("a"), calls("a"), calls("a"), calls("a")],
        [
            ("tool_call", "0/a", "proceed"),
            ("tool_call", "1/a", "proceed"),
            ("tool_result", "1/a", "proceed"),
            ("tool_call", "2/a", "proceed"),
            ("tool_result", "2/a", "proceed"),
            ("tool_call", "3/a", "proceed"),
            ("tool_result", "3/a", "proceed"),
        ],
    ),
}

# the cases whose agent sends its first request twice
RETRIED = {"retried request, reused id"}


def ordered(rows: list[tuple[str, str | None, str]]) -> list[tuple[str, str, str]]:
    return sorted((stage, step or "", answer) for stage, step, answer in rows)


@pytest.mark.parametrize(
    "case, client",
    [*((case, "openai") for case in CASES), ("plain", "anthropic")],
)
def test_a_bridge_check_names_the_replayed_step_of_the_call_it_checked(
    tmp_path: Path, case: str, client: Client
) -> None:
    turns, expected = CASES[case]
    uuids = run(tmp_path, turns, client, judge(), retry=case in RETRIED)

    read = records(scanned(tmp_path, recorded=as_reader())["recorded"])

    tools = [r for r in read if r["metadata.stage"].startswith("tool")]
    assert ordered(
        [
            (r["metadata.stage"], _text(r["metadata.replay_step_id"]), r["answer"])
            for r in tools
        ]
    ) == ordered([(s, joined(step, uuids), a) for s, step, a in expected])
    # the check's own key and citations are read mode's as before: never a
    # replayed step's
    assert not {r["metadata.step_id"] for r in tools} & {
        joined(step, uuids) for _, step, _ in expected
    }
    # each generate's checks are read under the generate's id, as replay names it
    assert all(
        r["metadata.replay_step_id"] == r["metadata.step_id"]
        for r in read
        if r["metadata.stage"].startswith("model")
    )


def _text(value: object) -> str | None:
    # a dataframe holds a missing string as None or NaN
    return value if isinstance(value, str) else None


def replayed_bridged_steps(frame: Any) -> None:
    # replay rebuilds bridged agents' tool steps from the inspect_ai release
    # with the follow-up to #6446 that replays them as the conversation;
    # earlier releases skip them
    if not any(r["metadata.stage"].startswith("tool") for r in records(frame)):
        pytest.skip("this inspect_ai's replay skips bridged agents' tool steps")


@pytest.mark.parametrize("case", list(CASES))
def test_replayed_and_recorded_bridge_decisions_join_side_by_side(
    tmp_path: Path, case: str
) -> None:
    turns, expected = CASES[case]
    uuids = run(tmp_path, turns, "openai", judge(), retry=case in RETRIED)
    frames = scanned(tmp_path, replayed=as_scanner(judge()), recorded=as_reader())
    replayed_bridged_steps(frames["replayed"])

    # the join the docs give (replay.qmd, "Bridged Agents")
    keys = ["transcript_id", "metadata.conversation", "metadata.stage"]
    both = frames["replayed"].merge(
        frames["recorded"],
        left_on=[*keys, "metadata.step_id"],
        right_on=[*keys, "metadata.replay_step_id"],
        how="outer",
        suffixes=("_replayed", "_recorded"),
        indicator=True,
    )

    tools = [r for r in records(both) if r["metadata.stage"].startswith("tool")]
    paired = [
        (r["metadata.stage"], r["metadata.step_id_replayed"], r["answer_recorded"])
        for r in tools
        if r["_merge"] == "both"
    ]
    assert ordered(paired) == ordered(
        [(s, joined(step, uuids), a) for s, step, a in expected if step is not None]
    )
    # each pair saw the same call or result (each turn's result names the
    # turn) and so decided alike, as the replayed protocol is the recorded one
    assert all(
        r["metadata.metadata_replayed"] == r["metadata.metadata_recorded"]
        and r["answer_replayed"] == r["answer_recorded"]
        for r in tools
        if r["_merge"] == "both"
    )
    # an unmatched check is listed, and so is a replayed step no check joins
    assert sum(r["_merge"] == "right_only" for r in tools) == sum(
        step is None for _, step, _ in expected
    )
    unchecked = {
        (r["metadata.stage"], r["metadata.step_id_replayed"])
        for r in tools
        if r["_merge"] == "left_only"
    }
    assert unchecked == {(s, joined(step, uuids)) for s, step in UNCHECKED[case]}


# the replayed steps no check joins: the results the model received after a
# rejection, which the bridge never checked, and the reused id's second result,
# which its two checks leave unjoined
UNCHECKED: dict[str, list[tuple[str, str]]] = {
    "plain": [],
    "rejection": [("tool_result", "0/ok"), ("tool_result", "0/bad")],
    "reused id": [("tool_result", "1/a")],
    "duplicate ids": [],
    "retried request, reused id": [],
}


def test_replay_check_matches_a_bridge_check_with_the_step_it_checked(
    tmp_path: Path,
) -> None:
    run(tmp_path, [calls("a", "b"), calls("c")], "openai", observe_only(replay_check()))
    frames = scanned(
        tmp_path,
        replayed=as_scanner(replay_check()),
        recorded=as_reader("replay_check"),
    )
    replayed_bridged_steps(frames["replayed"])

    differences: Any = replay_check_differences(frames["replayed"], frames["recorded"])

    # replay hands a monitor the call and result a live bridge check saw
    assert differences.empty
