import json
from itertools import count
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, agent, agent_bridge, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent, SentinelEvent
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_ai.util import span
from inspect_scout import scan, scan_results_df, transcripts_from

from inspect_sentinel import (
    AfterGenerate,
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
    compare_decisions,
    replay_check,
    replay_check_differences,
)
from inspect_sentinel.scout._probe import _hash

Row = dict[str, Any]
Client = Literal["openai", "anthropic"]
Filter = Literal["candidates", "cached", "monitor", "rekeyed"]
READ_FILE = "Read a file."
PATH = {"type": "object", "properties": {"path": {"type": "string"}}}

# Agents run through an agent bridge, as Claude Code is, run their own tools
# and the bridge checks each call and result live, in `sentinel_bridge` spans.
# Read mode keys those checks apart from any replayed step, and names the step
# each joins as `metadata.replay_step_id`: `<model call uuid>/<call id>`, which
# inspect_ai's replay names a bridged call's steps.


def calls(*ids: str, path: str | None = None) -> ModelOutput:
    # one turn whose response reads the file named by each call id, or `path`
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=i, function="read_file", arguments={"path": path or i}
                        )
                        for i in ids
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


@agent(name="scaffold")
def scaffold(
    client: Client, retry: bool = False, filter: Filter | None = None
) -> Agent:
    """A tool loop over the `client` SDK that runs its tools itself, as Claude Code does.

    With `retry`, it sends its first request twice and uses the second response, as an agent that retries a request does. With `filter`, the bridge's requests go through a request filter: `"candidates"` generates a second candidate for the first request and hands over the first, `"cached"` generates for the second request but hands over its response to the first again, `"monitor"` generates with the model for the monitor role, and `"rekeyed"` generates twice for the first request and hands over the second response with its call under the first response's call id.
    """
    first = True
    cached: list[ModelOutput] = []

    async def choose(
        model: Model,
        input: list[Any],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput:
        nonlocal first
        if filter == "monitor":
            model = get_model(role="monitor", default=model)
        chosen = await model.generate(
            input=input, tools=tools, tool_choice=tool_choice, config=config
        )
        if filter == "candidates" and first:
            first = False
            await model.generate(
                input=[ChatMessageUser(content="another candidate")],
                tools=tools,
                tool_choice=tool_choice,
                config=config,
            )
        if filter == "rekeyed" and first:
            first = False
            selected = await model.generate(
                input=input, tools=tools, tool_choice=tool_choice, config=config
            )
            handed = selected.model_copy(deep=True)
            [call] = handed.message.tool_calls or []
            [discarded] = chosen.message.tool_calls or []
            call.id = discarded.id
            return handed
        if filter == "cached":
            cached.append(chosen)
            if len(cached) == 2:
                return cached[0]
        return chosen

    async def execute(state: AgentState) -> AgentState:
        async with agent_bridge(
            state, filter=choose if filter is not None else None
        ) as bridge:
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
def judge(bad: str = "bad", stop: bool = False) -> ProtocolGroup:
    """Rejects a call to read the file `bad`, and proceeds otherwise, recording the call's arguments or the result it saw; proceeds at each generate's output. With `stop`, ends the run at the first result it sees."""

    async def tool_call(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision:
        seen = {"args": step.call.arguments}
        if step.call.arguments.get("path") == bad:
            return Decision(action="reject", message="Not that file.", metadata=seen)
        return Decision(action="proceed", metadata=seen)

    async def tool_result(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision:
        seen = {"result": step.result.text}
        if stop:
            return Decision(action="terminate", explanation="Done.", metadata=seen)
        return Decision(action="proceed", metadata=seen)

    async def model_output(
        context: Context, step: AfterGenerate, escalations: Escalations
    ) -> Decision:
        return Decision(action="proceed")

    return ProtocolGroup(tool_call, tool_result, model_output)


def run(
    tmp_path: Path,
    turns: list[ModelOutput],
    client: Client,
    sentinel: Any,
    retry: bool = False,
    filter: Filter | None = None,
) -> list[str]:
    # the eval's log, in tmp_path / "logs", and the uuids of the model calls
    # the bridge made, in order
    [log] = eval(
        Task(
            dataset=[Sample(input="Read the files.")],
            solver=as_solver(scaffold(client, retry, filter)),
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
    # the model proposes "a" in both turns, so nothing in the log says which
    # proposal a check under "a" is of: no check of "a" joins, the first
    # turn's included, and replay builds no result step for "a". (The bridge
    # checks each turn's result once: since inspect_ai #6747 it doesn't check
    # the first again when the second request carries it.) "b", proposed
    # once, joins
    "reused id": (
        [calls("a", "b"), calls("a")],
        [
            ("tool_call", None, "proceed"),
            ("tool_call", "0/b", "proceed"),
            ("tool_result", None, "proceed"),
            ("tool_result", "0/b", "proceed"),
            ("tool_call", None, "proceed"),
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
    # then the model reuses "a": every response proposes "a", so no check
    # joins, and no result step is built for "a"
    "retried request, reused id": (
        [calls("a"), calls("a"), calls("a"), calls("a")],
        [
            ("tool_call", None, "proceed"),
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
        ],
    ),
    # a request filter generates two candidates under "a" and hands over the
    # first: replay builds both calls' steps, and the log doesn't say which
    # one the bridge checked, so the check joins neither. Nor does its result,
    # under an id proposed twice
    "filter hands over one of two candidates": (
        [calls("a", path="first"), calls("a", path="second")],
        [
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
        ],
    ),
    # a request filter generates for the second request but hands over its
    # first response again: the bridge checks the old call a second time,
    # after a newer proposal of "a" the agent never received, so neither
    # check joins either proposal. Each turn's result is checked once
    "filter hands over a cached response": (
        [calls("a", path="old"), calls("a", path="new")],
        [
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
        ],
    ),
    # a request filter generates with the monitor role's model, whose calls
    # replay builds no steps for, so the bridge's checks of them join none
    "filter generates as the monitor": (
        [calls("a")],
        [
            ("tool_call", None, "proceed"),
            ("tool_result", None, "proceed"),
        ],
    ),
    # the sentinel ends the run at the first result, so no later request
    # carries it and replay builds no result step: the call check joins, the
    # result check doesn't
    "terminated at the first result": (
        [calls("a")],
        [
            ("tool_call", "0/a", "proceed"),
            ("tool_result", None, "terminate"),
        ],
    ),
}

# the cases whose agent sends its first request twice
RETRIED = {"retried request, reused id"}
# the cases whose bridge requests go through a request filter
FILTERED: dict[str, Filter] = {
    "filter hands over one of two candidates": "candidates",
    "filter hands over a cached response": "cached",
    "filter generates as the monitor": "monitor",
}
# the cases whose live sentinel ends the run at the first result
STOPPED = {"terminated at the first result"}


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
    uuids = run(
        tmp_path,
        turns,
        client,
        judge(stop=case in STOPPED),
        retry=case in RETRIED,
        filter=FILTERED.get(case),
    )

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
    # each generate's check, one per model call the bridge made, is read under
    # the generate's id, as replay names it
    generates = [r for r in read if r["metadata.stage"].startswith("model")]
    assert len(generates) == len(uuids)
    assert all(r["metadata.replay_step_id"] == r["metadata.step_id"] for r in generates)


def _text(value: object) -> str | None:
    # a dataframe holds a missing string as None or NaN
    return value if isinstance(value, str) else None


@pytest.mark.parametrize("case", list(CASES))
def test_replayed_and_recorded_bridge_decisions_join_side_by_side(
    tmp_path: Path, case: str
) -> None:
    turns, expected = CASES[case]
    uuids = run(
        tmp_path,
        turns,
        "openai",
        judge(stop=case in STOPPED),
        retry=case in RETRIED,
        filter=FILTERED.get(case),
    )
    frames = scanned(tmp_path, replayed=as_scanner(judge()), recorded=as_reader())

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


@pytest.mark.parametrize("case", list(CASES))
def test_read_mode_names_only_steps_replay_built(tmp_path: Path, case: str) -> None:
    # read mode names a step for a check only where replay built that step, so
    # the two views agree on which steps exist: where replay builds no step
    # (a reused id's results, calls sharing an id in one response, a monitor
    # role's calls), the check names none rather than one no replayed row has
    turns, expected = CASES[case]
    uuids = run(
        tmp_path,
        turns,
        "openai",
        judge(stop=case in STOPPED),
        retry=case in RETRIED,
        filter=FILTERED.get(case),
    )
    frames = scanned(tmp_path, replayed=as_scanner(judge()), recorded=as_reader())

    built = {
        (r["metadata.stage"], r["metadata.step_id"])
        for r in records(frames["replayed"])
    }
    named = {
        (r["metadata.stage"], r["metadata.replay_step_id"])
        for r in records(frames["recorded"])
        if _text(r["metadata.replay_step_id"]) is not None
    }
    assert named <= built
    # and every step the case expects a check to join is one replay built
    assert {(s, joined(step, uuids)) for s, step, _ in expected if step} <= built


# the replayed steps no check joins: the results the model received after a
# rejection, which the bridge never checked, and the calls under an id
# proposed more than once, whose checks can't be told apart. A reused id has
# no replayed result step to leave unjoined
UNCHECKED: dict[str, list[tuple[str, str]]] = {
    "plain": [],
    "rejection": [("tool_result", "0/ok"), ("tool_result", "0/bad")],
    "reused id": [("tool_call", "0/a"), ("tool_call", "1/a")],
    "duplicate ids": [],
    "retried request, reused id": [
        ("tool_call", "0/a"),
        ("tool_call", "1/a"),
        ("tool_call", "2/a"),
        ("tool_call", "3/a"),
    ],
    "filter hands over one of two candidates": [
        ("tool_call", "0/a"),
        ("tool_call", "1/a"),
    ],
    "filter hands over a cached response": [
        ("tool_call", "0/a"),
        ("tool_call", "1/a"),
    ],
    "filter generates as the monitor": [],
    "terminated at the first result": [],
}


def rejects_a(stage: str, step: str) -> str:
    # the new protocol's decision at a step, whose call id is the file it reads
    return "reject" if stage == "tool_call" and step.endswith("/a") else "proceed"


@pytest.mark.parametrize("case", list(CASES))
def test_compare_decisions_pairs_a_new_protocols_decisions_with_a_bridges_checks(
    tmp_path: Path, case: str
) -> None:
    turns, expected = CASES[case]
    uuids = run(
        tmp_path,
        turns,
        "openai",
        judge(stop=case in STOPPED),
        retry=case in RETRIED,
        filter=FILTERED.get(case),
    )
    # a new protocol that rejects reading `a` rather than `bad`
    frames = scanned(
        tmp_path, replayed=as_scanner(judge(bad="a")), recorded=as_reader()
    )

    compared = records(compare_decisions(frames["replayed"], frames["recorded"]))

    tools = [r for r in compared if r["stage"].startswith("tool")]
    # each joined check beside the new protocol's decision at its step
    assert sorted(
        (r["stage"], r["step_id"], r["replayed"], r["recorded"], r["agree"])
        for r in tools
        if r["status"] == "matched"
    ) == sorted(
        (s, joined(step, uuids), rejects_a(s, step), a, rejects_a(s, step) == a)
        for s, step, a in expected
        if step is not None
    )
    # a check that joins no step is listed, and so is a step no check joins
    assert sum(r["status"] == "only recorded" for r in tools) == sum(
        step is None for _, step, _ in expected
    )
    assert {
        (r["stage"], r["step_id"]) for r in tools if r["status"] == "only replayed"
    } == {(s, joined(step, uuids)) for s, step in UNCHECKED[case]}


def test_known_limitation_a_filter_that_rekeys_a_call_pairs_its_check_with_the_discarded_generation(
    tmp_path: Path,
) -> None:
    # A known limitation, accepted on 2026-11-23 (design/pr-series.md,
    # Deferred, "Exact joins for bridge checks, from a recorded producer"): the
    # log doesn't record which model call produced a bridged call. A request
    # filter generates `a` (reading "first"), then `b` (reading "second"), and
    # hands over the second response with its call renamed `a`. The bridge
    # checks the call that reads "second" under `a`, and the discarded first
    # generation is the one model call that proposed `a`, so read mode pairs
    # the check with its step. This pins that wrong pairing, so recording the
    # producer has to change this test on purpose
    uuids = run(
        tmp_path,
        [calls("a", path="first"), calls("b", path="second")],
        "openai",
        judge(),
        filter="rekeyed",
    )
    frames = scanned(tmp_path, replayed=as_scanner(judge()), recorded=as_reader())
    discarded, selected = f"{uuids[0]}/a", f"{uuids[1]}/b"

    replayed = {
        (r["metadata.stage"], r["metadata.step_id"]): r["metadata.metadata"]
        for r in records(frames["replayed"])
        if r["metadata.stage"].startswith("tool")
    }
    assert replayed == {
        ("tool_call", discarded): {"args": {"path": "first"}},
        ("tool_result", discarded): {"result": "t0:a"},
        ("tool_call", selected): {"args": {"path": "second"}},
    }
    recorded = [
        (r["metadata.stage"], r["metadata.replay_step_id"], r["metadata.metadata"])
        for r in records(frames["recorded"])
        if r["metadata.stage"].startswith("tool")
    ]
    # the check of the call that reads "second" names the discarded step
    assert sorted(recorded, key=str) == sorted(
        [
            ("tool_call", discarded, {"args": {"path": "second"}}),
            ("tool_result", discarded, {"result": "t0:a"}),
        ],
        key=str,
    )
    compared = {
        (r["stage"], r["step_id"]): r["status"]
        for r in records(compare_decisions(frames["replayed"], frames["recorded"]))
        if r["stage"].startswith("tool")
    }
    assert compared == {
        ("tool_call", discarded): "matched",
        ("tool_result", discarded): "matched",
        ("tool_call", selected): "only replayed",
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

    differences: Any = replay_check_differences(frames["replayed"], frames["recorded"])

    # replay hands a monitor the call and result a live bridge check saw
    assert differences.empty


def assistant_messages(frame: Any) -> list[tuple[str, str, str]]:
    # each tool step's stage, step id and assistant_message digest, from
    # replay_check() rows
    found: list[tuple[str, str, str]] = []
    for row in records(frame):
        stage = str(row["metadata.stage"])
        if not stage.startswith("tool"):
            continue
        metadata = row["metadata.metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        found.append(
            (stage, str(row["metadata.step_id"]), metadata["step"]["assistant_message"])
        )
    return found


def test_replay_gives_a_renamed_call_the_assistant_message_the_bridge_handed_over(
    tmp_path: Path,
) -> None:
    # the second turn reuses "a" for another call, so the bridge hands it over
    # under a fresh id, which it records on the model call
    first, second, _ = run(
        tmp_path,
        [calls("a"), calls("a", path="other")],
        "openai",
        observe_only(replay_check()),
    )
    [log] = list_eval_logs(str(tmp_path / "logs"))
    sample = read_eval_log(log).samples
    assert sample is not None
    [renamed] = [
        e
        for e in sample[0].events
        if isinstance(e, ModelEvent) and str(e.uuid) == second
    ]
    assert renamed.agent_tool_call_ids is not None
    [fresh] = renamed.agent_tool_call_ids
    assert fresh != "a"
    frames = scanned(
        tmp_path,
        replayed=as_scanner(replay_check()),
        recorded=as_reader("replay_check"),
    )
    replayed = assistant_messages(frames["replayed"])
    recorded = assistant_messages(frames["recorded"])

    # replay names the renamed call's steps by its fresh id, and builds no
    # result step for "a", which the conversation proposed twice
    assert sorted((stage, step) for stage, step, _ in replayed) == sorted(
        [
            ("tool_call", f"{first}/a"),
            ("tool_call", f"{second}/{fresh}"),
            ("tool_result", f"{second}/{fresh}"),
        ]
    )
    # the message the bridge handed over: the second model call's output with
    # the fresh id in place of the model's (the digest leaves out message ids)
    handed = renamed.output.message.model_copy(deep=True)
    assert handed.tool_calls is not None
    handed.tool_calls[0].id = fresh
    expected = _hash(handed)
    assert expected != _hash(renamed.output.message)
    # the live checks of the renamed call, by the execution their sentinel
    # events record for its fresh id
    [execution] = {
        e.execution_id
        for e in sample[0].events
        if isinstance(e, SentinelEvent) and e.step_id == fresh
    }
    replayed_by_step = {(stage, step): digest for stage, step, digest in replayed}
    recorded_by_step = {(stage, step): digest for stage, step, digest in recorded}
    for stage in ("tool_call", "tool_result"):
        assert replayed_by_step[(stage, f"{second}/{fresh}")] == expected
        assert recorded_by_step[(stage, str(execution))] == expected


@agent(name="two_conversations")
def two_conversations() -> Agent:
    async def execute(state: AgentState) -> AgentState:
        async with agent_bridge(state) as bridge:
            # one bridge, two conversations, each in an agent span of its own
            for task in ("first task", "second task"):
                async with span(task, type="agent"):
                    await _openai_loop(task, False)
            return bridge.state

    return execute


MODEL_CALL_FIELDS = ("assistant_message", "input", "tools", "tool_choice")


def model_call_fields(frame: Any) -> list[tuple[str, dict[str, str]]]:
    # each tool_result row's step id and digests of the four fields from the
    # model call, from replay_check() rows
    found: list[tuple[str, dict[str, str]]] = []
    for row in records(frame):
        if row["metadata.stage"] != "tool_result":
            continue
        metadata = row["metadata.metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        step = metadata["step"]
        found.append(
            (str(row["metadata.step_id"]), {f: step[f] for f in MODEL_CALL_FIELDS})
        )
    return found


def test_replay_fills_in_what_live_left_none_for_a_call_handed_over_in_two_conversations(
    tmp_path: Path,
) -> None:
    # each conversation's model proposes "a" once: the bridge tracks handouts
    # across its conversations, so live it can't tie the second result to one;
    # replay counts proposals per conversation
    done = ModelOutput.from_content("mockllm/model", "done")
    [log] = eval(
        Task(
            dataset=[Sample(input="go")],
            solver=as_solver(two_conversations()),
            sentinel=observe_only(replay_check()),
        ),
        model=get_model(
            "mockllm/model", custom_outputs=[calls("a"), done, calls("a"), done]
        ),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error
    assert log.samples is not None
    first, _, second, _ = [
        str(e.uuid)
        for e in log.samples[0].events
        if isinstance(e, ModelEvent) and e.requested_model is not None
    ]
    frames = scanned(
        tmp_path,
        replayed=as_scanner(replay_check()),
        recorded=as_reader("replay_check"),
    )
    none = _hash(None)

    # live, the first conversation's result has the four fields and the
    # second's has them None
    live = [digests for _, digests in model_call_fields(frames["recorded"])]
    assert sorted(set(d.values()) == {none} for d in live) == [False, True]
    assert sum(none in d.values() for d in live) == 1
    # replay builds both results with the four fields set
    replayed = dict(model_call_fields(frames["replayed"]))
    assert replayed.keys() == {f"{first}/a", f"{second}/a"}
    assert all(none not in d.values() for d in replayed.values())
    # so replay differs from live in those fields at the second result only
    differences = replay_check_differences(frames["replayed"], frames["recorded"])
    rows = records(differences)
    assert {(row["step_id"], row["stage"]) for row in rows} == {
        (f"{second}/a", "tool_result")
    }
    assert {f"step.{f}" for f in MODEL_CALL_FIELDS} <= {row["field"] for row in rows}
