"""The replay-trust benchmark end to end on deterministic models, so the harness does not rot between real runs; and what it accepts, compares and prices."""

from pathlib import Path
from typing import Any, cast

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    Content,
    ContentImage,
    ContentText,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)

from benchmarks.replay_trust.benchmark import Settings, run_benchmark
from benchmarks.replay_trust.haystack import haystack
from benchmarks.replay_trust.measure import measure
from benchmarks.replay_trust.prompts import Request, request
from benchmarks.replay_trust.run import RUNS, MonitorRun, RunName, StepRecord
from benchmarks.replay_trust.spend import pricing, used


# two evals and four scans for each of four monitors; about 45 seconds alone
@pytest.mark.timeout(300)
def test_the_benchmark_runs_end_to_end_and_replay_matches_live(
    tmp_path: Path,
) -> None:
    # a short haystack that still compacts, calls its sub-agent, retries an
    # error and makes parallel calls
    results = run_benchmark(Settings(out=tmp_path, sizes=(14,), compact_at=1500))

    assert results["accepted"], (tmp_path / "summary.md").read_text()
    monitors: dict[str, Any] = results["monitors"]
    assert set(monitors) == {
        "tool_call_judge",
        "turn_judge",
        "network_rule",
        "failure_count",
    }
    for name, monitor in monitors.items():
        pooled = monitor["pooled"]
        prompt = pooled["same_prompt"]
        if monitor["calls_a_model"]:
            # every difference is a known one, and the deterministic judge
            # asks the same thing in both live runs and both replays
            assert prompt["compared"] > 0, name
            assert [d for d in prompt["differences"] if d["known"] is None] == []
            assert prompt["live_floor"] == prompt["replay_floor"] == prompt["compared"]
            assert set(prompt["asked_again"].values()) == {0}
        else:
            assert prompt is None, name
        verdict = pooled["same_verdict"]
        assert verdict["steps"] > 0, name
        assert verdict["live_replay"] == verdict["live_live"] == 0, name
        assert verdict["replay_replay"] == 0, name
        found = pooled["needles_found"]
        assert found["recall"]["live_a"] == found["recall"]["replay_1"], name
        assert (
            found["false_positive_rate"]["live_a"]
            == found["false_positive_rate"]["replay_1"]
        ), name

    # the deterministic judge flags every planted needle
    needles = haystack(14).needles
    assert monitors["tool_call_judge"]["pooled"]["needles_found"]["recall"]["live_a"][
        "k"
    ] == len(needles)

    # the person's recorded answer is given again in both replays
    for name, monitor in monitors.items():
        human = monitor["pooled"]["same_human_answers"]
        assert human["expected"] == 1, name
        assert human["asked"] == {"live_a": 1, "live_b": 1}, name
        assert human["given_again"] == {"replay_1": 1, "replay_2": 1}, name

    # four runs' calls for each judge, with their tokens measured
    spend = results["spend"]["by_monitor_and_size"]
    assert {s["monitor"] for s in spend} == {"tool_call_judge", "turn_judge"}
    for s in spend:
        assert s["calls"] == 4 * s["calls_per_run"] > 0
        assert s["mean_input_tokens"] > 0
    for name in ("results.json", "steps.json", "summary.md", "labels.json"):
        assert (tmp_path / name).exists()


def _step(key: str, **scores: float | None) -> StepRecord:
    return StepRecord(
        sample="h",
        stage="tool_call",
        key=key,
        calls=(key,),
        scores=cast(dict[RunName, float | None], scores),
    )


def _run(*steps: StepRecord) -> MonitorRun:
    return MonitorRun(
        name="rule", stage="tool_call", steps=list(steps), logs=("a", "b"), scans={}
    )


def _every_run(score: float | None) -> dict[str, float | None]:
    return {run: score for run in RUNS}


@pytest.mark.parametrize("calls_a_model", [True, False])
def test_an_empty_run_is_not_accepted(calls_a_model: bool) -> None:
    measured = measure(_run(), [], calls_a_model=calls_a_model, asked=[])
    assert not measured.accepted
    assert not measured.same_verdict.accepted
    assert not measured.needles_found.accepted
    assert measured.same_human_answers is not None
    assert not measured.same_human_answers.accepted
    if calls_a_model:
        assert measured.same_prompt is not None
        assert not measured.same_prompt.accepted


# a needle each run must reach; "other" is an ordinary step every run scored
@pytest.mark.parametrize(
    ("steps", "missed"),
    [
        ([_step("needle", **_every_run(0.9))], set[RunName]()),
        ([], set(RUNS)),
        (
            [_step("needle", live_a=0.9, live_b=0.9)],
            {"replay_1", "replay_2"},
        ),
        (
            [_step("needle", **{**_every_run(0.9), "live_a": None})],
            {"live_a"},
        ),
    ],
    ids=["reached", "no step covers it", "replay lost it", "live abstained"],
)
def test_a_missed_needle_is_reported_and_not_accepted(
    steps: list[StepRecord], missed: set[RunName]
) -> None:
    run = _run(_step("other", **_every_run(0.1)), *steps)
    found = measure(run, ["needle"], calls_a_model=False).needles_found
    assert {r for r, ids in found.missed.items() if ids} == missed
    assert all(ids == ["needle"] for r, ids in found.missed.items() if r in missed)
    assert found.accepted == (not missed)


def _model_event(*image: str) -> ModelEvent:
    content: list[Content] = [
        ContentText(text="Look at this."),
        *(ContentImage(image=i) for i in image),
    ]
    return ModelEvent(
        model="judge",
        input=list[ChatMessage]([ChatMessageUser(content=content)]),
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content("judge", "0.0"),
    )


def test_requests_differing_only_in_an_image_are_not_the_same() -> None:
    png, jpeg = "data:image/png;base64," + "A" * 200, "data:image/jpeg;base64,B"
    with_png = request(_model_event(png))
    assert with_png.text != request(_model_event(jpeg)).text
    assert with_png.text != request(_model_event()).text
    assert request(_model_event(png)).text == with_png.text
    # the image's data is compared in full and shortened where it is shown
    assert png in with_png.text
    assert png not in with_png.display
    assert "Look at this." in with_png.display


def test_cache_reads_and_writes_are_priced_at_their_own_rates() -> None:
    model = "anthropic/claude-haiku-5-5"
    price = pricing(model)
    assert price is not None
    usage = ModelUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_tokens_cache_read=2_000_000,
        input_tokens_cache_write=3_000_000,
    )
    [run] = used({"live_a": [Request(text="", tokens=0, usage=usage)]}, model).values()
    cost = price.cost
    assert run["dollars"] == pytest.approx(
        cost.input
        + cost.output
        + 2 * cost.input_cache_read
        + 3 * cost.input_cache_write
    )
    assert run["dollars"] < cost.input * 6 + cost.output
    assert run["dollars_basis"] == (
        "estimated" if price.estimated else "inspect_ai model data"
    )
    assert (
        run["input_tokens"],
        run["cache_read_tokens"],
        run["cache_write_tokens"],
    ) == (
        1_000_000,
        2_000_000,
        3_000_000,
    )
