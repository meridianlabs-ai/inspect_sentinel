"""The replay-trust benchmark end to end on deterministic models, so the harness does not rot between real runs."""

from pathlib import Path
from typing import Any

import pytest

from benchmarks.replay_trust.benchmark import Settings, run_benchmark
from benchmarks.replay_trust.haystack import haystack


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

    # four runs' calls for each judge, with their tokens measured
    spend = results["spend"]["by_monitor_and_size"]
    assert {s["monitor"] for s in spend} == {"tool_call_judge", "turn_judge"}
    for s in spend:
        assert s["calls"] == 4 * s["calls_per_run"] > 0
        assert s["mean_input_tokens"] > 0
    for name in ("results.json", "steps.json", "summary.md", "labels.json"):
        assert (tmp_path / name).exists()
