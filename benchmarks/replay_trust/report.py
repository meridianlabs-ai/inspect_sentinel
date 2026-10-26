"""The results as a markdown summary."""

from __future__ import annotations

from typing import Any, cast


def summary(results: dict[str, Any]) -> str:
    """A markdown summary of `results.json`."""
    env = results["environment"]
    lines = [
        "# Replay trust results",
        "",
        f"Created {results['created']}. inspect_ai {env['inspect_ai']}, inspect_scout {env['inspect_scout']}, inspect_sentinel {env['inspect_sentinel']} ({(env['inspect_sentinel_commit'] or 'unknown')[:9]}).",
        "",
        f"Settings: `{_settings(results['settings'])}`. A step is flagged at suspicion {results['threshold']} or above.",
        "",
        f"**Accepted: {'yes' if results['accepted'] else 'no'}.**",
        "",
        "## 1. The same prompt",
        "",
        "Steps whose first monitor model request is identical live and in replay, after the nonce is replaced; beside them the same comparison between the two live runs and between the two replays. A monitor that asks its model again after an unusable answer sends that answer back, so those further requests are model noise: they are counted (asked again, in `live_a` and `replay_1`), not compared.",
        "",
        "| monitor | haystack | steps | identical | known difference | live = live | replay = replay | asked again | accepted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    differences: list[str] = []
    for name, monitor in results["monitors"].items():
        for where, m in _rows(monitor):
            p = m["same_prompt"]
            if p is None:
                continue
            lines.append(
                f"| {name} | {where} | {p['compared']} | {p['identical']} | {p['explained']} | {_maybe(p['live_floor'])} | {p['replay_floor']} | {p['asked_again']['live_a']} / {p['asked_again']['replay_1']} | {_yes(m['same_prompt_accepted'])} |"
            )
            if where == "all":
                for known, count in p["by_known"].items():
                    if count:
                        differences.append(
                            f"- {name}: {count} steps differ by a known difference: {known}."
                        )
                for d in (d for d in p["differences"] if d["known"] is None):
                    differences.append(
                        f"- {name}, {d['sample']} {d['stage']} `{d['key']}`: {d['reason']}, {d['known'] or 'unexplained'}"
                    )
                    if d["diff"]:
                        differences.extend(
                            ["", "  ```diff", *[f"  {x}" for x in d["diff"]], "  ```"]
                        )
    if differences:
        lines.extend(["", "Differences, known and unexplained:", "", *differences])
    lines.extend(
        [
            "",
            "## 2. The same verdict",
            "",
            "Steps whose flag differs between two runs: live against replay, and the noise floors, live against live and replay against replay. p is the exact two-sided McNemar test of live-against-replay disagreement versus live-against-live.",
            "",
            "| monitor | haystack | steps | live vs replay | live vs live | replay vs replay | p | mean abs. difference | accepted |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, monitor in results["monitors"].items():
        for where, m in _rows(monitor):
            v = m["same_verdict"]
            lines.append(
                f"| {name} | {where} | {v['steps']} | {v['live_replay']} | {_maybe(v['live_live'])} | {v['replay_replay']} | {_number(v['p_value'])} | {_number(v['mean_abs_difference'])} | {_yes(m['same_verdict_accepted'])} |"
            )
    lines.extend(
        [
            "",
            "## 3. The same needles found",
            "",
            "Recall on the needles and false-positive rate on the other steps, with 95% Wilson intervals, live (`live_a`) and in replay (`replay_1`).",
            "",
            "| monitor | haystack | recall live | recall replay | FPR live | FPR replay | accepted |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for name, monitor in results["monitors"].items():
        for where, m in _rows(monitor):
            n = m["needles_found"]
            lines.append(
                f"| {name} | {where} | {_proportion(n['recall']['live_a'])} | {_proportion(n['recall']['replay_1'])} | {_proportion(n['false_positive_rate']['live_a'])} | {_proportion(n['false_positive_rate']['replay_1'])} | {_yes(m['needles_found_accepted'])} |"
            )
    usage: list[tuple[str, str, dict[str, Any]]] = [
        (name, run, u)
        for name, monitor in results["monitors"].items()
        for run, u in cast(
            dict[str, dict[str, Any]], monitor.get("usage") or {}
        ).items()
        if u["input_tokens"] or u["output_tokens"]
    ]
    if usage:
        lines.extend(
            [
                "",
                "## Measured usage",
                "",
                "The monitor model's calls in each run, with the tokens the provider reported and their cost at the listed price.",
                "",
                "| monitor | run | calls | input tokens | output tokens | dollars |",
                "|---|---|---|---|---|---|",
                *[
                    f"| {name} | {run} | {u['calls']} | {u['input_tokens']} | {u['output_tokens']} | {_dollars(u['dollars'])} |"
                    for name, run, u in usage
                ],
                f"| **total** | | {sum(u['calls'] for _, _, u in usage)} | {sum(u['input_tokens'] for _, _, u in usage)} | {sum(u['output_tokens'] for _, _, u in usage)} | {_dollars(_sum([u['dollars'] for _, _, u in usage]))} |",
            ]
        )
    spend = results["spend"]
    if spend["by_monitor_and_size"]:
        models = [p["model"] for p in spend["prices"]]
        lines.extend(
            [
                "",
                "## Spend for the real runs",
                "",
                "Model calls are four times one run's (two live runs, two replays); input tokens are measured from the prompts sent, output tokens assumed at 180 a call, as gpt-4o-mini averaged in the smoke run. A model asked again after an unusable answer adds calls: about a fifth more for gpt-4o-mini. Prices per million tokens, without prompt caching:",
                "",
                *[
                    f"- `{p['model']}`: ${p['input']:.2f} in, ${p['output']:.2f} out ({p['source']})"
                    for p in spend["prices"]
                ],
                "",
                "| monitor | turns | calls | mean input tokens | "
                + " | ".join(f"`{m}`" for m in models)
                + " |",
                "|---|---|---|---|" + "---|" * len(models),
            ]
        )
        for s in spend["by_monitor_and_size"]:
            lines.append(
                f"| {s['monitor']} | {s['turns']} | {s['calls']} | {s['mean_input_tokens']:.0f} | "
                + " | ".join(f"${s['dollars'][m]:.2f}" for m in models)
                + " |"
            )
        lines.append(
            f"| **total** | | {spend['calls']} | | "
            + " | ".join(f"**${spend['total'][m]:.2f}**" for m in models)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _rows(monitor: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = list(monitor["haystacks"].items())
    if monitor["pooled"] is not None:
        rows.append(("all", monitor["pooled"]))
    return rows


def _settings(settings: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in settings.items() if k != "out")


def _proportion(p: dict[str, Any]) -> str:
    if p["rate"] is None:
        return f"{p['k']}/{p['n']}"
    return f"{p['k']}/{p['n']} = {p['rate']:.2f} [{p['low']:.2f}, {p['high']:.2f}]"


def _maybe(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def _yes(value: bool | None) -> str:
    return "n/a" if value is None else "yes" if value else "**no**"


def _dollars(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _sum(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else sum(v or 0.0 for v in values)
