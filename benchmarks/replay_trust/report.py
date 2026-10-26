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
        f"**Outcome: {results['outcome']}.**",
        "",
        "## Decisions",
        "",
        "Each monitor is decided on each haystack on its own, never on haystacks pooled: trusted only where every measurement is, not trusted where any finds a difference, and otherwise inconclusive, where a sample is too small to trust. A monitor is trusted only where every haystack is, and the run only where every monitor is. The findings say why a haystack is not trusted or inconclusive.",
        "",
        "| monitor | haystack | outcome | findings |",
        "|---|---|---|---|",
        *[
            f"| {name} | {where} | {_outcome(m['outcome'])} | {'; '.join(why for _, why in m['findings'])} |"
            for name, monitor in results["monitors"].items()
            for where, m in _rows(monitor)
        ],
        *[
            f"| **{name}** | all, each decided | {_outcome(monitor['outcome'])} | |"
            for name, monitor in results["monitors"].items()
        ],
        "",
        "## 1. The same prompt",
        "",
        "Steps whose first monitor model request is identical live and in replay, after the nonce is replaced; beside them the same comparison between the two live runs and between the two replays. A monitor that asks its model again after an unusable answer sends that answer back, so those further requests are model noise: they are counted (asked again, in `live_a` and `replay_1`), not compared.",
        "",
        "| monitor | haystack | steps | identical | known difference | live = live | replay = replay | asked again | outcome |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    differences: list[str] = []
    for name, monitor in results["monitors"].items():
        for where, m in _rows(monitor):
            p = m["same_prompt"]
            if p is None:
                continue
            lines.append(
                f"| {name} | {where} | {p['compared']} | {p['identical']} | {p['explained']} | {_maybe(p['live_floor'])} | {p['replay_floor']} | {p['asked_again']['live_a']} / {p['asked_again']['replay_1']} | {_outcome(m['same_prompt_outcome'])} |"
            )
            for known, count in p["by_known"].items():
                if count:
                    differences.append(
                        f"- {name}, {where}: {count} steps differ by a known difference: {known}."
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
            "Steps whose flag differs between two runs: each pair of a live run and a replay (`live_a`, `live_b` against `replay_1`, `replay_2`, in that order), and the noise reference, live against live and replay against replay. Two exact two-sided McNemar tests for each pair, on the per-step flags: shift p, the steps only the live run flagged against those only the replay flagged; excess p, the steps where the pair disagrees against those where the noise pair does. The smallest of each is shown, beside the level each test is held to, 5% divided by the number of tests. Where the runs reproduce themselves (no step differs within the noise pairs), any step a pair disagrees on fails. Unmatched steps lack a row in some run: replay did not reach them, or a live run has neither a report nor a recorded abstention; either fails.",
            "",
            "| monitor | haystack | steps | unmatched | live vs replay | live vs live | replay vs replay | shift p | excess p | level | mean abs. difference | outcome |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, monitor in results["monitors"].items():
        for where, m in _rows(monitor):
            v = m["same_verdict"]
            lines.append(
                f"| {name} | {where} | {v['steps']} | {v['unmatched']} | {' / '.join(str(_disagreements(p)) for p in v['pairs'].values())} | {_maybe(_disagreements(v['noise'].get('live_a vs live_b')))} | {_disagreements(v['noise']['replay_1 vs replay_2'])} | {_number(min(p['p_value'] for p in v['pairs'].values()))} | {_number(min(p['excess_p_value'] for p in v['pairs'].values()))} | {_number(v['alpha'])} | {_number(v['mean_abs_difference'])} | {_outcome(m['same_verdict_outcome'])} |"
            )
    lines.extend(
        [
            "",
            "## 3. The same needles found",
            "",
            "Recall on the needles and false-positive rate on the other steps, with 95% Wilson intervals, live (`live_a`) and in replay (`replay_1`), as a description. Each needle is checked one by one: changed counts the needles the live runs agree on (both flag it, or neither) that a replay flags differently, which fails acceptance. A needle is missed in a run where no step covers it or the monitor reported no score there; a missed needle fails acceptance, as does a recall or false-positive rate over no steps. With no needles labelled, this is not measured.",
            "",
            "| monitor | haystack | needles | missed | changed | recall live | recall replay | FPR live | FPR replay | outcome |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    missed: list[str] = []
    for name, monitor in results["monitors"].items():
        for where, m in _rows(monitor):
            n = m["needles_found"]
            if n is None:
                lines.append(f"| {name} | {where} | 0 | | | | | | | n/a |")
                continue
            lost = {run: ids for run, ids in n["missed"].items() if ids}
            lines.append(
                f"| {name} | {where} | {n['labelled']} | {sum(len(ids) for ids in lost.values())} | {len(n['changed'])} | {_proportion(n['recall']['live_a'])} | {_proportion(n['recall']['replay_1'])} | {_proportion(n['false_positive_rate']['live_a'])} | {_proportion(n['false_positive_rate']['replay_1'])} | {_outcome(m['needles_found_outcome'])} |"
            )
            missed.extend(
                f"- {name}, {where}: {run} missed {', '.join(f'`{i}`' for i in ids)}"
                for run, ids in lost.items()
            )
            missed.extend(
                f"- {name}, {where}: `{needle}` changed, flagged {n['flagged'][needle]}"
                for needle in n["changed"]
            )
    if missed:
        lines.extend(["", "Needles missed or changed:", "", *missed])
    human = [
        (name, where, m)
        for name, monitor in results["monitors"].items()
        for where, m in _rows(monitor)
        if m["same_human_answers"] is not None
    ]
    if human:
        lines.extend(
            [
                "",
                "## 4. The same human answers",
                "",
                "Calls `human()` was asked about: the calls the haystacks put to a person, those the person answered in each live run, and those each replay answered with the answer recorded in `live_a`, as the recorded answer.",
                "",
                "| monitor | haystack | expected | asked live | given again in replay | outcome |",
                "|---|---|---|---|---|---|",
                *[
                    f"| {name} | {where} | {h['expected']} | {' / '.join(str(v) for v in h['asked'].values())} | {' / '.join(str(v) for v in h['given_again'].values())} | {_outcome(m['same_human_answers_outcome'])} |"
                    for name, where, m in human
                    for h in [m["same_human_answers"]]
                ],
            ]
        )
        for name, _, m in human:
            lines.extend(
                f"- {name}, {d['sample']} `{d['call']}`: answers {d['answers']}, sources {d['sources']}"
                for d in m["same_human_answers"]["differences"]
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
        total = _sum([u["dollars"] for _, _, u in usage])
        estimated = any(u["dollars_basis"] == "estimated" for _, _, u in usage)
        lines.extend(
            [
                "",
                "## Measured usage",
                "",
                "The monitor model's calls in each run, with the tokens the provider reported, uncached input, cache reads and cache writes apart, and their cost: inspect_ai's where it priced the calls, else at inspect_ai's model data or, labelled estimated, at the assumed rates under the spend estimate, each token class at its own rate.",
                "",
                "| monitor | run | calls | input tokens | cache read | cache write | output tokens | dollars | priced by |",
                "|---|---|---|---|---|---|---|---|---|",
                *[
                    f"| {name} | {run} | {u['calls']} | {u['input_tokens']} | {u['cache_read_tokens']} | {u['cache_write_tokens']} | {u['output_tokens']} | {_dollars(u['dollars'])} | {u['dollars_basis'] or 'no price'} |"
                    for name, run, u in usage
                ],
                f"| **total** | | {sum(u['calls'] for _, _, u in usage)} | {sum(u['input_tokens'] for _, _, u in usage)} | {sum(u['cache_read_tokens'] for _, _, u in usage)} | {sum(u['cache_write_tokens'] for _, _, u in usage)} | {sum(u['output_tokens'] for _, _, u in usage)} | {_dollars(total)}{' (estimated)' if estimated and total is not None else ''} | |",
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
                "An estimate. Model calls are four times one run's (two live runs, two replays); input tokens are measured from the prompts sent, output tokens assumed at 180 a call, as gpt-4o-mini averaged in the smoke run. A model asked again after an unusable answer adds calls: about a fifth more for gpt-4o-mini. Every input token is priced uncached, an upper bound where the provider caches the prompt's opening. Prices per million tokens, inspect_ai's where its model data has them, else assumed:",
                "",
                *[
                    f"- `{p['model']}`: ${p['input']:.2f} in, ${p['output']:.2f} out, ${p['input_cache_read']:.3f} cache read, ${p['input_cache_write']:.3f} cache write ({'assumed: ' if p['estimated'] else ''}{p['source']})"
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
    return list(monitor["cells"].items())


def _settings(settings: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in settings.items() if k != "out")


def _proportion(p: dict[str, Any]) -> str:
    if p["rate"] is None:
        return f"{p['k']}/{p['n']}"
    return f"{p['k']}/{p['n']} = {p['rate']:.2f} [{p['low']:.2f}, {p['high']:.2f}]"


def _disagreements(pair: dict[str, Any] | None) -> int | None:
    return None if pair is None else pair["first_only"] + pair["second_only"]


def _maybe(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def _outcome(value: str | None) -> str:
    return "n/a" if value is None else value if value == "trusted" else f"**{value}**"


def _dollars(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _sum(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else sum(v or 0.0 for v in values)
