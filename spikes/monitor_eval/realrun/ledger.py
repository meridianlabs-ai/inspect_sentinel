"""The shared spend ledger: list-price cost from an eval log's recorded token usage, appended one line per run."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.log import EvalLog

LEDGER = Path("/home/eric/.scratch/r8-overnight/spend.jsonl")
STOP_AT_USD = 235.0

# USD per million tokens: input, output, cache write, cache read. Input tokens are charged at the full rate even where a provider counts cache reads inside them, so the figure errs high.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "openai/gpt-5-mini": (0.25, 2.00, 0.25, 0.025),
    "google/gemini-2.5-flash": (0.30, 2.50, 0.30, 0.075),
}


def total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(
        float(json.loads(line).get("usd", 0))
        for line in LEDGER.read_text().splitlines()
        if line.strip()
    )


def check() -> float:
    spent = total()
    if spent >= STOP_AT_USD:
        raise SystemExit(
            f"Spend ledger at ${spent:.2f}; new runs stop at ${STOP_AT_USD}."
        )
    return spent


def record(log: EvalLog, what: str, agent: str = "generator") -> float:
    usd = 0.0
    tokens_in = tokens_out = 0
    models = []
    for name, u in (log.stats.model_usage or {}).items():
        key = next((k for k in PRICES if name.endswith(k.split("/", 1)[1])), None)
        if key is None:
            raise SystemExit(f"No price for {name}; add it to ledger.PRICES.")
        p_in, p_out, p_cw, p_cr = PRICES[key]
        cw = u.input_tokens_cache_write or 0
        cr = u.input_tokens_cache_read or 0
        usd += (
            u.input_tokens * p_in + u.output_tokens * p_out + cw * p_cw + cr * p_cr
        ) / 1e6
        tokens_in += u.input_tokens + cw + cr
        tokens_out += u.output_tokens
        models.append(name)
    line = {
        "agent": agent,
        "what": what,
        "model": ",".join(models),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "usd": round(usd, 5),
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(line) + "\n")
    return usd
