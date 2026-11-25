"""The spend ledger: what each batch of generation or judging cost, by model, priced from its logs' token usage, so a run can stop before it crosses its budget.

`ledger.csv` holds one row per batch and model: when, what kind of batch, its label, the model, its tokens and their cost at `PRICES`. Generation batches are priced from their eval logs' `stats.model_usage`; judging from the monitors' results. Provider-side spend can lag behind and can differ from list prices, so a run keeps a margin under its budget (`MARGIN`).
"""

from __future__ import annotations

import csv
import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.model import ModelCost, ModelUsage

PRICES: dict[str, ModelCost] = {
    "anthropic/claude-sonnet-5-5": ModelCost(
        input=2.00, output=10.00, input_cache_read=0.20, input_cache_write=2.50
    ),
    "openai/gpt-6.2-sol": ModelCost(
        input=2.00, output=10.00, input_cache_read=0.20, input_cache_write=2.00
    ),
    "anthropic/claude-fable-5-2": ModelCost(
        input=8.00, output=40.00, input_cache_read=0.80, input_cache_write=10.00
    ),
    "openai/gpt-6.2-astra": ModelCost(
        input=8.00, output=40.00, input_cache_read=0.80, input_cache_write=8.00
    ),
    "together/zai-org/GLM-5.3": ModelCost(
        input=1.75, output=5.50, input_cache_read=0.325, input_cache_write=1.75
    ),
    "anthropic/claude-haiku-5-5": ModelCost(
        input=0.10, output=0.50, input_cache_read=0.01, input_cache_write=0.125
    ),
}
"""Dollars per million tokens, from the plan's list prices (2026-11-24): Sonnet-class $2/$10 and Fable-class about $8/$40 for input/output, GLM-5.3 on Together's US-only endpoint $1.75/$5.50 with cached input at $0.325 (1.25 times Together's list price, 2026-11-25; set TOGETHER_BASE_URL=https://us-api.together.ai/v1), Haiku 5.5 $0.10/$0.50 up to 100K-token prompts. Cache reads are priced at 0.1 times input for every model, and cache writes at 1.25 times for Anthropic and 1 times for OpenAI. These are assumptions: a cached read priced higher than the provider charges makes the ledger overstate spend, which errs toward stopping early."""

UNPRICED = ModelCost(
    input=10.00, output=50.00, input_cache_read=10.00, input_cache_write=12.50
)
"""The price assumed for a model not in `PRICES`, such as a setting's own scorer model: above every listed price, so an unknown model never makes the ledger understate spend."""

MARGIN = 25.0
"""Dollars a run stays under its budget on the projected figure, since provider-side spend lags."""

FIELDS = (
    "time",
    "kind",
    "label",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost",
)


@dataclass(frozen=True)
class Entry:
    """One batch's spend on one model."""

    kind: str
    """`generate` or `judge`."""

    label: str
    model: str
    usage: ModelUsage
    cost: float


def price(model: str, usage: ModelUsage) -> float:
    """What `usage` of `model` costs at `PRICES` (or `UNPRICED`). Input tokens are counted as inspect_ai reports them, without the cached ones."""
    found = PRICES.get(_bare(model), UNPRICED)
    return (
        usage.input_tokens * found.input
        + usage.output_tokens * found.output
        + (usage.input_tokens_cache_read or 0) * found.input_cache_read
        + (usage.input_tokens_cache_write or 0) * found.input_cache_write
    ) / 1_000_000


def entries(kind: str, label: str, usage: Mapping[str, ModelUsage]) -> list[Entry]:
    """An entry per model in `usage`, priced."""
    return [
        Entry(kind, label, model, u, price(model, u))
        for model, u in sorted(usage.items())
    ]


def record(ledger: Path, new: Sequence[Entry]) -> None:
    """Append `new` to the ledger file, creating it with its header."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    fresh = not ledger.exists()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with ledger.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for e in new:
            writer.writerow(
                {
                    "time": now,
                    "kind": e.kind,
                    "label": e.label,
                    "model": e.model,
                    "input_tokens": e.usage.input_tokens,
                    "output_tokens": e.usage.output_tokens,
                    "cache_read_tokens": e.usage.input_tokens_cache_read or 0,
                    "cache_write_tokens": e.usage.input_tokens_cache_write or 0,
                    "reasoning_tokens": e.usage.reasoning_tokens or 0,
                    "cost": f"{e.cost:.4f}",
                }
            )


def spent(ledger: Path, kind: str | None = None) -> float:
    """Dollars the ledger records, of `kind` or of every kind."""
    if not ledger.exists():
        return 0.0
    with ledger.open(newline="") as f:
        return sum(
            float(r["cost"])
            for r in csv.DictReader(f)
            if kind is None or r["kind"] == kind
        )


def _bare(model: str) -> str:
    # a model as a log names it, with any provider prefix the API adds
    # (anthropic/claude-sonnet-5-5-20260928) reduced to the priced name
    for name in PRICES:
        if (
            model == name
            or model.startswith(name + "-")
            or model.endswith("/" + name.split("/")[-1])
        ):
            return name
    return model
