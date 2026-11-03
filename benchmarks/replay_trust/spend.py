"""What the real runs would cost: model calls, tokens and dollars per monitor and haystack size.

Each model-calling monitor makes one call per monitored step in each of the two live runs and the two replays, so four times the calls of one run. The input tokens are measured from the prompts the monitor sent with the deterministic judge, which are the prompts a real model is sent, since `Host.score()`'s prompt does not depend on the model; a real model that gives an invalid answer is asked again, which this does not count. Output tokens are assumed, from a real model's in the smoke run, since the deterministic judge's answers are short. A real model that answers in a form `Host.score()` cannot use is asked again: gpt-4o-mini was asked 27 times for 23 steps in that run, so allow about a fifth more.

Prices are inspect_ai's where its model data has them, else assumed (`PRICES`), list prices with the provider's usual cache rates; a cost at assumed rates is labelled an estimate. The spend estimate assumes no prompt caching, so it is an upper bound where the provider caches the prompt's shared opening. The cost of calls made prices uncached input, cache reads and cache writes apart, as the provider reported them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import ModelCost, compute_model_cost, get_model_info

from .prompts import Request

RUNS = 4
"""Two live runs and two replays."""

OUTPUT_TOKENS = 180
"""The assumed output tokens of one call, an explanation and a score through the answer tool: gpt-4o-mini averaged 179 in the 20-turn smoke run of 2026-10-26."""


@dataclass(frozen=True)
class Price:
    """A model's assumed price in dollars per million tokens, for where inspect_ai has none."""

    model: str
    cost: ModelCost
    """Uncached input, output, cache reads and cache writes."""

    source: str


PRICES = [
    Price(
        "openai/gpt-4o-mini",
        ModelCost(
            input=0.15, output=0.60, input_cache_read=0.075, input_cache_write=0.15
        ),
        "OpenAI list price, not rechecked 2026-10-26; cache reads at half the input price and writes at the input price, OpenAI's usual rates",
    ),
    Price(
        "anthropic/claude-haiku-5-5",
        ModelCost(
            input=0.10, output=0.50, input_cache_read=0.01, input_cache_write=0.125
        ),
        "Anthropic list price, cached 2026-10-06; cache reads at 0.1 and 5-minute writes at 1.25 times the input price, Anthropic's usual multipliers",
    ),
    Price(
        "openai/gpt-4.1",
        ModelCost(
            input=2.00, output=8.00, input_cache_read=0.50, input_cache_write=2.00
        ),
        "OpenAI list price, not rechecked 2026-10-26; cache reads at a quarter of the input price and writes at the input price, OpenAI's usual rates",
    ),
    Price(
        "anthropic/claude-sonnet-5-5",
        ModelCost(
            input=2.00, output=10.00, input_cache_read=0.20, input_cache_write=2.50
        ),
        "Anthropic list price, cached 2026-10-06; cache reads at 0.1 and 5-minute writes at 1.25 times the input price, Anthropic's usual multipliers",
    ),
]
"""Candidate monitor models, two cheap and two stronger, with the prices assumed where inspect_ai's model data has none."""


@dataclass(frozen=True)
class Pricing:
    """The price a model's tokens are charged at, and where it came from."""

    cost: ModelCost
    estimated: bool
    """Whether the rates are assumed (`PRICES`), not inspect_ai's model data."""

    source: str


def pricing(model: str) -> Pricing | None:
    """`model`'s price: inspect_ai's, where its model data has one, else the one assumed in `PRICES`; None where neither has one."""
    info = get_model_info(model)
    if info is not None and info.cost is not None:
        return Pricing(info.cost, estimated=False, source="inspect_ai model data")
    assumed = next((p for p in PRICES if p.model == model), None)
    if assumed is None:
        return None
    return Pricing(assumed.cost, estimated=True, source=assumed.source)


@dataclass(frozen=True)
class Spend:
    """The real runs' calls and estimated cost for one monitor at one haystack size, with no prompt caching."""

    monitor: str
    turns: int
    calls_per_run: int
    calls: int
    mean_input_tokens: float
    input_tokens: int
    output_tokens: int
    dollars: dict[str, float]


def spend(monitor: str, turns: int, input_tokens: list[int]) -> Spend:
    """The spend for `monitor` at `turns`, from one run's calls and their input tokens, at each candidate's input and output price."""
    calls_per_run = len(input_tokens)
    total_input = RUNS * sum(input_tokens)
    total_output = RUNS * calls_per_run * OUTPUT_TOKENS
    dollars: dict[str, float] = {}
    for p in PRICES:
        found = pricing(p.model)
        assert found is not None
        dollars[p.model] = (
            total_input * found.cost.input + total_output * found.cost.output
        ) / 1_000_000
    return Spend(
        monitor=monitor,
        turns=turns,
        calls_per_run=calls_per_run,
        calls=RUNS * calls_per_run,
        mean_input_tokens=sum(input_tokens) / calls_per_run if calls_per_run else 0.0,
        input_tokens=total_input,
        output_tokens=total_output,
        dollars=dollars,
    )


def used(requests: dict[str, Iterable[Request]], model: str) -> dict[str, Any]:
    """What a monitor's calls used in each run, as the provider reported it, uncached input, cache reads and cache writes apart, and what it cost.

    The cost is inspect_ai's, where it priced every call; else each call priced by `pricing()`, which is an estimate where the rates are assumed; None where `model` has no price.
    """
    found = pricing(model)
    runs: dict[str, Any] = {}
    for run, made in requests.items():
        calls = list(made)
        usages = [r.usage for r in calls if r.usage is not None]
        if usages and all(u.total_cost is not None for u in usages):
            dollars: float | None = sum(u.total_cost or 0.0 for u in usages)
            basis: str | None = "inspect_ai"
        elif found is not None:
            dollars = sum(compute_model_cost(found.cost, u) for u in usages)
            basis = "estimated" if found.estimated else "inspect_ai model data"
        else:
            dollars, basis = None, None
        runs[run] = {
            "calls": len(calls),
            "input_tokens": sum(u.input_tokens for u in usages),
            "cache_read_tokens": sum(u.input_tokens_cache_read or 0 for u in usages),
            "cache_write_tokens": sum(u.input_tokens_cache_write or 0 for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "dollars": dollars,
            "dollars_basis": basis,
        }
    return runs
