"""What the real runs would cost: model calls, tokens and dollars per monitor and haystack size.

Each model-calling monitor makes one call per monitored step in each of the two live runs and the two replays, so four times the calls of one run. The input tokens are measured from the prompts the monitor sent with the deterministic judge, which are the prompts a real model is sent, since `Host.score()`'s prompt does not depend on the model; a real model that gives an invalid answer is asked again, which this does not count. Output tokens are assumed, from a real model's in the smoke run, since the deterministic judge's answers are short. A real model that answers in a form `Host.score()` cannot use is asked again: gpt-4o-mini was asked 27 times for 23 steps in that run, so allow about a fifth more.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .prompts import Request

RUNS = 4
"""Two live runs and two replays."""

OUTPUT_TOKENS = 180
"""The assumed output tokens of one call, an explanation and a score through the answer tool: gpt-4o-mini averaged 179 in the 20-turn smoke run of 2026-10-26."""


@dataclass(frozen=True)
class Price:
    """A model's price in dollars per million tokens."""

    model: str
    input: float
    output: float
    source: str


PRICES = [
    Price(
        "openai/gpt-4o-mini", 0.15, 0.60, "OpenAI list price, not rechecked 2026-10-26"
    ),
    Price(
        "anthropic/claude-haiku-5-5",
        0.10,
        0.50,
        "Anthropic list price, cached 2026-10-06",
    ),
    Price("openai/gpt-4.1", 2.00, 8.00, "OpenAI list price, not rechecked 2026-10-26"),
    Price(
        "anthropic/claude-sonnet-5-5",
        2.00,
        10.00,
        "Anthropic list price, cached 2026-10-06",
    ),
]
"""Candidate monitor models: two cheap, two stronger. Prices without prompt caching, so an upper bound where the provider caches the prompt's shared opening."""


@dataclass(frozen=True)
class Spend:
    """The real runs' calls and cost for one monitor at one haystack size."""

    monitor: str
    turns: int
    calls_per_run: int
    calls: int
    mean_input_tokens: float
    input_tokens: int
    output_tokens: int
    dollars: dict[str, float]


def spend(monitor: str, turns: int, input_tokens: list[int]) -> Spend:
    """The spend for `monitor` at `turns`, from one run's calls and their input tokens."""
    calls_per_run = len(input_tokens)
    total_input = RUNS * sum(input_tokens)
    total_output = RUNS * calls_per_run * OUTPUT_TOKENS
    return Spend(
        monitor=monitor,
        turns=turns,
        calls_per_run=calls_per_run,
        calls=RUNS * calls_per_run,
        mean_input_tokens=sum(input_tokens) / calls_per_run if calls_per_run else 0.0,
        input_tokens=total_input,
        output_tokens=total_output,
        dollars={
            p.model: (total_input * p.input + total_output * p.output) / 1_000_000
            for p in PRICES
        },
    )


def used(requests: dict[str, Iterable[Request]], model: str) -> dict[str, Any]:
    """What a monitor's calls used in each run, as the provider reported it, and what that cost at `model`'s price, if it is one of `PRICES`."""
    price = next((p for p in PRICES if p.model == model), None)
    runs: dict[str, Any] = {}
    for run, found in requests.items():
        calls = list(found)
        input = sum(r.used_input for r in calls)
        output = sum(r.used_output for r in calls)
        runs[run] = {
            "calls": len(calls),
            "input_tokens": input,
            "output_tokens": output,
            "dollars": (input * price.input + output * price.output) / 1_000_000
            if price
            else None,
        }
    return runs
