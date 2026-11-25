"""Judge model prices, for the cost per call: inspect_ai's where its model data has one, else the list prices assumed here, labelled estimates."""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.model import ModelCost, ModelUsage, compute_model_cost, get_model_info

ASSUMED: dict[str, tuple[ModelCost, str]] = {
    "anthropic/claude-haiku-5-5": (
        ModelCost(
            input=0.10, output=0.50, input_cache_read=0.01, input_cache_write=0.125
        ),
        "Anthropic list price for prompts up to 100K tokens, as cached 2026-11-03; cache reads at 0.1 and 5-minute writes at 1.25 times the input price",
    ),
    "anthropic/claude-sonnet-5-5": (
        ModelCost(
            input=2.00, output=10.00, input_cache_read=0.10, input_cache_write=2.50
        ),
        "Anthropic list price, as cached 2026-11-03; 5-minute cache writes at 1.25 times the input price",
    ),
    "openai/gpt-4.1-mini": (
        ModelCost(
            input=0.40, output=1.60, input_cache_read=0.10, input_cache_write=0.40
        ),
        "OpenAI list price, not rechecked 2026-11-24",
    ),
}
"""Prices in dollars per million tokens, for models inspect_ai has no price for, with where each came from."""


def price(model: str) -> tuple[ModelCost, str] | None:
    """`model`'s price and its source: inspect_ai's model data, else `ASSUMED`; None where neither has one."""
    info = get_model_info(model)
    if info is not None and info.cost is not None:
        return info.cost, "inspect_ai model data"
    return ASSUMED.get(model)


def cost(model: str, usages: Sequence[ModelUsage]) -> float | None:
    """What `usages` cost: the provider's reported cost where every call has one, else priced by `price()`; None where `model` has no price."""
    if usages and all(u.total_cost is not None for u in usages):
        return sum(u.total_cost or 0.0 for u in usages)
    found = price(model)
    if found is None:
        return None
    return sum(compute_model_cost(found[0], u) for u in usages)
