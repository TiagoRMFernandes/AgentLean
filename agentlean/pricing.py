"""Model pricing data (USD per million tokens) for cost estimation.

Prices are approximate and change over time — treat cost estimates as guidance,
not billing figures. Update this table as vendors revise pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for a model."""

    input_per_m: float
    output_per_m: float
    # Some models have a separate cached-input tier
    cached_input_per_m: Optional[float] = None


# Keys are the model ID strings passed to the API.
MODEL_PRICES: dict[str, ModelPrice] = {
    # --- Anthropic Claude 4 family ---
    "claude-opus-4-7": ModelPrice(input_per_m=15.0, output_per_m=75.0, cached_input_per_m=1.50),
    "claude-sonnet-4-6": ModelPrice(input_per_m=3.0, output_per_m=15.0, cached_input_per_m=0.30),
    "claude-haiku-4-5-20251001": ModelPrice(input_per_m=0.80, output_per_m=4.0, cached_input_per_m=0.08),
    # --- Anthropic Claude 3.x family (legacy) ---
    "claude-3-5-sonnet-20241022": ModelPrice(input_per_m=3.0, output_per_m=15.0, cached_input_per_m=0.30),
    "claude-3-5-haiku-20241022": ModelPrice(input_per_m=0.80, output_per_m=4.0, cached_input_per_m=0.08),
    "claude-3-opus-20240229": ModelPrice(input_per_m=15.0, output_per_m=75.0, cached_input_per_m=1.50),
    "claude-3-sonnet-20240229": ModelPrice(input_per_m=3.0, output_per_m=15.0),
    "claude-3-haiku-20240307": ModelPrice(input_per_m=0.25, output_per_m=1.25, cached_input_per_m=0.03),
    # --- OpenAI GPT-4o family ---
    "gpt-4o": ModelPrice(input_per_m=2.50, output_per_m=10.0, cached_input_per_m=1.25),
    "gpt-4o-mini": ModelPrice(input_per_m=0.15, output_per_m=0.60, cached_input_per_m=0.075),
    "gpt-4o-2024-11-20": ModelPrice(input_per_m=2.50, output_per_m=10.0, cached_input_per_m=1.25),
    "gpt-4-turbo": ModelPrice(input_per_m=10.0, output_per_m=30.0),
    "gpt-4-turbo-preview": ModelPrice(input_per_m=10.0, output_per_m=30.0),
    "gpt-4": ModelPrice(input_per_m=30.0, output_per_m=60.0),
    "gpt-3.5-turbo": ModelPrice(input_per_m=0.50, output_per_m=1.50),
    # --- OpenAI o1/o3 family ---
    "o1": ModelPrice(input_per_m=15.0, output_per_m=60.0, cached_input_per_m=7.50),
    "o1-mini": ModelPrice(input_per_m=3.0, output_per_m=12.0, cached_input_per_m=1.50),
    "o3-mini": ModelPrice(input_per_m=1.10, output_per_m=4.40, cached_input_per_m=0.55),
}

# Fallback price used when a model is not found in the table.
_FALLBACK_PRICE = ModelPrice(input_per_m=3.0, output_per_m=15.0)


def get_price(model: str) -> ModelPrice:
    """Return the ModelPrice for *model*, falling back to a generic price."""
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    # Partial-match heuristics so callers don't need exact snapshot IDs.
    lower = model.lower()
    if "opus" in lower:
        return ModelPrice(input_per_m=15.0, output_per_m=75.0)
    if "sonnet" in lower:
        return ModelPrice(input_per_m=3.0, output_per_m=15.0)
    if "haiku" in lower:
        return ModelPrice(input_per_m=0.80, output_per_m=4.0)
    if "gpt-4o-mini" in lower:
        return MODEL_PRICES["gpt-4o-mini"]
    if "gpt-4o" in lower:
        return MODEL_PRICES["gpt-4o"]
    if "gpt-4" in lower:
        return ModelPrice(input_per_m=10.0, output_per_m=30.0)
    return _FALLBACK_PRICE


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Return estimated USD cost for a single API call.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model ID string.

    Returns:
        Estimated cost in USD.
    """
    price = get_price(model)
    return (
        input_tokens * price.input_per_m / 1_000_000
        + output_tokens * price.output_per_m / 1_000_000
    )
