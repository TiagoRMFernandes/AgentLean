"""Statistics tracking for AgentLean sessions and individual API calls."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallStats:
    """Statistics for a single optimised API call.

    Attributes:
        original_input_tokens: Token count before optimisation.
        optimised_input_tokens: Token count after optimisation.
        output_tokens: Tokens in the model's response.
        model: Model ID used for this call.
        estimated_cost_usd: Estimated total cost (input + output) in USD.
        estimated_cost_saved_usd: Estimated savings vs. the unoptimised call.
        tool_outputs_distilled: Number of tool result messages that were truncated.
        turns_summarised: Number of conversation turns that were compressed.
        turns_dropped: Number of turns dropped from context.
        strategy_applied: The compression strategy used ("conservative", etc.).
        timestamp: Unix timestamp when this call was made.
        latency_ms: Wall-clock time from entering the wrapper to receiving a response.
    """

    original_input_tokens: int = 0
    optimised_input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0
    estimated_cost_saved_usd: float = 0.0
    tool_outputs_distilled: int = 0
    turns_summarised: int = 0
    turns_dropped: int = 0
    strategy_applied: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    @property
    def saved_pct(self) -> float:
        """Percentage of input tokens saved (0–100)."""
        if self.original_input_tokens == 0:
            return 0.0
        return (
            (self.original_input_tokens - self.optimised_input_tokens)
            / self.original_input_tokens
            * 100
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "original_input_tokens": self.original_input_tokens,
            "optimised_input_tokens": self.optimised_input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "estimated_cost_saved_usd": round(self.estimated_cost_saved_usd, 6),
            "saved_pct": round(self.saved_pct, 1),
            "tool_outputs_distilled": self.tool_outputs_distilled,
            "turns_summarised": self.turns_summarised,
            "turns_dropped": self.turns_dropped,
            "strategy_applied": self.strategy_applied,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class SessionStats:
    """Cumulative statistics across all calls in an AgentLean session.

    Attributes:
        total_calls: Number of API calls made through the wrapper.
        original_input_tokens: Total input tokens before any optimisation.
        optimised_input_tokens: Total input tokens after optimisation.
        total_output_tokens: Total output tokens across all calls.
        total_cost_usd: Estimated total spend in USD.
        total_saved_usd: Estimated total savings in USD.
        tool_outputs_distilled: Total tool results that were truncated.
        turns_summarised: Total turns compressed with summaries.
        turns_dropped: Total turns dropped from context.
        call_history: Ordered list of per-call statistics.
    """

    total_calls: int = 0
    original_input_tokens: int = 0
    optimised_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_saved_usd: float = 0.0
    tool_outputs_distilled: int = 0
    turns_summarised: int = 0
    turns_dropped: int = 0
    call_history: list[CallStats] = field(default_factory=list)

    @property
    def saved_pct(self) -> float:
        """Percentage of input tokens saved across the session (0–100)."""
        if self.original_input_tokens == 0:
            return 0.0
        return (
            (self.original_input_tokens - self.optimised_input_tokens)
            / self.original_input_tokens
            * 100
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (optimised input + output)."""
        return self.optimised_input_tokens + self.total_output_tokens

    def record(self, call: CallStats) -> None:
        """Incorporate a completed call's stats into the session totals."""
        self.total_calls += 1
        self.original_input_tokens += call.original_input_tokens
        self.optimised_input_tokens += call.optimised_input_tokens
        self.total_output_tokens += call.output_tokens
        self.total_cost_usd += call.estimated_cost_usd
        self.total_saved_usd += call.estimated_cost_saved_usd
        self.tool_outputs_distilled += call.tool_outputs_distilled
        self.turns_summarised += call.turns_summarised
        self.turns_dropped += call.turns_dropped
        self.call_history.append(call)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "total_calls": self.total_calls,
            "original_input_tokens": self.original_input_tokens,
            "optimised_input_tokens": self.optimised_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "saved_pct": round(self.saved_pct, 1),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_saved_usd": round(self.total_saved_usd, 6),
            "tool_outputs_distilled": self.tool_outputs_distilled,
            "turns_summarised": self.turns_summarised,
            "turns_dropped": self.turns_dropped,
        }

    def __repr__(self) -> str:
        return (
            f"AgentLeanStats("
            f"original_tokens={self.original_input_tokens:,}, "
            f"optimised_tokens={self.optimised_input_tokens:,}, "
            f"saved_pct={self.saved_pct:.1f}, "
            f"estimated_cost_saved={self.total_saved_usd:.4f})"
        )
