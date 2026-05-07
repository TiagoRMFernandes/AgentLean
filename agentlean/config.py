"""Configuration dataclass for AgentLean."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .exceptions import ConfigurationError

Strategy = Literal["conservative", "balanced", "aggressive"]


@dataclass
class AgentLeanConfig:
    """Configuration for the AgentLean middleware.

    Args:
        strategy: Compression preset. "conservative" applies only truncation;
            "balanced" adds sliding window and light summarisation;
            "aggressive" maximises compression at some quality risk.
        context_window_turns: Number of recent conversation turns kept verbatim.
            Older turns are progressively compressed.
        summarise_turns_up_to: Turns beyond context_window_turns but within this
            range are summarised. Turns beyond this number are dropped.
        max_tool_output_tokens: Hard cap (in tokens) applied to each tool result
            before it enters context.
        preserve_system_prompt: When True, the system prompt is never modified.
        preserve_first_message: When True, the first user message (original task)
            is always kept verbatim.
        budget_usd: Optional maximum spend (USD) per AgentLean session.
        budget_tokens: Optional maximum total input+output tokens per session.
        summarisation_model: The model used to generate turn summaries (should be
            a cheap, fast model).
        warn_at_budget_pct: Emit a warning when budget usage reaches this fraction.
        hard_stop_at_budget: Raise BudgetExhaustedError when budget is exhausted.
            If False, AgentLean just warns and continues.
        on_call_complete: Optional callback invoked after each API call with the
            per-call CallStats. Useful for logging or monitoring integration.
    """

    strategy: Strategy = "balanced"
    context_window_turns: int = 5
    summarise_turns_up_to: int = 15
    max_tool_output_tokens: int = 500
    preserve_system_prompt: bool = True
    preserve_first_message: bool = True
    budget_usd: float | None = None
    budget_tokens: int | None = None
    summarisation_model: str = "claude-haiku-4-5-20251001"
    warn_at_budget_pct: float = 0.70
    hard_stop_at_budget: bool = True
    on_call_complete: object = field(default=None, repr=False)

    # --- strategy presets ------------------------------------------------- #

    @classmethod
    def conservative(cls) -> "AgentLeanConfig":
        """Only truncate tool outputs. ~20-30% token savings."""
        return cls(
            strategy="conservative",
            context_window_turns=20,
            summarise_turns_up_to=50,
            max_tool_output_tokens=800,
        )

    @classmethod
    def balanced(cls) -> "AgentLeanConfig":
        """Truncate + sliding window + light summarisation. ~40-60% savings."""
        return cls(strategy="balanced")

    @classmethod
    def aggressive(cls) -> "AgentLeanConfig":
        """Heavy compression, shorter window. ~60-80% savings, some quality risk."""
        return cls(
            strategy="aggressive",
            context_window_turns=3,
            summarise_turns_up_to=8,
            max_tool_output_tokens=250,
        )

    def __post_init__(self) -> None:
        if self.strategy not in ("conservative", "balanced", "aggressive"):
            raise ConfigurationError(
                f"strategy must be one of 'conservative', 'balanced', 'aggressive'; "
                f"got '{self.strategy}'"
            )
        if self.context_window_turns < 1:
            raise ConfigurationError("context_window_turns must be >= 1")
        if self.summarise_turns_up_to < self.context_window_turns:
            raise ConfigurationError(
                "summarise_turns_up_to must be >= context_window_turns"
            )
        if self.max_tool_output_tokens < 50:
            raise ConfigurationError("max_tool_output_tokens must be >= 50")
        if self.budget_usd is not None and self.budget_usd <= 0:
            raise ConfigurationError("budget_usd must be > 0")
        if self.budget_tokens is not None and self.budget_tokens <= 0:
            raise ConfigurationError("budget_tokens must be > 0")
        if not (0 < self.warn_at_budget_pct < 1):
            raise ConfigurationError("warn_at_budget_pct must be between 0 and 1")
