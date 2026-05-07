"""Token and USD budget management.

The BudgetManager tracks cumulative spend across a session and:
  - Emits warnings at configurable thresholds (default 70% and 90%).
  - Automatically escalates the compression strategy as the budget runs low.
  - Optionally raises BudgetExhaustedError when 100% is reached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from ..exceptions import BudgetExhaustedError
from ..pricing import estimate_cost

logger = logging.getLogger(__name__)

Strategy = Literal["conservative", "balanced", "aggressive"]


@dataclass
class BudgetState:
    """Live state of a budget within a session."""

    budget_tokens: int | None = None
    budget_usd: float | None = None
    used_tokens: int = 0
    used_usd: float = 0.0
    warnings_emitted: set[str] = field(default_factory=set)

    @property
    def token_fraction(self) -> float:
        if self.budget_tokens is None or self.budget_tokens == 0:
            return 0.0
        return self.used_tokens / self.budget_tokens

    @property
    def usd_fraction(self) -> float:
        if self.budget_usd is None or self.budget_usd == 0:
            return 0.0
        return self.used_usd / self.budget_usd

    @property
    def max_fraction(self) -> float:
        """The more constrained of the two budget fractions."""
        return max(self.token_fraction, self.usd_fraction)

    @property
    def tokens_remaining(self) -> int | None:
        if self.budget_tokens is None:
            return None
        return max(0, self.budget_tokens - self.used_tokens)

    @property
    def usd_remaining(self) -> float | None:
        if self.budget_usd is None:
            return None
        return max(0.0, self.budget_usd - self.used_usd)


class BudgetManager:
    """Manages per-session token and USD budgets.

    Args:
        budget_tokens: Maximum input+output tokens for the session.
        budget_usd: Maximum USD spend for the session.
        base_strategy: The strategy configured by the user.
        warn_at_pct: Fraction (0–1) at which to emit the first warning.
        hard_stop: If True, raise BudgetExhaustedError when exhausted.
    """

    def __init__(
        self,
        *,
        budget_tokens: int | None = None,
        budget_usd: float | None = None,
        base_strategy: Strategy = "balanced",
        warn_at_pct: float = 0.70,
        hard_stop: bool = True,
    ) -> None:
        self._state = BudgetState(
            budget_tokens=budget_tokens,
            budget_usd=budget_usd,
        )
        self._base_strategy = base_strategy
        self._warn_at_pct = warn_at_pct
        self._hard_stop = hard_stop

    @property
    def has_budget(self) -> bool:
        return self._state.budget_tokens is not None or self._state.budget_usd is not None

    @property
    def state(self) -> BudgetState:
        return self._state

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        """Record token and cost usage for one completed API call.

        Args:
            input_tokens: Optimised input token count.
            output_tokens: Tokens in the model response.
            model: Model ID used for cost estimation.
        """
        self._state.used_tokens += input_tokens + output_tokens
        self._state.used_usd += estimate_cost(input_tokens, output_tokens, model)
        self._check_warnings()

    def check_before_call(self) -> None:
        """Raise BudgetExhaustedError if the budget is already exhausted.

        Only raises when hard_stop=True. Should be called before each API call.
        """
        if not self.has_budget or not self._hard_stop:
            return
        if self._state.max_fraction >= 1.0:
            self._raise_exhausted()

    def effective_strategy(self) -> Strategy:
        """Return the strategy to use for the current call.

        As the budget is consumed, the strategy automatically escalates:
          - 0–70%  → base strategy
          - 70–90% → at least "balanced"
          - 90%+   → "aggressive"
        """
        if not self.has_budget:
            return self._base_strategy

        fraction = self._state.max_fraction
        if fraction >= 0.90:
            return "aggressive"
        if fraction >= self._warn_at_pct:
            return _escalate(self._base_strategy, "balanced")
        return self._base_strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_warnings(self) -> None:
        fraction = self._state.max_fraction
        pct = int(fraction * 100)

        warn_thresholds = [
            (self._warn_at_pct, f"warn_{int(self._warn_at_pct * 100)}"),
            (0.90, "warn_90"),
        ]
        for threshold, key in warn_thresholds:
            if fraction >= threshold and key not in self._state.warnings_emitted:
                self._state.warnings_emitted.add(key)
                threshold_pct = int(threshold * 100)
                logger.warning(
                    "AgentLean budget warning: %d%% of %s budget used "
                    "(threshold: %d%%). Compression will be escalated automatically.",
                    min(pct, 100),
                    self._budget_type_label(),
                    threshold_pct,
                )

        # Only warn here — do not raise. Raising happens in check_before_call().

    def _raise_exhausted(self) -> None:
        if self._state.budget_tokens is not None and self._state.token_fraction >= 1.0:
            raise BudgetExhaustedError(
                "tokens",
                self._state.budget_tokens,
                self._state.used_tokens,
            )
        if self._state.budget_usd is not None and self._state.usd_fraction >= 1.0:
            raise BudgetExhaustedError(
                "usd",
                self._state.budget_usd,
                self._state.used_usd,
            )

    def _budget_type_label(self) -> str:
        labels = []
        if self._state.budget_tokens is not None:
            labels.append("token")
        if self._state.budget_usd is not None:
            labels.append("USD")
        return "/".join(labels) or "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STRATEGY_LEVELS: dict[Strategy, int] = {
    "conservative": 0,
    "balanced": 1,
    "aggressive": 2,
}
_LEVEL_TO_STRATEGY: dict[int, Strategy] = {v: k for k, v in _STRATEGY_LEVELS.items()}


def _escalate(current: Strategy, minimum: Strategy) -> Strategy:
    """Return *current* if it's already at or above *minimum*, else return *minimum*."""
    level = max(_STRATEGY_LEVELS[current], _STRATEGY_LEVELS[minimum])
    return _LEVEL_TO_STRATEGY[level]
