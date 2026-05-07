"""Tests for the budget manager."""

from __future__ import annotations

import pytest

from agentlean.exceptions import BudgetExhaustedError
from agentlean.strategies.budget import BudgetManager


class TestBudgetManager:
    def test_no_budget_no_restriction(self):
        mgr = BudgetManager()
        assert not mgr.has_budget
        mgr.record_call(10_000, 1_000, "claude-sonnet-4-6")
        mgr.check_before_call()  # should not raise

    def test_token_budget_tracking(self):
        mgr = BudgetManager(budget_tokens=10_000)
        mgr.record_call(4_000, 1_000, "claude-sonnet-4-6")
        assert mgr.state.used_tokens == 5_000
        assert mgr.state.tokens_remaining == 5_000

    def test_usd_budget_tracking(self):
        mgr = BudgetManager(budget_usd=1.0)
        # claude-sonnet-4-6: $3/M input, $15/M output
        mgr.record_call(100_000, 10_000, "claude-sonnet-4-6")
        assert mgr.state.used_usd > 0
        assert mgr.state.usd_remaining < 1.0

    def test_hard_stop_on_token_exhaustion(self):
        mgr = BudgetManager(budget_tokens=1_000, hard_stop=True)
        mgr.record_call(500, 200, "claude-sonnet-4-6")  # 700 total — within budget
        mgr.check_before_call()  # should not raise yet
        mgr.record_call(200, 200, "claude-sonnet-4-6")  # 1100 total — over budget
        with pytest.raises(BudgetExhaustedError) as exc_info:
            mgr.check_before_call()  # should raise now
        assert exc_info.value.budget_type == "tokens"

    def test_no_hard_stop_when_disabled(self):
        mgr = BudgetManager(budget_tokens=100, hard_stop=False)
        mgr.record_call(200, 100, "claude-sonnet-4-6")
        mgr.check_before_call()  # should not raise

    def test_strategy_escalation_at_70_pct(self):
        mgr = BudgetManager(
            budget_tokens=10_000,
            base_strategy="conservative",
            warn_at_pct=0.70,
        )
        mgr.record_call(7_500, 0, "claude-sonnet-4-6")  # 75%
        assert mgr.effective_strategy() == "balanced"

    def test_strategy_escalation_at_90_pct(self):
        mgr = BudgetManager(
            budget_tokens=10_000,
            base_strategy="conservative",
            warn_at_pct=0.70,
        )
        mgr.record_call(9_500, 0, "claude-sonnet-4-6")  # 95%
        assert mgr.effective_strategy() == "aggressive"

    def test_no_escalation_below_threshold(self):
        mgr = BudgetManager(
            budget_tokens=10_000,
            base_strategy="conservative",
            warn_at_pct=0.70,
        )
        mgr.record_call(5_000, 0, "claude-sonnet-4-6")  # 50%
        assert mgr.effective_strategy() == "conservative"

    def test_budget_exhausted_error_message(self):
        err = BudgetExhaustedError("tokens", 100, 200)
        assert "200" in str(err)
        assert "100" in str(err)
        assert err.budget_type == "tokens"
        assert err.limit == 100
        assert err.used == 200

    def test_warnings_emitted_once(self, caplog):
        import logging
        mgr = BudgetManager(budget_tokens=10_000, warn_at_pct=0.70, hard_stop=False)
        with caplog.at_level(logging.WARNING):
            mgr.record_call(7_500, 0, "claude-sonnet-4-6")  # triggers warn_70
            mgr.record_call(1_000, 0, "claude-sonnet-4-6")  # should NOT re-trigger warn_70
        warnings = [r for r in caplog.records if "70%" in r.message or "75%" in r.message]
        assert len(warnings) == 1
