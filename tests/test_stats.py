"""Tests for statistics tracking."""

from __future__ import annotations

import pytest

from agentlean.stats import CallStats, SessionStats


class TestCallStats:
    def test_saved_pct_zero_when_no_original(self):
        stats = CallStats()
        assert stats.saved_pct == 0.0

    def test_saved_pct_calculation(self):
        stats = CallStats(original_input_tokens=1000, optimised_input_tokens=400)
        assert stats.saved_pct == 60.0

    def test_to_dict_contains_expected_keys(self):
        stats = CallStats(
            original_input_tokens=1000,
            optimised_input_tokens=400,
            output_tokens=200,
            model="claude-sonnet-4-6",
            estimated_cost_usd=0.001,
            estimated_cost_saved_usd=0.0018,
        )
        d = stats.to_dict()
        assert "original_input_tokens" in d
        assert "optimised_input_tokens" in d
        assert "saved_pct" in d
        assert d["saved_pct"] == 60.0

    def test_to_dict_rounding(self):
        stats = CallStats(
            estimated_cost_usd=0.0012345678,
            estimated_cost_saved_usd=0.0009876543,
        )
        d = stats.to_dict()
        # Should be rounded to 6 decimal places
        assert d["estimated_cost_usd"] == round(0.0012345678, 6)


class TestSessionStats:
    def test_initial_state(self):
        session = SessionStats()
        assert session.total_calls == 0
        assert session.saved_pct == 0.0

    def test_record_accumulates(self):
        session = SessionStats()
        call1 = CallStats(original_input_tokens=1000, optimised_input_tokens=600, output_tokens=100)
        call2 = CallStats(original_input_tokens=2000, optimised_input_tokens=1200, output_tokens=200)
        session.record(call1)
        session.record(call2)
        assert session.total_calls == 2
        assert session.original_input_tokens == 3000
        assert session.optimised_input_tokens == 1800
        assert session.total_output_tokens == 300

    def test_session_saved_pct(self):
        session = SessionStats()
        session.record(CallStats(original_input_tokens=1000, optimised_input_tokens=400))
        assert session.saved_pct == 60.0

    def test_total_tokens(self):
        session = SessionStats()
        session.record(CallStats(optimised_input_tokens=500, output_tokens=200))
        assert session.total_tokens == 700

    def test_to_dict(self):
        session = SessionStats()
        session.record(CallStats(original_input_tokens=100, optimised_input_tokens=50))
        d = session.to_dict()
        assert d["total_calls"] == 1
        assert d["saved_pct"] == 50.0

    def test_repr_format(self):
        session = SessionStats()
        session.record(
            CallStats(
                original_input_tokens=84_230,
                optimised_input_tokens=31_450,
                estimated_cost_saved_usd=0.038,
            )
        )
        r = repr(session)
        assert "84,230" in r
        assert "31,450" in r
        assert "62.7" in r

    def test_call_history_maintained(self):
        session = SessionStats()
        for i in range(5):
            session.record(CallStats(original_input_tokens=100 * (i + 1)))
        assert len(session.call_history) == 5
