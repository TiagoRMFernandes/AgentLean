"""Budget cap example.

Demonstrates how to set a USD or token budget and observe how AgentLean
automatically escalates compression as the budget runs low.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import anthropic

from agentlean import AgentLean, AgentLeanConfig
from agentlean.exceptions import BudgetExhaustedError

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def make_mock_response(input_tokens: int = 1000, output_tokens: int = 300) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text="Response text.")]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    resp.stop_reason = "end_turn"
    return resp


def run_budget_example() -> None:
    # ----- Token budget -----
    print("=== Token Budget Example ===\n")

    client = MagicMock(spec=anthropic.Anthropic)
    client.__class__ = anthropic.Anthropic
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate increasing input token counts as context grows
        return make_mock_response(input_tokens=2000 + call_count * 1500, output_tokens=400)

    client.messages.create.side_effect = side_effect

    lean = AgentLean(
        client,
        provider="anthropic",
        budget_tokens=30_000,  # 30K token cap for the session
        strategy="conservative",  # start conservative, will escalate automatically
    )

    messages = [{"role": "user", "content": "Begin the research task."}]

    for i in range(12):
        try:
            response = lean.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content[0].text})
            messages.append({"role": "user", "content": f"Continue with step {i + 2}."})

            budget_state = lean._budget.state
            strategy = lean._budget.effective_strategy()
            print(
                f"Call {i + 1:2d}: {lean.stats.call_history[-1].strategy_applied:12s} | "
                f"tokens used: {budget_state.used_tokens:6,} / 30,000 "
                f"({budget_state.token_fraction * 100:.0f}%)"
            )

        except BudgetExhaustedError as e:
            print(f"\nBudget exhausted at call {i + 1}: {e}")
            break

    print(f"\nFinal stats: {lean.stats}")

    # ----- USD budget -----
    print("\n\n=== USD Budget Example ===\n")

    client2 = MagicMock(spec=anthropic.Anthropic)
    client2.__class__ = anthropic.Anthropic
    client2.messages.create.return_value = make_mock_response(
        input_tokens=5_000, output_tokens=800
    )

    lean2 = AgentLean(
        client2,
        provider="anthropic",
        budget_usd=0.05,  # $0.05 cap
        strategy="balanced",
    )

    for i in range(20):
        try:
            lean2.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": f"Question {i + 1}"}],
            )
            state = lean2._budget.state
            print(
                f"Call {i + 1:2d}: ${state.used_usd:.4f} / $0.0500 used "
                f"({state.usd_fraction * 100:.1f}%)"
            )
        except BudgetExhaustedError as e:
            print(f"\nUSD budget exhausted: {e}")
            break


if __name__ == "__main__":
    run_budget_example()
