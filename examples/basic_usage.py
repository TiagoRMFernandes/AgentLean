"""Basic AgentLean usage example.

Shows how to wrap an Anthropic client and inspect savings after a simulated
multi-step agent run. No real API calls are made — a mock client is used.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import anthropic

from agentlean import AgentLean, AgentLeanConfig


def run_example() -> None:
    # ----- 1. Set up a real (or mock) client -----
    # In production, use: client = anthropic.Anthropic()
    client = MagicMock(spec=anthropic.Anthropic)
    client.__class__ = anthropic.Anthropic  # makes provider detection work

    call_number = 0

    def mock_create(**kwargs):
        nonlocal call_number
        call_number += 1
        # Simulate: without AgentLean the context would grow ~8k tokens/step.
        # With optimisation, the actual payload sent stays roughly stable after the
        # window kicks in. Here we approximate the optimised size based on message count.
        messages_sent = kwargs.get("messages", [])
        simulated_input = min(500 + len(str(messages_sent)) // 4, 8_000)
        resp = MagicMock()
        resp.content = [MagicMock(text="Here is what I found based on the research.")]
        resp.usage = MagicMock(input_tokens=simulated_input, output_tokens=450)
        resp.stop_reason = "end_turn"
        return resp

    client.messages.create.side_effect = mock_create

    # ----- 2. Wrap with AgentLean -----
    config = AgentLeanConfig(
        strategy="balanced",
        max_tool_output_tokens=500,
        context_window_turns=5,
    )
    lean = AgentLean(client, provider="anthropic", config=config)

    # ----- 3. Build a growing conversation (simulates a research agent) -----
    system = (
        "You are a research assistant. Use the provided tools to answer questions thoroughly. "
        "Always cite your sources and provide structured, well-organised responses."
    )

    large_web_result = (
        "<!DOCTYPE html><html><head><title>Python Frameworks</title></head><body>"
        "<nav><ul><li>Home</li><li>Blog</li></ul></nav>"
        "<main><article><h1>Top Python Web Frameworks in 2024</h1>"
        + ("<p>Python has many excellent web frameworks. " * 50)
        + "</article></main>"
        "<footer>Copyright 2024. Privacy Policy. Terms of Service.</footer></body></html>"
    )

    messages = [
        {
            "role": "user",
            "content": "Research the top Python web frameworks and compare their performance.",
        }
    ]

    # Simulate 8 agent steps with growing context
    for step in range(8):
        # Simulate the assistant calling a tool
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Step {step + 1}: Let me search for more information."},
                    {
                        "type": "tool_use",
                        "id": f"tu_{step:03d}",
                        "name": "web_search",
                        "input": {"query": f"python web framework benchmark step {step + 1}"},
                    },
                ],
            }
        )
        # Simulate the tool result (large HTML content)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tu_{step:03d}",
                        "content": large_web_result,
                    }
                ],
            }
        )

        # Make the API call through AgentLean
        response = lean.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content[0].text})

        step_stats = lean.stats.call_history[-1]
        print(
            f"Step {step + 1:2d}: "
            f"original={step_stats.original_input_tokens:6,} → "
            f"optimised={step_stats.optimised_input_tokens:6,} tokens "
            f"({step_stats.saved_pct:.0f}% saved, "
            f"tool results distilled: {step_stats.tool_outputs_distilled})"
        )

    # ----- 4. Print session summary -----
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(json.dumps(lean.stats.to_dict(), indent=2))
    print(f"\nOverall: {lean.stats}")


if __name__ == "__main__":
    run_example()
