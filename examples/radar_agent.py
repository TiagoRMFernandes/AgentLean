"""Radar research agent example.

A realistic multi-step research agent that uses AgentLean to keep costs low.
The agent iteratively searches, reads pages, and synthesises a report.

This example uses mock API responses so it can run without real API keys.
To use with a real Anthropic client, replace the mock setup with:
    client = anthropic.Anthropic()
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import anthropic

from agentlean import AgentLean, AgentLeanConfig
from agentlean.strategies.system import analyse_system_prompt

# ---------------------------------------------------------------------------
# System prompt (intentionally verbose for demonstration)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Radar, an expert research assistant specialised in technology analysis.
You are helpful, harmless, and honest.
You are helpful, harmless, and honest.
When asked to research a topic, you follow this structured process:
1. Decompose the question into sub-questions
2. Search for relevant information using the search tool
3. Read and analyse the results
4. Synthesise a comprehensive report

You should always:
- Use the search tool to find information
- Read web pages thoroughly
- Cite your sources
- Provide well-structured output

You should never:
- Make up facts
- Skip searching before answering
- Provide unstructured output
- Make up facts (this is very important — never fabricate information)

When formatting your response, please ensure you:
- Use markdown headers
- Include a summary section
- Provide numbered citations
- Use markdown headers (note: headers are important for readability)

Please note that your responses should be helpful and accurate.
It is important to note that you should always verify information.
As a helpful assistant, you strive to provide the best possible responses.
"""

# ---------------------------------------------------------------------------
# Mock tool implementations
# ---------------------------------------------------------------------------

MOCK_SEARCH_RESULTS = {
    "default": """<!DOCTYPE html>
<html>
<head><title>Research Results</title></head>
<body>
<nav><ul><li>Home</li><li>Search</li><li>About</li></ul></nav>
<header><div class="logo">TechResearch</div></header>
<aside>
  <div class="advertisement">Sponsored: Learn AI Online!</div>
  <div class="related">Related: Machine Learning Basics</div>
</aside>
<main>
  <article>
    <h1>The State of AI Agent Frameworks in 2024</h1>
    <p>AI agent frameworks have proliferated rapidly in 2024. The landscape includes
    LangChain, CrewAI, AutoGen, and custom solutions built on raw API calls.</p>
    <p>Key trends: multi-agent orchestration, tool use standardisation, cost optimisation,
    and the emergence of structured outputs for reliability.</p>
    <h2>Framework Comparison</h2>
    <p>LangChain offers the broadest ecosystem. CrewAI specialises in role-based agents.
    AutoGen focuses on conversational multi-agent systems. Raw API usage remains popular
    for maximum control and minimal overhead.</p>
    <h2>Cost Analysis</h2>
    <p>Token costs are the primary concern for production agent deployments. Research agents
    commonly send 50,000-200,000 input tokens per run, making cost optimisation critical.</p>
  </article>
</main>
<footer>Privacy Policy | Terms | Cookie Notice <button>Accept Cookies</button></footer>
</html>""",
}


def fake_search(query: str) -> str:
    """Return a simulated web search result."""
    return MOCK_SEARCH_RESULTS["default"]


def fake_read_page(url: str) -> str:
    """Return a simulated web page content."""
    return (
        f"Page content for {url}: "
        + ("This page contains detailed information about the topic. " * 80)
    )


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": "Read the full content of a web page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read."}
            },
            "required": ["url"],
        },
    },
]


# ---------------------------------------------------------------------------
# Mock client setup
# ---------------------------------------------------------------------------

_step = 0
_queries = [
    "AI agent frameworks 2024 comparison",
    "token cost optimisation LLM agents",
    "multi-agent orchestration best practices",
    "LangChain CrewAI AutoGen comparison",
    "production AI agent cost management",
]


class _TextBlock:
    type = "text"
    def __init__(self, text: str) -> None:
        self.text = text

class _ToolUseBlock:
    type = "tool_use"
    def __init__(self, bid: str, name: str, input: dict) -> None:
        self.id = bid
        self.name = name
        self.input = input


def make_mock_step_response(step: int) -> MagicMock:
    resp = MagicMock()
    resp.usage = MagicMock(input_tokens=3000 + step * 2000, output_tokens=500)
    resp.stop_reason = "tool_use" if step < 4 else "end_turn"

    if step < len(_queries):
        resp.content = [
            _TextBlock(f"Step {step + 1}: Let me search for more information about this topic."),
            _ToolUseBlock(
                bid=f"tu_{step:03d}",
                name="web_search",
                input={"query": _queries[step % len(_queries)]},
            ),
        ]
    else:
        resp.content = [
            _TextBlock(
                "## Research Report: AI Agent Frameworks\n\n"
                "Based on my research, here is a comprehensive analysis...\n\n"
                "### Key Findings\n"
                "1. LangChain leads in ecosystem breadth\n"
                "2. Cost is the #1 production concern\n"
                "3. Token optimisation can reduce costs by 40-80%\n"
            )
        ]
    return resp


def run_radar_agent() -> None:
    # ----- Analyse system prompt first -----
    print("=" * 60)
    print("SYSTEM PROMPT ANALYSIS")
    print("=" * 60)
    analysis = analyse_system_prompt(SYSTEM_PROMPT)
    print(analysis)
    for s in analysis.suggestions:
        print(f"  [{s.severity.upper():6s}] {s.description[:80]}...")
    print()

    # ----- Set up mock client -----
    global _step
    _step = 0

    client = MagicMock(spec=anthropic.Anthropic)
    client.__class__ = anthropic.Anthropic

    def mock_create(**kwargs: Any) -> MagicMock:
        global _step
        resp = make_mock_step_response(_step)
        _step += 1
        return resp

    client.messages.create.side_effect = mock_create

    # ----- Configure AgentLean -----
    config = AgentLeanConfig(
        strategy="balanced",
        context_window_turns=4,
        max_tool_output_tokens=400,
        preserve_first_message=True,
        on_call_complete=lambda stats: None,  # silent callback
    )

    lean = AgentLean(client, provider="anthropic", config=config)

    # ----- Run the agent loop -----
    print("=" * 60)
    print("RADAR RESEARCH AGENT RUN")
    print("=" * 60)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Research the current landscape of AI agent frameworks in 2024. "
                "Focus on: (1) the most popular frameworks, (2) their cost profiles, "
                "and (3) best practices for production deployments."
            ),
        }
    ]

    max_steps = 6
    for step in range(max_steps):
        response = lean.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )

        call_stats = lean.stats.call_history[-1]
        print(
            f"\nStep {step + 1}: strategy={call_stats.strategy_applied}, "
            f"original={call_stats.original_input_tokens:,}, "
            f"optimised={call_stats.optimised_input_tokens:,} "
            f"({call_stats.saved_pct:.0f}% saved)"
        )

        # Process tool calls
        assistant_content = response.content
        has_tool_use = any(
            getattr(block, "type", "") == "tool_use" for block in assistant_content
        )

        messages.append({"role": "assistant", "content": [
            {"type": getattr(b, "type", "text"),
             "text": getattr(b, "text", ""),
             **({"id": b.id, "name": b.name, "input": b.input}
                if getattr(b, "type", "") == "tool_use" else {})}
            for b in assistant_content
        ]})

        if has_tool_use:
            # Execute tools and add results
            tool_results = []
            for block in assistant_content:
                if getattr(block, "type", "") == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    print(f"         → Tool call: {tool_name}({json.dumps(tool_input)[:60]})")

                    if tool_name == "web_search":
                        result = fake_search(tool_input.get("query", ""))
                    elif tool_name == "read_page":
                        result = fake_read_page(tool_input.get("url", ""))
                    else:
                        result = "Tool not found."

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Final response — agent is done
            final_text = next(
                (getattr(b, "text", "") for b in assistant_content
                 if getattr(b, "type", "") == "text"),
                "",
            )
            print(f"\n{'=' * 60}")
            print("FINAL REPORT (excerpt):")
            print("=" * 60)
            print(final_text[:500])
            break

    # ----- Print final stats -----
    print(f"\n{'=' * 60}")
    print("AGENTLEAN SESSION STATS")
    print("=" * 60)
    stats = lean.stats.to_dict()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_radar_agent()
