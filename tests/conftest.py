"""Shared pytest fixtures for AgentLean tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def long_conversation():
    with open(FIXTURES_DIR / "long_conversation.json") as f:
        return json.load(f)


@pytest.fixture
def web_search_fixture():
    with open(FIXTURES_DIR / "web_search_output.json") as f:
        return json.load(f)


@pytest.fixture
def api_response_fixture():
    with open(FIXTURES_DIR / "api_response.json") as f:
        return json.load(f)


@pytest.fixture
def short_messages():
    return [
        {"role": "user", "content": "Hello, what is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "Thanks!"},
    ]


@pytest.fixture
def tool_result_messages():
    """Messages with a large tool result that should be distilled."""
    large_content = "word " * 600  # ~600 words, well over default 500-token limit
    return [
        {"role": "user", "content": "Search for Python frameworks"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "tool_use",
                    "id": "tu_001",
                    "name": "web_search",
                    "input": {"query": "Python frameworks"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_001",
                    "content": large_content,
                }
            ],
        },
        {"role": "assistant", "content": "Based on the search results, here are the top frameworks..."},
    ]


@pytest.fixture
def many_turn_messages():
    """16-turn conversation for sliding window tests."""
    messages = []
    for i in range(16):
        messages.append({"role": "user", "content": f"Question number {i + 1}: What is {i + 1} * 7?"})
        messages.append({"role": "assistant", "content": f"The answer to question {i + 1} is {(i + 1) * 7}."})
    return messages


@pytest.fixture
def default_model():
    return "claude-sonnet-4-6"
