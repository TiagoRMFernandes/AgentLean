"""Tests for the OpenAI provider wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentlean import AgentLean, AgentLeanConfig
from agentlean.providers.openai import _split_system


class MockOpenAIUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockOpenAIResponse:
    def __init__(self, content="Hello!", prompt_tokens=100, completion_tokens=50):
        choice = MagicMock()
        choice.message.content = content
        self.choices = [choice]
        self.usage = MockOpenAIUsage(prompt_tokens, completion_tokens)
        self.model = "gpt-4o"


def make_mock_openai_client(response=None):
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.__class__.__name__ = "OpenAI"
    client.__class__.__module__ = "openai"
    if response is None:
        response = MockOpenAIResponse()
    client.chat.completions.create.return_value = response
    return client


class TestOpenAIProvider:
    def test_completions_create_forwarded(self):
        client = make_mock_openai_client()
        lean = AgentLean(client, provider="openai")
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]

        lean.chat.completions.create(model="gpt-4o", max_tokens=100, messages=messages)

        assert client.chat.completions.create.called

    def test_stats_recorded(self):
        client = make_mock_openai_client(
            MockOpenAIResponse(prompt_tokens=200, completion_tokens=80)
        )
        lean = AgentLean(client, provider="openai")

        lean.chat.completions.create(
            model="gpt-4o",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert lean.stats.total_calls == 1
        assert lean.stats.total_output_tokens == 80

    def test_system_message_preserved(self):
        client = make_mock_openai_client()
        lean = AgentLean(client, provider="openai")

        lean.chat.completions.create(
            model="gpt-4o",
            max_tokens=100,
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"

    def test_extra_kwargs_pass_through(self):
        client = make_mock_openai_client()
        lean = AgentLean(client, provider="openai")

        lean.chat.completions.create(
            model="gpt-4o",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            n=2,
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("n") == 2

    def test_openai_tool_results_distilled(self):
        client = make_mock_openai_client()
        lean = AgentLean(
            client,
            provider="openai",
            config=AgentLeanConfig(strategy="conservative", max_tool_output_tokens=60),
        )
        large = "data " * 200
        messages = [
            {"role": "user", "content": "Search something"},
            {"role": "tool", "tool_call_id": "c1", "content": large},
        ]

        lean.chat.completions.create(model="gpt-4o", max_tokens=100, messages=messages)

        assert lean.stats.tool_outputs_distilled >= 1


class TestSplitSystem:
    def test_splits_system_message(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys_msg, body = _split_system(messages)
        assert sys_msg is not None
        assert sys_msg["role"] == "system"
        assert len(body) == 1
        assert body[0]["role"] == "user"

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "Hi"}]
        sys_msg, body = _split_system(messages)
        assert sys_msg is None
        assert len(body) == 1

    def test_empty_messages(self):
        sys_msg, body = _split_system([])
        assert sys_msg is None
        assert body == []
