"""Tests for the Anthropic provider wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentlean import AgentLean, AgentLeanConfig
from agentlean.providers.anthropic import count_anthropic_tokens


class MockAnthropicUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicResponse:
    def __init__(self, text="Hello!", input_tokens=100, output_tokens=50):
        self.content = [MagicMock(text=text)]
        self.usage = MockAnthropicUsage(input_tokens, output_tokens)
        self.stop_reason = "end_turn"
        self.model = "claude-sonnet-4-6"


def make_mock_anthropic_client(response=None):
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.__class__.__name__ = "Anthropic"
    client.__class__.__module__ = "anthropic"
    if response is None:
        response = MockAnthropicResponse()
    client.messages.create.return_value = response
    return client


class TestAnthropicProvider:
    def test_messages_create_forwarded(self):
        client = make_mock_anthropic_client()
        lean = AgentLean(client, provider="anthropic")
        messages = [{"role": "user", "content": "Hello"}]

        lean.messages.create(model="claude-sonnet-4-6", max_tokens=100, messages=messages)

        assert client.messages.create.called

    def test_stats_recorded(self):
        client = make_mock_anthropic_client(MockAnthropicResponse(input_tokens=200, output_tokens=80))
        lean = AgentLean(client, provider="anthropic")
        messages = [{"role": "user", "content": "Hello"}]

        lean.messages.create(model="claude-sonnet-4-6", max_tokens=100, messages=messages)

        assert lean.stats.total_calls == 1
        assert lean.stats.total_output_tokens == 80

    def test_extra_kwargs_passed_through(self):
        client = make_mock_anthropic_client()
        lean = AgentLean(client, provider="anthropic")

        lean.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            stop_sequences=["STOP"],
        )

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("stop_sequences") == ["STOP"]

    def test_system_prompt_passed_through(self):
        client = make_mock_anthropic_client()
        lean = AgentLean(client, provider="anthropic")

        lean.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        call_kwargs = client.messages.create.call_args.kwargs
        assert "system" in call_kwargs

    def test_optimisation_on_large_input(self, tool_result_messages):
        client = make_mock_anthropic_client()
        lean = AgentLean(
            client,
            provider="anthropic",
            config=AgentLeanConfig(strategy="balanced", max_tool_output_tokens=50),
        )

        lean.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=tool_result_messages,
        )

        assert lean.stats.tool_outputs_distilled >= 1

    def test_optimisation_error_falls_back_to_original(self, short_messages):
        """If optimisation crashes, original messages must still be sent."""
        client = make_mock_anthropic_client()
        lean = AgentLean(client, provider="anthropic")

        with patch.object(lean, "_run_pipeline", side_effect=RuntimeError("boom")):
            lean.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=short_messages,
            )

        # Should still have called the client
        assert client.messages.create.called
        # Original messages passed through
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == short_messages

    def test_on_call_complete_callback(self):
        client = make_mock_anthropic_client()
        received = []
        config = AgentLeanConfig(on_call_complete=received.append)
        lean = AgentLean(client, provider="anthropic", config=config)

        lean.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert len(received) == 1
        assert received[0].model == "claude-sonnet-4-6"


class TestCountAnthropicTokens:
    def test_counts_messages(self):
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        count = count_anthropic_tokens(messages, None, "claude-sonnet-4-6")
        assert count > 0

    def test_adds_system_tokens(self):
        messages = [{"role": "user", "content": "Hi"}]
        without_system = count_anthropic_tokens(messages, None, "claude-sonnet-4-6")
        with_system = count_anthropic_tokens(
            messages, "You are a very verbose assistant who always...", "claude-sonnet-4-6"
        )
        assert with_system > without_system
