"""Tests for the sliding context window strategy."""

from __future__ import annotations

import pytest

from agentlean.strategies.window import apply_sliding_window, _segment_into_turns


MODEL = "claude-sonnet-4-6"


class TestSegmentIntoTurns:
    def test_basic_segmentation(self, short_messages):
        turns = _segment_into_turns(short_messages)
        # "Hello" / "4" is turn 1, "Thanks!" is turn 2
        assert len(turns) == 2

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        turns = _segment_into_turns(messages)
        assert len(turns) == 1
        assert turns[0].messages[0]["role"] == "user"

    def test_tool_messages_grouped_with_assistant(self):
        messages = [
            {"role": "user", "content": "Search something"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "results"}]},
            {"role": "assistant", "content": "Here are the results"},
            {"role": "user", "content": "Thanks"},
        ]
        turns = _segment_into_turns(messages)
        # Should be 3 turns: [user+assistant+tool_result_user], [assistant], [user]
        # Actually: turn 1 = user+assistant, turn 2 = user(tool_result)+assistant, turn 3 = user
        assert len(turns) == 3

    def test_empty_messages(self):
        turns = _segment_into_turns([])
        assert turns == []


class TestApplySlidingWindow:
    def test_short_conversation_unchanged(self, short_messages):
        result, summarised, dropped = apply_sliding_window(
            short_messages,
            context_window_turns=5,
            summarise_turns_up_to=15,
            model=MODEL,
        )
        assert summarised == 0
        assert dropped == 0
        assert len(result) == len(short_messages)

    def test_long_conversation_compressed(self, many_turn_messages):
        # 16 turns, keep last 5, summarise up to 15, drop beyond
        result, summarised, dropped = apply_sliding_window(
            many_turn_messages,
            context_window_turns=5,
            summarise_turns_up_to=15,
            model=MODEL,
        )
        assert summarised > 0 or dropped > 0
        assert len(result) < len(many_turn_messages)

    def test_recent_turns_preserved_verbatim(self, many_turn_messages):
        result, _, _ = apply_sliding_window(
            many_turn_messages,
            context_window_turns=3,
            summarise_turns_up_to=10,
            model=MODEL,
        )
        # The last few messages should appear unchanged
        last_user = many_turn_messages[-2]["content"]
        last_assistant = many_turn_messages[-1]["content"]
        result_contents = [m.get("content", "") for m in result]
        assert last_user in result_contents
        assert last_assistant in result_contents

    def test_first_message_preserved(self, many_turn_messages):
        first_content = many_turn_messages[0]["content"]
        result, _, _ = apply_sliding_window(
            many_turn_messages,
            context_window_turns=2,
            summarise_turns_up_to=5,
            model=MODEL,
            preserve_first_message=True,
        )
        result_contents = [m.get("content", "") for m in result]
        assert first_content in result_contents

    def test_system_message_preserved(self, many_turn_messages):
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant."},
            *many_turn_messages,
        ]
        result, _, _ = apply_sliding_window(
            messages_with_system,
            context_window_turns=3,
            summarise_turns_up_to=8,
            model=MODEL,
        )
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

    def test_summary_messages_are_user_role(self, many_turn_messages):
        result, summarised, _ = apply_sliding_window(
            many_turn_messages,
            context_window_turns=3,
            summarise_turns_up_to=10,
            model=MODEL,
        )
        summary_msgs = [m for m in result if "[Summary of earlier turn" in str(m.get("content", ""))]
        assert len(summary_msgs) == summarised

    def test_empty_messages(self):
        result, s, d = apply_sliding_window(
            [],
            context_window_turns=5,
            summarise_turns_up_to=15,
            model=MODEL,
        )
        assert result == []
        assert s == 0
        assert d == 0

    def test_custom_summariser_called(self, many_turn_messages):
        calls = []

        def fake_summariser(turn_messages, model):
            calls.append(len(turn_messages))
            return "Summary: discussed math."

        result, summarised, _ = apply_sliding_window(
            many_turn_messages,
            context_window_turns=3,
            summarise_turns_up_to=10,
            model=MODEL,
            summariser=fake_summariser,
        )
        assert len(calls) == summarised

    def test_dropped_count_correct(self, many_turn_messages):
        # 16 turns, keep 2 verbatim, summarise up to 5, drop the rest
        _, summarised, dropped = apply_sliding_window(
            many_turn_messages,
            context_window_turns=2,
            summarise_turns_up_to=5,
            model=MODEL,
            preserve_first_message=False,
        )
        assert summarised + dropped + 2 == 16  # 2 kept verbatim + summarised + dropped = 16
