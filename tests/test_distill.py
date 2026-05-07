"""Tests for tool output distillation."""

from __future__ import annotations

import pytest

from agentlean.strategies.distill import (
    distill_tool_outputs,
    _distill_html,
    _distill_json,
    _looks_like_html,
    _looks_like_json,
    _make_stub,
)
from agentlean.tokenizers import count_tokens


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 100


class TestDistillToolOutputs:
    def test_short_result_unchanged(self, short_messages):
        result, n_distilled = distill_tool_outputs(
            short_messages, max_tokens_per_result=500, model=MODEL
        )
        assert n_distilled == 0
        assert result == short_messages

    def test_large_tool_result_is_distilled(self, tool_result_messages):
        result, n_distilled = distill_tool_outputs(
            tool_result_messages, max_tokens_per_result=50, model=MODEL
        )
        assert n_distilled >= 1

        # The tool result message should be shorter now
        tool_msg = next(
            m for m in result
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"] if isinstance(b, dict))
        )
        content = tool_msg["content"][0]["content"]
        assert count_tokens(content, MODEL) <= 200  # well under original

    def test_stub_appended(self, tool_result_messages):
        result, n_distilled = distill_tool_outputs(
            tool_result_messages, max_tokens_per_result=50, model=MODEL
        )
        tool_msg = next(
            m for m in result
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"] if isinstance(b, dict))
        )
        content = tool_msg["content"][0]["content"]
        assert "[Tool output distilled" in content

    def test_openai_tool_message_distilled(self):
        long_content = "important data " * 200
        messages = [
            {"role": "tool", "tool_call_id": "call_123", "name": "search", "content": long_content}
        ]
        result, n_distilled = distill_tool_outputs(messages, max_tokens_per_result=50, model=MODEL)
        assert n_distilled == 1
        assert count_tokens(result[0]["content"], MODEL) < count_tokens(long_content, MODEL)

    def test_non_tool_messages_pass_through(self, short_messages):
        result, n_distilled = distill_tool_outputs(
            short_messages, max_tokens_per_result=5, model=MODEL
        )
        # short_messages has no tool results, nothing should be distilled
        assert n_distilled == 0

    def test_multiple_tool_results_counted(self):
        large = "token " * 300
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": large},
                    {"type": "tool_result", "tool_use_id": "b", "content": large},
                ],
            }
        ]
        _, n_distilled = distill_tool_outputs(messages, max_tokens_per_result=50, model=MODEL)
        assert n_distilled == 2


class TestHtmlDistillation:
    def test_detects_html(self):
        assert _looks_like_html("<div>hello</div>")
        assert _looks_like_html("<html><body><p>text</p></body></html>")
        assert not _looks_like_html('{"key": "value"}')
        assert not _looks_like_html("plain text with no tags")

    def test_strips_html_tags(self, web_search_fixture):
        html = web_search_fixture["html_output"]
        result = _distill_html(html, MAX_TOKENS, MODEL)
        assert "<" not in result or "[...]" in result
        assert count_tokens(result, MODEL) <= MAX_TOKENS + 10  # small tolerance

    def test_removes_nav_boilerplate(self, web_search_fixture):
        html = web_search_fixture["html_output"]
        result = _distill_html(html, 500, MODEL)
        # Navigation links should be stripped
        assert "Privacy Policy" not in result or len(result) < len(html)


class TestJsonDistillation:
    def test_detects_json(self):
        assert _looks_like_json('{"key": "value"}')
        assert _looks_like_json('[1, 2, 3]')
        assert not _looks_like_json("plain text")
        assert not _looks_like_json("<html>")

    def test_distills_large_json(self, web_search_fixture):
        json_str = str(web_search_fixture["json_api_response"]).replace("'", '"')
        import json
        json_str = json.dumps(web_search_fixture["json_api_response"])
        result = _distill_json(json_str, MAX_TOKENS, MODEL, query="python frameworks performance")
        assert count_tokens(result, MODEL) <= MAX_TOKENS + 20

    def test_query_guided_json_filtering(self):
        import json
        data = {
            "framework_name": "FastAPI",
            "performance_rps": 47000,
            "creator": "Sebastián Ramírez",
            "unrelated_field_1": "x" * 100,
            "unrelated_field_2": "y" * 100,
        }
        result = _distill_json(json.dumps(data), 200, MODEL, query="performance rps")
        # performance_rps should be included
        assert "47000" in result or "performance" in result.lower()


class TestStub:
    def test_stub_format(self):
        stub = _make_stub(8200, 480, "web_search")
        assert "8,200" in stub
        assert "480" in stub
        assert "web_search" in stub
