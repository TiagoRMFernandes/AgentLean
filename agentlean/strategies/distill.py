"""Tool output distillation strategy.

Detects tool_result / function response messages and compresses their content
before they re-enter the context window on the next LLM call. Three extraction
pipelines are applied based on content type:

- HTML  → strip tags + boilerplate, keep body text, truncate
- JSON  → extract only query-relevant keys, truncate
- Text  → sentence-boundary truncation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..tokenizers import count_tokens, truncate_to_token_limit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def distill_tool_outputs(
    messages: list[dict[str, Any]],
    max_tokens_per_result: int,
    model: str,
    *,
    original_query: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """Return a new messages list with tool outputs compressed.

    Only messages that contain tool results and exceed *max_tokens_per_result*
    are modified. All other messages are returned unchanged (same dict objects).

    Args:
        messages: Conversation messages as passed to the LLM API.
        max_tokens_per_result: Maximum tokens allowed per tool result block.
        model: Model ID (used for token counting).
        original_query: Optional original user query; used to guide JSON key
            extraction towards relevant fields.

    Returns:
        A tuple of (new_messages, n_distilled) where n_distilled is the number
        of tool result blocks that were actually compressed.
    """
    new_messages: list[dict[str, Any]] = []
    n_distilled = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool" or _is_function_response(msg):
            # OpenAI-style: role=tool, content=string
            new_msg, distilled = _distill_openai_tool_message(
                msg, max_tokens_per_result, model, original_query
            )
            new_messages.append(new_msg)
            n_distilled += distilled
            continue

        if isinstance(content, list):
            # Anthropic-style: content is a list of blocks
            new_content, distilled = _distill_content_blocks(
                content, max_tokens_per_result, model, original_query
            )
            if distilled:
                new_messages.append({**msg, "content": new_content})
                n_distilled += distilled
            else:
                new_messages.append(msg)
            continue

        new_messages.append(msg)

    return new_messages, n_distilled


# ---------------------------------------------------------------------------
# Anthropic-style content block handling
# ---------------------------------------------------------------------------


def _distill_content_blocks(
    blocks: list[Any],
    max_tokens: int,
    model: str,
    query: str,
) -> tuple[list[Any], int]:
    """Process a list of Anthropic content blocks, distilling tool_result blocks."""
    new_blocks: list[Any] = []
    n_distilled = 0

    for block in blocks:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        if block.get("type") == "tool_result":
            new_block, distilled = _distill_anthropic_tool_result(
                block, max_tokens, model, query
            )
            new_blocks.append(new_block)
            n_distilled += distilled
        else:
            new_blocks.append(block)

    return new_blocks, n_distilled


def _distill_anthropic_tool_result(
    block: dict[str, Any],
    max_tokens: int,
    model: str,
    query: str,
) -> tuple[dict[str, Any], int]:
    """Distill a single Anthropic tool_result content block."""
    content = block.get("content", "")
    tool_use_id = block.get("tool_use_id", "")

    if isinstance(content, str):
        original_text = content
    elif isinstance(content, list):
        # Content is itself a list of text/image blocks
        text_parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        original_text = "\n".join(text_parts)
    else:
        return block, 0

    original_tokens = count_tokens(original_text, model)
    if original_tokens <= max_tokens:
        return block, 0

    distilled_text, original_tok, distilled_tok = _distill_text(
        original_text, max_tokens, model, query
    )
    stub = _make_stub(original_tok, distilled_tok, tool_use_id)
    final_text = distilled_text + "\n\n" + stub

    if isinstance(content, str):
        new_content: Any = final_text
    else:
        new_content = [{"type": "text", "text": final_text}]

    return {**block, "content": new_content}, 1


# ---------------------------------------------------------------------------
# OpenAI-style tool message handling
# ---------------------------------------------------------------------------


def _is_function_response(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "function" and "content" in msg


def _distill_openai_tool_message(
    msg: dict[str, Any],
    max_tokens: int,
    model: str,
    query: str,
) -> tuple[dict[str, Any], int]:
    content = msg.get("content", "")
    if not isinstance(content, str):
        return msg, 0

    original_tokens = count_tokens(content, model)
    if original_tokens <= max_tokens:
        return msg, 0

    tool_name = msg.get("name", msg.get("tool_call_id", "unknown"))
    distilled_text, original_tok, distilled_tok = _distill_text(
        content, max_tokens, model, query
    )
    stub = _make_stub(original_tok, distilled_tok, tool_name)
    return {**msg, "content": distilled_text + "\n\n" + stub}, 1


# ---------------------------------------------------------------------------
# Content-type detection and distillation pipelines
# ---------------------------------------------------------------------------


def _distill_text(
    text: str, max_tokens: int, model: str, query: str
) -> tuple[str, int, int]:
    """Select and apply the appropriate distillation pipeline.

    Returns:
        (distilled_text, original_token_count, distilled_token_count)
    """
    original_tokens = count_tokens(text, model)
    stripped = text.strip()

    if _looks_like_html(stripped):
        distilled = _distill_html(stripped, max_tokens, model)
    elif _looks_like_json(stripped):
        distilled = _distill_json(stripped, max_tokens, model, query)
    else:
        distilled = truncate_to_token_limit(stripped, max_tokens, model)

    distilled_tokens = count_tokens(distilled, model)
    return distilled, original_tokens, distilled_tokens


# --- HTML pipeline ---


_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

# Blocks likely to be navigation / chrome rather than body content
_BOILERPLATE_RE = re.compile(
    r"<(nav|header|footer|aside|script|style|noscript|iframe|form|button|input|select|textarea|label|"
    r"advertisement|[a-z]*ad[a-z]*|menu|sidebar|breadcrumb|cookie|banner|popup|modal)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<(html|body|div|p|span|article|section|main|h[1-6])\b", text, re.I))


def _distill_html(html: str, max_tokens: int, model: str) -> str:
    # Remove likely-boilerplate sections first
    cleaned = _BOILERPLATE_RE.sub("", html)
    # Strip all remaining tags
    text = _HTML_TAG_RE.sub(" ", cleaned)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return truncate_to_token_limit(text, max_tokens, model)


# --- JSON pipeline ---


def _looks_like_json(text: str) -> bool:
    return bool(text and text[0] in ("{", "["))


def _distill_json(text: str, max_tokens: int, model: str, query: str) -> str:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return truncate_to_token_limit(text, max_tokens, model)

    query_words = set(re.split(r"\W+", query.lower())) - {"", "the", "a", "an", "of", "in", "to"}

    if isinstance(data, dict):
        filtered = _filter_dict(data, query_words, depth=0)
        result = json.dumps(filtered, indent=None, default=str)
    elif isinstance(data, list):
        result = _distill_json_array(data, query_words, max_tokens, model)
    else:
        result = str(data)

    if count_tokens(result, model) > max_tokens:
        result = truncate_to_token_limit(result, max_tokens, model)
    return result


def _filter_dict(data: dict[str, Any], query_words: set[str], depth: int) -> dict[str, Any]:
    """Recursively keep only keys relevant to the query (up to depth 3)."""
    if depth > 3 or not query_words:
        return data

    relevant: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(word in key_lower for word in query_words):
            relevant[key] = value
        elif depth == 0:
            # Always include top-level scalar fields
            if not isinstance(value, (dict, list)):
                relevant[key] = value
            elif isinstance(value, dict):
                sub = _filter_dict(value, query_words, depth + 1)
                if sub:
                    relevant[key] = sub

    # If nothing matched, return top-level scalars as a fallback
    if not relevant:
        return {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    return relevant


def _distill_json_array(
    data: list[Any], query_words: set[str], max_tokens: int, model: str
) -> str:
    """Summarise a JSON array by keeping the first N items that fit."""
    items: list[str] = []
    total_tokens = 0
    for item in data:
        item_str = json.dumps(item, default=str)
        item_tokens = count_tokens(item_str, model)
        if total_tokens + item_tokens > max_tokens * 0.9:
            items.append(f"... and {len(data) - len(items)} more items")
            break
        items.append(item_str)
        total_tokens += item_tokens
    return "[" + ", ".join(items) + "]"


# --- Stub ---


def _make_stub(original_tokens: int, distilled_tokens: int, source: str) -> str:
    return (
        f"[Tool output distilled from ~{original_tokens:,} to ~{distilled_tokens:,} tokens. "
        f"Source: {source}]"
    )
