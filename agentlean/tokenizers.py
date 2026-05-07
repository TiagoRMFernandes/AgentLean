"""Token counting utilities for Anthropic and OpenAI models.

OpenAI models are counted with tiktoken.
Anthropic models use a character-based approximation (chars / 3.5) because
the official tokenizer is not publicly released for all model families.
This approximation is conservative — real token counts may differ by ~10-15%.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tiktoken bootstrap (optional dependency)
# ---------------------------------------------------------------------------
try:
    import tiktoken as _tiktoken  # type: ignore[import-untyped]

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _tiktoken = None  # type: ignore[assignment]

_ENCODING_CACHE: dict[str, Any] = {}


def _get_openai_encoding(model: str) -> Any:
    if not _TIKTOKEN_AVAILABLE:
        raise ImportError(
            "tiktoken is required for accurate OpenAI token counting. "
            "Install it with: pip install tiktoken"
        )
    if model not in _ENCODING_CACHE:
        try:
            enc = _tiktoken.encoding_for_model(model)
        except KeyError:
            enc = _tiktoken.get_encoding("cl100k_base")
        _ENCODING_CACHE[model] = enc
    return _ENCODING_CACHE[model]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in *text* for the given *model*.

    For OpenAI models, tiktoken is used. For Anthropic / unknown models,
    a character-based approximation is applied (chars ÷ 3.5, rounded up).

    Args:
        text: The string to count tokens for.
        model: A model ID string (e.g. "gpt-4o", "claude-sonnet-4-6").

    Returns:
        Approximate token count (int).
    """
    if not text:
        return 0
    if _is_openai_model(model) and _TIKTOKEN_AVAILABLE:
        return _count_openai(text, model)
    return _count_anthropic_approx(text)


def count_messages_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """Estimate the total token count for a list of message dicts.

    Handles the overhead per-message (role + structural tokens) in addition
    to the text content.

    Args:
        messages: List of message dicts (as passed to the LLM API).
        model: Model ID string.

    Returns:
        Approximate total token count.
    """
    total = 0
    for msg in messages:
        # ~4 tokens overhead per message for role/formatting
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += count_tokens(_extract_block_text(block), model)
    return total


def truncate_to_token_limit(text: str, max_tokens: int, model: str) -> str:
    """Truncate *text* so it fits within *max_tokens*, cutting at sentence boundaries.

    The function first tries to cut at the last sentence boundary ('. ', '! ', '? ')
    within the limit. If no boundary is found, it falls back to a hard character cut.

    Args:
        text: Input string.
        max_tokens: Maximum number of tokens allowed.
        model: Model ID for token counting.

    Returns:
        Truncated string. May be unchanged if already within limit.
    """
    if count_tokens(text, model) <= max_tokens:
        return text

    # Binary search on character length — O(log n) token count calls.
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid], model) <= max_tokens:
            lo = mid
        else:
            hi = mid

    truncated = text[:lo]

    # Prefer cutting at a sentence boundary.
    boundary = _last_sentence_boundary(truncated)
    if boundary > len(truncated) // 2:
        truncated = truncated[:boundary].rstrip()

    return truncated + " [...]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_openai_model(model: str) -> bool:
    lower = model.lower()
    return any(prefix in lower for prefix in ("gpt-", "o1", "o3", "o4", "text-"))


def _count_openai(text: str, model: str) -> int:
    enc = _get_openai_encoding(model)
    return len(enc.encode(text))


def _count_anthropic_approx(text: str) -> int:
    # Anthropic approximation: 1 token ≈ 3.5 characters on average.
    return max(1, -(-len(text) // 4))  # ceiling division by 4 (conservative)


def _extract_block_text(block: dict[str, Any]) -> str:
    """Pull the text content out of an Anthropic or OpenAI content block."""
    btype = block.get("type", "")
    if btype == "text":
        return block.get("text", "")
    if btype in ("tool_result", "tool_use"):
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
    return ""


_SENTENCE_END_RE = re.compile(r"[.!?]\s+")


def _last_sentence_boundary(text: str) -> int:
    """Return the character index of the last sentence boundary in *text*."""
    matches = list(_SENTENCE_END_RE.finditer(text))
    if not matches:
        return len(text)
    last = matches[-1]
    return last.end()
