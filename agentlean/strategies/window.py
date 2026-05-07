"""Sliding context window with decay.

Conversation turns are divided into three zones:
  1. Recent (indices >= len - context_window_turns)  → kept verbatim
  2. Middle (within summarise_turns_up_to)            → summarised to key facts
  3. Far past (beyond summarise_turns_up_to)          → dropped

The system prompt and the first user message are always preserved
(when preserve_first_message=True).

A "turn" is a consecutive user+assistant pair. Tool messages are grouped
with the assistant turn they belong to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation round-trip (user + assistant + any tool messages)."""

    index: int
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.messages


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def apply_sliding_window(
    messages: list[dict[str, Any]],
    *,
    context_window_turns: int,
    summarise_turns_up_to: int,
    model: str,
    summariser: Callable[[list[dict[str, Any]], str], str] | None = None,
    preserve_first_message: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """Apply sliding context window compression to a message list.

    Args:
        messages: Conversation messages (may include system).
        context_window_turns: Number of recent turns to keep verbatim.
        summarise_turns_up_to: Maximum turn index (from oldest) to summarise;
            turns beyond this limit are dropped.
        model: Model ID (for token counting in summariser).
        summariser: Optional callable that takes a list of messages and the model
            string, and returns a summary string. If None, a simple inline
            extractor is used (no extra API call required).
        preserve_first_message: Always preserve the first user message.

    Returns:
        (new_messages, turns_summarised, turns_dropped)
    """
    if not messages:
        return messages, 0, 0

    # Separate any leading system message — it's always preserved.
    system_msgs: list[dict[str, Any]] = []
    body_msgs: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system" and not body_msgs:
            system_msgs.append(msg)
        else:
            body_msgs.append(msg)

    turns = _segment_into_turns(body_msgs)
    n_turns = len(turns)

    if n_turns <= context_window_turns:
        # Nothing to do
        return messages, 0, 0

    # Determine which turns fall in each zone.
    recent_start = n_turns - context_window_turns

    turns_summarised = 0
    turns_dropped = 0
    result_messages: list[dict[str, Any]] = list(system_msgs)

    for i, turn in enumerate(turns):
        is_first = i == 0 and preserve_first_message

        if i >= recent_start:
            # Recent zone: keep verbatim
            result_messages.extend(turn.messages)

        elif i < summarise_turns_up_to:
            # Middle zone: summarise (or keep if first)
            if is_first:
                result_messages.extend(turn.messages)
            else:
                summary_msg = _summarise_turn(turn, model, summariser)
                result_messages.append(summary_msg)
                turns_summarised += 1

        else:
            # Far past: drop (but always keep first user message)
            if is_first:
                result_messages.extend(turn.messages)
            else:
                turns_dropped += 1

    return result_messages, turns_summarised, turns_dropped


# ---------------------------------------------------------------------------
# Turn segmentation
# ---------------------------------------------------------------------------


def _segment_into_turns(messages: list[dict[str, Any]]) -> list[Turn]:
    """Group messages into logical turns (user → assistant → tool*) sequences."""
    turns: list[Turn] = []
    current_turn: Turn | None = None

    for msg in messages:
        role = msg.get("role", "")

        if role == "user":
            # A new user message starts a new turn
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = Turn(index=len(turns), messages=[msg])

        elif role in ("assistant", "tool", "function"):
            if current_turn is None:
                current_turn = Turn(index=0, messages=[])
            current_turn.messages.append(msg)

        else:
            # Unknown role — attach to current turn or start a new one
            if current_turn is None:
                current_turn = Turn(index=0, messages=[])
            current_turn.messages.append(msg)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def _summarise_turn(
    turn: Turn,
    model: str,
    summariser: Callable[[list[dict[str, Any]], str], str] | None,
) -> dict[str, Any]:
    """Return a synthetic summary message representing a compressed turn."""
    if summariser is not None:
        try:
            summary_text = summariser(turn.messages, model)
        except Exception as exc:
            logger.warning("Summariser raised an error: %s — falling back to inline.", exc)
            summary_text = _inline_summary(turn)
    else:
        summary_text = _inline_summary(turn)

    return {
        "role": "user",
        "content": f"[Summary of earlier turn {turn.index + 1}]: {summary_text}",
    }


def _inline_summary(turn: Turn) -> str:
    """Build a lightweight summary from the turn without an extra LLM call."""
    parts: list[str] = []

    for msg in turn.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str) and content.strip():
            excerpt = _first_n_chars(content, 200)
            parts.append(f"{role.capitalize()}: {excerpt}")

        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    excerpt = _first_n_chars(block.get("text", ""), 200)
                    if excerpt:
                        parts.append(f"{role.capitalize()}: {excerpt}")
                elif btype == "tool_use":
                    tool_name = block.get("name", "tool")
                    inp = block.get("input", {})
                    inp_str = _first_n_chars(str(inp), 100)
                    parts.append(f"Called tool '{tool_name}' with: {inp_str}")
                elif btype == "tool_result":
                    result_preview = _first_n_chars(_extract_tool_result_text(block), 100)
                    parts.append(f"Tool returned: {result_preview}")

    if not parts:
        return "(empty turn)"
    return " | ".join(parts)


def _extract_tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _first_n_chars(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."
