"""Anthropic provider wrapper for AgentLean.

Wraps an ``anthropic.Anthropic`` client so that ``lean.messages.create()``
is a drop-in replacement for ``client.messages.create()``.

Anthropic message structure reference:
  - messages: list of {"role": "user"|"assistant", "content": str | list[block]}
  - content blocks: {"type": "text"|"tool_use"|"tool_result", ...}
  - system: top-level string or list of content blocks
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from ..stats import CallStats
from ..tokenizers import count_messages_tokens, count_tokens

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AnthropicMessagesProxy:
    """A proxy for ``client.messages`` that applies AgentLean optimisation.

    Users call ``lean.messages.create(...)`` exactly as they would call
    ``client.messages.create(...)``.
    """

    def __init__(
        self,
        client: Any,
        optimise_fn: Callable[..., tuple[list[dict[str, Any]], str | Any, CallStats]],
        finalise_fn: Callable[..., None],
    ) -> None:
        self._client = client
        self._optimise_fn = optimise_fn
        self._finalise_fn = finalise_fn

    def create(self, **kwargs: Any) -> Any:
        """Intercept a messages.create call, optimise, and forward to Anthropic.

        All kwargs are passed through after context optimisation.
        """
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        system: str | list[Any] | None = kwargs.get("system")
        model: str = kwargs.get("model", "")

        t0 = time.monotonic()
        optimised_messages, optimised_system, call_stats = self._optimise_fn(
            messages=messages,
            system=system,
            model=model,
        )

        # Build the forwarded kwargs with optimised context
        forwarded = dict(kwargs)
        forwarded["messages"] = optimised_messages
        if optimised_system is not None:
            forwarded["system"] = optimised_system

        response = self._client.messages.create(**forwarded)
        call_stats.latency_ms = (time.monotonic() - t0) * 1000

        # Extract token usage from the Anthropic response and finalise stats
        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        api_input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0

        self._finalise_fn(call_stats, output_tokens, api_input_tokens)

        return response


def count_anthropic_tokens(
    messages: list[dict[str, Any]],
    system: str | list[Any] | None,
    model: str,
) -> int:
    """Count approximate total input tokens for an Anthropic request."""
    total = count_messages_tokens(messages, model)
    if system:
        system_text = system if isinstance(system, str) else _flatten_system(system)
        total += count_tokens(system_text, model)
    return total


def _flatten_system(system: list[Any]) -> str:
    """Flatten a structured system prompt (list of blocks) to plain text."""
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                content = block.get("content", "")
                parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)
