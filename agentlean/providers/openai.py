"""OpenAI provider wrapper for AgentLean.

Wraps an ``openai.OpenAI`` client so that ``lean.chat.completions.create()``
is a drop-in replacement for ``client.chat.completions.create()``.

OpenAI message structure reference:
  - messages: list of {"role": "system"|"user"|"assistant"|"tool"|"function", "content": str}
  - tool messages: {"role": "tool", "tool_call_id": ..., "content": str}
  - function messages (legacy): {"role": "function", "name": ..., "content": str}
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ..stats import CallStats
from ..tokenizers import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)


class OpenAICompletionsProxy:
    """Proxy for ``client.chat.completions`` that applies AgentLean optimisation.

    Users call ``lean.chat.completions.create(...)`` exactly as they would
    call ``client.chat.completions.create(...)``.
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
        """Intercept a chat.completions.create call, optimise, and forward."""
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        model: str = kwargs.get("model", "")

        # OpenAI uses a system message in the messages list, not a top-level param
        system_msg, body_messages = _split_system(messages)
        system_text = system_msg.get("content", "") if system_msg else None

        t0 = time.monotonic()
        optimised_messages, optimised_system, call_stats = self._optimise_fn(
            messages=body_messages,
            system=system_text,
            model=model,
        )

        # Reconstruct the full messages list with system at the front
        full_messages: list[dict[str, Any]] = []
        if optimised_system is not None and system_msg is not None:
            full_messages.append({**system_msg, "content": optimised_system})
        elif system_msg is not None:
            full_messages.append(system_msg)
        full_messages.extend(optimised_messages)

        forwarded = dict(kwargs)
        forwarded["messages"] = full_messages

        response = self._client.chat.completions.create(**forwarded)
        call_stats.latency_ms = (time.monotonic() - t0) * 1000

        # Extract usage from the OpenAI response and finalise stats
        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        api_input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0

        self._finalise_fn(call_stats, output_tokens, api_input_tokens)

        return response


class OpenAIChatProxy:
    """Proxy for ``client.chat`` that exposes ``completions``."""

    def __init__(self, completions: OpenAICompletionsProxy) -> None:
        self.completions = completions


def count_openai_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """Count approximate total input tokens for an OpenAI request."""
    return count_messages_tokens(messages, model)


def _split_system(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Separate the leading system message (if any) from the rest."""
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return None, messages
