"""AgentLean core class — the main entry-point for the middleware.

Usage::

    import anthropic
    from agentlean import AgentLean

    client = anthropic.Anthropic()
    lean = AgentLean(client, provider="anthropic")

    response = lean.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=messages,
    )
    print(lean.stats)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from .config import AgentLeanConfig
from .exceptions import OptimisationError, UnsupportedProviderError
from .providers.anthropic import AnthropicMessagesProxy, count_anthropic_tokens
from .providers.openai import OpenAIChatProxy, OpenAICompletionsProxy, count_openai_tokens
from .stats import CallStats, SessionStats
from .strategies.budget import BudgetManager
from .strategies.distill import distill_tool_outputs
from .strategies.window import apply_sliding_window
from .tokenizers import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai"]


class AgentLean:
    """Transparent middleware wrapper around an LLM API client.

    Intercepts every API call to optimise the context (messages + system prompt)
    before it reaches the provider, tracking token savings across the session.

    Args:
        client: An initialised ``anthropic.Anthropic`` or ``openai.OpenAI`` client.
        provider: Either ``"anthropic"`` or ``"openai"``. If omitted, AgentLean
            attempts to detect the provider from the client type.
        config: An :class:`AgentLeanConfig` instance. If omitted, the default
            balanced configuration is used.
        budget_tokens: Shorthand for ``config.budget_tokens``.
        budget_usd: Shorthand for ``config.budget_usd``.
        strategy: Shorthand for ``config.strategy``.
    """

    def __init__(
        self,
        client: Any,
        *,
        provider: Provider | None = None,
        config: AgentLeanConfig | None = None,
        budget_tokens: int | None = None,
        budget_usd: float | None = None,
        strategy: str | None = None,
    ) -> None:
        if config is None:
            config = AgentLeanConfig()
        if budget_tokens is not None:
            config.budget_tokens = budget_tokens
        if budget_usd is not None:
            config.budget_usd = budget_usd
        if strategy is not None:
            config.strategy = strategy  # type: ignore[assignment]

        self._config = config
        self._client = client
        self._provider = provider or _detect_provider(client)
        self._stats = SessionStats()
        self._budget = BudgetManager(
            budget_tokens=config.budget_tokens,
            budget_usd=config.budget_usd,
            base_strategy=config.strategy,
            warn_at_pct=config.warn_at_budget_pct,
            hard_stop=config.hard_stop_at_budget,
        )

        # Expose the appropriate proxy interface
        if self._provider == "anthropic":
            self.messages = AnthropicMessagesProxy(client, self._optimise, self._finalise_call)
        elif self._provider == "openai":
            _completions = OpenAICompletionsProxy(client, self._optimise, self._finalise_call)
            self.chat = OpenAIChatProxy(_completions)
        else:
            raise UnsupportedProviderError(self._provider)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SessionStats:
        """Cumulative session statistics."""
        return self._stats

    @property
    def config(self) -> AgentLeanConfig:
        """The active configuration."""
        return self._config

    def reset_stats(self) -> None:
        """Reset session statistics (does not reset budget usage)."""
        self._stats = SessionStats()

    def analyse_system_prompt(self, prompt: str) -> Any:
        """Advisory analysis of *prompt*. Returns a SystemPromptAnalysis."""
        from .strategies.system import analyse_system_prompt

        return analyse_system_prompt(prompt, model=self._config.summarisation_model)

    # ------------------------------------------------------------------
    # Core optimisation pipeline
    # ------------------------------------------------------------------

    def _optimise(
        self,
        messages: list[dict[str, Any]],
        system: Any,
        model: str,
    ) -> tuple[list[dict[str, Any]], Any, CallStats]:
        """Run the full optimisation pipeline on *messages* and *system*.

        Returns:
            (optimised_messages, optimised_system, call_stats)

        If any optimisation step raises an exception, it is caught and logged,
        and the original messages are returned unchanged — AgentLean never
        causes a call to fail.
        """
        call_stats = CallStats(model=model, strategy_applied=self._config.strategy)

        # Check budget before doing any work
        self._budget.check_before_call()
        effective_strategy = self._budget.effective_strategy()
        call_stats.strategy_applied = effective_strategy

        # Count original tokens before any transformation
        if self._provider == "anthropic":
            original_tokens = count_anthropic_tokens(messages, system, model)
        else:
            system_text = system if isinstance(system, str) else None
            all_msgs = (
                [{"role": "system", "content": system_text}] + messages
                if system_text
                else messages
            )
            original_tokens = count_openai_tokens(all_msgs, model)

        call_stats.original_input_tokens = original_tokens

        try:
            optimised_messages, optimised_system, call_stats = self._run_pipeline(
                messages=messages,
                system=system,
                model=model,
                strategy=effective_strategy,
                call_stats=call_stats,
            )
        except Exception as exc:
            logger.warning(
                "AgentLean optimisation failed (%s), passing original messages through. "
                "Error: %s",
                type(exc).__name__,
                exc,
            )
            call_stats.optimised_input_tokens = original_tokens
            optimised_messages = messages
            optimised_system = system

        # Approximate optimised token count — will be overwritten by the API's
        # actual reported count in _finalise_call().
        if call_stats.optimised_input_tokens == 0:
            if self._provider == "anthropic":
                call_stats.optimised_input_tokens = count_anthropic_tokens(
                    optimised_messages, optimised_system, model
                )
            else:
                call_stats.optimised_input_tokens = count_messages_tokens(
                    optimised_messages, model
                )

        return optimised_messages, optimised_system, call_stats

    def _finalise_call(
        self,
        call_stats: CallStats,
        output_tokens: int,
        api_input_tokens: int,
    ) -> None:
        """Record stats and budget after the API response is received.

        Called by provider proxies once they have the actual token counts
        from the API response.
        """
        from .pricing import estimate_cost

        if api_input_tokens:
            call_stats.optimised_input_tokens = api_input_tokens
        call_stats.output_tokens = output_tokens
        call_stats.estimated_cost_usd = estimate_cost(
            call_stats.optimised_input_tokens, output_tokens, call_stats.model
        )
        call_stats.estimated_cost_saved_usd = estimate_cost(
            max(0, call_stats.original_input_tokens - call_stats.optimised_input_tokens),
            0,
            call_stats.model,
        )
        self._budget.record_call(
            call_stats.optimised_input_tokens, output_tokens, call_stats.model
        )
        self._stats.record(call_stats)
        if self._config.on_call_complete is not None:
            try:
                self._config.on_call_complete(call_stats)  # type: ignore[operator]
            except Exception:
                pass

    def _run_pipeline(
        self,
        messages: list[dict[str, Any]],
        system: Any,
        model: str,
        strategy: str,
        call_stats: CallStats,
    ) -> tuple[list[dict[str, Any]], Any, CallStats]:
        """Execute optimisation strategies in sequence."""
        optimised_messages = list(messages)
        optimised_system = system

        # 1. Tool output distillation (all strategies)
        if strategy in ("conservative", "balanced", "aggressive"):
            query = _extract_first_user_text(messages)
            optimised_messages, n_distilled = distill_tool_outputs(
                optimised_messages,
                max_tokens_per_result=self._config.max_tool_output_tokens,
                model=model,
                original_query=query,
            )
            call_stats.tool_outputs_distilled = n_distilled

        # 2. Sliding context window (balanced + aggressive)
        if strategy in ("balanced", "aggressive"):
            optimised_messages, turns_summarised, turns_dropped = apply_sliding_window(
                optimised_messages,
                context_window_turns=self._config.context_window_turns,
                summarise_turns_up_to=self._config.summarise_turns_up_to,
                model=model,
                summariser=self._make_summariser() if self._config.summarisation_model else None,
                preserve_first_message=self._config.preserve_first_message,
            )
            call_stats.turns_summarised = turns_summarised
            call_stats.turns_dropped = turns_dropped

        # Count final tokens (approximate, will be overwritten by API response)
        if self._provider == "anthropic":
            call_stats.optimised_input_tokens = count_anthropic_tokens(
                optimised_messages, optimised_system, model
            )
        else:
            call_stats.optimised_input_tokens = count_messages_tokens(
                optimised_messages, model
            )

        return optimised_messages, optimised_system, call_stats

    def _make_summariser(self):  # type: ignore[return]
        """Return a summariser function if the config model is set, else None."""
        summarisation_model = self._config.summarisation_model
        if not summarisation_model or self._provider != "anthropic":
            return None

        def _summarise(turn_messages: list[dict[str, Any]], _model: str) -> str:
            """Call the cheap summarisation model to compress a turn."""
            turn_text = "\n".join(
                f"{m.get('role', 'unknown')}: {_extract_text(m)}"
                for m in turn_messages
            )
            prompt = (
                "Summarise the following conversation turn in 2-3 sentences. "
                "Keep only the key facts, decisions, and results.\n\n"
                f"Turn:\n{turn_text}"
            )
            response = self._client.messages.create(
                model=summarisation_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content
            if content and hasattr(content[0], "text"):
                return content[0].text
            return turn_text[:300]

        return _summarise


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def optimise_messages(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    strategy: str = "balanced",
    max_context_tokens: int = 16_000,
    model: str = "claude-sonnet-4-6",
    config: AgentLeanConfig | None = None,
) -> list[dict[str, Any]]:
    """Optimise a messages list without wrapping a client.

    This functional API is useful when you build raw API calls and just want
    to compress the context before sending.

    Args:
        messages: The conversation messages to optimise.
        system: Optional system prompt (not modified, used for token counting).
        strategy: One of "conservative", "balanced", "aggressive".
        max_context_tokens: Target maximum token budget for the messages.
        model: Model ID for token counting.
        config: Optional AgentLeanConfig. If provided, its values take precedence
            over strategy / max_context_tokens.

    Returns:
        Optimised messages list.
    """
    if config is None:
        config = AgentLeanConfig(strategy=strategy)  # type: ignore[arg-type]

    query = _extract_first_user_text(messages)

    # Tool distillation
    optimised, _ = distill_tool_outputs(
        messages,
        max_tokens_per_result=config.max_tool_output_tokens,
        model=model,
        original_query=query,
    )

    # Sliding window (if not conservative)
    if config.strategy != "conservative":
        optimised, _, _ = apply_sliding_window(
            optimised,
            context_window_turns=config.context_window_turns,
            summarise_turns_up_to=config.summarise_turns_up_to,
            model=model,
            preserve_first_message=config.preserve_first_message,
        )

    return optimised


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_provider(client: Any) -> Provider:
    """Infer the provider from the client's class name."""
    class_name = type(client).__name__.lower()
    module_name = getattr(type(client), "__module__", "").lower()

    if "anthropic" in module_name or "anthropic" in class_name:
        return "anthropic"
    if "openai" in module_name or "openai" in class_name:
        return "openai"

    raise UnsupportedProviderError(f"<could not detect from {type(client).__name__}>")


def _extract_first_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the first user message (for query-guided distillation)."""
    for msg in messages:
        if msg.get("role") == "user":
            return _extract_text(msg)
    return ""


def _extract_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""
