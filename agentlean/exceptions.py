"""Custom exceptions for AgentLean."""

from __future__ import annotations


class AgentLeanError(Exception):
    """Base exception for all AgentLean errors."""


class BudgetExhaustedError(AgentLeanError):
    """Raised when the configured token or USD budget is exhausted.

    Attributes:
        budget_type: Either "tokens" or "usd".
        limit: The configured limit.
        used: The amount consumed when the error was raised.
    """

    def __init__(self, budget_type: str, limit: float, used: float) -> None:
        self.budget_type = budget_type
        self.limit = limit
        self.used = used
        super().__init__(
            f"Budget exhausted: used {used:.4f} {budget_type} of {limit:.4f} limit"
        )


class OptimisationError(AgentLeanError):
    """Raised internally when an optimisation step fails.

    AgentLean catches this and falls back to the original messages.
    """


class UnsupportedProviderError(AgentLeanError):
    """Raised when an unsupported provider is specified."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"Unsupported provider: '{provider}'. Supported: 'anthropic', 'openai'."
        )


class ConfigurationError(AgentLeanError):
    """Raised when the AgentLeanConfig contains invalid values."""
