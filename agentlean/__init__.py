"""AgentLean — LLM middleware that dramatically reduces token costs in agentic workflows.

Quick start::

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

from .config import AgentLeanConfig
from .core import AgentLean, optimise_messages
from .exceptions import (
    AgentLeanError,
    BudgetExhaustedError,
    ConfigurationError,
    OptimisationError,
    UnsupportedProviderError,
)
from .stats import CallStats, SessionStats
from .strategies.system import SystemPromptUsageTracker, analyse_system_prompt

__all__ = [
    # Main class
    "AgentLean",
    # Functional API
    "optimise_messages",
    # Configuration
    "AgentLeanConfig",
    # Stats
    "SessionStats",
    "CallStats",
    # Exceptions
    "AgentLeanError",
    "BudgetExhaustedError",
    "ConfigurationError",
    "OptimisationError",
    "UnsupportedProviderError",
    # System prompt utilities
    "analyse_system_prompt",
    "SystemPromptUsageTracker",
]

__version__ = "0.1.0"
