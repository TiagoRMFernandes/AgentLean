"""System prompt analysis utilities.

This module provides *advisory* analysis of system prompts — it never
auto-modifies them. Results are returned as structured suggestions so the
developer can decide what to trim manually.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..tokenizers import count_tokens


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SystemPromptSuggestion:
    """A single optimisation suggestion for a system prompt.

    Attributes:
        category: High-level category (e.g., "redundancy", "verbosity").
        severity: "low" | "medium" | "high" — impact if addressed.
        description: Human-readable explanation of the issue.
        excerpt: The relevant excerpt from the prompt (first 200 chars).
        estimated_savings_tokens: Rough estimate of tokens that could be saved.
    """

    category: str
    severity: str
    description: str
    excerpt: str
    estimated_savings_tokens: int = 0


@dataclass
class SystemPromptAnalysis:
    """Full analysis result for a system prompt.

    Attributes:
        token_count: Total token count of the prompt.
        suggestions: List of optimisation suggestions.
        redundant_sections: Sections detected as duplicated.
        verbose_sections: Sections that are unusually long relative to their
            semantic density.
        usage_tracked_sections: Section labels that have been tracked across
            multiple runs (populated when ``track_usage`` is called).
    """

    token_count: int
    suggestions: list[SystemPromptSuggestion] = field(default_factory=list)
    redundant_sections: list[str] = field(default_factory=list)
    verbose_sections: list[str] = field(default_factory=list)
    usage_tracked_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "suggestion_count": len(self.suggestions),
            "suggestions": [
                {
                    "category": s.category,
                    "severity": s.severity,
                    "description": s.description,
                    "excerpt": s.excerpt,
                    "estimated_savings_tokens": s.estimated_savings_tokens,
                }
                for s in self.suggestions
            ],
            "redundant_sections": self.redundant_sections,
            "verbose_sections": self.verbose_sections,
        }

    def __repr__(self) -> str:
        high = sum(1 for s in self.suggestions if s.severity == "high")
        med = sum(1 for s in self.suggestions if s.severity == "medium")
        total_savings = sum(s.estimated_savings_tokens for s in self.suggestions)
        return (
            f"SystemPromptAnalysis(tokens={self.token_count}, "
            f"suggestions={len(self.suggestions)} [{high} high / {med} medium], "
            f"potential_savings≈{total_savings} tokens)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse_system_prompt(prompt: str, model: str = "claude-sonnet-4-6") -> SystemPromptAnalysis:
    """Analyse *prompt* and return structured optimisation suggestions.

    This function is purely advisory — it never modifies the prompt.

    Args:
        prompt: The system prompt text to analyse.
        model: Model ID used for token counting.

    Returns:
        A ``SystemPromptAnalysis`` with suggestions and metrics.
    """
    token_count = count_tokens(prompt, model)
    analysis = SystemPromptAnalysis(token_count=token_count)

    _check_repetition(prompt, analysis, model)
    _check_verbose_lists(prompt, analysis, model)
    _check_redundant_preamble(prompt, analysis, model)
    _check_excessive_examples(prompt, analysis, model)
    _check_contradictory_instructions(prompt, analysis)
    _check_filler_phrases(prompt, analysis, model)

    return analysis


class SystemPromptUsageTracker:
    """Track which sections of a system prompt are referenced in model outputs.

    Usage::

        tracker = SystemPromptUsageTracker(system_prompt)
        # ... after each run ...
        tracker.record_output(model_response_text)
        # ... after many runs ...
        dead_weight = tracker.unused_sections()
    """

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self._sections = _extract_sections(system_prompt)
        self._usage_counts: dict[str, int] = {s: 0 for s in self._sections}
        self._total_outputs = 0

    def record_output(self, output_text: str) -> None:
        """Record a model output and check which sections it references."""
        self._total_outputs += 1
        lower_output = output_text.lower()
        for section in self._sections:
            # Heuristic: check if key phrases from the section appear in output
            keywords = _extract_keywords(section)
            if any(kw in lower_output for kw in keywords):
                self._usage_counts[section] += 1

    def unused_sections(self, threshold: float = 0.1) -> list[str]:
        """Return sections referenced in fewer than *threshold* fraction of outputs.

        Args:
            threshold: Sections used in fewer than this fraction of runs are
                considered potential dead weight.

        Returns:
            List of section labels/excerpts considered unused.
        """
        if self._total_outputs == 0:
            return []
        return [
            section
            for section, count in self._usage_counts.items()
            if count / self._total_outputs < threshold
        ]

    def usage_report(self) -> dict[str, float]:
        """Return usage frequency (0–1) for each detected section."""
        if self._total_outputs == 0:
            return {}
        return {
            section: count / self._total_outputs
            for section, count in self._usage_counts.items()
        }


# ---------------------------------------------------------------------------
# Internal analysis rules
# ---------------------------------------------------------------------------


def _check_repetition(prompt: str, analysis: SystemPromptAnalysis, model: str) -> None:
    """Detect repeated sentences or near-duplicate paragraphs."""
    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    seen: dict[str, int] = {}
    for i, sent in enumerate(sentences):
        normalised = re.sub(r"\s+", " ", sent.lower().strip())
        if len(normalised) < 20:
            continue
        if normalised in seen:
            analysis.suggestions.append(
                SystemPromptSuggestion(
                    category="redundancy",
                    severity="medium",
                    description=(
                        f"Sentence appears more than once (first at sentence {seen[normalised] + 1}, "
                        f"again at {i + 1}). Remove the duplicate."
                    ),
                    excerpt=sent[:200],
                    estimated_savings_tokens=count_tokens(sent, model),
                )
            )
            analysis.redundant_sections.append(sent[:100])
        else:
            seen[normalised] = i


def _check_verbose_lists(prompt: str, analysis: SystemPromptAnalysis, model: str) -> None:
    """Detect bulleted/numbered lists longer than 10 items."""
    list_block_re = re.compile(
        r"((?:^[ \t]*(?:[-*•]|\d+[.)]) .+\n?){10,})",
        re.MULTILINE,
    )
    for match in list_block_re.finditer(prompt):
        block = match.group(0)
        items = len(re.findall(r"^[ \t]*(?:[-*•]|\d+[.)])", block, re.MULTILINE))
        analysis.suggestions.append(
            SystemPromptSuggestion(
                category="verbosity",
                severity="medium",
                description=(
                    f"Found a list with {items} items. Consider grouping related items "
                    "or replacing with a brief prose summary."
                ),
                excerpt=block[:200],
                estimated_savings_tokens=max(0, count_tokens(block, model) - 50),
            )
        )
        analysis.verbose_sections.append(block[:100])


def _check_redundant_preamble(prompt: str, analysis: SystemPromptAnalysis, model: str) -> None:
    """Detect a long preamble before the actual instructions."""
    preamble_re = re.compile(
        r"^(You are .{0,300}?\.)\s",
        re.DOTALL,
    )
    m = preamble_re.match(prompt.strip())
    if m and count_tokens(m.group(1), model) > 100:
        analysis.suggestions.append(
            SystemPromptSuggestion(
                category="verbosity",
                severity="low",
                description=(
                    "The opening 'You are...' preamble is over 100 tokens. "
                    "Consider condensing the persona description."
                ),
                excerpt=m.group(1)[:200],
                estimated_savings_tokens=count_tokens(m.group(1), model) // 2,
            )
        )


def _check_excessive_examples(prompt: str, analysis: SystemPromptAnalysis, model: str) -> None:
    """Detect sections with more than 3 inline examples."""
    example_block_re = re.compile(
        r"(?:example|e\.g\.|for instance|such as)[:\s](.{0,500}?)\n\n",
        re.IGNORECASE | re.DOTALL,
    )
    examples = example_block_re.findall(prompt)
    if len(examples) > 3:
        savings = sum(count_tokens(ex, model) for ex in examples[3:])
        analysis.suggestions.append(
            SystemPromptSuggestion(
                category="verbosity",
                severity="low",
                description=(
                    f"Found {len(examples)} example blocks. "
                    "Consider keeping only the 2-3 most illustrative and removing the rest."
                ),
                excerpt=examples[0][:200],
                estimated_savings_tokens=savings,
            )
        )


def _check_contradictory_instructions(prompt: str, analysis: SystemPromptAnalysis) -> None:
    """Detect common contradictory instruction patterns."""
    # Each tuple is (pattern_a, pattern_b, description)
    contradictions = [
        (
            r"\balways\b",
            r"\bnever\b",
            "'always' and 'never' both appear — check for conflicting absolutes.",
        ),
        (
            r"\bdo not\b",
            r"\byou must\b",
            "'do not' and 'you must' both appear — check for conflicting obligations.",
        ),
        (
            r"\b(brief|concise|short)\b",
            r"\b(detailed|comprehensive|thorough|exhaustive)\b",
            "Instructions to be brief conflict with instructions to be detailed.",
        ),
    ]
    lower = prompt.lower()
    for pat_a, pat_b, description in contradictions:
        if re.search(pat_a, lower) and re.search(pat_b, lower):
            analysis.suggestions.append(
                SystemPromptSuggestion(
                    category="contradiction",
                    severity="high",
                    description=f"Potentially contradictory instructions: {description}",
                    excerpt=prompt[:200],
                    estimated_savings_tokens=0,
                )
            )


def _check_filler_phrases(prompt: str, analysis: SystemPromptAnalysis, model: str) -> None:
    """Detect common filler phrases that add tokens without semantic value."""
    fillers = [
        "please note that",
        "it is important to note",
        "it's worth mentioning",
        "as an ai language model",
        "as a helpful assistant",
        "i want you to",
        "make sure to always",
        "always make sure to",
        "needless to say",
        "it goes without saying",
    ]
    found: list[str] = []
    lower = prompt.lower()
    for filler in fillers:
        if filler in lower:
            found.append(filler)

    if found:
        analysis.suggestions.append(
            SystemPromptSuggestion(
                category="filler",
                severity="low",
                description=(
                    f"Found {len(found)} low-value filler phrase(s): "
                    + ", ".join(f'"{f}"' for f in found[:5])
                    + ". Remove or rephrase to save tokens."
                ),
                excerpt=", ".join(found[:3]),
                estimated_savings_tokens=len(found) * 8,
            )
        )


# ---------------------------------------------------------------------------
# Section extraction helpers
# ---------------------------------------------------------------------------


def _extract_sections(prompt: str) -> list[str]:
    """Split a prompt into logical sections by headers or double newlines."""
    sections = re.split(r"\n#{1,3}\s+|\n\n", prompt)
    return [s.strip()[:150] for s in sections if s.strip()]


def _extract_keywords(section: str) -> list[str]:
    """Extract lower-cased keywords (length >= 5) from a section excerpt."""
    words = re.findall(r"\b[a-z]{5,}\b", section.lower())
    return list(set(words))[:10]
