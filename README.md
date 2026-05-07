# AgentLean

[![PyPI version](https://badge.fury.io/py/agentlean.svg)](https://badge.fury.io/py/agentlean)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/agentlean/agentlean/actions/workflows/test.yml/badge.svg)](https://github.com/agentlean/agentlean/actions)

**Cut your AI agent costs by 40–80% with one line of code.**

AgentLean is a transparent middleware layer that sits between your agent orchestration code and LLM APIs (Anthropic and OpenAI). It automatically optimises the context before each API call — no changes to your agent logic required.

---

## The problem

In multi-step agent loops, **input token costs dominate**. On every iteration, your agent re-sends the full conversation history plus raw tool outputs (web page HTML, JSON API dumps, search results). A 10-step research agent run can easily rack up 80,000+ input tokens per call.

**Without AgentLean — a 10-step research agent on Claude Sonnet:**

| Call | Input tokens | Cost     |
|------|-------------|----------|
| 1    | 2,100       | $0.006   |
| 2    | 9,400       | $0.028   |
| 3    | 18,200      | $0.055   |
| ...  | ...         | ...      |
| 10   | 84,200      | $0.253   |
| **Total** | **~420,000** | **~$1.26** |

**With AgentLean (balanced strategy):**

| Call | Original | Optimised | Savings  |
|------|----------|-----------|----------|
| 1    | 2,100    | 2,100     | 0%       |
| 2    | 9,400    | 4,200     | 55%      |
| 3    | 18,200   | 7,100     | 61%      |
| ...  | ...      | ...       | ...      |
| 10   | 84,200   | 28,400    | 66%      |
| **Total** | **~420,000** | **~156,000** | **63% → ~$0.47** |

**Savings: ~$0.79 per run** — on a busy agent making 100 runs/day, that's **~$2,400/month**.

---

## Quickstart

```bash
pip install agentlean
# For Anthropic support:
pip install agentlean[anthropic]
# For OpenAI support:
pip install agentlean[openai]
# Everything:
pip install agentlean[all]
```

```python
import anthropic
from agentlean import AgentLean

client = anthropic.Anthropic()
lean = AgentLean(client, provider="anthropic")

# Use lean.messages.create() exactly like client.messages.create()
response = lean.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=messages,
)

print(lean.stats)
# AgentLeanStats(original_tokens=84,230, optimised_tokens=31,450, saved_pct=62.7, estimated_cost_saved=0.0380)
```

**OpenAI:**

```python
import openai
from agentlean import AgentLean

client = openai.OpenAI()
lean = AgentLean(client, provider="openai")

response = lean.chat.completions.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=messages,
)
```

---

## Features

### 1. Sliding Context Window with Decay

Older conversation turns are progressively compressed so recent context stays fresh without bloating input tokens.

- **Recent turns** (configurable, default: last 5) → kept verbatim
- **Middle turns** (up to turn 15) → summarised to key facts
- **Far past** (beyond turn 15) → dropped
- The first user message (original task) and system prompt are always preserved

### 2. Tool Output Distillation

Raw tool outputs (web search HTML, JSON API responses, large text dumps) are automatically trimmed before they re-enter context.

- **HTML content**: strips tags, nav, headers, footers, ads — keeps only body text
- **JSON responses**: extracts only keys relevant to the conversation query
- **Raw text**: sentence-boundary truncation (never cuts mid-word)
- Each distilled result gets a metadata stub: `[Tool output distilled from ~8,200 to ~480 tokens]`

### 3. System Prompt Optimisation (Advisory)

```python
analysis = lean.analyse_system_prompt(your_system_prompt)
print(analysis)
# SystemPromptAnalysis(tokens=2,847, suggestions=4 [1 high / 2 medium], potential_savings≈340 tokens)

for suggestion in analysis.suggestions:
    print(f"[{suggestion.severity}] {suggestion.description}")
```

Detects: redundant sentences, overly verbose lists, contradictory instructions, filler phrases, excessive examples. **Never auto-applies changes** — returns suggestions for your review.

### 4. Token Budget Manager

```python
from agentlean import AgentLean, AgentLeanConfig

lean = AgentLean(
    client,
    provider="anthropic",
    budget_usd=0.50,    # Hard cap at $0.50
    # or: budget_tokens=200_000
)
```

- Tracks cumulative spend across the session
- Warns at 70% and 90% of budget
- Automatically escalates compression as budget runs low
- Raises `BudgetExhaustedError` at 100% (configurable)

### 5. Observability

```python
# Per-session stats
print(lean.stats.to_dict())
{
    "total_calls": 10,
    "original_input_tokens": 420000,
    "optimised_input_tokens": 156000,
    "saved_pct": 62.9,
    "total_cost_usd": 0.468,
    "total_saved_usd": 0.792,
    "tool_outputs_distilled": 18,
    "turns_summarised": 45,
    "turns_dropped": 12,
}

# Callback for every call (useful for logging/monitoring)
config = AgentLeanConfig(on_call_complete=lambda stats: logger.info(stats.to_dict()))
```

---

## Configuration

```python
from agentlean import AgentLean, AgentLeanConfig

config = AgentLeanConfig(
    strategy="balanced",           # "conservative" | "balanced" | "aggressive"
    context_window_turns=5,        # Full-fidelity recent turns
    summarise_turns_up_to=15,      # Summarise up to this many older turns
    max_tool_output_tokens=500,    # Token cap per tool result
    preserve_system_prompt=True,   # Never compress system prompt
    preserve_first_message=True,   # Always keep original task message
    budget_usd=None,               # Optional USD spend cap
    budget_tokens=None,            # Optional token cap
    summarisation_model="claude-haiku-4-5-20251001",  # Cheap model for summaries
    warn_at_budget_pct=0.70,       # Warn at 70% budget usage
    hard_stop_at_budget=True,      # Raise error at 100%
)

lean = AgentLean(client, config=config)
```

### Strategy presets

| Strategy     | What it does                                  | Typical savings |
|-------------|-----------------------------------------------|-----------------|
| `conservative` | Only truncate oversized tool outputs        | 20–30%          |
| `balanced`     | Truncate + sliding window + summarisation   | 40–60%          |
| `aggressive`   | Heavy compression, smaller window           | 60–80%          |

```python
# Shorthand preset constructors
config = AgentLeanConfig.conservative()
config = AgentLeanConfig.balanced()
config = AgentLeanConfig.aggressive()
```

### Functional API (no client wrapping)

```python
from agentlean import optimise_messages

optimised = optimise_messages(
    messages=messages,
    system=system_prompt,
    strategy="balanced",
    max_context_tokens=16_000,
    model="claude-sonnet-4-6",
)
```

---

## How it works

Every call to `lean.messages.create()` (or `lean.chat.completions.create()`) passes through the optimisation pipeline **before** hitting the API:

```
Your Agent Code
      │
      ▼
AgentLean Wrapper
  ┌───────────────────────────────────┐
  │ 1. Count original input tokens    │
  │ 2. Distill tool outputs           │
  │    ├─ HTML → strip boilerplate    │
  │    ├─ JSON → extract relevant     │
  │    └─ Text → sentence truncation  │
  │ 3. Apply sliding context window   │
  │    ├─ Recent N turns: verbatim    │
  │    ├─ Middle turns: summarise     │
  │    └─ Old turns: drop             │
  │ 4. Check / escalate budget        │
  │ 5. Count optimised tokens         │
  └───────────────────────────────────┘
      │
      ▼
  LLM API (Anthropic / OpenAI)
      │
      ▼
  Response (passed through unchanged)
      │
      ▼
  Update stats & budget
```

**AgentLean is safe by design**: if any optimisation step throws an exception, it logs a warning and passes through the original unmodified messages. Your agent run never fails because of AgentLean.

---

## Installation options

```bash
# Minimal (no provider SDKs — use the functional API only)
pip install agentlean

# With Anthropic support
pip install agentlean[anthropic]

# With OpenAI support
pip install agentlean[openai]

# With accurate OpenAI token counting (tiktoken)
pip install agentlean[tokenizers]

# Everything
pip install agentlean[all]
```

> **Token counting note**: For OpenAI models, AgentLean uses `tiktoken` for accurate counts (install with `agentlean[tokenizers]`). For Anthropic models, a character-based approximation (chars ÷ 3.5) is used — accurate within ~10-15%.

---

## Contributing

Contributions welcome! Please open an issue first for major changes.

```bash
git clone https://github.com/agentlean/agentlean
cd agentlean
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
