# Harness Engineering CLI — Model Routing

## Architecture

```
User Request
    ↓
ModelRouter
    ↓
┌─────────────────────────────────────┐
│  1. Refresh model catalog           │
│  2. Filter by requirements          │
│  3. Score each model                │
│  4. Rank by weighted score          │
│  5. Build fallback chain            │
└──────────────┬──────────────────────┘
               ↓
         FallbackEngine
               ↓
    ┌──────────┴──────────┐
    │  Try Model A        │
    │  ↓ 429 / timeout    │
    │  Try Model B        │
    │  ↓ failure          │
    │  Try Model C        │
    │  ↓ success          │
    │  Return result      │
    └─────────────────────┘
               ↓
      Health Tracker (records outcomes)
      Budget Manager (enforces limits)
      EventBus (emits routing.decision events)
```

## Scoring System

Each model is scored on 8 dimensions, each returning [0.0, 1.0]:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `capability` | 0.20 | Context window size (log scale: 8K→0.2, 1M+→1.0) |
| `task_fit` | 0.15 | Alignment between model tags and task type |
| `tool_support` | 0.15 | Whether model supports function calling |
| `context_fit` | 0.10 | Whether model's context window fits the task |
| `cost` | 0.10 | Lower cost = higher score (free = 1.0) |
| `reliability` | 0.15 | Historical success rate (tracked per model) |
| `latency` | 0.05 | Lower latency = higher score |
| `free_bonus` | 0.10 | Bonus for free models when `prefer_free=True` |

Weights are configurable via `.harness/config.yaml`:

```yaml
routing:
  scoring_weights:
    capability: 0.25
    cost: 0.15
    reliability: 0.20
```

## Routing Modes

```
auto     → Best overall model (weighted scoring)
free     → Free models only
best     → Highest capability
fast     → Lowest latency
local    → Local models only (Ollama)
cheap    → Lowest cost
coding   → Coding-optimized models
reasoning → Reasoning-optimized models
```

## Health Tracking

Each model's health state is tracked with:

- **success** / **failure** counts
- **rate_limit_hits** (429)
- **timeouts**
- **consecutive_failures**
- **avg_latency_ms** (rolling window of 50)
- **reliability** (successes / total_calls)
- **cooldown_seconds** (exponential backoff after rate limits)

Models with `consecutive_failures >= 5` or active rate-limit cooldown are excluded from routing.

## Fallback Engine

### Chain Construction

1. Score all available models
2. Filter by health, tool support, context window
3. Take top N (default: 4) as fallback chain
4. Execute through FallbackEngine

### Retry Logic

- **Retryable errors** (5xx, timeout, network): retry with exponential backoff
- **Rate limited** (429): skip to next model immediately
- **Permanent errors** (401, 403, 404): don't retry, move to next model
- **Context overflow**: move to larger-context model

### Backoff Configuration

```python
RetryConfig(
    max_retries=2,           # retries per model before fallback
    base_delay_seconds=1.0,  # initial delay
    backoff_factor=2.0,      # exponential multiplier
    max_delay_seconds=30.0,  # cap
    jitter=True,             # random 50-100% of calculated delay
)
```

## Budget System

Enforced per-task:

| Budget | Default | Description |
|--------|---------|-------------|
| `max_iterations` | 30 | Maximum agent loop iterations |
| `max_tool_calls` | 100 | Maximum tool executions |
| `max_tokens` | 500,000 | Maximum total tokens |
| `max_cost` | $5.00 | Maximum total cost |
| `timeout_seconds` | 600 | Maximum wall-clock time |
| `max_calls_per_model` | None | Per-model call limit |
| `max_cost_per_model` | None | Per-model cost limit |

Configuration via `.harness/config.yaml`:

```yaml
budgets:
  max_iterations: 30
  max_tool_calls: 100
  max_cost_per_task: 2.0
  timeout_seconds: 300
```

## Providers

### Supported Providers

| Provider | Status | Notes |
|----------|--------|-------|
| OpenRouter | ✅ Active | Primary multi-model gateway |
| Ollama | ✅ Active | Local inference |
| LiteLLM | ✅ Added | Unified interface to 100+ providers |

### Adding a Provider

Implement `ModelProvider` interface:

```python
class MyProvider(ModelProvider):
    @property
    def name(self) -> str: ...
    async def generate(self, request: CompletionRequest) -> CompletionResponse: ...
    async def stream(self, request: CompletionRequest): ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def health_check(self) -> bool: ...
```

## Events

Routing emits structured events for observability:

```
routing.decision       → model selected, score, alternatives
routing.models_refreshed → model catalog refresh count
model.error            → provider failure details
```

## CLI Usage

```bash
# Auto routing (default)
harness run "Fix the bug"

# Free models only
harness run --mode free "Fix the bug"

# With budget
harness run --max-cost 1.0 --max-iterations 10 "Fix the bug"

# Specific model
harness run --model "deepseek/deepseek-v4-flash" "Fix the bug"

# Headless (CI/CD)
harness run --headless --json "Fix the bug"
```
