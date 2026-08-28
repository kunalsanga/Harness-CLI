# Model Intelligence

Model Intelligence is Harness's evidence-based system for selecting the right model for each engineering task.

## Core Principle

**Model availability ≠ model capability.**

A model being available through OpenRouter or installed in Ollama does NOT mean it is a strong autonomous coding agent. Harness builds evidence-based model intelligence to make the right choice.

## Architecture

```
USER TASK
    │
    ▼
TASK CLASSIFIER (deterministic heuristics, no LLM)
    │
    ▼
TASK REQUIREMENTS (what capabilities the task needs)
    │
    ▼
MODEL REGISTRY (centralized, provider-agnostic)
    │
    ├── BENCHMARK DATA (Harness-measured scores)
    │
    ├── LIVE HEALTH (real-time reliability/latency)
    │
    └── HISTORICAL PERFORMANCE (SQLite-backed with time decay)
            │
            ▼
      MODEL ROUTER (task-aware, multi-dimensional scoring)
            │
            ▼
       BEST MODEL SELECTION + Explanation
```

## Components

### ModelRegistry

Centralized, thread-safe model registry. Provider-agnostic.

- `register(profile)` — Register or update a model
- `get(model_id)` — Look up by ID
- `search(...)` — Filter by provider, free, tools, context, tags
- `update_capabilities(...)` — Set capability scores with provenance
- `update_health(...)` — Update operational health (reliability, latency)
- `summary()` — Overview stats

### ModelProfile

Complete profile combining provider metadata + capability evidence + operational data.

**Four evidence sources:**

1. Provider metadata (what the provider says)
2. Declared capabilities (explicitly configured)
3. Observed capabilities (Harness has seen it work)
4. Benchmarked capabilities (measured by Harness benchmarks)

**Unknown ≠ zero.** `coding_score = None` means "not measured," not "terrible."

### CapabilityConfidence

Four confidence levels:

- `UNKNOWN` — No data at all
- `DECLARED` — From provider metadata
- `OBSERVED` — Harness has seen it work in practice
- `BENCHMARKED` — Measured by Harness benchmark engine

### TaskClassifier

Fast deterministic heuristic classifier. No LLM call. Latency matters.

**Categories:** implementation, bug_fix, debugging, refactoring, testing, research, repository_analysis, documentation, security, performance

Uses keyword/pattern matching. Designed so a more sophisticated classifier can replace it later.

### TaskRequirementProfile

What a task requires from a model. Feeds into task-aware routing.

**compute_fit(model_capabilities):** Returns 0.0–1.0 scoring how well a model matches the task.

### Benchmark Engine

Controlled benchmarks in isolated temporary workspaces. Never touches user's project.

**Categories:** tool_use, navigation, coding, debugging, recovery, context, planning, verification

**Scoring weights (configurable):**

| Dimension | Weight |
|-----------|--------|
| coding | 25% |
| tool_use | 15% |
| reasoning | 15% |
| planning | 10% |
| repository_navigation | 10% |
| context_handling | 10% |
| error_recovery | 10% |
| instruction_following | 3% |
| verification | 2% |

### PerformanceHistory

SQLite-backed persistent model performance tracking. Supports configurable time decay.

Records: model, provider, task type, success/failure, latency, tokens, tool calls, iterations.

### Discovery

Normalizes provider ModelInfo into ModelProfile. Supports heuristic capability estimation from model naming conventions.

## CLI Commands

```bash
harness models list --free         # List free models
harness models recommend --task "Fix authentication bug"
harness models inspect <model>     # Detailed model profile
harness models compare <A> <B>     # Side-by-side comparison
harness models benchmark           # Show available benchmarks
harness models local               # List Ollama local models
harness models history             # Performance history
```

## Free Model Intelligence

The system distinguishes: free, paid, local, unknown.

Free model selection pipeline:

```
provider discovery
    ↓
free filter
    ↓
tool-capability filter
    ↓
context filter
    ↓
capability evidence
    ↓
health
    ↓
latency
    ↓
task fit
    ↓
best free model
```

Do NOT assume every `:free` model is a good coding agent.

## Security

- Benchmarks run in isolated temporary workspaces
- Never access user's real repository during benchmarks
- Never expose .env, credentials, private keys to benchmark tasks
- Permission system fully enforced

## Known Limitations

1. **Regex-based classifier** — Not perfect. Sufficient for routing.
2. **Heuristic capability estimation** — Provider metadata is rough. Benchmarks provide truth.
3. **No real benchmark execution yet** — Engine exists but benchmarks require real provider calls
4. **SQLite only for history** — Suitable for single-user. Multi-user persistence deferred.
