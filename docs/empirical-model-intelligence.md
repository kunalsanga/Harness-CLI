# Empirical Model Intelligence — M3.8

## Overview

M3.8 transforms Harness model selection from metadata-based to evidence-based routing. The system learns from real execution outcomes and uses empirical performance data to make routing decisions.

## Core Concepts

### Evidence Provenance

Every capability score has clear provenance:

| Source | Meaning |
|--------|---------|
| `PROVIDER_DECLARED` | From OpenRouter/Ollama metadata |
| `HARNESS_STATIC` | From Harness's own analysis |
| `HARNESS_BENCHMARKED` | From controlled benchmark runs |
| `REAL_WORLD_OBSERVED` | From actual agent executions |

### Outcome Taxonomy

Not all failures are equal:

| Outcome | Meaning |
|---------|---------|
| `SUCCESS` | Task completed, verification passed |
| `PARTIAL_SUCCESS` | Useful progress, verification failed |
| `FAILURE` | Task failed |
| `TIMEOUT` | Exceeded configured limits |
| `MODEL_ERROR` | Provider/model failure |
| `TOOL_ERROR` | Tool execution failure |
| `PERMISSION_DENIED` | Security policy blocked |
| `USER_ABORTED` | User stopped execution |

### Sample Confidence

Based on sample size — not made up:

| Samples | Confidence | Weight |
|---------|------------|--------|
| 0 | UNKNOWN | 0.0 |
| 1–4 | VERY_LOW | 0.1 |
| 5–19 | LOW | 0.3 |
| 20–49 | MEDIUM | 0.6 |
| 50+ | HIGH | 1.0 |

**Unknown ≠ Zero.** A model with 0 samples has UNKNOWN confidence, not zero capability.

## Architecture

```
TaskClassifier → TaskType + TaskRequirementProfile
                        ↓
              TaskAwareRouter
              ┌──────────────────────────────┐
              │ Static capability fit        │
              │ Empirical task performance   │
              │ Historical performance       │
              │ Confidence weighting         │
              └──────────────────────────────┘
                        ↓
                  Final Score (0.0–1.0)
                        ↓
              RoutingExplanation (provenance)
```

## Scoring Formula

```python
final_score = capability_fit + empirical_bonus + history_bonus

empirical_bonus = (task_success_rate × confidence_weight - 0.5) × 0.3
history_bonus = (success_rate - 0.5) × 0.1 + (recovery_rate - 0.5) × 0.05
```

Where `confidence_weight` is:
- UNKNOWN: 0.0 (no influence)
- VERY_LOW: 0.1
- LOW: 0.3
- MEDIUM: 0.6
- HIGH: 1.0

## Time Decay

Old performance becomes less influential over time:

```python
weight = exp(-λ × age)
λ = 0.693 / (half_life_days × 86400)
```

Default half-life: 30 days.

### Long-term vs Recent

The system tracks both:
- **Long-term**: All historical performance
- **Recent**: Last N tasks (default 20)

These are reported separately so users can detect performance degradation.

## Task-Specific Performance

Performance is segmented by task type:

| Task Type | Example |
|-----------|---------|
| `implementation` | Build new features |
| `bug_fix` | Fix failing tests |
| `debugging` | Diagnose issues |
| `refactoring` | Restructure code |
| `testing` | Write test suites |
| `research` | Investigate codebase |
| `documentation` | Write docs |
| `repository_analysis` | Understand structure |

A model may be excellent at bug fixes but poor at implementation. Routing uses task-specific performance.

## Cold Start

A new model with 0 historical runs:
- Ranks based on static capabilities + provider metadata
- Clearly shows "No empirical history"
- As evidence accumulates, empirical influence increases

## Data Storage

SQLite-backed (`~/.harness/empirical.db`):

- Thread-safe
- Indexed by model, task_type, timestamp, outcome
- 1-minute in-memory cache
- Stores metadata only — no source code, no API keys, no secrets

## CLI Commands

```bash
# Recommend with empirical data
harness models recommend --task "Fix failing tests"

# Only free models
harness models recommend --free --task "Fix tests"

# Inspect with empirical profile
harness models inspect <model>

# Compare empirical performance
harness models compare <model-a> <model-b>

# View performance history
harness models history
```

## Security

Empirical learning never stores:
- API keys
- Source code
- Passwords
- Environment variables
- Full prompts

Only metadata: task_type, success, latency, tool counts, verification result.

## Known Limitations

1. **Needs real runs** — Empirical intelligence requires actual model executions
2. **Free tier rate limiting** — Free models may be heavily rate-limited
3. **Session resume not implemented** — Storage foundation exists
4. **No Rust/C++** — All Python until profiling proves bottleneck
5. **No exploration** — Always exploits current best (exploration planned for future)

## Files

| File | Purpose |
|------|---------|
| `models/empirical.py` | Core empirical data model, aggregation, storage |
| `routing/task_aware.py` | Bridges classification + registry + empirical into routing |
| `tests/unit/test_empirical.py` | Core empirical tests |
| `tests/unit/test_empirical_routing.py` | Routing integration tests |
