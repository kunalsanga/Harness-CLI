# Harness Engineering CLI — Architecture

## System Overview

```
                    USER TASK
                        │
                        ▼
                    CLI (typer)
                        │
                        ▼
                  AGENT LOOP
                   /    |    \
                  /     |     \
           PLANNER  TOOLS  ROUTER
              |       |       |
              ▼       ▼       ▼
          Context  Tool     Model
          Engine   Runtime  Router
              |       |       |
              ▼       ▼       ▼
          Repository  File   Provider
          Analysis    Cache  Adapter
              |       |       |
              ▼       ▼       ▼
          Symbol    Search   OpenRouter
          Index     Cache    Ollama
              |       |       LiteLLM
              ▼       ▼
          Dependency Parallel
          Graph     Executor
```

## Package Structure

```
src/harness_core/
├── agent/              # Core agent loop
│   ├── loop.py         # AgentLoop — iterative execution
│   └── types.py        # AgentConfig, AgentRole, AgentResult
│
├── analysis/           # Repository intelligence
│   ├── repository.py   # RepositoryAnalyzer (async)
│   └── relevance.py    # RelevanceRanker (6-signal scoring)
│
├── benchmarks/         # Model benchmark engine
│   ├── engine.py       # AgentBenchmarkEngine (isolated workspaces)
│   ├── scoring.py      # BenchmarkScoringWeights, aggregate_results
│   ├── types.py        # BenchmarkTask, BenchmarkResult, BenchmarkSuiteResult
│   └── tasks/          # Built-in benchmark tasks
│       ├── tool_use.py
│       ├── navigation.py
│       ├── coding.py
│       └── debugging.py
│
├── cache/              # Caching layer
│   ├── file_cache.py   # FileContentCache (LRU)
│   └── search_cache.py # SearchCache (key-based)
│
├── classifier/         # Task classification
│   ├── classifier.py   # TaskClassifier (deterministic heuristics)
│   └── types.py        # TaskRequirementProfile + requirement profiles
│
├── cli/                # CLI entry point
│   └── main.py         # typer app with all commands
│
├── context/            # Context engineering
│   ├── engine.py       # ContextEngine
│   ├── pack.py         # ContextPackBuilder (token-budget assembly)
│   └── compaction.py   # ContextCompactor
│
├── indexing/           # Code analysis
│   ├── symbols.py      # SymbolIndex (regex-based parser)
│   └── dependency_graph.py  # DependencyGraph (import graph)
│
├── models/             # Model intelligence
│   ├── types.py        # ModelProfile, CapabilityProfile, CapabilityScore
│   ├── registry.py     # ModelRegistry (centralized, provider-agnostic)
│   ├── capabilities.py # CapabilityWeights, CODING_WEIGHTS, etc.
│   ├── history.py      # PerformanceHistory (SQLite, time decay)
│   └── discovery.py    # Provider → ModelProfile normalization
│
├── observability/      # Observability
│   ├── events.py       # EventBus
│   └── metrics.py      # MetricsCollector
│
├── permissions/        # Permission system
│   └── manager.py      # PermissionManager (allow/ask/deny)
│
├── providers/          # Model providers
│   ├── base.py         # ModelProvider interface, ModelInfo
│   ├── openrouter.py   # OpenRouter provider
│   ├── ollama.py       # Ollama provider
│   └── litellm.py      # LiteLLM provider
│
├── routing/            # Model routing
│   ├── router.py       # ModelRouter
│   ├── scoring.py      # 8-dimension scoring
│   ├── health.py       # Per-model health tracking
│   ├── fallback.py     # Fallback engine with retry
│   └── budgets.py      # Budget enforcement
│
├── tools/              # Tool system
│   ├── base.py         # Tool, ToolResult
│   ├── filesystem.py   # ReadFile, WriteFile, EditFile, ListFiles
│   ├── search.py       # Glob, Grep
│   ├── shell.py        # RunCommand
│   ├── git.py          # GitStatus, GitDiff, GitLog
│   └── parallel.py     # ParallelToolExecutor, ToolCallDeduplicator
│
└── verification/       # Verification engine
    └── engine.py       # VerificationEngine
```

## Key Design Decisions

### 1. Model is Replaceable

The LLM is a reasoning component behind a clean interface. The harness architecture remains stable when models change.

### 2. Evidence-Based Routing

Capability scores have four confidence levels:
- **Unknown** — No data (NOT zero)
- **Declared** — Provider says so
- **Observed** — Harness has seen it work
- **Benchmarked** — Measured by Harness benchmarks

### 3. Task-Aware Selection

TaskClassifier determines what capabilities a task needs. TaskRequirementProfile computes model-task fit.

### 4. Provider Agnosticism

ModelRegistry, ModelRouter, and all routing logic work with any provider. OpenRouter, Ollama, LiteLLM are first-class. Future providers can be added without redesign.

### 5. Caching Strategy

- FileContentCache: LRU, mtime-based invalidation
- SearchCache: parameter-aware, results cached by query
- Both avoid stale data. Both respect file system state.

### 6. Security Model

- Permission system: allow/ask/deny per tool, per agent, per session
- Workspace sandboxing: default boundary at project root
- Protected paths: .env, credentials, private keys
- Secret redaction in tool output
- Benchmarks in isolated temporary workspaces

### 7. Context Intelligence

- Repository analysis detects language, build system, test framework
- Relevance ranking scores files by 6 signals
- Symbol index provides fast navigation
- Dependency graph tracks import relationships
- Context pack assembles token-budget-aware context
- Deduplication prevents duplicate content
- Parallel execution for independent read operations
