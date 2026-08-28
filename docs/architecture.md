# Harness Engineering CLI — Architecture

## System Overview

```
User
  ↓
CLI (typer)
  ↓
AgentLoop ←── ContextEngine ←── FileContentCache
  ↓                 ↓                SearchCache
ModelRouter ←── RepositoryAnalyzer
  ↓                 ↓
ProviderAdapter  RelevanceRanker
  ├── OpenRouter  SymbolIndex
  ├── LiteLLM     DependencyGraph
  └── Ollama      ContextPackBuilder
  ↓
ToolRuntime ←── ParallelToolExecutor
  ↓             ToolCallDeduplicator
PermissionManager
  ↓
VerificationEngine
  ↓
EventBus ←── MetricsCollector
```

## Core Subsystems

### Agent Runtime (`src/harness_core/agent/`)
- **loop.py** — The core engine. Manages the understand→plan→execute→verify loop.
- **types.py** — Task, ToolCall, AgentConfig, status enums.

### Tool System (`src/harness_core/tools/`)
- **base.py** — Tool, ToolSchema, ToolResult abstractions
- **filesystem.py** — read_file, write_file, edit_file, list_files
- **search.py** — glob, grep
- **shell.py** — run_command
- **git.py** — git_status, git_diff, git_log, git_commit
- **parallel.py** — ParallelToolExecutor, ToolCallDeduplicator

### Model Routing (`src/harness_core/routing/`)
- **router.py** — ModelRouter (selects model based on scoring + health)
- **scoring.py** — 8-dimension weighted scoring (capability, task_fit, tools, context, cost, reliability, latency, availability)
- **health.py** — ModelHealthTracker (success/429/4xx/5xx/timeout tracking)
- **fallback.py** — FallbackEngine with retry, backoff, jitter
- **budgets.py** — BudgetManager (iterations, tool calls, tokens, cost, timeout)

### Provider System (`src/harness_core/providers/`)
- **base.py** — ModelProvider interface (generate, stream, list_models, health_check)
- **openrouter.py** — OpenRouter multi-model gateway
- **ollama.py** — Ollama local inference
- **litellm.py** — LiteLLM unified abstraction

### Context Intelligence (`src/harness_core/context/`)
- **engine.py** — ContextEngine (project discovery, token budgeting, context assembly)
- **pack.py** — ContextPackBuilder, ContextPack, ContextPiece, estimate_tokens()
- **compaction.py** — Session compaction for long-running agent sessions

### Caching (`src/harness_core/cache/`)
- **file_cache.py** — FileContentCache (LRU, metadata tracking, secret detection, invalidation)
- **search_cache.py** — SearchCache (key-based glob/grep/repo caching)

### Analysis (`src/harness_core/analysis/`)
- **repository.py** — RepositoryAnalyzer, RepositoryInfo, Ecosystem detection
- **relevance.py** — RelevanceRanker, RelevanceScore, RelevanceConfig

### Indexing (`src/harness_core/indexing/`)
- **symbols.py** — SymbolIndex (regex-based Python/JS/TS/Rust/Go parser)
- **dependency_graph.py** — DependencyGraph (import graph, cycle detection, transitive reachability)

### Verification (`src/harness_core/verification/`)
- **engine.py** — VerificationEngine (ecosystem detection, test/lint/typecheck/build)

### Permissions (`src/harness_core/permissions/`)
- **manager.py** — PermissionManager (allow/ask/deny, workspace sandbox, protected paths)

### Observability (`src/harness_core/observability/`)
- **events.py** — EventBus (decoupled event system)
- **metrics.py** — MetricsCollector (counters, gauges, timers, dashboard)

### CLI (`src/harness_core/cli/`)
- **main.py** — CLI commands: init, run, doctor, models, status, config

## Agent Loop

```
GOAL
  ↓
UNDERSTAND (task parsing, intent detection)
  ↓
PLAN (strategy selection, file identification)
  ↓
CONTEXT (ContextEngine: cache → analyze → rank → index → pack)
  ↓
SELECT MODEL (ModelRouter: scoring + health + fallback)
  ↓
EXECUTE (ToolRuntime: permission check → parallel/dedup → tool call)
  ↓
OBSERVE (result parsing, tool call dedup, cache update)
  ↓
VERIFY (VerificationEngine: tests, lint, build, typecheck)
  ↓
EVALUATE (success criteria check)
  ↓
SUCCESS?
  ├── YES → DELIVER (result, metrics, trace)
  └── NO  → DIAGNOSE → REPLAN → EXECUTE
```

## Context Intelligence Pipeline

```
Repository
  ↓
RepositoryAnalyzer (languages, structure, ecosystems)
  ↓
RelevanceRanker (score files by task fit)
  ↓
SymbolIndex (function/class/import lookup)
  ↓
DependencyGraph (import relationships)
  ↓
ContextPackBuilder (token-budget assembly + dedup)
  ↓
ContextPack (smallest useful context)
```

## Caching Architecture

```
read_file("src/foo.py")
  ↓
FileContentCache.get(path)
  ├── Cache HIT → return immediately
  └── Cache MISS → disk read → cache → return

grep("authentication")
  ↓
SearchCache.get(key)
  ├── Cache HIT → return cached results
  └── Cache MISS → execute search → cache → return
```

## Parallel Execution

```
Agent requests: read(A), read(B), grep(C), read(D)
  ↓
ToolCallDeduplicator
  ├── Dedup identical calls
  └── Filter already-cached results
  ↓
ParallelToolExecutor
  ├── Permission check (serialized)
  ├── Concurrent reads: read(A), read(B), grep(C), read(D)
  └── Collect results (deterministic ordering)
```

## Provider Architecture

```
Agent
  ↓
ModelRouter
  ├── Score models by: capability, task_fit, tools, context, cost, reliability
  ├── Check health (429/4xx/5xx tracking)
  ├── Apply fallback chain
  └── Select best available model
  ↓
ProviderAdapter
  ├── OpenRouter (multi-model gateway)
  ├── LiteLLM (unified abstraction)
  └── Ollama (local inference)
```

## Technology Stack

- **Language**: Python 3.12+
- **Package Manager**: uv
- **CLI Framework**: typer
- **Async**: asyncio
- **Testing**: pytest + pytest-asyncio
- **Type Checking**: pyright
- **Linting**: ruff

## Directory Structure

```
harness-engineering-cli/
├── src/harness_core/
│   ├── agent/           # Core agent loop and types
│   ├── tools/           # Tool system (10 tools + parallel + dedup)
│   ├── routing/         # Model router, scoring, health, fallback, budgets
│   ├── providers/       # Model providers (OpenRouter, Ollama, LiteLLM)
│   ├── context/         # Context engine, pack builder, compaction
│   ├── cache/           # File and search caching
│   ├── analysis/        # Repository analysis and relevance ranking
│   ├── indexing/        # Symbol index and dependency graph
│   ├── verification/    # Test/lint/build verification
│   ├── permissions/     # Permission system and sandbox
│   ├── observability/   # Events and metrics
│   └── cli/             # CLI commands
├── tests/
│   ├── unit/            # 280 unit tests
│   └── benchmarks/      # 10 performance benchmarks
├── docs/                # Architecture, vision, security, routing docs
├── pyproject.toml
└── README.md
```
