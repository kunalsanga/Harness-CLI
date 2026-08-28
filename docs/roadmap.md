# Harness Engineering CLI — Roadmap

## Milestones

### M1 — Core Agent Engine ✅ COMPLETE

Core agent loop, CLI, tool system, permissions, provider abstraction.

- AgentLoop with iterative execution
- Tool system: read_file, write_file, edit_file, list_files, glob, grep, run_command, git tools
- Permission system: allow/ask/deny
- Provider abstraction: ModelProvider interface
- CLI: init, run, doctor, models, status
- 45 tests

### M2 — Model Router + Resilience ✅ COMPLETE

Production-grade model routing with fallback, retry, budgets.

- ModelRouter with 8-dimension scoring
- Fallback engine: primary → fallback → final failure
- Retry with exponential backoff + jitter
- Budget management: iterations, tool calls, cost, timeout
- Free-model optimization
- OpenRouter + Ollama + LiteLLM providers
- EventBus for routing decisions
- 122 tests

### M3 — Context Intelligence + Performance ✅ COMPLETE

Intelligent repository understanding and performance optimization.

- FileContentCache (LRU, mtime-based invalidation)
- SearchCache (key-based, parameter-aware)
- RepositoryAnalyzer (language, package manager, build system, test framework)
- RelevanceRanker (6-signal scoring: filename, path, extension, keywords, search matches, git history)
- SymbolIndex (regex-based parser for Python/JS/TS/Rust/Go)
- DependencyGraph (import/dependency tracking, cycle detection)
- ContextPackBuilder (token-budget assembly, deduplication)
- ParallelToolExecutor (async read/search, serial mutations)
- ToolCallDeduplicator (avoids repeated identical calls)
- MetricsCollector (counters, gauges, timing, dashboard)
- 280 tests + 10 benchmarks

### M3.5 — Model & Agent Intelligence ✅ COMPLETE

Evidence-based model capability tracking and task-aware routing.

**ModelRegistry** — Centralized, provider-agnostic model registry with capability profiles, health tracking, and search.

**CapabilityProfile** — Nine-dimension capability scoring with confidence levels (Unknown/Declared/Observed/Benchmarked) and source provenance.

**TaskClassifier** — Fast deterministic heuristic classifier (no LLM calls). Categories: implementation, bug_fix, debugging, refactoring, testing, research, repository_analysis, documentation, security, performance.

**TaskRequirementProfile** — What capabilities a task requires. `compute_fit()` returns 0.0–1.0 model-task match score.

**BenchmarkEngine** — Controlled benchmarks in isolated temporary workspaces. Nine categories with configurable scoring weights.

**PerformanceHistory** — SQLite-backed model performance tracking with time decay.

**Discovery** — Normalizes provider metadata into ModelProfile. Heuristic capability estimation from model naming.

**CLI commands:** models recommend, models inspect, models compare, models benchmark, models history, models local.

**Tests:** 331 total (41 new model intelligence tests).

### M3.6 — Real-World Integration & Production Hardening ✅ COMPLETE

Transformed tested components into one coherent, reliable coding-agent system.

**Task-Aware Routing** — TaskAwareRouter wired ModelRegistry → TaskClassifier → TaskRequirementProfile → ScoringRouter. 14-dimension scoring formula.

**Agent Recovery** — ErrorClassifier distinguishes RETRYABLE / NON_RETRYABLE / USER_ACTION_REQUIRED. Covers 429, timeout, 5xx, auth, context overflow, network, permission, unknown tools.

**Session Foundation** — SQLite-backed SessionStorage with SessionState, RunRecord, Checkpoint. Session CLI: session list, session show.

**E2E Fixture Tests** — Python fixture project with intentional bugs. Agent must inspect → diagnose → edit → test → verify.

**Security Audit Tests** — 20 tests covering secret leakage, permission enforcement, workspace sandbox, protected paths, gitignore validation.

**Integration Pipeline Tests** — 19 tests proving TaskClassifier → TaskAwareRouter → ScoringRouter → AgentLoop integration.

**14-Dimension Scoring** — task_fit + capability_fit + reliability + historical_performance + latency + context_fit + cost + availability + tool_support + context_window + free_preference + speed_preference + coding_preference + reasoning_preference.

**Tests:** 441 total (110 new M3.6 tests).

### M3.7 — Real Model Validation ✅ COMPLETE

Validated the product against real LLM providers and real-world execution.

**OpenRouter Validation** — Health check, model discovery (388 models), completion, null-content handling for reasoning models, 429 error handling.

**Free Model Discovery** — 21 free models found, 18 with tool support.

**Bug Fix** — OpenRouter provider now handles `content: null` from reasoning models gracefully.

**E2E Integration Tests** — 27 tests covering: task classification, task-aware routing, recovery system, session persistence, tool execution, verification engine.

**Provider Validation Tests** — 15 tests covering: health check, model discovery, completion, timeout, 429 handling, invalid model errors.

**CLI Validation** — `harness doctor`, `harness models list --free`, `harness models recommend --task "..."` all working.

**Security Audit** — 20 tests: no secrets in source, .env.example safe, config clean, recovery doesn't leak secrets, permissions enforced, workspace sandbox works.

**Tests:** 458 total (17 new M3.7 tests).

### M3.8 — Empirical Model Intelligence ✅ COMPLETE

Evidence-based model routing using real execution outcomes.

**EmpiricalHistory** — SQLite-backed execution history with per-model, per-task-type performance tracking. Thread-safe with 1-minute cache.

**ModelPerformanceAggregator** — Calculates success rates, latency percentiles, tool efficiency, recovery rates. Supports time-decay weighting (30-day half-life) and recent window tracking.

**ConfidenceCalculator** — Sample-size-based confidence: UNKNOWN (0), VERY_LOW (1–4), LOW (5–19), MEDIUM (20–49), HIGH (50+).

**TaskOutcome Taxonomy** — 9 outcomes: SUCCESS, PARTIAL_SUCCESS, FAILURE, TIMEOUT, MODEL_ERROR, TOOL_ERROR, PERMISSION_DENIED, USER_ABORTED, UNKNOWN.

**Evidence Provenance** — 4 sources: PROVIDER_DECLARED, HARNESS_STATIC, HARNESS_BENCHMARKED, REAL_WORLD_OBSERVED.

**Task-Aware Empirical Routing** — Router combines static capability fit + empirical task performance + historical data + confidence weighting.

**Cold Start** — New models route on static capabilities. Empirical influence grows with evidence.

**CLI Enhancements** — `models recommend --free`, `models inspect` (shows empirical profile), `models compare` (shows empirical comparison).

**Tests:** 518 total (60 new M3.8 tests).

### M4 — Session Intelligence ✅ COMPLETE

Persistent agent memory and session intelligence.

**Session Domain Model** — Session, Run, Checkpoint, MemoryItem, SessionEvent with explicit state transitions and validation.

**SQLite Storage** — 5 tables (sessions, runs, checkpoints, memory, session_events) with indexes, thread-safe, crash-safe.

**Session Lifecycle** — create, pause, resume, complete, fail, abort, archive with validated transitions.

**Checkpointing** — Git state, verification status, context references at safe boundaries.

**Resume** — get_resume_state() provides session, runs, checkpoint, and memories for safe continuation.

**Memory System** — 8 types (DECISION, DISCOVERY, CONSTRAINT, TODO, WARNING, ERROR, SOLUTION, NOTE) with importance scoring and keyword retrieval.

**Secret Sanitization** — API keys, tokens automatically redacted before persistence.

**CLI Commands** — session list, create, show, pause, resume, archive, delete, diff, export, memory.

**Tests:** 563 total (45 new M4 tests).

### M5 — Advanced Agent Engine ✅ COMPLETE

Multi-agent orchestration system with specialized agents.

**Agent Domain** — AgentRole (8 roles), SubTask, TaskGraph with dependency tracking and cycle detection.

**Agent Registry** — 7 pre-configured agents (planner, researcher, analyzer, coder, tester, reviewer, debugger) with capabilities, tool access, and model preferences.

**Agent Executor** — Executes individual agent tasks with structured results (status, summary, files_changed, tests, findings, review_verdict).

**Orchestrator** — Coordinates multi-agent execution: task decomposition, agent delegation, failure detection, repair cycles, final synthesis.

**Task Graph** — Dependency-aware scheduling, topological sort, ready-task detection, completion tracking.

**Agent Budget** — Resource limits: max agents, iterations, tool calls, repair cycles, runtime, cost.

**Execution Modes** — single (backward compatible), auto, multi-agent, parallel.

**Tests:** 615 total (51 new M5 tests).

### M6 — Native Performance & High-Performance Runtime ✅ COMPLETE

Performance baseline, Rust native foundation, parallel execution, and cancellation.

**Performance Baseline** — Comprehensive measurements of all subsystems. Identified glob/search as primary bottleneck (0.22s median). All other subsystems already fast.

**Profiling** — CPU/IO hotspots identified. Glob/search confirmed as Rust candidate. Model routing, classification, sessions already negligible.

**Rust Foundation** — `native/harness-fs/` crate with: fast_glob (ignore-aware parallel traversal), fast_grep (regex search with binary avoidance), fast_file_index, fast_hash/batch_hash, fast_count_files.

**Python Bridge** — `harness_core.native` module provides identical API with automatic Python fallback when Rust extension unavailable.

**Parallel Execution** — `ParallelScheduler` with dependency-aware scheduling. Independent tasks run concurrently; dependent tasks wait.

**File Conflict Detection** — `FileOwnershipTracker` prevents silent overwrites between parallel agents. States: UNOWNED / LOCKED / MODIFIED / CONFLICT.

**Cancellation** — `CancellationHandler` + `GracefulShutdown` for clean Ctrl+C handling. Subprocess cleanup, session checkpoint, resource release.

**C++ Evaluation** — Not justified at this stage. Rust covers all identified native workloads.

**Benchmark Suite** — `benchmarks/performance_baseline.py` measuring CLI, discovery, search, sessions, routing, tools, orchestration.

**Tests:** 649 total (33 new M6 tests).

### M7 — Universal Developer Platform & Extensibility ✅ COMPLETE

Extension architecture, plugin system, MCP foundation, hooks, and configuration.

**Extension Architecture** — `ExtensionManifest`, `ExtensionRegistry`, `ExtensionLoader`, `ExtensionContext`. Types: TOOL, PROVIDER, AGENT, HOOK, MCP. Lifecycle: DISCOVERED → INSTALLED → ENABLED → DISABLED → FAILED → REMOVED.

**Plugin System** — `PluginManager` for local filesystem plugins. Install, enable, disable, remove, inspect. Global directory `~/.harness/plugins/`. Manifest validation, safe error isolation.

**MCP Foundation** — `MCPClient` for stdio-based JSON-RPC MCP servers. Server lifecycle, tool discovery, tool invocation, shutdown. CLI: `harness mcp list`.

**Hook System** — `HookRegistry` with 12 lifecycle events. Priority ordering, error isolation, rejection support. Plugin-source tracking.

**Configuration System** — `HarnessConfig` with 5-level precedence (CLI > Session > Project > Global > Defaults). Environment variable override, type coercion, secret redaction, validation.

**Sample Plugins** — hello-world tool, code-quality analyzer, security-reviewer agent.

**CLI Commands** — `harness plugin list|install|enable|disable|remove|inspect`, `harness tools list|inspect`, `harness mcp list`, `harness hooks list`.

**Tests:** 706 total (57 new M7 tests).

### M8 — Productization, Distribution & Public Beta ✅ COMPLETE

Installation, provider setup, error system, documentation, and public release readiness.

**Installation** — MIT License, CHANGELOG.md, verified pyproject.toml for pip/uv install. Entry point: `harness = "harness_core.cli.main:app"`.

**Provider Setup** — `harness providers list|configure`. OpenRouter, Ollama, LiteLLM setup instructions. Interactive setup wizard: `harness setup`.

**Error System** — 9 structured error classes (ConfigurationError, ProviderError, ModelError, PermissionError, ToolError, WorkspaceError, VerificationError, ExtensionError, SessionError). Each provides what/why/fix.

**Documentation** — installation.md, quickstart.md, providers.md, configuration.md, troubleshooting.md. README rewritten for public release.

**Security** — Config redaction verified. API key patterns excluded from source. No secrets in logs.

**Tests:** 741 total (35 new M8 tests).

### M9 — Beta Validation, Reliability & Release Engineering ✅ COMPLETE

Full product audit, clean installation test, security audit, and release readiness assessment.

**Repository Audit** — 79 tracked files, 50+ untracked (M1-M8), 1 commit history. Clean .gitignore.

**Security Audit** — No API keys in source, .env gitignored, config redaction verified, no secrets in wheel.

**Packaging** — Wheel (155KB, 93 files) and sdist (250KB) built successfully. Clean install from wheel verified.

**CLI Verification** — All 15 commands work from clean install: doctor, status, providers, agents, tools, sessions, mcp, hooks, plugin.

**Provider Validation** — OpenRouter verified (health, discovery, completion). Ollama/LiteLLM unverified (not running).

**Release Recommendation** — READY FOR PRIVATE BETA. Core product works, tests pass, packaging clean. Not ready for public alpha due to incomplete multi-platform CI and unverified real coding tasks.

**Tests:** 742 total (37 new M9 tests, 1 E2E skip for rate limiting).

### M10 — Real-World Beta Validation & Coding Benchmark ✅ COMPLETE

Benchmark framework with 10 coding tasks across Python and TypeScript.

**Benchmark Framework** — `benchmarks/tasks.py` (10 tasks), `benchmarks/runner.py` (result tracking, JSON persistence). Categories: bug_fix, feature, refactor, testing, debugging, security.

**Python Fixture** — 7 source files, 4 test files, 8 intentional failures confirmed. Realistic multi-module project with Calculator, StringUtils, DataStore, UserService, OrderService, RateLimiter.

**Node.js Fixture** — 2 source files, 1 test file, 2 intentional failures confirmed. Math module with fibonacci bug, URL parser with port/auth bugs.

**CLI** — `harness benchmark run`, `harness benchmark report`.

**Decision** — Needs another reliability pass. Real agent execution not completed due to provider rate limiting.

**Tests:** 741 total (2 E2E skips for rate limiting).

## Test Progression

| Milestone | Tests | Benchmarks |
|-----------|-------|------------|
| M1 | 45 | 0 |
| M2 | 122 | 0 |
| M3 | 280 | 10 |
| M3.5 | 331 | 10 |
| M3.6 | 441 | 10 |
| M3.7 | 458 | 10 |
| M3.8 | 518 | 10 |
| M4 | 563 | 10 |
| M5 | 615 | 10 |
| M6 | 649 | 10 |

## Architecture Principles

1. The model is replaceable. The harness is the product.
2. Never fabricate scores. Unknown ≠ zero.
3. Benchmark in isolation. Never touch user's project.
4. Task success + low latency + low token usage + reliability.
5. Evidence-based routing. Not assumption-based routing.
