# Harness Engineering CLI — Roadmap

## Milestone 1: Core Engine ✅ COMPLETE

**Goal:** Autonomous engineering loop that can inspect, modify, test, and verify.

**Components:**
- Agent runtime with loop (understand → plan → execute → verify)
- Tool system (read, edit, write, shell, search, git — 10 tools)
- Model provider abstraction
- OpenRouter provider
- Ollama provider
- Permission system (allow/ask/deny with workspace sandbox)
- CLI (init, run, doctor, models, status, config)
- 45 unit tests

**Verified:** Agent executes 26 tool calls across 24 iterations on a real repository before hitting free-tier rate limit.

---

## Milestone 2: Model Router + Resilience ✅ COMPLETE

**Goal:** Intelligent model selection with fallback, retry, and cost optimization.

**Components:**
- `ModelRouter` with 8-dimension weighted scoring
- `ModelHealthTracker` (success/429/4xx/5xx/timeout/latency/reliability)
- `FallbackEngine` (primary → fallback → fallback → failure)
- Retry with exponential backoff and jitter
- Error classification (retryable vs permanent vs rate-limited)
- `BudgetManager` (iterations, tool calls, tokens, cost, timeout)
- Free-model optimization (prefer_free, auto-discovery)
- LiteLLM provider
- Configurable routing via `.harness/config.yaml`
- Structured routing events on EventBus
- 77 new tests (122 total)
- Integration test: Model A → 429 → Model B → success with tool call

**Verified:** All 122 tests pass. Fallback chain works end-to-end with mocked providers.

---

## Milestone 3: Context Intelligence + Performance ✅ COMPLETE

**Goal:** Intelligent context selection with caching, indexing, and parallel execution.

**Components:**

### Phase A: Caching
- `FileContentCache` — LRU file content cache with metadata, invalidation, secret detection
- `SearchCache` — cached glob/grep/repo-discovery results with key-based lookup
- 31 cache tests

### Phase B: Repository Analysis
- `RepositoryAnalyzer` — detects languages, ecosystems, build systems, test frameworks, entry points
- `RelevanceRanker` — 6-signal weighted scoring (filename, path, extension, search, importance, test proximity)
- Extensible ecosystem detection (Python, Node, TypeScript, Rust, Go)
- 34 analysis tests

### Phase C: Indexing
- `SymbolIndex` — regex-based symbol parser for Python, JS/TS, Rust, Go
- `DependencyGraph` — import/dependency graph with cycle detection, transitive reachability
- Supports: functions, classes, methods, imports
- 33 indexing tests

### Phase D: Context Assembly
- `ContextPackBuilder` — token-budget-aware context assembly with priority ordering
- `ContextPack` — assembled context with deduplication, truncation, metadata
- `estimate_tokens()` — fast ~4 chars/token estimation
- Compaction support for long-running sessions
- 26 context pack tests

### Phase E: Parallel Execution
- `ParallelToolExecutor` — async concurrent execution with configurable concurrency
- `ToolCallDeduplicator` — content-hash deduplication of redundant tool calls
- Safe serialization for mutating operations
- 19 parallel tests

### Phase F: Observability
- `MetricsCollector` — thread-safe counters, gauges, timers with summary statistics
- Latency instrumentation for every subsystem
- Performance dashboard generation
- 18 metrics tests

### Phase G: Benchmarking
- 10 performance benchmarks covering: cache hit rates, repo analysis, symbol indexing, context pack assembly, parallel execution, metrics overhead
- Benchmark results recorded per-run

**Test Count:** 280 unit tests + 10 benchmarks = 290 total

**Verified:**
- All 280 unit tests pass
- All 10 benchmarks pass
- E2E: Agent successfully inspects repository using intelligent context
- Context cache reduces repeated file reads to near-zero latency
- Symbol index builds 50-file index in ~115ms, lookups in <0.001ms
- Context pack assembly: 100 files in 0.013ms per build
- Metrics overhead: 10,000 operations in 5ms (0.0005ms per operation)

---

## Milestone 4: Advanced Agents

**Goal:** Specialized agents with subagent orchestration.

**Components:**
- Specialized agents (Build, Plan, Research, Debug, Test, Review)
- Subagent system with independent context
- Session memory and compaction
- Git checkpoints (safe state snapshots)
- Model-specific context optimization

**Success:** Complex multi-step tasks with agent delegation and context intelligence.

---

## Milestone 5: Production Readiness

**Goal:** Research-grade, observable, reproducible system.

**Components:**
- Benchmarking framework (multi-model comparison)
- Observability (structured traces, cost tracking, latency)
- Replay capability (reproduce agent runs)
- Security hardening (secret detection, prompt injection defense)
- CI/CD pipeline
- Cross-platform packaging (pip, uv, standalone binaries)
- Comprehensive documentation

---

## Milestone 6: Public Beta

**Goal:** Production-quality release.

**Components:**
- Security audit
- Dependency audit
- Installation testing (Windows, macOS, Linux)
- Documentation review
- Performance testing
- Failure testing
- Model/provider failure testing

**Success:** Public release with clean installation and upgrade path.
