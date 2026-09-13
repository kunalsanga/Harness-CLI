# HARNESS 2.0 — Forensic Architecture Audit

**Date:** September 13, 2026
**Purpose:** Complete repository mapping before any implementation work
**Status:** AUDIT COMPLETE — DO NOT MODIFY FILES UNTIL REVIEWED

---

## 1. Executive Summary

The Harness repository contains **substantial, real infrastructure** across 30+ source modules, 100+ test files, and 4 distinct runtime paths. The system is genuinely capable but suffers from **fragmented architecture**: three competing execution runtimes, four competing UI rendering paths, and significant dead/scaffolded code that was never wired into the default path.

**What is real:**
- AgentLoop with full tool execution, context management, verification, and governor
- ModelRouter with 14-dimension scoring, fallback engine, health tracking, budget management
- TaskGraph, Scheduler, WorkerAgent with real concurrent execution
- EngineeringRuntime with requirements → plan → TaskGraph → Scheduler → verification
- Orchestrator with multi-agent decomposition and execution
- MemoryManager with knowledge graph, retriever, sanitizer, retention
- Session system with persistence, checkpointing, resume
- PermissionManager with autonomous mode, dangerous command detection
- ConversationRenderer with live Rich updates
- CompletionFormatter with structured success/failure/cancellation
- VerificationEngine with ecosystem detection
- Recovery system with failure classification and orchestrator
- EventBus as the canonical event system

**What is dead/dormant/legacy:**
- LiveStatus (dummy class in interactive.py, never used)
- `native/harness-fs/` (Rust crate exists but has no Python bindings, no integration)
- `src/harness_core/native/__init__.py` (empty)
- Many scratch_analyzer/scratch_audit files in root (dev artifacts)
- `ContextReuseManager` — exists in `context/reuse.py` but is never imported by AgentLoop
- Several test files reference components that have been refactored

**Critical problems identified:**
1. Three competing runtimes: AgentLoop (default), Orchestrator (multi-agent), EngineeringRuntime (unified)
2. Four UI rendering paths: console.print fallback, ConversationRenderer, LiveTerminalUI, CompletionFormatter
3. Duplicate EventBus wiring causing double-prompt rendering
4. Hardcoded fallback messages in provider error handling
5. Provider architecture locked to OpenRouter/NVIDIA/Ollama (no extensibility interface)
6. No streaming integration in the agent loop (stream method exists on providers but is unused)
7. Context engine is basic (no real compaction, no priority-based retrieval)
8. Memory system exists but is only partially wired into the default path

---

## 2. Repository Structure

```
harness-engineering-cli/
├── src/harness_core/
│   ├── cli/                    # Terminal UI layer
│   │   ├── main.py             # Typer CLI entry point (harness command)
│   │   ├── interactive.py      # InteractiveShell — main REPL loop
│   │   ├── conversation.py     # ConversationRenderer — live Rich UI
│   │   ├── completion.py       # CompletionFormatter — success/failure output
│   │   ├── runtime_dashboard.py # RuntimeViewModel + LiveTerminalUI (unified mode)
│   │   └── ui.py               # UI primitives (Color, Sym, fmt_elapsed, etc.)
│   ├── agent/                  # Core agent execution
│   │   ├── loop.py             # AgentLoop — single-agent execution engine
│   │   ├── workflows.py        # Workflow definitions (explain, fix, test, etc.)
│   │   ├── types.py            # Task, ToolCall, ToolResult, AgentConfig types
│   │   ├── todos.py            # TodoItem management, plan reconciliation
│   │   ├── intent.py           # Intent classification (explain, fix, build, etc.)
│   │   ├── completion.py       # can_complete_task, completion_blockers
│   │   ├── recovery.py         # Recovery logic
│   │   └── report.py           # Execution report building
│   ├── agents/                 # Multi-agent system
│   │   ├── orchestrator.py     # Orchestrator — multi-agent coordinator
│   │   ├── scheduler.py        # Scheduler — TaskGraph execution engine
│   │   ├── worker.py           # WorkerAgent — individual agent execution
│   │   ├── registry.py         # AgentRegistry + AgentProfile definitions
│   │   ├── domain.py           # TaskGraph, SubTask, AgentRole, AgentContract
│   │   ├── executor.py         # AgentExecutor for single-agent delegation
│   │   ├── locks.py            # WorkspaceLockManager + WorkspaceResource
│   │   ├── message_bus.py      # AgentMessageBus — inter-agent communication
│   │   ├── parallel.py         # Parallel execution utilities
│   │   └── cancellation.py     # Cancellation handling
│   ├── runtime/                # Unified engineering runtime
│   │   ├── runtime.py          # EngineeringRuntime — full lifecycle owner
│   │   ├── state.py            # ProjectState, RuntimeStage, RuntimeStatus
│   │   ├── context.py          # Role-scoped context assembly
│   │   ├── requirements.py     # Requirements, TraceabilityIndex, Evidence
│   │   └── governance.py       # ConvergenceGovernor — stagnation detection
│   ├── routing/                # Model routing intelligence
│   │   ├── router.py           # ModelRouter — 14-dimension scoring + fallback
│   │   ├── task_aware.py       # TaskAwareRouter — task classification
│   │   ├── scoring.py          # rank_models, ScoringContext, ScoringWeights
│   │   ├── fallback.py         # FallbackEngine — retry + model switching
│   │   ├── health.py           # ModelHealthTracker — reliability/latency
│   │   └── budgets.py          # BudgetManager — cost/token limits
│   ├── providers/              # Model provider adapters
│   │   ├── base.py             # ModelProvider ABC, ModelInfo, CompletionRequest/Response
│   │   ├── openrouter.py       # OpenRouter provider
│   │   ├── nvidia.py           # NVIDIA NIM provider
│   │   └── ollama.py           # Ollama local provider
│   ├── models/                 # Model intelligence
│   │   ├── registry.py         # ModelRegistry — provider-agnostic model store
│   │   ├── types.py            # ModelProfile, CapabilityScore, CapabilityConfidence
│   │   ├── capabilities.py     # Capability scoring
│   │   ├── discovery.py        # discover_provider — model discovery
│   │   ├── empirical.py        # EmpiricalHistory — real performance tracking
│   │   └── history.py          # PerformanceHistory — model performance
│   ├── tools/                  # Tool implementations
│   │   ├── base.py             # Tool ABC, ToolSchema, ToolResult
│   │   ├── filesystem.py       # ReadFile, WriteFile, EditFile, ListFiles
│   │   ├── shell.py            # RunCommand
│   │   ├── git.py              # GitStatus, GitDiff, GitLog, GitAdd, GitCommit, GitPush, etc.
│   │   ├── search.py           # Glob, Grep
│   │   ├── diagnosis.py        # Command failure diagnosis, test parsing
│   │   ├── paths.py            # Workspace path resolution
│   │   └── parallel.py         # Parallel tool execution
│   ├── context/                # Context management
│   │   ├── engine.py           # ContextEngine — workspace discovery + assembly
│   │   ├── compaction.py       # ContextCompactor — history summarization
│   │   ├── pack.py             # estimate_tokens
│   │   └── reuse.py            # ContextReuseManager (DORMANT — never imported)
│   ├── memory/                 # Persistent memory
│   │   ├── manager.py          # MemoryManager — central write surface
│   │   ├── store.py            # LocalMemoryStore — JSON persistence
│   │   ├── graph.py            # KnowledgeGraph — relationship tracking
│   │   ├── retriever.py        # MemoryRetriever — RAG retrieval
│   │   ├── indexer.py          # MemoryIndexer — embedding/keyword indexing
│   │   ├── sanitizer.py        # MemorySanitizer — content safety
│   │   ├── retention.py        # MemoryRetentionPolicy — pruning
│   │   └── domain.py           # MemoryEntry, GraphNode, GraphEdge types
│   ├── session/                # Session management
│   │   ├── manager.py          # SessionManager — lifecycle orchestration
│   │   ├── storage.py          # SessionStorage — JSON persistence
│   │   ├── session.py          # Session domain
│   │   └── domain.py           # Session, Run, Checkpoint, MemoryItem types
│   ├── planning/               # Task planning
│   │   ├── planner.py          # Planner — model-driven decomposition
│   │   ├── domain.py           # Plan, PlannedTask types
│   │   └── validator.py        # PlanValidator
│   ├── verification/           # Verification engine
│   │   ├── engine.py           # VerificationEngine — ecosystem detection + checks
│   │   └── integrity.py        # check_test_integrity — anti-test-weakening
│   ├── recovery/               # Failure recovery
│   │   ├── classifier.py       # FailureClassifier — categorize failures
│   │   ├── orchestrator.py     # RecoveryOrchestrator — repair sequences
│   │   ├── planner.py          # Recovery planner
│   │   └── validator.py        # Recovery validator
│   ├── permissions/            # Security
│   │   └── manager.py          # PermissionManager — autonomous + dangerous ops
│   ├── observability/          # Events + metrics
│   │   ├── events.py           # Event, EventBus — canonical event system
│   │   └── metrics.py          # Metrics collection
│   ├── classifier/             # Task classification
│   │   ├── classifier.py       # TaskClassifier — 14-dimension task profiling
│   │   └── types.py            # TaskType, TaskProfile types
│   ├── config/                 # Configuration
│   │   └── config.py           # Configuration management
│   ├── errors/                 # Error types
│   │   └── errors.py           # Structured error types
│   ├── analysis/               # Code analysis
│   │   ├── relevance.py        # File relevance scoring
│   │   └── repository.py       # Repository analysis
│   ├── indexing/               # Code indexing
│   │   ├── dependency_graph.py # Dependency graph
│   │   └── symbols.py          # Symbol extraction
│   ├── cache/                  # Caching
│   │   ├── file_cache.py       # File content cache
│   │   └── search_cache.py     # Search result cache
│   ├── benchmarks/             # Model benchmarking
│   │   ├── engine.py           # AgentBenchmarkEngine
│   │   ├── scoring.py          # Benchmark scoring
│   │   ├── types.py            # BenchmarkCategory, BenchmarkTask types
│   │   └── tasks/              # Benchmark task definitions
│   ├── extensions/             # Extension system
│   │   ├── registry.py         # Extension registry
│   │   ├── loader.py           # Extension loader
│   │   ├── manifest.py         # Extension manifest
│   │   └── context.py          # Extension context
│   ├── hooks/                  # Hook system
│   │   └── hooks.py            # Hook registration
│   ├── mcp/                    # Model Context Protocol
│   │   └── client.py           # MCP client
│   ├── plugins/                # Plugin system
│   │   └── manager.py          # Plugin manager
│   ├── security/               # Security (EMPTY)
│   │   └── __init__.py
│   └── native/                 # Native bindings (EMPTY)
│       └── __init__.py
├── native/harness-fs/          # Rust filesystem crate (NO Python bindings)
│   ├── Cargo.toml              # PyO3 + walkdir + ignore + rayon + grep
│   └── src/                    # Rust source
├── tests/
│   ├── unit/                   # 75+ unit test files
│   ├── e2e/                    # 12+ E2E test files
│   ├── integration/            # (empty directory)
│   ├── benchmarks/
│   └── fixtures/
├── docs/                       # Documentation
├── examples/
├── pyproject.toml              # Build config: hatchling, Python 3.12+
└── .harness/config.yaml        # Project config template
```

---

## 3. Which Runtime Is Actually Used

### Default path: `uv run harness` (no arguments)

```
harness_core.cli.main:app
  → _main_callback()
    → run_interactive(mode="auto")
      → InteractiveShell.__init__()
      → InteractiveShell._setup_provider()    # Creates AgentLoop
      → InteractiveShell._setup_session()     # Creates/resumes session
      → InteractiveShell.run()                # REPL loop
        → _read_input()                       # prompt_toolkit
        → _execute_task(goal)
          → ConversationRenderer.start(goal)  # Renders prompt
          → AgentLoop.run(goal)               # Executes task
          → CompletionFormatter               # Renders result
```

**The default runtime is AgentLoop.** The EngineeringRuntime and Orchestrator are NOT used.

### Runtime paths by mode:

| Mode | Runtime | Status |
|------|---------|--------|
| `auto` (default) | AgentLoop | **ACTIVE — primary path** |
| `single` | AgentLoop | Active (same as auto) |
| `multi-agent` | Orchestrator → Scheduler → WorkerAgent → AgentLoop | Active but secondary |
| `unified` | EngineeringRuntime → Planner → TaskGraph → Scheduler → WorkerAgent → AgentLoop | Active but secondary |
| `free` | AgentLoop with free routing | Active variant |
| `local` | AgentLoop with local routing | Active variant |
| `best` | AgentLoop with best routing | Active variant |
| `fast` | AgentLoop with fast routing | Active variant |
| `cheap` | AgentLoop with cheap routing | Active variant |

---

## 4. UI Rendering Paths

### Path 1: Console.print fallback (plain mode or no ConversationRenderer)
- Location: `interactive.py` lines ~400-500
- Activated when: `self.plain or _conv() is None`
- Behavior: Direct `console.print` with Rich markup
- Status: **Active fallback** — used in plain mode and non-TTY

### Path 2: ConversationRenderer (default rich mode)
- Location: `conversation.py`
- Activated when: `ConversationRenderer is not None and not self.plain`
- Behavior: Rich `Live` widget with in-place updates
- Status: **Active — primary UI path**

### Path 3: LiveTerminalUI (unified mode only)
- Location: `runtime_dashboard.py`
- Activated when: `mode="unified"` and `interactive_ui=True`
- Behavior: Throttled dashboard via RuntimeViewModel
- Status: **Active only in unified mode** — dormant in default path

### Path 4: CompletionFormatter (all modes)
- Location: `completion.py`
- Activated when: Task completes
- Behavior: Structured success/failure/cancellation block
- Status: **Active — always used for final output**

### Duplicate rendering problem:
`ConversationRenderer.start(goal)` is called TWICE:
1. Explicitly in `InteractiveShell._execute_task()` before `AgentLoop.run()`
2. Via EventBus `on_task_started` handler which also calls `conv.start(goal)`

This causes the prompt to appear twice and thinking state to reset.

---

## 5. EventBus Architecture

### Event types emitted by AgentLoop:
```
task.started, task.completed, task.failed, task.paused, task.phase
thinking.status
todo.updated, todo.started, todo.completed, todo.failed
plan.created
tool.call, tool.result
model.error, model.switched
iteration.started
routing.decision, router.models_refreshed
task.classified
diagnosis.triggered
progress.stalled
execution.nudge
test.completed, test_integrity.warning
verification.started, verification.completed
error.occurred
```

### Event types emitted by EngineeringRuntime:
```
runtime_started, runtime_stage_changed, runtime_completed, runtime_failed
requirements_created, plan_created
task.started, task.completed, task.failed
recovery_started, recovery_exhausted
governor.stagnation, model.escalation
verification_completed
```

### Event types emitted by Scheduler:
```
scheduler.started, scheduler.completed
task.ready, task.queued, task.started, task.completed, task.failed, task.cancelled
resource_lock_requested, resource_lock_waiting, resource_lock_acquired, resource_lock_released
```

### EventBus consumers:
- InteractiveShell (event handlers → ConversationRenderer)
- RuntimeViewModel (attaches to bus, projects state for LiveTerminalUI)
- MemoryManager (records episodes on completion)
- ConvergenceGovernor (observes tool results for stagnation)

---

## 6. Provider Architecture

### Current providers:

| Provider | File | Auth | Streaming | Models |
|----------|------|------|-----------|--------|
| OpenRouter | `openrouter.py` | API key (env) | Yes (SSE) | Dynamic from /models |
| NVIDIA NIM | `nvidia.py` | API key (env) | Yes (SSE) | Dynamic from /models + fallback |
| Ollama | `ollama.py` | None (local) | Yes (NDJSON) | Dynamic from /api/tags |

### ModelProvider ABC interface:
```python
class ModelProvider(ABC):
    name: str                           # Property
    async def generate(request) -> CompletionResponse
    async def stream(request) -> AsyncGenerator[str, None]
    async def list_models() -> list[ModelInfo]
    async def health_check() -> bool
```

### What's missing from the interface:
- `discover()` — no dynamic provider discovery
- `authenticate()` — no auth validation
- `capabilities()` — no capability query
- `cancel()` — no request cancellation
- `usage()` / `cost()` — no usage tracking per provider
- `get_model()` — no single model lookup

### Provider error handling:
- `fallback.py` classifies errors as: TRANSIENT, PERMANENT, CAPABILITY_MISMATCH, RATE_LIMITED, PAYMENT_REQUIRED
- `AgentLoop` classifies model failures via `_classify_model_failure()` and `_classify_failure_reason()`
- **Problem:** Both classify errors independently with slightly different logic
- **Problem:** Provider-specific UI messages are in `ui.py` `render_provider_error()` — correct separation
- **Problem:** No structured `ProviderError` dataclass — errors are raw strings

---

## 7. Model Routing

### 14-dimension scoring system:
- Task type (implementation, bug_fix, refactoring, research, testing)
- Capability match (coding, reasoning, planning, etc.)
- Historical success rate
- Historical latency
- Tool efficiency
- Cost
- Context window fit
- Free/BYOK/local preference
- Health/reliability
- Classification confidence
- User preference
- Provider diversity
- Model freshness
- Streaming support

### Routing modes:
- `auto` — 14-dimension scoring
- `free` — free models only, no fallback to paid
- `best` — highest score regardless of cost
- `fast` — prefer fast/cheap models
- `local` — local models only
- `cheap` — prefer lowest cost
- `reasoning_high` — prefer reasoning models (agent policy)
- `coding` — prefer coding models (agent policy)
- `deterministic` — deterministic verification (verifier policy)

### Budget management:
- `BudgetManager` tracks: total tokens, per-model limits, cost limits, iteration limits
- Hard limits enforced before routing decisions

---

## 8. Multi-Agent System

### Agent roles (16 registered):
```
planner, architect, researcher, ui_designer, frontend, backend,
database, integration, tester, debugger, reviewer, security_reviewer,
verifier, git_release, coder, analyzer
```

### Execution flow (multi-agent mode):
```
Orchestrator.execute(task_description)
  → decompose_task() via Planner
    → TaskGraph created
  → Scheduler.execute(graph)
    → For each ready task:
      → Acquire workspace locks
      → Create WorkerAgent with AgentContract
      → WorkerAgent.run()
        → AgentLoop.run(prompt)
      → FailureClassifier.classify()
      → RecoveryOrchestrator.handle_failure() if failed
    → TaskGraph.update_status()
  → OrchestratorResult synthesized
```

### Execution flow (unified mode):
```
EngineeringRuntime.run(original_request)
  → Requirements structured
  → Planner.plan() → Plan
  → convert_plan_to_graph() → TaskGraph
  → build_traceability_index()
  → Scheduler.execute(graph)
    → [Same as multi-agent]
  → _verify_requirements() — requirement-level verification
  → RuntimeOutcome
```

### What's real vs scaffolded:
- **REAL:** TaskGraph, Scheduler, WorkerAgent, AgentContract, WorkspaceLockManager, AgentMessageBus
- **REAL:** FailureClassifier, RecoveryOrchestrator
- **REAL:** WorkerAgent creates real AgentLoop instances per worker
- **REAL:** Resource locking prevents concurrent file modification
- **SCAFFOLDED:** ContextReuseManager (never imported)
- **SCAFFOLDED:** Security module (empty)
- **SCAFFOLDED:** Native module (empty)

---

## 9. Tool System

### Available tools:

| Tool | Category | Status |
|------|----------|--------|
| read_file | filesystem | **Real** |
| write_file | filesystem | **Real** |
| edit_file | filesystem | **Real** |
| list_files | filesystem | **Real** |
| glob | search | **Real** |
| grep | search | **Real** |
| run_command | shell | **Real** |
| git_status | git | **Real** |
| git_diff | git | **Real** |
| git_log | git | **Real** |
| git_add | git | **Real** |
| git_commit | git | **Real** |
| git_push | git | **Real** |
| git_remote | git | **Real** |
| git_identity | git | **Real** |

### Tool execution governance:
- `AgentLoop` has execution governors: stagnation detection, repeat blocking, diagnosis mode
- `PermissionManager` enforces permissions per tool
- `WorkspaceLockManager` prevents concurrent file modifications across workers
- Tool results are truncated to `MAX_TOOL_RESULT_TOKENS = 8000` tokens
- History compaction: `HISTORY_CHAR_BUDGET = 120,000` characters

---

## 10. Context Management

### ContextEngine:
- Workspace discovery (files, languages, package manager, tests, git, entry points)
- Context assembly with priority-based pieces
- Budget-aware selection
- **Basic** — no real compaction, no file relevance scoring in production

### Context in AgentLoop:
- System prompt with workspace snapshot
- Tool history with compaction (recent calls verbatim, older collapsed to TASK STATE summary)
- Token ceiling enforcement (shrinks oversized tool results)
- Corrections/governance messages appended

### What's missing:
- Real file relevance scoring (exists in `analysis/relevance.py` but unused by AgentLoop)
- Context reuse across iterations (ContextReuseManager is dormant)
- Stale context detection
- Worker-specific context (exists in EngineeringRuntime but not in default path)

---

## 11. Memory System

### Architecture:
```
MemoryManager (write surface)
  ├── MemorySanitizer (content safety)
  ├── MemoryIndexer (keyword indexing)
  ├── LocalMemoryStore (JSON persistence)
  ├── KnowledgeGraph (relationship tracking)
  ├── MemoryRetriever (RAG retrieval)
  └── MemoryRetentionPolicy (pruning)
```

### Memory types:
- SUCCESS, FAILURE, ARCHITECTURE, DECISION, EPISODIC, NOTE, DISCOVERY, CONSTRAINT, TODO, WARNING

### Integration status:
- **Wired in:** EngineeringRuntime (unified mode), Scheduler, Orchestrator
- **NOT wired in:** AgentLoop (default path) — memory is not injected into prompts
- **Write surface:** Runtime writes on task completion/failure
- **Read surface:** get_context_for_planner(), get_context_for_worker(), get_failures_for_recovery()

---

## 12. Session System

### Architecture:
```
SessionManager
  ├── SessionStorage (JSON persistence)
  ├── Session domain (Session, Run, Checkpoint, MemoryItem)
  └── Event logging
```

### Session lifecycle:
- Created on interactive shell start
- Runs tracked per task
- Checkpoints created at safe boundaries
- Resume state available for paused/failed sessions
- Export as JSON or Markdown

### Integration status:
- **Wired in:** InteractiveShell creates/resumes sessions, tracks runs
- **NOT wired in:** EngineeringRuntime does not use SessionManager

---

## 13. Verification System

### VerificationEngine:
- Ecosystem detection (Python/pytest, Node/npm, Rust/cargo, Go, TypeScript)
- Check execution with timeout
- Result tracking (pass/fail/output/error)

### Integrity checking:
- `check_test_integrity()` detects if test files were modified alongside implementation
- Emits `test_integrity.warning` event

### Integration:
- AgentLoop runs verification when `verify_on_complete=True` and files were modified
- EngineeringRuntime runs requirement-level verification via TraceabilityIndex
- Both emit `verification.started` and `verification.completed` events

---

## 14. Recovery System

### Architecture:
```
FailureClassifier → categorize(task, result)
  → RecoveryOrchestrator.handle_failure(graph, task, classification)
    → RecoveryPlanner → plan repair sequence
    → Insert recovery tasks into TaskGraph
    → RecoveryValidator → verify recovery
```

### Failure categories:
- TEST_FAILURE, BUILD_FAILURE, LINT_FAILURE, TYPE_ERROR, RUNTIME_ERROR, etc.

### Integration:
- Scheduler invokes FailureClassifier after each worker completes
- RecoveryOrchestrator inserts retest tasks into TaskGraph
- EngineeringRuntime tracks recovery events and exhaustion
- **NOT wired in:** AgentLoop has its own simpler recovery (diagnosis mode + governor)

---

## 15. Permissions / Security

### PermissionManager:
- Three modes: autonomous (default), interactive, strict
- Safe command auto-approval (development commands, git, testing)
- Dangerous command blocking (rm -rf /, shutdown, credential access)
- Workspace boundary enforcement
- Protected path detection (.env, credentials, etc.)

### Secret protection:
- `_SECRET_PATTERNS` in interactive.py redacts API keys from display
- SessionManager sanitizes content before persistence
- `.env` file exists in repo (needs verification it's not committed with secrets)

---

## 16. Tests

### Test inventory:

**Unit tests (75+ files):**
- Agent loop, agent profiles, agent results, analysis, autonomous mode
- Cache, canonical tool result, completion invariant, completion UX
- Context (budget, compaction, pack, reuse)
- E2E fixture, empirical routing, forensic audit fixes
- Free routing, git accounting, git identity, governance
- Handoff integration, indexing, integration pipeline
- Intent classification, interactive shell, knowledge graph
- Locks, performance, extensions, productization
- Memory (manager, RAG integration, retrieval, scheduler wiring, store)
- Message bus, metrics, model intelligence, NVIDIA provider
- Observability, parallel, permissions, planning
- Production readiness, production UX, project context
- Provider auth failure, recovery, requirements
- Router policy, routing, runtime dashboard, runtime governance
- Runtime state, scheduler, scheduler locks, security audit
- Session, task plan UX, tools, tool failure propagation
- Traceability, UX interaction, UX upgrade
- Validation semantics, verification, welcome screen
- Worker agent

**E2E tests (12 files):**
- CLI memory integration, CLI unified runtime
- Concurrency E2E, integration pipeline
- Interactive E2E, live UI E2E
- Memory pipeline, permission E2E
- Professional CLI, provider validation
- Recovery integration, unified runtime

### Test gaps:
- No streaming tests
- No real provider integration tests (all mocked)
- No terminal resize tests
- No TUI snapshot tests
- No cross-platform tests
- No benchmark execution tests (only task definitions)

---

## 17. Rust / Native Status

### `native/harness-fs/`:
- Cargo.toml exists with PyO3, walkdir, ignore, rayon, grep dependencies
- **NO Python bindings** — `src/harness_core/native/__init__.py` is empty
- **NO integration** with the Python codebase
- Status: **Scaffolded only** — never completed

---

## 18. Hardcoded Messages Identified

### In AgentLoop (`loop.py`):
```python
_FAILURE_MESSAGES = {
    "model_unavailable": "⚠ Model temporarily unavailable",
    "model_rate_limited": "⚠ All free models rate limited (429)",
    "provider_auth_failure": "✗ Provider authentication failed",
    ...
}
```
These are **typed** and **canonical** — acceptable as runtime-owned messages.

### In `conversation.py`:
```python
def derive_intent(goal):
    # Returns fixed strings like:
    "I'll inspect the project structure and core runtime before explaining it."
    "I'll check the repository status and push the changes."
    "I'll start working on your request."
```
These are **semi-hardcoded** — derived from intent classification but with fixed templates.

### In `ui.py`:
```python
def render_provider_error(console, error):
    # Classifies error and shows fixed messages:
    "Free models temporarily unavailable"
    "Rate limited"
    "Model requires payment"
    ...
```
This is **correctly separated** — UI-layer rendering of structured errors.

### In `interactive.py`:
```python
# Inline in _execute_task():
console.print("  I'll inspect the existing project first, then plan the architecture, ...")
```
This is **hardcoded** — a static message printed in unified mode startup.

### In workflows (`workflows.py`):
Referenced but not fully read — likely contains hardcoded workflow-specific messages.

---

## 19. State Duplication

### Identified duplications:

1. **Tool display names** — implemented in THREE places:
   - `interactive.py` `_tool_display_name()`
   - `conversation.py` `_compact_tool_display()`
   - `ui.py` `_tool_display()`
   - `runtime_dashboard.py` `tool_display_name()`
   All are nearly identical but with subtle differences.

2. **Elapsed formatting** — implemented in THREE places:
   - `interactive.py` `_format_elapsed()`
   - `ui.py` `fmt_elapsed()`
   - `runtime_dashboard.py` `format_elapsed()` (delegates to ui.py)

3. **Error classification** — implemented in TWO places:
   - `AgentLoop._classify_model_failure()`
   - `AgentLoop._classify_failure_reason()` (duplicate of the above)

4. **Provider health checking** — each provider does its own health_check independently.

5. **Model registry** — `ModelRegistry` is instantiated separately in different paths.

---

## 20. Performance Bottlenecks (Likely)

1. **Synchronous file I/O in tools** — tools use `aiofiles` but some paths may block
2. **Model routing refresh** — `refresh_models()` makes HTTP calls with 5-minute cache
3. **Context assembly** — workspace discovery walks the filesystem
4. **EventBus dispatch** — all events go through async handlers, could bottleneck under load
5. **Memory retrieval** — JSON-based storage, no real vector DB
6. **Verification** — runs subprocess commands sequentially

---

## 21. What Needs to Change for Production

### Critical (must fix):
1. **Single runtime path** — consolidate AgentLoop + EngineeringRuntime
2. **Single UI path** — remove duplicate rendering, dead LiveStatus
3. **Fix EventBus double-wiring** — ConversationRenderer.start() called twice
4. **Remove hardcoded messages** — derive all from runtime events
5. **Structured provider errors** — create ProviderError dataclass
6. **Streaming integration** — wire provider.stream() through EventBus to UI

### Important (should fix):
7. **Provider interface expansion** — add discover(), authenticate(), capabilities(), cancel()
8. **Context engine upgrade** — real file relevance, priority retrieval, compaction
9. **Memory integration** — wire into default AgentLoop path
10. **Error classification consolidation** — single error classification path
11. **Tool display deduplication** — single canonical implementation
12. **Cross-platform testing** — Windows path handling, signals, Unicode

### Strategic (will fix):
13. **Rust native core** — replace hot paths (filesystem, process execution, event bus)
14. **Harness Cloud Gateway** — hosted free models
15. **Real streaming TUI** — ratatui/crossterm native terminal UI
16. **Plugin/extension system** — currently scaffolded
17. **MCP integration** — currently scaffolded

---

## 22. Component Maturity Matrix

| Component | Status | Notes |
|-----------|--------|-------|
| AgentLoop | **PRODUCTION** | Real execution, governors, context management |
| ModelRouter | **PRODUCTION** | 14-dim scoring, fallback, health, budgets |
| TaskGraph | **PRODUCTION** | Real dependency tracking, validation |
| Scheduler | **PRODUCTION** | Real concurrent execution, locks, recovery |
| WorkerAgent | **PRODUCTION** | Real agent execution via AgentLoop |
| Orchestrator | **PRODUCTION** | Real multi-agent coordination |
| EngineeringRuntime | **PRODUCTION** | Full lifecycle with verification |
| EventBus | **PRODUCTION** | Real event system, async handlers |
| ConversationRenderer | **PRODUCTION** | Live Rich UI, correct event mapping |
| CompletionFormatter | **PRODUCTION** | Structured success/failure output |
| PermissionManager | **PRODUCTION** | Autonomous mode, dangerous ops |
| SessionManager | **PRODUCTION** | Persistence, resume, export |
| VerificationEngine | **PRODUCTION** | Ecosystem detection, checks |
| MemoryManager | **PARTIAL** | Real infrastructure, not wired in default path |
| RecoveryOrchestrator | **PRODUCTION** | Real failure classification and repair |
| ContextEngine | **BASIC** | Workspace discovery, no real relevance |
| ContextReuseManager | **DORMANT** | Never imported |
| ProviderInterface | **BASIC** | Missing discover/auth/capabilities/cancel |
| Streaming | **SCAFFOLDED** | Provider.stream() exists, not wired to UI |
| Native/Rust | **SCAFFOLDED** | Cargo.toml exists, no bindings |
| MCP | **SCAFFOLDED** | Client exists, not integrated |
| Plugin system | **SCAFFOLDED** | Manager exists, not integrated |
| Security module | **EMPTY** | Only __init__.py |
| Benchmarks | **PARTIAL** | Engine + tasks exist, no execution integration |

---

## 23. Key Architectural Decisions Required

### Decision 1: Single Runtime Path
**Question:** Should `mode="auto"` use AgentLoop or EngineeringRuntime?
**Recommendation:** EngineeringRuntime is the superior architecture (requirements → plan → TaskGraph → Scheduler → verification). AgentLoop should become a thin wrapper that delegates to EngineeringRuntime for non-interactive tasks, and remains the single-task executor for interactive mode.

### Decision 2: UI Framework
**Question:** Keep Rich or move to native TUI?
**Recommendation:** Keep Rich for now (it works), but design the UI layer with a clean abstraction boundary so it can be swapped to ratatui later. The ConversationRenderer is already well-abstracted.

### Decision 3: Provider Interface
**Question:** Expand ModelProvider ABC or create a new Gateway interface?
**Recommendation:** Create a new `ModelGateway` interface that wraps providers and adds discovery, auth, capabilities, health, streaming, and cancellation. Keep `ModelProvider` as the low-level adapter interface.

### Decision 4: Rust Migration Scope
**Question:** What to migrate first?
**Recommendation:** Start with filesystem operations (the harness-fs crate already exists) and process execution. These are the most performance-sensitive and have the clearest Rust advantage.

### Decision 5: Harness Cloud
**Question:** When to start building the cloud gateway?
**Recommendation:** After the Python architecture is solid. The CLI should be able to point to a cloud endpoint as just another provider. The Gateway architecture should be designed now but implemented after the core is production-ready.

---

## 24. Files That Must Not Be Changed

- `src/harness_core/providers/base.py` — Provider ABC is correct
- `src/harness_core/routing/fallback.py` — Error classification is sound
- `src/harness_core/routing/scoring.py` — 14-dimension scoring is production-grade
- `src/harness_core/observability/events.py` — EventBus is the canonical event system
- `src/harness_core/permissions/manager.py` — Security model is solid
- `src/harness_core/agents/domain.py` — Domain types are well-designed

---

## 25. Next Steps

This audit is complete. No files have been modified.

**Awaiting decision on:**
1. Which runtime path to consolidate on
2. Whether to proceed with implementation or review this audit first
3. Priority order for the 10 implementation stages

**Do not proceed to Stage 1 (canonicalize runtime/event model) until this audit is reviewed and approved.**
