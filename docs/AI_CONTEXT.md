# Harness — Canonical AI Architecture Reference

> Module map checked 2026-09-29. For current interactive provider and model
> policy, see [`.agent-context/ARCHITECTURE.md`](../.agent-context/ARCHITECTURE.md).
> Update this file when architectural boundaries change.

---

## A. Product Purpose

Harness is an autonomous software-engineering CLI. The normal interactive
runtime and `harness run` currently instantiate OpenRouter. Provider adapters
for other services remain in the repository and some have separate commands,
tests, or compatibility uses; their presence does not make them active in the
normal task runtime. Treat this file as a broad module map, not a claim that all
provider adapters are currently enabled.

---

## B. Architecture Map

```
uv run harness
   └─ cli/main.py               CLI entrypoint (Typer)
        └─ cli/interactive.py   InteractiveShell (conversation-first TUI)
             └─ runtime/runtime.py  EngineeringRuntime (thin orchestration seam)
                  ├─ planning/planner.py    Planner → Plan / PlanningResult
                  ├─ agents/domain.py       TaskGraph (dependency graph of SubTasks)
                  ├─ agents/scheduler.py   Scheduler (concurrent task execution)
                  │    └─ agents/worker.py WorkerAgent (per-task isolated agent)
                  │         └─ agent/loop.py  AgentLoop (core tool-use loop)
                  │              ├─ routing/router.py   ModelRouter (14-dim scoring)
                  │              └─ providers/<name>.py ModelProvider (HTTP gateway)
                  ├─ recovery/orchestrator.py  RecoveryOrchestrator
                  ├─ verification/engine.py    VerificationEngine
                  └─ observability/events.py   EventBus (sole observability channel)
```

---

## C. Request Lifecycle

1. **Input** — user types task in `InteractiveShell`.
2. **Planning** — `Planner` decomposes task into a `Plan` with structured `Requirements`.
3. **TaskGraph** — `EngineeringRuntime` converts `Plan` into `TaskGraph` (`SubTask` nodes with dependencies).
4. **Scheduling** — `Scheduler` executes ready tasks concurrently (max 3 by default), assigning each to a `WorkerAgent`.
5. **AgentLoop** — `WorkerAgent` drives `AgentLoop`: model call → tool use → result → repeat until completion.
6. **Routing** — `ModelRouter` scores registered providers/models via 14 dimensions; 9router delegates selection to the gateway.
7. **Verification** — `VerificationEngine` runs checks (tests, linting, etc.) before marking a requirement complete.
8. **Recovery** — `RecoveryOrchestrator` classifies failures and retries or re-plans via `FailureClassifier`.
9. **Response** — completion is surfaced to the TUI via `EventBus`.

---

## D. Module Map

| Area | Location | Responsibility | Read when |
|---|---|---|---|
| CLI entry | `src/harness_core/cli/main.py` | Typer app, all CLI commands | Adding/fixing CLI commands |
| Interactive shell | `src/harness_core/cli/interactive.py` | TUI, conversation loop | Fixing UX/shell behavior |
| Conversation | `src/harness_core/cli/conversation.py` | Conversation state + rendering | Fixing message rendering |
| Runtime dashboard | `src/harness_core/cli/runtime_dashboard.py` | Live execution dashboard | Fixing dashboard display |
| EngineeringRuntime | `src/harness_core/runtime/runtime.py` | Project lifecycle orchestration | Changing execution flow |
| Runtime state | `src/harness_core/runtime/state.py` | `ProjectState`, `RuntimeStage` | Changing runtime state shape |
| Runtime governance | `src/harness_core/runtime/governance.py` | Budget/governance enforcement | Governance/budget issues |
| Requirements | `src/harness_core/runtime/requirements.py` | Structured requirements + traceability | Requirement/evidence system |
| AgentLoop | `src/harness_core/agent/loop.py` | Core tool-use loop (largest file) | Agent behavior issues |
| Agent types | `src/harness_core/agent/types.py` | `Task`, `TodoItem`, `ToolCall`, etc. | Type changes |
| Task todos | `src/harness_core/agent/todos.py` | TodoItem reconciliation | Todo plan behavior |
| Ask user | `src/harness_core/agent/ask_user.py` | Interactive clarification | Steering/ask-user issues |
| Steering | `src/harness_core/agent/steering.py` | `SteeringBuffer` integration | Steering pipeline |
| Completion | `src/harness_core/agent/completion.py` | Task completion invariants | Completion gating |
| TaskGraph / domain | `src/harness_core/agents/domain.py` | `TaskGraph`, `SubTask`, `AgentRole` | Task decomposition |
| Scheduler | `src/harness_core/agents/scheduler.py` | Concurrent task scheduling | Scheduling behavior |
| WorkerAgent | `src/harness_core/agents/worker.py` | Per-task isolated agent | Agent lifecycle |
| AgentRegistry | `src/harness_core/agents/registry.py` | Agent role → capability mapping | Adding agent roles |
| MessageBus | `src/harness_core/agents/message_bus.py` | Inter-agent communication | Multi-agent messaging |
| Planner | `src/harness_core/planning/planner.py` | Task planning/decomposition | Planning behavior |
| Planning domain | `src/harness_core/planning/domain.py` | `Plan`, `PlanningResult` | Planning type changes |
| Provider base | `src/harness_core/providers/base.py` | `ModelProvider` ABC, `CompletionRequest/Response` | Adding a provider |
| Groq | `src/harness_core/providers/groq.py` | Groq API integration | Groq issues |
| NVIDIA | `src/harness_core/providers/nvidia.py` | NVIDIA NIM integration | NVIDIA issues |
| OpenRouter | `src/harness_core/providers/openrouter.py` | OpenRouter integration | OpenRouter issues |
| Ollama | `src/harness_core/providers/ollama.py` | Local Ollama integration | Ollama issues |
| 9router | `src/harness_core/providers/ninerouter.py` | 9router gateway (automatic routing) | 9router issues |
| LiteLLM | `src/harness_core/providers/litellm.py` | LiteLLM provider | LiteLLM issues |
| ModelRouter | `src/harness_core/routing/router.py` | 14-dim scoring, fallback chains | Routing behavior |
| Fallback engine | `src/harness_core/routing/fallback.py` | Provider fallback logic | Fallback issues |
| Health tracker | `src/harness_core/routing/health.py` | Provider health state | Health tracking |
| Scoring | `src/harness_core/routing/scoring.py` | Model scoring dimensions | Scoring changes |
| Budgets (routing) | `src/harness_core/routing/budgets.py` | Per-request cost enforcement | Budget issues |
| Task-aware router | `src/harness_core/routing/task_aware.py` | Task-type routing policy | Task routing |
| Context engine | `src/harness_core/context/engine.py` | Context assembly | Context issues |
| Project intelligence | `src/harness_core/intelligence/__init__.py` | Canonical facade: scan/refresh/search/symbols/deps over SymbolIndex+DependencyGraph+RelevanceRanker+SearchCache+native fast_* | Project search/symbol/dependency needs |
| Context pipeline | `src/harness_core/context/pipeline.py` | Context pre-processing pipeline | Context pipeline |
| Context reuse | `src/harness_core/context/reuse.py` | `ContextReuseManager` (skip re-reads) | Context efficiency |
| Context compaction | `src/harness_core/context/compaction.py` | Long-context pruning | Token budget issues |
| Context pack | `src/harness_core/context/pack.py` | Token estimation, packing | Packing issues |
| Memory manager | `src/harness_core/memory/manager.py` | Persistent learned context | Memory behavior |
| Memory store | `src/harness_core/memory/store.py` | Memory persistence | Storage issues |
| Memory retriever | `src/harness_core/memory/retriever.py` | RAG retrieval | Retrieval issues |
| Memory graph | `src/harness_core/memory/graph.py` | Knowledge graph | Graph issues |
| Recovery classifier | `src/harness_core/recovery/classifier.py` | Failure classification | Recovery behavior |
| Recovery orchestrator | `src/harness_core/recovery/orchestrator.py` | Retry/re-plan logic | Recovery issues |
| Verification engine | `src/harness_core/verification/engine.py` | Runs test/lint checks | Verification issues |
| Verification integrity | `src/harness_core/verification/integrity.py` | Test integrity checks | Test validity |
| Tools base | `src/harness_core/tools/base.py` | `Tool` ABC | Adding tools |
| Filesystem tools | `src/harness_core/tools/filesystem.py` | File read/write/edit | Filesystem issues |
| Shell tools | `src/harness_core/tools/shell.py` | Sandboxed shell execution | Shell issues |
| Git tools | `src/harness_core/tools/git.py` | Git operations | Git issues |
| Search tools | `src/harness_core/tools/search.py` | Code/text search | Search issues |
| Path utilities | `src/harness_core/tools/paths.py` | Workspace boundary enforcement | Path/scope issues |
| Permissions | `src/harness_core/permissions/manager.py` | Tool permission enforcement | Permission issues |
| Credentials | `src/harness_core/config/credentials.py` | `CredentialResolver` | Auth/credential issues |
| Config | `src/harness_core/config/config.py` | `HarnessConfig` loading | Config issues |
| DotEnv loader | `src/harness_core/config/dotenv.py` | `.env` loading | Env-var issues |
| EventBus | `src/harness_core/observability/events.py` | Decoupled event system | Adding observability |
| Metrics | `src/harness_core/observability/metrics.py` | Run metrics collection | Metrics issues |
| Semantic events | `src/harness_core/observability/semantic.py` | Semantic event helpers | Semantic event issues |
| Session manager | `src/harness_core/session/manager.py` | Session lifecycle | Session issues |
| Session storage | `src/harness_core/session/storage.py` | Session persistence | Storage issues |
| Model registry | `src/harness_core/models/registry.py` | Known model metadata | Model capability lookup |
| Model empirical | `src/harness_core/models/empirical.py` | Empirical model performance | Empirical routing |

---

## E. Entrypoints

| Command | Entry | Notes |
|---|---|---|
| `uv run harness` | `cli/main.py → _main_callback → interactive.py` | No subcommand → interactive shell |
| `uv run harness run "<task>"` | `cli/main.py → run()` | Single task execution |
| `uv run harness shell` | `cli/main.py → shell()` | Explicit interactive shell |
| `uv run harness doctor` | `cli/main.py → doctor()` | System health check |
| `uv run harness providers` | `cli/main.py → providers_app` | Provider management |
| `uv run harness auth` | `cli/auth.py → auth_app` | Credential management |
| `uv run harness models` | `cli/main.py → models_app` | Model intelligence |
| `uv run harness benchmark` | `cli/main.py → benchmark_app` | Benchmarking |

---

## F. Data Flow

```
User prompt
   → InteractiveShell (conversation parsing)
   → EngineeringRuntime.execute()
   → Planner.plan()             ← LLM call for decomposition
   → TaskGraph (SubTask nodes)
   → Scheduler.run()
   → WorkerAgent.run()
   → AgentLoop.run()
        → ContextPipeline       ← context assembly + reuse
        → ModelRouter.route()   ← select provider/model
        → ModelProvider.stream() ← LLM streaming
        → Tool.execute()        ← tool calls with permission check
        → VerificationEngine    ← test/lint checks
        → RecoveryOrchestrator  ← on failure
   → EventBus.emit()            ← observability throughout
   → TUI rendering
```

---

## G. Provider Flow

All providers implement `ModelProvider` from `src/harness_core/providers/base.py`:

```python
class ModelProvider(abc.ABC):
    async def generate(request: CompletionRequest) -> CompletionResponse
    async def stream(request: CompletionRequest) -> AsyncGenerator
    async def list_models() -> list[ModelInfo]
    async def health_check() -> bool
```

Credentials flow: `.env` / environment variable → `CredentialResolver` → provider constructor.

Provider env keys:
- Groq: `GROQ_API_KEY`
- NVIDIA: `NVIDIA_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- 9router: `NINE_ROUTER_API_KEY`
- Ollama: no key (host-based)

---

## H. 9router

```
Harness
   └─ NineRouterProvider  (src/harness_core/providers/ninerouter.py)
        └─ HTTP POST → 9router gateway (localhost:20128/v1)
             model="qd/auto"
             └─ 9router selects underlying model automatically
```

**Critical invariant:** 9router owns underlying model selection. Harness never picks the underlying model. The selected model appears as `last_selected_model` for diagnostics only — not exposed to users. Do not expand `qd/auto` logic into Harness routing code.

Credential: `NINE_ROUTER_API_KEY` → `CredentialResolver` → `NineRouterProvider.__init__`.

---

## I. Free Mode

"Free mode" refers to preferring zero-cost models during routing (`prefer_free=True` in `RouterConfig`, `routing_mode="free"`). This is a **Harness routing policy**, not a separate product. It works with any provider that has free model offerings (e.g. OpenRouter free tier, Ollama local). Do not confuse with 9router or with a dedicated "Harness Free" product.

---

## J. Task / Todo Model

```
Plan (planning/domain.py)
   └─ PlanningResult
        └─ TaskGraph (agents/domain.py)
             └─ SubTask (dependency graph node)
                  └─ AgentContract → WorkerAgent
                       └─ AgentLoop
                            └─ TodoItem (agent/types.py)
                                 reconciled by agent/todos.py
```

- `TaskGraph.ready_tasks()` — returns tasks whose dependencies are complete.
- `Scheduler` polls `ready_tasks()` and dispatches `WorkerAgent` instances.
- `WorkerAgent` executes one `SubTask`; `AgentLoop` manages its internal `TodoItem` list.
- `TodoItem` is internal to `AgentLoop` — do not conflate with `SubTask`.

---

## K. Context / Memory

Three distinct systems — do not conflate:

| System | Location | Purpose |
|---|---|---|
| `ContextReuseManager` | `context/reuse.py` | Avoid re-reading unchanged files during a run |
| `ContextPipeline` | `context/pipeline.py` | Assemble + compress context for each model call |
| `MemoryManager` | `memory/manager.py` | Persistent cross-session learned context (RAG) |
| `SteeringBuffer` | `runtime/steering.py` (via `agent/steering.py`) | Mid-run user steering signals |

`ContextReuseManager` is snapshot-based: call `record_read()` after reading, `is_unchanged()` before re-reading. `invalidate()` after any write.

---

## L. Recovery

```
AgentLoop detects failure
   → RecoveryOrchestrator (recovery/orchestrator.py)
        → FailureClassifier (recovery/classifier.py)
             → FailureCategory enum (12 categories)
        → recovery planner (recovery/planner.py)
             → re-queue SubTask or escalate
```

**Invariant:** Recovery only mutates tasks through `TaskGraph` and `Scheduler` interfaces. It does not bypass `EngineeringRuntime` state. `RECOVERY_MARKER` and `RETEST_MARKER` in task IDs identify recovery-injected tasks.

---

## M. Verification

`VerificationEngine` (`verification/engine.py`) runs `VerificationCheck` commands (tests, lint, type checks). Returns `VerificationReport`. `VerificationEngine` is invoked by `AgentLoop` — do not skip it or fake results. `verify_check_integrity` (`verification/integrity.py`) validates that tests actually exercised the claimed code.

---

## N. UI / TUI

The TUI is **conversation-first**: `InteractiveShell` (`cli/interactive.py`) manages the REPL loop. The live execution dashboard is in `cli/runtime_dashboard.py`. All rendering is driven by `EventBus` events — components subscribe and react; nothing polls internal state directly. `cli/conversation.py` manages conversation rendering. `cli/ui.py` contains shared Rich UI helpers.

---

## O. Security

| Concern | Mechanism |
|---|---|
| Credential access | `CredentialResolver` only; never raw `os.environ` in providers |
| Display masking | `mask_secret()` in `config/credentials.py` |
| Tool permissions | `PermissionManager` (`permissions/manager.py`); `.harness/config.yaml` policies |
| Workspace boundary | `paths.py` — `cwd_in_workspace()`, `resolve_in_workspace()` |
| Shell sandboxing | `tools/shell.py` — gated by `PermissionManager` |
| Secret exposure | Credentials must never appear in EventBus events, logs, or model prompts |

Default permission policies (`.harness/config.yaml`):
- `bash: ask`, `edit: allow`, `network: ask`, `git_push: ask`

---

## P. Testing

```bash
# All tests
uv run pytest tests/ -x -q

# Targeted by subsystem
uv run pytest tests/unit/test_routing.py -x -q
uv run pytest tests/unit/test_ninerouter.py -x -q
uv run pytest tests/unit/test_recovery.py -x -q
uv run pytest tests/unit/test_planning.py -x -q
uv run pytest tests/unit/test_tools.py -x -q
uv run pytest tests/unit/test_verification.py -x -q
uv run pytest tests/unit/test_worker_agent.py -x -q

# Integration
uv run pytest tests/integration/ -x -q

# E2E
uv run pytest tests/e2e/ -x -q
```

Test config in `pyproject.toml` → `[tool.pytest.ini_options]`, `testpaths = ["tests"]`, `asyncio_mode = "auto"`.

---

## Q. Task-to-Context Navigation

| Task | Primary context | First files to read | Tests |
|---|---|---|---|
| Fix provider (Groq/NVIDIA/OpenRouter/Ollama) | `providers/AGENTS.md` | `base.py` + `<provider>.py` | `test_groq_discovery.py`, `test_nvidia_provider.py` |
| Fix 9router | `providers/AGENTS.md` + `routing/AGENTS.md` | `ninerouter.py` + `router.py` | `test_ninerouter.py`, `test_routing.py` |
| Fix model routing / fallback | `routing/AGENTS.md` | `router.py` + `fallback.py` + `scoring.py` | `test_routing.py`, `test_empirical_routing.py` |
| Fix CLI command | `cli/AGENTS.md` | `main.py` | `test_production_ux.py`, `test_m8_productization.py` |
| Fix interactive shell / TUI | `cli/AGENTS.md` | `interactive.py` + `conversation.py` | `test_interactive_shell.py` |
| Fix agent behavior | `agent/AGENTS.md` | `loop.py` + `types.py` | `test_agent_loop_integration.py`, `test_canonical_runtime.py` |
| Fix planning | `planning/AGENTS.md` | `planner.py` + `domain.py` | `test_planning.py` |
| Fix task scheduling | `agents/AGENTS.md` | `scheduler.py` + `worker.py` + `domain.py` | `test_scheduler.py`, `test_worker_agent.py` |
| Fix recovery | `recovery/AGENTS.md` | `classifier.py` + `orchestrator.py` | `test_recovery.py` |
| Fix verification | `verification/AGENTS.md` | `engine.py` + `integrity.py` | `test_verification.py` |
| Fix tools (filesystem/shell/git) | `tools/AGENTS.md` | `base.py` + relevant tool file | `test_tools.py`, `test_git_accounting.py` |
| Fix context / memory | `context/AGENTS.md` | `reuse.py` + `engine.py` + `pipeline.py` | `test_context_reuse.py`, `test_context_budget.py` |
| Fix project intelligence (scan/search/symbols/deps) | `src/harness_core/intelligence/__init__.py` + `indexing/AGENTS.md` | `intelligence/__init__.py` + `indexing/symbols.py` + `indexing/dependency_graph.py` | `test_intelligence.py`, `test_intelligence_integration.py` |
| Fix credentials / auth | `config/` | `credentials.py` + `config.py` | `test_provider_auth_failure.py` |
| Fix permissions | `permissions/` | `manager.py` | `test_permissions.py`, `test_permission_fix.py` |
| Fix session | `session/` | `manager.py` + `storage.py` | `test_session_m4.py` |
| Fix observability/metrics | `observability/` | `events.py` + `metrics.py` | `test_metrics.py`, `test_observability.py` |

---

## R. Common Agent Failure Modes

| Failure mode | Correct approach |
|---|---|
| Grepping entire repo before reading context | Read this file → subsystem `AGENTS.md` → targeted files |
| Creating a new provider without subclassing `ModelProvider` | Always subclass `providers/base.py:ModelProvider` |
| Treating 9router as a static model with a known underlying ID | 9router uses `qd/auto`; gateway selects — do not expose or hardcode |
| Adding routing logic inside `NineRouterProvider` | 9router owns routing; Harness only sends `qd/auto` |
| Claiming verification without running `VerificationEngine` | Always invoke `VerificationEngine`; never set `VerificationStatus.PASSED` manually |
| Bypassing `Scheduler` to run tasks directly | All task execution goes through `Scheduler → WorkerAgent → AgentLoop` |
| Adding a second `EventBus` | Use the existing `EventBus` instance passed through constructors |
| Stale session context leaking into new tasks | `ContextReuseManager.invalidate_all()` on new task start |
| Creating files outside workspace scope | Check with `paths.py:resolve_in_workspace()` before any write |
| Re-reading unchanged files every model call | Use `ContextReuseManager.is_unchanged()` to skip re-reads |
| Fabricating EventBus events | Only emit events through `EventBus.emit()` with proper `Event` dataclass |
| Over-planning a trivial task | `Planner` handles planning; do not add ad-hoc decomposition in `AgentLoop` |

---

## S. Do Not Rediscover

```
ENTRYPOINT:                    uv run harness
CLI MODULE:                    src/harness_core/cli/main.py
PROVIDER BASE:                 src/harness_core/providers/base.py (ModelProvider ABC)
CREDENTIAL RESOLUTION:         src/harness_core/config/credentials.py (CredentialResolver)
9ROUTER GATEWAY MODEL:         qd/auto  (never change; gateway decides)
9ROUTER CREDENTIAL KEY:        NINE_ROUTER_API_KEY
CORE EXECUTION CHAIN:          EngineeringRuntime → Scheduler → WorkerAgent → AgentLoop
EVENTBUS:                      src/harness_core/observability/events.py
WORKSPACE BOUNDARY:            src/harness_core/tools/paths.py
PERMISSION MANAGER:            src/harness_core/permissions/manager.py
TASK GRAPH:                    src/harness_core/agents/domain.py (TaskGraph, SubTask)
CONTEXT REUSE:                 src/harness_core/context/reuse.py (ContextReuseManager)
PROJECT INTELLIGENCE:          src/harness_core/intelligence/__init__.py (ProjectIntelligence facade)
FREE MODE:                     RouterConfig(prefer_free=True) in routing/router.py
HARNESS CONFIG:                .harness/config.yaml (runtime) + pyproject.toml (build)
```
