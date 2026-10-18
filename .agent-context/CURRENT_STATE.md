# Current State

Last checked: 2026-10-18 (Autonomous Git Workflow & Zero-Tool-Call Resolution)

## Autonomous Git Workflow & Zero-Tool-Call Fix (2026-10-18)

- **Zero-Tool-Call Defect Diagnosed & Fixed**:
  - `WorkerAgent` wrapped the user prompt in an agent preamble (`You are a specialized agent: ... Objective: <goal>`), preventing `classify_workflow()` from matching leading keywords. As a result, deterministic workflows were skipped and fell through to the LLM. When OpenRouter was rate-limited (HTTP 429), the task failed with 0 tool calls.
  - Fixed `classify_workflow()` in `workflows.py` to unwrap `Objective:` and recognize push instructions (`push this project`, `push to github`, etc.).
  - Fixed `AgentRegistry`: `GIT_RELEASE` role was missing git tools (`git_status`, `git_diff`, `git_log`, `git_remote`, `git_identity`, `git_add`, `git_commit`, `git_push`) in `allowed_tools`, which caused tool filtering to block Git operations.
  - Fixed `_role_for_intent` in `runtime.py` to route `TaskIntent.GIT` to `AgentRole.GIT_RELEASE`.
  - Fixed `run_git_push_workflow`: clean working tree now skips stage/commit but still pushes unpushed commits; remote URL in goal is auto-configured to `origin` when no remote exists.
  - Fixed `GitLogTool` to return success `(no commits yet)` on brand new repos without commits.
  - Fixed CLI tool failure diagnostics in `interactive.py` to show the actual error rather than generic placeholders.
- **Acceptance Scenario Proven**: `tests/unit/test_deterministic_git_push_acceptance.py` proves a complete initial commit and push to bare remote executes with 0 LLM calls under a failing/rate-limited provider.
- **Tests**: Full unit suite: **1,667 passed, 0 failed**.

## OpenRouter & Provider Reliability Audit (2026-10-18)

- **Root Cause Fixed**: Conflation of model-level HTTP 403 (e.g. partner-harness model restrictions) with global provider credential/authentication failure in `fallback.py` + `http_adapters.py`, which caused Harness to abort immediately and falsely claim `OPENROUTER_API_KEY` was invalid.
- **Failover Chain Fixed**: `router.py` now preserves dynamic resilient routes (`openrouter/free`) in free fallback chains even when numerous concrete models are discovered.
- **Error Classification & Redaction**: Added structured `PERMISSION`, `CONTEXT_LENGTH`, and `CONFIGURATION` categories; automated secret/bearer token redaction in fallback logs and error summaries.
- **Credential Resolution**: Fixed `.env` parsing when shell env vars contain whitespace or empty strings, ensuring robust cross-directory resolution.
- **Tests**: +24 unit regression tests in `tests/unit/test_openrouter_reliability_audit.py`. Complete test suite: **1,767 passed, 7 skipped, 0 failed**. Smoke tests (`harness --help`, `harness doctor`) passed.

## Project intelligence wired (2026-09-30, same takeover session)

- New canonical facade `harness_core/intelligence/__init__.py`:
  `ProjectIntelligence` — one instance per workspace wiring the EXISTING
  `SymbolIndex` + `DependencyGraph` + `RelevanceRanker` + `SearchCache` +
  native `fast_*` (Python fallbacks when Rust ext is not built). No duplicate
  systems: every underlying component is reused, not reimplemented.
- Capabilities: full scan with incremental refresh (mtime/size/hash diffing,
  hash-guarded), delete handling, per-file `on_file_changed()` hook,
  literal+regex search (cache-backed, positive-results-only to avoid stale
  negatives), indexed glob, symbol lookup/definitions, dependency queries
  (`dependencies_of/dependents_of/related/find_related_tests`),
  `context_candidates()` (RelevanceRanker + dependency-seed boost), secrets
  never indexed (.env/.pem/.key), skip-dirs, perf stats.
- Wired into: `GlobTool`/`GrepTool` (optional `intelligence` ctor arg; fast
  path only when the index is warm and the search dir is inside the indexed
  root; output contract byte-identical to the rglob fallback — 42 existing
  tool tests unchanged), `ContextPipeline` (new `intelligence` param; SYMBOL
  and DEPENDENCY candidate sources now live, previously unused enum values),
  `AgentLoop` (new `intelligence` param; `_search_matches_for(goal)` extracts
  content-match evidence so the pipeline's GREP source finally receives
  search matches), CLI sites `interactive.py::_setup_provider` and
  `main.py::_run` (facade built + scanned once, shared by tools and loop;
  build failure is non-fatal).
- Windows quirk handled: `fast_file_index` returns 8.3 short paths
  (`KUNALS~1`) that do NOT match `Path.resolve()` output; the facade keys
  everything off `self.root` resolved once and rejects entries outside it.
- DependencyGraph got `remove_outgoing_edges()` — re-indexing a changed file
  rebuilds its OWN imports while preserving incoming edges owned by importers
  (plain `remove_file()` during re-index silently dropped other files' edges).
- Tests: +42 (`tests/unit/test_intelligence.py` 27,
  `test_intelligence_integration.py` 15). Full suite **1,639 passed, 0
  failed**. `harness --help` and `harness tools list` smoke OK.

## Production-engineering takeover (2026-09-30)

- Full unit suite: **1,597 passed, 0 failed** (was 12 failures). Fixes:
  stale routing fixtures reconciled to the paid-model opt-in contract,
  `ConversationRenderer.start()` now seeds a monotonic `started_at` fallback
  (execution.state event still overwrites), footer wording/tri-state/providers-
  registry test drift reconciled, read-only planning shortcut honored in
  plan-backed TODO tests. See KI-005 for the full list.
- Architecture audit (prompt §1): EXISTING and intact — canonical
  ExecutionState/TaskExecutionTimeline, provider-neutral factory + adapters,
  FallbackEngine category-gated rotation, canonical semantic events,
  ContextPipeline/ReuseManager/compaction, Rust `harness-fs` + Python fallbacks.
  PARTIAL/unwired — `indexing/` (`SymbolIndex`, `DependencyGraph`) and
  `analysis/` (`RepositoryAnalyzer`, `RelevanceRanker`) are implemented but
  imported nowhere outside their packages; `GlobTool`/`GrepTool` are pure-Python
  rglob loops that use neither the native layer nor the symbol index;
  `cache/search_cache.py`/`file_cache.py` exist unwired. MISSING — incremental
  (hash-based) index updates, filesystem watcher, multi-layer search dispatcher,
  symbol tool exposed to the model, task-graph session resume. These are the
  highest-value next steps (prompt §11–§13: project intelligence).
- Rust native extension is not currently built (`is_native_available()` False);
  Python fallbacks cover the same API, so all callers remain functional.

## Provider-neutral architecture (validated offline 2026-09-30)

- `providers/factory.py` builds configured adapters from `.harness/config.yaml`
  + `CredentialResolver`; default remains OpenRouter when no providers list exists.
- New adapters: OpenAI, Anthropic, Gemini, OpenAI-compatible (via `http_adapters.py`),
  wrapped by `NormalizedModelProvider`. `RoutedModelProvider` sends Planner/auxiliary
  calls through `ModelRouter`/`FallbackEngine`.
- Core routing no longer owns `DEFAULT_OPENROUTER_FREE_MODELS` /
  `openrouter_free_models`. Free candidates come from discovery + configured models
  + provider `routing_hints()` (OpenRouter exposes `openrouter/free` there).
- Capability fields are tri-state; unknown tool support scores neutral (0.5), not false.
- Discovery failures are recorded and surface as explicit errors when the catalog is empty.
- Auth failover messages include HTTP status + provider credential hint (no secrets).
- Deterministic validation (no live API calls):
  - `python -m compileall -q src tests` → OK
  - Focused provider/routing/runtime suite → **196 passed**
    (`test_free_routing`, `test_openrouter_failover`, `test_model_health_failover`,
    `test_provider_factory`, `test_provider_auth_failure`, `test_provider_error_propagation`,
    `test_runtime_governance`, `test_execution_experience`, `test_agent_loop_integration`,
    `test_canonical_runtime`, `test_ninerouter`, `test_groq_discovery`)
- Full suite and live provider calls were not run in this takeover session.

## Remaining provider-neutral follow-ups

- Auxiliary modules that still accept a raw `ModelProvider`
  (`context/summarizer.py`, `recovery/planner.py`, `memory/retention.py`) work when
  callers pass `RoutedModelProvider`, but some non-CLI entrypoints may still inject
  a bare adapter.
- Streaming is implemented on adapters but is not the AgentLoop hot path.
- `CompletionResponse.usage` remains a dict; typed `TokenUsage` is exposed via property.
- UI still has a few OpenRouter-oriented short display names for common free models.
- `.agent-context/DECISIONS.md` still records the older OpenRouter-only interactive
  constraint; architecture docs now describe multi-provider factory wiring.

## P1 execution experience (verified 2026-09-30)

- Canonical per-task `ExecutionState` / `TaskExecutionTimeline` remain in place.
- Prior focused execution/runtime/UI + P0 failover evidence: **265 passed**.
- `uv run harness --help` succeeded earlier; live OpenRouter doctor/auth status is
  external credential-dependent and was not re-tested in the takeover session.

## HARNESS Reliability v0.2 — Freebuff master prompt sprint (2026-10-10)

Baseline audit + verification of existing Harness implementation against the
master prompt's P0–P2 requirements. No architectural changes made — all
critical systems were found intact and tested.

### P0 — Correctness and safety: PASS

- Intent classification: `test_intent_classification.py` 19/19 + `test_intent_contract.py` 32/32 = 51/51 pass.
- Read-only enforcement at tool boundary: `write_file`/`edit_file` blocked for EXPLAIN/INSPECT/QUESTION tasks (`AgentLoop._execute_tool_checked` line 1670). `test_readonly_enforcement.py` logic passes (only Unicode print issue on Windows).
- EXPLAIN plan filter: model-proposed plans filtered to inspection-only steps; composite "explain and fix" correctly classified as non-read-only.
- Completion invariant: `test_completion_invariant.py` 16/16 + `test_completion_ux.py` 20/20 = 36/36 pass. `can_complete_task()` gates every COMPLETED transition. No false success claims.
- Conversation intent ("hi"): classified as OTHER, no planning for modifications triggered.

### P1 — Execution reliability: PASS

- Provider error classification: `FallbackEngine` classifies AUTH (401/403), RATE_LIMIT (429), MODEL_UNAVAILABLE, TIMEOUT, NETWORK, PAYMENT (402), INVALID_REQUEST, and SERVER errors separately. Auth/payment errors do NOT trigger model rotation.
- Bounded retries: `RetryConfig(max_retries=1, backoff_factor=2.0, max_delay=30s)` with jitter. Per-attempt timeout 90s, total timeout 120s.
- Retry-after honored from `ProviderRequestError.retry_after`.
- Health tracking: `ModelHealthTracker` records per-model cooldown; known-unavailable models skipped at any chain position (fail-fast).
- Task pause on transient failure after progress: `AgentLoop` line 2469+ — model_unavailable/rate_limited/auth/payment with existing progress → PAUSED, state preserved.
- No-progress loop termination: stagnation counter at 2 warns, 3 stops. Diagnosis mode bounded at 5 iterations.

### P1 — Git operations: PASS

- `test_git_accounting.py` 4/4 pass: real git push through deterministic workflow is accounted as tool calls + events.
- Git identity safety: `GitCommitTool` blocks `git config user.name/email` in commit message; reports missing identity with setup instructions.
- Push diagnostics: auth failure (401/403/permission) appends GitHub auth setup hints.
- No force push without explicit `force=True`; uses `--force-with-lease` when forced.
- `test_professional_cli.py`: 4/4 — success renders evidence-based summary, failure renders honest summary, JSON output preserved, Ctrl+C renders cancellation (exit 130).

### P2 — TUI lifecycle: PASS

- Single `ConversationRenderer` per task (`self._conv`), created in `_execute_task()`, stopped in `finally` block. `self._conv = None` after stop.
- `conv.start(goal)` renders prompt once. `conv.stop()` guaranteed cleanup.
- `_first_task` flag prevents duplicate clear on first task.
- Plain mode and rich mode both handle completion rendering with fallback.
- Elapsed timer: seeded at `conv.start()` with `time.monotonic()`, overwritten by canonical `execution.state` event.
- No fake tool calls, no fabricated progress.

### E2E critical flows: 28/28 pass

- `test_critical_flows.py`: explanation no-modifications, coding creates file, fallback on primary failure, all-provider failure structured error, permission denial graceful, verification gate, context reuse, scheduler/taskgraph, convergence governor, budget enforcement, model health tracking.

### Full test suite: 1,639 unit tests passed, 0 failed

89/89 E2E tests pass (excluding `test_provider_validation.py` which requires live OpenRouter credentials and is skipped on this machine).

### Scratch files cleaned up

Removed: `interactive_old.py`, `scratch/`, `scratch_out.txt`, `scratch_print.py`, `scratch_print2.py`, `scratch_test.py`, `test_output2.txt`, `test_readonly_enforcement.py`.

### User modifications preserved

Two files modified (not by this sprint):
- `src/harness_core/cli/interactive.py`: simplified main loop — replaced complex `asyncio.wait` dual-task pattern with straightforward sequential loop. Clean improvement.
- `tests/e2e/test_professional_cli.py`: updated assertions to match current Qwen-style output format.

### Remaining known gaps (from prior audit, not addressed this sprint)

- `indexing/` (`SymbolIndex`, `DependencyGraph`) and `analysis/` (`RepositoryAnalyzer`, `RelevanceRanker`) implemented but not fully wired outside their packages.
- Rust native extension not built (`is_native_available()` False); Python fallbacks cover the API.
- Task-graph session resume not yet implemented.
- Incremental hash-based index updates and filesystem watcher not yet implemented.

### Verification method

All claims above verified by running the actual test files. No live provider calls were made (no `OPENROUTER_API_KEY` available). Provider failure behavior verified through deterministic mock tests.
