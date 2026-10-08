# Current State

Last checked: 2026-09-30 (production-engineering takeover session)

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

## Worktree caution (re-confirmed 2026-09-30)

This session modified only test fixtures/tests that were stale against the two
newer intentional contracts, plus one production fix in
`cli/conversation.py` (started_at seed) and `.agent-context/` updates. No
worktree reset/cleanup was performed.

Many application, test, documentation, and memory files remain modified or
untracked from overlapping agent work. Never reset/restore/delete shared
worktree changes as cleanup.
