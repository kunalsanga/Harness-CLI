# Agent Context Changelog

## 2026-09-30 — Project intelligence facade wired (same takeover session)

- Added `harness_core/intelligence/` — canonical `ProjectIntelligence` facade
  over SymbolIndex/DependencyGraph/RelevanceRanker/SearchCache/native fast_*
  (reuses existing components; no duplicate systems).
- Scan/refresh: full scan with hash-guarded incremental refresh, deletion
  handling, per-file change hook, secrets excluded, perf stats.
- Search: literal (escaped) + regex with positive-result caching; indexed
  glob; symbol queries; dependency queries incl. related-tests.
- Wiring: GlobTool/GrepTool fast path (contract-identical fallback),
  ContextPipeline SYMBOL+DEPENDENCY sources, AgentLoop search-match evidence,
  CLI tool-wiring sites (interactive + run path).
- Fixed during bring-up: missing `_literal_to_regex`, `_full_scan` ordering
  (module candidates must exist before import resolution), `from X import Y`
  resolution preferring `X.Y` submodule over package `__init__`, edge loss on
  re-index (`remove_outgoing_edges`), Windows 8.3 short-path mismatch, empty-
  result cache poisoning, `file_list(limit=0)` truncation.
- Tests: +42 new; full suite **1,639 passed, 0 failed**. No live API calls.
  No commit.

## 2026-09-30 — Production-engineering takeover: suite to green + architecture audit

- Full unit suite **1,597 passed, 0 failed** (was 12 failures). No live API
  calls. No commit.
- Production fix: `ConversationRenderer.start()` seeds a monotonic
  `started_at` fallback so elapsed time is not dead before the first
  execution.state event (the event value still overwrites).
- Reconciled 12 stale test fixtures to newer intentional contracts while
  preserving each test's protected invariant: routing fixtures now respect the
  paid-model opt-in gate (free models or `allow_paid_models=True`); footer
  wording uses canonical "Completed"; `ModelProfile.supports_tools` tri-state
  None; `providers list --json` asserts the provider-neutral registry;
  plan-backed TODO tests use an engineering goal (read-only goals intentionally
  skip planning).
- Architecture audit completed (EXISTING/PARTIAL/MISSING/CONFLICTING/RISKY).
  Key finding: `indexing/` (SymbolIndex, DependencyGraph) and `analysis/`
  (RepositoryAnalyzer, RelevanceRanker) are implemented but unwired; search
  tools use neither the native Rust layer nor the symbol index; no incremental
  index or watcher. Identified as the highest-value next implementation
  surface (project intelligence, prompt §11–§13).
- Updated KI-005 (resolved) and CURRENT_STATE.md.

## 2026-09-30 — Provider-neutral takeover (offline validated)

- Finished Codex's unfinished provider-agnostic wiring without reverting shared
  worktree changes.
- Fixed discovery silence, free-chain dynamic-route ordering, unknown capability
  scoring, auth/error messaging, provider/model pin parsing, and OpenRouter
  stream `ProviderRequestError` propagation.
- Updated stale free-pool tests; added `test_provider_factory.py`.
- Validated: `compileall src tests` OK; focused provider/routing/runtime suite
  **196 passed**. No live API calls. No commit.

## 2026-09-30 — Provider-neutral routing implementation (unverified)

- Added configured provider construction and native OpenAI, Anthropic, Gemini,
  and OpenAI-compatible HTTP adapters.
- Added normalized provider error categories, token-usage metadata, and
  tri-state model capability metadata.
- Replaced core router fixed OpenRouter free-model IDs with adapter discovery
  and routing hints; preserved free mode and the paid-model opt-in.
- Began routing CLI/runtime model requests through the shared configured
  provider list and router.
- Static implementation only so far: no tests, CLI, compile checks, or live
  provider requests were run. Further review is required before calling this
  architecture validated.

## 2026-09-30 — P1 execution experience

- Added a canonical per-task execution state/timeline while retaining existing
  task status and phase types; task/phase elapsed time is monotonic and model
  and tool durations remain nested diagnostics.
- Made the existing plan and live TODO states visible and included them in each
  working model context. Read-only tasks skip the extra plan call; engineering
  plans are bounded by existing task complexity.
- Connected semantic tool/model/verification events to compact conversation
  progress and completion metadata, including model history and phase duration.
- Updated read-only fake-provider fixtures for the no-extra-plan policy.
- Focused execution/runtime/UI and model-failover tests: 265 passed. `harness
  --help` succeeded; `harness doctor` reported OpenRouter `[FAIL]` once with no
  diagnostic category. Source compileall completed with one existing
  SyntaxWarning in a benchmark task string.

## 2026-09-29 — P0 orchestration: unified cooldowns, fail-fast, Retry-After, health-aware /models

- Health model extended (not replaced): every temporary failure (429,
  provider_unavailable/wrapped 502, 5xx, timeout, network) now sets a
  model-specific `cooldown_until` (bounded exponential 60s→300s, honored
  from provider Retry-After hints). `is_healthy` blocks selection during any
  active cooldown. Success clears cooldowns. New statuses: TEMPORARILY_
  UNAVAILABLE, TIMEOUT; `display_status` + `cooldown_remaining()` for UI.
- Fail-fast: FallbackEngine and RouterConfig empty-chain paths return a
  structured per-model summary (status + cooldown seconds) immediately — no
  provider requests start when every configured model is known-unusable.
  Live-verified: previously 5m00s hang → now instant structured failure.
- Per-attempt timeout: `FallbackConfig.model_attempt_timeout_seconds`
  (default 90s, configurable via `.harness/config.yaml`
  `routing.fallback.model_attempt_timeout_seconds`) bounds a single hung
  provider request; wrapped in asyncio.wait_for inside the fallback engine.
- Normalized error categories (`rate_limit`, `model_unavailable`, `timeout`,
  `network`, `server`, `auth`, `payment`, `invalid_request`, `unknown`) now
  ride on FallbackResult.error_category and attempt records; AgentLoop does
  not parse OpenRouter-specific strings (one added phrase for the fail-fast
  summary so tasks PAUSE with correct reason instead of failing generic).
- New `model.cooldown` EventBus event (model, category, cooldown_seconds);
  runtime dashboard logs it. `/models` now shows Configured + Health
  (Ready/Rate Limited/Unavailable/Timeout/Unknown) + Cooldown per model.
- Fail-fast skip rule: unhealthy models are skipped at ANY chain position
  (previously the primary was always attempted), eliminating wasted attempts
  on known-dead models while keeping cooldown-expired models ordered first.
- New test file `test_model_health_failover.py` (15 tests, cases A–M);
  realigned 3 stale expectations (429 status, fail-fast summary wording,
  empty-chain message). Focused suites: 254 passed. Full unit: 1,546 passed,
  2 failed (pre-existing completion-UX copy mismatches).
- External state observed: OpenRouter free tier is rate-limiting the account
  across ALL four pool models (shared budget); cooldowns re-arm on each
  planning attempt. This is the dominant live failure mode now.

## 2026-09-29 — Free-pool failover fixed: wrapped 200 provider errors now fail over

- Live probe established the real Nemotron failure: OpenRouter returns
  **HTTP 200** wrapping `{code: 502, message: "Upstream error from Nvidia:
  ResourceExhausted: Worker local total request limit reached (16/16)",
  error_type: provider_unavailable}`. The provider previously raised a code-
  and type-less message, which classified as UNKNOWN and terminated the whole
  task (`openrouter/free` was never attempted).
- Fixes: `providers/openrouter.py` now includes `code` and `error_type` in the
  raised wrapped-error message; `routing/fallback.py` classifies
  `provider_unavailable` as MODEL_UNAVAILABLE (failover-eligible); UI adds a
  distinct pool-exhausted message for model_unavailable exhaustion.
- Realigned 5 stale unit tests to the accepted failover policy (bounded
  rotation through all 4 pool models; 401/403 stop; free-pool chain).
- Verified live: Gemma 31B rate-limited → Nemotron 502/provider_unavailable →
  **`openrouter/free` executed a tool and continued** — the exact previously
  broken path now works. A simple task completes on the primary. KI-001 has
  self-resolved (credential authenticates; `harness doctor` → [OK] OpenRouter).
- Full unit suite: **1,531 passed, 2 failed** — both pre-existing completion-UX
  copy mismatches in files with another agent's uncommitted work.

## 2026-09-29 — OpenRouter live failure root-caused (401, external)

- Reproduced with the project `.env` key via the project's own dotenv loader:
  key loads correctly; direct chat requests to OpenRouter return **HTTP 401
  "User not found."** for all three free-pool targets (Gemma 26B, Gemma 31B,
  `openrouter/free`); `GET /auth/key` returns the same 401. No-auth and
  fabricated-key controls confirm Harness constructs valid requests.
- Verdict: **external credential problem** — the key does not match an
  OpenRouter account. No Harness request/credential bug exists. `harness
  doctor` independently confirms the failure through Harness's own path.
- Minimal hardening (verified cause: truncation could hide 401-vs-403):
  `routing/fallback.py` auth `final_error` now leads with the raw provider
  detail so the numeric status survives downstream truncation;
  `cli/interactive.py` `_friendly_model_error` classifies 401 vs 403 from the
  numeric status first, with the truncated-auth phrase mapping to the
  credential message. Free-model failover policy unchanged.
- Added regression test `test_auth_final_error_status_survives_truncation`
  (401 and 403 distinguished after 120-char truncation).
- Focused suites: `test_provider_auth_failure.py` +
  `test_provider_error_propagation.py` + `test_free_routing.py` = **54 passed**.

## 2026-09-29 — Initial shared context package

- Added project, architecture, current state, decisions, roadmap, active work,
  known issues, and testing notes grounded in the checked-out source and live
  verification from this session.
- Updated root `AGENTS.md` to establish a short mandatory context reading order
  and retain task-specific subsystem guidance.
- Recorded the live connectivity failure and the unverified earlier
  auth/access-status report separately.
- No application code changed as part of this context bootstrap.

## 2026-09-29 — Interactive UI work recorded

- Compact interactive header and concise completion footer.
- `/models` displays the configured OpenRouter free pool without claiming
  unknown tool capability.
- Model failover notices are event-driven and concise; normal error text is
  user-facing while verbose mode preserves diagnostics.
- Focused free-routing tests: 18 passed. Live task completion remained blocked
  by network connectivity in this environment.

## 2026-09-29 — Repository hygiene audit

- Anchored the local `/models/` ignore pattern to the repository root, exposing
  the active `src/harness_core/models/` package to Git; added `.uv-cache/` to
  ignored local caches.
- Removed `docs/HARNESS_PROJECT_TREE_AFTER_CLEANUP.txt` (generated tree dump)
  and `docs/PROJECT_IMPLEMENTATION_AUDIT.md` (unfilled audit template).
- Clarified current OpenRouter task execution versus retained inactive provider
  adapters in README and provider/routing/quickstart/troubleshooting docs;
  labeled broad architecture and prior audit/roadmap documents historical.
- Smoke checks: CLI help, provider help, interactive `/models` and `/exit`,
  imports, and source compileall succeeded. Full suite: 1,616 passed, 4
  skipped, 20 failed. Offline package build could not resolve uncached
  `hatchling`.
- No application behavior or provider implementation was removed. No commit
  was made.
