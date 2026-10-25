# Agent Context Changelog

## 2026-10-19 — OpenRouter Provider Investigation & Multi-Provider Fallback

- **Multi-Provider Auto-Discovery**:
  - `src/harness_core/providers/factory.py`: added `auto_discover: bool = False` to `_provider_configs()` and `create_providers()`. Automatically registers all providers with valid credentials in `CredentialResolver` when no explicit providers section is configured in `config.yaml`.
  - `src/harness_core/cli/interactive.py` & `main.py`: passes `auto_discover=True` on startup.
- **Provider Diversity in Free Routing**:
  - `src/harness_core/routing/router.py`: `_build_free_chain` interleaves distinct available providers when constructing fallback chains, preventing a single provider from monopolizing all fallback positions.
- **Exhausted Provider Fast Failover**:
  - `src/harness_core/routing/fallback.py`: tracks `exhausted_providers`. On detecting account-wide quota limits (`is_account_quota`), skips all remaining candidate models from the exhausted provider immediately, failing over cleanly to alternative providers.
- **Deterministic Testing**:
  - Added tests verifying multi-provider auto-discovery, quota skip behavior, and chain diversity in `tests/unit/test_openrouter_reliability_audit.py` and `tests/unit/test_provider_factory.py`.
  - Full suite: **1,755 passed, 1 skipped, 0 failures** in 79.35s.

## 2026-10-19 — Task Completion Reliability & Honest Plan Reporting

- **Timeout Immediate Failover**:
  - `FallbackConfig`: bounded `model_attempt_timeout_seconds` to 45s and `total_timeout_seconds` to 90s.
  - `FallbackEngine`: when a request times out (`ERROR_CATEGORY_TIMEOUT`), immediately emits cooldown and failover to next model in chain without wasting another 45-90 seconds retrying the same hung endpoint.
- **Quota & Shared Rate Limit Fail-Fast**:
  - Expanded detection for provider-wide and account-level daily quotas (`free model limit`, `daily rate limit`, `quota exceeded`, `resource_exhausted`, `free-models-per-day`).
  - Aborts early when all remaining candidate models share the same provider with exhausted quota.
- **Bounded Interactive Task Deadlines**:
  - `EngineeringRuntime.execute_interactive`: catches `(asyncio.TimeoutError, TimeoutError)` cleanly, marks tasks as failed with timeout blockers, and returns a truthful `RuntimeOutcome`.
  - `InteractiveShell`: passes a 240.0s task deadline default to `execute_interactive`.
- **Honest Plan Item Reconciliation & Falsification Prevention**:
  - `apply_tool_result`: file editing tools (`edit_file`, `write_file`) activate implementation TODOs to `IN_PROGRESS` with evidence, never prematurely marking them `COMPLETED` on intermediate edits.
  - `reconcile_on_evidence`:
    - Added `task_failed: bool = False` flag.
    - When a task fails, aborts, or pauses: implementation TODOs are NEVER marked completed without verification; any unverified completed items are reverted to `FAILED` with `unverified=True`.
    - Implementation items only complete during successful reconciliation when files were modified AND verification/tests have not failed.
- **Diff Preview & Failure Reporting Truthfulness**:
  - `ConversationRenderer._render_diff_preview`: generates balanced diff preview showing both removed (`-`) and added (`+`) lines.
  - `ConversationRenderer._render_plan`: displays unverified items honestly with warning mark `!` (yellow), never a misleading green `✓` when a task fails.
  - `InteractiveShell` & `ConversationRenderer.render_completion`: explicitly report partial edits and list the unverified files, and clearly state whether tests and verification were run or stopped beforehand.
- **Test Baseline**:
  - `tests/unit/test_task_completion_reliability.py`: 7 passed in 0.17s.
  - Core focused regression suite (104 tests across failover, invariant, ux, routing, windows): 100% passed.

## 2026-10-19 — Windows Coding Reliability & Shell Execution

- **Unified Diff Accuracy Fix**:
  - `EditFileTool.build_diff()` split without `keepends=True`, uses `lineterm=""`, and joins with `"\n"`. Eliminates line-merging defects (e.g. `-old;+new;` concatenated on the same line) when target files lack trailing newlines. Ensures addition/deletion counts match actual patches.
- **Windows-Aware Execution & Normalization**:
  - `RunCommandTool`:
    - Executes via PowerShell (`powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ...`) with `$LASTEXITCODE` / `$?` returncode propagation and alias compatibility (`ls`, `cat`, `type`).
    - Automatically strips redundant leading `cd <workspace> &&` / `cd <workspace>;` prefixes pointing to `cwd`.
    - Normalizes POSIX drive notation (`/c/Users/...` -> `C:/Users/...`).
    - Unescapes Unix-style backslash space escapes (`\ ` -> ` `).
    - Rewrites `python3` to `python` when `python3` is missing on Windows PATH.
- **Failure Diagnostics & Anti-Pattern Redirection**:
  - `classify_command_failure` detects shell file inspection attempts (`cat`, `type`, `head`, `tail`, `ls`, `dir`) as `file_inspection_via_shell`.
  - In `_postprocess_result` and `RunCommandTool`: returns immediate actionable guidance redirecting the model to use dedicated tools (`read_file`, `list_files`, `grep`) instead of looping through shell commands.
- **Windows System Prompt Context**:
  - `AgentLoop._system_prompt()` injects explicit `WINDOWS EXECUTION ENVIRONMENT` rules when `platform.system() == "Windows"` (PowerShell shell, `run_command` already in workspace, double-quoting paths with spaces, dedicated file tools over shell inspection).
- **Provider Fallback & Shared Quota Handling**:
  - Short-circuits fallback when explicit account-wide free-tier limit or daily quota exhaustion (`free model limit`, `daily rate limit`, `quota exceeded`) is detected, avoiding futile 5-minute timeout loops across identical provider endpoints.
  - Interactive shell displays clear quota guidance and surfaces partial changes saved when tasks stop prematurely.
- **Test Baseline**:
  - `tests/unit/test_windows_reliability.py`: 16 passed in 1.96s.
  - Core focused suite (125 tests across UX, failover, MCP, instructions, shell, git): 125 passed in 25.41s.

## 2026-10-19 — Claude Code Terminal UX Parity

- **Live Status & Non-Blocking Multi-Stage Spinner**:
  - Upgraded `ConversationRenderer` with animated spinner (`⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏`) driven smoothly by monotonic clock (10 fps) without separate threads.
  - Multi-stage indicators reflecting real observable runtime state: `Connecting…`, `Considering…`, `Investigating…`, `Reading files…`, `Editing code…`, `Running tests…`, `Preparing response…`.
  - Guaranteed terminal cleanup on stop, interruption, or exceptions: restores cursor visibility and cleanly terminates Rich Live region.
- **Truthful Token Accounting**:
  - Live turn token indicator (`↓ 1.2k tokens` or `↓ ~1.2k tokens (est)`) distinguished from total session tokens and input context.
  - Provider-reported usage is labelled `exact`. Approximated or streaming character heuristics are explicitly labelled `(est)`.
  - Wired provider response usage in `AgentLoop` onto `task.token_usage`, `task.total_input_tokens`, `task.total_output_tokens`, `task.tokens_estimated`, emitting `model.completed` with metrics.
- **Live Code-Change Visibility (Diff Preview)**:
  - `WriteFileTool` and `EditFileTool` return line change statistics (`added`, `removed`) and unified diff snippets in `ToolResult.metadata`.
  - `AgentLoop` tracks per-file additions/deletions on `task.file_change_stats` and emits `file.changed` events.
  - `ConversationRenderer` renders a compact 2-5 line diff preview box (`src/app.py (+18 -6)`) with colored `-` and `+` changes and dynamic terminal width adaptation.
- **Live Test Results**:
  - `AgentLoop` parses test suite outputs (pytest, unittest, npm) into passed/failed counts, emitting `test.completed`.
  - Live status displays `✓ Tests completed  12 passed` or `✗ Tests failed  8 passed, 2 failed`.
- **Structured Turn Completion Summary**:
  - Status, elapsed duration, tool call count, model routing chain, honest token usage (`Tokens: 3.2k in, 850 out (exact)`), file changes with line stats, and test run outcomes.
- **Test Baseline**:
  - Dedicated deterministic suite: `tests/unit/test_claude_ux_parity.py` (12 passed).
  - All UX tests: 106 passed.
  - Core integration & parity suite: 153 passed.

## 2026-10-19 — MCP Security & Lifecycle Audit Hardening

- **Security & Lifecycle Audit of MCP Subsystem**:
  - Implemented recursive path confinement `_inspect_mcp_path_confinement()` in `AgentLoop`: verifies all parameter keys and values against directory traversal (`../`), absolute paths (`/etc/passwd`), Windows drive paths (`C:\...`), and UNC network paths (`\\server\share`).
  - Hardened server configuration parser `_parse_server_config()`: server working directory `cwd` in `.mcp.json` is strictly confined within `workspace_root`.
  - Process lifecycle & leak prevention: `MCPClient.shutdown_server()` terminates child processes, closes `stdin`/`stdout`/`stderr` streams, and unregisters handles; `atexit.register(client.shutdown_all)` ensures cleanup on interpreter exit. Initialization failure terminates spawned subprocesses immediately.
  - JSON-RPC protocol desynchronization guard: `_send_request` enforces matching request `id`, ignoring unsolicited notifications (`id: null`) or stale responses.
  - Permission & tool shadowing protection: `AgentLoop.__init__` protects core native tools and aliases against collision or hijacking by external MCP tools. Enforced `tool.schema.permission_required` in `_execute_tool_checked()`; non-autonomous mode blocks unapproved MCP calls.
  - Comprehensive secret redaction: expanded `MCPToolAdapter._sanitize_text()` to scrub sensitive CLI arguments (`--token`, `-t`, `--key`, `-k`, `--password`, `-p`, `--secret`), ambient system environment secrets (`TOKEN`, `KEY`, `SECRET`, `PASSWORD`, `AUTH`), and API key/bearer patterns (`sk-`, `sk_`, `Bearer ...`).
  - Added 13 deterministic adversarial security tests to `tests/unit/test_mcp_tool_integration.py` (total 30 tests in file, all passing).
  - Full unit test suite: **1,717 passed, 1 skipped, 0 failed** in 74.17s.

## 2026-10-19 — Project Instruction Discovery & Native MCP Tool Integration

- **Project Instruction Discovery**:
  - Implemented automatic discovery of `.harness/instructions.md`, `AGENTS.md`, and `CLAUDE.md` in `AgentLoop._load_project_instructions()`.
  - Added deterministic precedence, strict workspace confinement checks, SHA-256 deduplication, bounded character budgeting (default 4,000 tokens), and clean section framing.
  - Zero-instruction fallback preserves original system prompt behavior.
  - 10 unit tests added in `tests/unit/test_project_instructions.py`.
- **Native MCP Tool Integration**:
  - Implemented `MCPToolAdapter` (`src/harness_core/mcp/adapter.py`) subclassing canonical `Tool` interface.
  - Implemented `MCPManager` (`src/harness_core/mcp/manager.py`) discovering stdio MCP servers from `.harness/config.yaml` (`mcp:` section, both `servers:` map and flat map) and `.mcp.json` (`mcpServers:`).
  - Enforced deterministic namespacing (`mcp_{server}_{tool}`) and collision-free alias generation (bare alias allowed ONLY if no conflict with native tools or across MCP servers).
  - MCP tools pass through native execution controls: argument schema normalization, path confinement (`_confine_call`), cwd confinement, timeout, and permission rules (`PermissionRule(tool_pattern="mcp_", action="allow")`).
  - Secret redaction: Server environment variables (`env`) are scrubbed from outputs and errors.
  - Integrated into `AgentLoop`, `InteractiveShell`, and CLI commands (`harness mcp list`, `harness tools list`).
  - 17 unit and integration tests added in `tests/unit/test_mcp_tool_integration.py` including end-to-end execution against a mock stdio JSON-RPC MCP server.
  - Test results: Full unit suite **1,704 passed, 1 skipped, 0 failed** in 69.19s.

## 2026-10-19 — REA-Powered Claude Code Parity & Agent Execution Loop Hardening

- Root cause diagnosed (historical zero-tool-call E2E failure):
  1. OpenRouter `/key` endpoint returned HTTP 200 with `remaining: 0` daily free requests. `OpenRouterProvider.health_check()` previously treated any HTTP 200 as healthy, causing live E2E tests to run against exhausted accounts and immediately fail with HTTP 429 rate limit across all free models, resulting in zero tool calls.
  2. Open-source and free models frequently output Claude Code tool names (`Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`) and argument variations (`file_path`, `cmd`, `old_str`/`new_str`), which previously failed as `unknown_tool` or `missing_argument`.
  3. Models generating tool calls directly inside assistant text blocks (`<tool_call>...` or markdown codeblocks) were ignored if the provider adapter did not extract them into `message.tool_calls`.
  4. Parallel tool calls in a single turn were previously converted into multiple independent assistant turns, violating model provider conversation schema requirements.
- Core fixes implemented & audited:
  - `OpenRouterProvider.health_check()` now checks `free_model_daily_requests.remaining > 0` for free keys, accurately marking exhausted keys unhealthy; defensively handles `None` or malformed payloads without crashing.
  - `AgentLoop`: Hardened `_TOOL_ALIASES` with direct registration precedence and registered target validation. Argument normalization expanded to `old_content`/`new_content`, `search`/`replace`.
  - `AgentLoop`: Hardened `_extract_tool_calls_from_content()`: ignores quoted examples, skips ordinary JSON responses, requires registered tool names and dict arguments, and avoids blanket backslash replacement that corrupts `\n`, `\t`, and quotes.
  - `AgentLoop`: Tool argument parsing hardened against unparsed strings, markdown fences, and malformed JSON, with targeted Windows drive-letter repair.
  - `AgentLoop`: Turns preserve `thought`/reasoning and group parallel tool calls under a single assistant turn before providing matching tool results.
  - `search.py`: Fixed `_glob_via_walk` and `_grep_via_walk` hidden-directory pruning bug (`rel.parts[:-1]`), allowing workspaces in dot-named parents and enabling intentional dot-file searches (`.env`, `.gitignore`).
- Test results:
  - Full unit test suite: **1,677 passed, 1 skipped, 0 failed** in 170.18s.
  - Non-live E2E suite: **88 passed, 1 skipped, 0 failed** in 23.92s.
  - Parity & audit regression suite: **11 passed, 0 failed**. Smoke tests pass cleanly. No live credentials required.


## 2026-10-18 — Deterministic Git Push Workflow & Zero-Tool-Call Root Cause Resolution

- Root cause diagnosed (3.2s zero-tool-call failure): `WorkerAgent` wrapped the user's task goal in an agent instruction preamble (`You are a specialized agent: ... Objective: <goal>`), causing `classify_workflow()` in `workflows.py` to evaluate the prompt as length > 50 characters and miss push keywords, bypassing deterministic workflows and falling through to rate-limited LLM free models.
- Missing Git tools fixed: `AgentRole.GIT_RELEASE` in `agents/registry.py` was missing `git_*` tools in `allowed_tools`, causing `WorkerAgent._filter_tools()` to strip Git tools and fail with `"Tool not available: git_status"`. Added all Git tools to `GIT_RELEASE` and mapped `TaskIntent.GIT: AgentRole.GIT_RELEASE` in `runtime.py`. Also added Git tools to `AgentRole.CODER`.
- Empty repo log handling: Updated `GitLogTool` in `tools/git.py` to recognize empty repositories without initial commits (`fatal: your current branch 'main' does not have any commits yet`) and return success with `"(no commits yet)"` instead of exiting with an error.
- Deterministic push auto-configuration: `run_git_push_workflow()` in `workflows.py` unwraps `Objective:` prefixes, handles clean trees by proceeding directly to push/verify, automatically extracts remote URLs from user goals when unconfigured, and safely executes `git remote add origin <url>`.
- Acceptance scenario verified: Added `tests/unit/test_deterministic_git_push_acceptance.py`. Harness CLI runtime successfully executes end-to-end Git push from an uncommitted repo to a local bare remote with **zero LLM model calls** permitted (`FailingProvider` returning HTTP 429).
- Test results: Full unit suite **1,667 passed, 0 failed** in 116.24s. Targeted deterministic suite: 3 passed. No live API calls. No commit.

## 2026-10-18 — OpenRouter & Provider Reliability Audit: fixes & verification

- Root cause diagnosed: OpenRouter returned HTTP 403 for models with partner restrictions (`thinkingmachines/inkling-small:free`), which `fallback.py` and `http_adapters.py` treated as global provider auth failure (`AUTHENTICATION`), aborting the fallback chain and falsely claiming `OPENROUTER_API_KEY` was invalid.
- Dynamic fallback preserved: `router.py` now ensures dynamic fallback routes (`openrouter/free`) are kept at the end of the fallback candidate list rather than being sliced off by `max_fallback_chain`.
- Structured error classification: Added `ProviderErrorCategory.PERMISSION`, `CONTEXT_LENGTH`, `CONFIGURATION` to `base.py`; updated `fallback.py` to classify HTTP 403 as `PERMISSION_DENIED` and fail over cleanly with a 1-hour model cooldown instead of aborting the session.
- Secret redaction: Added `_safe_error_detail()` in `fallback.py` to scrub bearer tokens and API keys from error messages, logs, and summaries.
- Credential resolution: Enhanced `dotenv.py` and `credentials.py` to properly handle whitespace, empty environment strings, and cross-directory launches.
- Tests: +24 unit regression tests in `tests/unit/test_openrouter_reliability_audit.py`. Full suite: **1,767 passed, 7 skipped, 0 failed**. Smoke tests pass. No commit.

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
