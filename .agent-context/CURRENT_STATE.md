# Current State

Last checked: 2026-10-19 (OpenRouter Provider Investigation & Multi-Provider Fallback)

## OpenRouter Provider Investigation & Multi-Provider Fallback (2026-10-19)
- **Investigation & Root Cause Analysis**:
  - Investigated OpenRouter dashboard evidence (294 requests, 1.88M tokens, $0.00 spend across `:free` models: North Mini Code, Gemma, Nemotron, LFM).
  - Confirmed that average request volume (~6,395 tokens/request) is standard for an autonomous loop with prompts, directory context, tool schemas, and file contents.
  - Confirmed $0.00 spend reflects 100% `:free` model usage, which is subject to OpenRouter's account-level daily free-tier request limits (~200 requests/day for zero-deposit accounts).
  - Identified critical architectural gap: `create_providers()` previously defaulted to only `[{"id": "openrouter", "enabled": True}]` when `.harness/config.yaml` was absent, ignoring credentials present in `.env` (such as `GROQ_API_KEY`, `NVIDIA_API_KEY`). As a result, the fallback chain consisted entirely of models hosted on the single provider `openrouter`. When OpenRouter hit an account-wide limit, model rotation was futile because all requests hit the same exhausted API key.
- **Multi-Provider Auto-Discovery**:
  - `src/harness_core/providers/factory.py`: added `auto_discover: bool = False` to `_provider_configs()` and `create_providers()`. When enabled, dynamically discovers all configured providers with valid credentials from `CredentialResolver` (`openrouter`, `groq`, `gemini`, `anthropic`, `openai`, `nvidia`).
  - `src/harness_core/cli/interactive.py` and `main.py`: passes `auto_discover=True` when loading project configuration so available credentials immediately provide genuine multi-provider fallback.
- **Provider Diversity in Fallback Chains**:
  - `src/harness_core/routing/router.py`: `_build_free_chain` enforces provider diversity across candidate models when multiple providers offer free models, ensuring alternative providers (e.g. Groq) are included rather than filling all chain slots with a single provider.
- **Exhausted Provider Fast Failover**:
  - `src/harness_core/routing/fallback.py`: tracks `exhausted_providers`. When a provider hits an account-wide quota limit (`is_account_quota`), subsequent candidate models belonging to that exhausted provider are skipped immediately without redundant HTTP requests, jumping directly to alternative non-exhausted providers in the chain.
- **Deterministic Testing**:
  - Added `test_account_quota_skips_same_provider_and_fails_over_to_distinct_provider` and `test_free_chain_ensures_provider_diversity` in `tests/unit/test_openrouter_reliability_audit.py`.
  - Added `test_create_providers_auto_discovers_configured_credentials` in `tests/unit/test_provider_factory.py`.
  - Full suite: **1,755 passed, 1 skipped, 0 failures** in 79.35s.

## Task Completion Reliability & Honest Plan Reporting (2026-10-19)
- **Timeout Immediate Failover**:
  - `FallbackConfig`: reduced `model_attempt_timeout_seconds` from 90s to 45s, and `total_timeout_seconds` from 120s to 90s.
  - `FallbackEngine`: when a request times out (`ERROR_CATEGORY_TIMEOUT`), the engine immediately breaks out of retry loops and fails over to the next candidate model in the chain without wasting another 45-90 seconds retrying the exact same hung endpoint.
- **Provider Quota & Account-Wide Rate Limit Exhaustion**:
  - Expanded detection for provider-wide and account-level daily quotas (`free model limit`, `daily rate limit`, `quota exceeded`, `resource_exhausted`, `free-models-per-day`).
  - When quota exhaustion is detected and all remaining candidate models in the chain share the same provider, fails over promptly with clear guidance rather than stalling through redundant 429 requests.
- **Bounded Interactive Task Deadlines**:
  - `EngineeringRuntime.execute_interactive`: catches `(asyncio.TimeoutError, TimeoutError)` cleanly, marks tasks as failed with clear timeout blockers, and returns a truthful `RuntimeOutcome` instead of crashing or hanging indefinitely.
  - `InteractiveShell`: passes a 240.0s task deadline default to `execute_interactive`.
- **Honest Plan Item Reconciliation & Falsification Prevention**:
  - `apply_tool_result`: file editing tools (`edit_file`, `write_file`) activate implementation TODOs to `IN_PROGRESS` with evidence, never prematurely marking them `COMPLETED` on intermediate edits.
  - `reconcile_on_evidence`:
    - Added `task_failed: bool = False` flag.
    - When a task fails, aborts, or pauses: implementation TODOs are NEVER marked completed without verification; any unverified completed items are reverted to `FAILED` with `unverified=True` and explicit reason.
    - Implementation items only complete during successful reconciliation when files were modified AND verification/tests have not failed.
- **Diff Preview & Failure Reporting Truthfulness**:
  - `ConversationRenderer._render_diff_preview`: generates balanced diff preview showing both removed (`-`) and added (`+`) lines instead of only displaying deletions.
  - `ConversationRenderer._render_plan`: displays unverified items honestly with warning mark `!` (yellow), never a misleading green `✓` when a task fails.
  - `InteractiveShell` & `ConversationRenderer.render_completion`: explicitly report partial edits and list the unverified files, and clearly state whether tests and verification were run or stopped beforehand.
- **Test Baseline**:
  - `tests/unit/test_task_completion_reliability.py`: 7 passed in 0.17s.
  - Core focused regression suite (104 tests across failover, invariant, ux, routing, windows): 100% passed.

## Windows Coding Reliability & Shell Execution (2026-10-19)
- **Unified Diff Accuracy**:
  - Fixed `EditFileTool.build_diff()` to split lines without `keepends=True`, use `lineterm=""`, and join with `"\n"`. Prevents merged diff lines (e.g. `-old;+new;`) when files lack trailing newlines, ensuring exact added/removed counts and clean diff preview boxes.
- **Windows-Aware Execution & Normalization**:
  - `RunCommandTool`:
    - Executes via PowerShell (`powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ...`) with `$LASTEXITCODE` / `$?` returncode propagation and alias compatibility (`ls`, `cat`, `type`). Falls back to `cmd.exe /c`.
    - Automatically strips redundant leading `cd <workspace> &&` / `cd <workspace>;` prefixes pointing to `cwd` or `.`.
    - Normalizes POSIX drive notation `/c/Users/...` to `C:/Users/...`.
    - Unescapes Unix-style backslash space escapes (`\ ` -> ` `).
    - Rewrites `python3` to `python` when `python3` is not found on Windows PATH.
- **Failure Diagnostics & Anti-Pattern Redirection**:
  - `classify_command_failure` identifies shell file inspection attempts (`cat`, `type`, `head`, `tail`, `ls`, `dir`) as `file_inspection_via_shell`.
  - In `_postprocess_result` and `RunCommandTool`: returns immediate actionable guidance redirecting the model to use dedicated tools (`read_file`, `list_files`, `grep`) instead of looping through shell commands.
- **Windows System Prompt Context**:
  - `AgentLoop._system_prompt()` injects explicit `WINDOWS EXECUTION ENVIRONMENT` rules when `platform.system() == "Windows"`:
    - Specifies PowerShell shell environment and states that `run_command` already runs in workspace root (no `cd`).
    - Mandates double-quoting paths with spaces and forbids Unix backslash escapes (`New\ folder`).
    - Enforces dedicated file tools (`read_file`, `list_files`) over shell inspection commands.
- **Provider Fallback & Shared Quota Handling**:
  - Short-circuits fallback when explicit account-wide free-tier limit or daily quota exhaustion (`free model limit`, `daily rate limit`, `quota exceeded`) is detected, avoiding futile 5-minute timeout loops across identical provider endpoints.
  - Interactive shell displays clear quota guidance and surfaces partial changes saved when tasks stop prematurely.
- **Test Baseline**:
  - `tests/unit/test_windows_reliability.py`: 16 passed in 1.96s.
  - Core focused suite (125 tests across UX, failover, MCP, instructions, shell, git): 125 passed in 25.41s.
- **Live Status & Non-Blocking Spinner**:
  - `ConversationRenderer` animated spinner (`⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏`) smoothly updating at 10 fps via monotonic clock without separate threads or race conditions.
  - Multi-stage progress indicators driven by real observable events: `Connecting…`, `Considering…`, `Investigating…`, `Reading files…`, `Editing code…`, `Running tests…`, `Preparing response…`.
  - Terminal cleanup: `stop()` guarantees `show_cursor(True)` and clean shutdown of Live region on finish, failure, cancel, or Ctrl+C.
- **Truthful Token Accounting**:
  - Distinguishes current-turn generated output tokens (`↓ ...`) from total session and input tokens.
  - Formatted cleanly (`120 tokens`, `1.2k tokens`). Provider-reported usage is labelled `exact`. Approximated or streaming character heuristics (`len(content)//4`) are explicitly labelled `(est)` (e.g. `↓ ~1.2k tokens (est)`).
  - `AgentLoop` populates `task.token_usage`, `task.total_input_tokens`, `task.total_output_tokens`, `task.tokens_estimated` and emits `model.completed` with exact/estimated flags.
- **Live Code-Change Visibility (Diff Preview)**:
  - `WriteFileTool` and `EditFileTool` calculate exact `added` and `removed` line counts and unified diff snippets in `ToolResult.metadata`.
  - `AgentLoop` records file change statistics on `task.file_change_stats` and `task.files_changed`, emitting `file.changed` events.
  - `ConversationRenderer` renders a compact 2-5 line diff preview box (`src/app.py (+18 -6)`) with syntax-highlighted `-` and `+` lines adapting dynamically to terminal width.
- **Live Test Results**:
  - `AgentLoop` parses test suite outputs (pytest, unittest, npm) into passed/failed counts, emitting `test.completed`.
  - Live status renders `✓ Tests completed  12 passed` or `✗ Tests failed  8 passed, 2 failed`.
- **Turn Completion Summary**:
  - Structured summary showing status, elapsed time, tool calls, model chain, exact vs estimated tokens (`Tokens: 3.2k in, 850 out (exact)`), file changes with per-file `(+added -removed)` stats, and test run outcomes.
- **Test Baseline**:
  - Deterministic suite: `tests/unit/test_claude_ux_parity.py` (12 passed in 1.02s).
  - All UX-related unit tests: 106 passed in 12.82s.
  - Core integration & parity suite: 153 passed in 22.11s.

## MCP Security & Lifecycle Audit (2026-10-19)
- **Arbitrary & Nested Path Confinement**:
  - Implemented `AgentLoop._inspect_mcp_path_confinement()` to recursively inspect all MCP parameter keys and values.
  - Secures arbitrary path parameter keys (`dest`, `source`, `target`, `filepath`, `dir`, etc.), nested dictionaries (`{"opts": {"target": "..."}}`), and list values against directory traversal (`../`), absolute paths (`/etc/passwd`), Windows drive paths (`C:\Windows\System32`), and UNC network paths (`\\server\share`).
  - Server configuration `cwd` in `.mcp.json` is strictly confined within `workspace_root`; escapes fallback safely to workspace root.
- **Process Lifecycle & Zombie/Handle Leak Prevention**:
  - `MCPClient.shutdown_server()` terminates process, closes `stdin`/`stdout`/`stderr` streams, and unregisters from `_processes` to prevent file handle exhaustion.
  - Registered `atexit.register(client.shutdown_all)` so all child MCP servers are terminated cleanly on interpreter exit or crash.
  - Startup failure or initialization timeout terminates spawned subprocesses immediately.
  - JSON-RPC protocol desynchronization guard: `_send_request` matches request `id`, safely skipping unsolicited notifications (`id: null`) or mismatched responses.
- **Permission & Tool Shadowing Protection**:
  - Core/native tools are protected against shadowing or hijacking by untrusted MCP servers in `AgentLoop.__init__`. MCP tools cannot overwrite existing tools or core aliases (`bash`, `read_file`, etc.).
  - Enforced `tool.schema.permission_required` (`deny` or `ask` requiring `request_approval`) in `_execute_tool_checked()`.
  - Non-autonomous (interactive) mode strictly denies unapproved MCP tool calls.
- **Comprehensive Secret Redaction**:
  - `MCPToolAdapter._sanitize_text()` scrubs server configuration arguments (sensitive flags like `--token`, `-t`, `--key`, `-k`, `--password`, `-p`, `--secret`), ambient system environment variables (`TOKEN`, `KEY`, `SECRET`, `PASSWORD`, `AUTH`), and API key/bearer token patterns (`sk-`, `sk_`, `Bearer ...`).
- **Test Baseline**:
  - Focused MCP suite: **30 passed in 3.71s** (`tests/unit/test_mcp_tool_integration.py`).
  - Full unit test suite: **1,717 passed, 1 skipped, 0 failed** in 74.17s.

## Project Instructions Discovery & Injection (2026-10-19)
- **Automatic Instruction Discovery in System Prompt**:
  - Implemented `AgentLoop._load_project_instructions()` to discover `.harness/instructions.md`, `AGENTS.md`, and `CLAUDE.md`.
  - Enforced deterministic precedence: `.harness/instructions.md` > `AGENTS.md` > `CLAUDE.md`.
  - Workspace path confinement check: paths verified to resolve strictly inside `workspace_root`; directory traversal (`../`) and escaping symlinks rejected.
  - SHA-256 content deduplication prevents injecting duplicated instruction blocks.
  - Bounded token/character budget with clean truncation notice (`[... Project instructions truncated to budget ...]`).
  - Section framing: `[{rel_name}]\n{content}`.
  - Zero-file fallback: prompt unchanged when no instruction files exist.
  - Test suite: `tests/unit/test_project_instructions.py` (10 passed in 0.54s).

## Native MCP Tool Integration (2026-10-19)
- **Adapter & Runtime Lifecycle**:
  - `MCPToolAdapter` (`src/harness_core/mcp/adapter.py`): Canonical `Tool` subclass translating MCP tools into Harness `ToolSchema`, normalizing JSON schema parameters, and wrapping `MCPClient.call_tool()`.
  - Handles MCP content blocks, `isError`, and JSON-RPC errors, returning canonical `ToolResults.success()` or `ToolResults.error()`.
  - Secret redaction: Server environment variables (`env`) are scrubbed from outputs and errors.
  - Configuration discovery (`src/harness_core/mcp/manager.py`): Supports `.harness/config.yaml` (`mcp:` section, both `servers:` map and flat map) and standard root `.mcp.json` (`mcpServers:`), with `.harness/config.yaml` taking deterministic precedence.
  - Deterministic namespacing & collision avoidance: Tools assigned primary name `mcp_{server}_{tool}`. Unambiguous bare tool names exposed as aliases ONLY when no collision with native tools or other MCP servers exists.
  - Execution guards: Path confinement (`_confine_call`), workspace boundary checks, and timeout handling apply to MCP tools identically to native tools.
  - CLI integration: `harness mcp list` reads workspace-configured servers; `harness tools list` shows core + MCP tools; `InteractiveShell` discovers tools and ensures clean shutdown (`mcp_client.shutdown_all()`).
  - Optionality: Zero overhead and identical behavior when no MCP servers are configured.

## REA Reverse Engineering & Claude Code Parity Sprint (2026-10-19)

- **Claude Code Reverse Engineering via REA & Observable Artifacts**:
  - REA (`rea-agents@6.7.0`) verified in environment; CLI engine verified healthy.
  - Claude Code (`2.1.302`) inspected via CLI, schema manifest, and active session transcript (`10a1d624-7826-4e9c-a02c-d75547490122.jsonl`).
  - Mapped Claude Code tool conventions (`Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`), turn grouping, thought preservation, and XML/markdown fallback tool calling to Harness.
- **AgentLoop Reliability, Hardening & Compatibility Audit**:
  - **Tool & Argument Aliasing Guardrails**: Direct tool registrations take strict precedence over aliases. Aliases resolve ONLY if the target tool is actually registered in the active runtime (e.g. read-only roles cannot have 'Bash' silently resolve). Argument normalization handles `old_content`/`new_content`, `search`/`replace`, `cmd`, and `file_path` without overwriting existing arguments.
  - **Safe Content Tool Extraction & No Quoted Examples**: Quoted examples (preceded by 'for example', 'e.g.', 'sample', etc.), normal JSON responses (with 'status', 'items', 'result'), and invalid schemas are ignored. Replaced naive backslash replacement with targeted drive-letter regex escaping so `\n`, `\t`, and quotes are never corrupted in string content.
  - **OpenRouter Defensive Health Check**: `OpenRouterProvider.health_check()` handles `None` or malformed payloads safely, accurately reports `remaining == 0` on free tier as unhealthy, while ensuring paid and unconstrained accounts remain healthy.
  - **Workspace Search Policy**: Excludes workspace-internal hidden directories (`.git`, `.venv`, `.cache`) while allowing workspace root to exist inside dot-prefixed parents and allowing intentional searches for dot-files (`.env`, `.gitignore`).
- **Test Baseline (Audited & Verified)**:
  - Full unit test suite: **1,677 passed, 1 skipped, 0 failed** in 170.18s.
  - Non-live E2E suite: **88 passed, 1 skipped, 0 failed** in 23.92s.
  - Parity & audit regression test suite (`tests/unit/test_agent_execution_parity.py`): **11 passed, 0 failed**.
  - CLI smoke tests: `uv run harness --help` and import smoke test passed cleanly.


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
