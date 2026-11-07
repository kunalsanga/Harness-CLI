# Current State

Last checked: 2026-11-07 (Harness V2 Interactive Session History & Transcript Navigation)

## Harness V2 — Interactive Session History & Transcript Navigation (2026-11-07)
- **Interactive `/sessions` Browser & Transcript Navigation**:
  - Implemented keyboard-first interactive session browser using `prompt_toolkit` Application, `FormattedTextControl`, and ANSI styling from Rich.
  - Date-grouped session hierarchy (`Today`, `Yesterday`, `This Week`, `Month/Year`).
  - Keyboard navigation: Up/Down arrow (`k`/`j`) line movement, PageUp/PageDown (5-item jump), Home/End (first/last session).
  - Pressing `Enter` opens the selected session's full chronological transcript viewer:
    - User prompts highlighted with `❯`.
    - Live tool activity icons (`✓`/`✗`).
    - Markdown-rendered assistant responses.
    - Verification outcome and run duration metrics.
  - Transcript navigation: scrollable with Up/Down, PageUp/PageDown, Home/End.
  - Transcript search: `/` activates live search, typing buffers query, `Enter` jumps to next match, `n`/`N` navigates matches, `Esc` clears search.
  - Strict read-only guarantee: 0 LLM calls, 0 tool executions, 0 file or state mutations.
  - Back navigation: `Esc`, `q`, or `Backspace` returns from transcript viewer back to session browser.
  - Explicit Resume dialog: pressing `R` opens `[Y/n]` resume confirmation with workspace reconciliation recap; pressing `N`/`Esc` cancels back to browser; `Y`/`Enter` executes resumption and context restore.
- **Accurate Derived Session Status**:
  - Implemented `derive_session_status(session, storage, current_session_id)`:
    - Eliminates false `Active` mislabeling on historical sessions.
    - Derives `Completed`, `Failed`, `Paused`, `Interrupted`, `Active`, and `Archived` based on persisted run outcomes and memory identity.
  - Updated SQLite session lifecycle: transitions active sessions to `completed` or `failed` on run completion.
- **Verification Evidence**:
  - `tests/unit/test_session_browser.py`: **36/36 passed in 2.77s**.
  - `tests/unit/test_readonly_explain_regression.py`: **16/16 passed in 0.27s**.
  - `tests/e2e/test_interactive_acceptance_hardening.py`: **16/16 passed in 2.83s**.
  - CLI smoke test: `uv run harness --help` and import check passed.
  - Validated against live database `~/.harness/sessions.db`: confirmed session `17006d8db86d` is accurately marked `Completed`, `f44265f95ddb` marked `Failed`, and abandoned sessions marked `Interrupted`.

## Read-Only Explanation Task Regression Resolution & Windows Shell Reliability (2026-11-07)
- **Root Cause Analysis on Mission Failure (Session `f44265f95ddb`)**:
  - Investigated the regression on read-only goal (`explain this project` against workspace `app1`): 35s execution duration, 31 tool calls, 32.3k input tokens, provider timeout, contradictory ladder state, and false verification failure.
  - Confirmed and resolved 5 distinct architectural and runtime root causes:
    1. **Consecutive Tool Aggregation Overwrite (`src/harness_core/cli/conversation.py`)**:
       - Fixed `complete_last()`: when a sub-call failed (e.g. attempting to read a directory), the entire aggregated summary was previously replaced with `✗ Inspected 21 files -> tool_error`, masking 20 successful reads.
       - Now truthfully preserves `✓ Inspected 20 files` and logs only the failed read as its own distinct item (`✗ Read ... -> error`), resetting the read streak cleanly.
    2. **Windows PowerShell Statement Chaining & Quoting (`src/harness_core/tools/shell.py`)**:
       - Implemented `translate_powershell_chain()`: converts bash/POSIX `&&` and `||` outside quotes to PowerShell 5.1 compatible syntax (`cmd1; if ($?) { cmd2 }`), eliminating `The token '&&' is not a valid statement separator in this version`.
       - Updated `normalize_windows_command()`: safely formats directory paths containing spaces with `Set-Location -LiteralPath '...'`, eliminating parameter binding errors on spaces (`New folder`).
       - Added early success return when commands normalize to a redundant `cd` without subprocess invocation.
    3. **Contradictory Plan Completion on Failed Tasks (`src/harness_core/agent/todos.py`)**:
       - Added `_SUMMARIZE = ("summarize", "synthesize", "report", ...)` category to `todos.py`. Read operations (`read_file`, `list_files`) no longer match or complete "Summarize findings".
       - Fixed `reconcile_on_evidence()` under `task_failed=True`: pending/in-progress items across ALL categories are failed, and any prematurely completed summary steps are reverted to `FAILED`, preventing contradictory ladders where all steps appear checked while the task failed.
    4. **Inappropriate Verification Failure on Read-Only Tasks (`src/harness_core/runtime/state.py` & `runtime.py`)**:
       - Added `VerificationStatus.SKIPPED = "skipped"`.
       - In `EngineeringRuntime`: when `contract.should_skip_verification()` is true, verification is marked `SKIPPED` (or `PASSED` when task completed cleanly), NEVER `FAILED`.
       - Renderers format `Verification: Not applicable (read-only task)` with dim styling, eliminating bogus red `Verification Failed (read-only explain task — no verification needed)`.
    5. **Read-Only Tool Loop Runaway & Token Bloat (`src/harness_core/agent/loop.py` & `filesystem.py`)**:
       - Bounded read-only tasks: injected stopping correction when `_readonly_tool_calls >= 12`, directing the model to synthesize findings into its final response.
       - Hard bounded at 18 read operations, preventing context accumulation beyond provider limits.
       - In `ReadFileTool`: added explicit diagnostic when target path is a directory (`Cannot read '...': it is a directory, not a file. Use list_files...`).
- **Verification Evidence**:
  - Created `tests/unit/test_readonly_explain_regression.py`: **16/16 passed in 0.21s**.
  - `tests/unit/test_intent_contract.py`: **31/31 passed in 1.77s**.
  - `tests/unit/test_task_completion_reliability.py`: **7/7 passed in 0.18s**.
  - `tests/unit/test_windows_reliability.py`: **16/16 passed in 2.03s**.
  - Full UX suite (`test_claude_ux_parity.py`, `test_tui_ux.py`, `test_ux_interaction.py`): **91/91 passed in 1.10s**.
  - Full unit test suite: **1,917 passed, 1 skipped, 0 failures** in 134.60s.
- **Live Interactive Reproduction Test Evidence (Session `17006d8db86d`)**:
  - Re-executed exact mission against workspace `C:\Users\kunal sanga\Desktop\New folder\app1` with goal `explain this project` via fresh interactive Harness shell session:
    - **Outcome**: `✓ Completed` in **19.6s** (previous run: `✗ Failed` in 35s).
    - **Tool Calls**: **1 tool call** (`List .`) (previous run: 31 runaway calls).
    - **Tokens**: **9.5k input tokens, 686 output tokens** (previous run: 32.3k input tokens, 97 output tokens — **71% reduction in input token consumption**).
    - **Quality of Response**: High-density, structured architecture explanation accurately detailing FastAPI backend, SQLite/SQLAlchemy models, endpoint contracts (`/api/v1/tasks`, `/health`, `/api/v1/stats`), React 18 / Vite / TypeScript frontend (`api.ts`, `types.ts`, `priority_filter.test.js`), and test suites.
    - **Verification State**: Truthfully marked as passed/skipped without emitting bogus `Verification Failed` banners.
    - **Session Integrity**: Session `17006d8db86d` persisted with `outcome: success`, `verification_passed: 1`; original failed session `f44265f95ddb` preserved intact.
    - **Zero Invariant Violations**: TaskForge untouched (`git status` unchanged), 0 git commits/pushes.

## Harness V2 — Real-World Dogfooding & TaskForge Validation (2026-11-07)
- **Autonomous Cross-Stack Implementation**:
  - Implemented priority filtering across FastAPI backend (`main.py`) and React/TypeScript frontend (`Dashboard.tsx`, `App.tsx`, `api.ts`, `types.ts`) in workspace `app1`.
  - Self-diagnosed and hardened 4 real engineering defects:
    1. Input validation: added strict rejection for invalid priority query parameters (`HTTP 400 Bad Request`).
    2. Statistical semantics: cleanly partitioned global distributions (`by_priority`, `global_by_status`) from filtered population metrics (`by_status`, `completion_rate`, `filtered_tasks_count`).
    3. Asynchronous race condition: added `AbortController` request cancellation to eliminate stale responses on rapid filter switching.
    4. TypeScript variable scoping: resolved declaration order issues in `Dashboard.tsx`.
- **Validation Evidence in Workspace (`app1`)**:
  - Backend pytest: **11/11 passed in 1.16s** (including invalid inputs, zero-match queries, regression checks).
  - Frontend type safety: `tsc --noEmit` passed with 0 errors.
  - Frontend automated unit tests: **5/5 passed in 31.6ms** (`node --test test/*.test.js`) verifying array filtering, empty states, completion calculations, URL formation, and `AbortController` cancellation.
  - Production build: `tsc && vite build` bundled 1,577 modules in 1.53s.
  - Live daemons: FastAPI (`http://127.0.0.1:8000`) and Vite (`http://127.0.0.1:5173`) running and verified with 8 seeded development tasks across all priority tiers.
- **Strategic Direction (Feature Freeze)**:
  - Freeze UI feature development; shift focus to **Session Interruption & Recovery Validation**, **3-Project Benchmark** (Python test debugging, multi-layer full-stack, unfamiliar repository), and empirical reliability metric tracking.

## Harness V2 — Interactive Acceptance Hardening & Defect Resolution (2026-11-06)
- **Interactive Hardening & Verified Defect Resolution**:
  - `src/harness_core/agent/loop.py`:
    - Resolved `AttributeError: 'ToolCall' object has no attribute 'get'` in tool execution dispatch: `tool_calls_to_process` handles both raw dictionary structures and typed `ToolCall` instances, safely reading `call_id = getattr(tool_call_data, "id", "")`, `tool_name = tool_call_data.tool_name`, and `raw_args = tool_call_data.arguments`.
  - `src/harness_core/cli/interactive.py`:
    - Fixed `/diff` command: guaranteed that newly created and untracked files (`git status --porcelain`) are ALWAYS prominently listed alongside tracked unified diffs, rather than being silenced when tracked files were also modified.
    - Added `highlight=False` on untracked file output so console syntax highlighting does not inject ANSI escape artifacts into filenames or path slashes.
  - `src/harness_core/cli/session_browser.py`:
    - Fixed `Text.append()`: replaced invalid `highlight=False` kwargs with `Text.from_markup()` on the command palette footer.
    - Fixed plain mode session browser: added `markup=False` when printing session IDs to prevent Rich from stripping square bracket session tags.
- **Full Acceptance Suite Verification (`tests/e2e/test_interactive_acceptance_hardening.py`)**:
  - Section 1 (Fresh Launch): 3/3 passed. Verified clean welcome wordmark with session tag and 0 recap dumping across both empty and pre-populated databases.
  - Section 2 (Session Browser): 3/3 passed. Verified date grouping (`Today`, `Yesterday`, `This Week`), exact timestamps, status badges, and search filtering across titles, IDs, and past prompt contents with real SQLite storage.
  - Section 3 (Historical Viewer): 2/2 passed. Verified chronological multi-run replay with exact prompts, tool calls, Markdown outputs, and run summary bars. Verified invariant: 0 LLM calls, 0 tool executions, database and workspace untouched.
  - Section 4 (Resume Matrix): 3/3 passed. Verified explicit resumption for completed, paused, and missing workspace sessions.
  - Section 5 (Live Execution & Dynamic Ladder): 1/1 passed. Verified real file creation, file modification, test execution, test repair, ladder state progression, and tool call event capture.
  - Section 6 (File Activity & Diff): 2/2 passed. Verified tracked + untracked file visibility, line capping, and secret redaction (`sk-***[REDACTED]`, `Bearer ***[REDACTED]`).
  - Section 7 (Provider Failover): 1/1 passed. Verified transparent stage updates (`Failing over to Groq (llama-3.3-70b-versatile)...`) upon 429 rate limit cooldown.
  - Section 8 (Performance & Stability): 1/1 passed. Verified rapid tool call deduplication (20 start events collapsed to 1 line, zero frozen spinner artifacts on stop).
- **Test Evidence**:
  - `tests/e2e/test_interactive_acceptance_hardening.py`: **16/16 passed in 2.79s**.
  - `tests/unit/test_session_browser.py`: **18/18 passed in 1.21s**.
  - Full targeted acceptance & regression suite: **237 passed in 7.78s**.
  - Full unit test suite: **1,901 passed, 1 skipped, 0 failures** in 118.70s.
  - CLI smoke test: `uv run harness --help` and import check passed.
- **Root Cause Resolution ("Configured models are rate limited")**:
  - `allow_paid_models` filtering was dropping all models when `NvidiaProvider` and `GroqProvider` had `is_free=False`, leaving only OpenRouter free models in the candidate chain when running with default settings.
  - Set `is_free=True` for NVIDIA NIM and Groq developer endpoints (cost 0.0), unlocking all 86+ NVIDIA models and Groq models under both `auto` and `free` routing modes.
  - Fixed `factory._provider_configs()`: auto-discovery now merges discovered credentials with custom project config overrides, ensuring all available providers remain eligible.
  - Fixed `FallbackEngine.classify_error()`: maps token context overflows to `CONTEXT_OVERFLOW` to avoid false rate-limit cooldowns.
  - Implemented cross-provider candidate interleaving in `ModelRouter.select_models()` and `_build_free_chain()`: candidate 1 is Provider A, candidate 2 is Provider B, candidate 3 is Provider C, ensuring immediate cross-provider failover on the very first failure.
- **Azure OpenAI Adapter**:
  - `src/harness_core/providers/azure.py`:
    - Full `AzureOpenAIProvider` subclassing `ModelProvider`.
    - Supports native Azure deployments (`/openai/deployments/{deployment}/chat/completions?api-version=...`), OpenAI-compatible paths (`/chat/completions`), `api-key` and Bearer token headers, tool calling, streaming, health check, and error category normalization.
  - Registered `azure` in `_PROVIDER_ENV_KEYS`, `_PROVIDER_BASE_URL_ENVS`, `_PROVIDER_DEFAULT_MODEL_ENVS` in `credentials.py`, and `factory.py`.
- **CLI Commands Auto-Discovery**:
  - `src/harness_core/cli/main.py`: updated `harness doctor`, `harness models list`, `harness models benchmark`, and `harness eval` to pass `auto_discover=True` to `create_providers()`.
- **Verification Evidence**:
  - `tests/unit/test_azure_adapter.py`: 7/7 passed.
  - `tests/unit/test_universal_provider_failover.py`: 20/20 passed (covering all 20 failover & resilience scenarios).
  - `tests/unit/test_free_routing.py`: 19/19 passed.
  - `tests/unit/test_security_audit.py`: 21/21 passed.
  - `tests/unit/test_multi_provider_cli.py`: 5/5 passed.
  - Full unit test suite: 1,883 passed, 1 skipped.
  - Live CLI Verification: OpenRouter 429 quota exhaustion autonomously failed over to FreeLLMAPI / Groq on attempt 2 without halting. Priority routing (`HARNESS_PROVIDER_PRIORITY="groq,..."`) routed directly to Groq.

## Multi-Provider AI Routing, Fault Tolerance & Pre-Deployment Hardening (2026-11-01)
- **Multi-Provider Adapters & Factory Auto-Discovery**:
  - `src/harness_core/config/credentials.py`:
    - Added `freellmapi` (`FREELLMAPI_API_KEY`) and updated `nvidia` (`NVIDIA_NIM_API_KEY`, `NVIDIA_API_KEY`) in `_PROVIDER_ENV_KEYS`.
    - Added `_PROVIDER_BASE_URL_ENVS` mapping custom base URLs (`FREELLMAPI_BASE_URL`, `NVIDIA_NIM_BASE_URL`, `OPENAI_BASE_URL`, etc.).
  - `src/harness_core/providers/freellmapi.py`:
    - Created `FreeLLMAPIProvider` subclassing `OpenAICompatibleProvider`. Implements `/models` and `/health` probes, context & tool inference, and fallback to `freellmapi/auto`.
  - `src/harness_core/providers/http_adapters.py`:
    - Enhanced `OpenAIProvider`: custom base URL (`OPENAI_BASE_URL`), configurable model (`OPENAI_MODEL`), `/models` health checking, and chat/tool/context capability augmentation for GPT-4o, GPT-4, o1, o3-mini, and GPT-5 models.
    - Updated `_error_category()`: classifies 429 quota exhaustion (`"insufficient_quota"`, `"exceeded your current quota"`) as `PAYMENT_REQUIRED` before generic rate limiting.
  - `src/harness_core/providers/nvidia.py`:
    - Enhanced `NvidiaProvider`: supports `NVIDIA_NIM_API_KEY` / `NVIDIA_API_KEY`, `NVIDIA_NIM_BASE_URL` / `NVIDIA_BASE_URL`, `NVIDIA_NIM_MODEL` / `NVIDIA_MODEL`.
    - Augmented `list_models()` to accurately infer `supports_tools=True` and 128k context windows for instruct/chat/coder models (`llama-3.1`, `nemotron`, `mixtral`, etc.).
  - `src/harness_core/providers/factory.py`:
    - Added `freellmapi` and `nvidia` resolution.
    - Updated `_provider_configs()` with auto-discovery across all providers and priority sorting (`HARNESS_PROVIDER_PRIORITY` or `providers_priority` in config).
- **Intelligent Automatic Failover & Routing**:
  - `src/harness_core/routing/fallback.py`:
    - In `execute()`, when an authentication error (401/403) or quota exhaustion occurs on provider A, marks provider A exhausted and checks if other non-exhausted providers exist in `model_chain`. If alternative providers exist, emits `model.failover` and breaks to the next provider instead of aborting the entire task.
    - For model-specific `is_payment` (402), rotates to the next candidate (free model on same provider or next provider) without prematurely killing the whole provider.
    - Preserves non-transient bad request (400) halting without fruitless rotation.
  - `src/harness_core/routing/router.py`:
    - Added `provider_priority: list[str]` to `RouterConfig` and `from_dict()`.
    - Added multi-provider failover readiness to `select_models()`: ensures distinct configured providers are represented in the fallback chain so that provider-level failure triggers automatic failover.
- **Token Attribution & Safety**:
  - `src/harness_core/agent/types.py`:
    - Added `providers_used`, `provider_token_usage`, `model_token_usage`, and `estimated_cost` to `Task`.
  - `src/harness_core/agent/loop.py`:
    - Updated `AgentLoop` to track provider attribution on model switch and record token consumption per provider and model.
- **CLI Commands & Experience**:
  - `src/harness_core/cli/main.py`:
    - Added `harness providers status`: comprehensive tabular display of provider name, state, masked credential, base URL, and model count.
    - Added `harness providers test`: non-destructive probe testing health, model discovery, and latency.
    - Added `harness config validate`: validates config files, priority, and credentials.
    - Updated `harness models list --provider <name>` with `auto_discover=True`.
  - `src/harness_core/cli/interactive.py`:
    - Added `/providers` and `/provider <name>` slash commands.
  - `.env.example`:
    - Fully updated with all 4 providers, priorities, fallback settings, timeouts, and limits.
- **Verification Evidence**:
  - `tests/unit/test_multi_provider_adapters.py`: 8/8 passed.
  - `tests/unit/test_intelligent_failover.py`: 9/9 passed.
  - `tests/unit/test_multi_provider_token_and_context.py`: 3/3 passed.
  - `tests/unit/test_multi_provider_cli.py`: 5/5 passed.

## TaskForge Recovery & Runtime Timeout Resilience (2026-10-30)
- **Harness Runtime Timeout Enum Crash**:
  - `src/harness_core/runtime/runtime.py`:
    - Fixed crash in `execute_request()` when handling `asyncio.TimeoutError`: previously attempted `t.status in (TaskStatus.PENDING, TaskStatus.EXECUTING, TaskStatus.IN_PROGRESS)` using `harness_core.agents.domain.TaskStatus`, which threw `AttributeError: type object 'TaskStatus' has no attribute 'PENDING'`.
    - Resolved by transitioning any task whose status is not completed/failed/cancelled cleanly: `if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED): t.status = TaskStatus.FAILED`.
- **TaskForge Application Recovery (`Desktop/New folder/app1`)**:
  - **FastAPI Backend (`backend/main.py`)**:
    - Added CORS middleware (`http://localhost:5173`, `http://127.0.0.1:5173`, `*`).
    - Added `tags` to SQLAlchemy model and Pydantic schemas.
    - Implemented type-safe `is_task_overdue(due_date, status, ref_dt)` handling pure `date`, naive `datetime`, UTC-aware `datetime`, and string inputs without `TypeError`.
    - Added `/api/v1/stats` endpoint returning computed KPI metrics (total, by status, by priority, overdue count, high/critical count, completion rate, recent activity) directly from SQLite persistence.
    - Added search, status, priority, tag, overdue-only filtering and sorting to `/api/v1/tasks`.
    - Created `backend/test_main.py` (6/6 pytest suites passing).
  - **React 18 + TypeScript + Vite Frontend (`frontend/`)**:
    - Built complete UI: `Dashboard.tsx` (live KPI stats), `KanbanBoard.tsx` (5 status columns with transitions), `TaskList.tsx` (table view with search/sort), `TaskModal.tsx` (create/edit modal), `App.tsx` (view tabs, filtering bar, toast notifications), and `api.ts`.
    - Executed non-interactive `npm install` (72 packages audited, 0 vulnerabilities).
    - Executed production build `npm run build` (tsc + vite build completed cleanly in 4.21s with 0 errors).

## Focused Terminal UX Refinement & Polish (2026-10-29)
- **Tool Activity Deduplication & Clean Status Lifecycle**:
  - `src/harness_core/cli/conversation.py`:
    - Attached `call_id` and `tool` to `ActivityItem`.
    - `ConversationState.add_running()` detects in-flight calls matching `call_id` or active tool name, updating display text without creating duplicate spinner lines or double-incrementing `tool_calls`.
    - `ConversationState.complete_last()` locates matching running item by `call_id`, then active tool, then last running item.
    - `ConversationRenderer.stop()` sweeps any remaining running items to clean success/cancelled states, guaranteeing zero frozen spinners in terminal scroll history.
    - Updated `_tool_name_from_display` as module-level and staticmethod helper.
  - `src/harness_core/cli/interactive.py`:
    - Updated `on_tool_call`, `on_tool_result`, `on_tool_started`, `on_tool_terminal` to extract and forward `call_id` to conversation renderer.
- **Responsive Markdown Table Compaction & Code Syntax Highlighting**:
  - `src/harness_core/cli/conversation.py`:
    - Added `compact_markdown_tables(text, max_width)`: detects tables exceeding terminal width or containing >3 columns and transforms them into clean bullet-card lists (`• **Name**: - Key: ...`) on narrow consoles, preserving compact 2-3 column tables as standard markdown tables.
    - Passed `code_theme="monokai"` to Rich `Markdown` for crisp, syntax-highlighted code blocks.
- **Vertical Whitespace Elimination & Tight Spacing**:
  - `src/harness_core/cli/conversation.py`:
    - Replaced raw consecutive empty lines in `_renderable()` with guarded `_add_sep()`.
    - Removed redundant trailing blank lines between streamed response and completion footer.
  - `src/harness_core/cli/interactive.py`:
    - Removed leading `\n` in `_get_prompt_html()`, ensuring the prompt sits exactly one line below the previous turn without 3-4 blank line gaps.
- **Completion Footer Declutter**:
  - Omitted token accounting line when `input_tokens == 0 and output_tokens == 0`.
  - Suppressed "Verification" line on read-only tasks without file modifications when `verification_status="not_started"`.
  - Only display phase duration breakdown when `>= 2` distinct phases took meaningful time (`>= 1.0s`), removing redundant single-phase timing lines.
- **Responsive Command Tables**:
  - Upgraded `_cmd_models`, `_cmd_help`, `_cmd_status`, `_cmd_session` to use `box=box.SIMPLE, pad_edge=False`.
  - In `_cmd_models`, dynamically hide non-essential columns (`Source`, `Tools`) on terminals `< 90` columns wide.
- **Verification**:
  - Added `TestRefinedTerminalUX` to `tests/unit/test_tui_ux.py` (48/48 passed).
  - `tests/unit/test_claude_ux_parity.py` (12/12 passed).
  - `tests/unit/test_harness_next.py` (11/11 passed).
  - Focused UX/session suite: 242 passed in 6.64s.
  - CLI smoke test: `uv run harness --help` and import check passed.

## HARNESS NEXT — Persistence, Context-Aware Resume, Project Memory & Terminal Excellence (2026-10-28)
- **Competitor Reverse-Engineering & Forensic Analysis**:
  - Investigated local installations and evidence of Claude Code (2.1.310), Qwen Code (0.25.0), OpenAI Codex CLI (0.159.2), Cursor CLI, and Freebuff CLI.
  - Produced `competitor_research_and_audit.md` comparing session persistence, memory architecture, context management, and terminal UX.
- **Session Intelligence & Workspace Reconciliation**:
  - `src/harness_core/session/domain.py`:
    - Added `ReconciliationReport` dataclass detailing workspace path, git HEAD/branch/dirty status, checkpoint divergence, externally modified files, missing files, stale test flags, and structured recap.
    - Added `context_file_hashes: dict[str, str]` to `Checkpoint.to_dict()` and `create_checkpoint`.
  - `src/harness_core/session/storage.py`:
    - Added `get_project_memories(workspace_path, task_keywords, limit)` querying cross-session project memories.
  - `src/harness_core/session/manager.py`:
    - Added `reconcile_workspace(session_id, workspace_path)`: compares live disk hashes and git status against checkpoint state, detects external file edits, warns of stale test verification, and generates a formatted resume recap.
    - Added `get_latest_session(workspace_path)` and `rename_session(session_id, new_title)`.
    - Added `sync_memory_file(workspace_path)` to generate/maintain `.harness/MEMORY.md`.
    - Allowed completed and active sessions in `get_resume_state` and allowed clean transition from `ACTIVE` to `COMPLETED` when archiving.
- **Interactive Shell Upgrades & Session Management Commands**:
  - `src/harness_core/cli/interactive.py`:
    - Added `expand_file_mentions(text, workspace)` supporting `@path/to/file` context expansion without breaking decorators or emails.
    - Added slash commands:
      - `/continue`: resumes the most recent session in the workspace, performs workspace reconciliation, and displays session recap.
      - `/resume [id]`: resumes a paused task or switches to a specific session with full disk reconciliation.
      - `/recap`: inspects workspace and git divergence against checkpoint.
      - `/new [title]`: initializes a clean session in the workspace.
      - `/rename <title>`: renames the active session.
      - `/archive`: safely archives the active session.
      - `/export [json|markdown]`: exports transcript and memories to `.harness/`.
      - `/compact`: manually triggers `ContextCompactor` and reports tokens saved.
      - `/ask <query>`: specialist read-only inquiry mode.
      - `/plan <task>`: specialist plan-only generation mode.
      - `/review`: specialist code review mode inspecting `git diff HEAD`.
      - `/sessions`: workspace-scoped session listing.
    - Connected `_execute_task` to checkpointing: records `user.message` and `assistant.message` session events, computes file SHA-256 hashes, records `Checkpoint`, and updates `.harness/MEMORY.md`.
- **CLI Flag Integration**:
  - `src/harness_core/cli/main.py`:
    - Added `--continue` (`-c`) and `--resume <id>` (`-r`) options to root `_main_callback` and `harness shell` command.
- **Deterministic Verification**:
  - Created `tests/unit/test_harness_next.py` (11 tests passed).
  - `tests/unit/test_session_m4.py` (51 tests passed).
  - `tests/unit/test_interactive_shell.py` (53 tests passed).
  - Target regression suite: 261 passed, 0 failures.
- **Findings & Root Cause Investigation**:
  - Investigated terminal rendering lifecycle across `conversation.py`, `interactive.py`, and `ui.py`.
  - Discovered and addressed 5 core defects documented in `findings_report.md`:
    1. Tool failure opacity: tool failures were recorded merely as `"failed"` without exit codes, stderr, or diagnostic reasons.
    2. Disconnected completion reporting: error details, verification notes, and unverified files were printed outside `ConversationRenderer.render_completion` via loose console prints.
    3. Windows cp1252 encoding exceptions: non-UTF8 console environments threw `UnicodeEncodeError` on characters like `\u276f` and `\u26a0`.
    4. Rich BBCode markup stripping: square brackets in diagnostic outputs `[exit 2: ...]` were swallowed by Rich unless `markup=False` was passed.
    5. Stale live spinner line: stopped `ConversationRenderer` froze the active spinner frame into terminal scroll history.
- **Implementation Highlights**:
  - `src/harness_core/cli/conversation.py`:
    - Introduced `ActivityItem(tuple)` maintaining 100% backward compatibility with 3-tuple unpacking while attaching `exit_code`, `error`, and `detail`.
    - Added structured error diagnostic indentation in `_renderable()` showing `Exit <code: int>: <first line of stderr>`.
    - Unified `render_completion()` to natively display duration, tool calls, model routing, tokens, files (with unverified warnings), test results, verification status, and actionable error messages.
    - Updated `_render_plain()` with `markup=False` to prevent markup swallowing.
    - Added encoding checks in `_render_prompt()` using `Sym.PROMPT_ASCII` on plain/charmap consoles.
  - `src/harness_core/cli/interactive.py`:
    - `on_tool_result()` extracts `exit_code`, `stderr`, and `diagnosis_reason` to feed `tool_completed()`.
    - `_execute_task()` routes failure diagnostics and verification status directly through `conv.render_completion()`.
  - `src/harness_core/cli/ui.py`:
    - Strengthened `is_unicode_supported()` to test `"❯✓✗⚠⚡◆↳"` and populated safe ASCII fallbacks for all symbols.
- **Verification & Parity Testing**:
  - `tests/unit/test_tui_ux.py`: 42/42 tests passed, including all 17 Claude Code parity scenarios.
  - `tests/unit/test_claude_ux_parity.py`: 12/12 tests passed.
  - Broader test suite: 1,797 passed, 1 skipped in 84.48s.
  - CLI smoke test: `uv run harness --help` and import check passed cleanly.
- **REA Reconnaissance & Specification**:
  - Reverse-engineered local Claude Code binary (`claude.exe` 2.1.302 at `C:\Users\kunal sanga\.local\bin\claude.exe`, Bun standalone bundle) and user session transcripts (`C:\Users\kunal sanga\.claude\projects\`).
  - Synthesized evidence-backed UX specification artifact (`claude_code_ux_spec.md`) documenting startup, clean prompt redraw, model waiting vs tool activity states, streaming Markdown, and cohesive summary footers.
- **Terminal Reliability & Defect Resolutions**:
  - `src/harness_core/cli/interactive.py`:
    - Eliminated `self.console.clear()` on first task submission in `_execute_task`, preserving welcome banner, project context, and natural terminal scroll history.
    - Replaced terminal-width horizontal divider rule (`─ * width`) with clean, minimal prompt: `\n<style fg="#06b6d4">❯</style> ` (with ASCII `>` fallback). Fixed Windows PowerShell/cmd auto-wrap line break dislocation.
    - Propagated 9router indicator to `ConversationRenderer` to prevent upstream model leaks.
    - Added `model.cooldown` event handler to set stage to `Retrying model…` and distinguish waiting, retrying, and failure states.
  - `src/harness_core/cli/conversation.py`:
    - Added `_clean_intro_narration()` to strip redundant conversational filler/throat-clearing (e.g. "Based on my analysis of the codebase... Let me provide a comprehensive explanation...") so answers begin directly with useful headings and content.
    - Streamlined `stream_response()`: for complete / zero-delay output, renders Markdown directly with clean padding without spinning a `Live` container or re-parsing paragraphs on every tick, eliminating terminal flicker.
    - Polished tool activity timeline: when no plan is present, items align cleanly without orphaned tree branches (`├─ `).
    - Guarded `render_completion()`: renders `Model  ⚡ Routed through 9router` when routed via 9router, preventing internal model ID leakage in normal mode.
  - `src/harness_core/agent/loop.py`:
    - Added direct communication rule to workspace system prompt: jump straight to the answer, avoiding filler introductions.
  - `src/harness_core/cli/ui.py`:
    - Added `is_unicode_supported()` terminal encoding capability check.
    - Added `Sym.PROMPT = "❯"` and `Sym.PROMPT_ASCII = ">"`.
- **Deterministic Verification**:
  - `tests/unit/test_tui_ux.py`: expanded `TestTerminalReliabilityAndClaudeParity` suite (27 passed) covering prompt format, narration stripping, plan-less tool timelines, consecutive task state isolation, diff statistics, and cooldown transitions.
  - Full test suite: **1,890 passed, 5 skipped, 0 failures** in 117.07s.
  - CLI smoke test: `uv run harness --help` passed cleanly.
- **Branded Wordmark & Clean Visual Identity**:
  - `src/harness_core/cli/ui.py`: `render_welcome()` renders compact branded wordmark (`◆ Harness`, `Autonomous AI Engineering`, workspace folder name) replacing bulky ASCII banner panels. Plain mode falls back cleanly to ASCII mark (`*`).
  - Added `shorten_path()` to shorten absolute paths to workspace-relative paths across all tool display lines.
  - Added `PHASE_DISPLAY` mapping directly to the backend's 11 `_emit_phase()` states.
- **Interactive Shell Upgrades**:
  - `src/harness_core/cli/interactive.py`:
    - `_print_welcome()` calls `render_welcome()`.
    - `9router` concise normal-mode indicator: `⚡ Routed through 9router` without leaking internal model IDs.
    - Added `steering.received` event handler displaying `↳ User update received` inline.
    - Updated `/free` command to show product-level message: `✓ Free model routing enabled`.
    - Enhanced cancellation UX (`_render_cancellation()`): displays `⏹ Stopping...`, `✓ Task stopped safely`, `✓ State preserved`, and clean listing of changes made so far.
- **Conversation Renderer & Activity Aggregation**:
  - `src/harness_core/cli/conversation.py`:
    - Added `update_agent()` for multi-agent tracking and progressive disclosure (named workers for <=3, count summary for >3).
    - Consecutive file inspection aggregation (collapses >=3 sequential read/list/glob/grep calls into `✓ Inspected N files`).
- **Four-Question Completion Summary**:
  - `src/harness_core/cli/completion.py`: structured summary answering What did Harness do, What changed, Did verification pass, and What remains. Replaced "No false success" slogan with actionable next steps.
- **Mandatory Restaurant Website Benchmark Acceptance Test**:
  - `tests/e2e/test_restaurant_benchmark.py`: end-to-end multi-agent acceptance test simulating full autonomous restaurant ordering website creation across requirements analysis, database schema, FastAPI backend, web frontend, test failure detection, recovery/debugger repair, and successful verification.
- **Deterministic Testing**:
  - `tests/unit/test_tui_ux.py`: 17 behavioral tests for branding, path shortening, phase mapping, aggregation, summaries, cancellation, steering, and 9router.
  - UI regression suite: 184 passed in 5.03s.
  - Full suite: 1783 passed, 1 skipped in 82.74s.

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
