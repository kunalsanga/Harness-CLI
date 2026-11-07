# Agent Context Changelog

## 2026-11-07 — Harness V2 Interactive Session History & Transcript Navigation

- **Interactive `/sessions` Browser & Historical Transcript Viewer (`src/harness_core/cli/session_browser.py`)**:
  - Implemented `InteractiveSessionBrowser` with keyboard navigation, date grouping (`Today`, `Yesterday`, `This Week`, `Month/Year`), full chronological transcript inspection, and explicit resume confirmation.
  - Keyboard navigation: Up/Down arrow (`k`/`j`), PageUp/PageDown, Home/End.
  - Transcript inspection: Enter opens full conversation transcript with user prompts, tool activity, Markdown assistant answers, and verification status. Up/Down/PageUp/PageDown scroll transcript.
  - Search filtering: `/` activates live search in both session list and transcript views; typing filters items; Enter jumps to match; `n`/`N` navigates matches; Esc clears search.
  - Strict read-only viewing guarantee: 0 LLM calls, 0 tool executions, 0 state mutations.
  - Explicit resume dialog: pressing `R` opens `[Y/n]` resume confirmation; pressing `N`/`Esc` cancels back to browser; `Y`/`Enter` confirms resumption and context restore.
  - Fallback support: non-TTY and `--plain` flag fall back to clean date-grouped static tables (`render_session_browser`).
- **Accurate Derived Session Status (`derive_session_status`)**:
  - Resolved false `Active` mislabeling on historical sessions: dynamically inspects latest run outcomes to derive `Completed`, `Failed`, `Paused`, `Interrupted`, `Active`, and `Archived`.
  - Updated SQLite session lifecycle in `InteractiveShell` to mark sessions `completed` or `failed` on run completion.
- **Verification Evidence**:
  - `tests/unit/test_session_browser.py`: 36/36 passed in 2.77s.
  - `tests/unit/test_readonly_explain_regression.py`: 16/16 passed in 0.27s.
  - `tests/e2e/test_interactive_acceptance_hardening.py`: 16/16 passed in 2.83s.
  - Real database audit (`~/.harness/sessions.db`): confirmed session `17006d8db86d` is accurately marked `Completed`, `f44265f95ddb` marked `Failed`, and abandoned sessions marked `Interrupted`.

## 2026-11-07 — Read-Only Explanation Task Regression Resolution & Windows Shell Reliability

- **Failure Analysis & Resolution on Session `f44265f95ddb`**:
  - Resolved 5 distinct root causes from read-only task failure on `app1` (`explain this project`):
    1. **Tool Aggregation Overwrite (`conversation.py`)**: Fixed `complete_last()` so a single failed read inside an aggregated block preserves successful reads (`✓ Inspected 20 files`) and isolates the failure (`✗ Read ...`) rather than masking the whole batch as failed.
    2. **Windows PowerShell Statement Chaining & Quoting (`shell.py`)**: Added `translate_powershell_chain()` to convert POSIX `&&` and `||` to PowerShell 5.1 syntax (`cmd1; if ($?) { cmd2 }`), and quoted space-containing paths safely with `Set-Location -LiteralPath '...'`.
    3. **Contradictory Plan Completion on Failed Tasks (`todos.py`)**: Added `_SUMMARIZE` category so "Summarize findings" is not completed by read tools. On task failure, `reconcile_on_evidence()` marks incomplete steps failed and reverts prematurely completed summary steps.
    4. **Inappropriate Verification Failure on Read-Only Tasks (`state.py`, `runtime.py`, `conversation.py`)**: Added `VerificationStatus.SKIPPED`. Failed read-only tasks now report `Verification: Not applicable (read-only task)` with dim styling instead of `Verification: Failed`.
    5. **Read-Only Inspection Bounding & Directory Diagnostics (`loop.py`, `filesystem.py`)**: Nudges read-only tasks to synthesize after 12 read operations, hard caps at 18 operations, and provides clear directory diagnostics when `read_file` is called on a folder.
- **Verification Evidence**:
  - `tests/unit/test_readonly_explain_regression.py`: 16/16 passed in 0.21s.
  - Full unit test suite: **1,917 passed, 1 skipped, 0 failures** in 134.60s.
  - Smoke tests and import checks passed.
  - **Live Post-Fix Reproduction Run (Session `17006d8db86d`)**:
    - Task `explain this project` in workspace `app1` completed cleanly in 19.6s (vs 35s failure).
    - Tool calls reduced from 31 to 1 (`List .`).
    - Input tokens reduced from 32.3k to 9.5k (-71%), output tokens increased from 97 to 686 (+607%).
    - Produced complete, accurate architectural overview; 0 file modifications; previous failed session `f44265f95ddb` preserved intact.

- **TaskForge Real-World Dogfooding & Implementation**:
  - Validated autonomous cross-stack engineering in disposable workspace `app1` (FastAPI + React/TypeScript).
  - Executed feature development: task priority filtering on the Dashboard.
  - Performed independent self-diagnosis and resolved 4 engineering defects:
    1. Input validation on `/api/v1/stats` and `/api/v1/tasks` (strict 400 rejection for unknown priorities).
    2. Partitioned global distribution metrics from filtered population statistics on the Dashboard.
    3. Added `AbortController` cancellation to eliminate asynchronous race conditions on rapid filter switching.
    4. Resolved variable hoisting order in `Dashboard.tsx`.
- **Automated & Live Verification**:
  - Backend pytest: 11/11 passed in 1.16s (with 3 upstream deprecation warnings noted).
  - Frontend typecheck: `tsc --noEmit` passed with 0 errors.
  - Frontend automated tests: 5/5 passed in 31.6ms via Node 24 built-in `node:test`.
  - Production build: `vite build` completed in 1.53s.
  - Live daemons: FastAPI (`http://127.0.0.1:8000`) and Vite (`http://127.0.0.1:5173`) running with 8 seeded development tasks.
- **Strategic Direction**:
  - UI feature development frozen. Focus shifted to Session Recovery testing, 3-Project Generalization Benchmark, and Task Success Rate tracking.

## 2026-11-06 — Harness V2: Premium Autonomous Coding Agent Experience

- **Clean Fresh Launch & Branded Welcome**:
  - Removed auto-hijack of past active/paused sessions on fresh launch in `interactive.py:_setup_session()`. `harness shell` now initializes clean without dumping old task recaps.
  - Added session tag (`Session: <id[:8]>`) and `/sessions, /models, /diff, /help` command discovery in `ui.py:render_welcome()`.
  - Fixed overwritten `session_id` in `InteractiveShell.__init__`.
- **Date-Grouped Session Browser & Historical Viewer**:
  - Built `src/harness_core/cli/session_browser.py`:
    - `group_sessions_by_date()` organizing sessions into `Today`, `Yesterday`, `This Week`, `Month Year`.
    - `render_session_browser()` displaying rich tabular session history with run counts, status colors, and keyword/workspace/status search.
    - `render_historical_transcript()` providing read-only playback of past prompts, tool activities, Markdown outputs, and verification results without calling models or running tools.
  - Rewired `/sessions` slash command to support browsing, searching, transcript viewing (`/sessions view <id>`), and resuming (`/sessions resume <id>`).
- **Live File Activity Tracking & `/diff` Command**:
  - Added `files_activity` tracking (`inspected`, `created`, `modified`, `deleted`, `failed`) and in-flight `Files` rendering in `ConversationRenderer`.
  - Removed `workflow=True` gate in `on_tool_call` so all live tool executions start rendering immediately.
  - Enhanced `/diff` command: renders syntax-highlighted git diff (`-U3`) with colored insertions/deletions, sensitive token/key redaction, and clean line capping.
- **Dynamic Ladder & Failover Transparency**:
  - Updated `on_model_switched` and `on_model_cooldown` to set explicit stage messages (`Failing over to Groq...`, `Retrying model...`), preventing silent hangs or frozen spinners during model switches.
- **Verification Evidence**:
  - Created `tests/unit/test_session_browser.py` (18/18 passed in 1.06s).
  - Targeted suites: 203 passed in 4.00s.
  - Full unit test suite: **1,901 passed, 1 skipped, 0 failures** in 104.96s.
  - CLI smoke test and import check passed.

## 2026-11-01 — Universal Multi-Provider Resilience & Intelligent API Key Failover

- **Root Cause Fixes**:
  - Unlocked NVIDIA NIM and Groq developer endpoints in all routing modes by marking them `is_free=True` with 0.0 cost.
  - Merged project config with auto-discovered environment credentials in `factory._provider_configs()` so providers are never omitted when project configs override one provider.
  - Implemented cross-provider candidate interleaving in `ModelRouter.select_models()` and `_build_free_chain()`, preventing single-provider monopolization of the fallback chain.
  - Mapped token context overflows to `CONTEXT_OVERFLOW` in `fallback.classify_error()`.
- **Azure OpenAI Integration**:
  - Implemented `AzureOpenAIProvider` in `src/harness_core/providers/azure.py` supporting `/openai/deployments/...` native endpoints and OpenAI-compatible paths.
  - Registered Azure in `CredentialResolver` with `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_DEPLOYMENT`.
- **CLI Auto-Discovery**:
  - Updated `harness doctor`, `harness models list`, `harness models benchmark`, and `harness eval` to pass `auto_discover=True` to `create_providers()`.
- **Regression Suite Verification**:
  - Verified 1,883 unit tests passing cleanly with 0 regressions.
  - Live failover verified: OpenRouter 429 quota exhaustion immediately routes to FreeLLMAPI/Groq with 0 interruption.

## 2026-11-01 — Multi-Provider AI Routing, Fault Tolerance & Pre-Deployment Hardening

- **Multi-Provider Adapters & Auto-Discovery**:
  - Implemented `FreeLLMAPIProvider` with health check, `/models` discovery, and automatic fallback.
  - Enhanced `OpenAIProvider` with custom `OPENAI_BASE_URL`, `OPENAI_MODEL`, `/models` health checking, and capability inference for GPT-4o/o1/o3-mini.
  - Enhanced `NvidiaProvider` supporting `NVIDIA_NIM_API_KEY`/`NVIDIA_API_KEY`, custom base URL, and tool-calling capability detection.
  - Enhanced `ProviderFactory` with auto-discovery and priority ordering (`HARNESS_PROVIDER_PRIORITY`).
  - Added custom base URL resolution across all providers in `CredentialResolver`.
- **Intelligent Automatic Failover**:
  - `FallbackEngine`: Fixed failover on 401/403 and account quota exhaustion so that when provider A fails, the chain switches seamlessly to the next provider instead of aborting the task.
  - Handled 402 payment errors with model/provider rotation.
  - Maintained request state, conversation history, and tool outputs intact across failovers.
  - Multi-provider representation in `ModelRouter.select_models()`.
- **Token Attribution & Safety**:
  - Added provider and model token usage tracking and estimated cost accounting to `Task` and `AgentLoop`.
- **Pre-Deployment CLI Diagnostics**:
  - Implemented `harness providers status`, `harness providers test`, `harness config validate`, `harness models list --provider <name>`, and interactive `/providers` slash command.
  - Completely updated `.env.example`.
- **Comprehensive Automated Verification**:
  - Verified across 25 unit tests covering multi-provider adapters, intelligent failover scenarios, token attribution, and CLI diagnostics.

## 2026-10-30 — TaskForge Recovery & Runtime Timeout Resilience

- **Harness Runtime Timeout Resilience**:
  - `src/harness_core/runtime/runtime.py`: Fixed `AttributeError: type object 'TaskStatus' has no attribute 'PENDING'` during task timeout handling by using safe status transition `if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED): t.status = TaskStatus.FAILED`.
- **TaskForge Recovery (`Desktop/New folder/app1`)**:
  - **FastAPI Backend**:
    - Added CORS middleware (`http://localhost:5173`, `*`).
    - Added tag support, advanced filtering, and overdue logic (`is_task_overdue`) preventing `TypeError` between `date` and `datetime`.
    - Added `/api/v1/stats` computing live metrics directly from SQLite persistence.
    - Verified with `test_main.py` (6/6 pytest suites passing in 1.50s).
  - **React 18 + Vite Frontend**:
    - Implemented `Dashboard.tsx`, `KanbanBoard.tsx`, `TaskList.tsx`, `TaskModal.tsx`, `App.tsx`, and `api.ts`.
    - Non-interactive `npm install` completed with 0 vulnerabilities.
    - Production build (`npm run build`) succeeded with 0 errors in 4.21s.

## 2026-10-29 — Harness CLI: Focused Terminal UX Refinement

- **Tool Event Deduplication**:
  - Attached `call_id` and `tool` to `ActivityItem`.
  - Deduped concurrent `tool.call` (workflow) and `tool.started` events; updated item display without duplicating rows or double-incrementing tool calls.
  - Guaranteed zero frozen spinners on `renderer.stop()` by sweeping active items to clean success/cancelled states.
- **Responsive Markdown Table Compaction & Monokai Syntax**:
  - Implemented `compact_markdown_tables()` in `conversation.py` to transform wide 4+ column tables or tables exceeding terminal width into clean bullet card structures on narrow/80-column consoles.
  - Enabled Monokai syntax highlighting (`code_theme="monokai"`) for Rich Markdown code blocks.
- **Vertical Whitespace Elimination**:
  - Guarded consecutive empty lines in `_renderable()` via `_add_sep()`.
  - Removed leading `\n` in `InteractiveShell._get_prompt_html()`, eliminating multi-line vertical gaps between turns.
- **Completion Footer Declutter**:
  - Omitted zero-token lines (`input_tokens == 0 and output_tokens == 0`).
  - Suppressed "Verification" status line on read-only tasks without file modifications when not started.
  - Omitted redundant phase duration breakdowns when fewer than 2 distinct phases took meaningful time (`>= 1.0s`).
- **Responsive Command Tables**:
  - Applied `box=box.SIMPLE, pad_edge=False` to `/help`, `/status`, `/model`, `/models`, and `/session` commands.
  - In `/models`, conditionally hid `Source` and `Tools` columns on narrow screens (<90 cols) to prevent horizontal wrapping.
- **Verification**:
  - Added `TestRefinedTerminalUX` to `tests/unit/test_tui_ux.py` (48/48 passed).
  - 242 tests passing across focused UX and session test suites.

## 2026-10-28 — HARNESS NEXT: Intelligence, Memory, Reliability & Terminal UX Transformation

- **Competitor Reverse-Engineering & Forensic Benchmark**:
  - Analyzed locally installed CLI tools via REA: Claude Code (2.1.310), Qwen Code (0.25.0), OpenAI Codex CLI (0.159.2), Cursor CLI, and Freebuff CLI.
  - Formulated `competitor_research_and_audit.md` mapping architectural patterns to Harness engineering.
- **Session Intelligence, Checkpointing & Disk Reconciliation**:
  - `src/harness_core/session/domain.py`: Added `ReconciliationReport` with full git, workspace, and checkpoint drift analysis. Added `context_file_hashes` tracking file SHA-256 digests.
  - `src/harness_core/session/storage.py`: Added cross-session workspace memory retrieval (`get_project_memories`).
  - `src/harness_core/session/manager.py`:
    - Added `reconcile_workspace(session_id, workspace_path)` to detect modified/missing files, check git HEAD drift, flag stale verification, and produce a structured resume recap.
    - Added `sync_memory_file(workspace_path)` maintaining `.harness/MEMORY.md`.
    - Added `rename_session` and `get_latest_session`.
    - Enabled seamless transitions when archiving active sessions.
- **Interactive Shell Upgrades & Session Commands**:
  - `src/harness_core/cli/interactive.py`:
    - Implemented `expand_file_mentions(text, workspace)` supporting `@path/to/file` context attachments.
    - Added slash commands: `/continue`, `/resume [id]`, `/recap`, `/new [title]`, `/rename <title>`, `/archive`, `/export [json|markdown]`, `/compact`, `/ask <query>`, `/plan <task>`, `/review`, and `/sessions`.
    - Added checkpoint creation and `user.message` / `assistant.message` event logging during task execution.
- **CLI Flag Integration**:
  - `src/harness_core/cli/main.py`: Added `--continue` (`-c`) and `--resume <id>` (`-r`) options to root CLI callback and `harness shell`.
- **Deterministic Verification**:
  - Added `tests/unit/test_harness_next.py` (11/11 passed).
  - Verified `tests/unit/test_session_m4.py` (51/51 passed).
  - Verified `tests/unit/test_interactive_shell.py` (53/53 passed).
  - All 261 targeted regression tests pass cleanly.

- **Findings & Root Causes**:
  - Investigated terminal rendering lifecycle across `conversation.py`, `interactive.py`, and `ui.py`.
  - Authored `findings_report.md` artifact detailing root causes: tool failure opacity, disconnected console prints, Windows cp1252 charmap encoding errors, Rich markup tag stripping, and stale live spinner lines.
- **Architectural Fixes**:
  - `src/harness_core/cli/conversation.py`:
    - Introduced `ActivityItem(tuple)` subclass to store `(icon, display, status)` while adding `exit_code`, `error`, and `detail` with 100% backward compatibility for all existing callers unpacking 3-tuples.
    - Updated `complete_last()` and `tool_completed()` to accept `exit_code`, `error`, and `detail`.
    - Enhanced `_renderable()`: when a tool fails, renders an indented diagnostic line with `Exit <code: int>` and the first line of stderr/error.
    - Enhanced `stop()`: sets `self._active = False` before `self._live.update(self._renderable())` so the live spinner line disappears cleanly and does not freeze into the terminal scroll history.
    - Enhanced `render_completion()`: accepts `verification_status`, `verification_summary`, and `unverified_files`; unified failure and verification reporting.
    - Added `markup=False` to `_render_plain()` to prevent Rich from silently stripping square bracket markup like `[exit 2: ...]`.
    - Made `_render_prompt()` check encoding and use ASCII fallback `>` in plain mode / charmap consoles to prevent `UnicodeEncodeError`.
  - `src/harness_core/cli/interactive.py`:
    - Updated `on_tool_result()`: extracts `exit_code`, `error`, `stderr`, and `metadata.diagnosis_reason`, passing them directly to `_conv().tool_completed()`.
    - Updated `on_tool_terminal()`: forwards `error_class` to `_conv().tool_completed()`.
    - Refactored `_execute_task()`: passes `error_detail`, `verification_status`, `verification_summary`, and `unverified_files` into `conv.render_completion()`, eliminating duplicate/disconnected console prints.
  - `src/harness_core/cli/ui.py`:
    - Tested comprehensive Unicode character set (`❯✓✗⚠⚡◆↳`) in `is_unicode_supported()` and added safe fallback constants across `Sym`.
- **Deterministic Testing**:
  - `tests/unit/test_tui_ux.py`: 42/42 tests passed (including all 17 Claude Code parity scenarios).
  - `tests/unit/test_claude_ux_parity.py`: 12/12 tests passed.
  - Full regression suite: **1,797 passed, 1 skipped in 84.48s**.
  - CLI smoke test: `uv run harness --help` and import check passed cleanly.

- **REA Reconnaissance & Specification**:
  - Reverse-engineered local Claude Code binary (`claude.exe` 2.1.302, Bun standalone bundle) and user session transcripts (`.claude/projects/`).
  - Created UX specification artifact (`claude_code_ux_spec.md`) documenting startup flow, minimal prompt, model wait states, tool execution, and compact summary footers.
- **Terminal Architecture & Reliability Fixes**:
  - `src/harness_core/cli/interactive.py`:
    - Removed `self.console.clear()` on first task submission in `_execute_task` to prevent screen wipe and preserve scrollback history.
    - Eliminated terminal auto-wrapping bug by replacing full-width divider `─ * width` with a minimal prompt: `\n<style fg="#06b6d4">❯</style> `.
    - Propagated 9router indication to conversation renderer to suppress upstream model leakage in completion.
    - Added `model.cooldown` event handler to transition stage to `Retrying model…` during rate limits.
  - `src/harness_core/cli/conversation.py`:
    - Added `_clean_intro_narration()` to strip redundant throat-clearing/filler prefixes so answers begin immediately with helpful content.
    - Eliminated nested `Live` re-rendering flicker in `stream_response()` on zero-delay / completed responses by rendering directly via Markdown.
    - Polished tool activity timeline: when no plan is present, tool activity aligns cleanly without orphaned tree branches (`├─ `).
    - Added `set_9router()` and updated `render_completion()` to display `Model  ⚡ Routed through 9router` without leaking internal model IDs.
  - `src/harness_core/agent/loop.py`:
    - Added direct communication rule to system prompt to prevent conversational filler.
  - `src/harness_core/cli/ui.py`:
    - Added `is_unicode_supported()` terminal encoding check and `Sym.PROMPT = "❯"`.
- **Deterministic Testing & Full Verification**:
  - Added and expanded `TestTerminalReliabilityAndClaudeParity` in `tests/unit/test_tui_ux.py` (27 passed).
  - Full suite: **1,890 passed, 5 skipped, 0 failures** in 117.07s.
  - Smoke test: `uv run harness --help` passed cleanly.

## 2026-10-25 — Terminal UI Overhaul & Restaurant Website Coding Benchmark

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
