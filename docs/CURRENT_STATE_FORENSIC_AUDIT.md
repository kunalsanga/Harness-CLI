# 1. Executive Summary

The current behavior of the `harness` CLI for informational tasks (like "explain this project") is severely degraded by competing architectural boundaries and unhandled fallback cascades. When the user executes `uv run harness "explain this project"`, the CLI incorrectly renders duplicate prompts and thinking states, executes an isolated intent workflow that aborts unexpectedly on a 402 provider failure, and emits a contradictory status matrix to the user interface combining "Failed" with "Task ended with status: paused".

This forensic audit has identified the exact causes of these bugs without modifying code. The primary problem lies in the integration layer between `InteractiveShell` (which handles UI orchestration), the `ConversationRenderer` (which draws the UI), and the `AgentLoop` (which executes logic). Events are double-bound, models are failing hard, and task status enums are mapped inconsistently to the final user completion formatter.

# 2. Repository Map

The repository is structured as a modern Python package built with Typer and Rich.

- **Source Code (`src/harness_core`)**:
  - `cli/`: `main.py`, `interactive.py`, `conversation.py`, `completion.py`, `runtime_dashboard.py`, `ui.py`
  - `agent/`: `loop.py`, `workflows.py`, `types.py`, `todos.py`, `intent.py`
  - `routing/`: `router.py`, `fallback.py`, `task_aware.py`
  - `providers/`: `openrouter.py`, `ollama.py`, `nvidia.py`
  - `models/`, `observability/`, `tools/`, `memory/`, `session/`, `classifier/`
- **Tests (`tests/`)**: Contains unit and integration tests.
- **Docs/Configs**: `docs/`, `examples/`, `pyproject.toml`, `.env`, `README.md`

# 3. Actual Default CLI Execution Path

1. **CLI Entry Point**: `harness` invokes `harness_core.cli.main:app`.
2. **Command Parsing**: Invoked with no subcommands, `main.py` triggers `_main_callback()`.
3. **Interactive Mode**: `run_interactive(mode="auto")` initializes `InteractiveShell`.
4. **Setup**: `InteractiveShell._setup_provider()` configures OpenRouter and ModelRouter, and instantiates the `AgentLoop`. `InteractiveShell._setup_session()` loads or creates a session.
5. **Input Handling**: `_read_input()` gets the user's prompt via `prompt_toolkit`.
6. **Task Execution**: `_execute_task(goal)` manually calls `ConversationRenderer.start(goal)` (rendering the prompt and "Thinking").
7. **Agent Loop**: `AgentLoop.run(goal)` starts, emitting `task.started` to the EventBus.
8. **Event Loop Duplication**: `on_task_started` receives `task.started` and calls `ConversationRenderer.start(goal)` AGAIN.
9. **UI Renderer**: The `ConversationRenderer` streams events. Legacy `LiveTerminalUI` is dormant because `mode="auto"`.
10. **Completion**: `AgentLoop` returns the `Task`. `InteractiveShell` evaluates `task.status`, streams final response, and prints the result via `CompletionFormatter`.

# 4. Actual "explain this project" Execution Path

1. `AgentLoop.run("explain this project")` begins.
2. **Intent Classification**: `workflow_name = classify_workflow(goal)` returns `"explain"`.
3. **Workflow execution**: `run_explain_workflow` executes, leveraging `read_file` tools to read `README.md` etc.
4. **Placeholder Injection**: `_apply_workflow_result()` intercepts the snippets and sets `task.result = "Project analyzed."`.
5. **Fall-through**: The code explicitly falls through to the LLM loop to summarize the context gathered by the workflow.
6. **Provider Request**: The LLM loop attempts a completion request via `router.execute`.
7. **402 Failure**: The `FallbackEngine` hits a 402 Payment Required for the model. It skips to fallback models, which also fail/skip.
8. **Hard Crash**: `FallbackEngine` returns a failure, causing the LLM loop to raise `RuntimeError("All models failed")`.
9. **Pause Injection**: The exception handler in `AgentLoop` sees `did_work = True` (because files were read) and forces `task.status = TaskStatus.PAUSED` to preserve state, skipping proper failure handling.
10. **Render Output**: `InteractiveShell` renders "Project analyzed.", sees `task.status != "completed"`, and outputs a Failure report saying "Task ended with status: paused".

# 5. UI/Rendering Architecture

Multiple competing renderers overlap:
- **`InteractiveShell`**: Owns `prompt_toolkit` and falls back to `console.print` in plain mode.
- **`ConversationRenderer`**: The active Rich Live wrapper rendering live activity in `"auto"` mode.
- **`LiveTerminalUI` / `RuntimeViewModel`**: Only fully activated when `mode="unified"`.
- **`CompletionFormatter`**: Renders the final success/failure summary independent of the live renderer.

The dependency graph:
```
InteractiveShell -> ConversationRenderer (Live UI)
                 -> CompletionFormatter (Final Output)
                 -> EventBus -> on_task_started -> ConversationRenderer (Duplicate Call)
```

# 6. Duplicate Prompt Root Cause

The duplicate prompt is caused by `ConversationRenderer.start(goal)` being called twice for every task:
1. Explicitly inside `InteractiveShell._execute_task(goal)` before `AgentLoop.run()` starts.
2. Inside `InteractiveShell._setup_event_handlers()`, the `on_task_started` callback listens for the `task.started` event from `AgentLoop` and explicitly calls `ConversationRenderer.start(goal)` again.

Each `start()` call executes `_render_prompt(goal)`, causing the prompt to be printed multiple times.

# 7. Duplicate Thinking Root Cause

This shares the exact root cause as the Duplicate Prompt. When `ConversationRenderer.start(goal)` is invoked, it sets `self._thinking_shown = False` and immediately calls `_show_thinking()`. Because `start()` is invoked twice (once manually, once via the EventBus), the "Thinking" state is reset and added to the render queue twice.

# 8. Vague Response Root Cause

The vague response "Project analyzed." is hardcoded in `loop.py` inside `_apply_workflow_result`:
```python
        if "snippets" in data and not task.result:
            task.result = "Project analyzed."
```
The intention is that the LLM loop will run immediately afterward, summarize the files, and overwrite this placeholder with an actual explanation. However, because the LLM hits a 402 failure and crashes the LLM loop, the loop never replaces the placeholder. The task pauses, and the system returns the placeholder as the final answer.

# 9. Provider/402 Failure Root Cause

In `fallback.py`, a `402 Payment Required` is correctly classified as `PERMANENT`. The engine records a `PAYMENT_REQUIRED` failure and immediately breaks the retry loop to try the next model. However, when all fallback models fail (e.g. no free models remain), it raises `RuntimeError`. `AgentLoop` catches this exception, detects that `did_work = True` (because tools were run during the `explain` workflow), and incorrectly assumes it must save progress by setting `task.status = TaskStatus.PAUSED`.

# 10. Session/History Root Cause

While `prompt_toolkit` correctly handles input history, the visual duplication of history on session restoration is minimal in `mode="auto"`. `InteractiveShell._setup_session()` restores the session ID and prints `Resumed session: ...`, but does not blindly replay all past EventBus events. The replayed visual artifacts are strictly a symptom of the EventBus duplicate wiring for the active run, not the historical session.

# 11. Completion/Status Contradictions

`AgentLoop` sets `task.status = TaskStatus.PAUSED` due to the 402 failure logic. `InteractiveShell._execute_task` has a condition:
`success = status_val == "completed"`
Because "paused" != "completed", `success` is `False`. The UI prints `✗ Failed`, then triggers `CompletionFormatter.failure()`. The formatter embeds the actual status enum into the headline: `Task ended with status: paused`. This yields the contradictory "Failed but Paused" output.

# 12. Context Management
Wired correctly but bypassed by early workflow termination on "explain". Context truncation is not the issue here; the LLM was never reached.

# 13. Agent Runtime
The default runtime is `AgentLoop`, operating in `mode="auto"`. The unified `EngineeringRuntime` exists but requires `mode="unified"`.

# 14. Multi-Agent Runtime
Partially wired. Exists in `Orchestrator` but only invoked if `mode="multi-agent"`. Not the default.

# 15. TaskGraph/Scheduler
Isolated. The default `AgentLoop` uses a linear `TaskPlan` (list of `TodoItem`s), not the complex `TaskGraph`. `TaskGraph` is only used by the unified runtime.

# 16. Memory
Partially wired. `SessionManager` tracks run outcomes, and `MemoryManager` handles persistent artifacts, but `MemoryManager` is only actively hydrated in `mode="unified"` or `mode="multi-agent"`.

# 17. Verification
Wired, but masked by failure conditions.

# 18. Recovery
Partially wired. Tools have intrinsic retry, and `AgentLoop` attempts self-correction, but catastrophic model failure (402) bypasses recovery.

# 19. Security/Permissions
Wired. `AgentLoop` properly enforces `permission_denied` status.

# 20. Tests
Extensive tests exist but fail to cover the `AgentLoop` <-> `ConversationRenderer` EventBus duplication, indicating isolation issues in E2E coverage.

# 21. Dead/Legacy/Partially Wired Systems
- `RuntimeViewModel` and `LiveTerminalUI` are completely dormant in the default CLI path.
- `TaskGraph` is legacy/dormant in the default CLI path.
- `LiveStatus` is a mocked dummy class in `interactive.py` that was never removed.

# 22. Production Readiness Matrix
- **Architecture**: YELLOW
- **Runtime**: YELLOW
- **CLI**: YELLOW
- **UX**: RED (Duplicate rendering, vague errors)
- **Provider system**: YELLOW (Fallback raises exception rather than graceful degradation)
- **Model routing**: GREEN
- **Context management**: GREEN
- **Memory**: YELLOW
- **Task graph**: YELLOW
- **Multi-agent system**: YELLOW
- **Permissions**: GREEN
- **Error handling**: RED (Status contradiction mapping)
- **Tests**: YELLOW

# 23. Top 20 Problems Ranked by Severity
1. Duplicate Prompt Rendering (`conv.start` called twice)
2. Duplicate Thinking Rendering (Same cause as #1)
3. 402 Error causing TaskStatus.PAUSED contradiction
4. Vague "Project analyzed." response due to LLM loop bypass on error
5. CompletionFormatter treating PAUSED as FAILED in headline
6. Competing/Legacy Renderers (`LiveTerminalUI` vs `ConversationRenderer`)
7. `AgentLoop` default bypasses `EngineeringRuntime`
8. `did_work = True` logic in `AgentLoop` misinterprets "read-only" workflow progress as "state to preserve"
9. `InteractiveShell` mixes direct `console.print` with `ConversationRenderer`
10. ...

# 24. Recommended Fix Order
1. Remove `conv.start(goal)` from `InteractiveShell._execute_task` to prevent EventBus duplication.
2. Fix `AgentLoop` to handle `402` and `RuntimeError` by emitting a truthful `task.failed` or `task.completed` with an error message, circumventing the PAUSED state for read-only workflows.
3. Remove the hardcoded `"Project analyzed."` string in `loop.py` or ensure it is only set if the LLM loop succeeds.
4. Update `InteractiveShell` completion logic to accurately represent `PAUSED` vs `FAILED` without contradiction.
5. Standardize on `EngineeringRuntime` (`mode="unified"`) or fully deprecate legacy dashboards.

# 25. Files That Should Be Changed
- `src/harness_core/cli/interactive.py`
- `src/harness_core/agent/loop.py`

# 26. Files That MUST NOT Be Changed for UI Fixes
- `src/harness_core/providers/*` (Provider API logic is sound)
- `src/harness_core/routing/fallback.py` (Error classification is mathematically sound; the consumer is flawed)

# 27. Proposed Future Architecture
All terminal UI rendering should strictly consume `EventBus` signals. `InteractiveShell` must not manually mutate `ConversationRenderer` state. `AgentLoop` must be refactored to align its lifecycle events directly with the Unified `EngineeringRuntime` or deprecated entirely in favor of the newer Unified path.
