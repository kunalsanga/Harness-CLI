"""The core agent loop — orchestrates the engineering workflow."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from harness_core.agent.types import (
    AgentConfig,
    FailureReason,
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.context.engine import ContextEngine
from harness_core.observability.events import Event, EventBus
from harness_core.permissions.manager import PermissionManager
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.routing.budgets import BudgetManager
from harness_core.routing.router import ModelRouter, RouterConfig
from harness_core.tools.base import Tool
from harness_core.verification.engine import VerificationEngine
from harness_core.verification.integrity import check_test_integrity, is_test_like_path
from harness_core.agent.todos import (
    apply_tool_result,
    apply_tool_started,
    fallback_todo_plan,
    reconcile_on_evidence,
    sanitize_todo_titles,
    select_todo,
)
from harness_core.agent.report import build_execution_report, files_from_task
from harness_core.agent.completion import can_complete_task, completion_blockers
from harness_core.tools.diagnosis import (
    classify_command_failure,
    normalize_shell_command,
    parse_test_counts,
)
from harness_core.tools.paths import cwd_in_workspace, resolve_in_workspace

if TYPE_CHECKING:
    from harness_core.routing.task_aware import TaskAwareRouter


# ── Execution governor limits ────────────────────────────────────────────
# Repeated identical failures: diagnose AND block blind retries at 2.
REPEAT_DIAGNOSIS_THRESHOLD = 2
REPEAT_BLOCK_THRESHOLD = 2
# Stagnation: warn at 2 no-progress iterations, stop at 3.
STAGNATION_WARN_AT = 2
STAGNATION_STOP_AT = 3
# How many times we may re-prompt a model that produced zero tool calls.
MAX_NO_TOOL_NUDGES = 2

# Phase 10: typed failure messages. Keys match FailureReason values.
_FAILURE_MESSAGES: dict[str, str] = {
    "model_unavailable": "⚠ Model temporarily unavailable",
    "model_rate_limited": "⚠ All free models rate limited (429)",
    "provider_auth_failure": "✗ Provider authentication failed",
    "payment_required": "✗ Model requires payment (402)",
    "tool_failure": "✗ Tool operation failed",
    "test_failure": "✗ Tests failed",
    "verification_failure": "✗ Verification failed",
    "user_cancelled": "⚠ Task cancelled",
    "task_stagnation": "✗ Task stagnation — no progress",
    "completion_invariant": "✗ Completion invariant violated",
    "permission_denied": "⚠ Permission denied",
    "budget_exceeded": "⚠ Budget exceeded",
    "unknown": "✗ Unknown error",
}

# Phase 11: pause-specific messaging
_PAUSED_MESSAGES: dict[str, str] = {
    "model_unavailable": "⚠ Model temporarily unavailable — task paused, state preserved",
    "model_rate_limited": "⚠ Free models temporarily unavailable (429) — task paused, state preserved",
    "provider_auth_failure": "⚠ Provider authentication failed — task paused, state preserved",
    "payment_required": "⚠ All viable models require payment — task paused, state preserved",
}


def _classify_model_failure(err: str) -> str:
    """Map a model error string to a typed FailureReason value."""
    low = (err or "").lower()
    if "429" in low or "rate limit" in low or "too many requests" in low:
        return "model_rate_limited"
    if "402" in low or "payment required" in low:
        return "payment_required"
    if "401" in low or "unauthorized" in low:
        return "provider_auth_failure"
    if "403" in low or "forbidden" in low:
        return "provider_auth_failure"
    if "all" in low and "failed" in low:
        return "model_unavailable"
    if "network" in low or "connection" in low or "timeout" in low:
        return "model_unavailable"
    return "model_unavailable"


# How many iterations may be spent in diagnosis mode before stopping.
MAX_DIAGNOSIS_ITERATIONS = 5
# History compaction: keep recent tool calls verbatim, summarize older.
HISTORY_CHAR_BUDGET = 120_000
KEEP_RECENT_CALLS = 10

_TEST_COMMAND_MARKERS = (
    "pytest", "npm test", "yarn test", "pnpm test", "bun test",
    "cargo test", "go test", "jest", "vitest", "mocha", "phpunit",
    "dotnet test", "gradle test", "mvn test",
)


class AgentLoop:
    """The core agent loop that drives the engineering workflow."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: list[Tool],
        workspace_root: Path | None = None,
        config: AgentConfig | None = None,
        event_bus: EventBus | None = None,
        router: ModelRouter | None = None,
        task_aware: TaskAwareRouter | None = None,
    ) -> None:
        self.provider = provider
        self.tools = {t.schema.name: t for t in tools}
        self.workspace_root = workspace_root or Path.cwd()
        self.config = config or AgentConfig()
        self.event_bus = event_bus or EventBus()
        self.router = router
        self.task_aware = task_aware
        self.budget = BudgetManager() if router is None else router.budget
        self.context_engine = ContextEngine(self.workspace_root)
        self.permission_manager = PermissionManager(
            self.workspace_root,
            autonomous_mode=self.config.autonomous_mode,
        )
        self.verification_engine = VerificationEngine(self.workspace_root)
        self._current_phase: str = ""
        self._recent_denials: list[tuple[str, str]] = []  # (tool_name, args_key)
        self._consecutive_denials: int = 0
        self._recent_failures: list[tuple[str, str, int]] = []  # (tool_name, args_key, exit_code)
        self._consecutive_failures: int = 0
        self._last_command_hash: str | None = None
        # Execution governor state
        self._project_info: dict[str, Any] | None = None
        self._failure_counts: dict[str, int] = {}  # action_key -> consecutive failures
        self._seen_actions: set[str] = set()  # action keys attempted (progress detection)
        self._stagnation_counter: int = 0
        self._diagnosis_active: bool = False
        self._diagnosis_iterations: int = 0
        self._no_tool_nudges: int = 0
        self._corrections: list[str] = []  # injected guidance messages
        self._modified_files: list[str] = []  # files written/edited (raw paths)
        self._had_test_failure: bool = False
        self._last_model_used: str = ""  # for model rotation on no-tool responses
        self._active_task: Task | None = None  # currently running task (cancellation reporting)
        self._completed_operations: set[str] = set()  # successful operation keys (Phase 8)
        self._pending_workflow_events: list = []  # deferred events from workflows

    def _workspace_snapshot_text(self) -> str:
        """Compact workspace snapshot for model context.

        Built from project discovery so the model never has to guess
        whether a workspace exists or probe the environment endlessly.
        """
        info = self._project_info
        if not info:
            return ""
        lines: list[str] = [f"Workspace root: {info.get('root', '')}"]
        files = info.get("files", [])
        if files:
            lines.append("Files: " + ", ".join(files[:20]))
        if info.get("languages"):
            lines.append("Languages: " + ", ".join(info["languages"]))
        if info.get("package_manager"):
            lines.append(f"Package manager: {info['package_manager']}")
        if info.get("has_tests"):
            lines.append("Test suite: detected")
        if info.get("has_git"):
            lines.append("Git repository: yes")
        if info.get("readme"):
            lines.append(f"README: {info['readme']}")
        return "\n".join(lines)

    def _workspace_has_files(self) -> bool:
        return bool(self._project_info and self._project_info.get("files"))

    def _system_prompt(self) -> str:
        """Build the workspace-aware system prompt."""
        base = """You are an autonomous software engineering agent operating INSIDE a real workspace.

Your goal is to complete engineering tasks reliably. You must:
1. Understand the task
2. Plan your approach
3. Execute tools to inspect and modify code
4. Verify your changes work
5. Report results with evidence

You have filesystem and execution tools. Use them to read, edit, write, search, and run commands.

WORKSPACE RULES:
- The workspace and its files are REAL and accessible through your tools.
- NEVER claim you lack visibility into the project. If you need information, use a tool.
- NEVER ask the user for information you can discover with tools (project type,
  tech stack, file names, test commands). Discover it yourself.
- Inspect relevant files BEFORE modifying them.
- Prefer acting over explaining. Do not answer a coding task with prose only."""

        snapshot = self._workspace_snapshot_text()
        if snapshot:
            base += f"\n\nCURRENT WORKSPACE (already discovered, do not re-probe):\n{snapshot}"

        return base + """

CRITICAL RULE: NEVER claim a task is complete when a required tool call failed.
The runtime execution results (exit codes, stderr) are the source of truth.
If a command exits with a non-zero exit code, the task is NOT complete.
Your text response cannot override actual tool failures.

If a command fails:
- Read the error output carefully
- Diagnose the root cause BEFORE retrying
- Fix the implementation code
- Run the command again
- Only claim success when the command passes
- NEVER run the exact same failing command more than twice; diagnose instead.
- NEVER weaken, delete, or skip tests just to make them pass. Fix the implementation.

IMPORTANT: If a tool call returns "permission denied", do NOT retry the same command.
Instead:
- Try a different approach that does not require the blocked operation
- If verification is blocked, skip verification and report what was completed
- Never retry a denied command more than once
- Accept the permission constraint and work within it

CRITICAL GIT RULE: NEVER invent Git identity (user.name, user.email).
- If git_identity check fails, report that identity is missing and stop.
- Do NOT run: git config user.name "Some Name"
- Do NOT run: git config user.email "some@email.com"
- Configure your OWN identity manually if needed.
- The agent must NEVER write fake placeholder identities into repositories.

Always verify your work before claiming success. Do not claim success without evidence.
However, if verification itself is blocked by permissions, report that clearly.


When you are done, summarize what you did and provide evidence of success."""

    @staticmethod
    def _action_key(tool_name: str, arguments: dict[str, Any]) -> str:
        """Normalized identity for a tool call (used for repetition detection).

        For run_command, the command is normalized (cd / shell wrappers
        stripped) so retrying the same underlying command through different
        shell formats counts as the same action — blocking shell-guessing loops.
        """
        args = arguments
        if tool_name == "run_command" and isinstance(arguments.get("command"), str):
            args = dict(arguments)
            args["command"] = normalize_shell_command(arguments["command"])
        return f"{tool_name}:{json.dumps(args, sort_keys=True)}"

    @staticmethod
    def _is_test_command(command: str) -> bool:
        """Detect whether a shell command runs a test suite."""
        cmd = command.lower()
        if any(marker in cmd for marker in _TEST_COMMAND_MARKERS):
            return True
        # e.g. "node test.js", "node tests/run.js"
        import re
        return bool(re.search(r"\b(node|python|python3|deno|bun)\s+\S*test\S*", cmd))

    def _compact_task_state(self, task: Task, compacted_count: int) -> str:
        """Build the compact TASK STATE summary for older tool history."""
        completed_steps = [
            i.description for i in task.task_plan.items
            if i.status == TodoStatus.COMPLETED
        ]
        lines = [
            "TASK STATE (summary of earlier work)",
            f"Goal: {task.goal}",
        ]
        if self._workspace_has_files():
            lines.append(f"Workspace: {self._project_info.get('root')}")
        if completed_steps:
            lines.append("Completed: " + "; ".join(completed_steps[:6]))
        if self._modified_files:
            lines.append("Changed files: " + ", ".join(self._modified_files[-10:]))
        lines.append(f"{compacted_count} earlier tool call(s) were made and are summarized above.")
        last_failure = next(
            (tc for tc in reversed(task.tool_calls)
             if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied),
            None,
        )
        if last_failure and last_failure.result:
            cmd = last_failure.arguments.get("command", last_failure.tool_name)
            lines.append(f"Latest failure: {cmd} (exit code {last_failure.result.exit_code})")
            err = (last_failure.result.error or "")[:300]
            if err:
                lines.append(f"Error: {err}")
        return "\n".join(lines)

    def _build_messages(
        self,
        task: Task,
        context: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for the model.

        Keeps a compact task state: recent tool calls verbatim, older
        history summarized, corrections (governor guidance) appended last.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]

        # Add context
        if context:
            for piece in context:
                messages.append(
                    {"role": "system", "content": f"[Context: {piece.source}]\n{piece.content}"}
                )

        # Add task
        messages.append({"role": "user", "content": task.goal})

        # Determine which tool calls to keep verbatim (context management).
        calls_with_results = [tc for tc in task.tool_calls if tc.result]
        total_chars = sum(
            len(tc.result.output or "") + len(tc.result.error or "")
            for tc in calls_with_results
        )
        if len(calls_with_results) > KEEP_RECENT_CALLS and total_chars > HISTORY_CHAR_BUDGET:
            compacted = calls_with_results[:-KEEP_RECENT_CALLS]
            recent = calls_with_results[-KEEP_RECENT_CALLS:]
            messages.append(
                {"role": "system", "content": self._compact_task_state(task, len(compacted))}
            )
        else:
            recent = calls_with_results

        # Add previous tool call results
        for tc in recent:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tc.result.output or tc.result.error or "(no output)",
                }
            )

        # Governor guidance (nudge / diagnosis / stagnation) — most recent last
        for correction in self._corrections[-3:]:
            messages.append({"role": "user", "content": correction})

        return messages

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Get LLM-compatible tool schemas."""
        return [t.to_llm_schema() for t in self.tools.values()]

    def _check_repeated_deny(self, call: ToolCall) -> bool:
        """Check if this exact tool call has already been denied.

        Returns True if we should block the repeated denied call.
        """
        args_key = json.dumps(call.arguments, sort_keys=True)
        for prev in self._recent_denials:
            if prev[0] == call.tool_name and prev[1] == args_key:
                return True
        return False

    def _record_denial(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record a permission denial for loop guard tracking."""
        args_key = json.dumps(arguments, sort_keys=True)
        self._recent_denials.append((tool_name, args_key))
        if len(self._recent_denials) > 50:
            self._recent_denials = self._recent_denials[-50:]
        self._consecutive_denials += 1

    def _reset_denial_tracking(self) -> None:
        """Reset consecutive denial counter after a successful tool call."""
        self._consecutive_denials = 0

    def _record_failure(self, tool_name: str, arguments: dict[str, Any], exit_code: int = -1) -> None:
        """Record a tool execution failure for repetition detection."""
        args_key = json.dumps(arguments, sort_keys=True)
        self._recent_failures.append((tool_name, args_key, exit_code))
        if len(self._recent_failures) > 50:
            self._recent_failures = self._recent_failures[-50:]
        self._consecutive_failures += 1

    def _reset_failure_tracking(self) -> None:
        """Reset consecutive failure counter after a successful tool call."""
        self._consecutive_failures = 0

    def _is_repeating_failure(self, call: ToolCall) -> bool:
        """Check if this exact operation has failed too many times.

        Hard-blocks blind retries at REPEAT_BLOCK_THRESHOLD; diagnosis
        mode is triggered earlier, at REPEAT_DIAGNOSIS_THRESHOLD.
        """
        key = self._action_key(call.tool_name, call.arguments)
        return self._failure_counts.get(key, 0) >= REPEAT_BLOCK_THRESHOLD

    async def _trigger_diagnosis(self, call: ToolCall, attempts: int) -> None:
        """Switch from blind execution into diagnosis mode."""
        self._diagnosis_active = True
        self._diagnosis_iterations = 0
        command = call.arguments.get("command", call.tool_name)
        await self._emit_phase("diagnosing")
        await self._emit_thinking("Repeated failure detected — diagnosing before retrying.")
        await self.event_bus.emit(
            Event(
                type="diagnosis.triggered",
                source="agent_loop",
                data={
                    "tool": call.tool_name,
                    "command": command,
                    "attempts": attempts,
                },
            )
        )
        self._corrections.append(
            f"WARNING: The same operation has failed {attempts} times in a row: "
            f"{command}. Do NOT run it again unchanged. Switch to diagnosis:\n"
            "1. Read the failing test/source files involved.\n"
            "2. Inspect the stderr/stack trace and identify the failing assertion or error.\n"
            "3. Determine the root cause in the IMPLEMENTATION code.\n"
            "4. Apply one targeted fix to the implementation. "
            "Do NOT weaken, delete, or skip tests to force a pass.\n"
            "5. Then re-run the command once."
        )

    async def _record_execution_outcome(self, call: ToolCall, result: ToolResult) -> None:
        """Update governor state after a tool execution."""
        key = self._action_key(call.tool_name, call.arguments)
        command = call.arguments.get("command", "")

        if result.execution_failed and not result.is_perm_denied:
            self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
            if call.tool_name == "run_command" and self._is_test_command(command):
                self._had_test_failure = True
                await self._emit_phase("testing")
            if self._failure_counts[key] == REPEAT_DIAGNOSIS_THRESHOLD:
                await self._trigger_diagnosis(call, self._failure_counts[key])
            return

        # Successful execution
        self._failure_counts.pop(key, None)
        if call.tool_name == "run_command":
            if self._is_test_command(command):
                await self._emit_phase("testing")
                if self._diagnosis_active:
                    # Tests pass again — diagnosis resolved
                    self._diagnosis_active = False
                    self._diagnosis_iterations = 0
        elif call.tool_name in ("write_file", "edit_file"):
            path = call.arguments.get("path", call.arguments.get("file_path", ""))
            if path and path not in self._modified_files:
                self._modified_files.append(path)
            # Code changed: earlier command failures may now be obsolete
            self._failure_counts.clear()
            await self._emit_phase("fixing" if self._diagnosis_active else "implementing")

    def _iteration_made_progress(self, iter_calls: list[ToolCall]) -> bool:
        """True when this iteration produced meaningful progress.

        Progress = a new (never-attempted) action, or a successful file
        modification. Repeating known actions or emitting only prose is
        not progress.
        """
        if not iter_calls:
            return False
        for tc in iter_calls:
            key = self._action_key(tc.tool_name, tc.arguments)
            if key not in self._seen_actions:
                return True
            if (
                tc.tool_name in ("write_file", "edit_file")
                and tc.result is not None
                and tc.result.status == ToolResultStatus.SUCCESS
            ):
                return True
        return False

    # ── Failure reason classification ─────────────────────────────────

    def _classify_failure_reason(self, err: str) -> str:
        """Map an error string to a typed FailureReason value."""
        low = (err or "").lower()
        if "429" in low or "rate limit" in low or "too many requests" in low:
            return "model_rate_limited"
        if "402" in low or "payment required" in low:
            return "payment_required"
        if "401" in low or "unauthorized" in low:
            return "provider_auth_failure"
        if "403" in low or "forbidden" in low:
            return "provider_auth_failure"
        if "all" in low and "failed" in low:
            return "model_unavailable"
        if "network" in low or "connection" in low or "timeout" in low:
            return "model_unavailable"
        return "unknown"

    # ── Workflow helpers ──────────────────────────────────────────────

    def _workflow_context(self) -> Any:
        """Build a WorkflowContext for the currently-running task."""
        from harness_core.agent.workflows import WorkflowContext
        return WorkflowContext(
            workspace=self.workspace_root,
            tools=self.tools,
            event_bus=self.event_bus,
            modified_files=self._modified_files,
            git_commit=getattr(self._active_task, "git_commit", None) if self._active_task else None,
            git_push=getattr(self._active_task, "git_push", None) if self._active_task else None,
        )

    def _apply_workflow_result(self, task: Task, result: Any) -> None:
        """Copy workflow outputs onto the task and emit per-step events.

        Emits the structured todo lifecycle events that the UX layer
        listens for. Called once per workflow run.
        """
        from harness_core.observability.events import Event
        data = result.data or {}
        if data.get("commit"):
            task.git_commit = data["commit"]
        if data.get("push"):
            task.git_push = data["push"]
        for op in result.completed_operations:
            self._completed_operations.add(op)
            task.completed_operations.append(op)
        # Stash TODO events for emission by an async wrapper
        for item in task.task_plan.items:
            if not getattr(item, "_workflow_emitted", False):
                item._workflow_emitted = True
                ev_type = (
                    "todo.completed"
                    if item.status == TodoStatus.COMPLETED
                    else "todo.failed" if item.status == TodoStatus.FAILED
                    else "todo.started" if item.status == TodoStatus.IN_PROGRESS
                    else None
                )
                if ev_type:
                    # Defer to the loop's event loop via the existing bus
                    # by storing pending events for an async emit pass.
                    self._pending_workflow_events.append(Event(
                        type=ev_type,
                        source="workflow",
                        data={
                            "todo_id": item.id,
                            "title": item.description,
                            "status": item.status.value,
                            "evidence": item.evidence,
                            "error": item.error,
                        },
                    ))

    async def _flush_workflow_events(self) -> None:
        """Emit any deferred workflow events on the bus."""
        pending = self._pending_workflow_events
        self._pending_workflow_events = []
        for ev in pending:
            await self.event_bus.emit(ev)
        await self._emit_todo_update(self._active_task) if self._active_task else None

    # ── Plan validation ────────────────────────────────────────────────

    def _validate_plan_steps(self, steps: list[str]) -> list[str]:
        """Keep only actionable engineering tasks; drop conversational prose.

        Every TODO must begin with an actionable engineering verb. Questions,
        requests for information, claims of missing visibility, and chatter are
        rejected — the workspace is discoverable via tools.
        """
        return sanitize_todo_titles(steps)

    def _default_plan_steps(self, goal: str = "") -> list[str]:
        """Deterministic fallback plan from the task + workspace.

        Never conversational. Used when the model proposes nothing usable.
        """
        return fallback_todo_plan(goal, self._project_info)

    # ── Completion verification ────────────────────────────────────────

    async def _verify_task_completion(self, task: Task) -> None:
        """Verify claimed completion truthfully.

        Runs test-integrity review plus ecosystem verification checks
        when the task modified files. On failure the task is marked
        FAILED — success is never claimed without evidence.
        """
        if not self.config.verify_on_complete or not self._modified_files:
            return

        # Test integrity review (Phase: don't let agents weaken tests)
        try:
            integrity = check_test_integrity(self.workspace_root, self._modified_files)
        except Exception:
            integrity = None
        if integrity and integrity.suspicious:
            await self.event_bus.emit(
                Event(
                    type="test_integrity.warning",
                    source="agent_loop",
                    data={
                        "warning": integrity.warning,
                        "files": integrity.test_files_modified,
                    },
                )
            )

        await self._emit_phase("verifying")
        await self.event_bus.emit(
            Event(
                type="verification.started",
                source="agent_loop",
                data={"files": list(self._modified_files)},
            )
        )

        try:
            checks = await self.verification_engine.detect_ecosystem()
        except Exception:
            checks = []

        if not checks:
            # No automated checks for this ecosystem — verify files exist
            missing = []
            for f in self._modified_files:
                p = Path(f)
                if not p.is_absolute():
                    try:
                        p = Path(self.workspace_root) / f
                    except TypeError:
                        p = Path(f)
                if not p.exists():
                    missing.append(f)
            if missing:
                task.status = TaskStatus.FAILED
                task.verification_passed = False
                task.verification_summary = f"modified files missing: {', '.join(missing)}"
                task.error = (
                    "Verification failed: modified files do not exist on disk: "
                    f"{', '.join(missing)}"
                )
            else:
                task.verification_passed = True
                task.verification_summary = "changed files present; no automated checks detected"
            await self.event_bus.emit(
                Event(
                    type="verification.completed",
                    source="agent_loop",
                    data={"passed": task.verification_passed, "checks_run": 0},
                )
            )
            return

        report = await self.verification_engine.run_checks(checks[:2])
        passed = report.all_passed
        task.verification_passed = passed
        details = "; ".join(
            f"{r.check_name}: {'passed' if r.passed else 'FAILED'}"
            for r in report.results
        )
        task.verification_summary = details
        await self.event_bus.emit(
            Event(
                type="verification.completed",
                source="agent_loop",
                data={
                    "passed": passed,
                    "checks_run": report.checks_run,
                    "checks_passed": report.checks_passed,
                },
            )
        )

        if not passed:
            failed_output = ""
            for r in report.results:
                if not r.passed:
                    snippet = (r.output or r.error or "")[-800:]
                    failed_output += f"\n[{r.check_name}]\n{snippet}\n"
            task.status = TaskStatus.FAILED
            task.error = (
                "Verification failed: the implementation does not pass the "
                f"project's checks.\n{details}\n{failed_output}".strip()
            )

    def _should_block_completion(self, task: Task) -> bool:
        """Determine if the task should NOT be marked COMPLETED.

        Hard invariant: TOOL FAILURE ≠ TASK SUCCESS

        Returns True when the task must NOT transition to COMPLETED.
        """
        if not task.tool_calls:
            return False  # No tool calls — model can finish freely

        # Check if any tool calls failed (execution failure, not permission denied)
        failed_executions = [
            tc for tc in task.tool_calls
            if tc.result is not None
            and tc.result.execution_failed
            and not tc.result.is_perm_denied
        ]

        if not failed_executions:
            return False  # All tool calls succeeded

        # There are failed executions. Check if there was any subsequent success
        # after the last failure (recovery happened).
        last_failure_idx = -1
        for i, tc in enumerate(task.tool_calls):
            if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied:
                last_failure_idx = i

        # Check if there's a success after the last failure
        has_recovery = False
        if last_failure_idx >= 0:
            for tc in task.tool_calls[last_failure_idx + 1:]:
                if tc.result and tc.result.status == ToolResultStatus.SUCCESS:
                    has_recovery = True
                    break

        if has_recovery:
            return False  # Agent recovered successfully

        # Failed executions exist with no recovery — block completion
        return True

    async def _emit_phase(self, phase: str) -> None:
        """Emit a task phase change event for progress tracking."""
        if phase != self._current_phase:
            self._current_phase = phase
            await self.event_bus.emit(
                Event(
                    type="task.phase",
                    source="agent_loop",
                    data={"phase": phase, "task_id": ""},
                )
            )

    async def _emit_thinking(self, message: str, task: Task | None = None) -> None:
        """Emit a thinking status event (high-level execution intent)."""
        if task:
            task.thinking = message
        await self.event_bus.emit(
            Event(
                type="thinking.status",
                source="agent_loop",
                data={"message": message},
            )
        )

    async def _emit_todo_update(self, task: Task) -> None:
        """Emit current TODO list state (display + structured items)."""
        items = task.task_plan.display()
        completed = task.task_plan.completed_count
        total = task.task_plan.total_count
        await self.event_bus.emit(
            Event(
                type="todo.updated",
                source="agent_loop",
                data={
                    "items": items,
                    "todos": task.task_plan.to_event_items(),
                    "completed": completed,
                    "failed": task.task_plan.failed_count,
                    "total": total,
                },
            )
        )

    async def _emit_todo_event(self, event_type: str, item: TodoItem) -> None:
        """Emit a per-TODO lifecycle event with structured metadata."""
        await self.event_bus.emit(
            Event(
                type=event_type,
                source="agent_loop",
                data={
                    "todo_id": item.id,
                    "title": item.description,
                    "status": item.status.value,
                    "evidence": item.evidence,
                    "error": item.error,
                },
            )
        )

    async def _todo_started(self, task: Task, call: ToolCall) -> None:
        """Mark the best-matching TODO in progress from a real tool start."""
        before = select_todo(task.task_plan, call.tool_name, call.arguments)
        was_pending = before is not None and before.status == TodoStatus.PENDING
        item = apply_tool_started(task.task_plan, call)
        if item is not None and was_pending:
            await self._emit_todo_event("todo.started", item)
            await self._emit_todo_update(task)

    async def _todo_result(self, task: Task, call: ToolCall, result: ToolResult) -> None:
        """Update TODO state from the real tool outcome (evidence-based)."""
        item = apply_tool_result(task.task_plan, call, result)
        if item is None:
            return
        if item.status == TodoStatus.COMPLETED and item.completed_at:
            await self._emit_todo_event("todo.completed", item)
            await self._emit_todo_update(task)
        elif item.status == TodoStatus.FAILED:
            await self._emit_todo_event("todo.failed", item)
            await self._emit_todo_update(task)

    async def _postprocess_result(self, task: Task, call: ToolCall, result: ToolResult) -> None:
        """Enrich a tool result with diagnosis, test accounting, and git state."""
        tool = call.tool_name

        # Command failure diagnosis + test accounting
        if tool == "run_command":
            command = str(call.arguments.get("command", ""))
            if result.execution_failed:
                diagnosis = classify_command_failure(
                    command,
                    exit_code=result.exit_code,
                    stdout=result.output or "",
                    stderr=result.stderr or "",
                    timed_out=(result.status == ToolResultStatus.TIMEOUT),
                )
                result.metadata["diagnosis_category"] = diagnosis.category
                result.metadata["diagnosis_reason"] = diagnosis.reason
                hint = (
                    f"\nDiagnosis: {diagnosis.category} — {diagnosis.reason}"
                )
                if diagnosis.category in ("command_syntax", "missing_executable"):
                    hint += (
                        "\nDo NOT retry with different shell wrappers (cd, cmd /c, powershell). "
                        "Fix the underlying command or use the workspace-aware run_command."
                    )
                if result.error:
                    result.error = result.error + hint
                else:
                    result.error = hint.strip()
            # Test accounting (success or failure) if this ran a suite
            if self._is_test_command(command):
                counts = parse_test_counts(result.output or "")
                if result.status == ToolResultStatus.SUCCESS and counts is None:
                    # Passed but no parseable summary — still a passing run
                    task.tests_run = task.tests_run or 1
                    task.tests_passed = task.tests_passed if task.tests_passed is not None else task.tests_run
                elif counts is not None:
                    passed, total = counts
                    task.tests_run = total
                    task.tests_passed = passed
                await self.event_bus.emit(
                    Event(
                        type="test.completed",
                        source="agent_loop",
                        data={
                            "command": command,
                            "passed": task.tests_passed,
                            "total": task.tests_run,
                            "success": result.status == ToolResultStatus.SUCCESS,
                        },
                    )
                )

        # Structured git accounting
        elif tool == "git_commit":
            await self._emit_phase("committing")
            if result.status == ToolResultStatus.SUCCESS:
                commit_hash = result.metadata.get("commit_hash") or self._parse_commit_hash(result.output or "")
                if commit_hash:
                    task.git_commit = commit_hash
        elif tool == "git_add" or tool == "git_stage":
            await self._emit_phase("committing")
        elif tool == "git_push":
            await self._emit_phase("pushing")
            if result.status == ToolResultStatus.SUCCESS:
                remote = result.metadata.get("remote", "origin")
                branch = result.metadata.get("branch", "")
                task.git_push = f"{remote}/{branch}" if branch else remote

    @staticmethod
    def _parse_commit_hash(output: str) -> str:
        """Extract a short commit hash from `git commit` output."""
        import re
        m = re.search(r"\[([^\s\]]+)\s+([0-9a-f]{7,40})\]", output)
        if m:
            return m.group(2)
        m = re.search(r"\b([0-9a-f]{40})\b", output)
        return m.group(1)[:12] if m else ""

    async def _stagnation_recovery(self, task: Task) -> bool:
        """Stagnation stopped the loop — complete truthfully if evidence is green.

        If files were changed, the last test run (if any) succeeded, and
        verification passes, the work is done even though the model kept
        issuing no-progress actions. Otherwise the task genuinely failed.
        The completion invariant is the gate: can_complete_task() must pass.
        """
        if not self._modified_files:
            return False
        for tc in reversed(task.tool_calls):
            if tc.tool_name == "run_command" and self._is_test_command(
                str(tc.arguments.get("command", ""))
            ):
                if tc.result is not None and tc.result.execution_failed:
                    return False
                break
        await self._verify_task_completion(task)
        if task.status == TaskStatus.FAILED:
            return False
        await self._reconcile_todos(task)

        # Completion invariant: runtime truth, not stagnation mercy
        if not can_complete_task(task):
            blockers = completion_blockers(task)
            task.status = TaskStatus.FAILED
            task.error = (
                "Stopped repeating no-progress iterations but "
                + "; ".join(blockers)
            )
            return False

        task.status = TaskStatus.COMPLETED
        task.error = None
        task.result = (
            "Stopped repeating no-progress iterations; final state verified. "
            + (task.result or "")
        )
        await self._emit_phase("complete")
        return True

    async def _reconcile_todos(self, task: Task) -> None:
        """Complete/fail open TODOs using real execution evidence."""
        did_inspect = any(
            tc.tool_name in ("read_file", "list_files", "glob", "grep")
            and tc.result and tc.result.status == ToolResultStatus.SUCCESS
            for tc in task.tool_calls
        )
        did_implement = bool(self._modified_files)
        tests_passed: bool | None = None
        if task.tests_run > 0 and task.tests_passed is not None:
            tests_passed = task.tests_passed >= task.tests_run
        did_commit = bool(task.git_commit)
        did_push = bool(task.git_push)

        changed = reconcile_on_evidence(
            task.task_plan,
            did_inspect=did_inspect,
            did_implement=did_implement,
            tests_passed=tests_passed,
            verified=task.verification_passed,
            did_commit=did_commit,
            did_push=did_push,
        )
        if changed:
            for item in changed:
                event_type = (
                    "todo.completed" if item.status == TodoStatus.COMPLETED else "todo.failed"
                )
                await self._emit_todo_event(event_type, item)
            await self._emit_todo_update(task)

    def _get_failure_summary(self, task: Task) -> str:
        """Build a summary of failed tool calls for the task error message."""
        failures = []
        for tc in task.tool_calls:
            if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied:
                cmd = tc.arguments.get("command", tc.tool_name)
                exit_code = tc.result.exit_code
                stderr = tc.result.stderr
                parts = [f"Command: {cmd}"]
                if exit_code is not None:
                    parts.append(f"Exit code: {exit_code}")
                if stderr:
                    # Truncate long stderr
                    truncated = stderr[:500] + ("..." if len(stderr) > 500 else "")
                    parts.append(f"stderr: {truncated}")
                failures.append("\n  ".join(parts))
        return "\n\nFailed commands:\n" + "\n---\n".join(failures) if failures else ""

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with permission and governor checking.

        Guarantees call.result is populated on every path (including
        denials and blocked retries) so accounting stays truthful.
        """
        result = await self._execute_tool_checked(call)
        call.result = result
        return result

    # File tools whose "path" argument must stay inside the workspace.
    _PATH_TOOLS_REQUIRED = {"read_file", "write_file", "edit_file"}
    _PATH_TOOLS_OPTIONAL = {"list_files", "glob", "grep"}
    _CWD_TOOLS = {"run_command", "git_status", "git_diff", "git_log", "git_add",
                  "git_commit", "git_push", "git_remote", "git_identity"}

    def _confine_call(self, call: ToolCall) -> ToolResult | None:
        """Enforce the workspace boundary on paths and working directory.

        Returns a denial ToolResult if a path escapes the workspace, else None.
        """
        name = call.tool_name
        args = call.arguments

        if name in self._PATH_TOOLS_REQUIRED or name in self._PATH_TOOLS_OPTIONAL:
            raw = args.get("path") or args.get("file_path")
            if raw or name in self._PATH_TOOLS_REQUIRED:
                resolved, err = resolve_in_workspace(self.workspace_root, raw)
                if err is not None:
                    return ToolResult(
                        status=ToolResultStatus.PERMISSION_DENIED,
                        output="",
                        error=f"Path confinement: {err}",
                        retryable=False,
                    )
                if resolved is not None:
                    if "path" in args or name in self._PATH_TOOLS_REQUIRED:
                        args["path"] = str(resolved)
                    if "file_path" in args:
                        args["file_path"] = str(resolved)

        if name in self._CWD_TOOLS:
            cwd, _ = cwd_in_workspace(self.workspace_root, args.get("cwd"))
            args["cwd"] = cwd

        return None

    async def _execute_tool_checked(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with permission checking.

        Added validation to ensure required arguments are present and to
        provide structured errors for missing arguments, improving reliability.
        """
        tool = self.tools.get(call.tool_name)
        if not tool:
            self._reset_denial_tracking()
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Unknown tool: {call.tool_name}",
                retryable=False,
            )
        # --- BEGIN VALIDATION BLOCK ---
        
        required_args = tool.schema.parameters.get("required", [])
        missing = [arg for arg in required_args if arg not in call.arguments]
        if missing:
            # Record a failure to enforce bounded correction attempts
            self._record_failure(call.tool_name, call.arguments, exit_code=-1)
            missing_msg = ", ".join(missing)
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Missing required argument(s): {missing_msg}",
                retryable=False,
            )
        # --- END VALIDATION BLOCK ---
        tool = self.tools.get(call.tool_name)
        if not tool:
            self._reset_denial_tracking()
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Unknown tool: {call.tool_name}",
                retryable=False,
            )

        
        
        required_args = tool.schema.parameters.get("required", [])
        missing = [arg for arg in required_args if arg not in call.arguments]
        if missing:
            # Record a failure to enforce bounded correction attempts
            self._record_failure(call.tool_name, call.arguments, exit_code=-1)
            missing_msg = ", ".join(missing)
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Missing required argument(s): {missing_msg}",
                retryable=False,
            )
        # Hard workspace boundary: confine file paths and working directory.
        confinement_denial = self._confine_call(call)
        if confinement_denial is not None:
            self._record_denial(call.tool_name, call.arguments)
            return confinement_denial

        # Check if this exact call was already denied
        if self._check_repeated_deny(call):
            self._consecutive_denials += 1
            return ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output="",
                error=(
                    "Permission denied (already rejected). "
                    "This command cannot be retried under the current policy. "
                    "Ask for user approval or choose a different approach."
                ),
                retryable=False,
            )

        # Check permission
        permission = self.permission_manager.check_permission(call.tool_name, call.arguments)
        if permission == "deny":
            self._record_denial(call.tool_name, call.arguments)
            return ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output="",
                error="Permission denied by policy",
                retryable=False,
            )
        if permission == "ask" and not self.permission_manager.request_approval(
            call.tool_name, str(call.arguments)
        ):
            self._record_denial(call.tool_name, call.arguments)
            return ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output="",
                error=(
                    "Permission denied. This command requires approval. "
                    "Ask the user for permission or use a different approach."
                ),
                retryable=False,
            )

        # Check for repeated failures of the same operation
        if self._is_repeating_failure(call):
            key = self._action_key(call.tool_name, call.arguments)
            count = self._failure_counts.get(key, 0)
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=(
                    f"BLOCKED: this exact operation has already failed {count} times. "
                    f"Running it again without changes is not allowed. "
                    f"Diagnose the root cause (read the relevant files and error output), "
                    f"apply a fix to the implementation, then retry."
                ),
                retryable=False,
            )

        # Phase 8: skip operations that already succeeded (duplicate prevention).
        # Tools whose results are idempotent: git_status, git_remote, git_log,
        # git_diff, read_file, list_files, glob. Other tools always run.
        op_key = self._action_key(call.tool_name, call.arguments)
        if (
            call.tool_name
            in {
                "git_status", "git_remote", "git_log", "git_diff",
                "read_file", "list_files", "glob",
            }
            and op_key in self._completed_operations
        ):
            # Return the cached "already done" result without re-executing.
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output="(already executed — cached)",
                metadata={"cached": True, "operation": op_key},
            )

        # Execute with a bounded timeout so a hung tool can't freeze the CLI.
        call_timeout = float(call.arguments.get("timeout") or tool.schema.timeout_seconds or 30.0)
        call_timeout = min(max(call_timeout, 1.0), 300.0)
        start = time.time()
        try:
            result = await asyncio.wait_for(tool.execute(call.arguments), timeout=call_timeout)
            call.duration_ms = (time.time() - start) * 1000
            call.result = result
            self._reset_denial_tracking()
            if result.execution_failed:
                self._record_failure(
                    call.tool_name, call.arguments, result.exit_code or -1
                )
            else:
                # Record successful operation for Phase 8 duplicate prevention
                self._completed_operations.add(op_key)
                self._reset_failure_tracking()
            await self._record_execution_outcome(call, result)
            return result
        except asyncio.TimeoutError:
            call.duration_ms = (time.time() - start) * 1000
            self._reset_denial_tracking()
            self._record_failure(call.tool_name, call.arguments, -1)
            timeout_result = ToolResult(
                status=ToolResultStatus.TIMEOUT,
                output="",
                error=f"Command timed out after {call_timeout:.0f}s",
                retryable=True,
            )
            call.result = timeout_result
            await self._record_execution_outcome(call, timeout_result)
            return timeout_result
        except Exception as e:
            call.duration_ms = (time.time() - start) * 1000
            self._reset_denial_tracking()
            self._record_failure(call.tool_name, call.arguments, -1)
            error_result = ToolResult(
                status=ToolResultStatus.ERROR, output="", error=str(e),
                retryable=True,
            )
            call.result = error_result
            await self._record_execution_outcome(call, error_result)
            return error_result

    async def run(self, goal: str) -> Task:
        """Run the agent loop for a given goal."""
        task = Task(goal=goal, max_iterations=self.config.max_iterations)
        self._active_task = task

        # Reset per-task governor state (loops may be reused across tasks)
        self._corrections = []
        self._seen_actions = set()
        self._failure_counts = {}
        self._stagnation_counter = 0
        self._diagnosis_active = False
        self._diagnosis_iterations = 0
        self._no_tool_nudges = 0
        self._modified_files = []
        self._had_test_failure = False
        self._last_model_used = ""
        self._current_phase = ""
        self._recent_denials = []
        self._consecutive_denials = 0
        self._recent_failures = []
        self._consecutive_failures = 0
        self._completed_operations = set()
        self._pending_workflow_events = []
        self.budget.reset()

        await self.event_bus.emit(
            Event(type="task.started", source="agent_loop", data={"goal": goal})
        )
        await self._emit_phase("understanding")

        # Phase 16: intent fast paths — deterministic workflows run without
        # unbounded LLM iteration when the intent matches.
        from harness_core.agent.workflows import (
            classify_workflow,
            run_git_push_workflow,
            run_explain_workflow,
        )
        workflow_name = classify_workflow(goal)
        if workflow_name in ("git_push", "explain", "test"):
            await self._emit_phase("workflow")
            if workflow_name == "git_push":
                wf_result = await run_git_push_workflow(
                    ctx=self._workflow_context(),
                    plan=task.task_plan,
                )
            elif workflow_name == "test":
                from harness_core.agent.workflows import run_test_workflow
                wf_result = await run_test_workflow(
                    ctx=self._workflow_context(),
                    plan=task.task_plan,
                )
            else:  # explain
                wf_result = await run_explain_workflow(
                    ctx=self._workflow_context(),
                    plan=task.task_plan,
                )
            self._apply_workflow_result(task, wf_result)
            await self._flush_workflow_events()
            if wf_result.success:
                # Completion invariant: workflow success ≠ task COMPLETED.
                # Every workflow must pass can_complete_task before finishing.
                await self._reconcile_todos(task)
                if can_complete_task(task):
                    await self._emit_phase("complete")
                    task.status = TaskStatus.COMPLETED
                else:
                    blockers = completion_blockers(task)
                    await self._emit_phase("complete")
                    task.status = TaskStatus.FAILED
                    task.error = (
                        "Workflow completed but task cannot be marked done: "
                        + "; ".join(blockers)
                    )
            else:
                await self._emit_phase("complete")
                task.status = TaskStatus.FAILED
                task.error = wf_result.failure_reason
                task.failure_reason = wf_result.failure_reason
            # Emit final event
            await self.event_bus.emit(
                Event(
                    type="task.completed",
                    source="agent_loop",
                    data={
                        "task_id": task.id,
                        "status": task.status.value,
                        "failure_reason": task.failure_reason,
                        "iterations": task.iterations,
                        "tool_calls": len(task.tool_calls),
                        "stats": task.execution_stats.summary(),
                        "attempted": task.execution_stats.attempted,
                        "succeeded": task.execution_stats.succeeded,
                        "failed": task.execution_stats.failed,
                        "recovered": task.execution_stats.recovered,
                        "unresolved": task.execution_stats.unresolved,
                        "verification_passed": task.verification_passed,
                        "verification_summary": task.verification_summary,
                        "files_changed": list(self._modified_files),
                        "completed_operations": list(self._completed_operations),
                        "todos": task.task_plan.to_event_items(),
                        "todos_completed": task.task_plan.completed_count,
                        "todos_failed": task.task_plan.failed_count,
                        "todos_total": task.task_plan.total_count,
                        "tests_run": task.tests_run,
                        "tests_passed": task.tests_passed,
                        "models_used": list(task.models_used),
                        "model_fallbacks": task.model_fallbacks,
                        "git_commit": task.git_commit,
                        "git_push": task.git_push,
                        "paused_reason": task.paused_reason,
                    },
                )
            )
            return task

        # Classify task if task_aware router is available
        task_type = None
        task_profile = None
        classification_confidence = 0.0
        if self.task_aware is not None:
            # Build a minimal request for classification
            classify_request = CompletionRequest(messages=[{"role": "user", "content": goal}])
            task_type, task_profile, classification_confidence = self.task_aware.classify_task(classify_request)
            await self.event_bus.emit(
                Event(
                    type="task.classified",
                    source="agent_loop",
                    data={
                        "task_type": task_type.value if task_type else "unknown",
                        "confidence": classification_confidence,
                    },
                )
            )

        # Discover project — workspace intelligence before planning
        project_info = await self.context_engine.discover_project()
        self._project_info = project_info

        # Assemble context
        context = await self.context_engine.assemble_context(goal, project_info)

        # Planning phase: ask model to create a concise plan grounded in
        # the discovered workspace (never in imagined context).
        await self._emit_phase("planning")
        try:
            plan_user_content = goal
            snapshot = self._workspace_snapshot_text()
            if snapshot:
                plan_user_content += (
                    "\n\nThe workspace has already been discovered:\n"
                    f"{snapshot}\n"
                    "Base the plan on these real files. Do not ask the user for "
                    "information that is discoverable with tools."
                )
            plan_messages = [
                {"role": "system", "content": (
                    "You are planning an engineering task. Respond with a numbered list of steps.\n"
                    "Be concise: 3-6 steps maximum. Each step must be one short, EXECUTABLE line\n"
                    "(an action the agent will perform), e.g. 'Inspect calculator.js'.\n"
                    "No questions. No explanations. No requests for more information.\n"
                    "Example:\n"
                    "1. Inspect current code\n"
                    "2. Implement changes\n"
                    "3. Run tests\n"
                    "4. Verify results"
                )},
                {"role": "user", "content": plan_user_content},
            ]
            plan_request = CompletionRequest(messages=plan_messages)
            if self.router is not None:
                plan_result = await self.router.execute(plan_request)
                if plan_result.succeeded and plan_result.response:
                    plan_text = plan_result.response.content or ""
                else:
                    plan_text = ""
            else:
                plan_response = await self.provider.generate(plan_request)
                plan_text = plan_response.content or ""
            # Parse plan steps from response
            plan_steps = []
            for line in plan_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering prefixes like "1.", "1)", "- "
                import re
                cleaned = re.sub(r"^[\d]+[.)\s]+[-•*]?\s*", "", line).strip()
                if cleaned:
                    plan_steps.append(cleaned)
            # Validate: TODOs must be actionable engineering tasks (verb-first).
            # Conversational prose is rejected and never displayed.
            plan_steps = self._validate_plan_steps(plan_steps)
            if not plan_steps:
                plan_steps = self._default_plan_steps(goal)
            if plan_steps:
                task.plan = plan_steps
                # Create dynamic task plan (runtime owns status from here on)
                for step in plan_steps:
                    task.task_plan.add(step)
                await self.event_bus.emit(
                    Event(
                        type="plan.created",
                        source="agent_loop",
                        data={
                            "steps": plan_steps,
                            "task_id": task.id,
                            "todos": task.task_plan.to_event_items(),
                        },
                    )
                )
                await self._emit_todo_update(task)
        except Exception:
            pass  # Planning is best-effort; don't fail the task

        # Emit initial thinking
        await self._emit_thinking(f"I understand the task. Starting execution.", task)
        await self._emit_phase("implementing")

        while task.iterations < task.max_iterations:
            task.iterations += 1
            task.status = TaskStatus.EXECUTING
            self.budget.record_iteration()

            await self.event_bus.emit(
                Event(
                    type="iteration.started",
                    source="agent_loop",
                    data={"iteration": task.iterations, "task_id": task.id},
                )
            )

            # Build messages
            messages = self._build_messages(task, context)

            # Check budget
            ok, reason = self.budget.check_all()
            if not ok:
                task.status = TaskStatus.FAILED
                task.error = f"Budget exceeded: {reason}"
                break

            # Call model — via router if available, else direct provider
            request = CompletionRequest(
                messages=messages,
                model=self.config.model_preference,
                tools=self._tool_schemas() if self.tools else None,
            )

            try:
                if self.router is not None:
                    response = None
                    for attempt in range(2):
                        fallback_result = await self.router.execute(request)
                        if fallback_result.succeeded:
                            response = fallback_result.response
                            prev_model = self._last_model_used
                            self._last_model_used = fallback_result.model_used
                            # Account models + switches (quiet, change-only)
                            if fallback_result.model_used and fallback_result.model_used not in task.models_used:
                                task.models_used.append(fallback_result.model_used)
                            if prev_model and fallback_result.model_used and fallback_result.model_used != prev_model:
                                task.model_fallbacks += 1
                                await self.event_bus.emit(
                                    Event(
                                        type="model.switched",
                                        source="agent_loop",
                                        data={
                                            "from": prev_model,
                                            "to": fallback_result.model_used,
                                            "reason": "unavailable",
                                        },
                                    )
                                )
                            break
                        if attempt == 0:
                            # Failed models were marked unhealthy (402/401/429);
                            # a rebuilt chain skips them and reaches viable models.
                            await self.event_bus.emit(
                                Event(
                                    type="model.error",
                                    source="agent_loop",
                                    data={
                                        "error": fallback_result.final_error or "All models failed",
                                        "retrying": True,
                                    },
                                )
                            )
                            continue
                        raise RuntimeError(fallback_result.final_error or "All models failed")
                else:
                    response = await self.provider.generate(request)
            except Exception as e:
                err_str = str(e)
                failure_reason = self._classify_failure_reason(err_str)
                task.failure_reason = failure_reason
                await self.event_bus.emit(
                    Event(
                        type="model.error",
                        source="agent_loop",
                        data={
                            "error": err_str,
                            "reason": failure_reason,
                        },
                    )
                )
                # MODEL FAILURE ≠ TASK FAILURE. If real work already happened,
                # preserve state and pause instead of discarding progress.
                did_work = bool(self._modified_files) or any(
                    tc.result is not None for tc in task.tool_calls
                )
                if did_work:
                    task.status = TaskStatus.PAUSED
                    task.paused_reason = (
                        _PAUSED_MESSAGES.get(failure_reason, failure_reason)
                        or "No usable model is currently available. "
                        "Execution state preserved."
                    )
                    task.error = None
                    await self._reconcile_todos(task)
                    await self.event_bus.emit(
                        Event(
                            type="task.paused",
                            source="agent_loop",
                            data={
                                "task_id": task.id,
                                "reason": failure_reason,
                                "reason": "model_unavailable",
                                "error": str(e),
                                "completed_todos": task.task_plan.completed_count,
                                "total_todos": task.task_plan.total_count,
                                "files_changed": list(self._modified_files),
                            },
                        )
                    )
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Provider error: {e}"
                break

            # Process response
            iter_calls: list[ToolCall] = []
            if response.tool_calls:
                # Execute tool calls
                for tool_call_data in response.tool_calls:
                    func = tool_call_data.get("function", {})
                    call = ToolCall(
                        id=tool_call_data.get("id", ""),
                        tool_name=func.get("name", ""),
                        arguments=json.loads(func.get("arguments", "{}")),
                    )
                    task.tool_calls.append(call)
                    iter_calls.append(call)
                    self.budget.record_tool_call()

                    await self.event_bus.emit(
                        Event(
                            type="tool.call",
                            source="agent_loop",
                            data={"tool": call.tool_name, "args": call.arguments},
                        )
                    )

                    # Runtime-owned TODO state: mark matching item in progress
                    await self._todo_started(task, call)

                    result = await self._execute_tool(call)

                    # Enrich with diagnosis / test accounting / git state
                    await self._postprocess_result(task, call, result)
                    # Evidence-based TODO transitions (never from model prose)
                    await self._todo_result(task, call, result)

                    # Track execution stats
                    task.execution_stats.record_attempt()
                    if result.status == ToolResultStatus.SUCCESS:
                        task.execution_stats.record_success(call.tool_name)
                    elif result.status == ToolResultStatus.PERMISSION_DENIED:
                        task.execution_stats.record_permission_denied(call.tool_name)
                    else:
                        task.execution_stats.record_failure(call.tool_name)

                    event_data: dict[str, Any] = {
                        "tool": call.tool_name,
                        "status": result.status.value,
                        "output_len": len(result.output),
                    }
                    if result.exit_code is not None:
                        event_data["exit_code"] = result.exit_code
                    if result.error:
                        event_data["error"] = result.error
                    if result.stderr:
                        event_data["stderr"] = result.stderr

                    await self.event_bus.emit(
                        Event(
                            type="tool.result",
                            source="agent_loop",
                            data=event_data,
                        )
                    )

                    # Check if we've hit tool call limit
                    if len(task.tool_calls) >= self.config.max_tool_calls:
                        task.status = TaskStatus.FAILED
                        task.error = "Tool call limit reached"
                        break
            else:
                # No tool calls — model wants to finish.
                # GUARD: a workspace task must not complete with zero tool calls.
                # Models that claim "I don't have visibility" are wrong — the
                # workspace is real and accessible through tools.
                if (
                    not task.tool_calls
                    and self.tools
                    and self._workspace_has_files()
                    and self._no_tool_nudges < MAX_NO_TOOL_NUDGES
                ):
                    self._no_tool_nudges += 1
                    files_preview = ", ".join(
                        (self._project_info or {}).get("files", [])[:10]
                    )
                    # Rotate away from the model that ignored its tools, so the
                    # next iteration tries a different (hopefully tool-capable) model.
                    if self.router is not None and self._last_model_used:
                        self.router.health.record_no_tool_usage(self._last_model_used)
                    self._corrections.append(
                        "You responded without using any tools. That is not acceptable here: "
                        f"the workspace at {self._project_info.get('root')} is real and "
                        f"contains these files: {files_preview}. "
                        "Use your tools (read_file, list_files, grep, run_command, ...) to "
                        "inspect the project and do the work. Do NOT claim you lack "
                        "visibility — proceed with the task using tools now."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="execution.nudge",
                            source="agent_loop",
                            data={"reason": "zero_tool_calls", "nudge": self._no_tool_nudges},
                        )
                    )
                    continue

                # When the model finishes with text only (no tool calls in this
                # iteration) and the nudge budget is exhausted, skip all pending
                # TODOs — the model chose to complete without further tool use.
                if not iter_calls:
                    for item in task.task_plan.items:
                        if item.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS):
                            task.task_plan.skip_id(
                                item.id,
                                "Model completed with text response; no tool evidence",
                            )

                # HARD INVARIANT: TOOL FAILURE ≠ TASK SUCCESS
                # The runtime execution results are the source of truth.
                # A model text response claiming success does NOT override
                # actual tool failures.
                if self._should_block_completion(task):
                    task.status = TaskStatus.FAILED
                    failure_summary = self._get_failure_summary(task)
                    task.error = (
                        "Task cannot be marked complete: required tool operations failed."
                        f"{failure_summary}\n\n"
                        "Diagnose the failures and fix the implementation. "
                        "Do not claim success when commands fail."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={
                                "task_id": task.id,
                                "reason": "tool_failures_not_recovered",
                                "failed_tools": len([
                                    tc for tc in task.tool_calls
                                    if tc.result and tc.result.execution_failed
                                ]),
                            },
                        )
                    )
                    break

                # Verification phase — truthful completion requires evidence.
                task.result = response.content
                await self._verify_task_completion(task)
                if task.status == TaskStatus.FAILED:
                    await self._reconcile_todos(task)
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={
                                "task_id": task.id,
                                "reason": "verification_failed",
                            },
                        )
                    )
                    break

                # Reconcile remaining TODOs against real execution evidence
                await self._reconcile_todos(task)

                # COMPLETION INVARIANT
                # No code path sets COMPLETED without passing through
                # can_complete_task. 4/5 is not completion; the runtime
                # is the source of truth, not the model's text.
                if not can_complete_task(task):
                    blockers = completion_blockers(task)
                    task.status = TaskStatus.FAILED
                    task.error = (
                        "Task cannot be marked complete: " + "; ".join(blockers)
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={
                                "task_id": task.id,
                                "reason": "completion_invariant_violated",
                                "blockers": blockers,
                                "completed_todos": task.task_plan.completed_count,
                                "total_todos": task.task_plan.total_count,
                            },
                        )
                    )
                    break

                await self._emit_phase("complete")
                task.status = TaskStatus.COMPLETED
                break

            # Progress & stagnation detection (governor)
            progress = self._iteration_made_progress(iter_calls)
            for tc in iter_calls:
                self._seen_actions.add(self._action_key(tc.tool_name, tc.arguments))
            if progress:
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1
                if self._stagnation_counter == STAGNATION_WARN_AT:
                    await self.event_bus.emit(
                        Event(
                            type="progress.stalled",
                            source="agent_loop",
                            data={"stagnant_iterations": self._stagnation_counter},
                        )
                    )
                    self._corrections.append(
                        "No meaningful progress detected in the last iterations "
                        "(same actions repeated, no new inspection or code changes). "
                        "Stop repeating the same operations. Either diagnose the "
                        "blocker by reading the relevant files and errors, or take a "
                        "different concrete action."
                    )
                elif self._stagnation_counter >= STAGNATION_STOP_AT:
                    # If the evidence is already green, finish truthfully instead
                    # of failing a completed task.
                    if await self._stagnation_recovery(task):
                        break
                    task.status = TaskStatus.FAILED
                    task.error = (
                        "Stopped: no meaningful progress detected for "
                        f"{self._stagnation_counter} consecutive iterations. "
                        "The same operations were repeated without advancing the task. "
                        "Partial work (if any) is preserved in the workspace."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={"task_id": task.id, "reason": "stagnation"},
                        )
                    )
                    break

            # Diagnosis mode budget: never loop forever in diagnosis
            if self._diagnosis_active:
                self._diagnosis_iterations += 1
                if self._diagnosis_iterations > MAX_DIAGNOSIS_ITERATIONS:
                    task.status = TaskStatus.FAILED
                    task.error = (
                        "Unable to resolve the repeated failure safely after "
                        f"{MAX_DIAGNOSIS_ITERATIONS} diagnostic iterations. "
                        "Stopping automatic repair rather than repeatedly "
                        "modifying files. Review the failing command output manually."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={"task_id": task.id, "reason": "diagnosis_exhausted"},
                        )
                    )
                    break

            # Safety valve: check AFTER processing tool calls (works for both
            # tool-call and text-response iterations)
            if self._consecutive_denials >= 3:
                task.status = TaskStatus.FAILED
                task.error = (
                    "Task blocked: too many consecutive permission denials. "
                    "The current permission policy prevents required operations. "
                    "Configure permissions or run with interactive approval enabled."
                )
                break

        if task.status not in (
            TaskStatus.COMPLETED, TaskStatus.FAILED,
            TaskStatus.PAUSED, TaskStatus.CANCELLED,
        ):
            if self._consecutive_denials >= 3:
                task.status = TaskStatus.FAILED
                task.error = task.error or "Task blocked by permission policy"
            else:
                task.status = TaskStatus.FAILED
                task.error = task.error or "Max iterations reached"

        # Record performance if task_aware is available
        if self.task_aware is not None and self.router is not None:
            decisions = self.router.get_routing_decisions()
            model_used = decisions[-1].selected_model if decisions else "unknown"
            provider_used = decisions[-1].selected_provider if decisions else "unknown"
            self.task_aware.record_task_result(
                model_id=model_used,
                provider=provider_used,
                task_type=task_type.value if task_type else "unknown",
                success=task.status == TaskStatus.COMPLETED,
                tool_calls=len(task.tool_calls),
                iterations=task.iterations,
            )

        await self.event_bus.emit(
            Event(
                type="task.completed",
                source="agent_loop",
                data={
                    "task_id": task.id,
                    "status": task.status.value,
                    "failure_reason": task.failure_reason,
                    "iterations": task.iterations,
                    "tool_calls": len(task.tool_calls),
                    "stats": task.execution_stats.summary(),
                    "attempted": task.execution_stats.attempted,
                    "succeeded": task.execution_stats.succeeded,
                    "failed": task.execution_stats.failed,
                    "recovered": task.execution_stats.recovered,
                    "unresolved": task.execution_stats.unresolved,
                    "verification_passed": task.verification_passed,
                    "verification_summary": task.verification_summary,
                    "files_changed": list(self._modified_files),
                    "completed_operations": list(self._completed_operations),
                    "todos": task.task_plan.to_event_items(),
                    "todos_completed": task.task_plan.completed_count,
                    "todos_failed": task.task_plan.failed_count,
                    "todos_total": task.task_plan.total_count,
                    "tests_run": task.tests_run,
                    "tests_passed": task.tests_passed,
                    "models_used": list(task.models_used),
                    "model_fallbacks": task.model_fallbacks,
                    "git_commit": task.git_commit,
                    "git_push": task.git_push,
                    "paused_reason": task.paused_reason,
                },
            )
        )

        return task
