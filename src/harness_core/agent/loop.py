"""The core agent loop — orchestrates the engineering workflow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
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
        """Normalized identity for a tool call (used for repetition detection)."""
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"

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

    # ── Plan validation ────────────────────────────────────────────────

    _INVALID_PLAN_MARKERS = (
        "i don't have", "i do not have", "i don't see", "i cannot see",
        "i can't see", "no visibility", "not able to see", "unable to see",
        "please provide", "please share", "please tell", "please specify",
        "can you provide", "can you share", "can you tell", "could you provide",
        "what technology", "what tech stack", "what framework", "which framework",
        "what language", "which language", "what type of project",
        "more context", "more information", "need more info",
    )

    def _validate_plan_steps(self, steps: list[str]) -> list[str]:
        """Keep only executable action steps; drop conversational prose.

        Rejects questions, requests for information, and claims of
        missing visibility — the workspace is discoverable via tools.
        """
        valid: list[str] = []
        for step in steps:
            s = step.strip().strip("-•*").strip().strip('"\'')
            if not s or len(s) > 140:
                continue
            low = s.lower()
            if "?" in s:
                continue
            if any(marker in low for marker in self._INVALID_PLAN_MARKERS):
                continue
            valid.append(s)
            if len(valid) >= 8:
                break
        return valid

    def _default_plan_steps(self) -> list[str]:
        """Structured fallback plan when the model proposes nothing usable."""
        steps = [
            "Inspect project files",
            "Analyze current implementation",
            "Implement the requested changes",
        ]
        if self._project_info and self._project_info.get("has_tests"):
            steps.append("Run tests and fix any failures")
        steps.append("Verify results")
        return steps

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
        """Emit current TODO list state."""
        items = task.task_plan.display()
        completed = task.task_plan.completed_count
        total = task.task_plan.total_count
        await self.event_bus.emit(
            Event(
                type="todo.updated",
                source="agent_loop",
                data={
                    "items": items,
                    "completed": completed,
                    "total": total,
                },
            )
        )

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

    async def _execute_tool_checked(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with permission checking."""
        tool = self.tools.get(call.tool_name)
        if not tool:
            self._reset_denial_tracking()
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Unknown tool: {call.tool_name}",
                retryable=False,
            )

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

        # Execute
        start = time.time()
        try:
            result = await tool.execute(call.arguments)
            call.duration_ms = (time.time() - start) * 1000
            call.result = result
            self._reset_denial_tracking()
            if result.execution_failed:
                self._record_failure(
                    call.tool_name, call.arguments, result.exit_code or -1
                )
            else:
                self._reset_failure_tracking()
            await self._record_execution_outcome(call, result)
            return result
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
        self.budget.reset()

        await self.event_bus.emit(
            Event(type="task.started", source="agent_loop", data={"goal": goal})
        )
        await self._emit_phase("understanding")

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
            # Validate: TODOs must be executable actions, not conversational prose
            plan_steps = self._validate_plan_steps(plan_steps)
            if not plan_steps:
                plan_steps = self._default_plan_steps()
            if plan_steps:
                task.plan = plan_steps
                # Create dynamic task plan
                for step in plan_steps:
                    task.task_plan.add(step)
                await self.event_bus.emit(
                    Event(
                        type="plan.created",
                        source="agent_loop",
                        data={"steps": plan_steps, "task_id": task.id},
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
                            self._last_model_used = fallback_result.model_used
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
                task.status = TaskStatus.RECOVERING
                await self.event_bus.emit(
                    Event(
                        type="model.error",
                        source="agent_loop",
                        data={"error": str(e)},
                    )
                )
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

                    result = await self._execute_tool(call)

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

        if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
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
                },
            )
        )

        return task
