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

if TYPE_CHECKING:
    from harness_core.routing.task_aware import TaskAwareRouter


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

    def _system_prompt(self) -> str:
        """Build the system prompt."""
        return """You are an autonomous software engineering agent.

Your goal is to complete engineering tasks reliably. You must:
1. Understand the task
2. Plan your approach
3. Execute tools to inspect and modify code
4. Verify your changes work
5. Report results with evidence

You have access to tools. Use them to read, edit, write, search, and run commands.

CRITICAL RULE: NEVER claim a task is complete when a required tool call failed.
The runtime execution results (exit codes, stderr) are the source of truth.
If a command exits with a non-zero exit code, the task is NOT complete.
Your text response cannot override actual tool failures.

If a command fails:
- Read the error output carefully
- Diagnose the root cause
- Fix the code
- Run the command again
- Only claim success when the command passesIMPORTANT: If a tool call returns "permission denied", do NOT retry the same command.
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

    def _build_messages(
        self,
        task: Task,
        context: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for the model."""
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

        # Add previous tool call results
        for tc in task.tool_calls:
            if tc.result:
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
        """Check if this exact command has failed repeatedly without code changes.

        Detects: node test.js → fail → node test.js → fail → node test.js → fail
        without any intervening file edits.
        """
        args_key = json.dumps(call.arguments, sort_keys=True)
        command = call.arguments.get("command", "")
        if not command:
            return False
        # Count how many times the same command failed recently
        same_cmd_failures = sum(
            1 for (tn, ak, _) in self._recent_failures[-10:]
            if tn == call.tool_name and ak == args_key
        )
        return same_cmd_failures >= 3

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

        # Check for repeated failures of the same command
        if self._is_repeating_failure(call):
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=(
                    f"This command has already failed {self._consecutive_failures} times "
                    f"without any changes. Edit the code first, then retry."
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
            return error_result

    async def run(self, goal: str) -> Task:
        """Run the agent loop for a given goal."""
        task = Task(goal=goal, max_iterations=self.config.max_iterations)

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

        # Discover project
        project_info = await self.context_engine.discover_project()

        # Assemble context
        context = await self.context_engine.assemble_context(goal, project_info)

        # Planning phase: ask model to create a concise plan
        await self._emit_phase("planning")
        try:
            plan_messages = [
                {"role": "system", "content": (
                    "You are planning an engineering task. Respond with a numbered list of steps.\n"
                    "Be concise: 3-6 steps maximum. Each step should be one short line.\n"
                    "Do NOT include explanation. ONLY the numbered list.\n"
                    "Example:\n"
                    "1. Inspect current code\n"
                    "2. Implement changes\n"
                    "3. Run tests\n"
                    "4. Verify results"
                )},
                {"role": "user", "content": goal},
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
                    fallback_result = await self.router.execute(request)
                    if not fallback_result.succeeded:
                        raise RuntimeError(fallback_result.final_error or "All models failed")
                    response = fallback_result.response
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
                # No tool calls — model is done
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

                task.result = response.content
                task.status = TaskStatus.COMPLETED
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
                },
            )
        )

        return task
