"""The core agent loop — orchestrates the engineering workflow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
    Task,
    TaskStatus,
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
    ) -> None:
        self.provider = provider
        self.tools = {t.schema.name: t for t in tools}
        self.workspace_root = workspace_root or Path.cwd()
        self.config = config or AgentConfig()
        self.event_bus = event_bus or EventBus()
        self.router = router
        self.budget = BudgetManager() if router is None else router.budget
        self.context_engine = ContextEngine(self.workspace_root)
        self.permission_manager = PermissionManager(self.workspace_root)
        self.verification_engine = VerificationEngine(self.workspace_root)

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

Always verify your work before claiming success. Do not claim success without evidence.

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

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with permission checking."""
        tool = self.tools.get(call.tool_name)
        if not tool:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=f"Unknown tool: {call.tool_name}",
            )

        # Check permission
        permission = self.permission_manager.check_permission(call.tool_name, call.arguments)
        if permission == "deny":
            return ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output="",
                error="Permission denied",
            )
        if permission == "ask" and not self.permission_manager.request_approval(
            call.tool_name, str(call.arguments)
        ):
            return ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output="",
                error="Permission denied by user",
            )

        # Execute
        start = time.time()
        try:
            result = await tool.execute(call.arguments)
            call.duration_ms = (time.time() - start) * 1000
            call.result = result
            return result
        except Exception as e:
            call.duration_ms = (time.time() - start) * 1000
            error_result = ToolResult(
                status=ToolResultStatus.ERROR, output="", error=str(e)
            )
            call.result = error_result
            return error_result

    async def run(self, goal: str) -> Task:
        """Run the agent loop for a given goal."""
        task = Task(goal=goal, max_iterations=self.config.max_iterations)

        await self.event_bus.emit(
            Event(type="task.started", source="agent_loop", data={"goal": goal})
        )

        # Discover project
        project_info = await self.context_engine.discover_project()

        # Assemble context
        context = await self.context_engine.assemble_context(goal, project_info)

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

                    await self.event_bus.emit(
                        Event(
                            type="tool.result",
                            source="agent_loop",
                            data={
                                "tool": call.tool_name,
                                "status": result.status.value,
                                "output_len": len(result.output),
                            },
                        )
                    )

                # Check if we've hit limits
                if len(task.tool_calls) >= self.config.max_tool_calls:
                    task.status = TaskStatus.FAILED
                    task.error = "Tool call limit reached"
                    break
            else:
                # No tool calls — model is done
                task.result = response.content
                task.status = TaskStatus.COMPLETED
                break

        if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task.status = TaskStatus.FAILED
            task.error = "Max iterations reached"

        await self.event_bus.emit(
            Event(
                type="task.completed",
                source="agent_loop",
                data={
                    "task_id": task.id,
                    "status": task.status.value,
                    "iterations": task.iterations,
                    "tool_calls": len(task.tool_calls),
                },
            )
        )

        return task
