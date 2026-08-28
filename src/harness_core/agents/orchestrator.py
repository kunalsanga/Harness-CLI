"""
Orchestrator for M5 — coordinates multi-agent execution.

Responsibilities:
  - Understand overall objective
  - Create task graph via TaskPlanner
  - Delegate to specialized agents
  - Monitor progress
  - Detect failures and trigger repair
  - Invoke review and verification
  - Produce final synthesized result
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .domain import (
    AgentMessage,
    AgentResult,
    AgentRole,
    AgentStatus,
    MessageType,
    ReviewVerdict,
    SubTask,
    TaskGraph,
    TaskStatus,
)
from .executor import AgentExecutor
from .registry import AgentConfig, AgentRegistry


class ExecutionMode(enum.Enum):
    """How the orchestrator executes tasks."""
    SINGLE = "single"          # One agent at a time (legacy mode)
    AUTO = "auto"              # Orchestrator decides
    MULTI_AGENT = "multi_agent"  # Always use specialized agents
    PARALLEL = "parallel"      # Maximize parallelism


@dataclass
class AgentBudget:
    """Resource limits for a multi-agent execution."""

    max_agents: int = 8
    max_parallel_agents: int = 3
    max_iterations_per_agent: int = 30
    max_total_iterations: int = 100
    max_tool_calls: int = 500
    max_repair_cycles: int = 3
    max_runtime_seconds: float = 300.0
    max_cost: float = 1.0

    # Tracking (mutable)
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_agents_used: int = 0
    repair_cycles: int = 0
    start_time: float = field(default_factory=time.time)

    def can_start_agent(self) -> bool:
        """Check if we can start another agent."""
        return (
            self.total_agents_used < self.max_agents
            and self.total_iterations < self.max_total_iterations
            and self.total_tool_calls < self.max_tool_calls
            and self.repair_cycles < self.max_repair_cycles
            and (time.time() - self.start_time) < self.max_runtime_seconds
        )

    def record_agent(self, iterations: int = 0, tool_calls: int = 0) -> None:
        """Record agent execution costs."""
        self.total_agents_used += 1
        self.total_iterations += iterations
        self.total_tool_calls += tool_calls

    def record_repair(self) -> bool:
        """Record a repair cycle. Returns False if budget exceeded."""
        self.repair_cycles += 1
        return self.repair_cycles <= self.max_repair_cycles

    def to_dict(self) -> dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "agents_used": self.total_agents_used,
            "max_agents": self.max_agents,
            "iterations": self.total_iterations,
            "max_iterations": self.max_total_iterations,
            "tool_calls": self.total_tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "repair_cycles": self.repair_cycles,
            "max_repair_cycles": self.max_repair_cycles,
            "elapsed_seconds": round(elapsed, 1),
            "max_runtime": self.max_runtime_seconds,
        }


@dataclass
class OrchestratorResult:
    """Final result from orchestrator execution."""

    success: bool = False
    summary: str = ""
    task_graph: TaskGraph | None = None
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    files_changed: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    review_verdict: ReviewVerdict | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    budget: AgentBudget | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "review_verdict": self.review_verdict.value if self.review_verdict else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "agents_used": len(self.agent_results),
            "duration_ms": round(self.duration_ms, 1),
        }


class Orchestrator:
    """Multi-agent orchestrator.

    Coordinates specialized agents to accomplish complex tasks.
    Uses TaskPlanner for decomposition, AgentExecutor for execution,
    and produces a synthesized final result.
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        executor: AgentExecutor | None = None,
        budget: AgentBudget | None = None,
        provider=None,
        tools: list | None = None,
        router=None,
        task_aware=None,
        event_bus=None,
        workspace_path: str = "",
    ) -> None:
        self.registry = registry or AgentRegistry()
        self.budget = budget or AgentBudget()
        self._messages: list[AgentMessage] = []

        # Create executor with real infrastructure
        self.executor = executor or AgentExecutor(
            provider=provider,
            tools=tools or [],
            workspace_path=workspace_path,
            router=router,
            task_aware=task_aware,
            event_bus=event_bus,
        )

        # Keep references for sub-executors
        self._provider = provider
        self._tools = tools or []
        self._router = router
        self._task_aware = task_aware
        self._event_bus = event_bus
        self._workspace_path = workspace_path

    def decompose_task(self, task_description: str) -> TaskGraph:
        """Decompose a user task into a TaskGraph of subtasks.

        Uses role-based heuristics to create appropriate subtasks.
        """
        graph = TaskGraph()

        # Phase 1: Research
        research = SubTask(
            description=f"Research codebase for: {task_description}",
            role=AgentRole.RESEARCHER,
            priority=10,
        )
        graph.add_task(research)

        # Phase 2: Plan
        plan = SubTask(
            description=f"Plan implementation for: {task_description}",
            role=AgentRole.PLANNER,
            dependencies=[research.task_id],
            priority=9,
        )
        graph.add_task(plan)

        # Phase 3: Implement
        implement = SubTask(
            description=f"Implement: {task_description}",
            role=AgentRole.CODER,
            dependencies=[plan.task_id],
            priority=8,
        )
        graph.add_task(implement)

        # Phase 4: Test
        test = SubTask(
            description="Run tests and verify implementation",
            role=AgentRole.TESTER,
            dependencies=[implement.task_id],
            priority=7,
        )
        graph.add_task(test)

        # Phase 5: Review
        review = SubTask(
            description="Review implementation quality",
            role=AgentRole.REVIEWER,
            dependencies=[test.task_id],
            priority=6,
        )
        graph.add_task(review)

        return graph

    async def execute(
        self,
        task_description: str,
        mode: ExecutionMode = ExecutionMode.AUTO,
        workspace_path: str = "",
        context: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        """Execute a task using multi-agent orchestration.

        This is the main entry point for M5 multi-agent execution.
        """
        start_time = time.time()
        self.budget = AgentBudget()

        result = OrchestratorResult(budget=self.budget)

        if mode == ExecutionMode.SINGLE:
            return await self._execute_single(
                task_description, workspace_path or self._workspace_path, context
            )

        # Multi-agent mode: decompose and execute
        graph = self.decompose_task(task_description)
        result.task_graph = graph

        # Validate graph
        errors = graph.validate()
        if errors:
            result.errors.extend(errors)
            result.summary = f"Task decomposition failed: {'; '.join(errors)}"
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        # Execute task graph
        agent_results: dict[str, AgentResult] = {}
        max_rounds = 20  # Safety limit

        for round_num in range(max_rounds):
            if graph.is_complete():
                break

            if not self.budget.can_start_agent():
                result.warnings.append("Budget exceeded — stopping execution")
                break

            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                if not graph.has_failures():
                    result.warnings.append("No ready tasks and none failed — possible deadlock")
                break

            # Execute ready tasks
            for task in ready_tasks:
                task.status = TaskStatus.RUNNING

                agent_config = self.registry.get_default_for_role(task.role)
                if agent_config is None:
                    task.status = TaskStatus.FAILED
                    task.error = f"No agent registered for role {task.role.value}"
                    result.errors.append(task.error)
                    continue

                agent_result = await self.executor.execute(
                    config=agent_config,
                    task=task,
                    context=context,
                    workspace_path=workspace_path or self._workspace_path,
                    previous_results=agent_results,
                )

                agent_results[agent_config.name] = agent_result
                self.budget.record_agent(
                    iterations=agent_result.iterations,
                    tool_calls=agent_result.tool_calls,
                )

                # Record message
                self._messages.append(AgentMessage(
                    sender=agent_config.name,
                    task_id=task.task_id,
                    message_type=MessageType.RESULT,
                    content=agent_result.summary,
                ))

                if agent_result.status == AgentStatus.FAILED:
                    task.status = TaskStatus.FAILED
                    task.error = "; ".join(agent_result.errors)
                    result.errors.append(f"Agent {agent_config.name} failed: {task.error}")

                    # Trigger debugger for failures
                    if self.budget.record_repair():
                        debug_task = SubTask(
                            description=f"Debug failure in: {task.description}",
                            role=AgentRole.DEBUGGER,
                            dependencies=[],
                        )
                        graph.add_task(debug_task)
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = agent_result.summary
                    task.files_changed = agent_result.files_changed
                    result.files_changed.extend(agent_result.files_changed)

        # Synthesize final result
        result.agent_results = agent_results
        result.success = graph.is_complete() and not graph.has_failures()
        result.files_changed = list(set(result.files_changed))

        # Collect test results
        for ar in agent_results.values():
            result.tests_passed += ar.tests_passed
            result.tests_total += ar.tests_total

        # Collect review verdict
        for ar in agent_results.values():
            if ar.review_verdict:
                result.review_verdict = ar.review_verdict

        # Build summary
        completed = graph.get_completed_count()
        total = graph.get_total_count()
        result.summary = (
            f"Executed {completed}/{total} tasks using {len(agent_results)} agents. "
            f"Files changed: {len(result.files_changed)}. "
            f"{'All tasks completed successfully.' if result.success else 'Some tasks failed.'}"
        )

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    async def _execute_single(
        self,
        task_description: str,
        workspace_path: str,
        context: dict[str, Any] | None,
    ) -> OrchestratorResult:
        """Execute in single-agent mode (backward compatible)."""
        start_time = time.time()
        result = OrchestratorResult(budget=self.budget)

        coder_config = self.registry.get_default_for_role(AgentRole.CODER)
        if coder_config is None:
            result.errors.append("No coder agent available")
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        task = SubTask(
            description=task_description,
            role=AgentRole.CODER,
        )

        agent_result = await self.executor.execute(
            config=coder_config,
            task=task,
            context=context,
            workspace_path=workspace_path,
        )

        result.agent_results = {coder_config.name: agent_result}
        result.success = agent_result.status == AgentStatus.COMPLETED
        result.files_changed = agent_result.files_changed
        result.summary = agent_result.summary
        result.duration_ms = (time.time() - start_time) * 1000

        return result

    def get_messages(self) -> list[AgentMessage]:
        """Get all inter-agent messages."""
        return list(self._messages)

    def get_registry(self) -> AgentRegistry:
        """Get the agent registry."""
        return self.registry
