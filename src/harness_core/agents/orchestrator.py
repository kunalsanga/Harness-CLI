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
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from harness_core.memory.manager import MemoryManager


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
        memory: "MemoryManager | None" = None,
        project_id: str = "",
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
        # Phase 8: Memory subsystem (optional; None = no memory)
        self._memory = memory
        self._project_id = project_id

    async def decompose_task(self, task_description: str, context: dict[str, Any] | None = None) -> tuple[TaskGraph | None, list[str]]:
        """Decompose a user task into a TaskGraph of subtasks via Autonomous Planner."""
        from harness_core.planning.planner import Planner
        from harness_core.observability.events import Event, EventBus
        from harness_core.agents.locks import WorkspaceResource

        event_bus = self._event_bus or EventBus()
        await event_bus.emit(Event(type="planning_started", data={"task_description": task_description}))

        if not self._provider:
            return None, ["No model provider configured for planning."]

        # Phase 8F: Pass memory to planner for RAG
        planner = Planner(
            provider=self._provider,
            registry=self.registry,
            memory=self._memory,
            project_id=self._project_id,
        )
        result = await planner.plan(task_description, context)
        
        if not result.success or not result.plan:
            await event_bus.emit(Event(type="plan_validation_failed", data={"errors": result.errors}))
            return None, result.errors

        await event_bus.emit(Event(type="planning_completed", data={"summary": result.plan.summary}))

        graph = TaskGraph()
        for p_task in result.plan.tasks:
            try:
                role_enum = AgentRole(p_task.role.lower())
            except ValueError:
                # Should be caught by PlanValidator, but just in case
                role_enum = AgentRole.CODER

            resources = []
            for res_dict in p_task.resources:
                if "path" in res_dict and "mode" in res_dict:
                    resources.append(WorkspaceResource.from_dict(res_dict))

            st = SubTask(
                task_id=p_task.task_id,
                description=p_task.objective,
                role=role_enum,
                dependencies=p_task.dependencies,
                priority=p_task.priority,
                resources=resources,
            )
            graph.add_task(st)

        # Let the TaskGraph run its own final validation (just to ensure parity)
        errors = graph.validate()
        if errors:
            await event_bus.emit(Event(type="plan_validation_failed", data={"errors": errors}))
            return None, errors

        await event_bus.emit(Event(type="task_graph_created", data={"total_tasks": graph.get_total_count()}))
        return graph, []

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

        # Multi-agent mode: autonomously plan and execute
        graph, errors = await self.decompose_task(task_description, context)
        
        if errors or not graph:
            result.errors.extend(errors or ["Unknown planning error."])
            result.summary = f"Task decomposition failed: {'; '.join(errors or [])}"
            result.duration_ms = (time.time() - start_time) * 1000
            return result
            
        result.task_graph = graph

        # Execute task graph using the new Scheduler
        from harness_core.agents.scheduler import Scheduler
        from harness_core.observability.events import EventBus

        # Phase 8E/8G: Pass memory subsystem to scheduler for RAG-execution and memory writes
        scheduler = Scheduler(
            event_bus=self._event_bus or EventBus(),
            registry=self.registry,
            provider=self._provider,
            tools=self._tools,
            workspace_path=workspace_path or self._workspace_path,
            max_concurrency=self.budget.max_parallel_agents,
            router=self._router,
            memory=self._memory,
            project_id=self._project_id,
        )

        await scheduler.execute(graph)

        agent_results = {}
        for agent_id, worker in scheduler.workers.items():
            if worker._result:
                agent_results[agent_id] = worker._result

        # Synthesize final result
        result.agent_results = agent_results
        result.success = graph.is_complete() and not graph.has_failures()
        
        for task in graph.tasks.values():
            result.files_changed.extend(task.files_changed)
            
        result.files_changed = list(set(result.files_changed))

        # Collect test results, reviews, and update budget
        for ar in agent_results.values():
            result.tests_passed += ar.tests_passed
            result.tests_total += ar.tests_total
            if ar.review_verdict:
                result.review_verdict = ar.review_verdict
            self.budget.record_agent(
                iterations=ar.iterations,
                tool_calls=ar.tool_calls,
            )

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
