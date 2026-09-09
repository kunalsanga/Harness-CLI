"""
Multi-agent task scheduler.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from harness_core.agents.domain import (
    AgentContract,
    TaskGraph,
    TaskStatus,
)
from harness_core.agents.locks import WorkspaceLockManager
from harness_core.agents.message_bus import AgentMessageBus
from harness_core.agents.registry import AgentRegistry
from harness_core.agents.worker import WorkerAgent
from harness_core.observability.events import Event, EventBus
from harness_core.recovery.classifier import FailureClassifier
from harness_core.recovery.orchestrator import RecoveryOrchestrator

logger = logging.getLogger(__name__)

class Scheduler:
    """Schedules and executes a TaskGraph concurrently."""

    def __init__(
        self,
        event_bus: EventBus,
        registry: AgentRegistry,
        provider: Any,
        tools: list,
        workspace_path: str,
        max_concurrency: int = 3,
        router: Any = None,
        lock_manager: WorkspaceLockManager | None = None,
        memory: Any = None,
        project_id: str = "",
        message_bus: AgentMessageBus | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.registry = registry
        self.provider = provider
        self.tools = tools
        self.workspace_path = workspace_path
        self.max_concurrency = max_concurrency
        self.router = router
        self.lock_manager = lock_manager or WorkspaceLockManager(workspace_path)
        self._injected_message_bus = message_bus
        self.active_workers: dict[str, asyncio.Task] = {}
        self.workers: dict[str, WorkerAgent] = {}
        self.message_bus: AgentMessageBus | None = None
        self.recovery_orchestrator = RecoveryOrchestrator(
            event_bus=self.event_bus,
            registry=self.registry,
            provider=self.provider,
            max_attempts=3,
            memory=memory,
            project_id=project_id,
        )
        # Phase 8E/8G: Memory subsystem for RAG-execution and memory writes
        self._memory = memory
        self._project_id = project_id
        self._is_running = False

    def set_message_bus(self, message_bus: AgentMessageBus | None) -> None:
        """Provide a pre-built AgentMessageBus for this scheduler run.

        The runtime (Phase 9) constructs one bus per project so it can inspect
        handoffs and publish structured messages; when set, ``execute()`` uses
        it instead of creating a private bus. Legacy callers are unaffected
        (default None -> scheduler builds its own bus as before).
        """
        self._injected_message_bus = message_bus

    def get_message_bus(self) -> AgentMessageBus | None:
        """Return the bus used by the most recent ``execute()`` run."""
        return self.message_bus

    async def execute(self, graph: TaskGraph) -> None:
        """Execute the task graph to completion or failure."""
        self.message_bus = self._injected_message_bus or AgentMessageBus(self.event_bus, graph, self.registry)
        self._is_running = True
        await self._emit("scheduler.started", {"total_tasks": graph.get_total_count()})

        while self._is_running and not graph.is_complete():
            # 1. Update readiness
            ready_tasks = graph.get_ready_tasks()
            for task in ready_tasks:
                await self._emit("task.ready", {"task_id": task.task_id})

            # 2. Check if we can launch new tasks
            running_count = len(self.active_workers)
            available_slots = self.max_concurrency - running_count

            # Find tasks that are READY
            dispatchable = [t for t in graph.tasks.values() if t.status == TaskStatus.READY]

            for task in dispatchable[:available_slots]:
                # Attempt to acquire locks
                if self.lock_manager and task.resources:
                    await self._emit("resource_lock_requested", {"task_id": task.task_id, "resources": [r.to_dict() for r in task.resources]})
                    acquired = await self.lock_manager.acquire(task.task_id, task.resources)
                    if not acquired:
                        # Find out who owns it (for observability)
                        owner = None
                        for r in task.resources:
                            owner = await self.lock_manager.get_owner(r)
                            if owner:
                                break
                        await self._emit("resource_lock_waiting", {"task_id": task.task_id, "owner": owner})
                        continue
                    await self._emit("resource_lock_acquired", {"task_id": task.task_id})

                task.status = TaskStatus.QUEUED
                await self._emit("task.queued", {"task_id": task.task_id})

                # Assign agent
                agent_config = self.registry.get_default_for_role(task.role)
                if not agent_config:
                    graph.update_task_status(task.task_id, TaskStatus.FAILED, f"No agent registered for role {task.role}")
                    await self._emit("task.failed", {"task_id": task.task_id, "error": f"No agent for role {task.role}"})
                    continue

                contract = AgentContract(
                    role=task.role,
                    task_id=task.task_id,
                    objective=task.description,
                    allowed_tools=agent_config.allowed_tools,
                    denied_tools=agent_config.denied_tools,
                    workspace_scope=agent_config.workspace_scope,
                    model_policy=agent_config.model_policy,
                    system_instructions=agent_config.system_instructions,
                    output_requirements=agent_config.output_requirements,
                    timeout_seconds=agent_config.timeout_seconds,
                    budget_cost=agent_config.budget_cost,
                    resources=task.resources,
                    runtime_context=getattr(task, "runtime_context", "") or "",
                )

                worker = WorkerAgent(
                    contract=contract,
                    provider=self.provider,
                    tools=self.tools,
                    event_bus=self.event_bus,
                    workspace_path=self.workspace_path,
                    router=self.router,
                    message_bus=self.message_bus,
                    memory=self._memory,
                    project_id=self._project_id,
                )

                task.assigned_agent = contract.agent_id
                self.workers[contract.agent_id] = worker

                # Start task
                task.status = TaskStatus.RUNNING
                await self._emit(
                    "task.started",
                    {
                        "task_id": task.task_id,
                        "agent_id": contract.agent_id,
                        # Phase 10: role is scheduler-owned truth; the live
                        # dashboard shows it without guessing.
                        "role": task.role.value,
                    },
                )

                worker_task = asyncio.create_task(self._run_worker(worker, task.task_id, graph))
                self.active_workers[task.task_id] = worker_task

            # Sleep briefly to yield control and let tasks run
            await asyncio.sleep(0.1)

            # Prevent infinite deadlock
            if not self.active_workers and not [t for t in graph.tasks.values() if t.status in (TaskStatus.READY, TaskStatus.CREATED)]:
                break

        self._is_running = False
        await self._emit("scheduler.completed", {"failures": graph.get_failed_count()})

    async def cancel_all(self) -> None:
        """Cancel all running tasks."""
        self._is_running = False
        for task_id, worker_task in list(self.active_workers.items()):
            worker_task.cancel()

    def cancel(self, task_id: str) -> None:
        """Cancel a specific task."""
        if task_id in self.active_workers:
            self.active_workers[task_id].cancel()

    async def _run_worker(self, worker: WorkerAgent, task_id: str, graph: TaskGraph) -> None:
        """Run a worker and process its result."""
        try:
            result = await worker.run()
            task = graph.get_task(task_id)

            if not task:
                return

            # Check if there is a failure (either explicitly FAILED, or failed tests)
            classification = FailureClassifier.classify(task, result)

            if classification:
                # We have a failure!
                error_msg = f"Task failed: {classification.summary}"
                graph.update_task_status(task_id, TaskStatus.FAILED, error_msg)
                await self._emit(
                    "task.failed",
                    {
                        "task_id": task_id,
                        "agent_id": worker.contract.agent_id,
                        "error": error_msg,
                        "role": worker.contract.role.value,
                    },
                )

                # Phase 8E: Record failure memory
                if self._memory and getattr(self._memory, "enabled", False):
                    try:
                        await self._memory.record_failure(
                            task_id=task.task_id,
                            description=task.description,
                            error=error_msg,
                            agent_role=task.role.value,
                            project_id=self._project_id,
                            metadata={
                                "classification": classification.category.value,
                                "files": task.files_changed,
                            },
                        )
                    except Exception:
                        pass  # Memory write failures must not break task execution

                # Attempt Recovery
                recovered = await self.recovery_orchestrator.handle_failure(graph, task, classification)
                if recovered:
                    # Recovery sequence inserted.
                    # We leave the original task as FAILED, but its dependents have been updated to wait for the recovery sequence!
                    logger.info(f"Recovery sequence initiated for {task_id}")
            else:
                graph.update_task_status(task_id, TaskStatus.COMPLETED)
                task.result = result.summary
                task.files_changed = result.files_changed
                task.model_id = result.model_id
                await self._emit(
                    "task.completed",
                    {
                        "task_id": task_id,
                        "agent_id": worker.contract.agent_id,
                        "role": worker.contract.role.value,
                    },
                )

                # Phase 8E: Record success memory
                if self._memory and getattr(self._memory, "enabled", False):
                    try:
                        await self._memory.record_success(
                            task_id=task.task_id,
                            description=task.description,
                            outcome=result.summary or "Completed successfully",
                            agent_role=task.role.value,
                            project_id=self._project_id,
                            tags=["completed"],
                            metadata={"files": task.files_changed},
                        )
                    except Exception:
                        pass  # Memory write failures must not break task execution

        except asyncio.CancelledError:
            graph.update_task_status(task_id, TaskStatus.CANCELLED)
            await self._emit("task.cancelled", {"task_id": task_id, "agent_id": worker.contract.agent_id})
        except Exception as e:
            graph.update_task_status(task_id, TaskStatus.FAILED, str(e))
            await self._emit("task.failed", {"task_id": task_id, "agent_id": worker.contract.agent_id, "error": str(e)})
        finally:
            if self.lock_manager:
                await self.lock_manager.release_all(task_id)
                await self._emit("resource_lock_released", {"task_id": task_id})
            if task_id in self.active_workers:
                del self.active_workers[task_id]

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = Event(type=event_type, source="scheduler", data=data)
        await self.event_bus.emit(event)
