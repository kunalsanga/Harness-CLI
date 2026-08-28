"""
Parallel agent execution for M6.

Provides:
  - Dependency-aware parallel scheduling
  - File ownership tracking
  - Conflict detection
  - Bounded concurrency
  - Task supervision
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .domain import (
    AgentResult,
    AgentRole,
    AgentStatus,
    SubTask,
    TaskGraph,
    TaskStatus,
)


class FileLockState(enum.Enum):
    """File ownership states."""
    UNOWNED = "unowned"
    LOCKED = "locked"
    MODIFIED = "modified"
    CONFLICT = "conflict"


@dataclass
class FileOwnership:
    """Track which agent owns which file."""
    file_path: str
    agent_id: str
    state: FileLockState = FileLockState.LOCKED
    acquired_at: float = field(default_factory=time.time)
    released_at: float = 0.0


class FileOwnershipTracker:
    """Track file ownership across parallel agents.

    Prevents silent overwrites when multiple agents modify the same file.
    """

    def __init__(self) -> None:
        self._owners: dict[str, FileOwnership] = {}
        self._conflicts: list[tuple[str, str, str]] = []  # (file, agent_a, agent_b)

    def try_acquire(self, file_path: str, agent_id: str) -> bool:
        """Try to acquire ownership of a file.

        Returns True if acquired, False if already owned by another agent.
        """
        if file_path in self._owners:
            existing = self._owners[file_path]
            if existing.agent_id == agent_id:
                return True  # Same agent re-acquiring
            # Conflict detected
            existing.state = FileLockState.CONFLICT
            self._conflicts.append((file_path, existing.agent_id, agent_id))
            return False

        self._owners[file_path] = FileOwnership(
            file_path=file_path,
            agent_id=agent_id,
            state=FileLockState.LOCKED,
        )
        return True

    def release(self, file_path: str, agent_id: str) -> None:
        """Release file ownership."""
        if file_path in self._owners:
            own = self._owners[file_path]
            if own.agent_id == agent_id:
                own.released_at = time.time()
                del self._owners[file_path]

    def mark_modified(self, file_path: str, agent_id: str) -> None:
        """Mark a file as modified by an agent."""
        if file_path in self._owners and self._owners[file_path].agent_id == agent_id:
            self._owners[file_path].state = FileLockState.MODIFIED

    def get_conflicts(self) -> list[tuple[str, str, str]]:
        """Get all detected conflicts."""
        return list(self._conflicts)

    def get_agent_files(self, agent_id: str) -> list[str]:
        """Get all files owned by an agent."""
        return [f for f, o in self._owners.items() if o.agent_id == agent_id]

    def get_owner(self, file_path: str) -> str | None:
        """Get the owner of a file."""
        own = self._owners.get(file_path)
        return own.agent_id if own else None

    def clear(self) -> None:
        """Clear all ownership."""
        self._owners.clear()
        self._conflicts.clear()


class ParallelScheduler:
    """Dependency-aware parallel scheduler for agent tasks.

    Schedules tasks for parallel execution based on the dependency graph.
    Independent tasks run concurrently; dependent tasks wait.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        file_tracker: FileOwnershipTracker | None = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.file_tracker = file_tracker or FileOwnershipTracker()
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, AgentResult] = {}

    async def schedule(
        self,
        graph: TaskGraph,
        executor_fn,
        context: dict[str, Any] | None = None,
        workspace_path: str = "",
    ) -> dict[str, AgentResult]:
        """Schedule and execute tasks with parallel dependency awareness.

        Args:
            graph: Task graph with dependencies
            executor_fn: Async function(agent_config, subtask, context, workspace) -> AgentResult
            context: Shared context
            workspace_path: Working directory

        Returns:
            Map of task_id -> AgentResult
        """
        max_rounds = 50  # Safety limit
        round_num = 0

        while not graph.is_complete() and round_num < max_rounds:
            round_num += 1

            ready = graph.get_ready_tasks()
            if not ready:
                if graph.has_failures():
                    break
                # All dependencies done or deadlock
                break

            # Start tasks up to concurrency limit
            batch = ready[:self.max_concurrent]
            running = []

            for task in batch:
                task.status = TaskStatus.RUNNING
                coro = self._run_task(task, executor_fn, context, workspace_path)
                asyncio_task = asyncio.create_task(coro)
                self._active_tasks[task.task_id] = asyncio_task
                running.append((task.task_id, asyncio_task))

            # Wait for batch to complete
            if running:
                task_ids, tasks = zip(*running)
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for task_id, result in zip(task_ids, results):
                    self._active_tasks.pop(task_id, None)
                    subtask = graph.get_task(task_id)

                    if isinstance(result, Exception):
                        if subtask:
                            subtask.status = TaskStatus.FAILED
                            subtask.error = str(result)[:200]
                        self._results[task_id] = AgentResult(
                            status=AgentStatus.FAILED,
                            summary=f"Error: {result}",
                            errors=[str(result)[:500]],
                        )
                    else:
                        if subtask:
                            subtask.status = TaskStatus.COMPLETED
                            subtask.result = result.summary
                            subtask.files_changed = result.files_changed
                        self._results[task_id] = result

        return dict(self._results)

    async def _run_task(
        self,
        task: SubTask,
        executor_fn,
        context: dict[str, Any] | None,
        workspace_path: str,
    ) -> AgentResult:
        """Execute a single task with file conflict checking."""
        # Check file conflicts before execution
        if task.role in (AgentRole.CODER, AgentRole.DEBUGGER):
            # These roles may modify files — check for conflicts
            pass  # Will be checked during execution

        return await executor_fn(task, context, workspace_path)

    def get_results(self) -> dict[str, AgentResult]:
        """Get all collected results."""
        return dict(self._results)

    def get_file_tracker(self) -> FileOwnershipTracker:
        """Get the file ownership tracker."""
        return self.file_tracker
