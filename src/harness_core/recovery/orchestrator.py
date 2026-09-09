"""
Recovery orchestrator to manage autonomous debugging loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from harness_core.agents.domain import AgentRole, SubTask, TaskGraph, TaskStatus
from harness_core.agents.registry import AgentRegistry
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import ModelProvider
from harness_core.recovery.classifier import FailureClassification
from harness_core.recovery.planner import RecoveryPlan, RecoveryPlanner
from harness_core.recovery.validator import RecoveryValidator

if TYPE_CHECKING:
    from harness_core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class RecoveryHistory:
    """Immutable history of recovery attempts per task."""
    task_id: str
    attempts: int = 0
    max_attempts: int = 3
    strategies_used: list[str] = field(default_factory=list)


class RecoveryOrchestrator:
    """Orchestrates autonomous recovery."""

    def __init__(
        self,
        event_bus: EventBus,
        registry: AgentRegistry,
        provider: ModelProvider,
        max_attempts: int = 3,
        memory: "MemoryManager | None" = None,
        project_id: str = "",
    ) -> None:
        self.event_bus = event_bus
        self.registry = registry
        self.provider = provider
        self.max_attempts = max_attempts
        self.planner = RecoveryPlanner(provider, registry)
        self.validator = RecoveryValidator(registry)
        self.project_id = project_id
        self.history: dict[str, RecoveryHistory] = {}
        # Phase 8E: Memory subsystem
        self._memory = memory

    async def handle_failure(
        self, graph: TaskGraph, failed_task: SubTask, classification: FailureClassification
    ) -> bool:
        """
        Attempt to orchestrate a recovery for a failed task.
        Returns True if a recovery sequence was successfully added to the graph.
        Returns False if recovery was exhausted or failed validation.
        """
        # Extract base task ID so we track attempts against the original failure
        base_task_id = failed_task.task_id
        if "_retest_" in base_task_id:
            base_task_id = base_task_id.split("_retest_")[0]
        if "_recovery_" in base_task_id:
            base_task_id = base_task_id.split("_recovery_")[0]
            
        history = self.history.setdefault(
            base_task_id,
            RecoveryHistory(task_id=base_task_id, max_attempts=self.max_attempts)
        )

        await self._emit("recovery_started", {
            "task_id": failed_task.task_id,
            "category": classification.category.value,
            "attempt": history.attempts + 1
        })

        # Phase 8E: Record this failure in memory so future agents can learn from it
        if self._memory and getattr(self._memory, "enabled", False):
            try:
                await self._memory.record_failure(
                    task_id=failed_task.task_id,
                    description=failed_task.description,
                    error=classification.summary,
                    agent_role=failed_task.role.value,
                    project_id=self.project_id,
                    metadata={
                        "classification": classification.category.value,
                        "recovery_attempt": history.attempts + 1,
                    },
                )
            except Exception:
                pass  # Memory write failures must not block recovery

        if history.attempts >= history.max_attempts:
            await self._emit("recovery_exhausted", {
                "task_id": failed_task.task_id,
                "reason": f"Max attempts ({history.max_attempts}) reached."
            })
            return False

        # If stagnation (same strategy used 3 times without progress), we could detect here,
        # but for simplicity, the max_attempts handles bounded execution.
        
        # 1. Plan
        plan = await self.planner.plan(
            failed_task.task_id, classification, context={"previous_attempts": history.attempts}
        )
        if not plan:
            await self._emit("recovery_plan_rejected", {"task_id": failed_task.task_id, "reason": "Planner returned None"})
            return False

        await self._emit("recovery_plan_created", {
            "task_id": failed_task.task_id,
            "strategy": plan.strategy.value,
            "specialist": plan.specialist_role.value
        })

        # 2. Validate
        errors = self.validator.validate(plan)
        if errors:
            await self._emit("recovery_plan_rejected", {
                "task_id": failed_task.task_id,
                "reason": "Validation failed",
                "errors": errors
            })
            return False

        # 3. Create Recovery Sequence
        # Sequence: DEBUGGER (fixes) -> RETEST (original role or TESTER)
        # We always ensure a retest task is generated to prevent self-certification.
        
        debug_task_id = f"{base_task_id}_recovery_{history.attempts + 1}"
        retest_task_id = f"{base_task_id}_retest_{history.attempts + 1}"

        # Debugger Task
        debug_task = SubTask(
            task_id=debug_task_id,
            description=f"RECOVERY ATTEMPT {history.attempts + 1}: {plan.objective}",
            role=plan.specialist_role,
            dependencies=[], # Will be linked by TaskGraph
            priority=failed_task.priority + 1,
            resources=list(failed_task.resources), # Re-use same workspace locks
        )

        # Retest Task (usually same as the failed task)
        retest_task = SubTask(
            task_id=retest_task_id,
            description=f"RETEST after recovery: {failed_task.description}",
            role=failed_task.role,
            dependencies=[],
            priority=failed_task.priority + 1,
            resources=list(failed_task.resources),
        )
        
        # For testing purposes, if original was TESTER, retest is TESTER. 
        # The TaskGraph insert_recovery_sequence handles chaining them.

        recovery_tasks = [debug_task, retest_task]

        try:
            graph.insert_recovery_sequence(failed_task.task_id, recovery_tasks)
            history.attempts += 1
            history.strategies_used.append(plan.strategy.value)
            
            await self._emit("recovery_task_created", {
                "task_id": failed_task.task_id,
                "recovery_tasks": [t.task_id for t in recovery_tasks]
            })
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert recovery sequence: {e}")
            await self._emit("recovery_plan_rejected", {
                "task_id": failed_task.task_id,
                "reason": f"Graph mutation failed: {str(e)}"
            })
            return False

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = Event(type=event_type, source="RecoveryOrchestrator", data=data)
        await self.event_bus.emit(event)
