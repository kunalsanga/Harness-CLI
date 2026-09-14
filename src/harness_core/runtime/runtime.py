"""EngineeringRuntime — the thin orchestration seam of the unified runtime.

Phase 9 principle: *model proposes, runtime decides.*

This class owns the lifecycle of one project execution and coordinates the
existing authorities — it does **not** re-implement any of them:

    TaskGraph            decides task readiness & task state
    Scheduler            decides execution (existing scheduler)
    AgentRegistry        decides role capabilities
    WorkspaceLockManager decides resource access
    RecoveryOrchestrator decides recovery mutation (owned by the scheduler)
    MemoryManager        provides persistent context (optional, non-fatal)
    EventBus             is the only observability mechanism
    Verifier (runtime)   decides requirement-level completion

The runtime adds only what nothing else owns: the project lifecycle stage,
structured requirements, requirement→task→evidence traceability, structured
artifact references, and interruption-safe cleanup.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness_core.agents.domain import (
    AgentMessage,
    AgentRole,
    MessageType,
    SubTask,
    TaskGraph,
    TaskStatus,
    WorkspaceResource,
)
from harness_core.agents.locks import WorkspaceLockManager
from harness_core.agents.message_bus import AgentMessageBus
from harness_core.agents.registry import AgentRegistry
from harness_core.agents.scheduler import Scheduler
from harness_core.observability.events import Event, EventBus
from harness_core.planning.domain import Plan, PlanningResult
from harness_core.planning.planner import Planner
from harness_core.runtime.context import assemble_role_context
from harness_core.runtime.requirements import (
    Evidence,
    EvidenceStatus,
    Requirements,
    TraceabilityIndex,
)
from harness_core.runtime.state import (
    ActiveAgent,
    ArtifactRef,
    ProjectState,
    RuntimeStage,
    RuntimeStatus,
    VerificationStatus,
)

if TYPE_CHECKING:
    from harness_core.memory.manager import MemoryManager

RECOVERY_MARKER = "_recovery_"
RETEST_MARKER = "_retest_"


def _is_recovery_task(task_id: str) -> bool:
    return RECOVERY_MARKER in task_id or RETEST_MARKER in task_id


@dataclass
class RuntimeOutcome:
    """Result of a project execution."""

    state: ProjectState
    status: RuntimeStatus
    traceability: TraceabilityIndex | None = None
    message_bus: AgentMessageBus | None = None
    graph: TaskGraph | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "state": self.state.to_dict(),
            "traceability": self.traceability.to_dict() if self.traceability else None,
            "duration_ms": round(self.duration_ms, 1),
        }


class EngineeringRuntime:
    """Owns the project execution lifecycle from request to terminal status."""

    def __init__(
        self,
        workspace_path: str | Path,
        provider: Any,
        *,
        project_id: str = "",
        registry: AgentRegistry | None = None,
        tools: list | None = None,
        event_bus: EventBus | None = None,
        memory: MemoryManager | None = None,
        max_concurrency: int = 3,
        router: Any = None,
        plan_provider: Any = None,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).resolve())
        self.project_id = project_id or self.workspace_path
        self.registry = registry or AgentRegistry()
        self.tools = tools or []
        self.event_bus = event_bus or EventBus()
        self.memory = memory
        self.router = router
        self.max_concurrency = max_concurrency
        # The provider used for *execution* is threaded to the scheduler;
        # planning may use its own provider boundary so callers can stub the
        # model without stubbing the (real) scheduler + worker execution.
        self._provider = provider
        self._plan_provider = plan_provider or provider

        self.lock_manager = WorkspaceLockManager(self.workspace_path)
        self.scheduler: Scheduler | None = None
        self.state = ProjectState(project_id=self.project_id, workspace=self.workspace_path)
        self._recovery_events: list[dict[str, Any]] = []
        self._started_at = time.time()
        self._interactive_config: Any = None  # AgentConfig passthrough for interactive mode

    # ── Observability ─────────────────────────────────────────────────────

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        await self.event_bus.emit(Event(type=event_type, source="EngineeringRuntime", data=data))

    async def _enter_stage(self, stage: RuntimeStage, detail: str = "") -> None:
        self.state.set_stage(stage, detail=detail)
        await self._emit("runtime_stage_changed", {"stage": stage.value, "detail": detail})

    # ── Public entry point ────────────────────────────────────────────────

    async def run(
        self,
        original_request: str,
        requirements: Requirements | None = None,
        *,
        plan: Plan | None = None,
        timeout_seconds: float = 0.0,
    ) -> RuntimeOutcome:
        """Execute one project from request to terminal status."""
        try:
            return await self._run_inner(
                original_request, requirements, plan=plan, timeout_seconds=timeout_seconds
            )
        except asyncio.CancelledError:
            # Interruption safety: mark non-terminal tasks CANCELLED, release
            # every lock and emit the failure event.  This runs in a detached
            # task because the current task is already cancelled — any further
            # ``await`` here would raise CancelledError again and abort cleanup.
            asyncio.get_running_loop().create_task(
                self._mark_interrupted("Execution cancelled")
            )
            raise
        except Exception as exc:  # unexpected failure: degrade to FAILED
            self.state.status = RuntimeStatus.FAILED
            self.state.set_stage(RuntimeStage.FAILED, detail=str(exc))
            self.state.register_blocker(f"Runtime error: {exc}")
            await self._emit("runtime_failed", {"reason": str(exc)})
            return RuntimeOutcome(
                state=self.state, status=RuntimeStatus.FAILED, traceability=self.state.traceability
            )

    async def _run_inner(
        self,
        original_request: str,
        requirements: Requirements | None,
        *,
        plan: Plan | None,
        timeout_seconds: float,
    ) -> RuntimeOutcome:
        started = time.time()
        self.state.original_request = original_request
        await self._emit("runtime_started", {"project_id": self.project_id})

        await self._enter_stage(RuntimeStage.DISCOVER, detail="project context loaded")

        if requirements is None:
            requirements = Requirements(objective=original_request)
        if not requirements.objective:
            requirements.objective = original_request
        self.state.requirements = requirements
        await self._enter_stage(RuntimeStage.UNDERSTAND, detail="requirements structured")
        req_count = len(requirements.to_requirements())
        await self._emit(
            "requirements_created",
            {"objective": requirements.objective, "requirement_count": req_count},
        )

        # PLAN (model proposes; runtime converts below) and traceability index.
        await self._enter_stage(RuntimeStage.PLAN, detail="retrieval-augmented planning")
        graph, planned_tasks = await self._build_task_graph(original_request, requirements, plan)
        if graph is None:
            self.state.status = RuntimeStatus.FAILED
            self.state.register_blocker("Planning produced no valid tasks.")
            self.state.set_stage(RuntimeStage.FAILED, detail="invalid or empty plan")
            await self._emit("runtime_failed", {"reason": "invalid or empty plan"})
            return RuntimeOutcome(state=self.state, status=RuntimeStatus.FAILED)

        # Traceability: requirements -> tasks. The mapping is runtime-recorded
        # (planner merely proposes task_ids/roles; the runtime validates and
        # records which requirement each task implements).
        index = build_traceability_index(requirements, planned_tasks)
        self.state.traceability = index
        self.state.task_graph = graph
        await self._enter_stage(RuntimeStage.DECOMPOSE, detail=f"{graph.get_total_count()} tasks")
        await self._emit("plan_created", {"total_tasks": graph.get_total_count()})

        # Provide role-scoped context to each task (requirements, traces,
        # artifact references, memory).
        await self._attach_task_contexts(graph, index)

        # EXECUTE through the real Scheduler; recovery handled inside it.
        await self._enter_stage(RuntimeStage.EXECUTE, detail="scheduler dispatch")
        bus = AgentMessageBus(self.event_bus, graph, self.registry)
        scheduler = Scheduler(
            event_bus=self.event_bus,
            registry=self.registry,
            provider=self._provider,
            tools=self.tools,
            workspace_path=self.workspace_path,
            max_concurrency=self.max_concurrency,
            router=self.router,
            lock_manager=self.lock_manager,
            memory=self.memory,
            project_id=self.project_id,
            message_bus=bus,
        )
        self.scheduler = scheduler

        task_handler = self._make_task_event_handler()
        recovery_handler = self._make_recovery_event_handler()
        convergence_handler = self._make_convergence_handler()
        self.event_bus.on("task.started", task_handler)
        self.event_bus.on("task.completed", task_handler)
        self.event_bus.on("task.failed", task_handler)
        self.event_bus.on("recovery_started", recovery_handler)
        self.event_bus.on("recovery_exhausted", recovery_handler)
        self.event_bus.on("tool.result", convergence_handler)

        execution = asyncio.ensure_future(scheduler.execute(graph))
        try:
            if timeout_seconds and timeout_seconds > 0:
                await asyncio.wait_for(execution, timeout=timeout_seconds)
            else:
                await execution
        finally:
            self.event_bus.off("task.started", task_handler)
            self.event_bus.off("task.completed", task_handler)
            self.event_bus.off("task.failed", task_handler)
            self.event_bus.off("recovery_started", recovery_handler)
            self.event_bus.off("recovery_exhausted", recovery_handler)
            self.event_bus.off("tool.result", convergence_handler)
            if not execution.done():
                execution.cancel()

        # Safety: release any locks still held (interrupt path included).
        for task_id in list(getattr(self.lock_manager, "_locks", {})):
            await self.lock_manager.release_all(task_id)

        # Lifecycle stages following execution.
        if self._graph_has_role(graph, AgentRole.TESTER):
            await self._enter_stage(RuntimeStage.INTEGRATE, detail="independent components joined")
            await self._enter_stage(RuntimeStage.TEST, detail="test tasks executed")
        if self._recovery_events:
            await self._enter_stage(
                RuntimeStage.DEBUG, detail=f"{len(self._recovery_events)} recovery events"
            )

        # Structured handoffs along dependency edges + refresh downstream context.
        await self._publish_dependency_handoffs(graph, bus)
        await self._attach_task_contexts(graph, index)

        # Review / verify stages (roles exist in the graph when relevant).
        self._record_artifacts_from_tasks(graph)
        if self._graph_has_role(graph, AgentRole.REVIEWER):
            await self._enter_stage(RuntimeStage.REVIEW, detail="review tasks executed")

        await self._enter_stage(RuntimeStage.VERIFY, detail="requirement verification")
        verification_ok, summary, results = self._verify_requirements(graph, index)
        self.state.verification_status = (
            VerificationStatus.PASSED if verification_ok else VerificationStatus.FAILED
        )
        self.state.verification_summary = summary
        self.state.verification_results = results
        await self._emit(
            "verification_completed",
            {"passed": verification_ok, "summary": summary, "results": results},
        )

        await self._record_memory_references(graph, index)

        # ── Terminal status decision ─────────────────────────────────────
        failed_originals = [
            t.task_id
            for t in graph.tasks.values()
            if t.status == TaskStatus.FAILED and not _is_recovery_task(t.task_id)
        ]
        if self.state.recovery_exhausted:
            status = RuntimeStatus.FAILED
            detail = f"recovery exhausted ({self.state.recovery_attempts} attempts)"
        elif failed_originals and not self._all_failures_recovered(graph, failed_originals):
            status = RuntimeStatus.FAILED
            detail = self._user_facing_error(graph, failed_originals)
        elif not verification_ok:
            status = RuntimeStatus.FAILED
            detail = f"Verification failed: {summary}"
        else:
            status = RuntimeStatus.SUCCESS
            detail = "all requirements verified"
            await self._enter_stage(RuntimeStage.DELIVER, detail=detail)
            await self._enter_stage(RuntimeStage.SUCCEEDED)

        self.state.status = status
        self.state.completed_at = time.time()
        if status == RuntimeStatus.FAILED:
            self.state.set_stage(RuntimeStage.FAILED, detail=detail)
            self.state.register_blocker(detail)
            await self._emit("runtime_failed", {"reason": detail})
        else:
            await self._emit("runtime_completed", {"status": status.value})

        return RuntimeOutcome(
            state=self.state,
            status=status,
            traceability=index,
            message_bus=bus,
            graph=graph,
            duration_ms=(time.time() - started) * 1000,
        )

    # ── Interactive execution (Stage 1: canonical runtime path) ────────────

    async def execute_interactive(
        self,
        goal: str,
        *,
        requirements: Requirements | None = None,
        timeout_seconds: float = 0.0,
    ) -> RuntimeOutcome:
        """Execute a single interactive task through the canonical runtime.

        Skips the model-driven planner: creates a single-task graph directly
        and runs it through the Scheduler → WorkerAgent → AgentLoop chain.
        This is the single authoritative execution path for interactive mode.
        """
        started = time.time()
        self.state.original_request = goal
        await self._emit("runtime_started", {"project_id": self.project_id})

        await self._enter_stage(RuntimeStage.DISCOVER, detail="project context loaded")

        if requirements is None:
            requirements = Requirements(objective=goal)
        if not requirements.objective:
            requirements.objective = goal
        self.state.requirements = requirements
        await self._enter_stage(RuntimeStage.UNDERSTAND, detail="requirements structured")
        await self._emit(
            "requirements_created",
            {"objective": requirements.objective, "requirement_count": len(requirements.to_requirements())},
        )

        # Build a single-task graph directly (no model planner needed).
        await self._enter_stage(RuntimeStage.PLAN, detail="interactive single-task plan")
        graph = TaskGraph()
        task = SubTask(
            task_id="interactive_task_1",
            description=goal,
            role=AgentRole.CODER,
            dependencies=[],
            priority=10,
        )
        task.runtime_context = ""
        graph.add_task(task)
        errors = graph.validate()
        if errors:
            self.state.status = RuntimeStatus.FAILED
            self.state.register_blocker(f"Task graph validation failed: {errors}")
            self.state.set_stage(RuntimeStage.FAILED, detail="invalid task graph")
            await self._emit("runtime_failed", {"reason": str(errors)})
            return RuntimeOutcome(state=self.state, status=RuntimeStatus.FAILED)

        # Traceability: single requirement → single task.
        index = build_traceability_index(requirements, [])
        self.state.traceability = index
        self.state.task_graph = graph
        await self._enter_stage(RuntimeStage.DECOMPOSE, detail="1 task")
        await self._emit("plan_created", {"total_tasks": 1})

        # Attach context to the single task.
        await self._attach_task_contexts(graph, index)

        # Execute through the real Scheduler.
        await self._enter_stage(RuntimeStage.EXECUTE, detail="scheduler dispatch")
        bus = AgentMessageBus(self.event_bus, graph, self.registry)
        scheduler = Scheduler(
            event_bus=self.event_bus,
            registry=self.registry,
            provider=self._provider,
            tools=self.tools,
            workspace_path=self.workspace_path,
            max_concurrency=self.max_concurrency,
            router=self.router,
            lock_manager=self.lock_manager,
            memory=self.memory,
            project_id=self.project_id,
            message_bus=bus,
            interactive_config=getattr(self, '_interactive_config', None),
        )
        self.scheduler = scheduler

        # Attach event handlers for task lifecycle tracking.
        task_handler = self._make_task_event_handler()
        recovery_handler = self._make_recovery_event_handler()
        convergence_handler = self._make_convergence_handler()
        self.event_bus.on("task.started", task_handler)
        self.event_bus.on("task.completed", task_handler)
        self.event_bus.on("task.failed", task_handler)
        self.event_bus.on("recovery_started", recovery_handler)
        self.event_bus.on("recovery_exhausted", recovery_handler)
        self.event_bus.on("tool.result", convergence_handler)

        execution = asyncio.ensure_future(scheduler.execute(graph))
        try:
            if timeout_seconds and timeout_seconds > 0:
                await asyncio.wait_for(execution, timeout=timeout_seconds)
            else:
                await execution
        finally:
            self.event_bus.off("task.started", task_handler)
            self.event_bus.off("task.completed", task_handler)
            self.event_bus.off("task.failed", task_handler)
            self.event_bus.off("recovery_started", recovery_handler)
            self.event_bus.off("recovery_exhausted", recovery_handler)
            self.event_bus.off("tool.result", convergence_handler)
            if not execution.done():
                execution.cancel()

        # Release any locks still held.
        for task_id in list(getattr(self.lock_manager, "_locks", {})):
            await self.lock_manager.release_all(task_id)

        # Record artifacts and verify.
        self._record_artifacts_from_tasks(graph)

        await self._enter_stage(RuntimeStage.VERIFY, detail="requirement verification")
        verification_ok, summary, results = self._verify_requirements(graph, index)
        self.state.verification_status = (
            VerificationStatus.PASSED if verification_ok else VerificationStatus.FAILED
        )
        self.state.verification_summary = summary
        self.state.verification_results = results
        await self._emit(
            "verification_completed",
            {"passed": verification_ok, "summary": summary, "results": results},
        )

        await self._record_memory_references(graph, index)

        # Terminal status decision.
        failed_originals = [
            t.task_id
            for t in graph.tasks.values()
            if t.status == TaskStatus.FAILED and not _is_recovery_task(t.task_id)
        ]
        if self.state.recovery_exhausted:
            status = RuntimeStatus.FAILED
            detail = f"recovery exhausted ({self.state.recovery_attempts} attempts)"
        elif failed_originals and not self._all_failures_recovered(graph, failed_originals):
            status = RuntimeStatus.FAILED
            detail = self._user_facing_error(graph, failed_originals)
        elif not verification_ok:
            status = RuntimeStatus.FAILED
            detail = f"Verification failed: {summary}"
        else:
            status = RuntimeStatus.SUCCESS
            detail = "all requirements verified"
            await self._enter_stage(RuntimeStage.DELIVER, detail=detail)
            await self._enter_stage(RuntimeStage.SUCCEEDED)

        self.state.status = status
        self.state.completed_at = time.time()
        if status == RuntimeStatus.FAILED:
            self.state.set_stage(RuntimeStage.FAILED, detail=detail)
            self.state.register_blocker(detail)
            await self._emit("runtime_failed", {"reason": detail})
        else:
            await self._emit("runtime_completed", {"status": status.value})

        return RuntimeOutcome(
            state=self.state,
            status=status,
            traceability=index,
            message_bus=bus,
            graph=graph,
            duration_ms=(time.time() - started) * 1000,
        )

    # ── Planning → TaskGraph ──────────────────────────────────────────────

    async def _build_task_graph(
        self, original_request: str, requirements: Requirements, plan: Plan | None
    ) -> tuple[TaskGraph | None, list[Any]]:
        """Return (TaskGraph, planned_tasks); None graph means planning failed."""
        if plan is None:
            planner = Planner(
                provider=self._plan_provider,
                registry=self.registry,
                memory=self.memory,
                project_id=self.project_id,
            )
            context: dict[str, Any] = {"requirements": requirements.to_dict()}
            if self.memory and getattr(self.memory, "enabled", False):
                try:
                    prior = await self.memory.get_context_for_planner(
                        user_request=original_request, project_id=self.project_id
                    )
                    if prior:
                        context["prior_context"] = prior
                except Exception:
                    pass
            result = await planner.plan(original_request, context)
            if not result.success or result.plan is None:
                return None, []
            plan = result.plan

        graph = convert_plan_to_graph(plan)
        if graph is None:
            return None, []
        return graph, list(plan.tasks)

    # ── Context assembly ──────────────────────────────────────────────────

    async def _attach_task_contexts(self, graph: TaskGraph, index: TraceabilityIndex) -> None:
        """Attach role-scoped project context to every task before dispatch."""
        for task in graph.tasks.values():
            if task.status not in (TaskStatus.CREATED, TaskStatus.READY, TaskStatus.BLOCKED):
                continue
            role = task.role if isinstance(task.role, AgentRole) else AgentRole.CODER
            memory_context = ""
            if self.memory and getattr(self.memory, "enabled", False):
                try:
                    memory_context = await self.memory.get_context_for_worker(
                        objective=task.description,
                        role=role.value,
                        project_id=self.project_id,
                    )
                except Exception:
                    memory_context = ""
            role_ctx = assemble_role_context(
                role=role,
                objective=task.description,
                requirements=self.state.requirements,
                traceability=index,
                project_state=self.state,
                task_id=task.task_id,
                message_bus=self.scheduler.get_message_bus() if self.scheduler else None,
                memory_context=memory_context,
            )
            task.runtime_context = role_ctx.to_prompt_block()

    # ── Event observation ─────────────────────────────────────────────────

    def _make_task_event_handler(self):
        async def _on_task_event(event: Event) -> None:
            data = event.data or {}
            task_id = data.get("task_id", "")
            if event.type == "task.started":
                agent_id = data.get("agent_id", "")
                self.state.active_agents[task_id] = ActiveAgent(
                    name=agent_id or task_id,
                    role=data.get("role", ""),
                    status="running",
                    task_id=task_id,
                )
            elif event.type == "task.completed":
                self.state.active_agents.pop(task_id, None)
            elif event.type == "task.failed":
                self.state.active_agents.pop(task_id, None)
                self.state.failed_task_ids.append(task_id)

        return _on_task_event

    def _make_recovery_event_handler(self):
        async def _on_recovery(event: Event) -> None:
            data = event.data or {}
            self._recovery_events.append({"type": event.type, **data})
            if event.type == "recovery_started":
                self.state.recovery_attempts += 1
            elif event.type == "recovery_exhausted":
                self.state.recovery_exhausted = True

        return _on_recovery

    def _make_convergence_handler(self):
        """Phase 10.5: observe real tool results for stagnation detection.

        Feeds the runtime-level ConvergenceGovernor with the same evidence the
        scheduler/worker produced and emits ``governor.stagnation`` (with the
        deterministic report) when repeated failures/commands indicate no
        progress.  The governor only reports evidence — the runtime decides.
        """
        from harness_core.runtime.governance import ConvergenceGovernor

        governor = ConvergenceGovernor()

        async def _on_tool_result(event: Event) -> None:
            data = event.data or {}
            tool = data.get("tool", "")
            status = data.get("status", "")
            exit_code = data.get("exit_code")
            args = data.get("args") or {}
            if not tool:
                return
            governor.record_tool(
                tool,
                args,
                succeeded=status == "success",
                exit_code=exit_code,
                stderr=data.get("stderr", ""),
            )
            report = governor.report()
            if report.stalled and not data.get("_stagnation_reported"):
                await self._emit(
                    "governor.stagnation",
                    {"report": report.to_dict(), "tool": tool},
                )
                data["_stagnation_reported"] = True
                if report.escalate_model:
                    await self._emit(
                        "model.escalation",
                        {
                            "reason": report.escalate_reason,
                            "repeated_command": report.repeated_command,
                            "recommended_switch": report.recommended_switch,
                        },
                    )

        return _on_tool_result

    # ── Helpers used by the run loop ──────────────────────────────────────

    @staticmethod
    def _graph_has_role(graph: TaskGraph, role: AgentRole) -> bool:
        return any(t.role == role for t in graph.tasks.values())

    def _all_failures_recovered(self, graph: TaskGraph, failed_original_ids: list[str]) -> bool:
        """A failed original task is recovered iff a retest of it completed."""
        retest_completed = {
            t.task_id
            for t in graph.tasks.values()
            if t.status == TaskStatus.COMPLETED and RETEST_MARKER in t.task_id
        }
        for base in failed_original_ids:
            # Retest ids look like ``<base>_retest_<n>`` — exact prefix check.
            if not any(rt.startswith(base) for rt in retest_completed):
                return False
        return True

    @staticmethod
    def _user_facing_error(graph: TaskGraph, failed_ids: list[str]) -> str:
        """Extract a concise, user-facing error from the first failed task.

        Never exposes internal task IDs or recovery internals.
        """
        for tid in failed_ids:
            task = graph.get_task(tid)
            if task is None:
                continue
            # Prefer a structured error if the task has one
            err = (task.error or "").strip()
            # Strip internal recovery/failure prefixes
            for prefix in ("Task failed:", "Provider error:", "Agent error:", "Runtime error:"):
                if err.startswith(prefix):
                    err = err[len(prefix):].strip()
            if err and len(err) > 5:
                # Take first meaningful line only
                first_line = err.splitlines()[0].strip()
                if len(first_line) > 120:
                    first_line = first_line[:117] + "..."
                return first_line
        return "Task did not complete successfully."

    async def _publish_dependency_handoffs(self, graph: TaskGraph, bus: AgentMessageBus) -> None:
        """Publish typed handoffs from each completed task to its dependents."""
        for task in graph.tasks.values():
            if task.status != TaskStatus.COMPLETED:
                continue
            dependents = graph.get_dependents(task.task_id)
            for dependent in dependents:
                payload: dict[str, Any] = {
                    "files_changed": list(task.files_changed or []),
                    "summary": (task.result or "")[:2000],
                    "artifacts": [
                        a.to_dict()
                        for a in self.state.artifacts
                        if a.produced_by_task == task.task_id
                    ],
                }
                message = AgentMessage(
                    sender_agent_id=f"task:{task.task_id}",
                    sender_task_id=task.task_id,
                    recipient_task_id=dependent.task_id,
                    message_type=MessageType.HANDOFF,
                    payload=payload,
                )
                try:
                    await bus.send(message)
                except Exception:
                    continue  # never break the run because a handoff failed

    def _record_artifacts_from_tasks(self, graph: TaskGraph) -> None:
        for task in graph.tasks.values():
            if task.status != TaskStatus.COMPLETED:
                continue
            for path in task.files_changed or []:
                kind = _infer_artifact_kind(task, path)
                self.state.register_artifact(
                    ArtifactRef(
                        kind=kind,
                        path=str(path),
                        produced_by_task=task.task_id,
                        summary=(task.result or "")[:300],
                    )
                )

    def _verify_requirements(
        self, graph: TaskGraph, index: TraceabilityIndex
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Decide requirement-level completion.

        Evidence comes from the TaskGraph (which tasks completed / failed) and
        from verifier/reviewer task summaries.  A requirement is VERIFIED only
        when all of its mapped tasks completed **and** no verifier failed it.
        """
        completed = {
            t.task_id for t in graph.tasks.values() if t.status == TaskStatus.COMPLETED
        }
        # A failed original task is *done* if a later retest of it completed
        # (new execution + evidence — never FAILED→SUCCESS without both).
        retest_completed = {
            t.task_id
            for t in graph.tasks.values()
            if RETEST_MARKER in t.task_id and t.status == TaskStatus.COMPLETED
        }

        def _task_done(task_id: str) -> bool:
            if task_id in completed:
                return True
            return any(rt.startswith(task_id) for rt in retest_completed)

        # 1. Task-completion evidence for every requirement: one PASS per
        # completed (or recovered) mapped task, one FAIL per still-broken task.
        for req_id, trace in index.traces.items():
            for mapped_task in trace.requirement.task_ids:
                if _task_done(mapped_task):
                    trace.add_evidence(
                        Evidence(
                            source=mapped_task,
                            status=EvidenceStatus.PASS,
                            detail="task completed"
                            + (" (recovered via retest)" if mapped_task not in completed else ""),
                        )
                    )
                else:
                    trace.add_evidence(
                        Evidence(
                            source=mapped_task,
                            status=EvidenceStatus.FAIL,
                            detail="task not completed",
                        )
                    )

        # 2. Verifier/reviewer judgement (explicit FAIL overrides task evidence).
        failures: list[str] = []
        for task in graph.tasks.values():
            if task.status != TaskStatus.COMPLETED:
                continue
            if task.role not in (AgentRole.VERIFIER, AgentRole.REVIEWER):
                continue
            verdicts = _parse_verifier_verdicts(task.result or "")
            for req_id, ok in verdicts.items():
                trace = index.traces.get(req_id)
                if trace is None:
                    continue
                detail = task.result or ""
                if ok:
                    trace.add_evidence(
                        Evidence(source=task.task_id, status=EvidenceStatus.PASS, detail=detail)
                    )
                else:
                    trace.add_evidence(
                        Evidence(source=task.task_id, status=EvidenceStatus.FAIL, detail=detail)
                    )
                    failures.append(req_id)

        # 3. Roll up.
        verified, failed, unverified, total = index.overall()
        results = [t.to_dict() for t in index.traces.values()]
        if failures:
            return False, f"verifier rejected requirements: {sorted(set(failures))}", results
        if failed:
            return False, f"{failed} requirement(s) failed evidence", results
        if total == 0:
            # No structured requirements were supplied (e.g. an ad-hoc CLI
            # request).  Nothing can be traced, so verification falls back to
            # graph-completion evidence: every planned (non-recovery) task must
            # have completed, or been recovered by a completed retest.
            originals = [t for t in graph.tasks.values() if not _is_recovery_task(t.task_id)]
            done = all(_task_done(t.task_id) for t in originals)
            if done:
                return True, "no structured requirements; all planned tasks completed", results
            missing = [t.task_id for t in originals if not _task_done(t.task_id)]
            return False, f"planned tasks incomplete: {missing}", results
        if verified == total:
            return True, f"all {total} requirements verified", results
        return False, f"verification incomplete ({verified}/{total} verified)", results

    async def _record_memory_references(self, graph: TaskGraph, index: TraceabilityIndex) -> None:
        if not self.memory or not getattr(self.memory, "enabled", False):
            return
        try:
            for artifact in self.state.artifacts:
                await self.memory.record_episode(
                    description=f"artifact {artifact.kind} produced",
                    outcome=f"{artifact.path} by {artifact.produced_by_task}",
                    agent_role="runtime",
                    project_id=self.project_id,
                    tags=["artifact", artifact.kind],
                    importance=0.3,
                    metadata={"path": artifact.path, "kind": artifact.kind},
                )
        except Exception:
            pass  # memory is best-effort

    async def request_cancel(self, reason: str = "User cancelled") -> None:
        """Public, idempotent interruption entry point for CLI-level Ctrl+C.

        The CLI (Phase 10) calls this on a fresh event loop after asyncio.run
        unwinds, because a task scheduled from inside an already-cancelled
        coroutine may be torn down before it finishes.  Idempotent: safe to
        call even when the CancelledError path already ran.
        """
        await self._mark_interrupted(reason)

    async def _mark_interrupted(self, reason: str) -> None:
        """Interruption: mark non-terminal tasks CANCELLED, release locks."""
        self.state.status = RuntimeStatus.CANCELLED
        self.state.set_stage(RuntimeStage.CANCELLED, detail=reason)
        self.state.register_warning(reason)
        if self.scheduler is not None:
            await self.scheduler.cancel_all()
        for task_id in list(getattr(self.lock_manager, "_locks", {})):
            await self.lock_manager.release_all(task_id)
        await self._emit("runtime_failed", {"reason": reason, "interrupted": True})


# ── Module-level helpers (pure) ───────────────────────────────────────────


def build_traceability_index(
    requirements: Requirements, planned_tasks: list[Any]
) -> TraceabilityIndex:
    """Record which requirement(s) each planned task implements.

    The planner may propose ``traceable_to`` ids; the runtime validates them
    (only known requirement ids are accepted).  Tasks without an explicit
    mapping fall back to a keyword match against requirement statements so
    traceability still holds when the model omits the field.
    """
    reqs = requirements.to_requirements()
    by_id = {r.req_id: r for r in reqs}
    for r in reqs:
        r.task_ids = []

    def _assign(task_id: str, req_id: str) -> None:
        if req_id in by_id and req_id not in by_id[req_id].task_ids:
            by_id[req_id].task_ids.append(task_id)

    for planned in planned_tasks:
        proposed = list(getattr(planned, "traceable_to", []) or [])
        if proposed:
            for req_id in proposed:
                _assign(planned.task_id, req_id)
            continue
        # Fallback keyword matching against requirement statements.  Acceptance
        # criteria are owned by the verifier, not the planner: they restate the
        # functional requirement, so matching against them would shadow the
        # real mapping.  Only functional/non-functional/constraint statements
        # participate in the fallback.
        tokens = _keyword_tokens(planned.objective)
        for req_id, req in by_id.items():
            if req.category in ("acceptance", "exclusion"):
                continue
            if not req.statement:
                continue
            if _tokens_overlap(tokens, _keyword_tokens(req.statement)):
                _assign(planned.task_id, req_id)

    return TraceabilityIndex.build(reqs)


def _tokens_overlap(a: set[str], b: set[str]) -> bool:
    """True when a token matches exactly or shares a >=4-char common prefix.

    Keeps the fallback honest for inflectional variants (register/registration,
    task/tasks) without requiring a full stemmer.
    """
    for x in a:
        for y in b:
            n = min(len(x), len(y))
            i = 0
            while i < n and x[i] == y[i]:
                i += 1
            if i >= 4:
                return True
    return False


def _keyword_tokens(text: str) -> set[str]:
    """Crude keyword set for fallback requirement matching (stopwords removed)."""
    stop = {
        "the", "a", "an", "and", "of", "to", "in", "for", "on", "with", "that",
        "is", "are", "be", "by", "as", "it", "its", "user", "users", "should",
        "must", "can", "this", "from", "at", "or", "system", "build", "create",
        "implement", "ensure", "all", "their",
    }
    out: set[str] = set()
    for word in text.lower().split():
        word = "".join(ch for ch in word if ch.isalnum())
        if word and word not in stop:
            out.add(word)
    return out


def convert_plan_to_graph(plan: Plan) -> TaskGraph | None:
    """Convert a Plan into a validated TaskGraph.

    The plan is model output — never authoritative.  Only this conversion
    (role mapping, resource parsing, dependency wiring) plus TaskGraph's own
    validation turns it into runtime state.
    """
    if isinstance(plan, PlanningResult):
        plan = plan.plan
    if not isinstance(plan, Plan) or not plan.tasks:
        return None

    graph = TaskGraph()
    for planned in plan.tasks:
        try:
            role = AgentRole(planned.role.lower())
        except ValueError:
            role = AgentRole.CODER
        resources = []
        for res in planned.resources or []:
            if isinstance(res, dict) and "path" in res and "mode" in res:
                try:
                    resources.append(WorkspaceResource.from_dict(res))
                except ValueError:
                    continue
        task = SubTask(
            task_id=planned.task_id,
            description=planned.objective,
            role=role,
            dependencies=list(planned.dependencies or []),
            priority=planned.priority,
            resources=resources,
        )
        task.runtime_context = ""
        graph.add_task(task)

    errors = graph.validate()
    if errors:
        return None
    return graph


def _infer_artifact_kind(task: SubTask, path: str) -> str:
    """Infer an artifact kind from the producing task's role + path."""
    role = task.role.value if isinstance(task.role, AgentRole) else str(task.role)
    low_path = str(path).lower()
    if task.role == AgentRole.ARCHITECT:
        return "architecture_decision"
    if task.role == AgentRole.REVIEWER:
        return "review_report"
    if task.role == AgentRole.TESTER:
        return "test_report"
    if task.role == AgentRole.DEBUGGER:
        return "failure_report"
    if "schema" in low_path or task.role == AgentRole.DATABASE:
        return "database_schema"
    if "contract" in low_path or "api" in low_path:
        return "api_contract"
    if "spec" in low_path or "ui" in low_path or role in ("ui_designer", "frontend"):
        return "ui_spec"
    return "source"


def _parse_verifier_verdicts(summary: str) -> dict[str, bool]:
    """Parse ``REQ-001: PASS|FAIL`` verdicts from a verifier task summary.

    Verifier agents report machine-readable verdicts in their summary; the
    runtime (never the model) decides what those verdicts mean for the
    requirement trace.
    """
    verdicts: dict[str, bool] = {}
    for line in (summary or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept compact JSON lists too: [{"req_id": "REQ-001", "status": "pass"}]
        if line.startswith("[") or line.startswith("{"):
            try:
                parsed = json.loads(line)
                items = parsed if isinstance(parsed, list) else [parsed]
                for item in items:
                    req_id = (item.get("req_id") or "").upper()
                    status = str(item.get("status", "")).lower()
                    if req_id.startswith("REQ-") and status in ("pass", "fail"):
                        verdicts[req_id] = status == "pass"
                continue
            except (ValueError, AttributeError):
                pass
        upper = line.upper()
        for marker in ("PASS", "FAIL"):
            idx = upper.find(marker)
            if idx == -1:
                continue
            head = upper[:idx].strip().rstrip(":")
            if ":" in head:
                # Handles "REQ-001: PASS" (and any prose before the marker).
                req_id = head.split(":")[-1].strip() or head.split(":")[0].strip()
            else:
                req_id = head
            if req_id.startswith("REQ-"):
                verdicts[req_id] = marker == "PASS"
            break
    return verdicts
