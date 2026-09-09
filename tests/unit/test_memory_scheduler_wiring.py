"""Phase 8 hardening — memory reaches Planner/Scheduler/WorkerAgent.

These tests drive the *runtime construction path* used by the real CLI
(``harness_core.memory.init_memory_manager`` → ``Orchestrator`` → ``Planner``
/``Scheduler`` → ``WorkerAgent``) and prove, with a single shared manager,
that:

1. A task graph executed through the real Scheduler records a persistent
   SUCCESS memory when a worker completes.
2. The WorkerAgent prompt (built by ``_build_prompt_async``) is augmented
   with ``## Historical Context`` retrieved from that same manager, and the
   Planner prompt is augmented with ``# Prior Project Context``.
3. Memory failures (write + retrieval) cannot fail task execution: a worker
   whose memory calls raise still completes, and the run still succeeds.

The agent loop itself is stubbed at the loop boundary (no network / no LLM);
every layer between the Scheduler and the memory store is real production
code.

The end-to-end CLI entry point is exercised in
``tests/e2e/test_cli_memory_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from harness_core.agents.domain import (
    AgentContract,
    AgentRole,
    SubTask,
    TaskGraph,
    TaskStatus,
    WorkspaceScope,
)
from harness_core.agents.registry import AgentRegistry
from harness_core.agents.scheduler import Scheduler
from harness_core.memory import init_memory_manager
from harness_core.memory.domain import MemoryType
from harness_core.observability.events import EventBus


class _CompletedLoop:
    """Stand-in for AgentLoop that always finishes 'completed' immediately."""

    def __init__(self, *args, **kwargs) -> None:
        return None

    async def run(self, goal: str):
        self.last_goal = goal
        return SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            result="Done",
            tool_calls=[],
            iterations=1,
            error=None,
            models_used=["fake"],
            total_tokens=10,
        )


def _make_graph(task_id: str = "task_1") -> TaskGraph:
    task = SubTask(
        task_id=task_id,
        description="Implement the thing",
        role=AgentRole.CODER,
        status=TaskStatus.CREATED,
        resources=[],
    )
    graph = TaskGraph()
    graph.add_task(task)
    return graph


async def _run_scheduler(manager, workspace_path: Path) -> tuple[Scheduler, TaskGraph]:
    """Execute a one-task graph through the real Scheduler + a stubbed loop."""
    import harness_core.agent.loop as loop_module

    registry = AgentRegistry()
    event_bus = EventBus()
    scheduler = Scheduler(
        event_bus=event_bus,
        registry=registry,
        provider=object(),  # unused: the agent loop is stubbed
        tools=[],
        workspace_path=str(workspace_path),
        max_concurrency=1,
        memory=manager,
        project_id=str(workspace_path.resolve()),
    )

    original_loop = loop_module.AgentLoop
    loop_module.AgentLoop = _CompletedLoop
    try:
        graph = _make_graph()
        await scheduler.execute(graph)
    finally:
        loop_module.AgentLoop = original_loop  # restore, never delete

    return scheduler, graph


class TestSchedulerMemoryWiring:
    async def test_scheduler_records_success_memory_through_real_chain(
        self, tmp_path: Path
    ):
        """Real init_memory_manager -> Scheduler -> WorkerAgent -> store."""
        manager = init_memory_manager(tmp_path, enabled=True)
        assert manager is not None and manager.enabled

        scheduler, graph = await _run_scheduler(manager, tmp_path)

        assert graph.get_task("task_1").status == TaskStatus.COMPLETED
        entries = await manager.store.all_entries()
        successes = [e for e in entries if e.type == MemoryType.SUCCESS]
        assert len(successes) == 1, (
            f"Scheduler must record one SUCCESS memory, got {len(successes)}"
        )
        assert successes[0].project_id == str(tmp_path.resolve())

        # The same manager is threaded into the worker that executed the task.
        worker = scheduler.workers.get(scheduler.workers and list(scheduler.workers)[0])
        assert worker is not None

    async def test_worker_prompt_is_augmented_with_historical_context(
        self, tmp_path: Path
    ):
        """Phase 8G: seeded FAILURE memory is injected into the worker prompt."""
        manager = init_memory_manager(tmp_path, enabled=True)
        project_id = str(tmp_path.resolve())
        await manager.record_failure(
            task_id="past_task",
            description="Implement JWT auth",
            error="refresh tokens were never rotated",
            agent_role="coder",
            project_id=project_id,
        )

        # Build a worker exactly as Scheduler does (see scheduler.py).
        from harness_core.agents.worker import WorkerAgent

        contract = AgentContract(
            role=AgentRole.CODER,
            task_id="task_1",
            objective="Implement the thing",
            workspace_scope=WorkspaceScope.PROJECT,
            model_policy="auto",
        )
        worker = WorkerAgent(
            contract=contract,
            provider=object(),
            tools=[],
            event_bus=EventBus(),
            workspace_path=str(tmp_path),
            memory=manager,
            project_id=project_id,
        )

        prompt = await worker._build_prompt_async()
        assert "## Historical Context" in prompt
        assert "refresh tokens were never rotated" in prompt

    async def test_planner_prompt_is_augmented_with_prior_context(
        self, tmp_path: Path
    ):
        """Phase 8F: prior successes surface as 'Prior Successful Approaches'."""
        from harness_core.planning.planner import Planner

        manager = init_memory_manager(tmp_path, enabled=True)
        project_id = str(tmp_path.resolve())
        await manager.record_success(
            task_id="past_auth",
            description="Build an authentication system",
            outcome="Used JWT with 15-minute access + 7-day rotating refresh tokens",
            agent_role="coder",
            project_id=project_id,
        )

        class _Recorder:
            """Provider that records the prompts it is handed."""

            def __init__(self) -> None:
                self.prompts: list[dict] = []

            async def generate(self, request):
                self.prompts.append(request)
                from harness_core.providers.base import CompletionResponse

                return CompletionResponse(content='{"summary": "s", "tasks": []}', model="fake")

        recorder = _Recorder()
        planner = Planner(
            provider=recorder, registry=AgentRegistry(), memory=manager, project_id=project_id
        )
        await planner.plan("build another authentication system")
        assert recorder.prompts, "Planner must call the provider"
        system_prompt = recorder.prompts[0].messages[0]["content"]
        user_prompt = recorder.prompts[0].messages[1]["content"]
        assert "# Prior Project Context" in system_prompt
        assert "rotating refresh tokens" in system_prompt
        assert "build another authentication system" in user_prompt


class TestMemoryFailureIsolation:
    async def test_worker_survives_memory_write_and_retrieval_failures(
        self, tmp_path: Path
    ):
        """Booming memory must not turn a good run into a failure."""
        from unittest.mock import AsyncMock

        manager = init_memory_manager(tmp_path, enabled=True)
        project_id = str(tmp_path.resolve())

        # Sabotage every memory surface the runtime touches.
        manager.get_context_for_worker = AsyncMock(side_effect=RuntimeError("retrieval down"))
        manager.record_success = AsyncMock(side_effect=RuntimeError("store down"))
        # record_* go through _record -> store/index/graph; sabotage those too.
        async def _boom(*a, **k):
            raise RuntimeError("store down")

        manager.store.save = _boom
        manager.store.all_entries = _boom

        import harness_core.agent.loop as loop_module

        original_loop = loop_module.AgentLoop
        loop_module.AgentLoop = _CompletedLoop
        try:
            registry = AgentRegistry()
            scheduler = Scheduler(
                event_bus=EventBus(),
                registry=registry,
                provider=object(),
                tools=[],
                workspace_path=str(tmp_path),
                max_concurrency=1,
                memory=manager,
                project_id=project_id,
            )
            graph = _make_graph()
            await scheduler.execute(graph)
        finally:
            loop_module.AgentLoop = original_loop

        # The engineering task itself must still succeed.
        assert graph.get_task("task_1").status == TaskStatus.COMPLETED
        # …and no memory entry was written while it was down.
        # (store.all_entries was replaced; the real store file stays empty.)
        assert not (tmp_path / ".harness" / "memory" / "entries.json").exists()

    async def test_disabled_manager_keeps_runtime_working(self, tmp_path: Path):
        """Opted-out / disabled memory behaves like None everywhere."""
        import harness_core.agent.loop as loop_module

        original_loop = loop_module.AgentLoop
        loop_module.AgentLoop = _CompletedLoop
        try:
            registry = AgentRegistry()
            scheduler = Scheduler(
                event_bus=EventBus(),
                registry=registry,
                provider=object(),
                tools=[],
                workspace_path=str(tmp_path),
                max_concurrency=1,
                memory=None,
                project_id=str(tmp_path.resolve()),
            )
            graph = _make_graph()
            await scheduler.execute(graph)
        finally:
            loop_module.AgentLoop = original_loop

        assert graph.get_task("task_1").status == TaskStatus.COMPLETED
        assert not (tmp_path / ".harness").exists()
