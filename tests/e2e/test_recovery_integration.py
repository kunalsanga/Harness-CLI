"""
End-to-End test for autonomous debugging and recovery (Phase 7).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from harness_core.agents.domain import AgentRole, SubTask, TaskGraph, TaskStatus, AgentStatus, MessageType
from harness_core.agents.registry import AgentRegistry, AgentProfile
from harness_core.observability.events import EventBus
from harness_core.agents.scheduler import Scheduler
from harness_core.providers.base import CompletionResponse

@pytest.mark.asyncio
async def test_end_to_end_recovery_flow():
    """
    Scenario:
    1. Backend -> Tester -> Reviewer -> Verifier
    2. Tester fails (2 failing tests).
    3. Scheduler invokes RecoveryOrchestrator.
    4. RecoveryOrchestrator spawns DEBUGGER -> TESTER_RETEST.
    5. Debugger fixes code.
    6. Tester_retest passes.
    7. Reviewer & Verifier proceed.
    """
    event_bus = EventBus()
    registry = AgentRegistry()
    registry.register(AgentProfile(name="backend", role=AgentRole.BACKEND, system_instructions=""))
    registry.register(AgentProfile(name="tester", role=AgentRole.TESTER, system_instructions=""))
    registry.register(AgentProfile(name="debugger", role=AgentRole.DEBUGGER, system_instructions=""))
    registry.register(AgentProfile(name="reviewer", role=AgentRole.REVIEWER, system_instructions=""))
    registry.register(AgentProfile(name="verifier", role=AgentRole.VERIFIER, system_instructions=""))
    
    mock_provider = MagicMock()
    
    # We will track which agent is running to fake responses
    run_counts = {}
    
    async def mock_gen(request, *args, **kwargs):
        await asyncio.sleep(0.01)
        # If it's a planner request, give it a debug_and_fix strategy
        if request and hasattr(request, "messages") and request.messages:
            content = str(request.messages[0].content)
            if "autonomous engineering recovery coordinator" in content:
                return CompletionResponse(
                    content='{"strategy": "debug_and_fix", "specialist_role": "debugger", "objective": "fix tests", "target_task_id": "tester_1"}',
                    model="test-planner"
                )
            if "FAILURE CLASSIFICATION PROMPT" in content or "categorize the following failure" in content:
                return CompletionResponse(
                    content='{"category": "test_failure", "confidence": 0.9, "summary": "tests fail", "recommended_role": "debugger"}',
                    model="test-classifier"
                )
        return CompletionResponse(content="Done", model="test-worker")
        
    mock_provider.generate = AsyncMock(side_effect=mock_gen)

    scheduler = Scheduler(event_bus, registry, mock_provider, [], "/workspace", max_concurrency=4)

    # To simulate task results, we will patch `worker.run()` inside the scheduler.
    # Since worker.run uses AgentLoop which calls provider, we can just intercept `WorkerAgent.run`
    from harness_core.agents.worker import WorkerAgent
    original_run = WorkerAgent.run
        
    async def fake_worker_run(self):
        role = self.contract.role
        run_counts[role] = run_counts.get(role, 0) + 1
        
        from harness_core.agents.domain import AgentResult
        self.set_status(AgentStatus.COMPLETED)
        
        # If this is the FIRST tester run, FAIL it with test failures.
        if role == AgentRole.TESTER and run_counts[role] == 1:
            return AgentResult(
                agent_id=self.contract.agent_id,
                role=self.contract.role,
                status=AgentStatus.COMPLETED,
                tests_passed=1,
                tests_total=3,
                summary="2 tests failed.",
                findings=[{"type": "test_failure", "test_name": "test_auth"}]
            )
            
        # For all other agents (and retests), succeed normally without JSON parsing
        return AgentResult(
            agent_id=self.contract.agent_id,
            role=self.contract.role,
            status=AgentStatus.COMPLETED,
            summary="Done"
        )

    WorkerAgent.run = fake_worker_run
    
    try:
        # Build Task Graph
        graph = TaskGraph()
        backend_task = SubTask(task_id="backend_1", role=AgentRole.BACKEND)
        tester_task = SubTask(task_id="tester_1", role=AgentRole.TESTER, dependencies=["backend_1"])
        reviewer_task = SubTask(task_id="reviewer_1", role=AgentRole.REVIEWER, dependencies=["tester_1"])
        verifier_task = SubTask(task_id="verifier_1", role=AgentRole.VERIFIER, dependencies=["reviewer_1"])
        
        graph.add_task(backend_task)
        graph.add_task(tester_task)
        graph.add_task(reviewer_task)
        graph.add_task(verifier_task)
        
        events = []
        async def capture(event):
            events.append(event)
            
        event_bus.on("*", capture)
        
        # Run with timeout to prevent hanging!
        try:
            await asyncio.wait_for(scheduler.execute(graph), timeout=5.0)
        except asyncio.TimeoutError:
            print("\n!!! SCHEDULER HANGED !!!")
            for t in graph.tasks.values():
                print(f"Task: {t.task_id}, Status: {t.status}, Deps: {t.dependencies}")
            raise
        
        # Assertions
        
        # 1. Tester was run TWICE (once failed, once retest)
        assert run_counts.get(AgentRole.TESTER, 0) == 2, "Tester should have run twice (original + retest)"
        
        # 2. Debugger was run ONCE (the recovery)
        assert run_counts.get(AgentRole.DEBUGGER, 0) == 1, "Debugger should have run once"
        
        # 3. Reviewer and Verifier should have run (which proves they were unblocked by the retest)
        assert run_counts.get(AgentRole.REVIEWER, 0) == 1
        assert run_counts.get(AgentRole.VERIFIER, 0) == 1
        
        # 4. Check events
        recovery_started = [e for e in events if e.type == "recovery_started"]
        assert len(recovery_started) == 1
        
        recovery_created = [e for e in events if e.type == "recovery_task_created"]
        assert len(recovery_created) == 1
        
        # 5. Original tester task should be FAILED in history, but new one COMPLETED
        assert graph.get_task("tester_1").status == TaskStatus.FAILED
        
        # 6. Graph should be complete
        assert graph.is_complete()
        
    finally:
        # Restore monkey patch
        WorkerAgent.run = original_run

@pytest.mark.asyncio
async def test_failure_exhaustion():
    """
    Test exhaustion of recovery attempts.
    """
    event_bus = EventBus()
    registry = AgentRegistry()
    registry.register(AgentProfile(name="tester", role=AgentRole.TESTER, system_instructions=""))
    registry.register(AgentProfile(name="debugger", role=AgentRole.DEBUGGER, system_instructions=""))
    
    mock_provider = MagicMock()
        
    async def mock_gen(request, *args, **kwargs):
        await asyncio.sleep(0.01)
        if request and hasattr(request, "messages") and request.messages:
            content = str(request.messages[0].content)
            if "autonomous engineering recovery coordinator" in content:
                return CompletionResponse(
                    content='{"strategy": "debug_and_fix", "specialist_role": "debugger", "objective": "fix tests", "target_task_id": "tester_1"}',
                    model="test-planner"
                )
            if "FAILURE CLASSIFICATION PROMPT" in content or "categorize the following failure" in content:
                return CompletionResponse(
                    content='{"category": "test_failure", "confidence": 0.9, "summary": "tests fail", "recommended_role": "debugger"}',
                    model="test-classifier"
                )
        return CompletionResponse(content="Done", model="test-worker")
        
    mock_provider.generate = AsyncMock(side_effect=mock_gen)

    scheduler = Scheduler(event_bus, registry, mock_provider, [], "/workspace", max_concurrency=4)
    # Set limit to 2 for faster test
    scheduler.recovery_orchestrator.max_attempts = 2

    from harness_core.agents.worker import WorkerAgent
    original_run = WorkerAgent.run
    
    async def fake_worker_run(self):
        from harness_core.agents.domain import AgentResult
        self.set_status(AgentStatus.COMPLETED)
        
        # ALWAYS fail tester
        if self.contract.role == AgentRole.TESTER:
            return AgentResult(
                agent_id=self.contract.agent_id,
                role=self.contract.role,
                status=AgentStatus.COMPLETED,
                tests_passed=0,
                tests_total=1,
                summary="tests keep failing"
            )
        
        # Succeed debugger to avoid JSON parsing error loop
        return AgentResult(
            agent_id=self.contract.agent_id,
            role=self.contract.role,
            status=AgentStatus.COMPLETED,
            summary="Fixed"
        )

    WorkerAgent.run = fake_worker_run
    
    try:
        graph = TaskGraph()
        tester_task = SubTask(task_id="tester_1", role=AgentRole.TESTER)
        graph.add_task(tester_task)
        
        events = []
        async def capture(event):
            events.append(event)
        event_bus.on("*", capture)
        
        await scheduler.execute(graph)
        
        exhausted = [e for e in events if e.type == "recovery_exhausted"]
        assert len(exhausted) == 1
        
        # Max attempts was 2, meaning original test + 2 recovery retests = 3 tester runs total
        test_failures = [e for e in events if e.type == "task.failed" and e.data.get("task_id", "").startswith("tester")]
        assert len(test_failures) == 3
        
    finally:
        WorkerAgent.run = original_run
