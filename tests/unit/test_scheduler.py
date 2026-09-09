import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from harness_core.agents.domain import AgentContract, AgentRole, AgentStatus, SubTask, TaskGraph, TaskStatus, AgentResult
from harness_core.agents.scheduler import Scheduler
from harness_core.agents.registry import AgentRegistry, AgentConfig
from harness_core.observability.events import EventBus

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture
def registry():
    reg = AgentRegistry()
    return reg

@pytest.fixture
def base_scheduler(event_bus, registry):
    return Scheduler(
        event_bus=event_bus,
        registry=registry,
        provider=AsyncMock(),
        tools=[],
        workspace_path=".",
        max_concurrency=2
    )

@pytest.mark.asyncio
async def test_single_task_execution(base_scheduler):
    graph = TaskGraph()
    task = SubTask(description="Test A", role=AgentRole.CODER)
    graph.add_task(task)
    
    with patch("harness_core.agents.worker.WorkerAgent.run", new_callable=AsyncMock) as mock_run:
        result = AgentResult(
            agent_id="mock-agent",
            role=AgentRole.CODER,
            status=AgentStatus.COMPLETED,
            summary="Done",
            model_id="test-model"
        )
        mock_run.return_value = result
        
        await base_scheduler.execute(graph)
        
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "Done"

@pytest.mark.asyncio
async def test_diamond_dependency_graph(base_scheduler):
    # A -> B & C -> D
    graph = TaskGraph()
    a = SubTask(description="A", role=AgentRole.CODER)
    b = SubTask(description="B", role=AgentRole.CODER, dependencies=[a.task_id])
    c = SubTask(description="C", role=AgentRole.CODER, dependencies=[a.task_id])
    d = SubTask(description="D", role=AgentRole.CODER, dependencies=[b.task_id, c.task_id])
    
    for t in [a, b, c, d]:
        graph.add_task(t)
        
    execution_order = []
    
    async def mock_run(self):
        # Record start time conceptually by appending
        execution_order.append(self.contract.objective)
        await asyncio.sleep(0.01)
        result = AgentResult(
            agent_id="mock-agent",
            role=AgentRole.CODER,
            status=AgentStatus.COMPLETED,
            summary="Done",
            model_id="test-model"
        )
        return result
        
    with patch("harness_core.agents.worker.WorkerAgent.run", new=mock_run):
        await base_scheduler.execute(graph)
        
    assert a.status == TaskStatus.COMPLETED
    assert b.status == TaskStatus.COMPLETED
    assert c.status == TaskStatus.COMPLETED
    assert d.status == TaskStatus.COMPLETED
    
    assert execution_order[0] == "A"
    assert set(execution_order[1:3]) == {"B", "C"}
    assert execution_order[3] == "D"

@pytest.mark.asyncio
async def test_failed_dependency_propagation(base_scheduler):
    graph = TaskGraph()
    a = SubTask(description="A", role=AgentRole.CODER)
    b = SubTask(description="B", role=AgentRole.CODER, dependencies=[a.task_id])
    
    graph.add_task(a)
    graph.add_task(b)
    
    async def mock_run(self):
        result = AgentResult(
            agent_id="mock-agent",
            role=AgentRole.CODER,
            status=AgentStatus.FAILED,
            errors=["Simulated failure"]
        )
        return result
        
    async def mock_generate(*args, **kwargs):
        from harness_core.providers.base import CompletionResponse
        return CompletionResponse(content='{"strategy": "ignore", "objective": "ignore", "specialist_role": "coder", "target_task_id": "a"}', model="test")
        
    base_scheduler.provider.generate = AsyncMock(side_effect=mock_generate)
        
    with patch("harness_core.agents.worker.WorkerAgent.run", new=mock_run):
        try:
            await asyncio.wait_for(base_scheduler.execute(graph), timeout=2.0)
        except asyncio.TimeoutError:
            print("Tasks:", [(t.task_id, t.status) for t in graph.tasks.values()])
            print("Active workers:", base_scheduler.active_workers)
            raise
        
    assert a.status == TaskStatus.FAILED
    assert b.status == TaskStatus.BLOCKED

@pytest.mark.asyncio
async def test_concurrency_limit(base_scheduler):
    base_scheduler.max_concurrency = 1
    graph = TaskGraph()
    a = SubTask(description="A", role=AgentRole.CODER)
    b = SubTask(description="B", role=AgentRole.CODER)
    
    graph.add_task(a)
    graph.add_task(b)
    
    active_count = 0
    max_active = 0
    
    async def mock_run(self):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.05)
        active_count -= 1
        
        result = AgentResult(
            agent_id="mock-agent",
            role=AgentRole.CODER,
            status=AgentStatus.COMPLETED,
            summary="Done",
            model_id="test-model"
        )
        return result
        
    with patch("harness_core.agents.worker.WorkerAgent.run", new=mock_run):
        await base_scheduler.execute(graph)
        
    assert max_active == 1
    assert a.status == TaskStatus.COMPLETED
    assert b.status == TaskStatus.COMPLETED

@pytest.mark.asyncio
async def test_cancellation(base_scheduler):
    graph = TaskGraph()
    a = SubTask(description="A", role=AgentRole.CODER)
    graph.add_task(a)
    
    async def mock_run(self):
        await asyncio.sleep(0.5)
        result = AgentResult(
            agent_id="mock-agent",
            role=AgentRole.CODER,
            status=AgentStatus.COMPLETED
        )
        return result
        
    with patch("harness_core.agents.worker.WorkerAgent.run", new=mock_run):
        execute_task = asyncio.create_task(base_scheduler.execute(graph))
        await asyncio.sleep(0.05)
        await base_scheduler.cancel_all()
        await execute_task
        
    assert a.status == TaskStatus.CANCELLED
