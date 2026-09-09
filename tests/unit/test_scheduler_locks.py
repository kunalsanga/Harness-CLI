import asyncio
import pytest
from unittest.mock import MagicMock

from harness_core.agents.domain import TaskGraph, SubTask, AgentRole, TaskStatus, WorkspaceScope
from harness_core.agents.scheduler import Scheduler
from harness_core.agents.registry import AgentRegistry
from harness_core.observability.events import EventBus
from harness_core.agents.locks import WorkspaceResource, ResourceMode
from harness_core.providers.base import ModelProvider, CompletionResponse

class AsyncMockWorker:
    def __init__(self, delay=0.1, result=None):
        self.delay = delay
        self.result = result

@pytest.mark.asyncio
async def test_scheduler_parallel_disjoint():
    """Prove disjoint tasks execute concurrently."""
    event_bus = EventBus()
    registry = AgentRegistry()
    mock_provider = MagicMock(spec=ModelProvider)
    # We mock out the provider so workers just instantly finish
    mock_provider.generate.return_value = CompletionResponse(content="Done", model="test")
    
    scheduler = Scheduler(event_bus, registry, mock_provider, [], "/workspace", max_concurrency=2)
    
    graph = TaskGraph()
    t1 = SubTask(task_id="t1", role=AgentRole.FRONTEND, resources=[WorkspaceResource("src/frontend/**", ResourceMode.WRITE)])
    t2 = SubTask(task_id="t2", role=AgentRole.DATABASE, resources=[WorkspaceResource("src/db/**", ResourceMode.WRITE)])
    
    graph.add_task(t1)
    graph.add_task(t2)
    
    # We want to trace if they are queued concurrently. 
    events = []
    async def handler(e):
        events.append((e.type, e.data.get("task_id")))
    
    event_bus.on("task.started", handler)
    event_bus.on("resource_lock_waiting", handler)
    
    await scheduler.execute(graph)
    
    # Should not have any lock waiting events
    wait_events = [e for e in events if e[0] == "resource_lock_waiting"]
    assert len(wait_events) == 0
    
    # Both tasks started
    started_tasks = {e[1] for e in events if e[0] == "task.started" and e[1] is not None}
    assert len(started_tasks) == 2
    assert graph.get_task("t1").status == TaskStatus.COMPLETED
    assert graph.get_task("t2").status == TaskStatus.COMPLETED

@pytest.mark.asyncio
async def test_scheduler_lock_conflict():
    """Prove conflicting tasks block appropriately."""
    event_bus = EventBus()
    registry = AgentRegistry()
    mock_provider = MagicMock(spec=ModelProvider)
    mock_provider.generate.return_value = CompletionResponse(content="Done", model="test")
    
    scheduler = Scheduler(event_bus, registry, mock_provider, [], "/workspace", max_concurrency=2)
    
    graph = TaskGraph()
    t1 = SubTask(task_id="t1", role=AgentRole.FRONTEND, resources=[WorkspaceResource("src/frontend/App.tsx", ResourceMode.WRITE)])
    t2 = SubTask(task_id="t2", role=AgentRole.FRONTEND, resources=[WorkspaceResource("src/frontend/App.tsx", ResourceMode.WRITE)])
    
    graph.add_task(t1)
    graph.add_task(t2)
    
    events = []
    async def handler(e):
        events.append((e.type, e.data.get("task_id")))
    
    event_bus.on("task.started", handler)
    event_bus.on("resource_lock_waiting", handler)
    event_bus.on("resource_lock_released", handler)
    
    await scheduler.execute(graph)
    
    # One of them had to wait!
    wait_events = [e for e in events if e[0] == "resource_lock_waiting"]
    assert len(wait_events) >= 1
    
    # Both eventually completed
    assert graph.get_task("t1").status == TaskStatus.COMPLETED
    assert graph.get_task("t2").status == TaskStatus.COMPLETED
