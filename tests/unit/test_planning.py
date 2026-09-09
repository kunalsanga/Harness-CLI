import pytest
import json
from unittest.mock import MagicMock

from harness_core.planning.domain import Plan, PlannedTask, PlanningResult
from harness_core.planning.planner import Planner
from harness_core.planning.validator import PlanValidator
from harness_core.agents.registry import AgentRegistry, AgentProfile
from harness_core.agents.domain import AgentRole, WorkspaceScope, TaskGraph, TaskStatus
from harness_core.observability.events import EventBus
from harness_core.providers.base import ModelProvider, CompletionResponse


@pytest.fixture
def registry():
    return AgentRegistry()

@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=ModelProvider)
    return provider


def test_validator_valid_plan(registry):
    validator = PlanValidator(registry)
    plan = Plan(tasks=[
        PlannedTask(task_id="t1", title="T1", objective="O1", role="architect", dependencies=[]),
        PlannedTask(task_id="t2", title="T2", objective="O2", role="coder", dependencies=["t1"]),
    ])
    errors = validator.validate(plan)
    assert not errors, f"Should be no errors, got {errors}"


def test_validator_cycle(registry):
    validator = PlanValidator(registry)
    plan = Plan(tasks=[
        PlannedTask(task_id="t1", title="T1", objective="O1", role="architect", dependencies=["t2"]),
        PlannedTask(task_id="t2", title="T2", objective="O2", role="coder", dependencies=["t1"]),
    ])
    errors = validator.validate(plan)
    assert any("cycle" in e.lower() for e in errors), "Should detect cycle"


def test_validator_unknown_role(registry):
    validator = PlanValidator(registry)
    plan = Plan(tasks=[
        PlannedTask(task_id="t1", title="T1", objective="O1", role="fake_role", dependencies=[]),
    ])
    errors = validator.validate(plan)
    assert any("unknown role" in e.lower() or "no registered" in e.lower() for e in errors), "Should detect unknown role"


def test_validator_duplicate_id(registry):
    validator = PlanValidator(registry)
    plan = Plan(tasks=[
        PlannedTask(task_id="t1", title="T1", objective="O1", role="architect", dependencies=[]),
        PlannedTask(task_id="t1", title="T1_dup", objective="O2", role="coder", dependencies=[]),
    ])
    errors = validator.validate(plan)
    assert any("duplicate task id" in e.lower() for e in errors), "Should detect duplicate ID"


def test_validator_missing_dependency(registry):
    validator = PlanValidator(registry)
    plan = Plan(tasks=[
        PlannedTask(task_id="t1", title="T1", objective="O1", role="architect", dependencies=["missing_task"]),
    ])
    errors = validator.validate(plan)
    assert any("unknown task" in e.lower() for e in errors), "Should detect missing dependency"


def test_validator_self_dependency(registry):
    validator = PlanValidator(registry)
    plan = Plan(tasks=[
        PlannedTask(task_id="t1", title="T1", objective="O1", role="architect", dependencies=["t1"]),
    ])
    errors = validator.validate(plan)
    assert any("depend on itself" in e.lower() for e in errors), "Should detect self dependency"


@pytest.mark.asyncio
async def test_planner_valid_json(mock_provider, registry):
    # Mock LLM returning valid JSON
    valid_json = json.dumps({
        "summary": "Mock Plan",
        "tasks": [
            {
                "task_id": "arch_1",
                "title": "Arch",
                "objective": "Do arch",
                "role": "architect",
                "dependencies": [],
                "workspace_scope": "project"
            }
        ]
    })
    # Wrap in markdown just to test planner's resilience
    mock_provider.generate.return_value = CompletionResponse(content=f"```json\n{valid_json}\n```", model="test")
    
    planner = Planner(mock_provider, registry)
    result = await planner.plan("Do something")
    
    assert result.success
    assert result.plan is not None
    assert len(result.plan.tasks) == 1
    assert result.plan.tasks[0].task_id == "arch_1"


@pytest.mark.asyncio
async def test_planner_invalid_json(mock_provider, registry):
    mock_provider.generate.return_value = CompletionResponse(content="This is not json", model="test")
    
    planner = Planner(mock_provider, registry)
    result = await planner.plan("Do something")
    
    assert not result.success
    assert result.plan is None
    assert any("parse" in e.lower() for e in result.errors)


@pytest.mark.asyncio
async def test_orchestrator_integration_valid(mock_provider, registry):
    from harness_core.agents.orchestrator import Orchestrator, ExecutionMode
    
    valid_json = json.dumps({
        "summary": "Full Stack App Plan",
        "tasks": [
            {
                "task_id": "arch",
                "title": "Architecture",
                "objective": "Design system",
                "role": "architect",
                "dependencies": []
            },
            {
                "task_id": "db",
                "title": "Database",
                "objective": "Design DB",
                "role": "database",
                "dependencies": ["arch"]
            },
            {
                "task_id": "backend",
                "title": "Backend",
                "objective": "Build API",
                "role": "backend",
                "dependencies": ["db"]
            },
            {
                "task_id": "frontend",
                "title": "Frontend",
                "objective": "Build UI",
                "role": "frontend",
                "dependencies": ["arch"]
            }
        ]
    })
    mock_provider.generate.return_value = CompletionResponse(content=valid_json, model="test")
    
    orchestrator = Orchestrator(provider=mock_provider, registry=registry)
    
    # We just want to see it decompose and create a graph properly, without actually executing
    graph, errors = await orchestrator.decompose_task("Build a full-stack app")
    
    assert not errors
    assert graph is not None
    assert graph.get_total_count() == 4
    
    arch_task = graph.get_task("arch")
    db_task = graph.get_task("db")
    frontend_task = graph.get_task("frontend")
    
    assert arch_task.role == AgentRole.ARCHITECT
    assert db_task.role == AgentRole.DATABASE
    assert "arch" in db_task.dependencies
    
    # Check that independent tasks (db and frontend) can execute concurrently once arch is done
    arch_task.status = TaskStatus.COMPLETED
    ready = graph.get_ready_tasks()
    
    ready_ids = {t.task_id for t in ready}
    assert "db" in ready_ids
    assert "frontend" in ready_ids
    
    # Backend shouldn't be ready yet
    assert "backend" not in ready_ids

@pytest.mark.asyncio
async def test_orchestrator_integration_invalid(mock_provider, registry):
    from harness_core.agents.orchestrator import Orchestrator, ExecutionMode
    
    invalid_json = json.dumps({
        "summary": "Bad Plan",
        "tasks": [
            {
                "task_id": "arch",
                "title": "Architecture",
                "objective": "Design system",
                "role": "fake_role_not_exist",
                "dependencies": []
            }
        ]
    })
    mock_provider.generate.return_value = CompletionResponse(content=invalid_json, model="test")
    
    orchestrator = Orchestrator(provider=mock_provider, registry=registry)
    graph, errors = await orchestrator.decompose_task("Build a full-stack app")
    
    assert graph is None
    assert errors
    assert any("fake_role_not_exist" in e for e in errors)
