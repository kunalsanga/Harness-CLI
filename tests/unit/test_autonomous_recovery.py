"""
Unit tests for the autonomous recovery components (Phase 7).
"""

import pytest

from harness_core.agents.domain import AgentResult, AgentStatus, AgentRole, SubTask, TaskGraph, TaskStatus
from harness_core.agents.registry import AgentRegistry, AgentProfile
from harness_core.recovery.classifier import FailureClassifier, FailureCategory
from harness_core.recovery.planner import RecoveryPlan, RecoveryStrategy, RecoveryPlanner
from harness_core.recovery.validator import RecoveryValidator

@pytest.fixture
def registry():
    r = AgentRegistry()
    r.register(AgentProfile(name="tester", role=AgentRole.TESTER))
    r.register(AgentProfile(name="debugger", role=AgentRole.DEBUGGER))
    r.register(AgentProfile(name="backend", role=AgentRole.BACKEND))
    return r

def test_failure_classifier_test_failure():
    task = SubTask(task_id="t1", role=AgentRole.TESTER)
    result = AgentResult(
        agent_id="test-agent",
        status=AgentStatus.COMPLETED,
        tests_passed=1,
        tests_total=3,
        findings=[{"type": "test_failure", "test_name": "test_api_auth"}]
    )
    
    classification = FailureClassifier.classify(task, result)
    assert classification is not None
    assert classification.category == FailureCategory.TEST_FAILURE
    assert "2 tests failed" in classification.summary
    assert "test_api_auth" in classification.failed_tests

def test_failure_classifier_timeout():
    task = SubTask(task_id="t1")
    result = AgentResult(
        status=AgentStatus.FAILED,
        errors=["Timeout: agent exceeded time limit"]
    )
    
    classification = FailureClassifier.classify(task, result)
    assert classification is not None
    assert classification.category == FailureCategory.TIMEOUT

def test_failure_classifier_no_failure():
    task = SubTask(task_id="t1")
    result = AgentResult(
        status=AgentStatus.COMPLETED,
        tests_passed=3,
        tests_total=3
    )
    
    classification = FailureClassifier.classify(task, result)
    assert classification is None

def test_recovery_validator(registry):
    validator = RecoveryValidator(registry)
    
    # Valid plan
    plan = RecoveryPlan(
        strategy=RecoveryStrategy.DEBUG_AND_FIX,
        specialist_role=AgentRole.DEBUGGER,
        target_task_id="t1",
        objective="Fix tests"
    )
    errors = validator.validate(plan)
    assert len(errors) == 0
    
    # Invalid role
    registry.unregister(AgentRole.SECURITY_REVIEWER.value) # Unregister the default one
    plan.specialist_role = AgentRole.SECURITY_REVIEWER # now it's missing in registry
    errors = validator.validate(plan)
    assert len(errors) > 0
    assert "No registered agent" in errors[0]

def test_task_graph_insert_recovery_sequence():
    graph = TaskGraph()
    t1 = SubTask(task_id="t1", status=TaskStatus.FAILED)
    t2 = SubTask(task_id="t2", dependencies=["t1"], status=TaskStatus.BLOCKED)
    t3 = SubTask(task_id="t3", dependencies=["t2"], status=TaskStatus.BLOCKED)
    
    graph.add_task(t1)
    graph.add_task(t2)
    graph.add_task(t3)
    
    debug = SubTask(task_id="debug_1", role=AgentRole.DEBUGGER)
    retest = SubTask(task_id="retest_1", role=AgentRole.TESTER)
    
    graph.insert_recovery_sequence("t1", [debug, retest])
    
    # Original t1 is FAILED
    assert graph.get_task("t1").status == TaskStatus.FAILED
    
    # t2 now depends on retest_1
    assert "retest_1" in graph.get_task("t2").dependencies
    assert "t1" not in graph.get_task("t2").dependencies
    
    # t2 should be unblocked (CREATED)
    assert graph.get_task("t2").status == TaskStatus.CREATED
    assert graph.get_task("t3").status == TaskStatus.CREATED
    
    # Debug task should have t1's dependencies (none)
    assert len(graph.get_task("debug_1").dependencies) == 0
    
    # Retest task depends on Debug
    assert "debug_1" in graph.get_task("retest_1").dependencies
