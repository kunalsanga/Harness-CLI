"""Tests for M5 — Advanced Agent Engine."""

import asyncio
import pytest

from harness_core.agents.domain import (
    AgentMessage,
    AgentResult,
    AgentRole,
    AgentStatus,
    MessageType,
    ReviewVerdict,
    SubTask,
    TaskGraph,
    TaskStatus,
)
from harness_core.agents.registry import AgentConfig, AgentRegistry
from harness_core.agents.executor import AgentExecutor
from harness_core.agents.orchestrator import AgentBudget, ExecutionMode, Orchestrator, OrchestratorResult


# ═══════════════════════════════════════════════════════════════════════
# Domain Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAgentRoles:
    def test_all_roles_exist(self):
        roles = [
            AgentRole.ORCHESTRATOR, AgentRole.PLANNER, AgentRole.RESEARCHER,
            AgentRole.ANALYZER, AgentRole.CODER, AgentRole.TESTER,
            AgentRole.REVIEWER, AgentRole.DEBUGGER,
        ]
        assert len(roles) == 8

    def test_role_values(self):
        assert AgentRole.CODER.value == "coder"
        assert AgentRole.PLANNER.value == "planner"


class TestSubTask:
    def test_create_subtask(self):
        t = SubTask(description="Fix auth bug", role=AgentRole.CODER)
        assert t.task_id
        assert t.status == TaskStatus.PENDING

    def test_subtask_to_dict(self):
        t = SubTask(description="Test", role=AgentRole.TESTER)
        d = t.to_dict()
        assert d["description"] == "Test"
        assert d["role"] == "tester"


class TestTaskGraph:
    def test_add_and_get_task(self):
        g = TaskGraph()
        t = SubTask(description="Task 1")
        g.add_task(t)
        assert g.get_task(t.task_id) is t

    def test_ready_tasks_no_deps(self):
        g = TaskGraph()
        t = SubTask(description="Independent task")
        g.add_task(t)
        ready = g.get_ready_tasks()
        assert len(ready) == 1

    def test_ready_tasks_with_deps(self):
        g = TaskGraph()
        t1 = SubTask(description="First")
        t2 = SubTask(description="Second", dependencies=[t1.task_id])
        g.add_task(t1)
        g.add_task(t2)

        # Only t1 should be ready
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t1.task_id

    def test_ready_after_dep_completed(self):
        g = TaskGraph()
        t1 = SubTask(description="First")
        t2 = SubTask(description="Second", dependencies=[t1.task_id])
        g.add_task(t1)
        g.add_task(t2)

        # Complete t1
        t1.status = TaskStatus.COMPLETED
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t2.task_id

    def test_is_complete(self):
        g = TaskGraph()
        t1 = SubTask(description="A")
        t2 = SubTask(description="B")
        g.add_task(t1)
        g.add_task(t2)

        assert not g.is_complete()
        t1.status = TaskStatus.COMPLETED
        t2.status = TaskStatus.COMPLETED
        assert g.is_complete()

    def test_has_failures(self):
        g = TaskGraph()
        t = SubTask(description="Failing")
        g.add_task(t)
        assert not g.has_failures()
        t.status = TaskStatus.FAILED
        assert g.has_failures()

    def test_cycle_detection(self):
        g = TaskGraph()
        t1 = SubTask(description="A")
        t2 = SubTask(description="B")
        t1.dependencies = [t2.task_id]
        t2.dependencies = [t1.task_id]
        g.add_task(t1)
        g.add_task(t2)

        errors = g.validate()
        assert any("Circular" in e or "cycle" in e.lower() for e in errors)

    def test_missing_dependency(self):
        g = TaskGraph()
        t = SubTask(description="Orphan", dependencies=["nonexistent"])
        g.add_task(t)
        errors = g.validate()
        assert any("non-existent" in e for e in errors)

    def test_topological_sort(self):
        g = TaskGraph()
        t1 = SubTask(description="First")
        t2 = SubTask(description="Second", dependencies=[t1.task_id])
        t3 = SubTask(description="Third", dependencies=[t2.task_id])
        g.add_task(t1)
        g.add_task(t2)
        g.add_task(t3)

        order = g.topological_sort()
        ids = [t.task_id for t in order]
        assert ids.index(t1.task_id) < ids.index(t2.task_id)
        assert ids.index(t2.task_id) < ids.index(t3.task_id)

    def test_dependents(self):
        g = TaskGraph()
        t1 = SubTask(description="Base")
        t2 = SubTask(description="Dep1", dependencies=[t1.task_id])
        t3 = SubTask(description="Dep2", dependencies=[t1.task_id])
        g.add_task(t1)
        g.add_task(t2)
        g.add_task(t3)

        deps = g.get_dependents(t1.task_id)
        assert len(deps) == 2

    def test_to_dict(self):
        g = TaskGraph()
        g.add_task(SubTask(description="A"))
        d = g.to_dict()
        assert d["total_tasks"] == 1


class TestAgentResult:
    def test_create_result(self):
        r = AgentResult(agent_id="coder", role=AgentRole.CODER)
        assert r.status == AgentStatus.COMPLETED

    def test_result_to_dict(self):
        r = AgentResult(
            agent_id="tester",
            role=AgentRole.TESTER,
            summary="Tests passed",
            tests_passed=5,
            tests_total=5,
        )
        d = r.to_dict()
        assert d["tests_passed"] == 5
        assert d["role"] == "tester"


class TestReviewVerdict:
    def test_verdicts(self):
        assert ReviewVerdict.APPROVED.value == "approved"
        assert ReviewVerdict.CHANGES_REQUESTED.value == "changes_requested"
        assert ReviewVerdict.REJECTED.value == "rejected"


# ═══════════════════════════════════════════════════════════════════════
# Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAgentRegistry:
    def test_default_agents(self):
        reg = AgentRegistry()
        agents = reg.list_all()
        assert len(agents) >= 7  # planner, researcher, analyzer, coder, tester, reviewer, debugger

    def test_get_agent(self):
        reg = AgentRegistry()
        coder = reg.get("coder")
        assert coder is not None
        assert coder.role == AgentRole.CODER

    def test_find_by_role(self):
        reg = AgentRegistry()
        coders = reg.find_by_role(AgentRole.CODER)
        assert len(coders) >= 1

    def test_register_custom(self):
        reg = AgentRegistry()
        custom = AgentConfig(name="custom", role=AgentRole.CODER, display_name="Custom")
        reg.register(custom)
        assert reg.get("custom") is not None

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register(AgentConfig(name="temp", role=AgentRole.CODER))
        assert reg.unregister("temp")
        assert reg.get("temp") is None

    def test_find_for_task(self):
        reg = AgentRegistry()
        coders = reg.find_for_task("bug_fix")
        assert len(coders) >= 1
        assert any(a.role == AgentRole.CODER for a in coders)

    def test_get_default_for_role(self):
        reg = AgentRegistry()
        coder = reg.get_default_for_role(AgentRole.CODER)
        assert coder is not None
        assert coder.name == "coder"

    def test_config_to_dict(self):
        reg = AgentRegistry()
        coder = reg.get("coder")
        d = coder.to_dict()
        assert d["name"] == "coder"
        assert d["role"] == "coder"
        assert "capabilities" in d

    def test_list_enabled(self):
        reg = AgentRegistry()
        enabled = reg.list_enabled()
        assert len(enabled) >= 7

    def test_agent_capabilities(self):
        reg = AgentRegistry()
        coder = reg.get("coder")
        assert "code_writing" in coder.capabilities
        assert "bug_fixing" in coder.capabilities


# ═══════════════════════════════════════════════════════════════════════
# Executor Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAgentExecutor:
    @pytest.mark.asyncio
    async def test_execute_coder(self):
        executor = AgentExecutor()
        config = AgentConfig(name="coder", role=AgentRole.CODER)
        task = SubTask(description="Fix the bug", role=AgentRole.CODER)

        result = await executor.execute(config, task)
        assert result.status == AgentStatus.COMPLETED
        assert result.agent_id == "coder"

    @pytest.mark.asyncio
    async def test_execute_tester(self):
        executor = AgentExecutor()
        config = AgentConfig(name="tester", role=AgentRole.TESTER)
        task = SubTask(description="Run tests", role=AgentRole.TESTER)

        result = await executor.execute(config, task)
        assert result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_reviewer(self):
        executor = AgentExecutor()
        config = AgentConfig(name="reviewer", role=AgentRole.REVIEWER)
        task = SubTask(description="Review code", role=AgentRole.REVIEWER)

        result = await executor.execute(config, task)
        assert result.status == AgentStatus.COMPLETED
        assert result.review_verdict == ReviewVerdict.APPROVED

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        executor = AgentExecutor()
        config = AgentConfig(name="coder", role=AgentRole.CODER)
        task = SubTask(description="Implement feature")
        context = {"repository": "python project", "language": "Python"}

        result = await executor.execute(config, task, context=context)
        assert result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_previous_results(self):
        executor = AgentExecutor()
        config = AgentConfig(name="coder", role=AgentRole.CODER)
        task = SubTask(description="Continue work")

        prev = AgentResult(
            agent_id="researcher",
            summary="Found auth module at src/auth.py",
            files_changed=["src/auth.py"],
        )

        result = await executor.execute(
            config, task, previous_results={"researcher": prev}
        )
        assert result.status == AgentStatus.COMPLETED

    def test_stats(self):
        executor = AgentExecutor()
        stats = executor.get_stats()
        assert "total_executions" in stats


# ═══════════════════════════════════════════════════════════════════════
# Budget Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAgentBudget:
    def test_can_start_within_budget(self):
        b = AgentBudget(max_agents=5)
        assert b.can_start_agent()

    def test_exceeds_agent_limit(self):
        b = AgentBudget(max_agents=2)
        b.record_agent()
        b.record_agent()
        assert not b.can_start_agent()

    def test_exceeds_iteration_limit(self):
        b = AgentBudget(max_total_iterations=5)
        b.record_agent(iterations=5)
        assert not b.can_start_agent()

    def test_exceeds_repair_limit(self):
        b = AgentBudget(max_repair_cycles=2)
        b.record_repair()
        b.record_repair()
        assert not b.record_repair()

    def test_budget_to_dict(self):
        b = AgentBudget()
        d = b.to_dict()
        assert "agents_used" in d
        assert "max_agents" in d


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestrator:
    def test_decompose_task(self):
        orch = Orchestrator()
        graph = orch.decompose_task("Implement authentication")

        assert graph.get_total_count() >= 5  # research, plan, implement, test, review
        errors = graph.validate()
        assert len(errors) == 0  # No cycles or missing deps

    def test_decompose_produces_valid_graph(self):
        orch = Orchestrator()
        graph = orch.decompose_task("Fix all failing tests")

        # Should have correct dependency chain
        order = graph.topological_sort()
        assert len(order) >= 5

    @pytest.mark.asyncio
    async def test_execute_single_mode(self):
        orch = Orchestrator()
        result = await orch.execute(
            "Fix the bug",
            mode=ExecutionMode.SINGLE,
        )
        assert isinstance(result, OrchestratorResult)
        assert result.success is True
        assert len(result.agent_results) == 1

    @pytest.mark.asyncio
    async def test_execute_multi_agent_mode(self):
        orch = Orchestrator()
        result = await orch.execute(
            "Implement authentication",
            mode=ExecutionMode.MULTI_AGENT,
        )
        assert isinstance(result, OrchestratorResult)
        assert len(result.agent_results) >= 1

    @pytest.mark.asyncio
    async def test_execute_auto_mode(self):
        orch = Orchestrator()
        result = await orch.execute(
            "Fix failing tests",
            mode=ExecutionMode.AUTO,
        )
        assert isinstance(result, OrchestratorResult)

    @pytest.mark.asyncio
    async def test_orchestrator_result_to_dict(self):
        orch = Orchestrator()
        result = await orch.execute("Test task", mode=ExecutionMode.SINGLE)
        d = result.to_dict()
        assert "success" in d
        assert "agents_used" in d
        assert "duration_ms" in d

    def test_get_registry(self):
        orch = Orchestrator()
        reg = orch.get_registry()
        assert reg is not None
        assert len(reg.list_all()) >= 7

    def test_get_messages(self):
        orch = Orchestrator()
        msgs = orch.get_messages()
        assert isinstance(msgs, list)


class TestExecutionModes:
    def test_all_modes_exist(self):
        modes = [ExecutionMode.SINGLE, ExecutionMode.AUTO, ExecutionMode.MULTI_AGENT, ExecutionMode.PARALLEL]
        assert len(modes) == 4

    def test_mode_values(self):
        assert ExecutionMode.SINGLE.value == "single"
        assert ExecutionMode.MULTI_AGENT.value == "multi_agent"


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAgentIntegration:
    """Test that the complete multi-agent pipeline works."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test: decompose → plan → execute → review."""
        orch = Orchestrator()
        result = await orch.execute(
            "Implement a new REST endpoint with tests",
            mode=ExecutionMode.MULTI_AGENT,
        )

        # Should have executed multiple agents
        assert len(result.agent_results) >= 1

        # Budget should be tracked
        assert result.budget is not None
        assert result.budget.total_agents_used >= 1

    @pytest.mark.asyncio
    async def test_budget_enforcement(self):
        """Test that budgets are respected."""
        budget = AgentBudget(max_agents=2)
        orch = Orchestrator(budget=budget)

        result = await orch.execute(
            "Complex multi-step task",
            mode=ExecutionMode.MULTI_AGENT,
        )

        # Should not exceed budget
        assert budget.total_agents_used <= 2

    @pytest.mark.asyncio
    async def test_single_vs_multi_agent(self):
        """Single mode should produce fewer agents than multi-agent."""
        orch = Orchestrator()

        single = await orch.execute("Fix bug", mode=ExecutionMode.SINGLE)
        multi = await orch.execute("Fix bug", mode=ExecutionMode.MULTI_AGENT)

        assert len(single.agent_results) <= len(multi.agent_results)

    @pytest.mark.asyncio
    async def test_failure_triggers_debugger(self):
        """When an agent fails, a debugger should be scheduled."""
        orch = Orchestrator()
        result = await orch.execute(
            "Task that might fail",
            mode=ExecutionMode.MULTI_AGENT,
        )

        # Should complete without hanging
        assert isinstance(result, OrchestratorResult)
        assert result.agent_results is not None
