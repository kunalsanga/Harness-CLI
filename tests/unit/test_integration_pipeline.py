"""Integration tests for the full agent pipeline (M3.6 Phase 1).

Proves that AgentLoop, TaskClassifier, ModelRegistry, TaskAwareRouter,
ContextEngine, and EventBus work together coherently.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, AgentRole, TaskStatus
from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.classifier.types import TaskRequirementProfile
from harness_core.models.registry import ModelRegistry
from harness_core.models.types import CapabilityConfidence, ModelProfile
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.routing.router import ModelRouter, RouterConfig
from harness_core.routing.task_aware import TaskAwareRouter
from harness_core.tools.base import Tool, ToolSchema
from harness_core.agent.types import ToolResult, ToolResultStatus


# ── Mock Tools ────────────────────────────────────────────────────────────────


class MockTool(Tool):
    """A mock tool for testing."""

    def __init__(self, name: str = "mock_tool", response: str = "ok") -> None:
        self._name = name
        self._response = response

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Mock tool: {self._name}",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(status=ToolResultStatus.SUCCESS, output=self._response)


# ── Mock Provider ─────────────────────────────────────────────────────────────


class MockProvider(ModelProvider):
    """A mock provider for testing."""

    def __init__(self) -> None:
        self.call_count = 0
        self._responses: list[CompletionResponse] = []

    def add_response(self, response: CompletionResponse) -> None:
        self._responses.append(response)

    @property
    def name(self) -> str:
        return "mock_provider"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        # Default: return a completion with no tool calls
        return CompletionResponse(
            content="Task completed successfully.",
            model="mock-model",
            provider="mock",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

    async def stream(self, request: CompletionRequest):
        yield ""

    async def list_models(self) -> list:
        return []

    async def health_check(self) -> bool:
        return True


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTaskClassifierIntegration:
    """Test that TaskClassifier works with real task descriptions."""

    def test_classify_bug_fix(self):
        c = TaskClassifier()
        task_type, confidence = c.classify_with_confidence(
            "Fix the authentication bug and make all tests pass"
        )
        assert task_type == TaskType.BUG_FIX
        assert confidence > 0.5

    def test_classify_implementation(self):
        c = TaskClassifier()
        task_type, _ = c.classify_with_confidence(
            "Implement a new REST API endpoint for user management"
        )
        assert task_type == TaskType.IMPLEMENTATION

    def test_classify_refactoring(self):
        c = TaskClassifier()
        task_type, _ = c.classify_with_confidence(
            "Refactor the database module to use connection pooling"
        )
        assert task_type == TaskType.REFACTORING

    def test_classify_testing(self):
        c = TaskClassifier()
        task_type, _ = c.classify_with_confidence(
            "Write unit tests for the authentication middleware"
        )
        assert task_type == TaskType.TESTING

    def test_classify_research(self):
        c = TaskClassifier()
        task_type, _ = c.classify_with_confidence(
            "Explain how the caching system works in this codebase"
        )
        assert task_type == TaskType.RESEARCH

    def test_get_requirement_profile(self):
        c = TaskClassifier()
        profile = c.get_profile("Fix the authentication bug")
        assert profile.task_type == "bug_fix"
        assert profile.coding is not None
        assert profile.tool_use is not None
        assert profile.verification is not None


class TestTaskAwareRouterIntegration:
    """Test that TaskAwareRouter bridges classifier, registry, and history."""

    def test_classify_from_request(self):
        router = TaskAwareRouter()
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Fix the authentication bug and make all tests pass"}]
        )
        task_type, profile, confidence = router.classify_task(request)
        assert task_type == TaskType.BUG_FIX
        assert isinstance(profile, TaskRequirementProfile)
        assert confidence > 0.5

    def test_score_model_with_registry_data(self):
        registry = ModelRegistry()
        task_aware = TaskAwareRouter(registry=registry)

        # Register a strong coding model
        p = ModelProfile(model_id="coder-33b", provider="openrouter")
        p.capabilities.coding.score = 0.95
        p.capabilities.coding.confidence = CapabilityConfidence.BENCHMARKED
        p.capabilities.tool_use.score = 0.90
        p.capabilities.tool_use.confidence = CapabilityConfidence.BENCHMARKED
        p.capabilities.verification.score = 0.85
        p.capabilities.verification.confidence = CapabilityConfidence.BENCHMARKED
        registry.register(p)

        # Register a weak model
        p2 = ModelProfile(model_id="small-8b", provider="openrouter")
        p2.capabilities.coding.score = 0.40
        p2.capabilities.coding.confidence = CapabilityConfidence.DECLARED
        registry.register(p2)

        # Classify a bug fix task
        classifier = TaskClassifier()
        task_type = classifier.classify("Fix the authentication bug")
        profile = classifier.get_profile("Fix the authentication bug")

        # Score both models
        score_strong, _ = task_aware.score_model_for_task("coder-33b", task_type, profile)
        score_weak, _ = task_aware.score_model_for_task("small-8b", task_type, profile)

        assert score_strong > score_weak
        assert score_strong > 0.5

    def test_rank_models_for_task(self):
        registry = ModelRegistry()
        task_aware = TaskAwareRouter(registry=registry)

        # Register models with different capabilities
        for i, (coding, tool_use) in enumerate([
            (0.95, 0.90), (0.60, 0.70), (0.80, 0.85),
        ]):
            p = ModelProfile(model_id=f"model-{i}", provider="test")
            p.capabilities.coding.score = coding
            p.capabilities.tool_use.score = tool_use
            registry.register(p)

        classifier = TaskClassifier()
        task_type = classifier.classify("Fix the authentication bug and make tests pass")
        profile = classifier.get_profile("Fix the authentication bug")

        ranked = task_aware.rank_models_for_task(
            ["model-0", "model-1", "model-2"], task_type, profile
        )

        # model-0 (0.95 coding) should be ranked first
        assert ranked[0][0] == "model-0"
        assert ranked[0][1] > ranked[1][1]

    def test_record_task_result(self, tmp_path):
        db = tmp_path / "test_history.db"
        from harness_core.models.history import PerformanceHistory
        history = PerformanceHistory(db)
        task_aware = TaskAwareRouter(history=history)

        task_aware.record_task_result(
            model_id="test-model",
            provider="test",
            task_type="bug_fix",
            success=True,
            latency_ms=1500,
            tool_calls=8,
            iterations=3,
        )

        perf = history.get_performance("test-model")
        assert perf.total_tasks == 1
        assert perf.success_count == 1

    def test_get_model_summary(self):
        registry = ModelRegistry()
        task_aware = TaskAwareRouter(registry=registry)

        p = ModelProfile(model_id="test-model", provider="test", is_free=True)
        p.capabilities.coding.score = 0.9
        registry.register(p)

        summary = task_aware.get_model_summary("test-model")
        assert summary["found"] is True
        assert summary["provider"] == "test"
        assert summary["is_free"] is True
        assert "capabilities" in summary
        assert summary["capabilities"]["coding"]["score"] == 0.9

    def test_get_model_summary_unknown(self):
        task_aware = TaskAwareRouter()
        summary = task_aware.get_model_summary("nonexistent")
        assert summary["found"] is False


class TestAgentLoopIntegration:
    """Test that AgentLoop works with TaskAwareRouter."""

    def test_agent_loop_with_task_aware(self):
        provider = MockProvider()
        tools = [MockTool("list_files", "file1.py\nfile2.py")]
        event_bus = EventBus()
        task_aware = TaskAwareRouter()

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=AgentConfig(max_iterations=2),
            event_bus=event_bus,
            task_aware=task_aware,
        )

        # Verify task_aware is wired
        assert agent.task_aware is task_aware

    @pytest.mark.asyncio
    async def test_agent_loop_emits_task_classified(self):
        provider = MockProvider()
        tools: list[Tool] = []
        event_bus = EventBus()
        task_aware = TaskAwareRouter()

        events_received: list[Event] = []

        async def capture_event(event: Event) -> None:
            events_received.append(event)

        event_bus.on("*", capture_event)

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=AgentConfig(max_iterations=1),
            event_bus=event_bus,
            task_aware=task_aware,
        )

        result = await agent.run("Fix the authentication bug")

        # Should have emitted task.classified
        classified_events = [e for e in events_received if e.type == "task.classified"]
        assert len(classified_events) == 1
        assert classified_events[0].data["task_type"] == "bug_fix"

    @pytest.mark.asyncio
    async def test_agent_loop_records_performance(self, tmp_path):
        from harness_core.models.history import PerformanceHistory

        db = tmp_path / "test_perf.db"
        history = PerformanceHistory(db)
        task_aware = TaskAwareRouter(history=history)

        # Create a mock router with a routing decision
        router = MagicMock()
        router.budget = MagicMock()
        router.budget.check_all.return_value = (True, None)
        router.get_routing_decisions.return_value = [
            MagicMock(selected_model="test-model", selected_provider="test")
        ]

        provider = MockProvider()
        tools: list[Tool] = []
        event_bus = EventBus()

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=AgentConfig(max_iterations=1),
            event_bus=event_bus,
            router=router,
            task_aware=task_aware,
        )

        result = await agent.run("Fix the authentication bug")

        # Performance should be recorded
        perf = history.get_performance("test-model")
        assert perf.total_tasks == 1


class TestContextEngineIntegration:
    """Test that ContextEngine discovers projects correctly."""

    @pytest.mark.asyncio
    async def test_discover_python_project(self, tmp_path):
        # Create a minimal Python project
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text('print("hello")\n')
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text('assert True\n')
        (tmp_path / ".git").mkdir()

        from harness_core.context.engine import ContextEngine
        engine = ContextEngine(tmp_path)
        info = await engine.discover_project()

        assert info["has_git"] is True
        assert info["has_tests"] is True
        assert "python" in info["languages"]
        assert len(info["files"]) >= 3

    @pytest.mark.asyncio
    async def test_assemble_context_respects_budget(self, tmp_path):
        from harness_core.context.engine import ContextEngine, ContextBudget

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text('print("hello")\n')

        engine = ContextEngine(tmp_path)
        info = await engine.discover_project()

        # Very small budget
        budget = ContextBudget(model_context_limit=500, reserved_output_tokens=100)
        context = await engine.assemble_context("test task", info, budget)

        # Should fit within budget
        total_tokens = sum(p.tokens_estimate for p in context)
        assert total_tokens <= budget.available_tokens


class TestEventBusIntegration:
    """Test that EventBus connects components."""

    @pytest.mark.asyncio
    async def test_event_flow(self):
        event_bus = EventBus()
        events: list[Event] = []

        async def handler(event: Event) -> None:
            events.append(event)

        # EventBus supports "*" for all events
        event_bus.on("*", handler)

        await event_bus.emit(Event(type="task.started", source="test", data={"goal": "test"}))
        await event_bus.emit(Event(type="task.classified", source="test", data={"task_type": "bug_fix"}))
        await event_bus.emit(Event(type="task.completed", source="test", data={"status": "completed"}))

        assert len(events) == 3
        assert events[0].type == "task.started"
        assert events[1].type == "task.classified"
        assert events[2].type == "task.completed"


class TestEndToEndPipeline:
    """End-to-end test: classify → route → execute → complete."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Simulate the full pipeline with mocked components."""
        # 1. Classify the task
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Fix the authentication bug and make all tests pass"
        )
        assert task_type == TaskType.BUG_FIX

        # 2. Get requirements
        profile = classifier.get_profile("Fix the authentication bug")
        assert profile.coding is not None
        assert profile.tool_use is not None
        assert profile.verification is not None

        # 3. Set up registry with models
        registry = ModelRegistry()
        strong = ModelProfile(model_id="strong-model", provider="test")
        strong.capabilities.coding.score = 0.95
        strong.capabilities.tool_use.score = 0.90
        strong.capabilities.verification.score = 0.85
        registry.register(strong)

        weak = ModelProfile(model_id="weak-model", provider="test")
        weak.capabilities.coding.score = 0.40
        weak.capabilities.tool_use.score = 0.50
        registry.register(weak)

        # 4. Rank models
        task_aware = TaskAwareRouter(registry=registry)
        ranked = task_aware.rank_models_for_task(
            ["strong-model", "weak-model"], task_type, profile
        )
        assert ranked[0][0] == "strong-model"

        # 5. Run agent loop (mocked provider)
        provider = MockProvider()
        tools: list[Tool] = [MockTool("list_files", "src/auth.py\ntests/test_auth.py")]
        event_bus = EventBus()
        router = MagicMock()
        router.budget = MagicMock()
        router.budget.check_all.return_value = (True, None)
        router.execute = AsyncMock(return_value=MagicMock(
            succeeded=True,
            response=CompletionResponse(
                content="I've identified the bug and fixed it.",
                model="strong-model",
                provider="test",
                usage={"prompt_tokens": 500, "completion_tokens": 200},
            ),
        ))
        router.get_routing_decisions.return_value = [
            MagicMock(selected_model="strong-model", selected_provider="test")
        ]

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=AgentConfig(max_iterations=5),
            event_bus=event_bus,
            router=router,
            task_aware=task_aware,
        )

        result = await agent.run("Fix the authentication bug and make all tests pass")

        # Verify pipeline completed
        assert result.status == TaskStatus.COMPLETED
        assert result.result is not None
        assert result.iterations == 1
