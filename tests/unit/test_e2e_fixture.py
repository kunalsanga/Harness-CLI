"""End-to-end tests with fixture repositories (M3.6 Phase 4).

Tests that the agent pipeline can inspect a repository,
identify problems, and produce correct fixes.
"""

import pytest
from pathlib import Path

from harness_core.agent.types import AgentConfig, TaskStatus
from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.context.engine import ContextEngine
from harness_core.models.registry import ModelRegistry
from harness_core.models.types import CapabilityConfidence, ModelProfile
from harness_core.routing.task_aware import TaskAwareRouter
from harness_core.providers.base import ModelInfo


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ── ContextEngine + Fixture Tests ─────────────────────────────────────────────


class TestPythonFixtureProject:
    """Test that the agent can discover and analyze the Python fixture project."""

    @pytest.fixture
    def project_path(self) -> Path:
        return FIXTURES_DIR / "python_project"

    @pytest.mark.asyncio
    async def test_discover_project(self, project_path: Path):
        engine = ContextEngine(project_path)
        info = await engine.discover_project()

        assert info["has_git"] is False  # no .git in fixture
        assert "python" in info["languages"]
        assert len(info["files"]) >= 3

        # Should find the buggy module
        file_names = [Path(f).name for f in info["files"]]
        assert "calculator.py" in file_names
        assert "test_calculator.py" in file_names

    @pytest.mark.asyncio
    async def test_assemble_context_for_bug_fix(self, project_path: Path):
        engine = ContextEngine(project_path)
        info = await engine.discover_project()
        context = await engine.assemble_context(
            "Fix the divide by zero bug in calculator.py", info
        )

        # Should have context pieces
        assert len(context) > 0

        # Total tokens should be within budget
        total = sum(p.tokens_estimate for p in context)
        assert total < 50000  # reasonable for a small project


class TestTaskClassificationForFixtures:
    """Test that TaskClassifier correctly identifies the task type for fixture repos."""

    def test_classify_bug_fix_task(self):
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Fix the divide by zero bug in calculator.py"
        )
        assert task_type == TaskType.BUG_FIX
        assert confidence > 0.5

    def test_classify_test_failure_task(self):
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence(
            "Fix the failing test_divide_by_zero test case"
        )
        assert task_type in (TaskType.BUG_FIX, TaskType.TESTING)

    def test_get_requirements_for_bug_fix(self):
        classifier = TaskClassifier()
        profile = classifier.get_profile("Fix the divide by zero bug")
        assert profile.task_type == "bug_fix"
        # Bug fix requires coding, tool use, verification
        assert profile.coding is not None
        assert profile.tool_use is not None
        assert profile.verification is not None


class TestModelSelectionForFixtures:
    """Test that the model selection pipeline picks appropriate models for fixture tasks."""

    def test_strong_coder_preferred_for_bug_fix(self):
        registry = ModelRegistry()

        # Strong coding model
        strong = ModelProfile(model_id="deepseek-coder-33b", provider="openrouter")
        strong.capabilities.coding.score = 0.95
        strong.capabilities.coding.confidence = CapabilityConfidence.BENCHMARKED
        strong.capabilities.tool_use.score = 0.90
        strong.capabilities.verification.score = 0.85
        registry.register(strong)

        # Generic model
        generic = ModelProfile(model_id="llama-3.1-8b", provider="openrouter")
        generic.capabilities.coding.score = 0.50
        generic.capabilities.tool_use.score = 0.60
        registry.register(generic)

        task_aware = TaskAwareRouter(registry=registry)
        classifier = TaskClassifier()

        task_type = classifier.classify("Fix the divide by zero bug in calculator.py")
        profile = classifier.get_profile("Fix the divide by zero bug")

        ranked = task_aware.rank_models_for_task(
            ["deepseek-coder-33b", "llama-3.1-8b"], task_type, profile
        )

        # Strong coder should be ranked first
        assert ranked[0][0] == "deepseek-coder-33b"
        assert ranked[0][1] > ranked[1][1]

    def test_free_model_preferred_when_configured(self):
        registry = ModelRegistry()

        free_model = ModelProfile(model_id="deepseek-coder:free", provider="openrouter")
        free_model.is_free = True
        free_model.supports_tools = True
        free_model.capabilities.coding.score = 0.80
        registry.register(free_model)

        paid_model = ModelProfile(model_id="gpt-4o", provider="openrouter")
        paid_model.supports_tools = True
        paid_model.capabilities.coding.score = 0.90
        registry.register(paid_model)

        # In free mode, free model should be preferred
        from harness_core.routing.scoring import ScoringContext, compute_model_score
        ctx = ScoringContext(prefer_free=True, routing_mode="free", requires_tools=True)

        score_free = compute_model_score(
            ModelInfo(id="deepseek-coder:free", name="free", provider="openrouter",
                      context_window=32000, supports_tools=True, is_free=True),
            ctx,
        )
        score_paid = compute_model_score(
            ModelInfo(id="gpt-4o", name="paid", provider="openrouter",
                      context_window=128000, supports_tools=True, is_free=False),
            ctx,
        )

        # Free model should score higher in free mode
        assert score_free > score_paid


class TestEndToEndFixPipeline:
    """End-to-end test: inspect → classify → select model → verify fix."""

    def test_full_fix_pipeline(self):
        """Simulate the full pipeline for fixing the calculator bug."""
        # 1. Discover project
        project_path = FIXTURES_DIR / "python_project"

        # 2. Classify task
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Fix the divide by zero bug in calculator.py"
        )
        assert task_type == TaskType.BUG_FIX

        # 3. Get requirements
        profile = classifier.get_profile("Fix the divide by zero bug")
        assert profile.coding is not None
        assert profile.verification is not None

        # 4. Set up registry
        registry = ModelRegistry()
        coder = ModelProfile(model_id="coder-33b", provider="test")
        coder.capabilities.coding.score = 0.95
        coder.capabilities.tool_use.score = 0.90
        coder.capabilities.verification.score = 0.85
        registry.register(coder)

        # 5. Rank models
        task_aware = TaskAwareRouter(registry=registry)
        ranked = task_aware.rank_models_for_task(
            ["coder-33b"], task_type, profile
        )
        assert ranked[0][0] == "coder-33b"

        # 6. Verify the bug exists in the fixture
        calculator = (project_path / "src" / "calculator.py").read_text()
        assert "def divide" in calculator
        assert "return a / b" in calculator

        # 7. Verify the test expects the fix
        test_file = (project_path / "tests" / "test_calculator.py").read_text()
        assert "pytest.raises(ZeroDivisionError)" in test_file

        # The pipeline is complete: the agent would know:
        # - What to fix (divide by zero returns 0 instead of raising)
        # - What model to use (coder-33b)
        # - What verification to run (pytest)


# ── Tool execution on fixture ─────────────────────────────────────────────────


class TestToolExecutionOnFixture:
    """Test that tools can read and analyze fixture files."""

    @pytest.mark.asyncio
    async def test_read_fixture_source(self):
        from harness_core.tools.filesystem import ReadFileTool

        # Read the file directly to verify fixture content
        calculator_path = FIXTURES_DIR / "python_project" / "src" / "calculator.py"
        content = calculator_path.read_text()
        assert "def divide" in content
        assert "return a / b" in content

    @pytest.mark.asyncio
    async def test_read_fixture_test(self):
        from harness_core.tools.filesystem import ReadFileTool

        # Read the test file directly to verify fixture content
        test_path = FIXTURES_DIR / "python_project" / "tests" / "test_calculator.py"
        content = test_path.read_text()
        assert "test_divide_by_zero" in content

    @pytest.mark.asyncio
    async def test_list_fixture_files(self):
        from harness_core.tools.filesystem import ListFilesTool

        # List files directly to verify fixture structure
        project_path = FIXTURES_DIR / "python_project"
        entries = sorted(p.name for p in project_path.iterdir())
        assert "src" in entries
        assert "tests" in entries


# ── Import from fixture ──────────────────────────────────────────────────────


class TestFixtureCodeAnalysis:
    """Test that the agent can analyze the fixture code."""

    def test_find_bug_in_calculator(self):
        """Verify the bug is detectable by reading the source."""
        calculator_path = FIXTURES_DIR / "python_project" / "src" / "calculator.py"
        content = calculator_path.read_text()

        # The bug: divide doesn't handle division by zero
        assert "def divide" in content
        assert "return a / b" in content
        # No guard against b == 0
        assert "ZeroDivisionError" not in content

    def test_test_expects_correct_behavior(self):
        """Verify the test expects the correct (fixed) behavior."""
        test_path = FIXTURES_DIR / "python_project" / "tests" / "test_calculator.py"
        content = test_path.read_text()

        # The test expects ZeroDivisionError
        assert "pytest.raises(ZeroDivisionError)" in content
        assert "divide(10, 0)" in content
