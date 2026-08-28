"""E2E integration tests — verify the full agent pipeline with real components."""

import asyncio
from pathlib import Path

import pytest

from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.models.registry import ModelRegistry
from harness_core.routing.task_aware import TaskAwareRouter
from harness_core.session.session import RunRecord, Session, SessionState
from harness_core.agent.recovery import classify_error, RecoveryStrategy


class TestTaskClassificationPipeline:
    """Verify task classification feeds into routing."""

    def test_bug_fix_classification(self):
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Fix the failing test case that crashes"
        )
        assert task_type in (TaskType.BUG_FIX, TaskType.DEBUGGING, TaskType.TESTING)
        assert confidence > 0.3

    def test_implementation_classification(self):
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Implement a new function for user authentication"
        )
        assert task_type in (TaskType.IMPLEMENTATION, TaskType.BUG_FIX)
        assert confidence > 0.2

    def test_refactoring_classification(self):
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Refactor the authentication module without changing behavior"
        )
        assert task_type in (TaskType.REFACTORING, TaskType.IMPLEMENTATION)
        assert confidence > 0.2

    def test_research_classification(self):
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(
            "Explain how the caching system works"
        )
        assert task_type in (TaskType.RESEARCH, TaskType.DOCUMENTATION)
        assert confidence > 0.2

    def test_profile_computes_fit(self):
        classifier = TaskClassifier()
        profile = classifier.get_profile("Fix the authentication bug that crashes")
        requirements = profile.get_requirements()
        assert isinstance(requirements, dict)
        assert len(requirements) > 0
        # Bug fix requires coding, tool_use, verification
        assert requirements.get("coding", 0) > 0.5
        assert requirements.get("verification", 0) > 0.5


class TestTaskAwareRouter:
    """Verify TaskAwareRouter works with real models."""

    def test_router_scoring_no_models(self):
        registry = ModelRegistry()
        router = TaskAwareRouter(registry=registry)
        classifier = TaskClassifier()

        task_type, _ = classifier.classify_with_confidence("Fix failing tests")
        profile = classifier.get_profile("Fix failing tests")

        ranked = router.rank_models_for_task([], task_type, profile)
        assert ranked == []

    def test_router_with_model(self):
        from harness_core.models.types import (
            CapabilityConfidence,
            CapabilityProfile,
            CapabilityScore,
            CapabilitySource,
            ModelProfile,
        )

        registry = ModelRegistry()
        caps = CapabilityProfile(
            coding=CapabilityScore(0.9, CapabilityConfidence.DECLARED, CapabilitySource.USER_DECLARED),
            tool_use=CapabilityScore(0.8, CapabilityConfidence.DECLARED, CapabilitySource.USER_DECLARED),
            reasoning=CapabilityScore(0.85, CapabilityConfidence.DECLARED, CapabilitySource.USER_DECLARED),
        )
        model = ModelProfile(
            model_id="test/model-v1",
            provider="test",
            display_name="Test Model",
            context_window=128000,
            supports_tools=True,
            capabilities=caps,
        )
        registry.register(model)

        router = TaskAwareRouter(registry=registry)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Fix failing tests")
        profile = classifier.get_profile("Fix failing tests")

        ranked = router.rank_models_for_task(
            ["test/model-v1"], task_type, profile
        )
        assert len(ranked) == 1
        model_id, score = ranked[0]
        assert model_id == "test/model-v1"
        assert score > 0

    def test_router_model_summary(self):
        from harness_core.models.types import ModelProfile

        registry = ModelRegistry()
        model = ModelProfile(
            model_id="test/model-v2",
            provider="test",
            display_name="Test Model V2",
            context_window=64000,
            supports_tools=True,
        )
        registry.register(model)

        router = TaskAwareRouter(registry=registry)
        summary = router.get_model_summary("test/model-v2")
        assert summary["found"] is True
        assert summary["model_id"] == "test/model-v2"
        assert summary["provider"] == "test"
        assert "capabilities" in summary


class TestRecoverySystem:
    """Verify error classification works correctly."""

    def test_rate_limit_recovery(self):
        decision = classify_error(Exception("HTTP 429 Too Many Requests"))
        assert decision.strategy == RecoveryStrategy.FALLBACK_MODEL

    def test_timeout_recovery(self):
        decision = classify_error(Exception("Request timed out"))
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_auth_failure_recovery(self):
        decision = classify_error(Exception("HTTP 401 Unauthorized"))
        assert decision.strategy == RecoveryStrategy.ABORT

    def test_permission_recovery(self):
        decision = classify_error(Exception("Permission denied"))
        assert decision.strategy == RecoveryStrategy.SKIP_TOOL

    def test_network_recovery(self):
        decision = classify_error(Exception("Connection refused"))
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_unknown_error_recovery(self):
        decision = classify_error(Exception("Something unexpected"))
        assert decision.strategy == RecoveryStrategy.RETRY


class TestSessionPersistence:
    """Verify sessions are persisted correctly."""

    def test_create_and_retrieve_session(self, tmp_path):
        db_path = tmp_path / "test_sessions.db"
        session = Session(db_path)

        sid = session.create_session("/test/project")
        assert len(sid) > 0

        sessions = session.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sid

    def test_record_and_retrieve_runs(self, tmp_path):
        db_path = tmp_path / "test_sessions.db"
        session = Session(db_path)

        sid = session.create_session()
        run = RunRecord(
            session_id=sid,
            goal="Fix bug",
            status="completed",
            model_used="test-model",
            iterations=3,
            tool_calls=5,
            started_at=1.0,
            completed_at=2.0,
        )
        session.record_run(run)

        runs = session.get_runs(sid)
        assert len(runs) == 1
        assert runs[0].goal == "Fix bug"
        assert runs[0].iterations == 3

    def test_session_state(self):
        state = SessionState("sess-1", "/project")
        run = RunRecord(goal="Task 1")
        state.add_run(run)
        assert len(state.runs) == 1
        assert state.runs[0].session_id == "sess-1"


class TestToolExecution:
    """Verify tools work with real filesystem."""

    @pytest.mark.asyncio
    async def test_read_file_tool(self, tmp_path):
        from harness_core.tools.filesystem import ReadFileTool
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        tool = ReadFileTool()
        result = await tool.execute({"path": str(test_file)})
        assert result.status.value == "success"
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_write_file_tool(self, tmp_path):
        from harness_core.tools.filesystem import WriteFileTool
        tool = WriteFileTool()
        target = tmp_path / "output.txt"
        result = await tool.execute({"path": str(target), "content": "written"})
        assert result.status.value == "success"
        assert target.read_text() == "written"

    @pytest.mark.asyncio
    async def test_edit_file_tool(self, tmp_path):
        from harness_core.tools.filesystem import EditFileTool
        test_file = tmp_path / "edit.py"
        test_file.write_text("old content")

        tool = EditFileTool()
        result = await tool.execute({
            "path": str(test_file),
            "old_string": "old content",
            "new_string": "new content",
        })
        assert result.status.value == "success"
        assert test_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_list_files_tool(self, tmp_path):
        from harness_core.tools.filesystem import ListFilesTool
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")

        tool = ListFilesTool()
        result = await tool.execute({"path": str(tmp_path)})
        assert result.status.value == "success"
        assert "a.py" in result.output
        assert "b.py" in result.output

    @pytest.mark.asyncio
    async def test_grep_tool(self, tmp_path):
        from harness_core.tools.search import GrepTool
        test_file = tmp_path / "search.py"
        test_file.write_text("def hello():\n    pass")

        tool = GrepTool()
        result = await tool.execute({"pattern": "hello", "path": str(tmp_path)})
        assert result.status.value == "success"
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_glob_tool(self, tmp_path):
        from harness_core.tools.search import GlobTool
        (tmp_path / "test_a.py").write_text("")
        (tmp_path / "test_b.py").write_text("")
        (tmp_path / "other.txt").write_text("")

        tool = GlobTool()
        result = await tool.execute({"pattern": "test_*.py", "path": str(tmp_path)})
        assert result.status.value == "success"
        assert "test_a.py" in result.output
        assert "test_b.py" in result.output


class TestVerificationEngine:
    """Verify the verification engine detects real failures."""

    @pytest.mark.asyncio
    async def test_detect_python_ecosystem(self, tmp_path):
        from harness_core.verification.engine import VerificationEngine
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n[tool.pytest]\n")
        engine = VerificationEngine(workspace_root=tmp_path)
        checks = await engine.detect_ecosystem()
        check_names = [c.name for c in checks]
        assert "pytest" in check_names

    @pytest.mark.asyncio
    async def test_detect_node_ecosystem(self, tmp_path):
        from harness_core.verification.engine import VerificationEngine
        (tmp_path / "package.json").write_text(
            '{"name": "test", "scripts": {"test": "jest"}}'
        )
        engine = VerificationEngine(workspace_root=tmp_path)
        checks = await engine.detect_ecosystem()
        check_names = [c.name for c in checks]
        assert "npm_test" in check_names

    @pytest.mark.asyncio
    async def test_run_single_check(self, tmp_path):
        from harness_core.verification.engine import (
            VerificationCheck,
            VerificationEngine,
        )
        engine = VerificationEngine(workspace_root=tmp_path)
        check = VerificationCheck(name="echo_test", command="echo hello")
        result = await engine._run_check(check)
        assert result.passed is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_run_failing_check(self, tmp_path):
        from harness_core.verification.engine import (
            VerificationCheck,
            VerificationEngine,
        )
        engine = VerificationEngine(workspace_root=tmp_path)
        check = VerificationCheck(name="false_test", command="false")
        result = await engine._run_check(check)
        assert result.passed is False
