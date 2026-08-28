"""Tests for agent recovery system and session persistence."""

import tempfile
import time

from harness_core.agent.recovery import (
    RecoveryDecision,
    RecoveryStrategy,
    classify_error,
)
from harness_core.session.session import (
    RunRecord,
    Session,
    SessionState,
)


class TestRecoveryClassification:
    """Test error classification and recovery strategy selection."""

    def test_429_fallback(self):
        decision = classify_error(Exception("HTTP 429 Too Many Requests"))
        assert decision.strategy == RecoveryStrategy.FALLBACK_MODEL
        assert decision.retry_delay_seconds > 0

    def test_rate_limit_fallback(self):
        decision = classify_error(Exception("Rate limit exceeded"))
        assert decision.strategy == RecoveryStrategy.FALLBACK_MODEL

    def test_timeout_retry(self):
        decision = classify_error(Exception("Request timed out"))
        assert decision.strategy == RecoveryStrategy.RETRY
        assert decision.retry_delay_seconds > 0

    def test_context_overflow_fallback(self):
        decision = classify_error(Exception("Context token limit exceeded"))
        assert decision.strategy == RecoveryStrategy.FALLBACK_MODEL

    def test_500_retry(self):
        decision = classify_error(Exception("HTTP 500 Internal Server Error"))
        assert decision.strategy == RecoveryStrategy.RETRY
        assert decision.retry_delay_seconds > 0

    def test_502_retry(self):
        decision = classify_error(Exception("Bad Gateway: 502"))
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_503_retry(self):
        decision = classify_error(Exception("Service Unavailable: 503"))
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_auth_abort(self):
        decision = classify_error(Exception("HTTP 401 Unauthorized"))
        assert decision.strategy == RecoveryStrategy.ABORT

    def test_forbidden_abort(self):
        decision = classify_error(Exception("HTTP 403 Forbidden"))
        assert decision.strategy == RecoveryStrategy.ABORT

    def test_not_found_abort(self):
        decision = classify_error(Exception("HTTP 404 Not Found"))
        assert decision.strategy == RecoveryStrategy.ABORT

    def test_permission_skip(self):
        decision = classify_error(Exception("Permission denied"))
        assert decision.strategy == RecoveryStrategy.SKIP_TOOL

    def test_unknown_tool_skip(self):
        decision = classify_error(Exception("Unknown tool: foo"))
        assert decision.strategy == RecoveryStrategy.SKIP_TOOL

    def test_network_retry(self):
        decision = classify_error(Exception("Connection refused"))
        assert decision.strategy == RecoveryStrategy.RETRY
        assert decision.retry_delay_seconds > 0

    def test_dns_retry(self):
        decision = classify_error(Exception("DNS resolution failed"))
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_unknown_error_retry(self):
        decision = classify_error(Exception("Something weird happened"))
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_decision_has_reason(self):
        decision = classify_error(Exception("Rate limited"))
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_timeout_asyncio(self):
        import asyncio
        try:
            raise asyncio.TimeoutError()
        except asyncio.TimeoutError as e:
            decision = classify_error(e)
            assert decision.strategy == RecoveryStrategy.RETRY


class TestRecoveryDecision:
    """Test RecoveryDecision dataclass."""

    def test_decision_fields(self):
        d = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            reason="test",
            retry_delay_seconds=1.5,
            fallback_model="model-a",
        )
        assert d.strategy == RecoveryStrategy.RETRY
        assert d.retry_delay_seconds == 1.5
        assert d.fallback_model == "model-a"

    def test_strategy_values(self):
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK_MODEL.value == "fallback_model"
        assert RecoveryStrategy.SKIP_TOOL.value == "skip_tool"
        assert RecoveryStrategy.ABORT.value == "abort"
        assert RecoveryStrategy.USER_ACTION.value == "user_action"


class TestSessionState:
    """Test in-memory session state."""

    def test_create_state(self):
        state = SessionState("sess-123", "/project")
        assert state.session_id == "sess-123"
        assert state.project_path == "/project"
        assert len(state.runs) == 0

    def test_add_run(self):
        state = SessionState("sess-1")
        run = RunRecord(goal="Fix bug", session_id="wrong")
        state.add_run(run)
        assert len(state.runs) == 1
        assert state.runs[0].session_id == "sess-1"

    def test_to_dict(self):
        state = SessionState("sess-1", "/p")
        d = state.to_dict()
        assert d["session_id"] == "sess-1"
        assert d["project_path"] == "/p"
        assert d["runs"] == 0


class TestRunRecord:
    """Test RunRecord defaults."""

    def test_default_run(self):
        r = RunRecord()
        assert len(r.run_id) > 0
        assert r.status == "pending"
        assert r.iterations == 0
        assert r.tool_calls == 0
        assert r.started_at > 0

    def test_run_with_fields(self):
        r = RunRecord(
            goal="Test",
            model_used="m",
            provider_used="p",
            task_type="coding",
        )
        assert r.goal == "Test"
        assert r.model_used == "m"


class TestSQLiteSession:
    """Test SQLite-backed session storage."""

    def _make_session(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        return Session(path)

    def test_create_session(self):
        s = self._make_session()
        sid = s.create_session("/project")
        assert len(sid) > 0

    def test_record_and_get_runs(self):
        s = self._make_session()
        sid = s.create_session()
        run = RunRecord(
            session_id=sid,
            goal="Fix bug",
            status="completed",
            model_used="test-model",
            provider_used="test-provider",
            iterations=3,
            tool_calls=5,
            result="Fixed",
            started_at=time.time(),
            completed_at=time.time(),
        )
        s.record_run(run)
        runs = s.get_runs(sid)
        assert len(runs) == 1
        assert runs[0].goal == "Fix bug"
        assert runs[0].iterations == 3

    def test_get_sessions(self):
        s = self._make_session()
        s.create_session("/p1")
        s.create_session("/p2")
        sessions = s.get_sessions()
        assert len(sessions) == 2

    def test_count(self):
        s = self._make_session()
        assert s.count() == 0
        s.create_session()
        assert s.count() == 1
        s.create_session()
        assert s.count() == 2

    def test_multiple_runs(self):
        s = self._make_session()
        sid = s.create_session()
        for i in range(5):
            run = RunRecord(
                session_id=sid,
                goal=f"Task {i}",
                status="completed",
                started_at=time.time(),
            )
            s.record_run(run)
        runs = s.get_runs(sid)
        assert len(runs) == 5

    def test_runs_ordered_by_time(self):
        s = self._make_session()
        sid = s.create_session()
        t = time.time()
        for i in range(3):
            run = RunRecord(
                session_id=sid,
                goal=f"Task {i}",
                started_at=t + i,
            )
            s.record_run(run)
        runs = s.get_runs(sid)
        goals = [r.goal for r in runs]
        assert goals == ["Task 0", "Task 1", "Task 2"]
