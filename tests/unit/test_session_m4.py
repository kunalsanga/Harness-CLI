"""Tests for M4 — Session Intelligence & Persistent Agent Memory."""

import time
import pytest
from pathlib import Path

from harness_core.session.domain import (
    Checkpoint,
    MemoryItem,
    MemoryType,
    Run,
    RunStatus,
    Session,
    SessionEvent,
    SessionStatus,
    can_transition,
)
from harness_core.session.storage import SessionStorage
from harness_core.session.manager import SessionManager


# ═══════════════════════════════════════════════════════════════════════
# Domain Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSessionDomain:
    """Test Session domain object."""

    def test_create_session(self):
        s = Session(workspace_path="/test", title="Test Session")
        assert s.session_id
        assert s.status == SessionStatus.ACTIVE
        assert s.workspace_path == "/test"
        assert s.title == "Test Session"

    def test_session_transition_valid(self):
        s = Session()
        assert s.status == SessionStatus.ACTIVE
        s.transition_to(SessionStatus.PAUSED)
        assert s.status == SessionStatus.PAUSED

    def test_session_transition_invalid(self):
        s = Session()
        s.transition_to(SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="Invalid session transition"):
            s.transition_to(SessionStatus.ACTIVE)

    def test_session_to_dict(self):
        s = Session(workspace_path="/p", title="T")
        d = s.to_dict()
        assert d["workspace_path"] == "/p"
        assert d["title"] == "T"
        assert d["status"] == "active"


class TestRunDomain:
    """Test Run domain object."""

    def test_create_run(self):
        r = Run(session_id="s1", task="Fix bug")
        assert r.run_id
        assert r.status == RunStatus.PENDING

    def test_run_transition_running(self):
        r = Run()
        r.transition_to(RunStatus.RUNNING)
        assert r.status == RunStatus.RUNNING

    def test_run_transition_complete(self):
        r = Run()
        r.transition_to(RunStatus.RUNNING)
        r.transition_to(RunStatus.COMPLETED)
        assert r.status == RunStatus.COMPLETED
        assert r.completed_at > 0
        assert r.duration_ms >= 0

    def test_run_transition_invalid(self):
        r = Run()
        r.transition_to(RunStatus.RUNNING)
        r.transition_to(RunStatus.COMPLETED)
        with pytest.raises(ValueError, match="Invalid run transition"):
            r.transition_to(RunStatus.RUNNING)

    def test_run_to_dict(self):
        r = Run(session_id="s1", task="test", model_id="model-x")
        d = r.to_dict()
        assert d["task"] == "test"
        assert d["model_id"] == "model-x"


class TestCheckpointDomain:
    """Test Checkpoint domain object."""

    def test_create_checkpoint(self):
        cp = Checkpoint(
            session_id="s1",
            git_head="abc123",
            git_branch="main",
            current_task="Fix auth",
        )
        assert cp.checkpoint_id
        assert cp.git_head == "abc123"
        assert cp.current_task == "Fix auth"

    def test_checkpoint_to_dict(self):
        cp = Checkpoint(session_id="s1", git_head="abc")
        d = cp.to_dict()
        assert d["git_head"] == "abc"
        assert d["tests_passing"] == 0


class TestMemoryDomain:
    """Test MemoryItem domain object."""

    def test_create_memory(self):
        m = MemoryItem(
            session_id="s1",
            memory_type=MemoryType.DECISION,
            content="Use JWT for auth",
            importance=0.8,
        )
        assert m.memory_id
        assert m.memory_type == MemoryType.DECISION
        assert m.importance == 0.8

    def test_memory_to_dict(self):
        m = MemoryItem(
            session_id="s1",
            memory_type=MemoryType.TODO,
            content="Add tests",
        )
        d = m.to_dict()
        assert d["memory_type"] == "todo"
        assert d["content"] == "Add tests"


class TestStateTransitions:
    """Test valid and invalid state transitions."""

    def test_session_lifecycle(self):
        s = Session()
        s.transition_to(SessionStatus.PAUSED)
        s.transition_to(SessionStatus.ACTIVE)
        s.transition_to(SessionStatus.COMPLETED)
        s.transition_to(SessionStatus.ARCHIVED)
        assert s.status == SessionStatus.ARCHIVED

    def test_session_failure_retry(self):
        s = Session()
        s.transition_to(SessionStatus.FAILED)
        s.transition_to(SessionStatus.ACTIVE)  # retry
        assert s.status == SessionStatus.ACTIVE

    def test_run_full_lifecycle(self):
        r = Run()
        r.transition_to(RunStatus.RUNNING)
        r.transition_to(RunStatus.COMPLETED)
        assert r.status == RunStatus.COMPLETED

    def test_run_interrupt(self):
        r = Run()
        r.transition_to(RunStatus.RUNNING)
        r.transition_to(RunStatus.INTERRUPTED)
        assert r.status == RunStatus.INTERRUPTED

    def test_can_transition_function(self):
        assert can_transition(SessionStatus.ACTIVE, SessionStatus.PAUSED, {
            SessionStatus.ACTIVE: {SessionStatus.PAUSED}
        })
        assert not can_transition(SessionStatus.COMPLETED, SessionStatus.ACTIVE, {
            SessionStatus.COMPLETED: {SessionStatus.ARCHIVED}
        })


# ═══════════════════════════════════════════════════════════════════════
# Storage Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSessionStorage:
    """Test SQLite-backed session storage."""

    def test_create_and_get_session(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session(workspace_path="/test", title="Test")
        storage.create_session(s)

        retrieved = storage.get_session(s.session_id)
        assert retrieved is not None
        assert retrieved.workspace_path == "/test"
        assert retrieved.title == "Test"

    def test_list_sessions(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        for i in range(5):
            storage.create_session(Session(title=f"Session {i}"))

        sessions = storage.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_by_status(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        storage.create_session(Session(title="Active", status=SessionStatus.ACTIVE))
        storage.create_session(Session(title="Paused", status=SessionStatus.PAUSED))

        active = storage.list_sessions(status=SessionStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].title == "Active"

    def test_update_session(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session(title="Original")
        storage.create_session(s)

        s.title = "Updated"
        s.transition_to(SessionStatus.PAUSED)
        storage.update_session(s)

        retrieved = storage.get_session(s.session_id)
        assert retrieved.title == "Updated"
        assert retrieved.status == SessionStatus.PAUSED

    def test_delete_session(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session(title="ToDelete")
        storage.create_session(s)
        storage.create_run(Run(session_id=s.session_id, task="test"))

        deleted = storage.delete_session(s.session_id)
        assert deleted == 1
        assert storage.get_session(s.session_id) is None
        assert storage.count_runs(s.session_id) == 0

    def test_run_persistence(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session()
        storage.create_session(s)

        run = Run(session_id=s.session_id, task="Fix auth", model_id="model-x")
        storage.create_run(run)

        runs = storage.get_runs(s.session_id)
        assert len(runs) == 1
        assert runs[0].task == "Fix auth"
        assert runs[0].model_id == "model-x"

    def test_checkpoint_persistence(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session()
        storage.create_session(s)

        cp = Checkpoint(
            session_id=s.session_id,
            git_head="abc123",
            git_branch="main",
            tests_passing=5,
            tests_total=10,
        )
        storage.create_checkpoint(cp)

        latest = storage.get_latest_checkpoint(s.session_id)
        assert latest is not None
        assert latest.git_head == "abc123"
        assert latest.tests_passing == 5

    def test_memory_persistence(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session()
        storage.create_session(s)

        m = MemoryItem(
            session_id=s.session_id,
            memory_type=MemoryType.DECISION,
            content="Use JWT auth",
            importance=0.8,
        )
        storage.add_memory(m)

        memories = storage.get_memories(s.session_id)
        assert len(memories) == 1
        assert memories[0].content == "Use JWT auth"
        assert memories[0].importance == 0.8

    def test_memory_search(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session()
        storage.create_session(s)

        storage.add_memory(MemoryItem(
            session_id=s.session_id,
            memory_type=MemoryType.DECISION,
            content="Authentication uses JWT tokens",
        ))
        storage.add_memory(MemoryItem(
            session_id=s.session_id,
            memory_type=MemoryType.TODO,
            content="Add payment integration",
        ))

        results = storage.search_memories(s.session_id, "JWT")
        assert len(results) == 1
        assert "JWT" in results[0].content

    def test_event_persistence(self, tmp_path):
        storage = SessionStorage(tmp_path / "test.db")
        s = Session()
        storage.create_session(s)

        event = SessionEvent(
            session_id=s.session_id,
            event_type="run.started",
            data={"task": "Fix bug"},
        )
        storage.record_event(event)

        events = storage.get_events(s.session_id)
        assert len(events) == 1
        assert events[0].event_type == "run.started"

    def test_thread_safety(self, tmp_path):
        """Storage should handle concurrent access."""
        import threading
        storage = SessionStorage(tmp_path / "test.db")
        s = Session()
        storage.create_session(s)

        errors = []

        def create_run(i):
            try:
                storage.create_run(Run(
                    session_id=s.session_id,
                    task=f"Task {i}",
                ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert storage.count_runs(s.session_id) == 10


# ═══════════════════════════════════════════════════════════════════════
# Manager Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSessionManager:
    """Test high-level session lifecycle management."""

    def test_create_session(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session(workspace_path="/test", title="My Session")
        assert s.session_id
        assert s.title == "My Session"

    def test_pause_and_resume(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        mgr.pause_session(s.session_id)
        session = mgr.storage.get_session(s.session_id)
        assert session.status == SessionStatus.PAUSED

        mgr.resume_session(s.session_id)
        session = mgr.storage.get_session(s.session_id)
        assert session.status == SessionStatus.ACTIVE

    def test_complete_and_archive(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        mgr.complete_session(s.session_id)
        session = mgr.storage.get_session(s.session_id)
        assert session.status == SessionStatus.COMPLETED

        mgr.archive_session(s.session_id)
        session = mgr.storage.get_session(s.session_id)
        assert session.status == SessionStatus.ARCHIVED

    def test_start_and_complete_run(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        run = mgr.start_run(s.session_id, "Fix auth", model_id="model-x")
        assert run.status == RunStatus.RUNNING

        completed = mgr.complete_run(
            run.run_id,
            outcome="success",
            verification_passed=True,
            tool_calls=5,
            iterations=3,
        )
        assert completed.status == RunStatus.COMPLETED
        assert completed.verification_passed is True

    def test_interrupt_run(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        run = mgr.start_run(s.session_id, "Task")
        interrupted = mgr.interrupt_run(run.run_id)
        assert interrupted.status == RunStatus.INTERRUPTED

    def test_create_checkpoint(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        cp = mgr.create_checkpoint(
            s.session_id,
            git_head="abc123",
            git_branch="main",
            tests_passing=8,
            tests_total=10,
        )
        assert cp.checkpoint_id

        latest = mgr.storage.get_latest_checkpoint(s.session_id)
        assert latest.git_head == "abc123"

    def test_get_resume_state(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        run = mgr.start_run(s.session_id, "Fix bug")
        mgr.create_checkpoint(s.session_id, git_head="abc", tests_passing=5, tests_total=10)
        mgr.add_decision(s.session_id, "Use JWT auth")

        state = mgr.get_resume_state(s.session_id)
        assert state is not None
        assert state["session"].session_id == s.session_id
        assert len(state["runs"]) == 1
        assert state["checkpoint"] is not None
        assert len(state["memories"]) == 1

    def test_get_resume_state_nonexistent(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        state = mgr.get_resume_state("nonexistent")
        assert state is None

    def test_memory_management(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        mgr.add_decision(s.session_id, "Use JWT for auth", importance=0.9)
        mgr.add_constraint(s.session_id, "Do not modify public API", importance=0.8)
        mgr.add_todo(s.session_id, "Add refresh token tests", importance=0.5)
        mgr.add_warning(s.session_id, "Legacy endpoint still used", importance=0.7)

        # Get by importance
        memories = mgr.storage.get_memories(s.session_id, min_importance=0.7)
        assert len(memories) == 3

        # Get by type
        decisions = mgr.storage.get_memories(s.session_id, memory_type=MemoryType.DECISION)
        assert len(decisions) == 1

    def test_relevant_memories(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        mgr.add_decision(s.session_id, "Authentication uses JWT tokens")
        mgr.add_discovery(s.session_id, "Payment service is in billing module")
        mgr.add_constraint(s.session_id, "Do not modify public API")

        relevant = mgr.get_relevant_memories(s.session_id, ["auth", "JWT"])
        assert len(relevant) > 0
        assert any("auth" in m.content.lower() or "jwt" in m.content.lower() for m in relevant)

    def test_secret_sanitization(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()

        # Content with API key
        content = "The key is sk-or-v1-abc123def456ghi789jkl012mno345pqr"
        item = mgr.add_memory(s.session_id, MemoryType.NOTE, content)

        # Verify sanitization
        stored = mgr.storage.get_memories(s.session_id)
        assert len(stored) == 1
        assert "sk-or-v1" not in stored[0].content
        assert "REDACTED" in stored[0].content

    def test_export_json(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()
        mgr.start_run(s.session_id, "Task")

        output = mgr.export_session(s.session_id, fmt="json")
        assert "session_id" in output
        assert "runs" in output

    def test_export_markdown(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()
        mgr.start_run(s.session_id, "Fix auth")
        mgr.add_decision(s.session_id, "Use JWT")

        output = mgr.export_session(s.session_id, fmt="markdown")
        assert "# Session:" in output
        assert "Fix auth" in output

    def test_invalid_session_error(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        with pytest.raises(ValueError, match="not found"):
            mgr.pause_session("nonexistent")


class TestEventLogging:
    """Test event persistence."""

    def test_events_recorded(self, tmp_path):
        mgr = SessionManager(SessionStorage(tmp_path / "test.db"))
        s = mgr.create_session()
        mgr.pause_session(s.session_id)
        mgr.resume_session(s.session_id)

        events = mgr.storage.get_events(s.session_id)
        types = [e.event_type for e in events]
        assert "session.created" in types
        assert "session.paused" in types
        assert "session.resumed" in types


class TestMemoryTypes:
    """Test all memory type categories."""

    def test_all_memory_types(self):
        types = [
            MemoryType.DECISION, MemoryType.DISCOVERY, MemoryType.CONSTRAINT,
            MemoryType.TODO, MemoryType.WARNING, MemoryType.ERROR,
            MemoryType.SOLUTION, MemoryType.NOTE,
        ]
        assert len(types) == 8

    def test_memory_type_values(self):
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.TODO.value == "todo"
        assert MemoryType.WARNING.value == "warning"
