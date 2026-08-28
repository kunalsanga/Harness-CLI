"""Tests for M6 — native performance, parallel execution, and cancellation."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest

# ── Native Module Tests ───────────────────────────────────────────────────


class TestNativeModule:
    """Test the native performance module (Python fallback)."""

    def test_import_native(self):
        from harness_core.native import is_native_available
        assert isinstance(is_native_available(), bool)

    def test_fast_glob_basic(self):
        from harness_core.native import fast_glob
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test.py").write_text("x = 1\n")
            (Path(tmpdir) / "other.js").write_text("y = 2\n")
            sub = Path(tmpdir) / "sub"
            sub.mkdir()
            (sub / "nested.py").write_text("z = 3\n")

            results = fast_glob(tmpdir, "*.py", max_files=100, respect_gitignore=False, include_hidden=False)
            assert len(results) >= 2
            assert any("test.py" in r for r in results)
            assert any("nested.py" in r for r in results)

    def test_fast_glob_max_files(self):
        from harness_core.native import fast_glob
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                (Path(tmpdir) / f"file_{i}.py").write_text(f"# {i}\n")

            results = fast_glob(tmpdir, "*.py", max_files=5, respect_gitignore=False, include_hidden=False)
            assert len(results) <= 5

    def test_fast_glob_nonexistent(self):
        from harness_core.native import fast_glob
        results = fast_glob("/nonexistent/path", "*.py", max_files=100, respect_gitignore=False, include_hidden=False)
        assert results == []

    def test_fast_grep_basic(self):
        from harness_core.native import fast_grep
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("AUTH_TOKEN = 'secret'\ndef foo():\n    pass\n")
            (Path(tmpdir) / "b.py").write_text("def bar():\n    return AUTH_TOKEN\n")

            results = fast_grep(tmpdir, "AUTH_TOKEN", max_results=100, case_insensitive=False, respect_gitignore=False)
            assert len(results) >= 1
            assert any(r["file"].endswith("a.py") for r in results)

    def test_fast_grep_case_insensitive(self):
        from harness_core.native import fast_grep
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("Hello World\nhello world\n")

            results = fast_grep(tmpdir, "hello", case_insensitive=True, max_results=100, respect_gitignore=False)
            assert len(results) >= 2

    def test_fast_grep_max_results(self):
        from harness_core.native import fast_grep
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                (Path(tmpdir) / f"file_{i}.py").write_text("TARGET = 1\n" * 5)

            results = fast_grep(tmpdir, "TARGET", max_results=3, respect_gitignore=False)
            assert len(results) <= 3

    def test_fast_file_index(self):
        from harness_core.native import fast_file_index
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("x = 1\n")
            (Path(tmpdir) / "b.js").write_text("y = 2\n")

            results = fast_file_index(tmpdir, max_files=100, respect_gitignore=False)
            assert len(results) >= 2
            assert all("path" in r for r in results)
            assert all("size" in r for r in results)

    def test_fast_hash(self):
        from harness_core.native import fast_hash
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test.txt"
            fpath.write_text("hello world")

            h = fast_hash(str(fpath))
            assert len(h) > 0
            assert isinstance(h, str)

    def test_fast_batch_hash(self):
        from harness_core.native import fast_batch_hash
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(5):
                fpath = Path(tmpdir) / f"file_{i}.txt"
                fpath.write_text(f"content {i}")
                files.append(str(fpath))

            results = fast_batch_hash(files)
            assert len(results) == 5
            assert all(isinstance(v, str) for v in results.values())

    def test_fast_count_files(self):
        from harness_core.native import fast_count_files
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                (Path(tmpdir) / f"file_{i}.py").write_text(f"# {i}\n")
            (Path(tmpdir) / "readme.md").write_text("# README\n")

            count = fast_count_files(tmpdir, respect_gitignore=False)
            assert count >= 11

    def test_fast_count_files_with_extensions(self):
        from harness_core.native import fast_count_files
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                (Path(tmpdir) / f"file_{i}.py").write_text(f"# {i}\n")
            (Path(tmpdir) / "readme.md").write_text("# README\n")

            count = fast_count_files(tmpdir, respect_gitignore=False, extensions=[".py"])
            assert count == 5


# ── File Ownership Tracker Tests ──────────────────────────────────────────


class TestFileOwnershipTracker:
    """Test file ownership tracking for parallel agents."""

    def test_acquire_file(self):
        from harness_core.agents.parallel import FileOwnershipTracker
        tracker = FileOwnershipTracker()
        assert tracker.try_acquire("src/a.py", "agent-1")
        assert tracker.get_owner("src/a.py") == "agent-1"

    def test_conflict_detection(self):
        from harness_core.agents.parallel import FileOwnershipTracker
        tracker = FileOwnershipTracker()
        assert tracker.try_acquire("src/a.py", "agent-1")
        assert not tracker.try_acquire("src/a.py", "agent-2")
        assert len(tracker.get_conflicts()) == 1

    def test_same_agent_reacquire(self):
        from harness_core.agents.parallel import FileOwnershipTracker
        tracker = FileOwnershipTracker()
        assert tracker.try_acquire("src/a.py", "agent-1")
        assert tracker.try_acquire("src/a.py", "agent-1")  # Same agent

    def test_release_file(self):
        from harness_core.agents.parallel import FileOwnershipTracker
        tracker = FileOwnershipTracker()
        tracker.try_acquire("src/a.py", "agent-1")
        tracker.release("src/a.py", "agent-1")
        # After release, another agent can acquire
        assert tracker.try_acquire("src/a.py", "agent-2")

    def test_get_agent_files(self):
        from harness_core.agents.parallel import FileOwnershipTracker
        tracker = FileOwnershipTracker()
        tracker.try_acquire("src/a.py", "agent-1")
        tracker.try_acquire("src/b.py", "agent-1")
        tracker.try_acquire("src/c.py", "agent-2")

        files = tracker.get_agent_files("agent-1")
        assert len(files) == 2
        assert "src/a.py" in files
        assert "src/b.py" in files

    def test_mark_modified(self):
        from harness_core.agents.parallel import FileOwnershipTracker, FileLockState
        tracker = FileOwnershipTracker()
        tracker.try_acquire("src/a.py", "agent-1")
        tracker.mark_modified("src/a.py", "agent-1")
        assert tracker._owners["src/a.py"].state == FileLockState.MODIFIED

    def test_clear(self):
        from harness_core.agents.parallel import FileOwnershipTracker
        tracker = FileOwnershipTracker()
        tracker.try_acquire("src/a.py", "agent-1")
        tracker.clear()
        assert tracker.get_owner("src/a.py") is None


# ── Parallel Scheduler Tests ──────────────────────────────────────────────


class TestParallelScheduler:
    """Test parallel task scheduling."""

    def test_independent_tasks_parallel(self):
        from harness_core.agents.parallel import ParallelScheduler
        from harness_core.agents.domain import SubTask, TaskGraph, AgentRole, TaskStatus

        scheduler = ParallelScheduler(max_concurrent=3)

        # Create graph with independent tasks
        graph = TaskGraph()
        t1 = SubTask(description="Task 1", role=AgentRole.RESEARCHER)
        t2 = SubTask(description="Task 2", role=AgentRole.RESEARCHER)
        t3 = SubTask(description="Task 3", role=AgentRole.RESEARCHER)
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        ready = graph.get_ready_tasks()
        assert len(ready) == 3  # All independent, all ready

    def test_dependent_tasks_sequential(self):
        from harness_core.agents.domain import SubTask, TaskGraph, AgentRole

        graph = TaskGraph()
        t1 = SubTask(description="First", role=AgentRole.RESEARCHER)
        t2 = SubTask(description="Second", role=AgentRole.CODER, dependencies=[t1.task_id])
        graph.add_task(t1)
        graph.add_task(t2)

        ready = graph.get_ready_tasks()
        assert len(ready) == 1  # Only t1 is ready
        assert ready[0].task_id == t1.task_id

    def test_concurrency_limit(self):
        from harness_core.agents.parallel import ParallelScheduler
        scheduler = ParallelScheduler(max_concurrent=2)
        assert scheduler.max_concurrent == 2


# ── Cancellation Tests ────────────────────────────────────────────────────


class TestCancellation:
    """Test cancellation and shutdown support."""

    def test_cancel(self):
        from harness_core.agents.cancellation import CancellationHandler
        handler = CancellationHandler()
        assert not handler.is_cancelled
        handler.cancel("test")
        assert handler.is_cancelled
        assert handler.cancel_reason == "test"

    def test_cleanup_callback(self):
        from harness_core.agents.cancellation import CancellationHandler
        handler = CancellationHandler()
        called = []
        handler.register_cleanup(lambda: called.append(True))
        handler.cancel()
        assert len(called) == 1

    def test_reset(self):
        from harness_core.agents.cancellation import CancellationHandler
        handler = CancellationHandler()
        handler.cancel("test")
        handler.reset()
        assert not handler.is_cancelled

    def test_check(self):
        from harness_core.agents.cancellation import CancellationHandler
        handler = CancellationHandler()
        assert not handler.check()
        handler.cancel()
        assert handler.check()

    def test_operation_timeout(self):
        from harness_core.agents.cancellation import OperationTimeout
        with OperationTimeout(0.1) as timeout:
            assert not timeout.expired()
            time.sleep(0.15)
            assert timeout.expired()

    def test_operation_timeout_remaining(self):
        from harness_core.agents.cancellation import OperationTimeout
        with OperationTimeout(1.0) as timeout:
            remaining = timeout.remaining()
            assert 0 < remaining <= 1.0

    def test_graceful_shutdown(self):
        from harness_core.agents.cancellation import CancellationHandler, GracefulShutdown
        handler = CancellationHandler()
        with GracefulShutdown(handler) as shutdown:
            assert not handler.is_cancelled
        # After exit, should be cancelled
        assert handler.is_cancelled


# ── Integration: Parallel Graph Execution ──────────────────────────────────


class TestParallelIntegration:
    """Integration tests for parallel execution with graph."""

    @pytest.mark.asyncio
    async def test_linear_graph_execution(self):
        from harness_core.agents.parallel import ParallelScheduler
        from harness_core.agents.domain import SubTask, TaskGraph, AgentRole, AgentResult, AgentStatus

        scheduler = ParallelScheduler(max_concurrent=2)

        graph = TaskGraph()
        t1 = SubTask(description="Research", role=AgentRole.RESEARCHER)
        t2 = SubTask(description="Code", role=AgentRole.CODER, dependencies=[t1.task_id])
        graph.add_task(t1)
        graph.add_task(t2)

        async def mock_executor(task, context, workspace):
            await asyncio.sleep(0.01)
            return AgentResult(
                status=AgentStatus.COMPLETED,
                summary=f"Done: {task.description}",
            )

        results = await scheduler.schedule(graph, mock_executor)
        assert len(results) == 2
        assert all(r.status == AgentStatus.COMPLETED for r in results.values())

    @pytest.mark.asyncio
    async def test_parallel_graph_execution(self):
        from harness_core.agents.parallel import ParallelScheduler
        from harness_core.agents.domain import SubTask, TaskGraph, AgentRole, AgentResult, AgentStatus

        scheduler = ParallelScheduler(max_concurrent=5)

        graph = TaskGraph()
        # Three independent tasks
        t1 = SubTask(description="Research A", role=AgentRole.RESEARCHER)
        t2 = SubTask(description="Research B", role=AgentRole.RESEARCHER)
        t3 = SubTask(description="Research C", role=AgentRole.RESEARCHER)
        # Dependent task
        t4 = SubTask(description="Implement", role=AgentRole.CODER,
                     dependencies=[t1.task_id, t2.task_id, t3.task_id])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)
        graph.add_task(t4)

        start = time.time()

        async def mock_executor(task, context, workspace):
            await asyncio.sleep(0.05)
            return AgentResult(
                status=AgentStatus.COMPLETED,
                summary=f"Done: {task.description}",
            )

        results = await scheduler.schedule(graph, mock_executor)
        elapsed = time.time() - start

        assert len(results) == 4
        # 3 parallel tasks (0.05s each) + 1 dependent task (0.05s) = ~0.10s total
        # Not 0.20s (which would be sequential)
        assert elapsed < 0.18  # Some margin for scheduling overhead

    @pytest.mark.asyncio
    async def test_task_failure(self):
        from harness_core.agents.parallel import ParallelScheduler
        from harness_core.agents.domain import SubTask, TaskGraph, AgentRole, AgentResult, AgentStatus

        scheduler = ParallelScheduler(max_concurrent=3)

        graph = TaskGraph()
        t1 = SubTask(description="Fail", role=AgentRole.CODER)
        graph.add_task(t1)

        async def failing_executor(task, context, workspace):
            raise RuntimeError("Simulated failure")

        results = await scheduler.schedule(graph, failing_executor)
        assert len(results) == 1
        result = list(results.values())[0]
        assert result.status == AgentStatus.FAILED
        assert "Simulated failure" in result.errors[0]

    def test_file_conflict_in_parallel(self):
        from harness_core.agents.parallel import FileOwnershipTracker

        tracker = FileOwnershipTracker()
        # Two agents try to modify the same file
        assert tracker.try_acquire("src/auth.py", "coder-1")
        assert not tracker.try_acquire("src/auth.py", "coder-2")

        conflicts = tracker.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][0] == "src/auth.py"
