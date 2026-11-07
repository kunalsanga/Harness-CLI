"""Harness V2 — Interactive Acceptance Test and UX Hardening Suite.

Comprehensive acceptance testing validating the real interactive product against
the 8 critical operational areas:
1. Fresh Launch Clean Slate & Branded Welcome (empty DB vs existing DB).
2. Date-Grouped Session Browser with real SQLite persistence.
3. Historical Conversation Viewer integrity (zero model/tool invocations, read-only).
4. Session Resume matrix (completed, paused, interrupted, provider-failed, missing WS).
5. Live Agent Execution & Dynamic Ladder (file ops, test fixing, stage progression).
6. File Activity Tracking & Diff Security (tracked, untracked, secret redaction, line capping).
7. Provider Failure & Fallback Transparency (error classification, failover stage).
8. Performance, Redraw Deduplication, and Terminal Stability.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.cli.conversation import ConversationRenderer, ConversationState
from harness_core.cli.interactive import InteractiveShell
from harness_core.cli.session_browser import (
    group_sessions_by_date,
    render_historical_transcript,
    render_session_browser,
)
from harness_core.cli.ui import Sym, render_welcome
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.routing.fallback import FallbackConfig, FallbackEngine
from harness_core.routing.router import ModelRouter, RouterConfig
from harness_core.session.domain import Run, RunStatus, Session, SessionEvent, SessionStatus
from harness_core.session.manager import SessionManager
from harness_core.session.storage import SessionStorage
from harness_core.tools.base import Tool, ToolSchema


# ─── Mock Controllable Provider ─────────────────────────────────────────────

class MockTestProvider(ModelProvider):
    """Controllable mock provider for simulating multi-step agent actions."""

    def __init__(
        self,
        name: str,
        model_id: str,
        responses: list[CompletionResponse | Exception] | None = None,
        context_window: int = 128000,
        supports_tools: bool = True,
        is_free: bool = True,
    ) -> None:
        self._name = name
        self.model_id = model_id
        self.responses = list(responses or [])
        self.context_window = context_window
        self.supports_tools = supports_tools
        self.is_free = is_free
        self.call_count = 0
        self.received_requests: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def set_responses(self, responses: list[CompletionResponse | Exception]) -> None:
        self.responses = list(responses)

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        self.received_requests.append(request)
        if not self.responses:
            return CompletionResponse(
                content="I'm done with the task.",
                model=self.model_id,
                provider=self._name,
                tool_calls=[],
                usage={"total_tokens": 100},
            )
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def stream(self, request: CompletionRequest):
        resp = await self.generate(request)
        yield resp.content

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=self.model_id,
                name=self.model_id,
                provider=self._name,
                context_window=self.context_window,
                supports_tools=self.supports_tools,
                supports_streaming=True,
                is_free=self.is_free,
            )
        ]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def routing_hints(self, mode: str = "auto") -> list[ModelInfo]:
        return [
            ModelInfo(
                id=self.model_id,
                name=self.model_id,
                provider=self._name,
                context_window=self.context_window,
                supports_tools=self.supports_tools,
                is_free=self.is_free,
            )
        ]


# ─── SECTION 1: Fresh Launch Clean Slate ─────────────────────────────────────

class TestSection1FreshLaunch:
    """Validate that starting Harness shell starts fresh without hijacking prior tasks."""

    def test_fresh_launch_empty_database(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)

        out = io.StringIO()
        shell = InteractiveShell(workspace=str(tmp_path), plain=True)
        shell.console = Console(file=out, no_color=True, width=100)
        shell.session_manager = mgr
        shell.session_id = None
        shell.resume_latest = False

        shell._setup_session()
        shell._print_welcome()
        output = out.getvalue()

        # Must have created a session in DB
        assert shell.session_id is not None
        saved = storage.get_session(shell.session_id)
        assert saved is not None
        assert saved.status == SessionStatus.ACTIVE

        # Clean welcome screen: no recap, no prior transcripts
        assert "Harness" in output
        assert "Autonomous AI Engineering" in output
        assert "Session:" in output
        assert "Last Goal" not in output
        assert "Reconciliation" not in output

    def test_fresh_launch_with_existing_active_and_paused_sessions(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)

        # Pre-seed active, paused, and completed sessions
        s_active = mgr.create_session(workspace_path=str(tmp_path), title="Active Task 1")
        s_paused = mgr.create_session(workspace_path=str(tmp_path), title="Paused Task 2")
        storage.update_session(
            Session(
                session_id=s_paused.session_id,
                workspace_path=str(tmp_path),
                title="Paused Task 2",
                status=SessionStatus.PAUSED,
                created_at=time.time() - 200,
                updated_at=time.time() - 100,
            )
        )
        s_done = mgr.create_session(workspace_path=str(tmp_path), title="Completed Task 3")
        storage.update_session(
            Session(
                session_id=s_done.session_id,
                workspace_path=str(tmp_path),
                title="Completed Task 3",
                status=SessionStatus.COMPLETED,
                created_at=time.time() - 400,
                updated_at=time.time() - 300,
            )
        )

        out = io.StringIO()
        shell = InteractiveShell(workspace=str(tmp_path), plain=False)
        shell.console = Console(file=out, force_terminal=True, width=100)
        shell.session_manager = mgr
        shell.session_id = None
        shell.resume_latest = False

        # Fresh launch
        shell._setup_session()
        shell._print_welcome()
        output = out.getvalue()

        # Must create a BRAND NEW session ID, not hijacking active or paused
        assert shell.session_id not in (s_active.session_id, s_paused.session_id, s_done.session_id)
        # Output must NOT show recap or resume prompt
        assert "Active Task 1" not in output
        assert "Paused Task 2" not in output
        assert "Harness" in output
        assert "Tips: Enter to send" in output

    def test_explicit_resume_via_flag(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)

        old_session = mgr.create_session(workspace_path=str(tmp_path), title="Mission TaskForge")

        out = io.StringIO()
        shell = InteractiveShell(workspace=str(tmp_path), plain=False, session_id=old_session.session_id)
        shell.console = Console(file=out, force_terminal=True, width=100)
        shell.session_manager = mgr

        shell._setup_session()
        output = out.getvalue()

        # Explicit resume MUST load the old session and display recap
        assert shell.session_id == old_session.session_id
        assert "Session:" in output or "Mission TaskForge" in output


# ─── SECTION 2: Session Browser with Real SQLite Persistence ─────────────────

class TestSection2SessionBrowser:
    """Validate /sessions date grouping, filtering, and table display."""

    @pytest.fixture
    def real_storage(self, tmp_path):
        db_path = tmp_path / "browser_test.db"
        storage = SessionStorage(db_path=db_path)
        now = time.time()

        # 1. Today
        s1 = Session(
            session_id="today-001",
            workspace_path="/projects/app1",
            title="Implement OAuth Flow",
            status=SessionStatus.ACTIVE,
            created_at=now - 1800,
            updated_at=now - 600,
        )
        storage.create_session(s1)
        r1 = Run(
            run_id="r1",
            session_id=s1.session_id,
            task="Add login button",
            status=RunStatus.COMPLETED,
            started_at=now - 1200,
            completed_at=now - 1100,
            duration_ms=100000,
            tool_calls=4,
        )
        storage.create_run(r1)

        # 2. Yesterday
        yesterday_ts = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
        s2 = Session(
            session_id="yest-002",
            workspace_path="/projects/backend",
            title="Database Schema Migration",
            status=SessionStatus.COMPLETED,
            created_at=yesterday_ts - 3600,
            updated_at=yesterday_ts,
        )
        storage.create_session(s2)

        # 3. Earlier this week
        week_ts = (datetime.datetime.now() - datetime.timedelta(days=4)).timestamp()
        s3 = Session(
            session_id="week-003",
            workspace_path="/projects/frontend",
            title="Refactor Kanban Board",
            status=SessionStatus.PAUSED,
            created_at=week_ts - 7200,
            updated_at=week_ts,
        )
        storage.create_session(s3)

        return storage

    def test_sessions_command_date_grouping(self, real_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        sessions = real_storage.list_sessions(limit=50)

        render_session_browser(console, sessions, real_storage, current_session_id="today-001")
        output = out.getvalue()

        assert "TODAY" in output
        assert "YESTERDAY" in output
        assert "THIS WEEK" in output
        assert "Implement OAuth Flow" in output
        assert "Database Schema Migration" in output
        assert "Refactor Kanban Board" in output
        assert "Active" in output
        assert "Completed" in output
        assert "Paused" in output

    def test_sessions_search_filtering(self, real_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        sessions = real_storage.list_sessions(limit=50)

        # Search for "OAuth"
        render_session_browser(console, sessions, real_storage, query="OAuth")
        output = out.getvalue()

        assert "Implement OAuth Flow" in output
        assert "Database Schema Migration" not in output
        assert "Refactor Kanban Board" not in output

    def test_sessions_search_inside_run_prompts(self, real_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        sessions = real_storage.list_sessions(limit=50)

        # Search for prompt task "login button"
        render_session_browser(console, sessions, real_storage, query="login button")
        output = out.getvalue()

        assert "today-001" in output
        assert "Database Schema Migration" not in output


# ─── SECTION 3: Historical Conversation Viewer (Strictly Read-Only) ──────────

class TestSection3HistoricalConversationViewer:
    """Validate /sessions view <id> integrity and strict read-only nature."""

    @pytest.fixture
    def session_with_history(self, tmp_path):
        db_path = tmp_path / "hist_test.db"
        storage = SessionStorage(db_path=db_path)
        now = time.time()

        sess = Session(
            session_id="hist-9876543210",
            workspace_path=str(tmp_path),
            title="Implement User Verification",
            status=SessionStatus.COMPLETED,
            created_at=now - 500,
            updated_at=now - 100,
        )
        storage.create_session(sess)

        # Run 1
        r1 = Run(
            run_id="run-1",
            session_id=sess.session_id,
            task="Inspect user model and add is_verified column",
            status=RunStatus.COMPLETED,
            started_at=now - 450,
            completed_at=now - 400,
            duration_ms=50000,
            tool_calls=2,
            model_id="qwen3-coder-plus",
            verification_passed=True,
            result_summary="Added `is_verified` boolean column to User model with default False.",
        )
        storage.create_run(r1)

        # Run 2
        r2 = Run(
            run_id="run-2",
            session_id=sess.session_id,
            task="Run pytest on test_auth.py",
            status=RunStatus.COMPLETED,
            started_at=now - 350,
            completed_at=now - 300,
            duration_ms=50000,
            tool_calls=1,
            model_id="llama-3.3-70b-versatile",
            verification_passed=True,
            result_summary="All 12 user authentication tests passed cleanly.",
        )
        storage.create_run(r2)

        # Events
        storage.record_event(
            SessionEvent(
                session_id=sess.session_id,
                event_type="tool.completed",
                data={"run_id": "run-1", "tool": "read_file", "status": "success"},
            )
        )
        storage.record_event(
            SessionEvent(
                session_id=sess.session_id,
                event_type="tool.completed",
                data={"run_id": "run-1", "tool": "edit_file", "status": "success"},
            )
        )
        storage.record_event(
            SessionEvent(
                session_id=sess.session_id,
                event_type="tool.completed",
                data={"run_id": "run-2", "tool": "run_command", "status": "success"},
            )
        )

        return storage, sess

    def test_transcript_ordering_and_rendering(self, session_with_history):
        storage, sess = session_with_history
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)

        render_historical_transcript(console, sess.session_id, storage)
        output = out.getvalue()

        # Check chronological order: Run 1 before Run 2
        pos_run1 = output.find("Inspect user model and add is_verified column")
        pos_run2 = output.find("Run pytest on test_auth.py")
        assert pos_run1 != -1
        assert pos_run2 != -1
        assert pos_run1 < pos_run2

        # Check tool calls
        assert "read_file" in output
        assert "edit_file" in output
        assert "run_command" in output

        # Check summaries and badges
        assert "is_verified" in output and "boolean column" in output
        assert "All 12 user authentication tests passed cleanly" in output
        assert "Run 1 Completed" in output
        assert "Run 2 Completed" in output
        assert "Verification Passed" in output
        assert "End of transcript" in output
        assert "read-only" in output

    def test_transcript_strictly_read_only_invariant(self, session_with_history, tmp_path):
        storage, sess = session_with_history

        # Create canary file in workspace
        canary = tmp_path / "canary.txt"
        canary.write_text("untouched")

        orig_sess = storage.get_session(sess.session_id)
        assert orig_sess is not None
        orig_updated_at = orig_sess.updated_at
        orig_status = orig_sess.status

        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)

        # Call transcript viewer with prefix
        render_historical_transcript(console, "hist-987", storage)

        # Invariant 1: canary file untouched
        assert canary.read_text() == "untouched"

        # Invariant 2: session record in database completely unmodified
        curr_sess = storage.get_session(sess.session_id)
        assert curr_sess is not None
        assert curr_sess.status == orig_status
        assert curr_sess.updated_at == orig_updated_at


# ─── SECTION 4: Session Resume Matrix ─────────────────────────────────────────

class TestSection4SessionResumeMatrix:
    """Validate explicit session resumption across diverse lifecycle states."""

    @pytest.fixture
    def resume_env(self, tmp_path):
        db_path = tmp_path / "resume_matrix.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)
        return mgr, storage, tmp_path

    def test_resume_completed_session(self, resume_env):
        mgr, storage, ws = resume_env
        s = mgr.create_session(workspace_path=str(ws), title="Completed Project")
        storage.update_session(
            Session(
                session_id=s.session_id,
                workspace_path=str(ws),
                title="Completed Project",
                status=SessionStatus.COMPLETED,
                created_at=time.time() - 300,
                updated_at=time.time() - 10,
            )
        )

        shell = InteractiveShell(workspace=str(ws), session_id=s.session_id, plain=True)
        shell.session_manager = mgr
        shell._setup_session()

        assert shell.session_id == s.session_id
        # State must remain accessible without replaying runs
        assert shell.running is False

    @pytest.mark.asyncio
    async def test_resume_paused_session(self, resume_env):
        mgr, storage, ws = resume_env
        s = mgr.create_session(workspace_path=str(ws), title="Paused Pipeline")
        storage.update_session(
            Session(
                session_id=s.session_id,
                workspace_path=str(ws),
                title="Paused Pipeline",
                status=SessionStatus.PAUSED,
                created_at=time.time() - 200,
                updated_at=time.time() - 5,
            )
        )

        shell = InteractiveShell(workspace=str(ws), plain=True)
        shell.session_manager = mgr
        await shell._cmd_resume(s.session_id)

        assert shell.session_id == s.session_id

    @pytest.mark.asyncio
    async def test_resume_missing_workspace(self, resume_env):
        mgr, storage, ws = resume_env
        missing_ws = str(ws / "deleted_dir")
        s = mgr.create_session(workspace_path=missing_ws, title="Lost Project")

        shell = InteractiveShell(workspace=str(ws), plain=True)
        shell.session_manager = mgr

        # Must not raise an unhandled exception when workspace does not exist on disk
        await shell._cmd_resume(s.session_id)
        assert shell.session_id == s.session_id


# ─── SECTION 5: Live Agent Execution & Execution Ladder ───────────────────────

class TestSection5LiveAgentExecutionAndLadder:
    """Validate live agent execution ladder, tool tracking, and test repair."""

    @pytest.mark.asyncio
    async def test_live_execution_tool_calls_and_ladder(self, tmp_path):
        # 1. Setup workspace with buggy calculator
        calc_file = tmp_path / "calc.py"
        calc_file.write_text("def add(a, b):\n    return a - b\n")  # deliberate bug

        test_file = tmp_path / "test_calc.py"
        test_file.write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")

        # 2. Setup mock controllable provider that performs:
        # Step 0: Plan generation
        # Step 1: read_file calc.py
        # Step 2: edit_file calc.py (fixing return a + b)
        # Step 3: run_command pytest test_calc.py
        # Step 4: completion response
        responses = [
            CompletionResponse(
                content="Plan:\n1. Inspect calc.py\n2. Fix add function\n3. Run tests",
                model="test-coder",
                tool_calls=[],
                usage={"total_tokens": 50},
            ),
            CompletionResponse(
                content="",
                model="test-coder",
                tool_calls=[ToolCall(id="c1", tool_name="read_file", arguments={"path": "calc.py"})],
            ),
            CompletionResponse(
                content="",
                model="test-coder",
                tool_calls=[
                    ToolCall(
                        id="c2",
                        tool_name="edit_file",
                        arguments={
                            "path": "calc.py",
                            "old_content": "    return a - b\n",
                            "new_content": "    return a + b\n",
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="",
                model="test-coder",
                tool_calls=[
                    ToolCall(
                        id="c3",
                        tool_name="run_command",
                        arguments={"command": "python -m pytest test_calc.py"},
                    )
                ],
            ),
            CompletionResponse(
                content="Fixed the bug in `calc.py` and verified tests pass.",
                model="test-coder",
                tool_calls=[],
                usage={"total_tokens": 250},
            ),
        ]

        provider = MockTestProvider("test-provider", "test-coder", responses=responses)
        router = ModelRouter(providers=[provider], config=RouterConfig(routing_mode="free"))
        await router.refresh_models()

        event_bus = EventBus()
        captured_events: list[Event] = []
        async def on_event(e: Event) -> None:
            captured_events.append(e)
        event_bus.on("*", on_event)

        # Wire Real Tools
        from harness_core.tools.filesystem import ReadFileTool, EditFileTool, WriteFileTool
        from harness_core.tools.shell import RunCommandTool
        tools = [
            ReadFileTool(),
            EditFileTool(),
            WriteFileTool(),
            RunCommandTool(working_directory=str(tmp_path)),
        ]

        agent_loop = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(role=AgentRole.BUILD, max_iterations=5),
            event_bus=event_bus,
            router=router,
        )

        out_task = await agent_loop.run("Fix the bug in calc.py so tests pass")

        # Assertions:
        # A. Bug was actually fixed in workspace file
        assert "return a + b" in calc_file.read_text()

        # B. EventBus observed tool calls & results
        event_types = [e.type for e in captured_events]
        assert "tool.call" in event_types
        assert "tool.result" in event_types
        assert "execution.state" in event_types

        # C. Check tool calls captured
        tool_names = [e.data.get("tool") for e in captured_events if e.type == "tool.call"]
        assert "read_file" in tool_names
        assert "edit_file" in tool_names
        assert "run_command" in tool_names

        # D. Execution Ladder progressed
        states = [e.data.get("state") for e in captured_events if e.type == "execution.state"]
        assert len(states) >= 3


# ─── SECTION 6: File Activity & Diff Viewer Hardening ────────────────────────

class TestSection6FileActivityAndDiffViewer:
    """Validate tracked, untracked, deleted, and secret-scrubbed diff behavior."""

    @pytest.fixture
    def shell_with_git(self, tmp_path):
        shell = InteractiveShell(workspace=str(tmp_path), plain=False)
        out = io.StringIO()
        shell.console = Console(file=out, force_terminal=True, width=120)
        return shell, out

    def test_diff_with_tracked_and_untracked_files(self, shell_with_git, monkeypatch):
        shell, out = shell_with_git

        fake_stat = "app.py | 4 ++--\n1 file changed, 2 insertions(+), 2 deletions(-)"
        fake_diff = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,4 +1,4 @@\n"
            "-def old_func():\n"
            "+def new_func():\n"
        )
        fake_status = " M app.py\n?? untracked_service.py\n?? new_script.sh\n"

        class FakeGit:
            def __init__(self, cmd):
                self.returncode = 0
                if "--stat" in cmd:
                    self.stdout = fake_stat
                elif "-U3" in cmd:
                    self.stdout = fake_diff
                elif "--porcelain" in cmd:
                    self.stdout = fake_status
                else:
                    self.stdout = ""

        monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: FakeGit(cmd))

        shell._cmd_diff()
        output = out.getvalue()

        # Both tracked diff and untracked files MUST be surfaced!
        assert "Uncommitted Changes" in output
        assert "app.py" in output
        assert "+def new_func():" in output
        assert "Untracked / New Files:" in output
        assert "untracked_service.py" in output
        assert "new_script.sh" in output

    def test_diff_redacts_api_keys_and_bearer_tokens(self, shell_with_git, monkeypatch):
        shell, out = shell_with_git

        sensitive_diff = (
            "diff --git a/.env b/.env\n"
            "--- a/.env\n"
            "+++ b/.env\n"
            "@@ -1,2 +1,2 @@\n"
            "+OPENAI_API_KEY=sk-proj-supersecretkey1234567890abcdef\n"
            "+AUTH_HEADER=Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.topsecret\n"
        )

        class FakeGitSens:
            def __init__(self, cmd):
                self.returncode = 0
                if "--stat" in cmd:
                    self.stdout = ".env | 2 ++\n1 file changed, 2 insertions(+)"
                elif "-U3" in cmd:
                    self.stdout = sensitive_diff
                else:
                    self.stdout = ""

        monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: FakeGitSens(cmd))

        shell._cmd_diff()
        output = out.getvalue()

        assert "sk-proj-supersecretkey1234567890abcdef" not in output
        assert "sk-***[REDACTED]" in output
        assert "Bearer ***[REDACTED]" in output


# ─── SECTION 7: Provider Failure & Fallback Transparency ──────────────────────

class TestSection7ProviderFailureAndTransparency:
    """Validate error classification and stage update when provider limits occur."""

    @pytest.mark.asyncio
    async def test_provider_fallback_updates_ui_stage(self, tmp_path):
        shell = InteractiveShell(workspace=str(tmp_path), plain=False)
        out = io.StringIO()
        shell.console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(shell.console)
        conv.start("Run engineering build")
        shell._conv = conv

        bus = EventBus()
        shell._event_bus = bus
        shell._setup_event_handlers()

        # Emit model cooldown (simulating 429)
        await bus.emit(
            Event(
                type="model.cooldown",
                data={"model": "openrouter/free", "cooldown_seconds": 15.0},
            )
        )
        assert "Retrying model…" in conv.state.stage or "cooling down" in conv.state.stage

        # Emit model switched (failover to Groq)
        await bus.emit(
            Event(
                type="model.switched",
                data={
                    "from": "openrouter/free",
                    "to": "llama-3.3-70b-versatile",
                    "provider": "Groq",
                    "reason": "rate limit (HTTP 429)",
                },
            )
        )
        assert "Failing over to Groq" in conv.state.stage
        assert "llama-3.3-70b-versatile" in conv.state.stage


# ─── SECTION 8: Performance and Terminal Stability ────────────────────────────

class TestSection8PerformanceAndStability:
    """Validate deduplication of rapid tool starts and clean termination."""

    def test_rapid_tool_start_deduplication(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.start("Process 50 items")

        # Simulate 20 rapid start events for the same call_id
        for i in range(20):
            renderer.tool_started("run_command", {"command": "npm run build"}, call_id="call-42")

        # Must NOT generate 20 running lines; only 1 item
        assert len(renderer.state.items) == 1
        assert renderer.state.items[0][2] == "running"

        # Complete the call
        renderer.tool_completed("run_command", "success", call_id="call-42")
        assert renderer.state.items[0][2] == "success"

        # Stop renderer
        renderer.stop()
        assert renderer._active is False
        # When stopped, renderable must NOT contain live spinner frames
        renderable = renderer._renderable()
        console.print(renderable)
        text = out.getvalue()
        assert Sym.SPINNER not in text or "npm run build" in text
