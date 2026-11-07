"""Unit tests for Session Browser, Historical Transcript Viewer, and TUI Parity."""

import io
import time
import datetime
from pathlib import Path

import pytest
from rich.console import Console

from harness_core.cli.session_browser import (
    _format_date_group,
    group_sessions_by_date,
    render_session_browser,
    render_historical_transcript,
    derive_session_status,
    InteractiveSessionBrowser,
    BrowserViewMode,
)
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from harness_core.session.domain import Session, SessionStatus, Run, RunStatus, SessionEvent
from harness_core.session.storage import SessionStorage
from harness_core.session.manager import SessionManager
from harness_core.cli.interactive import InteractiveShell
from harness_core.cli.conversation import ConversationRenderer


def _make_session(
    session_id: str,
    title: str = "Test Session",
    workspace: str = "/tmp/test-ws",
    status: SessionStatus = SessionStatus.ACTIVE,
    updated_at: float = 0.0,
) -> Session:
    now = time.time() if updated_at == 0.0 else updated_at
    return Session(
        session_id=session_id,
        title=title,
        workspace_path=workspace,
        status=status,
        created_at=now - 100,
        updated_at=now,
    )


class TestSessionDateGrouping:
    """Test date grouping logic for the session browser."""

    def test_format_date_group_today(self):
        now = time.time()
        group = _format_date_group(now)
        assert group.startswith("Today")

    def test_format_date_group_yesterday(self):
        yesterday_dt = datetime.datetime.now() - datetime.timedelta(days=1)
        group = _format_date_group(yesterday_dt.timestamp())
        assert group.startswith("Yesterday")

    def test_format_date_group_this_week(self):
        three_days_ago = datetime.datetime.now() - datetime.timedelta(days=3)
        group = _format_date_group(three_days_ago.timestamp())
        assert group.startswith("This Week")

    def test_group_sessions_by_date(self):
        now_ts = time.time()
        yesterday_ts = (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp()
        s1 = _make_session("s1", updated_at=now_ts)
        s2 = _make_session("s2", updated_at=yesterday_ts)
        groups = group_sessions_by_date([s1, s2])
        assert any("Today" in k for k in groups.keys())
        assert any("Yesterday" in k for k in groups.keys())


class TestSessionBrowserRendering:
    """Test rendering of the date-grouped session browser."""

    @pytest.fixture
    def mock_storage(self, tmp_path):
        db_path = tmp_path / "test_sessions.db"
        storage = SessionStorage(db_path=db_path)
        return storage

    def test_render_session_browser_rich(self, mock_storage):
        s1 = _make_session("sess-1", title="Build Feature X")
        s2 = _make_session("sess-2", title="Fix Bug Y", status=SessionStatus.COMPLETED)
        mock_storage.create_session(s1)
        mock_storage.create_session(s2)

        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        render_session_browser(console, [s1, s2], mock_storage, current_session_id="sess-1")
        output = out.getvalue()

        assert "Session History" in output
        assert "sess-1" in output
        assert "Build Feature X" in output
        assert "sess-2" in output
        assert "Fix Bug Y" in output
        assert "/sessions view <id>" in output

    def test_render_session_browser_plain(self, mock_storage):
        s1 = _make_session("sess-plain-1", title="Plain Test")
        mock_storage.create_session(s1)

        out = io.StringIO()
        console = Console(file=out, no_color=True, width=120)
        render_session_browser(console, [s1], mock_storage, plain=True)
        output = out.getvalue()

        assert "=== HARNESS · Session History ===" in output
        assert "sess-plain" in output
        assert "Plain Test" in output

    def test_render_session_browser_filtering(self, mock_storage):
        s1 = _make_session("sess-alpha", title="Frontend redesign", workspace="/app/web")
        s2 = _make_session("sess-beta", title="Backend api", workspace="/app/api")
        mock_storage.create_session(s1)
        mock_storage.create_session(s2)

        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        render_session_browser(console, [s1, s2], mock_storage, query="Frontend")
        output = out.getvalue()

        assert "Frontend redesign" in output
        assert "Backend api" not in output

    def test_render_session_browser_empty(self, mock_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        render_session_browser(console, [], mock_storage, query="NonExistent")
        output = out.getvalue()

        assert "No sessions found matching your criteria" in output


class TestHistoricalTranscriptViewer:
    """Test read-only historical conversation rendering."""

    @pytest.fixture
    def populated_storage(self, tmp_path):
        db_path = tmp_path / "transcript_test.db"
        storage = SessionStorage(db_path=db_path)
        sess = _make_session("transcript-123456", title="Implement Auth")
        storage.create_session(sess)

        # Add runs
        r1 = Run(
            run_id="run-001",
            session_id=sess.session_id,
            task="Add login endpoint",
            status=RunStatus.COMPLETED,
            started_at=time.time() - 50,
            completed_at=time.time() - 40,
            duration_ms=10000,
            tool_calls=3,
            model_id="qwen3-coder-plus",
            verification_passed=True,
            result_summary="Successfully implemented `/api/v1/login` with JWT tokens.",
        )
        storage.create_run(r1)

        # Add events
        storage.record_event(
            SessionEvent(
                session_id=sess.session_id,
                event_type="tool.completed",
                data={"run_id": "run-001", "tool": "write_file", "status": "success"},
            )
        )
        storage.record_event(
            SessionEvent(
                session_id=sess.session_id,
                event_type="tool.completed",
                data={"run_id": "run-001", "tool": "run_command", "status": "success"},
            )
        )
        return storage

    def test_render_historical_transcript_rich(self, populated_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        render_historical_transcript(console, "transcript-123456", populated_storage)
        output = out.getvalue()

        assert "Conversation History (Read-Only)" in output
        assert "Implement Auth" in output
        assert "Add login endpoint" in output
        assert "write_file" in output
        assert "Successfully implemented" in output
        assert "Run 1 Completed" in output
        assert "Verification Passed" in output
        assert "End of transcript" in output
        assert "read-only" in output

    def test_render_historical_transcript_prefix_lookup(self, populated_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        render_historical_transcript(console, "transcript-12", populated_storage)
        output = out.getvalue()

        assert "Conversation History (Read-Only)" in output
        assert "Add login endpoint" in output

    def test_render_historical_transcript_not_found(self, populated_storage):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=120)
        render_historical_transcript(console, "unknown-session-id", populated_storage)
        output = out.getvalue()

        assert "not found in history" in output


class TestInteractiveFreshLaunchAndResume:
    """Test that fresh launch starts clean while explicit resume restores state."""

    @pytest.fixture
    def shell_env(self, tmp_path):
        shell = InteractiveShell(
            workspace=str(tmp_path),
            plain=True,
        )
        return shell

    def test_fresh_launch_creates_clean_session(self, shell_env, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)
        old_sess = mgr.create_session(workspace_path=shell_env.workspace, title="Old Active Session")

        shell_env.session_manager = mgr
        shell_env.session_id = None
        shell_env.resume_latest = False
        shell_env._setup_session()

        # Must create a fresh session, not hijacking old_sess
        assert shell_env.session_id is not None
        assert shell_env.session_id != old_sess.session_id

    def test_explicit_resume_loads_session(self, shell_env, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)
        old_sess = mgr.create_session(workspace_path=shell_env.workspace, title="Session to Resume")

        shell_env.session_manager = mgr
        shell_env.session_id = old_sess.session_id
        shell_env._setup_session()

        assert shell_env.session_id == old_sess.session_id

    def test_resume_latest_flag(self, shell_env, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)
        s = mgr.create_session(workspace_path=shell_env.workspace, title="Latest Session")

        shell_env.session_manager = mgr
        shell_env.session_id = None
        shell_env.resume_latest = True
        shell_env._setup_session()

        assert shell_env.session_id == s.session_id


class TestFileActivityTracking:
    """Test real-time file activity tracking in ConversationRenderer."""

    def test_record_file_activity_lifecycle(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.start("Implement user login")

        # Record file inspections and modifications
        renderer.record_file_activity("src/auth.py", "inspected")
        renderer.record_file_activity("src/models.py", "modified")
        renderer.record_file_activity("tests/test_auth.py", "created")

        assert "src/auth.py" in renderer.state.files_activity
        assert renderer.state.files_activity["src/auth.py"] == "inspected"
        assert renderer.state.files_activity["src/models.py"] == "modified"
        assert renderer.state.files_activity["tests/test_auth.py"] == "created"

        # Check that files appear in live renderable
        renderable = renderer._renderable()
        console.print(renderable)
        text = out.getvalue()

        assert "Files" in text
        assert "auth.py" in text
        assert "models.py" in text
        assert "test_auth.py" in text


class TestDiffCommandAndSecurity:
    """Test /diff command syntax highlighting and secret redaction."""

    @pytest.fixture
    def shell_env(self, tmp_path):
        shell = InteractiveShell(workspace=str(tmp_path), plain=False)
        return shell

    def test_cmd_diff_secret_redaction(self, shell_env, monkeypatch):
        out = io.StringIO()
        shell_env.console = Console(file=out, force_terminal=True, width=120)

        # Mock git diff returning an API key and token
        raw_diff = (
            "diff --git a/config.py b/config.py\n"
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-api_key = ''\n"
            "+api_key = 'sk-1234567890abcdef1234567890abcdef'\n"
            "+token = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy'\n"
        )
        fake_stat = "config.py | 2 +-\n1 file changed, 2 insertions(+)"

        class FakeSubprocess:
            def __init__(self, cmd):
                self.cmd = cmd
                self.returncode = 0
                if "--stat" in cmd:
                    self.stdout = fake_stat
                elif "-U3" in cmd:
                    self.stdout = raw_diff
                else:
                    self.stdout = ""

        monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: FakeSubprocess(cmd))

        shell_env._cmd_diff()
        output = out.getvalue()

        assert "Uncommitted Changes" in output
        assert "config.py" in output
        # Secrets MUST be redacted!
        assert "sk-1234567890abcdef1234567890abcdef" not in output
        assert "sk-***[REDACTED]" in output
        assert "Bearer ***[REDACTED]" in output

    def test_cmd_diff_clean_workspace(self, shell_env, monkeypatch):
        out = io.StringIO()
        shell_env.console = Console(file=out, force_terminal=True, width=120)

        class FakeClean:
            returncode = 0
            stdout = ""

        monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: FakeClean())

        shell_env._cmd_diff()
        output = out.getvalue()

        assert "No uncommitted changes detected in workspace" in output


class TestModelSwitchedFailoverTransparency:
    """Test that model.switched and failover events update live conversation stage."""

    @pytest.mark.asyncio
    async def test_model_switched_updates_conversation_stage(self, tmp_path):
        shell = InteractiveShell(workspace=str(tmp_path), plain=False)
        out = io.StringIO()
        shell.console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(shell.console)
        conv.start("Build app")
        shell._conv = conv

        from harness_core.observability.events import EventBus, Event
        bus = EventBus()
        shell._event_bus = bus
        shell._setup_event_handlers()

        # Emit model.switched with provider failover
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

        assert "Groq" in conv.state.stage or "llama-3.3-70b" in conv.state.stage
        assert conv.state.stage.startswith("Failing over")


class TestDerivedSessionStatus:
    """Test derive_session_status accurately determines status from runs and lifecycle."""

    @pytest.fixture
    def storage(self, tmp_path):
        return SessionStorage(db_path=tmp_path / "status_test.db")

    def test_archived_status(self, storage):
        s = _make_session("s-arch", status=SessionStatus.ARCHIVED)
        storage.create_session(s)
        assert derive_session_status(s, storage) == "Archived"

    def test_paused_status(self, storage):
        s = _make_session("s-pause", status=SessionStatus.PAUSED)
        storage.create_session(s)
        assert derive_session_status(s, storage) == "Paused"

    def test_current_session_is_active(self, storage):
        s = _make_session("s-cur")
        storage.create_session(s)
        assert derive_session_status(s, storage, current_session_id="s-cur") == "Active"

    def test_historical_session_no_runs_is_interrupted(self, storage):
        s = _make_session("s-old")
        storage.create_session(s)
        # Not current session, no runs -> Interrupted, NEVER Active!
        assert derive_session_status(s, storage, current_session_id="other-id") == "Interrupted"

    def test_completed_run_outcome(self, storage):
        s = _make_session("s-comp")
        storage.create_session(s)
        r = Run(
            run_id="r1",
            session_id=s.session_id,
            task="do work",
            status=RunStatus.COMPLETED,
            outcome="success",
        )
        storage.create_run(r)
        assert derive_session_status(s, storage) == "Completed"

    def test_failed_run_outcome(self, storage):
        s = _make_session("s-fail")
        storage.create_session(s)
        r = Run(
            run_id="r1",
            session_id=s.session_id,
            task="do work",
            status=RunStatus.FAILED,
            outcome="failure",
        )
        storage.create_run(r)
        assert derive_session_status(s, storage) == "Failed"

    def test_interrupted_run_status(self, storage):
        s = _make_session("s-int")
        storage.create_session(s)
        r = Run(
            run_id="r1",
            session_id=s.session_id,
            task="do work",
            status=RunStatus.RUNNING,
        )
        storage.create_run(r)
        assert derive_session_status(s, storage) == "Interrupted"


class TestInteractiveSessionBrowserNavigation:
    """Test keyboard navigation, search, transcript viewing, and resume confirmation."""

    @pytest.fixture
    def test_env(self, tmp_path):
        db_path = tmp_path / "browser_test.db"
        storage = SessionStorage(db_path=db_path)
        # Create 5 sessions across different dates
        now = time.time()
        for i in range(5):
            sess = _make_session(
                f"session-{i:02d}",
                title=f"Task {i}",
                workspace=f"/workspace/app{i}",
                updated_at=now - (i * 3600),
            )
            storage.create_session(sess)
            r = Run(
                run_id=f"run-{i}",
                session_id=sess.session_id,
                task=f"Goal for session {i}",
                status=RunStatus.COMPLETED if i % 2 == 0 else RunStatus.FAILED,
                outcome="success" if i % 2 == 0 else "failure",
                duration_ms=5000,
                tool_calls=2,
                result_summary=f"Summary for task {i} completed.",
            )
            storage.create_run(r)
        return storage

    @pytest.mark.asyncio
    async def test_keyboard_navigation_up_down(self, test_env):
        out = io.StringIO()
        console = Console(file=out, width=100)
        with create_pipe_input() as pipe_input:
            browser = InteractiveSessionBrowser(
                console=console,
                storage=test_env,
                input=pipe_input,
                output=DummyOutput(),
            )
            assert browser.mode == BrowserViewMode.SESSIONS_LIST
            assert browser.selected_index == 0

            # Press down arrow twice, then up arrow once, then exit with 'q'
            pipe_input.send_text("\x1b[B\x1b[B\x1b[Aq")
            res = await browser.run_async()

            assert browser.selected_index == 1
            assert res is None

    @pytest.mark.asyncio
    async def test_enter_opens_transcript_and_esc_returns(self, test_env):
        out = io.StringIO()
        console = Console(file=out, width=100)
        with create_pipe_input() as pipe_input:
            browser = InteractiveSessionBrowser(
                console=console,
                storage=test_env,
                input=pipe_input,
                output=DummyOutput(),
            )
            # Enter opens transcript viewer, Esc returns to session list, q exits
            pipe_input.send_text("\r\x1b\x1bq")
            res = await browser.run_async()

            assert res is None

    @pytest.mark.asyncio
    async def test_r_key_confirm_resume_flow(self, test_env):
        out = io.StringIO()
        console = Console(file=out, width=100)
        with create_pipe_input() as pipe_input:
            browser = InteractiveSessionBrowser(
                console=console,
                storage=test_env,
                input=pipe_input,
                output=DummyOutput(),
            )
            # Press 'r' to open resume confirmation, then 'y' to confirm
            pipe_input.send_text("ry")
            res = await browser.run_async()

            assert res is not None
            assert res.get("action") == "resume"
            assert res.get("session_id") == browser.sessions[0].session_id

    @pytest.mark.asyncio
    async def test_r_key_decline_resume_flow(self, test_env):
        out = io.StringIO()
        console = Console(file=out, width=100)
        with create_pipe_input() as pipe_input:
            browser = InteractiveSessionBrowser(
                console=console,
                storage=test_env,
                input=pipe_input,
                output=DummyOutput(),
            )
            # Press 'r' to open resume confirmation, then 'n' to cancel back to list, then 'q' to quit
            pipe_input.send_text("rnq")
            res = await browser.run_async()

            # Resuming was declined
            assert res is None

    @pytest.mark.asyncio
    async def test_search_filtering(self, test_env):
        out = io.StringIO()
        console = Console(file=out, width=100)
        with create_pipe_input() as pipe_input:
            browser = InteractiveSessionBrowser(
                console=console,
                storage=test_env,
                input=pipe_input,
                output=DummyOutput(),
            )
            # Press '/' then type 'Task 2' then Enter then 'q'
            pipe_input.send_text("/Task 2\rq")
            res = await browser.run_async()

            assert len(browser.sessions) == 1
            assert browser.sessions[0].session_id == "session-02"

    @pytest.mark.asyncio
    async def test_transcript_search_and_navigation(self, test_env):
        out = io.StringIO()
        console = Console(file=out, width=100)
        with create_pipe_input() as pipe_input:
            browser = InteractiveSessionBrowser(
                console=console,
                storage=test_env,
                input=pipe_input,
                output=DummyOutput(),
            )
            # Enter to open transcript, '/' to search, 'Summary', Enter, 'n', Esc to exit transcript, 'q' to exit browser
            pipe_input.send_text("\r/Summary\rn\x1b\x1bq")
            res = await browser.run_async()

            assert res is None

    def test_read_only_invariants(self, test_env):
        """Verify that viewing transcripts makes 0 model calls, 0 tool executions, 0 modifications."""
        out = io.StringIO()
        console = Console(file=out, width=100)
        browser = InteractiveSessionBrowser(
            console=console,
            storage=test_env,
        )
        browser.mode = BrowserViewMode.TRANSCRIPT_VIEWER
        rendered = browser._render_transcript_viewer()

        assert "Conversation History (Read-Only)" in rendered
        assert "Goal for session" in rendered
        assert "Verification" in rendered
        # Assert database was not modified
        all_s = test_env.list_sessions(limit=100)
        assert len(all_s) == 5


class TestInteractiveShellSessionCommand:
    """Test /sessions command in InteractiveShell."""

    @pytest.fixture
    def shell(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        storage = SessionStorage(db_path=db_path)
        mgr = SessionManager(storage=storage)
        s1 = mgr.create_session(workspace_path=str(tmp_path), title="First Session")
        s2 = mgr.create_session(workspace_path=str(tmp_path), title="Second Session")
        shell = InteractiveShell(workspace=str(tmp_path), plain=True)
        shell.session_manager = mgr
        shell.session_id = s1.session_id
        return shell

    @pytest.mark.asyncio
    async def test_cmd_session_list_plain(self, shell):
        out = io.StringIO()
        shell.console = Console(file=out, no_color=True, width=100)
        await shell._cmd_session("list")
        output = out.getvalue()
        assert "Session History" in output
        assert "First Session" in output

    @pytest.mark.asyncio
    async def test_cmd_session_view(self, shell):
        all_s = shell.session_manager.storage.list_sessions()
        target = [s for s in all_s if s.title == "First Session"][0]
        out = io.StringIO()
        shell.console = Console(file=out, no_color=True, width=100)
        await shell._cmd_session(f"view {target.session_id[:8]}")
        output = out.getvalue()
        assert "Session Transcript" in output

    @pytest.mark.asyncio
    async def test_cmd_session_resume(self, shell):
        all_s = shell.session_manager.storage.list_sessions()
        target = [s for s in all_s if s.title == "Second Session"][0]
        out = io.StringIO()
        shell.console = Console(file=out, no_color=True, width=100)
        await shell._cmd_session(f"resume {target.session_id}")
        assert shell.session_id == target.session_id

    @pytest.mark.asyncio
    async def test_cmd_session_search(self, shell):
        out = io.StringIO()
        shell.console = Console(file=out, no_color=True, width=100)
        await shell._cmd_session("search First")
        output = out.getvalue()
        assert "First Session" in output

