"""Unit tests for Harness Next features:
- Interactive session commands (/continue, /resume, /recap, /new, /rename, /archive, /export, /compact, /ask, /plan, /review)
- CLI flags (--continue, --resume)
- File mention expansion (@file)
- Context compaction in interactive sessions
- Workspace reconciliation and checkpointing
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from harness_core.cli.interactive import InteractiveShell, expand_file_mentions
from harness_core.context.compaction import AgentMessage, ContextCompactor
from harness_core.session.domain import (
    Checkpoint,
    MemoryItem,
    MemoryType,
    Run,
    RunStatus,
    Session,
    SessionEvent,
    SessionStatus,
)
from harness_core.session.manager import SessionManager
from harness_core.session.storage import SessionStorage


@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "main.py").write_text("print('hello world')", encoding="utf-8")
    (ws / "utils.py").write_text("def add(a, b): return a + b", encoding="utf-8")
    (ws / ".harness").mkdir()
    return ws


@pytest.fixture
def session_manager(temp_workspace):
    db_path = temp_workspace / ".harness" / "test_sessions.db"
    storage = SessionStorage(db_path=str(db_path))
    return SessionManager(storage=storage)


class TestFileMentionExpansion:
    def test_expand_file_mentions_valid_file(self, temp_workspace):
        text = "Please examine @main.py and explain it"
        expanded = expand_file_mentions(text, temp_workspace)
        assert "[Referenced File: main.py]" in expanded
        assert "print('hello world')" in expanded
        assert "Please examine @main.py and explain it" in expanded

    def test_expand_file_mentions_multiple_files(self, temp_workspace):
        text = "Compare @main.py with @utils.py"
        expanded = expand_file_mentions(text, temp_workspace)
        assert "[Referenced File: main.py]" in expanded
        assert "[Referenced File: utils.py]" in expanded
        assert "def add(a, b)" in expanded

    def test_expand_file_mentions_non_file_tokens(self, temp_workspace):
        text = "Contact user@example.com or use @property decorator"
        expanded = expand_file_mentions(text, temp_workspace)
        assert expanded == text

    def test_expand_file_mentions_missing_file(self, temp_workspace):
        text = "Check @does_not_exist.py"
        expanded = expand_file_mentions(text, temp_workspace)
        assert expanded == text


class TestInteractiveSessionCommands:
    @pytest.mark.asyncio
    async def test_cmd_new_and_rename_and_archive(self, temp_workspace, session_manager):
        console = Console(record=True)
        shell = InteractiveShell(workspace=str(temp_workspace), plain=True)
        shell.console = console
        shell.session_manager = session_manager

        # /new
        shell._cmd_new_session("Feature Branch Session")
        assert shell.session_id is not None
        sess = session_manager.storage.get_session(shell.session_id)
        assert sess is not None
        assert sess.title == "Feature Branch Session"

        # /rename
        shell._cmd_rename_session("Renamed Feature Session")
        sess_renamed = session_manager.storage.get_session(shell.session_id)
        assert sess_renamed.title == "Renamed Feature Session"

        # /recap
        shell._cmd_recap()
        out = console.export_text()
        assert "Workspace:" in out

        # /archive
        shell._cmd_archive_session()
        assert shell.session_id is None
        sess_archived = session_manager.storage.get_session(sess.session_id)
        assert sess_archived.status == SessionStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_cmd_continue_and_resume(self, temp_workspace, session_manager):
        console = Console(record=True)
        shell = InteractiveShell(workspace=str(temp_workspace), plain=True)
        shell.console = console
        shell.session_manager = session_manager

        # Create 2 sessions
        s1 = session_manager.create_session(workspace_path=str(temp_workspace), title="Session 1")
        s2 = session_manager.create_session(workspace_path=str(temp_workspace), title="Session 2")

        # /continue should pick s2 (latest)
        shell._cmd_continue()
        assert shell.session_id == s2.session_id

        # /resume with s1 ID
        await shell._cmd_resume(s1.session_id)
        assert shell.session_id == s1.session_id

        # /sessions listing
        await shell._cmd_session("list")
        out = console.export_text()
        assert s1.session_id[:10] in out
        assert s2.session_id[:10] in out

    @pytest.mark.asyncio
    async def test_cmd_export(self, temp_workspace, session_manager):
        console = Console(record=True)
        shell = InteractiveShell(workspace=str(temp_workspace), plain=True)
        shell.console = console
        shell.session_manager = session_manager

        s = session_manager.create_session(workspace_path=str(temp_workspace), title="Export Test")
        shell.session_id = s.session_id

        shell._cmd_export_session("markdown")
        export_file = temp_workspace / ".harness" / f"session_{s.session_id[:8]}.md"
        assert export_file.exists()
        assert "Export Test" in export_file.read_text(encoding="utf-8")

        shell._cmd_export_session("json")
        export_json = temp_workspace / ".harness" / f"session_{s.session_id[:8]}.json"
        assert export_json.exists()
        assert "Export Test" in export_json.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_cmd_compact(self, temp_workspace, session_manager):
        console = Console(record=True)
        shell = InteractiveShell(workspace=str(temp_workspace), plain=True)
        shell.console = console
        shell.session_manager = session_manager

        s = session_manager.create_session(workspace_path=str(temp_workspace), title="Compact Test")
        shell.session_id = s.session_id

        # Add some dialogue events
        for i in range(12):
            session_manager.storage.record_event(
                SessionEvent(
                    session_id=s.session_id,
                    event_type="user.message" if i % 2 == 0 else "assistant.message",
                    data={"message": f"This is message step {i} with detailed information " * 10},
                )
            )

        shell._cmd_compact()
        out = console.export_text()
        assert "Context compacted" in out or "messages summarized" in out or "Context is already compact" in out

    @pytest.mark.asyncio
    async def test_specialist_modes_routing(self, temp_workspace, session_manager):
        console = Console(record=True)
        shell = InteractiveShell(workspace=str(temp_workspace), plain=True)
        shell.console = console
        shell.session_manager = session_manager

        executed_tasks = []

        async def fake_execute(goal):
            executed_tasks.append(goal)
            return "OK"

        shell._execute_task = fake_execute

        # /ask
        await shell._handle_command("/ask How does authentication work?")
        assert len(executed_tasks) == 1
        assert "[READ-ONLY INQUIRY" in executed_tasks[0]
        assert "How does authentication work?" in executed_tasks[0]

        # /plan <task>
        await shell._handle_command("/plan Add SQLite caching to the model router")
        assert len(executed_tasks) == 2
        assert "[PLAN-ONLY MODE" in executed_tasks[1]
        assert "Add SQLite caching" in executed_tasks[1]

        # /review (no git changes in mock, prints no changes detected)
        await shell._handle_command("/review")
        out = console.export_text()
        assert "No uncommitted git changes" in out or len(executed_tasks) == 3


class TestCLIResumeFlags:
    def test_main_callback_options_exist(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--continue" in result.stdout or "-c" in result.stdout
        assert "--resume" in result.stdout or "-r" in result.stdout

    def test_shell_command_options_exist(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["shell", "--help"])
        assert result.exit_code == 0
        assert "--continue" in result.stdout or "-c" in result.stdout
        assert "--resume" in result.stdout or "-r" in result.stdout
