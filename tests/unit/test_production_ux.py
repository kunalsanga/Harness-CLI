"""Regression tests for the production UX & runtime-ownership pass.

Covers: runtime-owned TODO state, actionable TODO validation, tool activity
rendering, live status rendering, quiet model switching, rate-limit pause
(state preservation), structured git results + diagnosis, workspace cwd,
path confinement, command timeouts, cancellation reporting, truthful
completion, and final execution accounting.
"""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from harness_core.agent.loop import AgentLoop
from harness_core.agent.report import build_execution_report, files_from_task
from harness_core.agent.todos import fallback_todo_plan, is_actionable_todo, sanitize_todo_titles
from harness_core.agent.types import (
    AgentConfig,
    Task,
    TaskPlan,
    TaskStatus,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.cli.interactive import InteractiveShell, LiveStatus
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.routing.fallback import FallbackResult
from harness_core.tools.base import Tool, ToolSchema
from harness_core.tools.filesystem import ReadFileTool, WriteFileTool
from harness_core.tools.shell import RunCommandTool


# ── Scripted provider ────────────────────────────────────────────────────


class ScriptedProvider(ModelProvider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = responses
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = CompletionResponse(content="Done.", model="scripted", provider="scripted")
        self._call_count += 1
        return resp

    async def stream(self, request: CompletionRequest):
        yield CompletionResponse(content="", model="scripted", provider="scripted")

    async def list_models(self) -> list[Any]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def text(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, model="scripted", provider="scripted")


def plan_response(steps: list[str]) -> CompletionResponse:
    return text("\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)))


def tool_response(tool: str, args: dict[str, Any], call_id: str = "call-x") -> CompletionResponse:
    return CompletionResponse(
        content=None,
        model="scripted",
        provider="scripted",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }
        ],
    )


class EventCollector:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.events: list[Event] = []

        async def _collect(event: Event) -> None:
            self.events.append(event)

        self.bus.on("*", _collect)

    def of_type(self, t: str) -> list[Event]:
        return [e for e in self.events if e.type == t]


def make_loop(provider, workspace: Path, collector=None, tools=None, max_iterations=15):
    return AgentLoop(
        provider=provider,
        tools=tools if tools is not None else [ReadFileTool(), WriteFileTool(), RunCommandTool()],
        workspace_root=workspace,
        config=AgentConfig(max_iterations=max_iterations, verify_on_complete=False),
        event_bus=collector.bus if collector else EventBus(),
    )


# ── 1-2. Runtime-owned TODO state ────────────────────────────────────────


class TestRuntimeTodoState:
    @pytest.mark.asyncio
    async def test_todo_completed_after_tool_success(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(["Inspect project files", "Analyze implementation"]),
            tool_response("read_file", {"path": str(target)}, "c1"),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector)

        task = await loop.run("Tell me about this project")

        statuses = {i.description: i.status for i in task.task_plan.items}
        assert statuses["Inspect project files"] == TodoStatus.COMPLETED
        inspect_item = next(i for i in task.task_plan.items if i.description == "Inspect project files")
        assert inspect_item.evidence and inspect_item.evidence.get("tool") == "read_file"
        assert inspect_item.completed_at is not None
        # Structured events were emitted
        assert collector.of_type("todo.completed"), "todo.completed event must be emitted"

    @pytest.mark.asyncio
    async def test_todo_failed_after_tool_failure(self, tmp_path: Path):
        provider = ScriptedProvider([
            plan_response(["Run tests"]),
            tool_response("run_command", {"command": "node test.js"}, "c1"),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector)

        task = await loop.run("Execute the test suite")

        run_item = next(i for i in task.task_plan.items if i.description == "Run tests")
        assert run_item.status == TodoStatus.FAILED
        assert run_item.error
        assert collector.of_type("todo.failed"), "todo.failed event must be emitted"

    @pytest.mark.asyncio
    async def test_final_counter_reflects_real_state(self, tmp_path: Path):
        """Never show 0/N after the work actually happened."""
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(["Inspect project files", "Implement changes"]),
            tool_response("read_file", {"path": str(target)}, "c1"),
            tool_response("write_file", {"path": str(target), "content": "x = 2"}, "c2"),
            text("Done."),
        ])
        loop = make_loop(provider, tmp_path)

        task = await loop.run("Enhance this project")

        assert task.task_plan.completed_count == task.task_plan.total_count
        assert task.task_plan.completed_count > 0


# ── 3-4. Actionable TODO validation ──────────────────────────────────────


class TestActionableTodos:
    def test_rejects_conversational_prose(self):
        garbage = [
            "The project's in my scope.",
            "The file tree gives me hope.",
            "Got the HTML and CSS.",
            "Ready for deployment.",
            "The project looks good",
        ]
        assert sanitize_todo_titles(garbage) == []

    def test_requires_action_verb_first(self):
        assert is_actionable_todo("Inspect repository structure")
        assert is_actionable_todo("Run automated tests")
        assert is_actionable_todo("Update responsive styling")
        assert not is_actionable_todo("Got the files")
        assert not is_actionable_todo("I think we should look at the code")

    def test_keeps_valid_engineering_tasks(self):
        steps = [
            "Inspect repository structure",
            "Update responsive styling",
            "Run automated tests",
            "Verify changed behavior",
        ]
        assert sanitize_todo_titles(steps) == steps

    def test_deterministic_fallback_plan(self):
        steps = fallback_todo_plan("enhance this project", {"has_tests": True})
        assert steps, "fallback plan must never be empty"
        for s in steps:
            assert is_actionable_todo(s), f"fallback step not actionable: {s}"

    @pytest.mark.asyncio
    async def test_model_prose_replaced_by_fallback_plan(self, tmp_path: Path):
        provider = ScriptedProvider([
            text("The file tree gives me hope. Ready for deployment."),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, tools=[], collector=collector)

        task = await loop.run("Enhance this project")

        assert task.task_plan.items, "plan must not be empty"
        for item in task.task_plan.items:
            assert is_actionable_todo(item.description)


# ── 5-6. Rendering ───────────────────────────────────────────────────────


class TestRendering:
    def test_activity_trail_tracks_tool_activity(self):
        console = Console(file=io.StringIO(), no_color=True)
        status = LiveStatus(console, plain=True)
        status.start("Enhance calculator")
        status.update_activity("read_file", {"path": "script.js"})
        status.update_activity_complete("read_file", "success")
        status.update_activity("edit_file", {"path": "script.js"})
        assert ("✓", "read script.js") in status.trail
        assert status.trail[-1][0] == "◐"
        status.stop()

    def test_plain_mode_prints_only_on_change(self):
        buf = io.StringIO()
        console = Console(file=buf, no_color=True)
        status = LiveStatus(console, plain=True)
        status.start("task")
        lines_after_start = buf.getvalue().count("\n")
        # Same-state updates must not reprint
        status.update_phase("understanding")
        status.update_phase("understanding")
        assert buf.getvalue().count("\n") == lines_after_start
        # A real change prints once
        status.update_phase("implementing")
        assert buf.getvalue().count("\n") == lines_after_start + 1
        status.stop()

    def test_rich_renderable_contains_canonical_state(self):
        buf = io.StringIO()
        console = Console(file=buf, no_color=True, width=100)
        status = LiveStatus(console, plain=True)
        status.start("Enhance this calculator")
        status.update_phase("implementing")
        status.update_todo_items([
            {"id": "1", "title": "Inspect project", "status": "completed", "evidence": None, "error": None},
            {"id": "2", "title": "Run tests", "status": "pending", "evidence": None, "error": None},
        ])
        renderable = status._renderable()
        console.print(renderable)
        out = buf.getvalue()
        assert "HARNESS" in out
        assert "Enhance this calculator" in out
        assert "Inspect project" in out
        assert "Run tests" in out
        assert "1/2" in out

    def test_model_switch_rendered_once_per_change(self):
        shell = InteractiveShell(plain=True)
        shell.console = Console(file=io.StringIO(), no_color=True)
        shell._event_bus = EventBus()
        shell._setup_event_handlers()
        bus = shell._event_bus

        async def emit():
            await bus.emit(Event(type="model.switched", data={
                "from": "model-a:free", "to": "model-b:free", "reason": "rate limit",
            }))

        asyncio.run(emit())
        out = shell.console.file.getvalue()
        assert out.count("Model fallback") == 1
        assert "model-a:free" in out and "model-b:free" in out


# ── 8. Rate-limit pause preserves state ──────────────────────────────────


class TestRateLimitPause:
    @pytest.mark.asyncio
    async def test_model_failure_after_work_pauses_not_fails(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")

        plan_ok = FallbackResult(
            response=CompletionResponse(
                content="1. Inspect project files\n2. Implement changes",
                model="m1", provider="scripted",
            ),
            model_used="m1", succeeded=True,
        )
        iter1_ok = FallbackResult(
            response=CompletionResponse(
                content=None, model="m1", provider="scripted",
                tool_calls=[{
                    "id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": str(target)})},
                }],
            ),
            model_used="m1", succeeded=True,
        )
        rate_limited = FallbackResult(
            succeeded=False, final_error="All 3 models are rate limited (429).",
        )

        router = MagicMock()
        router.budget = MagicMock()
        router.budget.check_all.return_value = (True, None)
        router.execute = AsyncMock(side_effect=[plan_ok, iter1_ok, rate_limited, rate_limited])
        router.get_routing_decisions.return_value = []

        collector = EventCollector()
        provider = ScriptedProvider([])
        loop = AgentLoop(
            provider=provider,
            tools=[ReadFileTool(), WriteFileTool()],
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10, verify_on_complete=False),
            event_bus=collector.bus,
            router=router,
        )

        task = await loop.run("Enhance this project")

        assert task.status == TaskStatus.PAUSED
        assert task.paused_reason
        # Work evidence preserved
        assert len(task.tool_calls) == 1
        assert task.tool_calls[0].result is not None
        inspect = next(
            (i for i in task.task_plan.items if i.description == "Inspect project files"), None
        )
        assert inspect is not None and inspect.status == TodoStatus.COMPLETED
        # task.paused emitted; task not reported as failed
        assert collector.of_type("task.paused")
        completed_events = collector.of_type("task.completed")
        assert completed_events[-1].data["status"] == "paused"

    @pytest.mark.asyncio
    async def test_model_failure_without_work_fails(self, tmp_path: Path):
        rate_limited = FallbackResult(
            succeeded=False, final_error="All models are rate limited (429).",
        )
        router = MagicMock()
        router.budget = MagicMock()
        router.budget.check_all.return_value = (True, None)
        router.execute = AsyncMock(side_effect=[rate_limited, rate_limited])

        loop = AgentLoop(
            provider=ScriptedProvider([]),
            tools=[ReadFileTool()],
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=5, verify_on_complete=False),
            event_bus=EventBus(),
            router=router,
        )
        # Planning also fails via router -> fallback plan used, no work done
        task = await loop.run("Enhance this project")
        assert task.status == TaskStatus.FAILED
        assert task.error and "Provider error" in task.error


# ── 9-12. Structured Git ─────────────────────────────────────────────────


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, timeout=15)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    return tmp_path


class TestStructuredGit:
    @pytest.mark.asyncio
    async def test_status_structured_result(self, git_repo: Path):
        from harness_core.tools.git import GitStatusTool
        (git_repo / "file.txt").write_text("hello", encoding="utf-8")
        tool = GitStatusTool()
        result = await tool.execute({"cwd": str(git_repo)})
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["operation"] == "status"
        assert result.metadata["clean"] is False
        assert any("file.txt" in f for f in result.metadata["files"])

    @pytest.mark.asyncio
    async def test_commit_structured_result(self, git_repo: Path):
        from harness_core.tools.git import GitCommitTool
        (git_repo / "file.txt").write_text("hello", encoding="utf-8")
        tool = GitCommitTool()
        result = await tool.execute({"message": "Add file", "cwd": str(git_repo)})
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata.get("commit_hash"), "commit hash must be returned"
        assert "Commit created" in result.output

    @pytest.mark.asyncio
    async def test_commit_clean_tree_is_success(self, git_repo: Path):
        from harness_core.tools.git import GitCommitTool
        (git_repo / "file.txt").write_text("hello", encoding="utf-8")
        _git(git_repo, "add", "-A")
        _git(git_repo, "commit", "-m", "init")
        tool = GitCommitTool()
        result = await tool.execute({"message": "Nothing to do", "cwd": str(git_repo)})
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata.get("nothing_to_commit") is True
        assert "clean" in result.output.lower()

    @pytest.mark.asyncio
    async def test_push_no_remote_diagnosed(self, git_repo: Path):
        from harness_core.tools.git import GitPushTool
        (git_repo / "file.txt").write_text("hello", encoding="utf-8")
        _git(git_repo, "add", "-A")
        _git(git_repo, "commit", "-m", "init")
        tool = GitPushTool()
        result = await tool.execute({"cwd": str(git_repo)})
        assert result.status == ToolResultStatus.ERROR
        assert "No git remote configured" in result.error

    @pytest.mark.asyncio
    async def test_status_outside_repo_diagnosed(self, tmp_path: Path):
        from harness_core.tools.git import GitStatusTool
        tool = GitStatusTool()
        result = await tool.execute({"cwd": str(tmp_path)})
        assert result.status == ToolResultStatus.ERROR
        assert "git repository" in result.error.lower()

    @pytest.mark.asyncio
    async def test_loop_accounts_commit_and_push(self, git_repo: Path):
        """git_commit success records the hash; git_push records remote/branch."""
        from harness_core.tools.git import GitAddTool, GitCommitTool, GitPushTool

        (git_repo / "file.txt").write_text("hello", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(["Commit changes"]),
            tool_response("git_commit", {"message": "Update project", "cwd": str(git_repo)}, "c1"),
            text("Done."),
        ])
        loop = AgentLoop(
            provider=provider,
            tools=[GitAddTool(), GitCommitTool(), GitPushTool()],
            workspace_root=git_repo,
            config=AgentConfig(max_iterations=6, verify_on_complete=False),
            event_bus=EventBus(),
        )
        task = await loop.run("Commit these changes")
        assert task.git_commit, "commit hash must be accounted on the task"
        commit_item = next(i for i in task.task_plan.items if "commit" in i.description.lower())
        assert commit_item.status == TodoStatus.COMPLETED


# ── 13. Workspace-aware cwd ──────────────────────────────────────────────


class TestWorkspaceCwd:
    @pytest.mark.asyncio
    async def test_commands_run_in_workspace(self, tmp_path: Path):
        probe = tmp_path / "cwd_probe.py"
        probe.write_text("import os\nprint(os.getcwd())\n", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(["Inspect project files"]),
            tool_response("run_command", {"command": "python cwd_probe.py"}, "c1"),
            text("Done."),
        ])
        loop = make_loop(provider, tmp_path)
        task = await loop.run("Show the working directory")
        call = task.tool_calls[0]
        assert call.result is not None and call.result.status == ToolResultStatus.SUCCESS
        assert str(tmp_path.resolve()).lower() in call.result.output.lower()


# ── 14. Path confinement ─────────────────────────────────────────────────


class TestPathConfinement:
    @pytest.mark.asyncio
    async def test_relative_path_resolves_inside_workspace(self, tmp_path: Path):
        provider = ScriptedProvider([
            plan_response(["Implement changes"]),
            tool_response("write_file", {"path": "new.txt", "content": "hello"}, "c1"),
            text("Done."),
        ])
        loop = make_loop(provider, tmp_path)
        await loop.run("Create a file")
        assert (tmp_path / "new.txt").exists()

    @pytest.mark.asyncio
    async def test_escaping_path_denied(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        call = ToolCall(tool_name="read_file", arguments={"path": "../outside.txt"})
        result = await loop._execute_tool(call)
        assert result.status == ToolResultStatus.PERMISSION_DENIED
        assert "outside the workspace" in result.error

    @pytest.mark.asyncio
    async def test_absolute_outside_path_denied(self, tmp_path: Path):
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "elsewhere"
        loop = make_loop(ScriptedProvider([]), ws)
        call = ToolCall(tool_name="write_file", arguments={"path": str(outside / "x.txt"), "content": "x"})
        result = await loop._execute_tool(call)
        assert result.status == ToolResultStatus.PERMISSION_DENIED


# ── 15. Timeouts ─────────────────────────────────────────────────────────


class SlowTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="slow_tool", description="sleeps", parameters={"type": "object", "properties": {}},
                          timeout_seconds=30.0)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(status=ToolResultStatus.SUCCESS, output="done")


class TestTimeouts:
    @pytest.mark.asyncio
    async def test_tool_timeout_returns_timeout_result(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path, tools=[SlowTool()])
        call = ToolCall(tool_name="slow_tool", arguments={"timeout": 1})
        start = time.time()
        result = await loop._execute_tool(call)
        elapsed = time.time() - start
        assert result.status == ToolResultStatus.TIMEOUT
        assert "timed out" in result.error.lower()
        assert elapsed < 4, "timeout must fire near the limit, not hang"


# ── 16. Cancellation reporting ───────────────────────────────────────────


class TestCancellationReport:
    def test_cancellation_reports_partial_state(self):
        shell = InteractiveShell(plain=True)
        shell.console = Console(file=io.StringIO(), no_color=True)
        task = Task(goal="Enhance calculator")
        task.task_plan.add("Inspect project")
        task.task_plan.add("Run tests")
        task.task_plan.complete_id(task.task_plan.items[0].id, {"tool": "read_file"})
        task.status = TaskStatus.EXECUTING
        shell._agent_loop = SimpleNamespace(_active_task=task)

        shell._render_cancellation()

        out = shell.console.file.getvalue()
        assert "Task cancelled" in out
        assert "1/2" in out  # truthful partial TODO accounting
        assert task.status == TaskStatus.CANCELLED


# ── 17-18. Truthful completion & accounting ─────────────────────────────


class TestFinalAccounting:
    def test_report_contains_real_accounting(self):
        task = Task(goal="Enhance calculator", status=TaskStatus.COMPLETED)
        task.task_plan.add("Inspect project")
        task.task_plan.add("Run tests")
        task.task_plan.complete_id(task.task_plan.items[0].id)
        task.task_plan.complete_id(task.task_plan.items[1].id)
        task.iterations = 9
        task.tests_run = 32
        task.tests_passed = 32
        task.verification_passed = True
        task.git_commit = "abc1234"
        task.models_used = ["minimax/minimax-m3:free"]
        task.model_fallbacks = 1
        tc = ToolCall(tool_name="edit_file", arguments={"path": "script.js"})
        tc.result = ToolResult(status=ToolResultStatus.SUCCESS, output="ok")
        task.tool_calls.append(tc)

        report = build_execution_report(task, 62.0, files_from_task(task))

        assert "Task completed" in report
        assert "TODO progress: 2/2" in report
        assert "32/32 passed" in report
        assert "Verification passed" in report
        assert "abc1234" in report
        assert "script.js" in report
        assert "9 iterations" in report

    def test_report_failed_task_shows_root_cause_not_prose(self):
        task = Task(goal="Fix tests", status=TaskStatus.FAILED)
        task.error = "AssertionError: expected 5 got 0"
        tc = ToolCall(tool_name="run_command", arguments={"command": "node test.js"})
        tc.result = ToolResult(status=ToolResultStatus.ERROR, output="",
                               error="AssertionError: expected 5 got 0", exit_code=1)
        task.tool_calls.append(tc)

        report = build_execution_report(task, 10.0, [])
        assert "Task failed" in report
        assert "Root cause" in report
        assert "AssertionError" in report

    @pytest.mark.asyncio
    async def test_completion_blocked_without_evidence(self, tmp_path: Path):
        """A failing command with no recovery must not yield a completed task."""
        provider = ScriptedProvider([
            plan_response(["Run tests"]),
            tool_response("run_command", {"command": "node test.js"}, "c1"),
            text("I completed the task. Everything looks good."),
        ])
        loop = make_loop(provider, tmp_path)
        task = await loop.run("Fix the tests")
        assert task.status == TaskStatus.FAILED


class TestStagnationRecovery:
    @pytest.mark.asyncio
    async def test_green_evidence_completes_despite_repetition(self, tmp_path: Path):
        """Work done + tests green ⇒ complete, even if the model keeps repeating."""
        (tmp_path / "test_ok.py").write_text("print('1/1 passed')\n", encoding="utf-8")
        app = tmp_path / "app.py"
        app.write_text("x = 1", encoding="utf-8")
        same_read = tool_response("read_file", {"path": str(app)}, "rX")
        provider = ScriptedProvider([
            plan_response(["Implement changes", "Run tests"]),
            tool_response("write_file", {"path": str(app), "content": "x = 2"}, "c1"),
            tool_response("run_command", {"command": "python test_ok.py"}, "c2"),
            same_read, same_read, same_read, same_read,
        ])
        loop = make_loop(provider, tmp_path, max_iterations=12)

        task = await loop.run("Enhance this project")

        assert task.status == TaskStatus.COMPLETED
        assert task.result and "no-progress" in task.result
        statuses = {i.description: i.status for i in task.task_plan.items}
        assert statuses["Implement changes"] == TodoStatus.COMPLETED
        assert statuses["Run tests"] == TodoStatus.COMPLETED
