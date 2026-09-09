"""Phase 10.5 — Git zero-tool-call accounting regression tests.

Proves that real git operations executed by deterministic workflows are
recorded as first-class ToolCalls with tool events and iteration/tool
accounting — the "Commit 06a6ae3 / Push origin/main with 0 tool calls"
invalid state must be impossible.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, Task, ToolResultStatus
from harness_core.agent.workflows import WorkflowContext, run_git_push_workflow
from harness_core.cli.runtime_dashboard import RuntimeViewModel
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.tools.base import Tool
from harness_core.tools.git import (
    GitAddTool,
    GitCommitTool,
    GitDiffTool,
    GitIdentityTool,
    GitLogTool,
    GitPushTool,
    GitRemoteTool,
    GitStatusTool,
)


class _MockProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "mock"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(content="Done.", model="mock", provider="mock")

    async def stream(self, request: CompletionRequest):
        yield CompletionResponse(content="", model="mock", provider="mock")

    async def list_models(self) -> list[Any]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def _git(*args: str, cwd: Path) -> str:
    """Run a real git command in a temp repo."""
    out = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "PATH": ""},
    )
    return out.stdout.strip()


@pytest.fixture()
def git_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A real git repo with one committed file plus a bare remote."""
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@harness.local", cwd=repo)
    _git("config", "user.name", "Harness Test", cwd=repo)
    (repo / "README.md").write_text("# hello\n", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "initial", cwd=repo)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git("remote", "add", "origin", str(remote), cwd=repo)
    # Dirty the tree so the push workflow has real work to do.
    (repo / "README.md").write_text("# hello\n\nUpdated for the push test.\n", encoding="utf-8")
    return repo, remote


def _build_tools(repo: Path) -> list[Tool]:
    return [
        GitStatusTool(working_directory=str(repo)),
        GitDiffTool(working_directory=str(repo)),
        GitLogTool(working_directory=str(repo)),
        GitIdentityTool(working_directory=str(repo)),
        GitRemoteTool(working_directory=str(repo)),
        GitAddTool(working_directory=str(repo)),
        GitCommitTool(working_directory=str(repo)),
        GitPushTool(working_directory=str(repo)),
    ]


class TestGitWorkflowAccounting:
    """Real git push through the deterministic workflow must be accounted."""

    async def _run_workflow(
        self, repo: Path, vm: RuntimeViewModel | None = None
    ) -> tuple[AgentLoop, Task, list[Event]]:
        event_bus = EventBus()
        if vm is not None:
            vm.attach(event_bus)
        loop = AgentLoop(
            provider=_MockProvider(),
            tools=_build_tools(repo),
            workspace_root=repo,
            config=AgentConfig(),
            event_bus=event_bus,
        )
        task = Task(
            goal="Push changes to remote",
            git_commit="initial",
        )
        loop._active_task = task  # noqa: SLF001 — unit test drives the private slot directly
        ctx: WorkflowContext = loop._workflow_context()
        plan = task.task_plan
        result = await run_git_push_workflow(
            ctx,
            plan,
            commit_message="Phase 10.5 accounting test",
        )
        events = event_bus.get_history()
        return loop, task, events, result  # type: ignore[return-value]

    def test_real_git_push_records_tool_calls_and_events(self, git_repo: tuple[Path, Path]):
        """A real push must yield ToolCalls, iterations, tool events, and a commit SHA."""
        repo, _remote = git_repo

        async def _main() -> None:
            loop, task, events, result = await self._run_workflow(repo)
            assert result.success, result.failure_reason

            # The zero-tool-call state is impossible: every git op is a ToolCall.
            assert len(task.tool_calls) >= 7, (
                f"expected >=7 recorded tool calls, got {len(task.tool_calls)}: "
                + ", ".join(tc.tool_name for tc in task.tool_calls)
            )
            assert task.iterations >= 7
            assert loop.budget.state.tool_calls >= 7
            assert task.git_commit, "commit SHA must be recorded"
            assert task.git_push, "push target must be recorded"

            # Real git evidence on the commit tool call.
            commit_call = next(tc for tc in task.tool_calls if tc.tool_name == "git_commit")
            assert commit_call.result.status == ToolResultStatus.SUCCESS
            assert commit_call.result.metadata and commit_call.result.metadata.get("commit_hash")
            assert task.git_commit in commit_call.result.metadata["commit_hash"]

            # Events were emitted for every op.
            tool_calls = [e for e in events if e.type == "tool.call"]
            tool_results = [e for e in events if e.type == "tool.result"]
            assert len(tool_calls) >= 7
            assert len(tool_results) >= 7
            result_tools = {e.data.get("tool") for e in tool_results}
            assert {"git_status", "git_remote", "git_add", "git_commit", "git_push"} <= result_tools

        asyncio.run(_main())

    def test_remote_verified_and_push_evidence(self, git_repo: tuple[Path, Path]):
        """The remote must actually receive the commit and verification uses evidence."""
        repo, remote = git_repo

        async def _main() -> None:
            _loop, task, _events, result = await self._run_workflow(repo)
            assert result.success

            # Verify the push actually reached the bare remote.
            head = subprocess.run(
                ["git", "--git-dir", str(remote), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            assert head.startswith(task.git_commit), (
                f"remote HEAD {head} must match the recorded commit {task.git_commit}"
            )

            push_call = next(tc for tc in task.tool_calls if tc.tool_name == "git_push")
            assert push_call.result.status == ToolResultStatus.SUCCESS
            assert push_call.result.metadata and push_call.result.metadata.get("branch") == "main"

        asyncio.run(_main())

    def test_view_model_receives_git_evidence(self, git_repo: tuple[Path, Path]):
        """RuntimeViewModel git state must be derived from real tool events."""
        repo, _remote = git_repo

        async def _main() -> None:
            vm = RuntimeViewModel()
            loop, task, events, _result = await self._run_workflow(repo, vm=vm)
            assert task.git_commit

            snap = vm.snapshot()
            git = snap["git"]

            # The dashboard can now truthfully say: Commit <sha>, pushed to <remote>.
            assert git["commit"] == task.git_commit
            assert git["remote"] == "origin", "remote must be visible in the dashboard"
            assert git["push"] == "origin/main", "push evidence must be visible"
            assert "commit" in git["operations"]
            assert "push" in git["operations"]

        asyncio.run(_main())

    def test_loop_events_carry_command_for_convergence(self, git_repo: tuple[Path, Path]):
        """tool.result events must be attributed with a tool name for the VM."""
        repo, _remote = git_repo

        async def _main() -> None:
            _loop, _task, events, _result = await self._run_workflow(repo)
            # Every tool.result event must name its tool so the view model and
            # convergence governor can attribute real operations.
            results = [e for e in events if e.type == "tool.result"]
            assert len(results) >= 7
            assert all(e.data.get("tool") for e in results)

        asyncio.run(_main())
