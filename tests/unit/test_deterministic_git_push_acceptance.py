"""Acceptance scenario test: deterministic git push with failing/unavailable model provider."""

import asyncio
from pathlib import Path
import subprocess
import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, Task, ToolResultStatus
from harness_core.agents.domain import TaskGraph, SubTask, TaskStatus, AgentRole
from harness_core.agents.locks import WorkspaceLockManager
from harness_core.agents.message_bus import AgentMessageBus
from harness_core.agents.registry import AgentRegistry
from harness_core.agents.scheduler import Scheduler
from harness_core.observability.events import EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.runtime.runtime import EngineeringRuntime
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
from harness_core.tools.shell import RunCommandTool


class FailingProvider(ModelProvider):
    """Provider that raises an exception if called, guaranteeing 0 LLM calls."""

    @property
    def name(self) -> str:
        return "failing_rate_limited"

    async def health_check(self) -> bool:
        return False

    async def list_models(self) -> list:
        return []

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        raise RuntimeError("Provider rate limited: HTTP 429 Too Many Requests")

    async def stream(self, request: CompletionRequest):
        raise RuntimeError("Provider rate limited: HTTP 429 Too Many Requests")


def _build_tools(repo: Path) -> list:
    return [
        GitStatusTool(working_directory=str(repo)),
        GitDiffTool(working_directory=str(repo)),
        GitLogTool(working_directory=str(repo)),
        GitIdentityTool(working_directory=str(repo)),
        GitRemoteTool(working_directory=str(repo)),
        GitAddTool(working_directory=str(repo)),
        GitCommitTool(working_directory=str(repo)),
        GitPushTool(working_directory=str(repo)),
        RunCommandTool(working_directory=str(repo)),
    ]


@pytest.mark.asyncio
async def test_deterministic_git_push_acceptance_scenario(tmp_path: Path):
    """
    Mandatory acceptance test:
    Given:
      - Separate project directory with uncommitted files and no initial commits
      - Valid bare git remote
      - Failing provider (HTTP 429 rate limit if any LLM call made)
    When:
      - Goal is 'push this project to this github repo <remote>'
    Then:
      - Completes without model calls
      - Initial commit is created
      - Remote receives the commit and file
      - Real tool calls are executed and recorded
    """
    repo = tmp_path / "project_app"
    repo.mkdir()
    remote = tmp_path / "remote.git"

    # Setup bare remote
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    # Setup local repo with files (uncommitted)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kunal Sanga"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kunal@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(remote.resolve())], check=True, capture_output=True)

    # Create project files
    (repo / "index.html").write_text("<!DOCTYPE html><html><body><h1>Calculator</h1></body></html>", encoding="utf-8")
    (repo / "script.js").write_text("console.log('calc ready');", encoding="utf-8")
    (repo / "README.md").write_text("# Calculator App", encoding="utf-8")

    event_bus = EventBus()
    provider = FailingProvider()
    tools = _build_tools(repo)

    runtime = EngineeringRuntime(
        workspace_path=str(repo),
        provider=provider,
        event_bus=event_bus,
        tools=tools,
    )

    goal = f"push this project to this github repo {str(remote.resolve())}"

    # Execute through the canonical runtime interactive entrypoint
    outcome = await runtime.execute_interactive(goal)

    # Must succeed despite failing model provider
    assert outcome.status.value == "success", f"Runtime failed: {outcome.state.blockers}"

    # Verify remote received the push
    remote_head = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert remote_head, "Remote HEAD should not be empty"

    # Verify remote has the index.html file
    ls_tree = subprocess.run(
        ["git", "--git-dir", str(remote), "ls-tree", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "index.html" in ls_tree
    assert "script.js" in ls_tree
    assert "README.md" in ls_tree


@pytest.mark.asyncio
async def test_deterministic_git_push_auto_adds_remote_from_goal(tmp_path: Path):
    """Verify that if origin is not configured, the URL in the goal is auto-configured as origin."""
    repo = tmp_path / "project_app2"
    repo.mkdir()
    remote = tmp_path / "remote2.git"

    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kunal Sanga"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kunal@example.com"], check=True, capture_output=True)
    # Notice: NO remote added to repo!

    (repo / "index.html").write_text("<h1>App</h1>", encoding="utf-8")

    event_bus = EventBus()
    provider = FailingProvider()
    tools = _build_tools(repo)

    runtime = EngineeringRuntime(
        workspace_path=str(repo),
        provider=provider,
        event_bus=event_bus,
        tools=tools,
    )

    goal = f"push this project to this github repo {str(remote.resolve())}"
    outcome = await runtime.execute_interactive(goal)

    assert outcome.status.value == "success", f"Runtime failed: {outcome.state.blockers}"
    remote_head = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert remote_head


@pytest.mark.asyncio
async def test_deterministic_git_push_when_already_committed(tmp_path: Path):
    """Verify that when working tree is clean because commits are already created, push proceeds."""
    repo = tmp_path / "project_app3"
    repo.mkdir()
    remote = tmp_path / "remote3.git"

    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kunal Sanga"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kunal@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(remote.resolve())], check=True, capture_output=True)

    (repo / "index.html").write_text("<h1>App</h1>", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "manual commit"], check=True, capture_output=True)
    # Tree is now CLEAN!

    event_bus = EventBus()
    provider = FailingProvider()
    tools = _build_tools(repo)

    runtime = EngineeringRuntime(
        workspace_path=str(repo),
        provider=provider,
        event_bus=event_bus,
        tools=tools,
    )

    goal = "push this project to origin"
    outcome = await runtime.execute_interactive(goal)

    assert outcome.status.value == "success", f"Runtime failed: {outcome.state.blockers}"
    remote_head = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert remote_head

