"""Phase 10 — professional CLI E2E (Parts 1, 17, 18, 19, 25).

Invokes the real ``harness run --mode unified`` command and proves the CLI
renders evidence-based summaries — a success panel only when the runtime
verifies, an honest failure panel otherwise, a cancellation path on Ctrl+C —
and that machine-readable --json output is preserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness_core.agents.domain import AgentResult, AgentStatus
from harness_core.agents.worker import WorkerAgent
from harness_core.providers.base import CompletionResponse

PLAN_JSON = """{
  "summary": "Create an artifact",
  "tasks": [
    {
      "task_id": "task_1",
      "title": "Build",
      "objective": "Create the artifact",
      "role": "coder",
      "dependencies": [],
      "success_criteria": ["created"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [],
      "traceable_to": []
    }
  ]
}"""


class _FakeProvider:
    name = "fake-openrouter"

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def generate(self, request):
        return CompletionResponse(content=PLAN_JSON, model="fake")

    async def list_models(self):
        return []


class _FakeOllama:
    name = "ollama"

    async def health_check(self) -> bool:
        return False

    async def close(self) -> None:
        return None


def _write_project_config(workspace: Path) -> None:
    (workspace / ".harness").mkdir(parents=True, exist_ok=True)
    (workspace / ".harness" / "config.yaml").write_text(
        "routing:\n  strategy: auto\n\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )


def _patch_cli_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    import harness_core.providers.ollama as ollama_module
    import harness_core.providers.openrouter as openrouter_module

    monkeypatch.setattr(openrouter_module, "OpenRouterProvider", _FakeProvider)
    monkeypatch.setattr(ollama_module, "OllamaProvider", _FakeOllama)


def _make_worker_success():
    """Success path: worker emits tool/test events and returns completed."""

    async def fake_run(self):
        await self.emit("tool.call", {"tool": "write_file", "args": {"path": "src/app.py"}})
        await self.emit("tool.result", {"tool": "write_file", "status": "success"})
        await self.emit(
            "test.completed", {"passed": 2, "total": 2, "success": True}
        )
        return AgentResult(
            agent_id=self.contract.agent_id,
            role=self.contract.role,
            status=AgentStatus.COMPLETED,
            summary="artifact created",
            files_changed=["src/app.py"],
            tests_passed=2,
            tests_total=2,
        )

    return fake_run


def test_cli_unified_success_renders_evidence_based_summary(tmp_path, monkeypatch):
    """Success summary only claims what the runtime actually observed."""
    _write_project_config(tmp_path)
    _patch_cli_boundaries(monkeypatch)
    monkeypatch.setattr(WorkerAgent, "run", _make_worker_success())

    from harness_core.cli.main import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["run", "--mode", "unified", "--headless", "Create artifact"]
    )
    assert result.exit_code == 0, result.output

    output = result.output
    # Part 18: concise professional completion summary.
    assert "Harness Complete" in output
    assert "Status: VERIFIED" in output
    assert "Changes: 1 modified, 0 created" in output
    assert "Tests: ✓ 2 passed" in output
    assert "Verification: passed" in output


def test_cli_unified_failure_renders_honest_summary(tmp_path, monkeypatch):
    """A failed run must never claim success (Part 19)."""
    _write_project_config(tmp_path)
    _patch_cli_boundaries(monkeypatch)

    async def fake_run(self):
        return AgentResult(
            agent_id=self.contract.agent_id,
            role=self.contract.role,
            status=AgentStatus.FAILED,
            summary="boom",
            errors=["boom"],
        )

    monkeypatch.setattr(WorkerAgent, "run", fake_run)

    from harness_core.cli.main import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["run", "--mode", "unified", "--headless", "Create artifact"]
    )
    assert result.exit_code == 1, result.output

    output = result.output
    assert "Harness Stopped" in output
    assert "No false success was reported." in output
    assert "Harness Complete" not in output


def test_cli_unified_json_preserved(tmp_path, monkeypatch):
    """--json output remains the authoritative ProjectState snapshot."""
    _write_project_config(tmp_path)
    _patch_cli_boundaries(monkeypatch)
    monkeypatch.setattr(WorkerAgent, "run", _make_worker_success())

    from harness_core.cli.main import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["run", "--mode", "unified", "--headless", "--json", "Create artifact"]
    )
    assert result.exit_code == 0, result.output

    import json

    snapshot = json.loads(result.output)
    assert snapshot["status"] == "success"
    assert snapshot["terminal"] is True


def test_cli_unified_ctrl_c_renders_cancellation(tmp_path, monkeypatch):
    """Ctrl+C cancels cleanly: summary rendered, exit code 130."""
    _write_project_config(tmp_path)
    _patch_cli_boundaries(monkeypatch)

    from harness_core.cli import main as cli_main
    from harness_core.runtime import runtime as runtime_module

    # Simulate the user pressing Ctrl+C mid-execution: asyncio.run unwinds
    # with KeyboardInterrupt and the CLI must render the cancellation panel.
    async def _raise_keyboard_interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_module.EngineeringRuntime, "run", _raise_keyboard_interrupt)

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli_main.app,
        ["run", "--mode", "unified", "--headless", "Create artifact"],
    )
    assert result.exit_code == 130, result.output
    output = result.output
    assert "Cancellation requested" in output
    assert "Runtime cancelled" in output
    assert "Locks released" in output
    assert "Session preserved" in output
