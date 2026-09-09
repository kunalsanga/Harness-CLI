"""Phase 9 — the REAL CLI executes through EngineeringRuntime (--mode unified).

Invokes the actual ``harness run --mode unified`` command via Typer's
CliRunner inside a real project directory and proves the command:

    CLI -> EngineeringRuntime -> Planner -> Scheduler -> WorkerAgent -> MemoryManager

reaches a terminal SUCCESS status, persists project-scoped memory, and exits 0.

The model/provider and the AgentLoop inside WorkerAgents are stubbed at the
process boundary (no network); every other layer is the real production path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from harness_core.memory.domain import MemoryType
from harness_core.memory.store import LocalMemoryStore

# ── Process-boundary stubs ─────────────────────────────────────────────────


class _FakeProvider:
    """Stand-in for OpenRouterProvider: healthy, no network."""

    name = "fake-openrouter"

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def generate(self, request):
        from harness_core.providers.base import CompletionResponse

        return CompletionResponse(
            content=(
                '{"summary": "single task plan", "tasks": ['
                '{"task_id": "task_1", "title": "Build", '
                '"objective": "Create artifact", "role": "coder", '
                '"dependencies": [], "success_criteria": ["created"], '
                '"workspace_scope": "project", "priority": 5, '
                '"model_policy": "auto", "expected_artifacts": [], '
                '"resources": []}]}'
            ),
            model="fake",
        )

    async def list_models(self):
        return []


class _FakeOllama:
    """Stand-in for OllamaProvider: not running."""

    name = "ollama"

    async def health_check(self) -> bool:
        return False

    async def close(self) -> None:
        return None


class _StubAgentLoop:
    """Stand-in for the AgentLoop executed inside WorkerAgents."""

    def __init__(self, *args, **kwargs) -> None:
        return None

    async def run(self, goal: str):
        return SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            result="Done",
            tool_calls=[],
            iterations=1,
            error=None,
            models_used=["fake"],
            total_tokens=10,
        )


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_project_config(workspace: Path) -> None:
    (workspace / ".harness").mkdir(parents=True, exist_ok=True)
    (workspace / ".harness" / "config.yaml").write_text(
        "routing:\n  strategy: auto\n\nmemory:\n  enabled: true\n",
        encoding="utf-8",
    )


def _invoke_unified(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    import harness_core.agent.loop as loop_module
    import harness_core.providers.ollama as ollama_module
    import harness_core.providers.openrouter as openrouter_module
    from harness_core.cli.main import app

    monkeypatch.setattr(openrouter_module, "OpenRouterProvider", _FakeProvider)
    monkeypatch.setattr(ollama_module, "OllamaProvider", _FakeOllama)
    monkeypatch.setattr(loop_module, "AgentLoop", _StubAgentLoop)

    monkeypatch.chdir(workspace)
    runner = CliRunner()
    return runner.invoke(
        app, ["run", "--mode", "unified", "--headless", "--json", "Create artifact"]
    )


def _load_entries(entries_path: Path) -> list:
    async def _read() -> list:
        store = LocalMemoryStore(entries_path)
        return await store.all_entries()

    return asyncio.run(_read())


def test_cli_unified_mode_runs_through_engineering_runtime(tmp_path, monkeypatch):
    _write_project_config(tmp_path)
    result = _invoke_unified(tmp_path, monkeypatch)

    assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"

    # The JSON snapshot is the authoritative ProjectState the CLI rendered.
    snapshot = json.loads(result.output)
    assert snapshot["status"] == "success"
    assert snapshot["terminal"] is True
    assert snapshot["stage"] == "succeeded"
    assert snapshot["progress"]["completed"] == 1
    assert snapshot["progress"]["total"] == 1
    assert snapshot["verification"]["status"] == "passed"

    # Persistent, project-scoped memory was written by the real CLI path.
    entries = _load_entries(tmp_path / ".harness" / "memory" / "entries.json")
    assert entries, "unified CLI run should persist memory entries"
    project_id = str(tmp_path.resolve())
    assert all(e.project_id == project_id for e in entries), "project isolation violated"
    assert any(e.type == MemoryType.SUCCESS for e in entries)
