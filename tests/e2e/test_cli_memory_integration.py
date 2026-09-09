"""Phase 8 hardening — memory initialised through the REAL CLI construction path.

These tests invoke the actual ``harness run`` command (via Typer's CliRunner)
inside a real project directory and prove that the normal CLI path:

    CLI -> Orchestrator -> Planner -> Scheduler -> WorkerAgent -> MemoryManager

initialises persistent project memory under ``.harness/memory`` by default,
honours the ``memory.enabled: false`` opt-out, and stays inert when the
directory is not an initialised Harness project.

The model/provider and the agent loop are stubbed at the process boundary
(no network); every layer between the CLI command and the memory store is the
real production code path.
"""

from __future__ import annotations

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
        # The only real generate() call in this flow is the Planner's
        # decomposition request. Return a valid single-coder-task plan.
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
    """Stand-in for the AgentLoop executed inside WorkerAgents.

    Returns a completed result immediately so the whole Scheduler flow runs
    deterministically; the loop is not the subject under test here.
    """

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


def _write_project_config(workspace: Path, memory_enabled: bool | None = None) -> None:
    (workspace / ".harness").mkdir(parents=True, exist_ok=True)
    lines = [
        "routing:",
        "  strategy: auto",
        "",
        "budgets:",
        "  max_cost_per_task: 1.0",
        "",
    ]
    if memory_enabled is not None:
        lines.append("memory:")
        lines.append(f"  enabled: {str(memory_enabled).lower()}")
    (workspace / ".harness" / "config.yaml").write_text("\n".join(lines), encoding="utf-8")


def _run_cli(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Run the real ``harness run --mode multi-agent`` command in ``workspace``."""
    import harness_core.agent.loop as loop_module
    import harness_core.providers.ollama as ollama_module
    import harness_core.providers.openrouter as openrouter_module
    from harness_core.cli.main import app

    monkeypatch.setattr(openrouter_module, "OpenRouterProvider", _FakeProvider)
    monkeypatch.setattr(ollama_module, "OllamaProvider", _FakeOllama)
    monkeypatch.setattr(loop_module, "AgentLoop", _StubAgentLoop)

    monkeypatch.chdir(workspace)
    runner = CliRunner()
    return runner.invoke(app, ["run", "--mode", "multi-agent", "--headless", "Create artifact"])


def _memory_store_path(workspace: Path) -> Path:
    return workspace / ".harness" / "memory" / "entries.json"


def _load_entries(entries_path: Path) -> list:
    """Load persisted memory entries via a dedicated event loop."""
    import asyncio


    async def _load() -> list:
        return await LocalMemoryStore(entries_path).all_entries()

    return asyncio.run(_load())


# ── Tests ──────────────────────────────────────────────────────────────────


class TestCliMemoryBootstrap:
    def test_multi_agent_run_initializes_persistent_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Default project: memory is enabled and populated during the run."""
        _write_project_config(tmp_path, memory_enabled=None)  # no memory key -> default on
        result = _run_cli(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output

        entries_path = _memory_store_path(tmp_path)
        assert entries_path.exists(), "CLI run must persist memory under .harness/memory"

        entries = _load_entries(entries_path)
        assert len(entries) >= 1, "Successful worker execution must record a memory"
        completed = [e for e in entries if e.type == MemoryType.SUCCESS]
        assert completed, f"Expected at least one SUCCESS memory, got: {[e.type.value for e in entries]}"
        # Entries are scoped to this project (resolved absolute workspace path)
        expected_project = str(tmp_path.resolve())
        assert all(e.project_id == expected_project for e in entries), (
            f"All entries must be project-scoped to {expected_project}"
        )

    def test_multi_agent_run_opt_out_via_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """memory.enabled: false must keep the run fully functional with no memory."""
        _write_project_config(tmp_path, memory_enabled=False)
        result = _run_cli(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output
        assert not _memory_store_path(tmp_path).exists(), (
            "Opted-out run must not create a memory store"
        )
        assert not (tmp_path / ".harness" / "memory").exists()

    def test_run_in_uninitialized_dir_does_not_touch_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Without .harness/config.yaml the CLI stays legacy: no memory side effects."""
        result = _run_cli(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output
        assert not (tmp_path / ".harness").exists(), (
            "Uninitialised project dirs must not gain .harness side effects"
        )
