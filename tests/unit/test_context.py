"""Unit tests for the context engine."""

import tempfile
from pathlib import Path

import pytest

from harness_core.context.engine import ContextBudget, ContextEngine


class TestContextEngine:
    """Tests for ContextEngine."""

    def test_estimate_tokens(self):
        engine = ContextEngine()
        # Roughly 1 token per 4 chars
        assert engine.estimate_tokens("hello") == 1  # 5 chars / 4 = 1

    @pytest.mark.asyncio
    async def test_discover_project(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("print('hi')")
        (tmp_path / ".git").mkdir()

        engine = ContextEngine(tmp_path)
        info = await engine.discover_project()

        assert info["has_git"] is True
        assert "python" in info["languages"]
        assert "app.py" in info["files"]

    @pytest.mark.asyncio
    async def test_assemble_context(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x = 1")

        engine = ContextEngine(tmp_path)
        project_info = await engine.discover_project()
        pieces = await engine.assemble_context("Fix the bug", project_info)

        assert len(pieces) > 0
        # Task should be highest priority
        task_piece = [p for p in pieces if p.source == "task"]
        assert len(task_piece) == 1

    def test_budget_available_tokens(self):
        budget = ContextBudget(
            model_context_limit=100000,
            system_prompt_tokens=1000,
            tool_schema_tokens=500,
            reserved_output_tokens=4096,
        )
        assert budget.available_tokens == 94404
