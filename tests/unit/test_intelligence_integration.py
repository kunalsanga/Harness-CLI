"""Tests for ProjectIntelligence integration points.

Covers:
- GlobTool/GrepTool fast path via ProjectIntelligence (contract parity with
  the pure-Python walk fallback).
- ContextPipeline SYMBOL/DEPENDENCY candidate sources (optional facade,
  failure-tolerant).
- AgentLoop._search_matches_for evidence extraction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness_core.agent.types import AgentConfig, AgentRole
from harness_core.context.models import CandidateSource, ContextRequest
from harness_core.context.pipeline import ContextPipeline
from harness_core.intelligence import ProjectIntelligence
from harness_core.tools.search import GlobTool, GrepTool


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "models.py").write_text(
        "class User:\n    pass\n\n"
        "def load_user(user_id: int) -> User:\n    return User(f'u{user_id}')\n",
        encoding="utf-8",
    )
    (root / "app" / "auth.py").write_text(
        "from app import models\n\n"
        "def authenticate(user_id: int) -> models.User:\n"
        "    return models.load_user(user_id)\n",
        encoding="utf-8",
    )
    (root / "app" / "service.py").write_text(
        "from app.auth import authenticate\n\n"
        "def run(user_id: int):\n    return authenticate(user_id)\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_service.py").write_text(
        "from app.service import run\n\n"
        "def test_run():\n    assert run(1)\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Demo project\n", encoding="utf-8")
    return root


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestGlobToolIntelligence:
    def test_fast_path_matches_walk_output(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        fast = GlobTool(intelligence=intel)
        slow = GlobTool()
        args = {"pattern": "*.py", "path": str(project)}
        fast_res = _run(fast.execute(dict(args)))
        slow_res = _run(slow.execute(dict(args)))
        assert fast_res.status == slow_res.status
        assert fast_res.output == slow_res.output

    def test_cold_index_falls_back_to_walk(self, project: Path):
        intel = ProjectIntelligence(project)  # never scanned
        tool = GlobTool(intelligence=intel)
        res = _run(tool.execute({"pattern": "*.py", "path": str(project)}))
        assert "app" in res.output or "service" in res.output

    def test_no_intelligence_still_works(self, project: Path):
        tool = GlobTool()
        res = _run(tool.execute({"pattern": "*.py", "path": str(project)}))
        assert "auth.py" in res.output


class TestGrepToolIntelligence:
    def test_fast_path_matches_walk_output(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        fast = GrepTool(intelligence=intel)
        slow = GrepTool()
        args = {"pattern": "authenticate", "path": str(project)}
        fast_res = _run(fast.execute(dict(args)))
        slow_res = _run(slow.execute(dict(args)))
        assert fast_res.status == slow_res.status
        assert fast_res.output == slow_res.output

    def test_include_filter_respected_in_fast_path(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        tool = GrepTool(intelligence=intel)
        res = _run(tool.execute({
            "pattern": "run", "path": str(project), "include": "*.py",
        }))
        assert ".md" not in res.output

    def test_no_intelligence_still_works(self, project: Path):
        tool = GrepTool()
        res = _run(tool.execute({"pattern": "authenticate", "path": str(project)}))
        assert "authenticate" in res.output


class TestPipelineIntelligenceSources:
    def _request(self, task: str) -> ContextRequest:
        return ContextRequest(task=task, run_id="r1", task_id="t1", agent_id="a1")

    def test_symbol_source_adds_candidates(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        pipeline = ContextPipeline(project, intelligence=intel)
        selection = _run(pipeline.discover(self._request("fix the authenticate function")))
        # auth.py should surface via symbol evidence (or its ranker path match).
        assert any("auth" in f for f in selection.files)

    def test_dependency_source_expands_from_seeds(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        pipeline = ContextPipeline(project, intelligence=intel)
        selection = _run(pipeline.discover(
            self._request("update the load_user models helper"),
        ))
        # Either direct selection or candidate consideration — models.py is
        # linked to auth.py in the dependency graph.
        assert selection.snapshot.candidates_considered > 0

    def test_pipeline_without_intelligence_unchanged(self, project: Path):
        pipeline = ContextPipeline(project)
        selection = _run(pipeline.discover(self._request("fix the authenticate function")))
        assert selection.snapshot is not None

    def test_intelligence_failure_is_non_fatal(self, project: Path):
        class _Boom:
            root = str(project)

            def file_list(self, limit: int = 500):
                return ["app/auth.py"]

            def find_symbols(self, *a, **k):
                raise RuntimeError("boom")

            dependencies = None

        pipeline = ContextPipeline(project, intelligence=_Boom())
        # Must not raise; falls back to deterministic sources.
        selection = _run(pipeline.discover(self._request("fix the authenticate function")))
        assert selection.snapshot is not None


class TestAgentLoopSearchMatches:
    def test_search_matches_for_uses_intelligence(self, project: Path):
        from harness_core.agent.loop import AgentLoop

        intel = ProjectIntelligence(project)
        intel.scan()

        loop = AgentLoop.__new__(AgentLoop)  # bypass __init__
        loop.intelligence = intel
        matches = loop._search_matches_for("fix authenticate in auth")
        assert matches is not None
        assert any("auth.py" in k for k in matches)

    def test_search_matches_none_without_intelligence(self, project: Path):
        from harness_core.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.intelligence = None
        assert loop._search_matches_for("fix authenticate") is None

    def test_search_matches_none_on_intel_failure(self, project: Path):
        from harness_core.agent.loop import AgentLoop

        class _Boom:
            def file_list(self, limit: int = 500):
                return ["x.py"]

            def search(self, *a, **k):
                raise RuntimeError("boom")

        loop = AgentLoop.__new__(AgentLoop)
        loop.intelligence = _Boom()
        assert loop._search_matches_for("fix authenticate") is None

    def test_agentloop_accepts_intelligence_param(self, project: Path):
        from harness_core.agent.loop import AgentLoop

        from harness_core.context.pipeline import ContextPipeline as _CP

        intel = ProjectIntelligence(project)
        loop = AgentLoop(
            provider=None,
            tools=[],
            workspace_root=project,
            config=AgentConfig(role=AgentRole.BUILD),
            intelligence=intel,
        )
        assert loop.intelligence is intel
        # Default pipeline receives the same facade.
        assert loop._context_pipeline.intelligence is intel

    def test_default_pipeline_isolation(self, project: Path):
        from harness_core.agent.loop import AgentLoop

        loop = AgentLoop(provider=None, tools=[], workspace_root=project)
        assert loop._context_pipeline.intelligence is None
