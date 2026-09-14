"""Regression tests for the Harness 2.0 context intelligence pipeline (Parts 2-6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core.context.budgets import (
    BudgetClass,
    ContextBudgetConfig,
    ContextBudgetManager,
)
from harness_core.context.models import (
    ContextCandidate,
    ContextEvidence,
    ContextRequest,
    CandidateSource,
    Freshness,
)
from harness_core.context.pipeline import ContextPipeline
from harness_core.context.relevance import AIRelevanceRanker, AIRelevanceConfig
from harness_core.context.reuse import ContextReuseManager
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider


# ── Budget calculation & thresholds ──────────────────────────────────────


class TestContextBudgets:
    def test_budget_usage_and_remaining(self):
        mgr = ContextBudgetManager(ContextBudgetConfig(max_total_tokens=1000))
        mgr.record(BudgetClass.TOOL_OUTPUT, 400)
        mgr.record(BudgetClass.FILE_CONTENT, 200)
        assert mgr.current_usage == 600
        assert mgr.remaining == 400
        assert not mgr.compaction_recommended

    def test_compaction_threshold(self):
        mgr = ContextBudgetManager(ContextBudgetConfig(max_total_tokens=1000))
        mgr.record(BudgetClass.TOOL_OUTPUT, 810)  # 81% > 80%
        assert mgr.compaction_recommended
        assert not mgr.emergency_truncation_required

    def test_emergency_threshold(self):
        mgr = ContextBudgetManager(ContextBudgetConfig(max_total_tokens=1000))
        mgr.record(BudgetClass.TOOL_OUTPUT, 960)  # 96% > 95%
        assert mgr.emergency_truncation_required

    def test_class_allocation(self):
        cfg = ContextBudgetConfig(max_total_tokens=1000)
        assert cfg.limit_for(BudgetClass.TOOL_OUTPUT) == 350

    def test_invalid_threshold_ordering_rejected(self):
        with pytest.raises(ValueError):
            ContextBudgetConfig(compaction_threshold=0.99, emergency_threshold=0.9)

    def test_latest_user_request_class_protected_in_allocation(self):
        cfg = ContextBudgetConfig(max_total_tokens=1000)
        # user_critical has a dedicated allocation and is never the one
        # compacted first (pipeline compacts TOOL_OUTPUT).
        assert cfg.limit_for(BudgetClass.USER_CRITICAL) > 0


# ── Candidate models / evidence ──────────────────────────────────────────


class TestContextModels:
    def test_evidence_aggregation_prefers_strongest(self):
        c = ContextCandidate(path="src/app.py")
        c.add_evidence(ContextEvidence(source=CandidateSource.PATH_MATCH, score=0.3))
        c.add_evidence(ContextEvidence(source=CandidateSource.GREP, score=0.7))
        assert c.score >= 0.7  # strongest dominates (plus corroboration bonus)

    def test_request_bounds_are_bounded(self):
        req = ContextRequest(task="fix the bug")
        assert req.max_candidates <= 64
        assert req.max_files <= 16

    def test_freshness_values(self):
        assert Freshness.CACHED.value == "cached"
        assert Freshness.SUMMARIZED.value == "summarized"


# ── Deterministic pipeline discovery ─────────────────────────────────────


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    pass\n")
    (tmp_path / "README.md").write_text("# Project\n")
    return tmp_path


class TestContextPipeline:
    @pytest.mark.asyncio
    async def test_first_read_is_fresh(self, ws: Path):
        reuse = ContextReuseManager()
        pipe = ContextPipeline(ws, reuse=reuse)
        req = ContextRequest(task="calculator add function", max_files=3)
        sel = await pipe.discover(req, project_files=["src/calculator.py", "src/utils.py"])
        assert "src/calculator.py" in sel.files
        assert sel.fresh_contents["src/calculator.py"]
        cand = next(c for c in sel.snapshot.selected if c.path == "src/calculator.py")
        assert cand.freshness is Freshness.FRESH
        # Snapshot recorded for reuse.
        assert reuse.has_snapshot("src/calculator.py")

    @pytest.mark.asyncio
    async def test_repeat_read_is_cached_not_reread(self, ws: Path):
        reuse = ContextReuseManager()
        pipe = ContextPipeline(ws, reuse=reuse)
        req = ContextRequest(task="calculator", max_files=3)
        first = await pipe.discover(req, project_files=["src/calculator.py"])
        assert "src/calculator.py" in first.fresh_contents
        # Second discovery: file unchanged → CACHED, no fresh content loaded.
        second = await pipe.discover(req, project_files=["src/calculator.py"])
        cand = next(c for c in second.snapshot.selected if c.path == "src/calculator.py")
        assert cand.freshness is Freshness.CACHED
        assert "src/calculator.py" not in second.fresh_contents

    @pytest.mark.asyncio
    async def test_changed_file_is_invalidated_and_reloaded(self, ws: Path):
        reuse = ContextReuseManager()
        pipe = ContextPipeline(ws, reuse=reuse)
        req = ContextRequest(task="calculator", max_files=3)
        await pipe.discover(req, project_files=["src/calculator.py"])
        (ws / "src" / "calculator.py").write_text("def add(a, b):\n    return a * b  # changed\n")
        sel = await pipe.discover(req, project_files=["src/calculator.py"])
        cand = next(c for c in sel.snapshot.selected if c.path == "src/calculator.py")
        assert cand.freshness is Freshness.INVALIDATED
        assert "changed" in sel.fresh_contents["src/calculator.py"]

    @pytest.mark.asyncio
    async def test_invalid_path_rejected(self, ws: Path):
        pipe = ContextPipeline(ws)
        req = ContextRequest(task="x", max_files=2)
        # Path traversal outside the workspace must be rejected by validation.
        sel = await pipe.discover(req, project_files=["../outside.py"])
        assert all(not c.path.startswith("..") for c in sel.snapshot.selected)
        assert all(c.exists for c in sel.snapshot.selected)

    @pytest.mark.asyncio
    async def test_bounded_output(self, ws: Path):
        for i in range(40):
            (ws / "src" / f"calc_{i}.py").write_text(f"# calc {i}\nX = {i}\n")
        pipe = ContextPipeline(ws)
        files = [f"src/calc_{i}.py" for i in range(40)]
        req = ContextRequest(task="calc", max_candidates=10, max_files=3)
        sel = await pipe.discover(req, project_files=files)
        assert len(sel.snapshot.selected) <= 3
        assert sel.snapshot.candidates_considered <= 10

    @pytest.mark.asyncio
    async def test_stale_cache_invalidation_on_write(self, ws: Path):
        reuse = ContextReuseManager()
        pipe = ContextPipeline(ws, reuse=reuse)
        await pipe.discover(ContextRequest(task="utils"), project_files=["src/utils.py"])
        # Simulate an external write: invalidate like the edit path does.
        reuse.invalidate("src/utils.py")
        assert not reuse.has_snapshot("src/utils.py")
        sel = await pipe.discover(ContextRequest(task="utils"), project_files=["src/utils.py"])
        cand = next(c for c in sel.snapshot.selected if c.path == "src/utils.py")
        assert cand.freshness is Freshness.FRESH  # re-read after invalidation


# ── AI relevance stage ───────────────────────────────────────────────────


class _FakeProvider(ModelProvider):
    """Returns scripted content."""

    def __init__(self, content: str = "", fail: bool = False):
        self._content = content
        self._fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self._fail:
            raise RuntimeError("provider down")
        return CompletionResponse(content=self._content, model="fake", provider="fake")

    async def stream(self, request: CompletionRequest):
        yield CompletionResponse(content=self._content, model="fake", provider="fake")

    async def list_models(self):
        return []

    async def health_check(self) -> bool:
        return not self._fail


def _cands(*paths: str) -> list[ContextCandidate]:
    out = []
    for p in paths:
        c = ContextCandidate(path=p)
        c.add_evidence(ContextEvidence(source=CandidateSource.PATH_MATCH, score=0.4))
        out.append(c)
    return out


class TestAIRelevance:
    @pytest.mark.asyncio
    async def test_ai_ranking_reorders(self):
        provider = _FakeProvider('["src/b.py", "src/a.py"]')
        ranker = AIRelevanceRanker(model=provider)
        result = await ranker.rank("task", _cands("src/a.py", "src/b.py", "src/c.py", "src/d.py"))
        assert result.used
        assert result.ranked[0].path == "src/b.py"
        # Evidence preserved and AI source recorded.
        assert any(
            e.source is CandidateSource.AI_RANKING for e in result.ranked[0].evidence
        )

    @pytest.mark.asyncio
    async def test_model_failure_falls_back_deterministically(self):
        provider = _FakeProvider(fail=True)
        ranker = AIRelevanceRanker(model=provider)
        base = _cands("src/a.py", "src/b.py", "src/c.py", "src/d.py")
        result = await ranker.rank("task", base)
        assert result.failed
        assert not result.used
        assert [c.path for c in result.ranked] == ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]

    @pytest.mark.asyncio
    async def test_invalid_and_foreign_paths_rejected(self):
        # Model tries to inject a path that was never a candidate + traversal.
        provider = _FakeProvider('["src/b.py", "../../etc/passwd", "src/unknown.py", "src/b.py"]')
        ranker = AIRelevanceRanker(model=provider)
        result = await ranker.rank(
            "task", _cands("src/a.py", "src/b.py", "src/c.py", "src/d.py")
        )
        assert result.used
        paths = [c.path for c in result.ranked]
        assert "src/unknown.py" not in paths
        assert not any(".." in p for p in paths)
        assert paths.count("src/b.py") == 1  # deduped

    @pytest.mark.asyncio
    async def test_bounded_output(self):
        provider = _FakeProvider('["' + '","'.join(f"src/f{i}.py" for i in range(50)) + '"]')
        cfg = AIRelevanceConfig(max_returned_paths=3)
        ranker = AIRelevanceRanker(model=provider, config=cfg)
        base = _cands(*[f"src/f{i}.py" for i in range(10)])
        result = await ranker.rank("task", base)
        assert len([c for c in result.ranked if any(
            e.source is CandidateSource.AI_RANKING for e in c.evidence
        )]) <= 3

    @pytest.mark.asyncio
    async def test_no_model_skips_gracefully(self):
        ranker = AIRelevanceRanker(model=None)
        base = _cands("src/a.py", "src/b.py", "src/c.py", "src/d.py")
        result = await ranker.rank("task", base)
        assert result.skipped and not result.failed
        assert [c.path for c in result.ranked] == ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]

    @pytest.mark.asyncio
    async def test_unparsable_response_falls_back(self):
        provider = _FakeProvider("I cannot answer that in JSON, sorry!")
        ranker = AIRelevanceRanker(model=provider)
        base = _cands("src/a.py", "src/b.py", "src/c.py", "src/d.py")
        result = await ranker.rank("task", base)
        assert result.failed
        assert [c.path for c in result.ranked] == ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]
