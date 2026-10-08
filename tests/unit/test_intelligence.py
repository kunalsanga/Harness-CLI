"""Unit tests for harness_core.intelligence.ProjectIntelligence.

Covers the canonical facade over SymbolIndex/DependencyGraph/RelevanceRanker/
SearchCache: scan + incremental refresh, search layers, symbol/dependency
queries, context candidates, and invalidation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core.intelligence import ProjectIntelligence


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A tiny deterministic workspace: app package + one test + a doc."""
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "\n"
        "def load_user(user_id: int) -> User:\n"
        "    return User(f'u{user_id}')\n",
        encoding="utf-8",
    )
    (root / "app" / "auth.py").write_text(
        "from app import models\n"
        "\n"
        "def authenticate(user_id: int) -> models.User:\n"
        "    return models.load_user(user_id)\n",
        encoding="utf-8",
    )
    (root / "app" / "service.py").write_text(
        "from app.auth import authenticate\n"
        "\n"
        "def run(user_id: int):\n"
        "    return authenticate(user_id)\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_service.py").write_text(
        "from app.service import run\n"
        "\n"
        "def test_run():\n"
        "    assert run(1)\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Demo project\n", encoding="utf-8")
    return root


class TestScan:
    def test_full_scan_stats(self, project: Path):
        intel = ProjectIntelligence(project)
        stats = intel.scan()
        assert stats["files"] == 6
        assert stats["source_files"] == 5
        assert stats["symbols"]["symbols"] >= 6
        assert stats["dependencies"]["edges"] >= 3
        assert stats["last_scan_ms"] >= 0

    def test_scan_is_incremental_when_warm(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        result = intel.refresh()
        assert result.get("incremental") is True
        assert result.get("changed") == 0
        assert result.get("deleted") == 0

    def test_scan_detects_new_file(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        (project / "app" / "extra.py").write_text("def extra() -> int:\n    return 1\n", encoding="utf-8")
        result = intel.refresh()
        assert result.get("changed") == 1
        assert len(intel.find_symbols("extra")) == 1

    def test_refresh_reindexes_changed_file_without_dupes(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        before = intel.symbols.stats["symbols"]
        (project / "app" / "models.py").write_text(
            "class User:\n    pass\n\nclass Admin(User):\n    pass\n",
            encoding="utf-8",
        )
        result = intel.refresh()
        assert result.get("changed") == 1
        after = intel.symbols.stats["symbols"]
        # symbols replaced, not accumulated
        assert after <= before + 2

    def test_refresh_handles_deletion(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        (project / "app" / "service.py").unlink()
        result = intel.refresh()
        assert result.get("deleted") == 1
        assert not any(
            s.file_path.replace("\\", "/").endswith("app/service.py")
            for s in intel.find_symbols("run")
        )

    def test_secrets_never_indexed(self, project: Path):
        (project / ".env").write_text("SECRET=x\n", encoding="utf-8")
        (project / "server.key").write_text("keydata\n", encoding="utf-8")
        intel = ProjectIntelligence(project)
        intel.scan()
        listed = intel.file_list()
        assert not any(".env" in p for p in listed)
        assert not any(".key" in p for p in listed)


class TestDependencies:
    def test_from_import_resolves_submodule(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        auth = str(project / "app" / "auth.py")
        models = str(project / "app" / "models.py")
        assert models in intel.dependencies_of(auth)

    def test_dependents_reverse_lookup(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        auth = str(project / "app" / "auth.py")
        models = str(project / "app" / "models.py")
        assert auth in intel.dependents_of(models)

    def test_related_includes_tests(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        service = str(project / "app" / "service.py")
        related = intel.related(service)
        assert any("test_service" in r for r in related)

    def test_reindex_does_not_drop_incoming_edges(self, project: Path):
        """Re-indexing models.py must not erase auth.py's edge to it."""
        intel = ProjectIntelligence(project)
        intel.scan()
        auth = str(project / "app" / "auth.py")
        models = str(project / "app" / "models.py")
        assert models in intel.dependencies_of(auth)
        # Touch + re-index the target file only.
        intel.on_file_changed(models)
        assert models in intel.dependencies_of(auth)

    def test_find_related_tests_by_naming(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        tests = intel.find_related_tests(project / "app" / "service.py")
        assert any("test_service" in t for t in tests)


class TestSymbols:
    def test_find_symbols_substring(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        names = [s.name for s in intel.find_symbols("auth")]
        assert "authenticate" in names

    def test_find_definitions(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        defs = intel.find_definitions("load_user")
        assert len(defs) == 1
        assert defs[0].kind == "function"

    def test_find_symbols_kind_filter(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        classes = intel.find_symbols("User", kind="class")
        assert all(s.kind == "class" for s in classes)
        assert len(classes) == 1


class TestSearch:
    def test_literal_search(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        hits = intel.search("authenticate")
        assert any("auth.py" in h["file"] for h in hits)
        assert all(set(h) == {"file", "line", "content"} for h in hits)

    def test_regex_search(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        hits = intel.search(r"def \w+\(", regex=True)
        assert len(hits) >= 3

    def test_search_is_cached(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        first = intel.search("authenticate")
        second = intel.search("authenticate")
        assert first == second

    def test_search_cache_invalidated_on_change(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        before = intel.search("logout")
        assert before == []
        (project / "app" / "auth.py").write_text(
            "from app import models\n\ndef logout() -> None:\n    pass\n",
            encoding="utf-8",
        )
        intel.refresh()
        after = intel.search("logout")
        assert len(after) == 1

    def test_glob_from_index(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        py = intel.glob("*.py")
        assert len(py) == 5
        assert all(p.endswith(".py") for p in py)


class TestContextCandidates:
    def test_candidates_ranked(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        cands = intel.context_candidates("fix authenticate in auth", max_results=5)
        assert cands
        assert "auth" in cands[0].path

    def test_candidates_boost_dependency_seeds(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        matches = {"app/auth.py": ["1: from app import models"]}
        cands = intel.context_candidates(
            "fix models", search_matches=matches, max_results=10
        )
        boosted = {c.path: c for c in cands}
        target = boosted.get("app/models.py")
        if target is not None:
            assert target.signals.get("dependency_seed") == 1.0

    def test_candidates_empty_before_scan(self, project: Path):
        intel = ProjectIntelligence(project)
        assert intel.context_candidates("anything") == []


class TestInvalidation:
    def test_invalidate_all(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        intel.invalidate()
        assert intel.file_list() == []
        stats = intel.scan()
        assert stats["files"] == 6

    def test_invalidate_single_file(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        intel.invalidate(project / "app" / "auth.py")
        assert len(intel.find_symbols("authenticate")) == 0
        assert len(intel.find_symbols("load_user")) == 1

    def test_on_file_changed_reindexes(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        (project / "app" / "auth.py").write_text(
            "from app import models\n\ndef logout() -> None:\n    pass\n",
            encoding="utf-8",
        )
        intel.on_file_changed(project / "app" / "auth.py")
        assert len(intel.find_symbols("logout")) == 1
        assert len(intel.find_symbols("authenticate")) == 0

    def test_on_file_changed_deleted_file(self, project: Path):
        intel = ProjectIntelligence(project)
        intel.scan()
        target = project / "app" / "service.py"
        target.unlink()
        intel.on_file_changed(target)
        assert not any(
            s.file_path.replace("\\", "/").endswith("app/service.py")
            for s in intel.find_symbols("run")
        )


class TestNoDuplicateSystems:
    def test_facade_reuses_existing_components(self, project: Path):
        from harness_core.analysis.relevance import RelevanceRanker
        from harness_core.cache.search_cache import SearchCache
        from harness_core.indexing.dependency_graph import DependencyGraph
        from harness_core.indexing.symbols import SymbolIndex

        intel = ProjectIntelligence(project)
        assert isinstance(intel.symbols, SymbolIndex)
        assert isinstance(intel.dependencies, DependencyGraph)
        assert isinstance(intel.ranker, RelevanceRanker)
        assert isinstance(intel.search_cache, SearchCache)
