"""Tests for repository analysis and relevance ranking (Phase B).

Covers: RepositoryAnalyzer ecosystem detection, file discovery,
RelevanceRanker scoring, keyword extraction, and ranking.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from harness_core.analysis.repository import (
    Ecosystem,
    RepositoryAnalyzer,
    RepositoryInfo,
    _detect_python,
    _detect_nodejs,
    _detect_typescript,
    _detect_rust,
    _detect_go,
)
from harness_core.analysis.relevance import (
    RelevanceConfig,
    RelevanceRanker,
    RelevanceScore,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a Python project."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "auth.py").write_text("class Auth: pass\n", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_auth.py").write_text("def test_auth(): pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def nodejs_project(tmp_path: Path) -> Path:
    """Create a Node.js project."""
    import json
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "test-app",
        "dependencies": {"express": "^4.0.0"},
        "devDependencies": {"jest": "^29.0.0"},
    }), encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.js").write_text("console.log('hello');\n", encoding="utf-8")
    (tmp_path / "src" / "auth.js").write_text("module.exports = {};\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "index.test.js").write_text("test('works', () => {});\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def rust_project(tmp_path: Path) -> Path:
    """Create a Rust project."""
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "integration.rs").write_text("#[test]\nfn test_it() {}\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def mixed_project(tmp_path: Path) -> Path:
    """Create a project with multiple ecosystems."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name":"test"}', encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text('[package]\nname="test"\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "src" / "index.js").write_text("console.log();\n", encoding="utf-8")
    return tmp_path


# ── Ecosystem Detector Tests ────────────────────────────────────────────────


class TestPythonDetector:
    def test_detects_python(self, python_project: Path):
        eco = _detect_python(python_project)
        assert eco is not None
        assert eco.name == "python"
        assert eco.confidence > 0
        assert "pyproject.toml" in eco.config_files

    def test_no_python(self, tmp_path: Path):
        eco = _detect_python(tmp_path)
        assert eco is None


class TestNodejsDetector:
    def test_detects_nodejs(self, nodejs_project: Path):
        eco = _detect_nodejs(nodejs_project)
        assert eco is not None
        assert eco.name == "nodejs"

    def test_no_nodejs(self, tmp_path: Path):
        eco = _detect_nodejs(tmp_path)
        assert eco is None


class TestRustDetector:
    def test_detects_rust(self, rust_project: Path):
        eco = _detect_rust(rust_project)
        assert eco is not None
        assert eco.name == "rust"
        assert "cargo test" in eco.test_commands

    def test_no_rust(self, tmp_path: Path):
        eco = _detect_rust(tmp_path)
        assert eco is None


class TestGoDetector:
    def test_detects_go(self, tmp_path: Path):
        (tmp_path / "go.mod").write_text("module test\n", encoding="utf-8")
        eco = _detect_go(tmp_path)
        assert eco is not None
        assert eco.name == "go"

    def test_no_go(self, tmp_path: Path):
        eco = _detect_go(tmp_path)
        assert eco is None


# ── RepositoryAnalyzer Tests ────────────────────────────────────────────────


class TestRepositoryAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_python(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        info = await analyzer.analyze()

        assert info.has_git
        assert len(info.ecosystems) >= 1
        assert info.primary_ecosystem is not None
        assert info.primary_ecosystem.name == "python"
        assert info.source_files > 0
        assert info.test_files > 0

    @pytest.mark.asyncio
    async def test_analyze_nodejs(self, nodejs_project: Path):
        analyzer = RepositoryAnalyzer(nodejs_project)
        info = await analyzer.analyze()

        assert info.primary_ecosystem is not None
        assert info.primary_ecosystem.name == "nodejs"

    @pytest.mark.asyncio
    async def test_analyze_rust(self, rust_project: Path):
        analyzer = RepositoryAnalyzer(rust_project)
        info = await analyzer.analyze()

        assert info.primary_ecosystem is not None
        assert info.primary_ecosystem.name == "rust"

    @pytest.mark.asyncio
    async def test_analyze_mixed(self, mixed_project: Path):
        analyzer = RepositoryAnalyzer(mixed_project)
        info = await analyzer.analyze()

        assert len(info.ecosystems) >= 2
        names = {e.name for e in info.ecosystems}
        assert "python" in names
        assert "nodejs" in names

    @pytest.mark.asyncio
    async def test_caching(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        info1 = await analyzer.analyze()
        info2 = await analyzer.analyze()
        assert info1 is info2  # same object (cached)

    @pytest.mark.asyncio
    async def test_force_refresh(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        info1 = await analyzer.analyze()
        info2 = await analyzer.analyze(force=True)
        assert info1 is not info2  # new object

    @pytest.mark.asyncio
    async def test_important_files(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        info = await analyzer.analyze()
        assert any("README" in f for f in info.important_files)

    @pytest.mark.asyncio
    async def test_to_dict(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        info = await analyzer.analyze()
        d = info.to_dict()
        assert "ecosystems" in d
        assert "total_files" in d

    @pytest.mark.asyncio
    async def test_get_source_files(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        await analyzer.analyze()
        source = analyzer.get_source_files()
        assert any(f.endswith(".py") for f in source)

    @pytest.mark.asyncio
    async def test_get_test_files(self, python_project: Path):
        analyzer = RepositoryAnalyzer(python_project)
        await analyzer.analyze()
        tests = analyzer.get_test_files()
        assert any("test_" in f for f in tests)


# ── RelevanceRanker Tests ───────────────────────────────────────────────────


class TestRelevanceRanker:
    def test_keyword_extraction(self):
        ranker = RelevanceRanker()
        keywords = ranker._extract_keywords("Fix the authentication middleware bug")
        assert "authentication" in keywords
        assert "middleware" in keywords
        assert "bug" in keywords
        # Stop words should be removed
        assert "the" not in keywords

    def test_filename_score(self):
        ranker = RelevanceRanker()
        score = ranker._score_filename("auth.py", ["auth", "authentication"])
        assert score > 0.5

    def test_filename_no_match(self):
        ranker = RelevanceRanker()
        score = ranker._score_filename("utils.py", ["authentication", "middleware"])
        assert score < 0.3

    def test_path_score(self):
        ranker = RelevanceRanker()
        score = ranker._score_path("src/auth/middleware.py", ["auth", "middleware"])
        assert score > 0.5

    def test_extension_coding(self):
        ranker = RelevanceRanker()
        score = ranker._score_extension("main.py", "Fix the bug in the code")
        assert score > 0.5

    def test_extension_config(self):
        ranker = RelevanceRanker()
        score = ranker._score_extension("pyproject.toml", "Update the project config")
        assert score > 0.4

    def test_importance_readme(self):
        ranker = RelevanceRanker()
        score = ranker._score_importance("README.md")
        assert score == 1.0

    def test_importance普通文件(self):
        ranker = RelevanceRanker()
        score = ranker._score_importance("random_file.xyz")
        assert score < 0.3

    def test_test_proximity_for_test_file(self):
        ranker = RelevanceRanker()
        score = ranker._score_test_proximity("test_auth.py", [])
        assert score >= 0.6

    def test_test_proximity_with_corresponding_test(self):
        ranker = RelevanceRanker()
        score = ranker._score_test_proximity("auth.py", ["tests/test_auth.py", "test_auth.py"])
        assert score >= 0.5

    def test_score_file(self):
        ranker = RelevanceRanker()
        result = ranker.score_file(
            "src/auth.py",
            "Fix the authentication bug",
            all_files=["src/auth.py", "tests/test_auth.py"],
        )
        assert result.total_score > 0
        assert "filename_match" in result.signals

    def test_rank_files(self):
        ranker = RelevanceRanker()
        files = [
            "src/utils.py",
            "src/auth.py",
            "tests/test_auth.py",
            "README.md",
            "random.xyz",
        ]
        ranked = ranker.rank_files(files, "Fix the authentication middleware")
        assert len(ranked) > 0
        # auth.py should rank higher than random.xyz
        auth_rank = next(i for i, r in enumerate(ranked) if "auth" in r.path and "test" not in r.path)
        random_rank = next(i for i, r in enumerate(ranked) if "random" in r.path)
        assert auth_rank < random_rank

    def test_rank_max_results(self):
        ranker = RelevanceRanker()
        files = [f"file_{i}.py" for i in range(100)]
        ranked = ranker.rank_files(files, "test", max_results=10)
        assert len(ranked) <= 10

    def test_search_match_boost(self):
        ranker = RelevanceRanker()
        search_matches = {"src/auth.py": ["auth.py:1: class Auth"]}
        result = ranker.score_file(
            "src/auth.py",
            "Fix authentication",
            search_matches=search_matches,
        )
        assert result.signals["search_match"] == 1.0

    def test_empty_task(self):
        ranker = RelevanceRanker()
        result = ranker.score_file("main.py", "")
        assert result.total_score >= 0

    def test_config(self):
        config = RelevanceConfig(filename_match=0.5, path_match=0.5)
        ranker = RelevanceRanker(config)
        result = ranker.score_file("auth.py", "auth")
        assert result.total_score > 0
