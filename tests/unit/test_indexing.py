"""Tests for the indexing subsystem (Phase C).

Covers: SymbolIndex (Python/JS/Rust/Go parsing, search, lookup),
DependencyGraph (edges, transitive deps, cycle detection).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core.indexing.dependency_graph import DependencyGraph
from harness_core.indexing.symbols import SymbolIndex


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def py_project(tmp_path: Path) -> Path:
    """Create a temporary Python project for indexing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "main.py").write_text(
        '"""Main module."""\n'
        'import os\n'
        'from pathlib import Path\n\n'
        'class AppConfig:\n'
        '    """Application config."""\n'
        '    def __init__(self, name: str):\n'
        '        self.name = name\n\n'
        '    def load(self) -> None:\n'
        '        pass\n\n'
        'def create_app(name: str) -> AppConfig:\n'
        '    return AppConfig(name)\n\n'
        'async def fetch_data() -> list[dict]:\n'
        '    return []\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "utils.py").write_text(
        'import json\n'
        'from typing import Any\n\n'
        'def helper(x: Any) -> Any:\n'
        '    return x\n\n'
        'def compute(a: int, b: int) -> int:\n'
        '    return a + b\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "auth.py").write_text(
        'from .main import AppConfig\n'
        'from .utils import helper\n\n'
        'class AuthManager:\n'
        '    def login(self, user: str) -> bool:\n'
        '        return True\n'
        '    def logout(self) -> None:\n'
        '        pass\n',
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        'from src.main import create_app\n\n'
        'def test_create_app():\n'
        '    app = create_app("test")\n'
        '    assert app.name == "test"\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def js_project(tmp_path: Path) -> Path:
    """Create a temporary JS/TS project for indexing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.ts").write_text(
        'import { helper } from "./utils";\n'
        'export function main() {\n'
        '    return helper();\n'
        '}\n'
        'export class App {\n'
        '    constructor() {}\n'
        '}\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "utils.ts").write_text(
        'export function helper(): string {\n'
        '    return "ok";\n'
        '}\n'
        'export const compute = (a: number) => a * 2;\n',
        encoding="utf-8",
    )
    return tmp_path


# ── SymbolIndex Tests ────────────────────────────────────────────────────────

class TestSymbolIndexPython:
    def test_index_python_file(self, py_project: Path):
        idx = SymbolIndex()
        count = idx.index_file(py_project / "src" / "main.py")
        assert count >= 3  # AppConfig, create_app, fetch_data

    def test_finds_classes(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        classes = idx.get_symbols_by_kind("class")
        names = [s.name for s in classes]
        assert "AppConfig" in names

    def test_finds_functions(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        funcs = idx.get_symbols_by_kind("function")
        names = [s.name for s in funcs]
        assert "create_app" in names
        assert "fetch_data" in names

    def test_finds_methods(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        methods = idx.get_symbols_by_kind("method")
        names = [s.name for s in methods]
        assert "__init__" in names
        assert "load" in names

    def test_detects_async(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        funcs = idx.get_symbols_by_kind("function")
        async_funcs = [s for s in funcs if s.is_async]
        assert len(async_funcs) >= 1
        assert async_funcs[0].name == "fetch_data"

    def test_finds_imports(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        imports = idx.get_imports_in_file(py_project / "src" / "main.py")
        modules = [i.module for i in imports]
        assert "os" in modules

    def test_find_definition(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        results = idx.find_definition("AppConfig")
        assert len(results) >= 1
        assert results[0].kind == "class"

    def test_find_imports_of_module(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "auth.py")
        results = idx.find_imports("main")
        assert len(results) >= 1

    def test_search_by_name(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        results = idx.search("app")
        assert len(results) >= 1

    def test_index_directory(self, py_project: Path):
        idx = SymbolIndex()
        count = idx.index_directory(py_project)
        assert count >= 5

    def test_dedup_indexing(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_file(py_project / "src" / "main.py")
        count2 = idx.index_file(py_project / "src" / "main.py")
        assert count2 == 0  # already indexed

    def test_stats(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_directory(py_project)
        stats = idx.stats
        assert stats["files"] >= 3
        assert stats["symbols"] >= 5


class TestSymbolIndexJavaScript:
    def test_index_js_file(self, js_project: Path):
        idx = SymbolIndex()
        count = idx.index_file(js_project / "src" / "index.ts")
        assert count >= 2  # main function, App class

    def test_finds_js_functions(self, js_project: Path):
        idx = SymbolIndex()
        idx.index_file(js_project / "src" / "index.ts")
        funcs = idx.get_symbols_by_kind("function")
        names = [s.name for s in funcs]
        assert "main" in names

    def test_finds_js_classes(self, js_project: Path):
        idx = SymbolIndex()
        idx.index_file(js_project / "src" / "index.ts")
        classes = idx.get_symbols_by_kind("class")
        names = [s.name for s in classes]
        assert "App" in names

    def test_finds_js_imports(self, js_project: Path):
        idx = SymbolIndex()
        idx.index_file(js_project / "src" / "index.ts")
        imports = idx.get_imports_in_file(js_project / "src" / "index.ts")
        assert len(imports) >= 1
        assert imports[0].module == "./utils"


class TestSymbolIndexEdgeCases:
    def test_empty_file(self, tmp_path: Path):
        (tmp_path / "empty.py").write_text("", encoding="utf-8")
        idx = SymbolIndex()
        count = idx.index_file(tmp_path / "empty.py")
        assert count == 0

    def test_nonexistent_file(self, tmp_path: Path):
        idx = SymbolIndex()
        count = idx.index_file(tmp_path / "nope.py")
        assert count == 0

    def test_clear(self, py_project: Path):
        idx = SymbolIndex()
        idx.index_directory(py_project)
        assert idx.stats["symbols"] > 0
        idx.clear()
        assert idx.stats["symbols"] == 0


# ── DependencyGraph Tests ────────────────────────────────────────────────────

class TestDependencyGraphBasic:
    def test_add_edge(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        deps = g.get_dependencies("a.py")
        assert any("b.py" in d for d in deps)

    def test_reverse_edge(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        deps = g.get_dependents("b.py")
        assert any("a.py" in d for d in deps)

    def test_multiple_edges(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("a.py", "c.py")
        g.add_edge("b.py", "c.py")
        deps = g.get_dependencies("a.py")
        assert any("b.py" in d for d in deps)
        assert any("c.py" in d for d in deps)


class TestDependencyGraphTransitive:
    def test_transitive_deps(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        g.add_edge("c.py", "d.py")
        deps = g.get_transitive_dependencies("a.py")
        assert any("b.py" in d for d in deps)
        assert any("c.py" in d for d in deps)
        assert any("d.py" in d for d in deps)

    def test_transitive_deps_max_depth(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        g.add_edge("c.py", "d.py")
        deps = g.get_transitive_dependencies("a.py", max_depth=1)
        assert any("b.py" in d for d in deps)
        assert not any("c.py" in d for d in deps)

    def test_transitive_dependents(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        deps = g.get_transitive_dependents("c.py")
        assert any("a.py" in d for d in deps)
        assert any("b.py" in d for d in deps)

    def test_related_files(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        g.add_edge("d.py", "a.py")
        related = g.get_related_files("a.py")
        assert any("b.py" in r for r in related)
        assert any("c.py" in r for r in related)
        assert any("d.py" in r for r in related)
        assert not any("a.py" in r and r.endswith("a.py") for r in related)


class TestDependencyGraphCycle:
    def test_no_cycle(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        assert not g.has_cycle()

    def test_self_cycle(self):
        g = DependencyGraph()
        g.add_edge("a.py", "a.py")
        assert g.has_cycle()

    def test_indirect_cycle(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        g.add_edge("c.py", "a.py")
        assert g.has_cycle()


class TestDependencyGraphModuleResolution:
    def test_register_and_resolve(self):
        g = DependencyGraph()
        g.register_module("src.main", "/path/to/main.py")
        result = g.resolve_module("src.main")
        assert result is not None

    def test_resolve_unknown(self):
        g = DependencyGraph()
        result = g.resolve_module("nope")
        assert result is None


class TestDependencyGraphStats:
    def test_stats(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.add_edge("b.py", "c.py")
        stats = g.stats
        assert stats["edges"] == 2
        assert stats["files"] >= 2

    def test_clear(self):
        g = DependencyGraph()
        g.add_edge("a.py", "b.py")
        g.clear()
        assert g.stats["edges"] == 0
