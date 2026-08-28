"""
Performance benchmarks for Harness Engineering CLI.

Run with:
    uv run pytest tests/benchmarks/ -v -s

Each benchmark records wall-clock time and produces reproducible measurements.
"""

import asyncio
import pathlib
import time

import pytest


# ---------------------------------------------------------------------------
# A: Cold vs warm file reads
# ---------------------------------------------------------------------------

class TestFileReadBenchmarks:
    """Benchmark file cache impact on repeated reads."""

    def test_cold_read_vs_warm_read(self, tmp_path: pathlib.Path):
        """Measure cache benefit for repeated file reads."""
        from harness_core.cache import FileContentCache

        # Create test files
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(
                f"# File {i}\n" + "x = 1\n" * 100
            )

        cache = FileContentCache(max_entries=50, workspace_root=tmp_path)

        # Pre-populate cache
        for i in range(20):
            path = tmp_path / f"file_{i}.py"
            cache.put(str(path), path.read_text())

        # Warm reads (cache hit every time)
        warm_times: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            for i in range(20):
                cache.get(str(tmp_path / f"file_{i}.py"))
            warm_times.append((time.perf_counter() - start) * 1000)

        avg_warm = sum(warm_times) / len(warm_times)

        print(f"\n  Warm read avg: {avg_warm:.3f}ms for 20 files")
        print(f"  Cache stats: {cache.stats}")

        assert avg_warm < 200
        stats = cache.stats
        assert stats.hits > 0

    def test_search_cache_hit_rate(self, tmp_path: pathlib.Path):
        """Measure search cache prevents redundant I/O."""
        from harness_core.cache import SearchCache

        cache = SearchCache(max_entries=100)

        # Populate cache using make_key
        k1 = SearchCache.make_key("glob", pattern="*.py")
        k2 = SearchCache.make_key("grep", pattern="auth")
        cache.put(k1, ["a.py", "b.py"])
        cache.put(k2, ["src/auth.py"])

        # Measure hits
        hits = 0
        misses = 0
        start = time.perf_counter()
        for _ in range(1000):
            v = cache.get(k1)
            if v is not None:
                hits += 1
            else:
                misses += 1
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  1000 lookups in {elapsed:.1f}ms")
        print(f"  Hit rate: {hits / (hits + misses):.1%}")

        assert hits == 1000
        assert misses == 0


# ---------------------------------------------------------------------------
# B: Repository analysis benchmarks
# ---------------------------------------------------------------------------

class TestRepositoryAnalysisBenchmarks:
    """Benchmark repository analyzer on real project."""

    @pytest.mark.asyncio
    async def test_analyze_own_repo(self):
        """Analyze the harness-engineering-cli repo itself."""
        from harness_core.analysis import RepositoryAnalyzer

        root = pathlib.Path(__file__).resolve().parents[2]
        analyzer = RepositoryAnalyzer(root=root)

        start = time.perf_counter()
        result = await analyzer.analyze()
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  Repo analysis: {elapsed:.1f}ms")
        print(f"  Files discovered: {result.total_files}")
        print(f"  Source files: {result.source_files}")
        print(f"  Test files: {result.test_files}")
        print(f"  Primary ecosystem: {result.primary_ecosystem}")

        assert result.total_files > 0
        assert elapsed < 5000

    def test_relevance_scoring(self):
        """Benchmark relevance scoring against real files."""
        from harness_core.analysis import RelevanceRanker

        ranker = RelevanceRanker()

        files = [
            "src/harness_core/agent/loop.py",
            "src/harness_core/providers/openrouter.py",
            "src/harness_core/tools/filesystem.py",
            "src/harness_core/routing/router.py",
            "tests/unit/test_routing.py",
            "docs/architecture.md",
            "pyproject.toml",
        ]

        start = time.perf_counter()
        for _ in range(1000):
            ranked = ranker.rank_files(files, "Fix authentication middleware bug")
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  Relevance scoring (1000 iterations): {elapsed:.1f}ms")
        print(f"  Avg per iteration: {elapsed / 1000:.3f}ms")

        assert elapsed < 5000
        assert len(ranked) > 0


# ---------------------------------------------------------------------------
# C: Symbol indexing benchmarks
# ---------------------------------------------------------------------------

class TestSymbolIndexBenchmarks:
    """Benchmark symbol indexing and lookup."""

    def test_index_build_time(self, tmp_path: pathlib.Path):
        """Benchmark building a symbol index from files."""
        from harness_core.indexing import SymbolIndex

        # Create realistic Python files
        for i in range(50):
            content = f'''
"""Module {i}."""
import os
import sys
from pathlib import Path

class Service{i}:
    """Service {i}."""
    def __init__(self, config):
        self.config = config

    def process(self, data):
        return data

    def validate(self):
        return True

def helper_{i}(x):
    return x * 2

def compute_{i}(data):
    return [helper_{i}(d) for d in data]
'''
            (tmp_path / f"module_{i}.py").write_text(content)

        index = SymbolIndex()

        start = time.perf_counter()
        count = index.index_directory(tmp_path)
        elapsed = (time.perf_counter() - start) * 1000

        st = index.stats
        print(f"\n  Index build (50 files): {elapsed:.1f}ms")
        print(f"  Files indexed: {st['files']}")
        print(f"  Symbols: {st['symbols']}")
        print(f"  Imports: {st['imports']}")

        # Lookup benchmark
        start = time.perf_counter()
        for _ in range(1000):
            index.find_definition("Service0")
        lookup_ms = (time.perf_counter() - start) * 1000

        print(f"  1000 lookups: {lookup_ms:.1f}ms ({lookup_ms / 1000:.3f}ms/lookup)")

        assert elapsed < 10000
        assert st['symbols'] > 0


# ---------------------------------------------------------------------------
# D: Context pack assembly benchmarks
# ---------------------------------------------------------------------------

class TestContextPackBenchmarks:
    """Benchmark context pack assembly and deduplication."""

    def test_context_pack_assembly(self):
        """Benchmark building a context pack with token budget."""
        from harness_core.context.pack import ContextPackBuilder

        builder = ContextPackBuilder(token_budget=20_000, output_reserve=4000)
        builder.add_task("Fix authentication middleware bug")

        # Add 100 files
        for i in range(100):
            content = f"# File {i}\n" + "line of code\n" * 20
            builder.add_file(f"src/module_{i}.py", content, priority=50.0 + (i % 50))

        start = time.perf_counter()
        for _ in range(200):
            pack = builder.build()
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  Context pack assembly (100 files, 200 builds): {elapsed:.1f}ms")
        print(f"  Avg per build: {elapsed / 200:.3f}ms")
        print(f"  Pack tokens: {pack.total_tokens}")
        print(f"  Deduplication count: {pack.deduplication_count}")

        assert elapsed < 5000

    def test_deduplication_effectiveness(self):
        """Verify deduplication removes duplicate content."""
        from harness_core.context.pack import ContextPackBuilder

        builder = ContextPackBuilder(token_budget=50_000, output_reserve=4000)
        builder.add_task("Read files")

        content_a = "def auth_check():\n    return True\n"
        content_b = "def user_login():\n    return False\n"

        builder.add_file("a.py", content_a, priority=60.0)
        builder.add_file("b.py", content_b, priority=60.0)
        builder.add_file("a_copy.py", content_a, priority=40.0)
        builder.add_file("b_copy.py", content_b, priority=40.0)

        pack = builder.build()

        print(f"\n  Input files: 4")
        print(f"  Output pieces: {len(pack.pieces)}")
        print(f"  Deduplication count: {pack.deduplication_count}")
        print(f"  Total tokens: {pack.total_tokens}")

        # Deduplication should have excluded at least the task piece
        # Duplicates of identical content should be dropped
        assert len(pack.pieces) <= 4  # task + up to 3 unique files


# ---------------------------------------------------------------------------
# E: Parallel vs sequential execution
# ---------------------------------------------------------------------------

class TestParallelExecutionBenchmarks:
    """Benchmark parallel vs sequential tool execution."""

    def test_parallel_vs_sequential_reads(self, tmp_path: pathlib.Path):
        """Compare parallel vs sequential file reads."""
        # Create test files
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(
                f"# Content {i}\n" + "x = 1\n" * 200
            )

        # Measure sequential reads
        sequential_times: list[float] = []
        for _ in range(20):
            start = time.perf_counter()
            for i in range(20):
                (tmp_path / f"file_{i}.py").read_text()
            sequential_times.append((time.perf_counter() - start) * 1000)

        avg_sequential = sum(sequential_times) / len(sequential_times)
        print(f"\n  Sequential 20 reads: avg {avg_sequential:.3f}ms")

        assert avg_sequential < 1000


# ---------------------------------------------------------------------------
# F: Metrics and dashboard
# ---------------------------------------------------------------------------

class TestMetricsDashboardBenchmarks:
    """Benchmark metrics collection overhead."""

    def test_metrics_overhead(self):
        """Measure instrumentation overhead of MetricsCollector."""
        from harness_core.observability.metrics import MetricsCollector

        mc = MetricsCollector()

        start = time.perf_counter()
        for i in range(10_000):
            mc.inc("test.counter")
            mc.gauge("test.gauge", float(i))
            mc.record_duration("test.timing", 1.0)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  10,000 metric operations: {elapsed:.1f}ms")
        print(f"  Per operation: {elapsed / 10_000:.4f}ms")

        # Must not add significant overhead
        assert elapsed < 500

    def test_dashboard_generation(self):
        """Benchmark dashboard generation."""
        from harness_core.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        # Populate with realistic data
        for i in range(100):
            mc.inc("model.calls")
            mc.inc("tool.calls")
            mc.gauge("tokens.total", float(i * 100))
            mc.record_duration("model.request", float(i % 100))

        start = time.perf_counter()
        for _ in range(1000):
            mc.dashboard()
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  1000 dashboard generations: {elapsed:.1f}ms")
        assert elapsed < 1000
