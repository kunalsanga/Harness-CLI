"""Performance baseline measurements for Harness Engineering CLI.

Measures:
  1. CLI startup
  2. Repository discovery
  3. File search (grep/glob)
  4. Context construction
  5. Session operations
  6. Tool dispatch
  7. Model routing
  8. Multi-agent orchestration overhead
  9. Memory usage
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────────


def _measure_time(func, *args, iterations=5, **kwargs) -> dict[str, float]:
    """Measure function execution time with statistical analysis."""
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    return {
        "median": times[len(times) // 2],
        "p95": times[int(len(times) * 0.95)],
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
    }


async def _async_measure_time(func, *args, iterations=5, **kwargs) -> dict[str, float]:
    """Measure async function execution time."""
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    return {
        "median": times[len(times) // 2],
        "p95": times[int(len(times) * 0.95)],
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
    }


def _run_async(func, *args, iterations=5, **kwargs) -> dict[str, float]:
    """Run an async function synchronously with timing."""
    loop = asyncio.new_event_loop()
    try:
        times = []
        for _ in range(iterations):
            gc.collect()
            start = time.perf_counter()
            loop.run_until_complete(func(*args, **kwargs))
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        times.sort()
        return {
            "median": times[len(times) // 2],
            "p95": times[int(len(times) * 0.95)],
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
        }
    finally:
        loop.close()


def _create_temp_repo(file_count: int = 100, dir_depth: int = 3) -> Path:
    """Create a temporary repository for benchmarking."""
    tmpdir = Path(tempfile.mkdtemp(prefix="harness_bench_"))

    # Create directory structure
    dirs = [tmpdir]
    for d in range(dir_depth):
        new_dirs = []
        for parent in dirs:
            for i in range(5):
                new_dir = parent / f"dir_{d}_{i}"
                new_dir.mkdir(exist_ok=True)
                new_dirs.append(new_dir)
        dirs = new_dirs

    # Create files
    files_per_dir = max(1, file_count // len(dirs)) if dirs else 1
    for d in dirs:
        for i in range(files_per_dir):
            fpath = d / f"file_{i}.py"
            content = f'"""Module {fpath.stem}."""\n\n'
            content += "import os\nimport sys\nfrom pathlib import Path\n\n"
            content += f"def func_{i}():\n"
            content += f'    """Function in {fpath.name}."""\n'
            content += f'    x = {i}\n'
            content += f"    return x + 1\n\n"
            content += f"class Test{i}:\n"
            content += f'    """Test class {i}."""\n'
            content += f"    def test_method(self):\n"
            content += f"        assert func_{i}() == {i + 1}\n\n"
            content += f'# TODO: fix issue in {fpath.name}\n'
            content += f'AUTHENTICATION_TOKEN = "placeholder_token_{i}"\n'
            fpath.write_text(content, encoding="utf-8")

    # Create a pyproject.toml
    (tmpdir / "pyproject.toml").write_text('[project]\nname = "bench"\n', encoding="utf-8")
    (tmpdir / "README.md").write_text("# Benchmark Repository\n", encoding="utf-8")
    (tmpdir / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")

    return tmpdir


# ── Benchmark Functions ───────────────────────────────────────────────────


def bench_cli_startup() -> dict[str, Any]:
    """Measure CLI startup time."""
    print("  Measuring CLI startup...")

    result = _measure_time(
        lambda: subprocess.run(
            [sys.executable, "-m", "harness_core.cli.main", "--help"],
            capture_output=True, text=True, cwd=str(Path.cwd()),
            timeout=30,
        ),
        iterations=10,
    )

    return {
        "name": "CLI Startup (--help)",
        "unit": "seconds",
        "measurements": result,
    }


def bench_doctor() -> dict[str, Any]:
    """Measure harness doctor execution time."""
    print("  Measuring harness doctor...")

    result = _measure_time(
        lambda: subprocess.run(
            [sys.executable, "-m", "harness_core.cli.main", "doctor"],
            capture_output=True, text=True, cwd=str(Path.cwd()),
            timeout=30,
        ),
        iterations=5,
    )

    return {
        "name": "harness doctor",
        "unit": "seconds",
        "measurements": result,
    }


def bench_repository_discovery() -> dict[str, Any]:
    """Measure project discovery time across different repo sizes."""
    print("  Measuring repository discovery...")

    results = {}
    for size_name, file_count in [("small", 10), ("medium", 100), ("large", 500)]:
        tmpdir = _create_temp_repo(file_count=file_count)
        try:
            from harness_core.context.engine import ContextEngine
            engine = ContextEngine(tmpdir)

            result = _run_async(engine.discover_project, iterations=5)
            results[size_name] = result
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "name": "Repository Discovery",
        "unit": "seconds",
        "sizes": results,
    }


def bench_file_search() -> dict[str, Any]:
    """Measure grep and glob search performance."""
    print("  Measuring file search...")

    results = {}
    for size_name, file_count in [("small", 10), ("medium", 100), ("large", 500)]:
        tmpdir = _create_temp_repo(file_count=file_count)
        try:
            from harness_core.tools.search import GlobTool, GrepTool
            glob_tool = GlobTool()
            grep_tool = GrepTool()
            glob_tool.workspace_root = tmpdir
            grep_tool.workspace_root = tmpdir

            glob_time = _run_async(
                glob_tool.execute, {"pattern": "**/*.py"}, iterations=5
            )

            grep_time = _run_async(
                grep_tool.execute,
                {"pattern": "AUTHENTICATION_TOKEN", "path": str(tmpdir)},
                iterations=5,
            )

            results[size_name] = {"glob": glob_time, "grep": grep_time}
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "name": "File Search (Glob + Grep)",
        "unit": "seconds",
        "sizes": results,
    }


def bench_session_operations() -> dict[str, Any]:
    """Measure session SQLite operations."""
    print("  Measuring session operations...")

    from harness_core.session.manager import SessionManager
    from harness_core.session.domain import MemoryType

    with tempfile.TemporaryDirectory(prefix="harness_session_bench_") as tmpdir:
        manager = SessionManager()

        # Measure session creation
        create_time = _measure_time(
            lambda: manager.create_session(workspace_path=tmpdir, title="Bench"),
            iterations=50,
        )

        session = manager.create_session(workspace_path=tmpdir, title="List Test")

        # Measure listing
        list_time = _measure_time(
            lambda: manager.storage.list_sessions(limit=100),
            iterations=50,
        )

        # Measure run creation
        run_time = _measure_time(
            lambda: manager.start_run(
                session.session_id, task="Bench task", model_id="test", provider="test"
            ),
            iterations=50,
        )

        # Measure checkpoint creation
        checkpoint_time = _measure_time(
            lambda: manager.create_checkpoint(
                session.session_id, git_head="abc123", git_branch="main"
            ),
            iterations=50,
        )

        # Measure memory add
        mem_time = _measure_time(
            lambda: manager.add_memory(
                session.session_id, MemoryType.NOTE,
                "Benchmark memory item for performance testing", importance=0.5,
            ),
            iterations=50,
        )

        return {
            "name": "Session Operations (SQLite)",
            "unit": "seconds",
            "measurements": {
                "create_session": create_time,
                "list_sessions": list_time,
                "create_run": run_time,
                "create_checkpoint": checkpoint_time,
                "add_memory": mem_time,
            },
        }


def bench_context_construction() -> dict[str, Any]:
    """Measure context engine performance."""
    print("  Measuring context construction...")

    results = {}
    for size_name, file_count in [("small", 10), ("medium", 100), ("large", 500)]:
        tmpdir = _create_temp_repo(file_count=file_count)
        try:
            from harness_core.context.engine import ContextEngine
            engine = ContextEngine(tmpdir)

            discover_time = _run_async(engine.discover_project, iterations=5)
            context_time = _run_async(
                engine.assemble_context, "Fix the authentication bug", iterations=5
            )

            results[size_name] = {
                "discover_project": discover_time,
                "assemble_context": context_time,
            }
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "name": "Context Construction",
        "unit": "seconds",
        "sizes": results,
    }


def bench_model_routing() -> dict[str, Any]:
    """Measure model routing overhead."""
    print("  Measuring model routing...")

    from harness_core.routing.scoring import ScoringContext, ScoringWeights, rank_models
    from harness_core.providers.base import ModelInfo

    # Create mock models for scoring benchmark
    mock_models = []
    for i in range(50):
        model = ModelInfo(
            id=f"model-{i}",
            name=f"Model {i}",
            provider="openrouter",
            context_window=4096,
            supports_tools=True,
            is_free=(i % 3 == 0),
        )
        mock_models.append(model)

    ctx = ScoringContext(
        task_description="Fix the authentication bug in the login module",
        requires_tools=True,
        task_type="bug_fix",
    )
    weights = ScoringWeights()

    scoring_time = _measure_time(
        lambda: rank_models(mock_models, ctx, weights), iterations=50
    )

    # Measure classification
    from harness_core.classifier.classifier import TaskClassifier
    classifier = TaskClassifier()
    classify_time = _measure_time(
        lambda: classifier.classify_with_confidence("Fix the authentication bug"),
        iterations=50,
    )

    # Measure task-aware routing
    from harness_core.routing.task_aware import TaskAwareRouter
    from harness_core.models.registry import ModelRegistry
    from harness_core.providers.base import CompletionRequest

    registry = ModelRegistry()
    task_aware = TaskAwareRouter(registry=registry, classifier=classifier)
    routing_time = _measure_time(
        lambda: task_aware.classify_task(
            CompletionRequest(messages=[{"role": "user", "content": "Fix the failing tests"}])
        ),
        iterations=50,
    )

    return {
        "name": "Model Routing",
        "unit": "seconds",
        "measurements": {
            "score_50_models": scoring_time,
            "classify_task": classify_time,
            "classify_task_aware": routing_time,
        },
    }


def bench_tool_dispatch() -> dict[str, Any]:
    """Measure tool dispatch overhead."""
    print("  Measuring tool dispatch...")

    from harness_core.tools.search import GlobTool, GrepTool
    from harness_core.tools.filesystem import ReadFileTool, ListFilesTool

    tools = {
        "glob": GlobTool(),
        "grep": GrepTool(),
        "read": ReadFileTool(),
        "list": ListFilesTool(),
    }

    tmpdir = _create_temp_repo(file_count=50)
    try:
        for tool in tools.values():
            tool.workspace_root = tmpdir

        results = {}
        results["glob"] = _run_async(
            tools["glob"].execute, {"pattern": "**/*.py"}, iterations=20
        )
        results["list_files"] = _run_async(
            tools["list"].execute, {"path": str(tmpdir)}, iterations=20
        )
        readme = tmpdir / "README.md"
        results["read_file"] = _run_async(
            tools["read"].execute, {"path": str(readme)}, iterations=20
        )

        return {"name": "Tool Dispatch", "unit": "seconds", "measurements": results}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_orchestration_overhead() -> dict[str, Any]:
    """Measure multi-agent orchestration overhead (no LLM calls)."""
    print("  Measuring orchestration overhead...")

    from harness_core.agents.orchestrator import Orchestrator, ExecutionMode

    orchestrator = Orchestrator()

    decompose_time = _measure_time(
        lambda: orchestrator.decompose_task("Fix the authentication bug and add tests"),
        iterations=50,
    )

    async def _orchestrate():
        return await orchestrator.execute(
            "Fix the authentication bug", mode=ExecutionMode.SINGLE,
        )

    orchestrate_time = _run_async(_orchestrate, iterations=10)

    graph = orchestrator.decompose_task("Test task")
    validate_time = _measure_time(lambda: graph.validate(), iterations=100)

    return {
        "name": "Orchestration Overhead",
        "unit": "seconds",
        "measurements": {
            "decompose_task": decompose_time,
            "orchestrate_single": orchestrate_time,
            "validate_graph": validate_time,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────


def run_all_benchmarks() -> dict[str, Any]:
    """Run all benchmarks and return results."""

    print("=" * 60)
    print("HARNESS ENGINEERING CLI — PERFORMANCE BASELINE")
    print("=" * 60)
    print()

    all_results = {}
    benchmarks = [
        ("cli_startup", bench_cli_startup),
        ("doctor", bench_doctor),
        ("repository_discovery", bench_repository_discovery),
        ("file_search", bench_file_search),
        ("session_operations", bench_session_operations),
        ("context_construction", bench_context_construction),
        ("model_routing", bench_model_routing),
        ("tool_dispatch", bench_tool_dispatch),
        ("orchestration_overhead", bench_orchestration_overhead),
    ]

    for name, bench_func in benchmarks:
        try:
            print(f"[{name}]")
            result = bench_func()
            all_results[name] = result
            print(f"  OK")
            print()
        except Exception as e:
            import traceback
            all_results[name] = {"error": str(e)}
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

    return all_results


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary of benchmark results."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    for key, data in results.items():
        if "error" in data:
            print(f"  {key}: FAILED ({data['error']})")
            continue

        name = data.get("name", key)
        unit = data.get("unit", "")
        measurements = data.get("measurements", {})
        sizes = data.get("sizes", {})

        if measurements:
            print(f"  {name}:")
            if isinstance(measurements, dict):
                for k, v in measurements.items():
                    if isinstance(v, dict) and "median" in v:
                        print(f"    {k}: {v['median']:.4f}s (p95: {v['p95']:.4f}s)")
                    elif isinstance(v, (int, float)):
                        print(f"    {k}: {v:.2f}{unit}")
            print()

        if sizes:
            print(f"  {name}:")
            for size_name, size_data in sizes.items():
                if isinstance(size_data, dict):
                    print(f"    [{size_name}]:")
                    for k, v in size_data.items():
                        if isinstance(v, dict) and "median" in v:
                            print(f"      {k}: {v['median']:.4f}s (p95: {v['p95']:.4f}s)")
            print()


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_summary(results)

    output_path = Path("benchmarks/baseline_results.json")
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
