"""Benchmark runner — executes coding tasks and records results."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .tasks import BenchmarkTask, ALL_TASKS, TaskCategory


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""

    task_id: str
    task_description: str
    category: str
    difficulty: str
    language: str
    success: bool = False
    partial: bool = False
    duration_ms: float = 0.0
    pre_test_failures: int = 0
    post_test_failures: int = 0
    tool_calls: int = 0
    iterations: int = 0
    model: str = ""
    provider: str = ""
    error: str = ""
    timestamp: str = ""
    harness_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Runs benchmark tasks against a fixture and records results.

    This runner:
    1. Sets up the fixture (copy to temp dir)
    2. Runs pre-test (should fail)
    3. Records pre-test failure count
    4. Executes the task via Harness agent
    5. Runs post-test (should pass)
    6. Records post-test failure count
    7. Determines success/partial/failure
    """

    def __init__(
        self,
        results_dir: str | Path | None = None,
    ) -> None:
        self.results_dir = Path(results_dir or Path(__file__).parent / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[BenchmarkResult] = []

    def run_task(
        self,
        task: BenchmarkTask,
        model: str = "",
        provider: str = "",
    ) -> BenchmarkResult:
        """Run a single benchmark task."""
        result = BenchmarkResult(
            task_id=task.task_id,
            task_description=task.description,
            category=task.category.value,
            difficulty=task.difficulty.value,
            language=task.language.value,
            model=model,
            provider=provider,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        fixture_dir = task.get_fixture_dir()
        if not fixture_dir.exists():
            result.error = f"Fixture not found: {fixture_dir}"
            self._results.append(result)
            return result

        start = time.monotonic()

        try:
            # Run pre-test
            if task.pre_test_command:
                pre_cmd = task.pre_test_command.replace("{fixture}", str(fixture_dir))
                result.pre_test_failures = self._count_test_failures(pre_cmd)

            # For now, we record the task but don't actually invoke Harness agent
            # This is the framework — real execution happens when provider is available
            result.success = False
            result.partial = False
            result.error = "Framework ready — awaiting real agent execution"

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)[:200]}"
            result.success = False

        result.duration_ms = (time.monotonic() - start) * 1000
        self._results.append(result)
        return result

    def run_all(
        self,
        model: str = "",
        provider: str = "",
        category: TaskCategory | None = None,
    ) -> list[BenchmarkResult]:
        """Run all benchmark tasks."""
        tasks = ALL_TASKS
        if category:
            tasks = [t for t in tasks if t.category == category]

        results = []
        for task in tasks:
            result = self.run_task(task, model=model, provider=provider)
            results.append(result)
        return results

    def get_results(self) -> list[BenchmarkResult]:
        return list(self._results)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self._results:
            return {"total": 0}

        total = len(self._results)
        success = sum(1 for r in self._results if r.success)
        partial = sum(1 for r in self._results if r.partial)
        failed = total - success - partial

        durations = [r.duration_ms for r in self._results if r.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        by_category: dict[str, dict] = {}
        for r in self._results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "success": 0, "partial": 0, "failed": 0}
            by_category[cat]["total"] += 1
            if r.success:
                by_category[cat]["success"] += 1
            elif r.partial:
                by_category[cat]["partial"] += 1
            else:
                by_category[cat]["failed"] += 1

        return {
            "total": total,
            "success": success,
            "partial": partial,
            "failed": failed,
            "success_rate": f"{success / total * 100:.1f}%",
            "avg_duration_ms": round(avg_duration),
            "by_category": by_category,
        }

    def save_results(self, filename: str = "") -> Path:
        """Save results to JSON."""
        if not filename:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.results_dir / filename
        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self._results],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _count_test_failures(command: str) -> int:
        """Run a test command and count failures."""
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse pytest output for failure count
            output = result.stdout + result.stderr
            for line in output.split("\n"):
                if "failed" in line and "passed" in line:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == "failed":
                            try:
                                return int(parts[i - 1])
                            except (ValueError, IndexError):
                                pass
            return 0
        except Exception:
            return -1
