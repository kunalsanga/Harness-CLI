"""
Benchmark types — task definitions, results, and scoring.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Optional


class BenchmarkCategory(enum.Enum):
    """Benchmark task categories."""

    TOOL_USE = "tool_use"
    REPOSITORY_NAVIGATION = "repository_navigation"
    CODE_GENERATION = "code_generation"
    BUG_FIXING = "bug_fixing"
    DEBUGGING = "debugging"
    ERROR_RECOVERY = "error_recovery"
    CONTEXT_HANDLING = "context_handling"
    PLANNING = "planning"
    VERIFICATION = "verification"


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""

    name: str = ""
    category: BenchmarkCategory = BenchmarkCategory.TOOL_USE
    description: str = ""
    setup_files: dict[str, str] = field(default_factory=dict)  # path → content
    instructions: str = ""
    expected_files: list[str] = field(default_factory=list)  # files that should exist/be modified
    expected_tests_pass: bool = False
    timeout_seconds: int = 120
    max_iterations: int = 15
    scoring_criteria: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""

    task_name: str = ""
    category: str = ""
    model_id: str = ""
    provider: str = ""

    # Outcome
    success: bool = False
    score: float = 0.0

    # Capability scores
    coding_score: Optional[float] = None
    tool_use_score: Optional[float] = None
    navigation_score: Optional[float] = None
    recovery_score: Optional[float] = None
    context_score: Optional[float] = None
    verification_score: Optional[float] = None
    planning_score: Optional[float] = None

    # Operational metrics
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_used: int = 0
    tool_calls: int = 0
    iterations: int = 0
    files_modified: int = 0

    # Metadata
    benchmark_version: str = "1.0"
    timestamp: float = field(default_factory=time.time)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "category": self.category,
            "model_id": self.model_id,
            "success": self.success,
            "score": round(self.score, 3),
            "coding_score": self.coding_score,
            "tool_use_score": self.tool_use_score,
            "navigation_score": self.navigation_score,
            "recovery_score": self.recovery_score,
            "context_score": self.context_score,
            "verification_score": self.verification_score,
            "planning_score": self.planning_score,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "files_modified": self.files_modified,
            "benchmark_version": self.benchmark_version,
            "error": self.error,
        }


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from a benchmark suite run."""

    model_id: str = ""
    provider: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)

    # Aggregate scores
    overall_score: float = 0.0
    coding_avg: Optional[float] = None
    tool_use_avg: Optional[float] = None
    navigation_avg: Optional[float] = None
    recovery_avg: Optional[float] = None
    context_avg: Optional[float] = None
    verification_avg: Optional[float] = None
    planning_avg: Optional[float] = None

    # Aggregate metrics
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    avg_tool_calls: float = 0.0
    success_rate: float = 0.0
    total_tasks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "overall_score": round(self.overall_score, 3),
            "coding_avg": round(self.coding_avg, 3) if self.coding_avg is not None else None,
            "tool_use_avg": round(self.tool_use_avg, 3) if self.tool_use_avg is not None else None,
            "navigation_avg": round(self.navigation_avg, 3) if self.navigation_avg is not None else None,
            "recovery_avg": round(self.recovery_avg, 3) if self.recovery_avg is not None else None,
            "context_avg": round(self.context_avg, 3) if self.context_avg is not None else None,
            "verification_avg": round(self.verification_avg, 3) if self.verification_avg is not None else None,
            "planning_avg": round(self.planning_avg, 3) if self.planning_avg is not None else None,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens": round(self.avg_tokens, 0),
            "avg_tool_calls": round(self.avg_tool_calls, 1),
            "success_rate": round(self.success_rate, 3),
            "total_tasks": self.total_tasks,
        }
