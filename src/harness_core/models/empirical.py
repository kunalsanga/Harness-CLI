"""
Empirical model intelligence — the core of M3.8.

Tracks real execution outcomes, aggregates per-task performance,
provides confidence scores, and feeds into routing decisions.

Separates:
  - BENCHMARK data (controlled experiments)
  - REAL_WORLD data (actual agent executions)

Distinguishes:
  - LONG-TERM performance (all history)
  - RECENT performance (last N tasks)

Each capability score has provenance:
  - PROVIDER_DECLARED
  - HARNESS_STATIC
  - HARNESS_BENCHMARKED
  - REAL_WORLD_OBSERVED
"""

from __future__ import annotations

import enum
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── Outcome taxonomy ──────────────────────────────────────────────────────

class TaskOutcome(enum.Enum):
    """Clear outcome classification — not just success/failure."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    PERMISSION_DENIED = "permission_denied"
    USER_ABORTED = "user_aborted"
    UNKNOWN = "unknown"


# ── Evidence provenance ──────────────────────────────────────────────────

class EvidenceSource(enum.Enum):
    """Where capability data came from."""

    PROVIDER_DECLARED = "provider_declared"
    HARNESS_STATIC = "harness_static"
    HARNESS_BENCHMARKED = "harness_benchmarked"
    REAL_WORLD_OBSERVED = "real_world_observed"


# ── Sample confidence ────────────────────────────────────────────────────

class SampleConfidence(enum.Enum):
    """Confidence based on sample size."""

    UNKNOWN = "unknown"       # 0 samples
    VERY_LOW = "very_low"    # 1-4 samples
    LOW = "low"              # 5-19 samples
    MEDIUM = "medium"        # 20-49 samples
    HIGH = "high"            # 50+ samples


# ── Execution record ─────────────────────────────────────────────────────

@dataclass
class ModelExecutionRecord:
    """A single real execution result — the atomic unit of empirical evidence."""

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    provider: str = ""

    # Task context
    task_id: str = ""
    task_type: str = ""
    task_description: str = ""

    # Outcome
    outcome: TaskOutcome = TaskOutcome.UNKNOWN
    verification_passed: bool = False

    # Timing
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0
    ttft_ms: float = 0.0

    # Resource usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    # Execution details
    tool_calls: int = 0
    failed_tool_calls: int = 0
    iterations: int = 0
    recovery_attempts: int = 0

    # Context
    context_tokens: int = 0
    context_files: int = 0

    # Fallback
    fallback_used: bool = False
    fallback_model: str = ""

    # Error info (sanitized — no secrets)
    error_type: str = ""
    error_message: str = ""

    # Provenance
    source: EvidenceSource = EvidenceSource.REAL_WORLD_OBSERVED
    benchmark_version: str = ""
    benchmark_name: str = ""

    # Metadata
    timestamp: float = field(default_factory=time.time)


# ── Task-specific performance ────────────────────────────────────────────

@dataclass
class TaskPerformance:
    """Performance metrics for a specific task type."""

    task_type: str = ""
    total_tasks: int = 0
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0

    success_rate: float = 0.0
    verification_rate: float = 0.0

    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    avg_ttft_ms: float = 0.0
    avg_iterations: float = 0.0
    avg_tool_calls: float = 0.0
    avg_failed_tool_calls: float = 0.0
    avg_recovery_attempts: float = 0.0

    avg_tokens: float = 0.0
    avg_cost: float = 0.0

    fallback_rate: float = 0.0

    # Confidence
    sample_confidence: SampleConfidence = SampleConfidence.UNKNOWN

    # Recent performance (last N tasks)
    recent_tasks: int = 0
    recent_success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "total_tasks": self.total_tasks,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 3),
            "verification_rate": round(self.verification_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "median_latency_ms": round(self.median_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "avg_iterations": round(self.avg_iterations, 1),
            "avg_tool_calls": round(self.avg_tool_calls, 1),
            "sample_confidence": self.sample_confidence.value,
            "recent_tasks": self.recent_tasks,
            "recent_success_rate": round(self.recent_success_rate, 3),
        }


# ── Model empirical profile ──────────────────────────────────────────────

@dataclass
class ModelEmpiricalProfile:
    """Complete empirical profile for a model — all task-specific performance."""

    model_id: str = ""

    # Overall
    overall: TaskPerformance = field(default_factory=TaskPerformance)

    # Per task type
    by_task_type: dict[str, TaskPerformance] = field(default_factory=dict)

    # Evidence provenance
    primary_source: EvidenceSource = EvidenceSource.PROVIDER_DECLARED
    last_observed: float = 0.0
    total_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "overall": self.overall.to_dict(),
            "by_task_type": {k: v.to_dict() for k, v in self.by_task_type.items()},
            "primary_source": self.primary_source.value,
            "last_observed": self.last_observed,
            "total_records": self.total_records,
        }


# ── Confidence calculator ────────────────────────────────────────────────

class ConfidenceCalculator:
    """Calculate sample-based confidence."""

    # Configurable thresholds
    THRESHOLDS = {
        SampleConfidence.UNKNOWN: (0, 0),
        SampleConfidence.VERY_LOW: (1, 4),
        SampleConfidence.LOW: (5, 19),
        SampleConfidence.MEDIUM: (20, 49),
        SampleConfidence.HIGH: (50, float("inf")),
    }

    @classmethod
    def from_sample_count(cls, count: int) -> SampleConfidence:
        """Get confidence level from sample count."""
        if count <= 0:
            return SampleConfidence.UNKNOWN
        if count <= 4:
            return SampleConfidence.VERY_LOW
        if count <= 19:
            return SampleConfidence.LOW
        if count <= 49:
            return SampleConfidence.MEDIUM
        return SampleConfidence.HIGH

    @classmethod
    def confidence_weight(cls, confidence: SampleConfidence) -> float:
        """Get a numeric weight for routing (0.0-1.0).

        UNKNOWN = 0.0 (no influence)
        VERY_LOW = 0.1
        LOW = 0.3
        MEDIUM = 0.6
        HIGH = 1.0
        """
        weights = {
            SampleConfidence.UNKNOWN: 0.0,
            SampleConfidence.VERY_LOW: 0.1,
            SampleConfidence.LOW: 0.3,
            SampleConfidence.MEDIUM: 0.6,
            SampleConfidence.HIGH: 1.0,
        }
        return weights.get(confidence, 0.0)


# ── Aggregation engine ──────────────────────────────────────────────────

class ModelPerformanceAggregator:
    """Calculate aggregated metrics from execution records.

    Supports:
    - Per-task-type segmentation
    - Long-term vs recent performance
    - Sample confidence
    - Time-decay weighting
    """

    def __init__(
        self,
        decay_half_life_days: float = 30.0,
        recent_window: int = 20,
    ) -> None:
        self.decay_half_life_days = decay_life_days = decay_half_life_days
        self.decay_rate = 0.693 / (decay_half_life_days * 86400)
        self.recent_window = recent_window

    def _time_weight(self, timestamp: float) -> float:
        """Calculate time-decay weight for a record."""
        age = time.time() - timestamp
        return math.exp(-self.decay_rate * age)

    def _percentile(self, values: list[float], p: float) -> float:
        """Calculate p-th percentile."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def aggregate_task(
        self,
        records: list[ModelExecutionRecord],
        task_type: str = "",
    ) -> TaskPerformance:
        """Aggregate records into TaskPerformance for a specific task type.

        If task_type is empty, aggregates all records.
        """
        if task_type:
            filtered = [r for r in records if r.task_type == task_type]
        else:
            filtered = list(records)

        if not filtered:
            return TaskPerformance(task_type=task_type)

        # Sort by timestamp
        filtered.sort(key=lambda r: r.timestamp)

        total = len(filtered)
        successes = sum(1 for r in filtered if r.outcome == TaskOutcome.SUCCESS)
        failures = sum(1 for r in filtered if r.outcome in (
            TaskOutcome.FAILURE, TaskOutcome.TIMEOUT, TaskOutcome.MODEL_ERROR,
            TaskOutcome.TOOL_ERROR, TaskOutcome.PERMISSION_DENIED,
        ))
        partials = sum(1 for r in filtered if r.outcome == TaskOutcome.PARTIAL_SUCCESS)
        verifications = sum(1 for r in filtered if r.verification_passed)

        # Time-weighted averages
        total_weight = 0.0
        w_success = 0.0
        w_verification = 0.0
        w_latency = 0.0
        w_ttft = 0.0
        w_iterations = 0.0
        w_tool_calls = 0.0
        w_failed_tool_calls = 0.0
        w_recovery = 0.0
        w_tokens = 0.0
        w_cost = 0.0
        w_fallback = 0.0

        latencies = []

        for r in filtered:
            w = self._time_weight(r.timestamp)
            total_weight += w
            w_success += w * (1 if r.outcome == TaskOutcome.SUCCESS else 0)
            w_verification += w * (1 if r.verification_passed else 0)
            w_latency += w * r.duration_ms
            w_ttft += w * r.ttft_ms
            w_iterations += w * r.iterations
            w_tool_calls += w * r.tool_calls
            w_failed_tool_calls += w * r.failed_tool_calls
            w_recovery += w * r.recovery_attempts
            w_tokens += w * r.total_tokens
            w_cost += w * r.estimated_cost
            w_fallback += w * (1 if r.fallback_used else 0)
            if r.duration_ms > 0:
                latencies.append(r.duration_ms)

        if total_weight == 0:
            return TaskPerformance(task_type=task_type)

        # Recent performance (last N tasks)
        recent = filtered[-self.recent_window:]
        recent_successes = sum(1 for r in recent if r.outcome == TaskOutcome.SUCCESS)
        recent_success_rate = recent_successes / len(recent) if recent else 0.0

        return TaskPerformance(
            task_type=task_type,
            total_tasks=total,
            success_count=successes,
            failure_count=failures,
            partial_count=partials,
            success_rate=w_success / total_weight,
            verification_rate=w_verification / total_weight,
            avg_latency_ms=w_latency / total_weight,
            median_latency_ms=self._percentile(latencies, 50) if latencies else 0.0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0.0,
            avg_ttft_ms=w_ttft / total_weight,
            avg_iterations=w_iterations / total_weight,
            avg_tool_calls=w_tool_calls / total_weight,
            avg_failed_tool_calls=w_failed_tool_calls / total_weight,
            avg_recovery_attempts=w_recovery / total_weight,
            avg_tokens=w_tokens / total_weight,
            avg_cost=w_cost / total_weight,
            fallback_rate=w_fallback / total_weight,
            sample_confidence=ConfidenceCalculator.from_sample_count(total),
            recent_tasks=len(recent),
            recent_success_rate=recent_success_rate,
        )

    def build_profile(
        self,
        records: list[ModelExecutionRecord],
    ) -> ModelEmpiricalProfile:
        """Build a complete empirical profile from all records for a model."""
        if not records:
            return ModelEmpiricalProfile()

        model_id = records[0].model_id
        provider = records[0].provider

        # Overall
        overall = self.aggregate_task(records, "")

        # Per task type
        task_types = set(r.task_type for r in records if r.task_type)
        by_task = {}
        for tt in sorted(task_types):
            by_task[tt] = self.aggregate_task(records, tt)

        # Primary source
        sources = [r.source for r in records]
        if EvidenceSource.REAL_WORLD_OBSERVED in sources:
            primary = EvidenceSource.REAL_WORLD_OBSERVED
        elif EvidenceSource.HARNESS_BENCHMARKED in sources:
            primary = EvidenceSource.HARNESS_BENCHMARKED
        else:
            primary = EvidenceSource.PROVIDER_DECLARED

        last_observed = max(r.timestamp for r in records)

        return ModelEmpiricalProfile(
            model_id=model_id,
            overall=overall,
            by_task_type=by_task,
            primary_source=primary,
            last_observed=last_observed,
            total_records=len(records),
        )


# ── Empirical history storage ───────────────────────────────────────────

class EmpiricalHistory:
    """SQLite-backed empirical execution history.

    Thread-safe. Stores execution records and supports
    fast queries for aggregation and routing.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".harness" / "empirical.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        self._cache: dict[str, ModelEmpiricalProfile] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 60.0  # 1 minute

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    record_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    task_description TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT 'unknown',
                    verification_passed INTEGER NOT NULL DEFAULT 0,
                    started_at REAL NOT NULL DEFAULT 0,
                    completed_at REAL NOT NULL DEFAULT 0,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    ttft_ms REAL NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost REAL NOT NULL DEFAULT 0,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    failed_tool_calls INTEGER NOT NULL DEFAULT 0,
                    iterations INTEGER NOT NULL DEFAULT 0,
                    recovery_attempts INTEGER NOT NULL DEFAULT 0,
                    context_tokens INTEGER NOT NULL DEFAULT 0,
                    context_files INTEGER NOT NULL DEFAULT 0,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    fallback_model TEXT NOT NULL DEFAULT '',
                    error_type TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'real_world_observed',
                    benchmark_version TEXT NOT NULL DEFAULT '',
                    benchmark_name TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emp_model
                ON executions(model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emp_task_type
                ON executions(task_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emp_timestamp
                ON executions(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emp_outcome
                ON executions(outcome)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emp_model_task
                ON executions(model_id, task_type)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=5)

    def record(self, record: ModelExecutionRecord) -> None:
        """Record an execution result."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO executions
                    (record_id, model_id, provider, task_id, task_type,
                     task_description, outcome, verification_passed,
                     started_at, completed_at, duration_ms, ttft_ms,
                     input_tokens, output_tokens, total_tokens, estimated_cost,
                     tool_calls, failed_tool_calls, iterations, recovery_attempts,
                     context_tokens, context_files,
                     fallback_used, fallback_model,
                     error_type, error_message,
                     source, benchmark_version, benchmark_name, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.record_id, record.model_id, record.provider,
                        record.task_id, record.task_type, record.task_description,
                        record.outcome.value, 1 if record.verification_passed else 0,
                        record.started_at, record.completed_at,
                        record.duration_ms, record.ttft_ms,
                        record.input_tokens, record.output_tokens,
                        record.total_tokens, record.estimated_cost,
                        record.tool_calls, record.failed_tool_calls,
                        record.iterations, record.recovery_attempts,
                        record.context_tokens, record.context_files,
                        1 if record.fallback_used else 0, record.fallback_model,
                        record.error_type, record.error_message,
                        record.source.value, record.benchmark_version,
                        record.benchmark_name, record.timestamp,
                    ),
                )
                conn.commit()
        # Invalidate cache
        self._cache = {}

    def get_records(
        self,
        model_id: str,
        task_type: str = "",
        limit: int = 1000,
    ) -> list[ModelExecutionRecord]:
        """Get execution records for a model, optionally filtered by task type."""
        with self._lock:
            with self._get_conn() as conn:
                if task_type:
                    cursor = conn.execute(
                        """SELECT * FROM executions
                        WHERE model_id = ? AND task_type = ?
                        ORDER BY timestamp DESC LIMIT ?""",
                        (model_id, task_type, limit),
                    )
                else:
                    cursor = conn.execute(
                        """SELECT * FROM executions
                        WHERE model_id = ?
                        ORDER BY timestamp DESC LIMIT ?""",
                        (model_id, limit),
                    )
                return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_profile(self, model_id: str) -> ModelEmpiricalProfile:
        """Get the complete empirical profile for a model."""
        # Check cache
        now = time.time()
        if model_id in self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache[model_id]

        records = self.get_records(model_id)
        aggregator = ModelPerformanceAggregator()
        profile = aggregator.build_profile(records)

        self._cache[model_id] = profile
        self._cache_time = now
        return profile

    def get_all_profiles(self) -> dict[str, ModelEmpiricalProfile]:
        """Get empirical profiles for all tracked models."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT model_id FROM executions"
                )
                model_ids = [row[0] for row in cursor.fetchall()]

        return {mid: self.get_profile(mid) for mid in model_ids}

    def count(self, model_id: str | None = None) -> int:
        """Count records."""
        with self._lock:
            with self._get_conn() as conn:
                if model_id:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM executions WHERE model_id = ?",
                        (model_id,),
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM executions")
                return cursor.fetchone()[0]

    def clear(self, model_id: str | None = None) -> int:
        """Clear records."""
        with self._lock:
            with self._get_conn() as conn:
                if model_id:
                    cursor = conn.execute(
                        "DELETE FROM executions WHERE model_id = ?",
                        (model_id,),
                    )
                else:
                    cursor = conn.execute("DELETE FROM executions")
                conn.commit()
                self._cache = {}
                return cursor.rowcount

    def _row_to_record(self, row: tuple) -> ModelExecutionRecord:
        """Convert a database row to a ModelExecutionRecord."""
        return ModelExecutionRecord(
            record_id=row[0],
            model_id=row[1],
            provider=row[2],
            task_id=row[3],
            task_type=row[4],
            task_description=row[5],
            outcome=TaskOutcome(row[6]),
            verification_passed=bool(row[7]),
            started_at=row[8],
            completed_at=row[9],
            duration_ms=row[10],
            ttft_ms=row[11],
            input_tokens=row[12],
            output_tokens=row[13],
            total_tokens=row[14],
            estimated_cost=row[15],
            tool_calls=row[16],
            failed_tool_calls=row[17],
            iterations=row[18],
            recovery_attempts=row[19],
            context_tokens=row[20],
            context_files=row[21],
            fallback_used=bool(row[22]),
            fallback_model=row[23],
            error_type=row[24],
            error_message=row[25],
            source=EvidenceSource(row[26]),
            benchmark_version=row[27],
            benchmark_name=row[28],
            timestamp=row[29],
        )
