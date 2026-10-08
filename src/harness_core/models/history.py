"""
Model performance history — SQLite-backed persistent tracking.

Records model performance across tasks with time-decay support.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PerformanceRecord:
    """A single performance observation."""

    model_id: str = ""
    provider: str = ""
    task_type: str = ""
    benchmark_name: str = ""
    success: bool = False
    failure_reason: str = ""
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_used: int = 0
    tool_calls: int = 0
    iterations: int = 0
    recovered: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelPerformance:
    """Aggregated performance for a model."""

    model_id: str = ""
    total_tasks: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_tokens: float = 0.0
    avg_tool_calls: float = 0.0
    avg_iterations: float = 0.0
    recovery_rate: float = 0.0
    last_task_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "total_tasks": self.total_tasks,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_ttft_ms": round(self.avg_ttft_ms, 1),
            "avg_tokens": round(self.avg_tokens, 0),
            "avg_tool_calls": round(self.avg_tool_calls, 1),
            "avg_iterations": round(self.avg_iterations, 1),
            "recovery_rate": round(self.recovery_rate, 3),
            "last_task_time": self.last_task_time,
        }


class PerformanceHistory:
    """SQLite-backed model performance history.

    Thread-safe. Supports time decay for older observations.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".harness" / "performance.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    benchmark_name TEXT NOT NULL DEFAULT '',
                    success INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    latency_ms REAL NOT NULL DEFAULT 0,
                    ttft_ms REAL NOT NULL DEFAULT 0,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    iterations INTEGER NOT NULL DEFAULT 0,
                    recovered INTEGER NOT NULL DEFAULT 0,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON performance(model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON performance(timestamp)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self._db_path), timeout=5)

    def record(self, record: PerformanceRecord) -> None:
        """Record a performance observation."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO performance
                    (model_id, provider, task_type, benchmark_name,
                     success, failure_reason, latency_ms, ttft_ms,
                     tokens_used, tool_calls, iterations, recovered, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.model_id,
                        record.provider,
                        record.task_type,
                        record.benchmark_name,
                        1 if record.success else 0,
                        record.failure_reason,
                        record.latency_ms,
                        record.ttft_ms,
                        record.tokens_used,
                        record.tool_calls,
                        record.iterations,
                        1 if record.recovered else 0,
                        record.timestamp,
                    ),
                )
                conn.commit()

    def get_performance(
        self,
        model_id: str,
        decay_half_life_days: float = 30.0,
    ) -> ModelPerformance:
        """Get aggregated performance for a model with time decay."""
        now = time.time()
        decay_rate = 0.693 / (decay_half_life_days * 86400)  # ln(2) / half_life

        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """SELECT success, latency_ms, ttft_ms, tokens_used,
                              tool_calls, iterations, recovered, timestamp
                       FROM performance
                       WHERE model_id = ?""",
                    (model_id,),
                )
                rows = cursor.fetchall()

        if not rows:
            return ModelPerformance(model_id=model_id)

        total = 0
        weighted_success = 0.0
        weighted_latency = 0.0
        weighted_ttft = 0.0
        weighted_tokens = 0.0
        weighted_tool_calls = 0.0
        weighted_iterations = 0.0
        weighted_recovered = 0.0
        total_weight = 0.0
        last_time = 0.0

        for row in rows:
            success, latency, ttft, tokens, tool_calls, iters, recovered, ts = row
            age = now - ts
            weight = pow(2, -decay_rate * age)

            total += 1
            total_weight += weight
            weighted_success += weight * success
            weighted_latency += weight * latency
            weighted_ttft += weight * ttft
            weighted_tokens += weight * tokens
            weighted_tool_calls += weight * tool_calls
            weighted_iterations += weight * iters
            weighted_recovered += weight * recovered
            last_time = max(last_time, ts)

        if total_weight == 0:
            return ModelPerformance(model_id=model_id)

        return ModelPerformance(
            model_id=model_id,
            total_tasks=total,
            success_count=int(weighted_success / total_weight + 0.5),
            failure_count=total - int(weighted_success / total_weight + 0.5),
            success_rate=weighted_success / total_weight,
            avg_latency_ms=weighted_latency / total_weight,
            avg_ttft_ms=weighted_ttft / total_weight,
            avg_tokens=weighted_tokens / total_weight,
            avg_tool_calls=weighted_tool_calls / total_weight,
            avg_iterations=weighted_iterations / total_weight,
            recovery_rate=weighted_recovered / total_weight,
            last_task_time=last_time,
        )

    def get_all_performance(
        self,
        decay_half_life_days: float = 30.0,
    ) -> list[ModelPerformance]:
        """Get performance for all tracked models."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT model_id FROM performance"
                )
                model_ids = [row[0] for row in cursor.fetchall()]

        return [
            self.get_performance(mid, decay_half_life_days)
            for mid in model_ids
        ]

    def count(self, model_id: str | None = None) -> int:
        """Count records, optionally for a specific model."""
        with self._lock:
            with self._get_conn() as conn:
                if model_id:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM performance WHERE model_id = ?",
                        (model_id,),
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM performance")
                return cursor.fetchone()[0]

    def clear(self, model_id: str | None = None) -> int:
        """Clear records. Returns number deleted."""
        with self._lock:
            with self._get_conn() as conn:
                if model_id:
                    cursor = conn.execute(
                        "DELETE FROM performance WHERE model_id = ?",
                        (model_id,),
                    )
                else:
                    cursor = conn.execute("DELETE FROM performance")
                conn.commit()
                return cursor.rowcount
