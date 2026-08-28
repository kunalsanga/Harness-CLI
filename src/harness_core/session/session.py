"""
Session persistence — stores agent run history in SQLite.

Tracks: session ID, project path, tasks, runs, selected models,
tool calls, verification results, status, timestamps.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunRecord:
    """A single agent run within a session."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    goal: str = ""
    status: str = "pending"  # pending, running, completed, failed
    model_used: str = ""
    provider_used: str = ""
    task_type: str = ""
    iterations: int = 0
    tool_calls: int = 0
    result: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class SessionState:
    """Session state tracking."""

    def __init__(self, session_id: str, project_path: str = "") -> None:
        self.session_id = session_id
        self.project_path = project_path
        self.runs: list[RunRecord] = []
        self.created_at: float = time.time()
        self.last_active: float = time.time()

    def add_run(self, run: RunRecord) -> None:
        """Add a run to the session."""
        run.session_id = self.session_id
        self.runs.append(run)
        self.last_active = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_path": self.project_path,
            "runs": len(self.runs),
            "created_at": self.created_at,
            "last_active": self.last_active,
        }


class Session:
    """SQLite-backed session storage.

    Thread-safe. Stores session and run history for agent tasks.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".harness" / "sessions.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    project_path TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    goal TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    model_used TEXT NOT NULL DEFAULT '',
                    provider_used TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    iterations INTEGER NOT NULL DEFAULT 0,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    result TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_session
                ON runs(session_id)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=5)

    def create_session(self, project_path: str = "") -> str:
        """Create a new session. Returns session ID."""
        session_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, project_path, created_at, last_active) VALUES (?, ?, ?, ?)",
                    (session_id, project_path, now, now),
                )
                conn.commit()
        return session_id

    def record_run(self, run: RunRecord) -> None:
        """Record a run."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO runs
                    (run_id, session_id, goal, status, model_used, provider_used,
                     task_type, iterations, tool_calls, result, error,
                     started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.run_id, run.session_id, run.goal, run.status,
                        run.model_used, run.provider_used, run.task_type,
                        run.iterations, run.tool_calls, run.result, run.error,
                        run.started_at, run.completed_at,
                    ),
                )
                # Update session last_active
                conn.execute(
                    "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                    (time.time(), run.session_id),
                )
                conn.commit()

    def get_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent sessions."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT session_id, project_path, created_at, last_active FROM sessions ORDER BY last_active DESC LIMIT ?",
                    (limit,),
                )
                return [
                    {"session_id": r[0], "project_path": r[1],
                     "created_at": r[2], "last_active": r[3]}
                    for r in cursor.fetchall()
                ]

    def get_runs(self, session_id: str) -> list[RunRecord]:
        """Get all runs for a session."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at",
                    (session_id,),
                )
                return [
                    RunRecord(
                        run_id=r[0], session_id=r[1], goal=r[2], status=r[3],
                        model_used=r[4], provider_used=r[5], task_type=r[6],
                        iterations=r[7], tool_calls=r[8], result=r[9],
                        error=r[10], started_at=r[11], completed_at=r[12],
                    )
                    for r in cursor.fetchall()
                ]

    def count(self) -> int:
        """Count total sessions."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                return cursor.fetchone()[0]
