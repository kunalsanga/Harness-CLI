"""
SQLite-backed session storage for M4.

Thread-safe, transactional, crash-safe persistence for:
  - sessions
  - runs
  - checkpoints
  - memory items
  - session events

Uses explicit schemas. No arbitrary deserialization.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .domain import (
    Checkpoint,
    MemoryItem,
    MemoryType,
    Run,
    RunStatus,
    Session,
    SessionEvent,
    SessionStatus,
)


class SessionStorage:
    """SQLite-backed session persistence.

    Thread-safe. One writer per session via SQLite locking.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".harness" / "sessions.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    workspace_path TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_workspace
                ON sessions(workspace_path);
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status);
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at);

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    task TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    model_id TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT '',
                    verification_passed INTEGER NOT NULL DEFAULT 0,
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL DEFAULT 0,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    iterations INTEGER NOT NULL DEFAULT 0,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    failed_tool_calls INTEGER NOT NULL DEFAULT 0,
                    recovery_attempts INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost REAL NOT NULL DEFAULT 0,
                    result_summary TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    fallback_model TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_runs_session
                ON runs(session_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status
                ON runs(status);

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    git_head TEXT NOT NULL DEFAULT '',
                    git_branch TEXT NOT NULL DEFAULT '',
                    git_dirty INTEGER NOT NULL DEFAULT 0,
                    changed_files TEXT NOT NULL DEFAULT '[]',
                    current_iteration INTEGER NOT NULL DEFAULT 0,
                    current_task TEXT NOT NULL DEFAULT '',
                    model_id TEXT NOT NULL DEFAULT '',
                    tool_history_summary TEXT NOT NULL DEFAULT '',
                    verification_status TEXT NOT NULL DEFAULT '',
                    tests_passing INTEGER NOT NULL DEFAULT 0,
                    tests_total INTEGER NOT NULL DEFAULT 0,
                    context_files TEXT NOT NULL DEFAULT '[]',
                    context_file_hashes TEXT NOT NULL DEFAULT '{}',
                    repository_summary TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_session
                ON checkpoints(session_id);

                CREATE TABLE IF NOT EXISTS memory (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'note',
                    content TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    source_run_id TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_session
                ON memory(session_id);
                CREATE INDEX IF NOT EXISTS idx_memory_type
                ON memory(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memory_importance
                ON memory(importance);

                CREATE TABLE IF NOT EXISTS session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_session
                ON session_events(session_id);
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON session_events(event_type);
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=10)

    # ── Session CRUD ─────────────────────────────────────────────────────

    def create_session(self, session: Session) -> Session:
        """Persist a new session."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO sessions
                    (session_id, workspace_path, title, status, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session.session_id,
                        session.workspace_path,
                        session.title,
                        session.status.value,
                        session.created_at,
                        session.updated_at,
                        json.dumps(session.metadata),
                    ),
                )
                conn.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return Session(
                    session_id=row[0],
                    workspace_path=row[1],
                    title=row[2],
                    status=SessionStatus(row[3]),
                    created_at=row[4],
                    updated_at=row[5],
                    metadata=json.loads(row[6]),
                )

    def update_session(self, session: Session) -> None:
        """Update an existing session."""
        session.updated_at = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE sessions
                    SET title = ?, status = ?, updated_at = ?, metadata = ?
                    WHERE session_id = ?""",
                    (
                        session.title,
                        session.status.value,
                        session.updated_at,
                        json.dumps(session.metadata),
                        session.session_id,
                    ),
                )
                conn.commit()

    def list_sessions(
        self,
        workspace_path: str | None = None,
        status: SessionStatus | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions with optional filters."""
        with self._lock:
            with self._get_conn() as conn:
                query = "SELECT * FROM sessions WHERE 1=1"
                params: list[Any] = []

                if workspace_path:
                    query += " AND workspace_path = ?"
                    params.append(workspace_path)
                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                query += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                return [
                    Session(
                        session_id=r[0],
                        workspace_path=r[1],
                        title=r[2],
                        status=SessionStatus(r[3]),
                        created_at=r[4],
                        updated_at=r[5],
                        metadata=json.loads(r[6]),
                    )
                    for r in cursor.fetchall()
                ]

    def delete_session(self, session_id: str) -> int:
        """Delete a session and all its data. Returns count of deleted rows."""
        with self._lock:
            with self._get_conn() as conn:
                # Delete in order (foreign keys)
                conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM memory WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM runs WHERE session_id = ?", (session_id,))
                cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount

    # ── Run CRUD ─────────────────────────────────────────────────────────

    def create_run(self, run: Run) -> Run:
        """Persist a new run."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO runs
                    (run_id, session_id, task, status, model_id, provider, task_type,
                     outcome, verification_passed, started_at, completed_at, duration_ms,
                     iterations, tool_calls, failed_tool_calls, recovery_attempts,
                     input_tokens, output_tokens, total_tokens, estimated_cost,
                     result_summary, error_message, fallback_used, fallback_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.run_id, run.session_id, run.task, run.status.value,
                        run.model_id, run.provider, run.task_type,
                        run.outcome, 1 if run.verification_passed else 0,
                        run.started_at, run.completed_at, run.duration_ms,
                        run.iterations, run.tool_calls, run.failed_tool_calls,
                        run.recovery_attempts,
                        run.input_tokens, run.output_tokens, run.total_tokens,
                        run.estimated_cost,
                        run.result_summary, run.error_message,
                        1 if run.fallback_used else 0, run.fallback_model,
                    ),
                )
                # Update session last_active
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (time.time(), run.session_id),
                )
                conn.commit()
        return run

    def update_run(self, run: Run) -> None:
        """Update an existing run."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE runs SET
                    status = ?, outcome = ?, verification_passed = ?,
                    completed_at = ?, duration_ms = ?,
                    iterations = ?, tool_calls = ?, failed_tool_calls = ?,
                    recovery_attempts = ?,
                    input_tokens = ?, output_tokens = ?, total_tokens = ?,
                    estimated_cost = ?, result_summary = ?, error_message = ?,
                    fallback_used = ?, fallback_model = ?
                    WHERE run_id = ?""",
                    (
                        run.status.value, run.outcome,
                        1 if run.verification_passed else 0,
                        run.completed_at, run.duration_ms,
                        run.iterations, run.tool_calls, run.failed_tool_calls,
                        run.recovery_attempts,
                        run.input_tokens, run.output_tokens, run.total_tokens,
                        run.estimated_cost, run.result_summary, run.error_message,
                        1 if run.fallback_used else 0, run.fallback_model,
                        run.run_id,
                    ),
                )
                conn.commit()

    def get_runs(self, session_id: str) -> list[Run]:
        """Get all runs for a session."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE session_id = ? ORDER BY started_at",
                    (session_id,),
                )
                return [self._row_to_run(r) for r in cursor.fetchall()]

    def get_run(self, run_id: str) -> Run | None:
        """Get a single run by ID."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
                row = cursor.fetchone()
                return self._row_to_run(row) if row else None

    # ── Checkpoint CRUD ──────────────────────────────────────────────────

    def create_checkpoint(self, cp: Checkpoint) -> Checkpoint:
        """Persist a checkpoint."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO checkpoints
                    (checkpoint_id, session_id, run_id, created_at,
                     git_head, git_branch, git_dirty, changed_files,
                     current_iteration, current_task, model_id, tool_history_summary,
                     verification_status, tests_passing, tests_total,
                     context_files, context_file_hashes, repository_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cp.checkpoint_id, cp.session_id, cp.run_id, cp.created_at,
                        cp.git_head, cp.git_branch, 1 if cp.git_dirty else 0,
                        json.dumps(cp.changed_files),
                        cp.current_iteration, cp.current_task, cp.model_id,
                        cp.tool_history_summary,
                        cp.verification_status, cp.tests_passing, cp.tests_total,
                        json.dumps(cp.context_files),
                        json.dumps(cp.context_file_hashes),
                        cp.repository_summary,
                    ),
                )
                conn.commit()
        return cp

    def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a session."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """SELECT * FROM checkpoints WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT 1""",
                    (session_id,),
                )
                row = cursor.fetchone()
                return self._row_to_checkpoint(row) if row else None

    def get_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a session."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at",
                    (session_id,),
                )
                return [self._row_to_checkpoint(r) for r in cursor.fetchall()]

    # ── Memory CRUD ──────────────────────────────────────────────────────

    def add_memory(self, item: MemoryItem) -> MemoryItem:
        """Persist a memory item."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO memory
                    (memory_id, session_id, memory_type, content, importance,
                     created_at, updated_at, source_run_id, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item.memory_id, item.session_id,
                        item.memory_type.value, item.content, item.importance,
                        item.created_at, item.updated_at,
                        item.source_run_id, json.dumps(item.tags),
                    ),
                )
                conn.commit()
        return item

    def get_memories(
        self,
        session_id: str,
        memory_type: MemoryType | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Get memory items for a session."""
        with self._lock:
            with self._get_conn() as conn:
                query = "SELECT * FROM memory WHERE session_id = ? AND importance >= ?"
                params: list[Any] = [session_id, min_importance]

                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type.value)

                query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                return [self._row_to_memory(r) for r in cursor.fetchall()]

    def search_memories(
        self,
        session_id: str,
        keyword: str,
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Simple keyword search across memory content."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """SELECT * FROM memory
                    WHERE session_id = ? AND content LIKE ?
                    ORDER BY importance DESC LIMIT ?""",
                    (session_id, f"%{keyword}%", limit),
                )
                return [self._row_to_memory(r) for r in cursor.fetchall()]

    def update_memory(self, item: MemoryItem) -> None:
        """Update a memory item."""
        item.updated_at = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE memory SET
                    content = ?, importance = ?, updated_at = ?, tags = ?
                    WHERE memory_id = ?""",
                    (
                        item.content, item.importance, item.updated_at,
                        json.dumps(item.tags), item.memory_id,
                    ),
                )
                conn.commit()

    def delete_memory(self, memory_id: str) -> int:
        """Delete a memory item."""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("DELETE FROM memory WHERE memory_id = ?", (memory_id,))
                conn.commit()
                return cursor.rowcount

    # ── Events ───────────────────────────────────────────────────────────

    def record_event(self, event: SessionEvent) -> SessionEvent:
        """Persist a session event."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO session_events
                    (event_id, session_id, event_type, timestamp, data)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        event.event_id, event.session_id,
                        event.event_type, event.timestamp,
                        json.dumps(event.data),
                    ),
                )
                conn.commit()
        return event

    def get_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SessionEvent]:
        """Get events for a session."""
        with self._lock:
            with self._get_conn() as conn:
                if event_type:
                    cursor = conn.execute(
                        """SELECT * FROM session_events
                        WHERE session_id = ? AND event_type = ?
                        ORDER BY timestamp DESC LIMIT ?""",
                        (session_id, event_type, limit),
                    )
                else:
                    cursor = conn.execute(
                        """SELECT * FROM session_events
                        WHERE session_id = ?
                        ORDER BY timestamp DESC LIMIT ?""",
                        (session_id, limit),
                    )
                return [self._row_to_event(r) for r in cursor.fetchall()]

    # ── Counts ───────────────────────────────────────────────────────────

    def count_sessions(self) -> int:
        with self._lock:
            with self._get_conn() as conn:
                return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def count_runs(self, session_id: str | None = None) -> int:
        with self._lock:
            with self._get_conn() as conn:
                if session_id:
                    return conn.execute(
                        "SELECT COUNT(*) FROM runs WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()[0]
                return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    # ── Row converters ───────────────────────────────────────────────────

    def _row_to_run(self, row: tuple) -> Run:
        return Run(
            run_id=row[0], session_id=row[1], task=row[2],
            status=RunStatus(row[3]), model_id=row[4], provider=row[5],
            task_type=row[6], outcome=row[7],
            verification_passed=bool(row[8]),
            started_at=row[9], completed_at=row[10], duration_ms=row[11],
            iterations=row[12], tool_calls=row[13], failed_tool_calls=row[14],
            recovery_attempts=row[15],
            input_tokens=row[16], output_tokens=row[17], total_tokens=row[18],
            estimated_cost=row[19],
            result_summary=row[20], error_message=row[21],
            fallback_used=bool(row[22]), fallback_model=row[23],
        )

    def _row_to_checkpoint(self, row: tuple) -> Checkpoint:
        return Checkpoint(
            checkpoint_id=row[0], session_id=row[1], run_id=row[2],
            created_at=row[3],
            git_head=row[4], git_branch=row[5],
            git_dirty=bool(row[6]),
            changed_files=json.loads(row[7]),
            current_iteration=row[8], current_task=row[9],
            model_id=row[10], tool_history_summary=row[11],
            verification_status=row[12],
            tests_passing=row[13], tests_total=row[14],
            context_files=json.loads(row[15]),
            context_file_hashes=json.loads(row[16]),
            repository_summary=row[17],
        )

    def _row_to_memory(self, row: tuple) -> MemoryItem:
        return MemoryItem(
            memory_id=row[0], session_id=row[1],
            memory_type=MemoryType(row[2]),
            content=row[3], importance=row[4],
            created_at=row[5], updated_at=row[6],
            source_run_id=row[7],
            tags=json.loads(row[8]),
        )

    def _row_to_event(self, row: tuple) -> SessionEvent:
        return SessionEvent(
            event_id=row[0], session_id=row[1],
            event_type=row[2], timestamp=row[3],
            data=json.loads(row[4]),
        )
