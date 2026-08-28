"""Session management — persistent agent memory for M4.

Provides:
  - Domain model (Session, Run, Checkpoint, MemoryItem, SessionEvent)
  - SQLite storage (SessionStorage)
  - Legacy compatibility (SessionState, RunRecord from session.py)
"""

from harness_core.session.domain import (
    Checkpoint,
    MemoryItem,
    MemoryType,
    Run,
    RunStatus,
    Session,
    SessionEvent,
    SessionStatus,
    can_transition,
)
from harness_core.session.storage import SessionStorage

__all__ = [
    "Session",
    "SessionStatus",
    "Run",
    "RunStatus",
    "Checkpoint",
    "MemoryItem",
    "MemoryType",
    "SessionEvent",
    "SessionStorage",
    "can_transition",
]
