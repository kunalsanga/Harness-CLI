"""
Session lifecycle manager for M4.

Orchestrates session creation, run management, checkpointing,
memory management, and resume operations.

Uses SessionStorage for persistence and domain objects for state.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .domain import (
    Checkpoint,
    MemoryItem,
    MemoryType,
    ReconciliationReport,
    Run,
    RunStatus,
    Session,
    SessionEvent,
    SessionStatus,
)
from .storage import SessionStorage


class SessionManager:
    """High-level session lifecycle management.

    Handles:
    - Creating and titling sessions
    - Starting and completing runs
    - Creating checkpoints
    - Managing memory
    - Crash detection
    """

    def __init__(self, storage: SessionStorage | None = None) -> None:
        self.storage = storage or SessionStorage()

    # ── Session lifecycle ────────────────────────────────────────────────

    def create_session(
        self,
        workspace_path: str = "",
        title: str = "",
    ) -> Session:
        """Create a new session."""
        session = Session(
            workspace_path=workspace_path,
            title=title or f"Session {time.strftime('%Y-%m-%d %H:%M')}",
        )
        self.storage.create_session(session)
        self._emit_event(session.session_id, "session.created", {
            "workspace": workspace_path,
            "title": session.title,
        })
        return session

    def pause_session(self, session_id: str) -> Session:
        """Pause a session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.transition_to(SessionStatus.PAUSED)
        self.storage.update_session(session)
        self._emit_event(session_id, "session.paused", {})
        return session

    def resume_session(self, session_id: str) -> Session:
        """Resume a paused session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.transition_to(SessionStatus.ACTIVE)
        self.storage.update_session(session)
        self._emit_event(session_id, "session.resumed", {})
        return session

    def complete_session(self, session_id: str) -> Session:
        """Mark a session as completed."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.transition_to(SessionStatus.COMPLETED)
        self.storage.update_session(session)
        self._emit_event(session_id, "session.completed", {})
        return session

    def fail_session(self, session_id: str) -> Session:
        """Mark a session as failed."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.transition_to(SessionStatus.FAILED)
        self.storage.update_session(session)
        self._emit_event(session_id, "session.failed", {})
        return session

    def abort_session(self, session_id: str) -> Session:
        """Abort a session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.transition_to(SessionStatus.ABORTED)
        self.storage.update_session(session)
        self._emit_event(session_id, "session.aborted", {})
        return session

    def archive_session(self, session_id: str) -> Session:
        """Archive a session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        if session.status == SessionStatus.ACTIVE:
            session.transition_to(SessionStatus.COMPLETED)
        session.transition_to(SessionStatus.ARCHIVED)
        self.storage.update_session(session)
        self._emit_event(session_id, "session.archived", {})
        return session

    def rename_session(self, session_id: str, new_title: str) -> Session:
        """Rename an existing session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.title = new_title.strip()
        self.storage.update_session(session)
        self._emit_event(session_id, "session.renamed", {"title": session.title})
        return session

    def get_latest_session(self, workspace_path: str | None = None) -> Session | None:
        """Get the most recently updated session, optionally scoped to a workspace."""
        sessions = self.storage.list_sessions(workspace_path=workspace_path, limit=1)
        return sessions[0] if sessions else None

    # ── Run management ───────────────────────────────────────────────────

    def start_run(
        self,
        session_id: str,
        task: str,
        model_id: str = "",
        provider: str = "",
        task_type: str = "",
    ) -> Run:
        """Start a new run in a session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        run = Run(
            session_id=session_id,
            task=task,
            model_id=model_id,
            provider=provider,
            task_type=task_type,
        )
        run.transition_to(RunStatus.RUNNING)
        self.storage.create_run(run)
        self._emit_event(session_id, "run.started", {
            "run_id": run.run_id,
            "task": task,
            "model": model_id,
        })
        return run

    def complete_run(
        self,
        run_id: str,
        outcome: str = "success",
        verification_passed: bool = False,
        result_summary: str = "",
        iterations: int = 0,
        tool_calls: int = 0,
        failed_tool_calls: int = 0,
        recovery_attempts: int = 0,
        total_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> Run:
        """Complete a run successfully."""
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.transition_to(RunStatus.COMPLETED)
        run.outcome = outcome
        run.verification_passed = verification_passed
        run.result_summary = result_summary
        run.iterations = iterations
        run.tool_calls = tool_calls
        run.failed_tool_calls = failed_tool_calls
        run.recovery_attempts = recovery_attempts
        run.total_tokens = total_tokens
        run.estimated_cost = estimated_cost

        self.storage.update_run(run)
        self._emit_event(run.session_id, "run.completed", {
            "run_id": run_id,
            "outcome": outcome,
            "verification": verification_passed,
        })
        return run

    def fail_run(
        self,
        run_id: str,
        error_message: str = "",
    ) -> Run:
        """Mark a run as failed."""
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.transition_to(RunStatus.FAILED)
        run.error_message = error_message
        run.outcome = "failure"

        self.storage.update_run(run)
        self._emit_event(run.session_id, "run.failed", {
            "run_id": run_id,
            "error": error_message[:200],
        })
        return run

    def interrupt_run(self, run_id: str) -> Run:
        """Mark a run as interrupted (Ctrl+C, crash, etc.)."""
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.transition_to(RunStatus.INTERRUPTED)
        run.outcome = "interrupted"

        self.storage.update_run(run)
        self._emit_event(run.session_id, "run.interrupted", {
            "run_id": run_id,
        })
        return run

    # ── Checkpointing ────────────────────────────────────────────────────

    def create_checkpoint(
        self,
        session_id: str,
        run_id: str = "",
        git_head: str = "",
        git_branch: str = "",
        git_dirty: bool = False,
        changed_files: list[str] | None = None,
        current_iteration: int = 0,
        current_task: str = "",
        model_id: str = "",
        verification_status: str = "",
        tests_passing: int = 0,
        tests_total: int = 0,
        context_files: list[str] | None = None,
        context_file_hashes: dict[str, str] | None = None,
        repository_summary: str = "",
    ) -> Checkpoint:
        """Create a checkpoint at a safe boundary."""
        cp = Checkpoint(
            session_id=session_id,
            run_id=run_id,
            git_head=git_head,
            git_branch=git_branch,
            git_dirty=git_dirty,
            changed_files=changed_files or [],
            current_iteration=current_iteration,
            current_task=current_task,
            model_id=model_id,
            verification_status=verification_status,
            tests_passing=tests_passing,
            tests_total=tests_total,
            context_files=context_files or [],
            context_file_hashes=context_file_hashes or {},
            repository_summary=repository_summary,
        )
        self.storage.create_checkpoint(cp)
        self._emit_event(session_id, "checkpoint.created", {
            "checkpoint_id": cp.checkpoint_id,
            "run_id": run_id,
            "git_head": git_head,
        })
        return cp

    def get_resume_state(self, session_id: str) -> dict[str, Any] | None:
        """Get the state needed to resume a session.

        Returns a dict with session, runs, checkpoint, and memories,
        or None if the session cannot be resumed.
        """
        session = self.storage.get_session(session_id)
        if session is None:
            return None

        # Check if resumable
        if session.status not in (SessionStatus.ACTIVE, SessionStatus.PAUSED, SessionStatus.FAILED, SessionStatus.COMPLETED):
            return None

        runs = self.storage.get_runs(session_id)
        checkpoint = self.storage.get_latest_checkpoint(session_id)

        # Get high-importance memories
        memories = self.storage.get_memories(
            session_id, min_importance=0.3, limit=50
        )

        # Detect interrupted runs
        interrupted_runs = [r for r in runs if r.status == RunStatus.INTERRUPTED]

        return {
            "session": session,
            "runs": runs,
            "checkpoint": checkpoint,
            "memories": memories,
            "interrupted_runs": interrupted_runs,
            "total_runs": len(runs),
            "completed_runs": len([r for r in runs if r.status == RunStatus.COMPLETED]),
            "failed_runs": len([r for r in runs if r.status == RunStatus.FAILED]),
        }

    def reconcile_workspace(
        self,
        session_id: str,
        workspace_path: str | None = None,
    ) -> ReconciliationReport:
        """Inspect and reconcile workspace and git state for a resumed session."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        runs = self.storage.get_runs(session_id)
        checkpoint = self.storage.get_latest_checkpoint(session_id)
        target_ws = Path(workspace_path or session.workspace_path or Path.cwd()).resolve()

        # 1. Inspect Git state in target workspace
        git_head = ""
        git_branch = ""
        git_dirty = False
        import hashlib
        import subprocess

        try:
            r_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(target_ws), capture_output=True, text=True, timeout=5)
            if r_head.returncode == 0:
                git_head = r_head.stdout.strip()
            r_branch = subprocess.run(["git", "branch", "--show-current"], cwd=str(target_ws), capture_output=True, text=True, timeout=5)
            if r_branch.returncode == 0:
                git_branch = r_branch.stdout.strip()
            r_status = subprocess.run(["git", "status", "--porcelain"], cwd=str(target_ws), capture_output=True, text=True, timeout=5)
            if r_status.returncode == 0:
                git_dirty = bool(r_status.stdout.strip())
        except Exception:
            pass

        # 2. Compare against checkpoint
        head_matches = True
        missing_files: list[str] = []
        externally_modified: list[str] = []
        stale_tests = False

        if checkpoint:
            if checkpoint.git_head and git_head and checkpoint.git_head != git_head:
                head_matches = False
                stale_tests = True

            for rel_path in checkpoint.changed_files:
                fpath = target_ws / rel_path
                if not fpath.exists():
                    missing_files.append(rel_path)
                    stale_tests = True
                elif rel_path in checkpoint.context_file_hashes:
                    try:
                        cur_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
                        if cur_hash != checkpoint.context_file_hashes[rel_path]:
                            externally_modified.append(rel_path)
                            stale_tests = True
                    except Exception:
                        pass

            if checkpoint.verification_status != "passed":
                stale_tests = True

        # 3. Analyze runs and tasks
        last_task = ""
        last_outcome = ""
        completed_steps: list[str] = []
        unresolved_steps: list[str] = []

        for r in runs:
            if r.status == RunStatus.COMPLETED:
                completed_steps.append(r.task)
            elif r.status in (RunStatus.FAILED, RunStatus.INTERRUPTED):
                unresolved_steps.append(f"{r.task} ({r.status.value})")

        if runs:
            last_run = runs[-1]
            last_task = last_run.task
            last_outcome = last_run.outcome or last_run.status.value

        # 4. Determine recommended action
        if unresolved_steps:
            recommended_action = f"Continue unfinished work: '{unresolved_steps[-1]}'"
        elif externally_modified:
            recommended_action = f"Review external changes to {', '.join(externally_modified[:3])} and run tests."
        elif stale_tests:
            recommended_action = "Run verification tests to validate current codebase state."
        elif last_task:
            recommended_action = f"Ready for next phase after: '{last_task}'"
        else:
            recommended_action = "Ready for instructions."

        # 5. Build recap text
        lines = [
            f"◆ Harness Session Resume",
            f"  Session: {session.title} ({session.session_id})",
            f"  Workspace: {target_ws.name}",
        ]
        if last_task:
            lines.append(f"  Last Goal: {last_task} [{last_outcome}]")
        if checkpoint and checkpoint.changed_files:
            lines.append(f"  Changed Files: {', '.join(checkpoint.changed_files[:5])}")
        if externally_modified:
            lines.append(f"  External Modifications: {', '.join(externally_modified)} (detected since last session)")
        if missing_files:
            lines.append(f"  Missing Files: {', '.join(missing_files)}")
        if checkpoint:
            v_status = checkpoint.verification_status or "None"
            t_info = f" ({checkpoint.tests_passing}/{checkpoint.tests_total} passed)" if checkpoint.tests_total > 0 else ""
            lines.append(f"  Verification: {v_status}{t_info}")
        lines.append(f"  Next Action: {recommended_action}")

        recap = "\n".join(lines)

        # 6. Re-activate session if paused or failed
        if session.status in (SessionStatus.PAUSED, SessionStatus.FAILED):
            session.transition_to(SessionStatus.ACTIVE)
            self.storage.update_session(session)

        self._emit_event(session_id, "session.reconciled", {
            "head_matches": head_matches,
            "modified_files": externally_modified,
            "missing_files": missing_files,
            "stale_tests": stale_tests,
        })

        return ReconciliationReport(
            session_id=session_id,
            workspace_path=str(target_ws),
            session_title=session.title,
            git_head=git_head,
            git_branch=git_branch,
            git_dirty=git_dirty,
            head_matches=head_matches,
            externally_modified_files=externally_modified,
            missing_files=missing_files,
            stale_tests=stale_tests,
            last_task=last_task,
            last_outcome=last_outcome,
            verification_status=checkpoint.verification_status if checkpoint else "",
            completed_steps=completed_steps,
            unresolved_steps=unresolved_steps,
            recommended_action=recommended_action,
            recap=recap,
        )

    # ── Memory management ────────────────────────────────────────────────

    def add_memory(
        self,
        session_id: str,
        memory_type: MemoryType,
        content: str,
        importance: float = 0.5,
        source_run_id: str = "",
        tags: list[str] | None = None,
    ) -> MemoryItem:
        """Add a memory item to a session."""
        # Sanitize content — redact potential secrets
        sanitized = self._sanitize_content(content)

        item = MemoryItem(
            session_id=session_id,
            memory_type=memory_type,
            content=sanitized,
            importance=max(0.0, min(1.0, importance)),
            source_run_id=source_run_id,
            tags=tags or [],
        )
        self.storage.add_memory(item)
        self._emit_event(session_id, "memory.created", {
            "memory_id": item.memory_id,
            "type": memory_type.value,
        })
        return item

    def get_relevant_memories(
        self,
        session_id: str,
        task_keywords: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Retrieve memories relevant to a task."""
        all_memories = self.storage.get_memories(session_id, limit=200)

        if not task_keywords:
            # Return high-importance memories
            return sorted(all_memories, key=lambda m: m.importance, reverse=True)[:limit]

        # Score memories by keyword relevance
        scored: list[tuple[float, MemoryItem]] = []
        keyword_set = set(k.lower() for k in task_keywords)

        for mem in all_memories:
            content_lower = mem.content.lower()
            relevance = sum(1 for k in keyword_set if k in content_lower)
            # Combine relevance with importance
            score = relevance * 0.6 + mem.importance * 0.4
            if relevance > 0 or mem.importance >= 0.7:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def get_project_memories(
        self,
        workspace_path: str,
        task_keywords: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Retrieve memories relevant to a project workspace."""
        all_memories = self.storage.get_project_memories(workspace_path, limit=200)

        if not task_keywords:
            return sorted(all_memories, key=lambda m: m.importance, reverse=True)[:limit]

        scored: list[tuple[float, MemoryItem]] = []
        keyword_set = set(k.lower() for k in task_keywords)

        for mem in all_memories:
            content_lower = mem.content.lower()
            relevance = sum(1 for k in keyword_set if k in content_lower)
            score = relevance * 0.6 + mem.importance * 0.4
            if relevance > 0 or mem.importance >= 0.7:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def sync_memory_file(self, workspace_path: str) -> Path:
        """Write human-readable project memory file in .harness/MEMORY.md."""
        ws = Path(workspace_path).resolve()
        harness_dir = ws / ".harness"
        harness_dir.mkdir(parents=True, exist_ok=True)
        mem_file = harness_dir / "MEMORY.md"

        memories = self.storage.get_project_memories(str(ws), limit=100)
        by_type: dict[str, list[MemoryItem]] = {}
        for m in memories:
            by_type.setdefault(m.memory_type.value, []).append(m)

        lines = [
            f"# Project Memory — {ws.name}",
            "",
            "> Persistent architectural context, confirmed decisions, and constraints.",
            "> Automatically synced by Harness across sessions.",
            "",
        ]

        section_titles = {
            "decision": "Confirmed Decisions",
            "architecture": "Architecture & Structure",
            "constraint": "Project Constraints",
            "discovery": "Discoveries & Conventions",
            "warning": "Known Issues & Warnings",
            "todo": "Open Items & Next Steps",
            "note": "General Notes",
        }

        for mtype, title in section_titles.items():
            items = by_type.get(mtype, [])
            if items:
                lines.append(f"## {title}")
                for item in items:
                    lines.append(f"- {item.content}")
                lines.append("")

        mem_file.write_text("\n".join(lines), encoding="utf-8")
        return mem_file

    def add_decision(self, session_id: str, content: str, importance: float = 0.7) -> MemoryItem:
        """Convenience: add a DECISION memory."""
        return self.add_memory(session_id, MemoryType.DECISION, content, importance)

    def add_discovery(self, session_id: str, content: str, importance: float = 0.6) -> MemoryItem:
        """Convenience: add a DISCOVERY memory."""
        return self.add_memory(session_id, MemoryType.DISCOVERY, content, importance)

    def add_constraint(self, session_id: str, content: str, importance: float = 0.8) -> MemoryItem:
        """Convenience: add a CONSTRAINT memory."""
        return self.add_memory(session_id, MemoryType.CONSTRAINT, content, importance)

    def add_todo(self, session_id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """Convenience: add a TODO memory."""
        return self.add_memory(session_id, MemoryType.TODO, content, importance)

    def add_warning(self, session_id: str, content: str, importance: float = 0.7) -> MemoryItem:
        """Convenience: add a WARNING memory."""
        return self.add_memory(session_id, MemoryType.WARNING, content, importance)

    # ── Event logging ────────────────────────────────────────────────────

    def _emit_event(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Record a session event."""
        event = SessionEvent(
            session_id=session_id,
            event_type=event_type,
            data=data,
        )
        self.storage.record_event(event)

    # ── Security ─────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_content(content: str) -> str:
        """Redact potential secrets from content before persistence."""
        import re

        # Redact API key patterns
        patterns = [
            (r'sk-or-[a-zA-Z0-9\-_]{20,}', '[REDACTED_API_KEY]'),
            (r'sk-[a-zA-Z0-9\-_]{20,}', '[REDACTED_API_KEY]'),
            (r'ghp_[a-zA-Z0-9]{36}', '[REDACTED_GITHUB_TOKEN]'),
            (r'github_pat_[a-zA-Z0-9_]{80,}', '[REDACTED_GITHUB_PAT]'),
            (r'AIza[a-zA-Z0-9_\-]{35}', '[REDACTED_GOOGLE_KEY]'),
            (r'xoxb-[a-zA-Z0-9\-]+', '[REDACTED_SLACK_TOKEN]'),
            (r'Bearer\s+[a-zA-Z0-9\-_.]+', 'Bearer [REDACTED]'),
            (r'Authorization:\s*Bearer\s+\S+', 'Authorization: Bearer [REDACTED]'),
        ]

        result = content
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    # ── Export ────────────────────────────────────────────────────────────

    def export_session(self, session_id: str, fmt: str = "json") -> str:
        """Export session data as JSON or Markdown."""
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        runs = self.storage.get_runs(session_id)
        memories = self.storage.get_memories(session_id)
        checkpoint = self.storage.get_latest_checkpoint(session_id)
        events = self.storage.get_events(session_id, limit=100)

        if fmt == "markdown":
            return self._export_markdown(session, runs, memories, checkpoint)

        # JSON export
        import json
        return json.dumps({
            "session": session.to_dict(),
            "runs": [r.to_dict() for r in runs],
            "memories": [m.to_dict() for m in memories],
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
            "events": [e.to_dict() for e in events],
        }, indent=2)

    def _export_markdown(
        self,
        session: Session,
        runs: list[Run],
        memories: list[MemoryItem],
        checkpoint: Checkpoint | None,
    ) -> str:
        """Export session as readable Markdown."""
        lines = [
            f"# Session: {session.title}",
            "",
            f"**ID:** `{session.session_id}`",
            f"**Status:** {session.status.value}",
            f"**Workspace:** {session.workspace_path}",
            f"**Created:** {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.created_at))}",
            "",
            "## Runs",
            "",
        ]

        for run in runs:
            status_icon = "✅" if run.status == RunStatus.COMPLETED else "❌" if run.status == RunStatus.FAILED else "⏸️"
            lines.append(f"### {status_icon} Run `{run.run_id}`")
            lines.append(f"- **Task:** {run.task}")
            lines.append(f"- **Status:** {run.status.value}")
            lines.append(f"- **Model:** {run.model_id or 'N/A'}")
            lines.append(f"- **Tool calls:** {run.tool_calls}")
            lines.append(f"- **Iterations:** {run.iterations}")
            if run.result_summary:
                lines.append(f"- **Result:** {run.result_summary}")
            if run.error_message:
                lines.append(f"- **Error:** {run.error_message}")
            lines.append("")

        if memories:
            lines.append("## Memories")
            lines.append("")
            by_type: dict[str, list[MemoryItem]] = {}
            for m in memories:
                by_type.setdefault(m.memory_type.value, []).append(m)

            for mtype, items in by_type.items():
                lines.append(f"### {mtype.title()}")
                for item in items:
                    lines.append(f"- {item.content}")
                lines.append("")

        if checkpoint:
            lines.append("## Last Checkpoint")
            lines.append("")
            lines.append(f"- **Git HEAD:** `{checkpoint.git_head or 'N/A'}`")
            lines.append(f"- **Branch:** `{checkpoint.git_branch or 'N/A'}`")
            lines.append(f"- **Tests:** {checkpoint.tests_passing}/{checkpoint.tests_total}")
            lines.append(f"- **Verification:** {checkpoint.verification_status or 'N/A'}")
            lines.append("")

        return "\n".join(lines)
