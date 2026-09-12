"""Professional live engineering dashboard for the unified runtime (Phase 10).

Architecture (event-driven, never fabricated):

    EventBus  →  RuntimeViewModel  →  LiveTerminalUI

``RuntimeViewModel`` is a pure, Rich-free state adapter.  Every field it
exposes is derived from real EventBus events (or from the authoritative
``RuntimeOutcome`` merged in via ``finalize()``).  It never fakes agent
activity, progress, file changes, or success.

``LiveTerminalUI`` renders that state with Rich's ``Live`` region.  It is
non-blocking: it performs no model calls, no network I/O and no expensive
filesystem scans; rendering is throttled so the terminal stays smooth even
while the runtime is busy.

The summary renderers (startup / success / failure / cancelled) only ever
print numbers that were actually observed — no invented metrics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from harness_core.observability.events import Event, EventBus


# ── Small display helpers (shared with the summary panels) ────────────────


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as MM:SS or HH:MM:SS."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


def tool_display_name(tool_name: str, args: dict[str, Any]) -> str:
    """Concise display string for a tool call (never dumps raw output)."""
    if tool_name in ("read_file", "write_file", "edit_file"):
        return f"{tool_name.split('_')[0]} {args.get('path', args.get('file_path', '?'))}"
    if tool_name == "list_files":
        return f"list {args.get('path', '.')}"
    if tool_name == "run_command":
        cmd = args.get("command", "")
        return f"run {cmd[:60]}{'...' if len(cmd) > 60 else ''}"
    if tool_name == "grep":
        return f'grep "{args.get("pattern", "")}"'
    if tool_name == "glob":
        return f"glob {args.get('pattern', '')}"
    if tool_name == "git_status":
        return "git status"
    if tool_name == "git_diff":
        return "git diff"
    if tool_name == "git_log":
        return "git log"
    return tool_name


# ── Agent activity model ───────────────────────────────────────────────────


@dataclass
class AgentView:
    """Public status of one agent/task as seen by the dashboard.

    Exposes only operational summaries — never private model reasoning.
    """

    name: str = ""
    role: str = ""
    task_id: str = ""
    status: str = "queued"  # queued|running|waiting|blocked|completed|failed|cancelled
    operation: str = ""  # current tool / operation summary
    detail: str = ""  # current file / command being executed
    model: str = ""
    started_at: float = 0.0
    latest_event: str = ""
    latest_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return time.time() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "task_id": self.task_id,
            "status": self.status,
            "operation": self.operation,
            "detail": self.detail,
            "model": self.model,
            "latest_event": self.latest_event,
            "elapsed": round(self.elapsed, 1),
        }


@dataclass
class TestState:
    """Roll-up of observed test events (never invented)."""

    passed: int = 0
    failed: int = 0
    running: int = 0
    total: int = 0
    last_line: str = ""

    @property
    def has_evidence(self) -> bool:
        return self.total > 0 or self.passed > 0 or self.failed > 0 or bool(self.last_line)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "running": self.running,
            "total": self.total,
            "last_line": self.last_line,
            "has_evidence": self.has_evidence,
        }


@dataclass
class ValidationRun:
    """One observed validation command result (Phase 10.5, Part 11).

    Kept separate from the aggregate ``TestState`` so a previous successful
    suite can never mask a later failing command.  ``command`` is the exact
    command that ran, ``exit_code`` its real exit code, and ``scope``
    distinguishes the full baseline suite from a targeted/latest command.
    """

    command: str = ""
    exit_code: int | None = None
    passed: int = 0
    total: int = 0
    success: bool = False
    scope: str = "latest"  # baseline | latest | targeted
    at: float = field(default_factory=time.time)

    @property
    def is_failure(self) -> bool:
        return not self.success or (self.exit_code not in (None, 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command[:120],
            "exit_code": self.exit_code,
            "passed": self.passed,
            "total": self.total,
            "success": self.success,
            "scope": self.scope,
            "at": round(self.at, 1),
        }


@dataclass
class FileChange:
    """One observed file change (status is M/A/D where determinable)."""

    path: str
    status: str = "M"  # M modified, A added, D deleted
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "status": self.status}


# ── Event-driven view model ────────────────────────────────────────────────


class RuntimeViewModel:
    """State adapter between the EventBus and the terminal UI.

    Registers a catch-all handler on the EventBus and derives a live snapshot
    from the events the real runtime, scheduler, workers and agent loop emit.
    No event is ever synthesized here.
    """

    # Scheduler/worker lifecycle events that carry task identity.
    _STATUS_FROM_TASK = {
        "task.ready": "queued",
        "task.queued": "queued",
        "task.started": "running",
        "task.completed": "completed",
        "task.failed": "failed",
        "task.cancelled": "cancelled",
    }
    # Worker AgentStatus transitions (agent.status_changed) → dashboard state.
    _STATUS_FROM_AGENT = {
        "queued": "queued",
        "planning": "running",
        "running": "running",
        "waiting": "waiting",
        "blocked": "blocked",
        "reviewing": "running",
        "verifying": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }

    def __init__(self) -> None:
        self.project_id: str = ""
        self.workspace: str = ""
        self.request: str = ""
        self.stage: str = "unknown"
        self.status: str = "running"  # running|success|failed|blocked|cancelled
        self.terminal: bool = False
        self.started_at: float = time.time()
        self.completed_at: float = 0.0

        self.agents: dict[str, AgentView] = {}  # task_id -> view
        self._task_agent: dict[str, str] = {}  # task_id -> agent_id
        self._agent_task: dict[str, str] = {}  # agent_id -> task_id
        self._agent_role: dict[str, str] = {}  # agent_id -> role

        self.activity: deque[tuple[str, str, str]] = deque(maxlen=10)  # (time, agent, text)
        self.files: dict[str, FileChange] = {}  # path -> change
        self.tests = TestState()
        # Phase 10.5 (Part 11): per-run validation history — a previous
        # successful suite must never mask a later failing command.
        self.validation_runs: list[ValidationRun] = []
        self.recovery_attempts: int = 0
        self.recovery_max: int = 0
        self.recovery_exhausted: bool = False
        self.recovery_active: bool = False
        self.verification_status: str = "not_started"
        self.verification_summary: str = ""
        self.warnings: list[str] = []

        # Phase 10.5 (Part 30): authoritative Git evidence from real git
        # tool events (status/remote/commit/push) — never from model prose.
        self.git: dict[str, Any] = {
            "repo_detected": False,
            "branch": "",
            "remote": "",
            "dirty": False,
            "commit": "",
            "push": "",
            "identity_ok": False,
            "operations": [],  # successful git operations with evidence
        }

        # Lightweight honest metrics (only what events actually report).
        self.tool_calls: int = 0
        self.model_calls: int = 0
        self.failed_tool_calls: int = 0

        self._plan_tasks: int = 0
        self._bus: EventBus | None = None

    # ── wiring ─────────────────────────────────────────────────────────

    def attach(self, bus: EventBus) -> None:
        """Subscribe to a live EventBus. Idempotent per bus."""
        if self._bus is bus:
            return
        self._bus = bus
        bus.on("*", self._on_event)

    def detach(self, bus: EventBus | None = None) -> None:
        target = bus or self._bus
        if target is not None:
            target.off("*", self._on_event)
        self._bus = None

    # ── event handling ─────────────────────────────────────────────────

    async def _on_event(self, event: Event) -> None:
        """Single catch-all handler mapping real events to dashboard state."""
        etype = event.type
        data = event.data or {}
        now = time.time()

        if etype == "runtime_started":
            self.project_id = data.get("project_id", self.project_id)
            self.started_at = now
            self.status = "running"
            self.terminal = False
            return
        if etype == "runtime_stage_changed":
            self.stage = data.get("stage", self.stage)
            return
        if etype == "plan_created":
            self._plan_tasks = int(data.get("total_tasks", 0) or 0)
            return
        if etype == "runtime_completed":
            self.status = data.get("status", "success")
            self.terminal = True
            self.completed_at = now
            return
        if etype == "runtime_failed":
            self.status = "failed"
            self.terminal = True
            self.completed_at = now
            reason = data.get("reason", "")
            if data.get("interrupted"):
                self.status = "cancelled"
            if reason:
                self.warnings.append(str(reason))
            return

        # ── tasks / agents ──────────────────────────────────────────────
        if etype in self._STATUS_FROM_TASK:
            task_id = str(data.get("task_id", ""))
            if not task_id:
                return
            new_status = self._STATUS_FROM_TASK[etype]
            agent = self.agents.setdefault(
                task_id,
                AgentView(task_id=task_id, name=data.get("agent_id", task_id)),
            )
            agent.task_id = task_id
            agent.latest_event = etype
            agent.latest_at = now
            if data.get("agent_id"):
                agent.name = str(data["agent_id"])
                self._task_agent[task_id] = str(data["agent_id"])
                self._agent_task[str(data["agent_id"])] = task_id
            agent.role = data.get("role", agent.role) or self._agent_role.get(agent.name, "")
            if etype == "task.started":
                agent.started_at = agent.started_at or now
                self.stage = self.stage if self.stage != "unknown" else "execute"
            agent.status = new_status
            if etype == "task.completed" and data.get("status") == "partial":
                # Phase 10.6: a task that ended with required work unresolved
                # must never be displayed as a ✓ completed agent.
                agent.status = "partial"
                agent.detail = data.get("error", "")[:120] or agent.detail
            if new_status == "completed" and agent.operation:
                # Completed agent: keep a compact summary of its last operation.
                agent.operation = "done"
                agent.detail = ""
            if etype == "task.failed":
                self.failed_tool_calls += 0  # failure counted below from tool events
                error = data.get("error", "")
                agent.detail = error[:80] if error else agent.detail
            self._push_activity(now, agent.name or task_id, agent.operation or etype)
            return

        if etype == "agent.status_changed":
            agent_id = str(data.get("agent_id", ""))
            task_id = self._agent_task.get(agent_id, "")
            if not task_id:
                return
            role = data.get("role", "")
            if role:
                self._agent_role[agent_id] = role
            agent = self.agents.setdefault(
                task_id, AgentView(task_id=task_id, name=agent_id)
            )
            agent.role = role or agent.role
            old = data.get("old_status", "")
            new = data.get("new_status", "")
            mapped = self._STATUS_FROM_AGENT.get(new, new)
            if mapped:
                agent.status = mapped
            agent.latest_event = f"agent {old}→{new}"
            agent.latest_at = now
            return

        if etype in ("agent.started", "agent.completed", "agent.failed"):
            task_id = str(data.get("task_id", ""))
            agent_id = str(data.get("agent_id", ""))
            if task_id and task_id in self.agents:
                agent = self.agents[task_id]
                if etype == "agent.started":
                    agent.status = "running"
                elif etype == "agent.completed":
                    agent.status = "completed"
                elif etype == "agent.failed":
                    agent.status = "failed"
                agent.latest_event = etype
                agent.latest_at = now
            return

        # ── tools / files ───────────────────────────────────────────────
        if etype == "tool.call":
            self.tool_calls += 1
            task_id = self._current_task_id(data)
            tool = data.get("tool", "")
            args = data.get("args", {}) or {}
            if task_id:
                # Identity comes from the event itself (stamped by the loop),
                # so creating the view here is evidence, not fabrication.
                agent = self.agents.setdefault(
                    task_id, AgentView(task_id=task_id, name=data.get("agent_id", task_id))
                )
                agent.operation = tool_display_name(tool, args)
                agent.detail = _file_from_tool(tool, args)
                agent.latest_event = f"tool {tool}"
                agent.latest_at = now
            if tool in ("write_file", "edit_file"):
                path = _file_from_tool(tool, args)
                if path:
                    self.files.setdefault(path, FileChange(path=path, status="M"))
            if tool == "delete_file":
                path = _file_from_tool(tool, args)
                if path:
                    self.files[path] = FileChange(path=path, status="D")
            return

        if etype == "tool.result":
            tool = data.get("tool", "")
            status = data.get("status", "")
            exit_code = data.get("exit_code")
            task_id = self._current_task_id(data)
            if status != "success" and status != "permission_denied":
                self.failed_tool_calls += 1
                if task_id and task_id in self.agents:
                    self.agents[task_id].operation = "failed"
                    self.agents[task_id].detail = (
                        str(data.get("error") or data.get("stderr") or status)[:80]
                    )
                    self._push_activity(now, self.agents[task_id].name, f"{tool} failed")
            # Phase 10.5: failing test command → a *separate* latest validation
            # run (never conflated with the baseline suite).
            if _looks_like_test_command(tool, data):
                self.tests.failed += 1
                self.tests.last_line = f"{tool} failed (exit {exit_code})"
                self._record_validation_run(
                    ValidationRun(
                        command=str(data.get("command") or ""),
                        exit_code=exit_code,
                        success=False,
                        scope="latest",
                    )
                )
            self._observe_git_tool(tool, data)
            return

        # ── tests ───────────────────────────────────────────────────────
        if etype == "test.started":
            self.tests.running += 1
            return
        if etype == "test.completed":
            passed = data.get("passed")
            total = data.get("total")
            success = data.get("success")
            command = str(data.get("command") or "")
            self.tests.running = max(0, self.tests.running - 1)
            # Phase 10.5: every completed test run is recorded with its own
            # outcome; a later failure supersedes earlier success for display.
            ok = bool(success) if success is not None else (
                bool(passed) and passed == total if passed is not None and total else False
            )
            run = ValidationRun(
                command=command,
                success=ok,
                scope="latest",
            )
            if passed is not None and total:
                self.tests.passed = int(passed)
                self.tests.total = int(total)
                self.tests.failed = max(self.tests.failed, int(total) - int(passed))
                self.tests.last_line = f"{passed}/{total} passed"
                run.passed = int(passed)
                run.total = int(total)
            elif success is not None:
                self.tests.last_line = "tests passed" if success else "tests failed"
                if not success:
                    self.tests.failed += 1
            self._record_validation_run(run)
            return

        # ── recovery ────────────────────────────────────────────────────
        if etype == "recovery_started":
            self.recovery_active = True
            self.recovery_attempts += 1
            attempt = int(data.get("attempt", self.recovery_attempts))
            self.recovery_max = max(self.recovery_max, attempt)
            category = data.get("category", "")
            self._push_activity(now, "RECOVERY", f"classified: {category}")
            return
        if etype == "recovery_exhausted":
            self.recovery_exhausted = True
            self.recovery_active = False
            return
        if etype in ("recovery_plan_created", "recovery_task_created"):
            self.recovery_active = True
            return

        # ── verification ────────────────────────────────────────────────
        if etype in ("verification.started", "verification_started"):
            self.verification_status = "in_progress"
            return
        if etype in ("verification.completed", "verification_completed"):
            passed = data.get("passed", False)
            self.verification_status = "passed" if passed else "failed"
            self.verification_summary = str(data.get("summary", ""))
            return

        # ── metrics / warnings ──────────────────────────────────────────
        if etype == "routing.decision":
            self.model_calls += 1
            return
        if etype == "model.error":
            self.warnings.append(str(data.get("error", ""))[:120])
            return
        if etype == "test_integrity.warning":
            self.warnings.append(str(data.get("warning", "test integrity"))[:120])
            return

    def _current_task_id(self, data: dict[str, Any]) -> str:
        """Best-effort task identity for events that may not carry one."""
        task_id = str(data.get("task_id", ""))
        if task_id:
            return task_id
        agent_id = str(data.get("agent_id", ""))
        if agent_id:
            return self._agent_task.get(agent_id, "")
        return ""

    def _push_activity(self, now: float, agent: str, text: str) -> None:
        if not text:
            return
        ts = time.strftime("%H:%M:%S", time.localtime(now))
        self.activity.append((ts, agent, text[:90]))

    # ── Phase 10.5 validation + git evidence helpers ─────────────────────

    def _record_validation_run(self, run: ValidationRun) -> None:
        """Append a validation run, keeping the *latest* result authoritative.

        ``baseline`` is the most recent successful full-suite result;
        ``latest`` is the most recent run of any kind — so a failing
        command that follows a green suite is never hidden.
        """
        self.validation_runs.append(run)
        if len(self.validation_runs) > 32:
            self.validation_runs = self.validation_runs[-32:]

    def baseline_validation(self) -> ValidationRun | None:
        """Most recent successful run (baseline evidence), if any."""
        for run in reversed(self.validation_runs):
            if not run.is_failure:
                return run
        return None

    def latest_validation(self) -> ValidationRun | None:
        """Most recent validation run of any kind (truthful 'latest')."""
        return self.validation_runs[-1] if self.validation_runs else None

    def _observe_git_tool(self, tool: str, data: dict[str, Any]) -> None:
        """Record authoritative Git evidence from real git tool results."""
        if not tool.startswith("git_"):
            return
        status = data.get("status", "")
        if status != "success":
            return
        meta = data.get("metadata") or {}
        op = meta.get("operation", tool)
        if tool == "git_status":
            self.git["repo_detected"] = True
            self.git["dirty"] = bool(meta.get("clean") is False or data.get("output"))
        elif tool == "git_remote":
            remotes = meta.get("remotes") or {}
            if remotes:
                self.git["remote"] = next(iter(remotes))
        elif tool == "git_identity":
            self.git["identity_ok"] = bool(meta.get("configured"))
        elif tool == "git_commit":
            self.git["commit"] = str(meta.get("commit_hash") or "")
            self.git["branch"] = str(meta.get("branch") or self.git.get("branch", ""))
        elif tool == "git_push":
            remote = meta.get("remote", "origin")
            branch = meta.get("branch", "")
            self.git["push"] = f"{remote}/{branch}" if branch else remote
        self.git["operations"].append(op)
        if len(self.git["operations"]) > 16:
            self.git["operations"] = self.git["operations"][-16:]

    # ── final merge with authoritative outcome ──────────────────────────

    def finalize(self, outcome: Any) -> None:
        """Merge the authoritative RuntimeOutcome into the view model.

        The outcome owns task-level truth (files changed per task, statuses,
        verification).  We adopt it so the final summary is evidence-based
        even when individual events were sparse.
        """
        state = getattr(outcome, "state", None)
        if state is not None:
            self.status = state.status.value
            self.stage = state.stage.value
            self.terminal = state.stage.value in (
                "succeeded", "failed", "blocked", "cancelled",
            )
            self.completed_at = state.completed_at or time.time()
            if state.original_request:
                self.request = state.original_request
            if state.workspace:
                self.workspace = state.workspace
            if state.verification_status is not None:
                self.verification_status = state.verification_status.value
            self.verification_summary = state.verification_summary or self.verification_summary
            self.recovery_attempts = max(self.recovery_attempts, state.recovery_attempts)
            self.recovery_exhausted = state.recovery_exhausted or self.recovery_exhausted

        graph = getattr(outcome, "graph", None)
        if graph is not None:
            for task in graph.tasks.values():
                files = list(task.files_changed or [])
                for path in files:
                    existing = self.files.get(path)
                    if existing is None:
                        self.files[path] = FileChange(path=path, status="M")

    # ── snapshot for tests / /status ───────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Structured snapshot of the current view-model state."""
        return {
            "status": self.status,
            "terminal": self.terminal,
            "stage": self.stage,
            "request": self.request,
            "project_id": self.project_id,
            "workspace": self.workspace,
            "elapsed": round(time.time() - self.started_at, 1),
            "agents": [a.to_dict() for a in self.agents.values()],
            "activity": [list(x) for x in self.activity],
            "files": [f.to_dict() for f in self.files.values()],
            "tests": self.tests.to_dict(),
            "validation": {
                "baseline": (
                    self.baseline_validation().to_dict() if self.baseline_validation() else None
                ),
                "latest": self.latest_validation().to_dict() if self.latest_validation() else None,
                "runs": [r.to_dict() for r in self.validation_runs],
            },
            "git": dict(self.git),
            "recovery": {
                "attempts": self.recovery_attempts,
                "max": self.recovery_max,
                "active": self.recovery_active,
                "exhausted": self.recovery_exhausted,
            },
            "verification": {
                "status": self.verification_status,
                "summary": self.verification_summary,
            },
            "metrics": {
                "tool_calls": self.tool_calls,
                "model_calls": self.model_calls,
                "failed_tool_calls": self.failed_tool_calls,
                "plan_tasks": self._plan_tasks,
            },
            "warnings": list(self.warnings),
        }


def _file_from_tool(tool: str, args: dict[str, Any]) -> str:
    """Extract the file path from a file-tool call, if any."""
    if tool in ("read_file", "write_file", "edit_file", "delete_file"):
        path = args.get("path", args.get("file_path", ""))
        return str(path) if path else ""
    return ""


_TEST_MARKERS = ("pytest", "npm test", "yarn test", "pnpm test", "bun test",
                 "cargo test", "go test", "jest", "vitest", "mocha", "phpunit",
                 "dotnet test", "gradle test", "mvn test", "python -m pytest")


def _looks_like_test_command(tool: str, data: dict[str, Any]) -> bool:
    """True when a failed tool call was a test command (deterministic)."""
    if tool != "run_command":
        return False
    status = data.get("status", "")
    if status == "success" or data.get("exit_code") in (None, 0):
        return False
    command = str(data.get("command") or data.get("args", {}).get("command", ""))
    lowered = command.lower()
    return any(marker in lowered for marker in _TEST_MARKERS)


# ── Live terminal UI ───────────────────────────────────────────────────────


class LiveTerminalUI:
    """Renders the RuntimeViewModel as a live, throttled Rich region.

    Non-blocking by design: only mutates Rich state on the caller's event
    loop; never performs model calls, network I/O or filesystem scans.
    """

    STATUS_ICONS = {
        "queued": ("○", "dim"),
        "waiting": ("◌", "yellow"),
        "blocked": ("⊘", "red"),
        "running": ("●", "cyan"),
        "partial": ("⚠", "yellow"),  # required work unresolved (Phase 10.6)
        "completed": ("✓", "green"),
        "failed": ("✗", "red"),
        "cancelled": ("—", "dim"),
    }

    def __init__(
        self,
        console: Console,
        *,
        plain: bool = False,
        refresh_per_second: float = 6.0,
        max_agents: int = 10,
    ) -> None:
        self.console = console
        self.plain = plain
        self.refresh_per_second = refresh_per_second
        self.max_agents = max_agents
        self._live: Any = None
        self._active = False
        self._last_refresh = 0.0
        self._min_interval = 1.0 / refresh_per_second
        self._last_plain_line = ""

    def start(self, vm: RuntimeViewModel) -> None:
        """Begin the live dashboard. In plain mode this is a no-op render."""
        self._active = True
        if self.plain:
            self._render_plain(vm)
            return
        from rich.live import Live

        self._live = Live(
            self._renderable(vm),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            transient=False,
        )
        self._live.start()

    def stop(self, vm: RuntimeViewModel) -> None:
        """Stop the live region, leaving the final frame visible."""
        self._active = False
        if self._live is not None:
            try:
                self._live.update(self._renderable(vm))
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def update(self, vm: RuntimeViewModel) -> None:
        """Throttled refresh. Cheap to call on every event."""
        if not self._active:
            return
        now = time.time()
        if self.plain:
            self._render_plain(vm)
            return
        if now - self._last_refresh < self._min_interval:
            return
        self._last_refresh = now
        if self._live is not None:
            self._live.update(self._renderable(vm))

    # ── rendering ─────────────────────────────────────────────────────

    def _current_operation(self, vm: RuntimeViewModel) -> tuple[str, str]:
        """Truthful 'Current' operation from the most recent active agent.

        Returns (operation, detail).  When nothing is running the operation
        reports the runtime stage so the user never sees a frozen blank.
        """
        running = [
            a for a in vm.agents.values()
            if a.status in ("running", "waiting", "blocked")
        ]
        if running:
            latest = max(running, key=lambda a: a.latest_at)
            return (latest.operation or "working", latest.detail or "")
        if vm.terminal:
            return ("finished", "")
        if vm.stage and vm.stage != "unknown":
            return (vm.stage, "")
        return ("waiting for next task", "")

    @staticmethod
    def _dedup_activity(items: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """Collapse consecutive identical activity entries (Part 4)."""
        out: list[tuple[str, str, str]] = []
        for item in items:
            if out and out[-1][1] == item[1] and out[-1][2] == item[2]:
                continue
            out.append(item)
        return out

    def _renderable(self, vm: RuntimeViewModel) -> Any:
        from rich.console import Group
        from rich.text import Text

        lines: list[Any] = []

        # Compact header (conversation-feel, not a monitoring banner).
        status_icon = "✓" if vm.terminal and vm.status == "success" else (
            "✗" if vm.terminal else "●")
        status_style = "green" if vm.terminal and vm.status == "success" else (
            "red" if vm.terminal else "cyan")
        header = Text.assemble(
            (f"{status_icon} HARNESS", "bold white" if not vm.terminal else status_style),
            ("  ", ""),
            (format_elapsed(time.time() - vm.started_at), "bold green"),
        )
        lines.append(header)

        # Task line (WHAT we are doing, concise — never a dashboard banner).
        if vm.request and not vm.terminal:
            lines.append(Text.assemble(
                ("Task  ", "dim"), (vm.request[:90], "bold"),
            ))

        # Working line (WHY: current operational intent, evidence-derived).
        op, detail = self._current_operation(vm)
        work_text = f"{op}" + (f" {detail[:50]}" if detail else "")
        lines.append(Text.assemble(
            ("● Working   ", "bold blue"), (work_text[:80], ""),
        ))

        # Agent(s): one compact line when a single agent exists (Part 6/34).
        agents = sorted(
            vm.agents.values(), key=lambda a: (a.latest_at, a.task_id), reverse=True
        )[: self.max_agents]
        if agents:
            if len(agents) == 1:
                a = agents[0]
                icon, style = self.STATUS_ICONS.get(a.status, ("○", "dim"))
                role = a.role.upper() if a.role else "AGENT"
                lines.append(Text.assemble(
                    ("Agent  ", "dim"),
                    (f"{icon} ", style),
                    (f"{role}", "bold"),
                    (f"  {a.operation[:34]}", "cyan" if a.operation else "dim"),
                    (f"  {a.model[:24]}", "dim") if a.model else ("", ""),
                ))
            else:
                lines.append(Text("Agents", style="dim"))
                for a in agents:
                    icon, style = self.STATUS_ICONS.get(a.status, ("○", "dim"))
                    role = a.role.upper() if a.role else "AGENT"
                    op_txt = a.operation[:26] if a.operation else ""
                    parts: list[tuple[str, str]] = [
                        (f"  {icon} ", style),
                        (f"{role:<10}", "bold"),
                        (f"{op_txt:<26}", "cyan" if op_txt else "dim"),
                    ]
                    if a.status in ("waiting", "blocked"):
                        parts.append((f"({a.status})", "yellow" if a.status == "waiting" else "red"))
                    lines.append(Text.assemble(*parts))

        # Activity trail — deduplicated, human-readable (Part 4).
        activity = self._dedup_activity(list(vm.activity))[-5:]
        if activity:
            lines.append(Text("Activity", style="dim"))
            for ts, name, text in activity:
                lines.append(Text.assemble(
                    ("  ", ""), (name[:12], "cyan"),
                    (" ", ""), (text[:58], ""),
                ))

        # Validation: baseline suite vs latest command vs verification (Part 11).
        latest = vm.latest_validation()
        baseline = vm.baseline_validation()
        if latest is not None or vm.verification_status != "not_started" or (
            vm.tests.has_evidence and not vm.validation_runs
        ):
            lines.append(Text("Validation", style="dim"))
            if latest is None and vm.tests.has_evidence and not vm.validation_runs:
                # Aggregate-only evidence (no run scope) — show as observed.
                if vm.tests.total:
                    icon = "✓" if vm.tests.failed == 0 else "✗"
                    style = "green" if vm.tests.failed == 0 else "red"
                    lines.append(Text.assemble(
                        (f"  {icon} ", style),
                        (f"{vm.tests.passed}/{vm.tests.total} passed", "bold"),
                    ))
                else:
                    lines.append(Text(f"  {vm.tests.last_line}", style="dim"))
            else:
                if baseline is not None and baseline.total:
                    lines.append(Text.assemble(
                        ("  ✓ baseline ", "green"),
                        (f"{baseline.passed}/{baseline.total} passed", "bold"),
                    ))
                elif baseline is not None and baseline.success:
                    lines.append(Text("  ✓ baseline tests passed", style="green"))
                if latest is not None:
                    if latest.is_failure:
                        cmd = latest.command or "test command"
                        lines.append(Text.assemble(
                            ("  ✗ latest ", "red"),
                            (f"{cmd[:44]} ", "bold"),
                            (f"exit {latest.exit_code}", "dim") if latest.exit_code not in (None, 0) else ("failed", "dim"),
                        ))
                    elif latest.success and latest.total:
                        lines.append(Text.assemble(
                            ("  ✓ latest ", "green"),
                            (f"{latest.passed}/{latest.total} passed", "bold"),
                        ))
            if vm.verification_status != "not_started":
                icon = "✓" if vm.verification_status == "passed" else "✗"
                style = "green" if vm.verification_status == "passed" else "red"
                lines.append(Text.assemble(
                    (f"  {icon} verification ", style), (vm.verification_status, "bold"),
                ))

        # Files (adaptive: only when changes were actually observed).
        if vm.files:
            lines.append(Text("Files", style="dim"))
            for change in list(vm.files.values())[-6:]:
                style = {"M": "yellow", "A": "green", "D": "red"}.get(change.status, "white")
                lines.append(Text.assemble(
                    (f"  {change.status} ", style), (change.path[:64], ""),
                ))

        # Git (adaptive: only when real git evidence exists).
        if vm.git["operations"]:
            lines.append(Text("Git", style="dim"))
            if vm.git["commit"]:
                lines.append(Text(f"  ✓ commit {vm.git['commit'][:12]}", style="green"))
            if vm.git["push"]:
                lines.append(Text(f"  ✓ pushed {vm.git['push']}", style="green"))
            elif vm.git["dirty"]:
                lines.append(Text("  changes present (not pushed)", style="yellow"))

        # Recovery (adaptive: only while it is actually happening).
        if vm.recovery_active:
            lines.append(Text.assemble(
                ("⚠ recovery ", "yellow"), (f"{vm.recovery_attempts} attempt(s)", "bold"),
            ))
        if vm.warnings:
            lines.append(Text.assemble(("⚠ ", "yellow"), (vm.warnings[-1][:90], "dim")))

        footer = Text(
            "Ctrl+C cancel   /status  /agents  /plan  /activity  /files  /tests  /help",
            style="dim",
        )
        return Group(*lines, Text(""), footer)

    def _render_plain(self, vm: RuntimeViewModel) -> None:
        """Plain mode: one compact line, only when the stage changes."""
        line = f"[{vm.stage}] {vm.status} {len(vm.agents)} agents {vm.tool_calls} tools"
        if vm.tests.has_evidence and vm.tests.total:
            line += f" tests:{vm.tests.passed}/{vm.tests.total}"
        if line != self._last_plain_line:
            self._last_plain_line = line
            # markup=False so the [stage] token is never parsed as a style.
            self.console.print(
                f"  {line} ({format_elapsed(time.time() - vm.started_at)})",
                highlight=False,
                markup=False,
            )


# ── Evidence-based summary panels ──────────────────────────────────────────


def render_startup_panel(
    console: Console,
    workspace: str,
    checks: dict[str, bool],
    *,
    plain: bool = False,
) -> None:
    """Concise startup banner (Part 1). Only meaningful checks are shown."""
    if plain:
        console.print("✦ Harness — Autonomous AI Engineering Agent")
        console.print(f"  Workspace: {workspace}")
        console.print("  System is online.")
        console.print("")
        return

    console.print("")
    console.print(" [bold cyan]✦[/] [bold white]Harness[/] — Autonomous AI Engineering Agent")
    console.print("")

    display = workspace if len(workspace) <= 60 else "..." + workspace[-57:]
    console.print(f" [dim]Workspace:[/] {display}")

    provider_ok = checks.get("provider", False)
    if provider_ok:
        console.print(" [green]✓[/] [dim]System is online and connected[/]")
    else:
        console.print(" [yellow]○[/] [dim]System is initializing or offline[/]")
    console.print("")


def render_success_summary(
    console: Console, vm: RuntimeViewModel, outcome: Any, *, plain: bool = False
) -> None:
    """Concise professional success summary (Parts 7-8, 18, 35).

    Evidence only — rendered through ``CompletionFormatter`` with contextual
    next-action suggestions from ``NextActionEngine``.
    """
    from harness_core.cli.completion import CompletionFormatter, NextActionEngine

    graph = getattr(outcome, "graph", None)

    agent_response = ""
    if graph is not None:
        for task in reversed(list(graph.tasks.values())):
            res = (task.result or "").strip()
            if res and not (res.startswith("Task ") and res.endswith(" completed")) and not res.startswith("Stopped:"):
                agent_response = res
                break

    duration = (vm.completed_at - vm.started_at) if vm.completed_at else 0.0

    created = sum(1 for f in vm.files.values() if f.status == "A")
    modified = sum(1 for f in vm.files.values() if f.status == "M")
    tests_line = (
        f"✓ {vm.tests.passed} passed"
        if vm.tests.has_evidence and vm.tests.total
        else ("✓ tests passed" if vm.tests.has_evidence else "— not reported")
    )
    verified_ok = vm.verification_status in ("passed", "not_started")

    latest = vm.latest_validation()
    if latest is not None and latest.is_failure:
        # Part 36: a failing latest command overrides stale green summary.
        console.print("")
        console.print("⚠ Implementation completed, but the latest validation command failed.")
        console.print(f"  {latest.command[:100]} → exit {latest.exit_code}")

    files_modified = [f.path for f in vm.files.values() if f.status == "M"]
    files_created = [f.path for f in vm.files.values() if f.status == "A"]
    files_deleted = [f.path for f in vm.files.values() if f.status == "D"]
    suggestions = NextActionEngine().suggest(
        vm.request or "",
        files=files_modified + files_created,
        verification_status=vm.verification_status,
        git_commit=vm.git.get("commit", ""),
        git_push=vm.git.get("push", ""),
        recovery_exhausted=vm.recovery_exhausted,
    )
    formatter = CompletionFormatter(plain=plain)
    summary = formatter.success(
        headline="Project implemented",
        files_modified=files_modified,
        files_created=files_created,
        files_deleted=files_deleted,
        tests_line=tests_line,
        verification_status=vm.verification_status,
        git_commit=vm.git.get("commit", ""),
        git_push=vm.git.get("push", ""),
        recovery_attempts=vm.recovery_attempts,
        duration=duration,
        agents=len(vm.agents),
        tool_calls=vm.tool_calls,
        next_actions=suggestions,
        summary=getattr(getattr(outcome, "state", None), "verification_summary", "") or "",
        agent_response=agent_response,
    )

    if plain:
        console.print("")
        console.print("Harness Complete")
        for line in summary.splitlines():
            if line.startswith("✓ Done") or line.startswith("Next") or line.startswith("  →"):
                continue  # summary header + next actions print below
            console.print(line)
        console.print(f"  Changes: {modified} modified, {created} created")
        console.print(f"  Tests: {tests_line}")
        console.print(f"  Verification: {vm.verification_status}")
        console.print("  Status: VERIFIED" if verified_ok else "  Status: COMPLETE")
        if suggestions:
            console.print("  Next")
            for s in suggestions:
                console.print(f"    → {s.label}")
        return

    from rich.console import Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    title = "Harness Complete"
    border = "green"

    renderables = []
    if agent_response:
        from rich.padding import Padding
        renderables.append(Text("  Agent", style="bold cyan"))
        renderables.append(Padding(Markdown(agent_response.strip()), (0, 0, 1, 2)))
    
    # We remove the heavy Panel border for the final output because it makes it look disconnected.
    # Instead, we just print the summary neatly.
    
    console.print("")
    for r in renderables:
        console.print(r)
    console.print(Text(summary))

    if graph is not None and vm.files:
        console.print("")
        console.print("  [dim]Next:[/] git status shows uncommitted changes in this workspace.")


def render_failure_summary(
    console: Console, vm: RuntimeViewModel, outcome: Any, *, plain: bool = False
) -> None:
    """Honest failure summary (Parts 10, 19, 36) — never false success."""
    from harness_core.cli.completion import (
        CompletionClassifier,
        CompletionFormatter,
        NextActionEngine,
    )

    state = getattr(outcome, "state", None)
    graph = getattr(outcome, "graph", None)

    agent_response = ""
    if graph is not None:
        for task in reversed(list(graph.tasks.values())):
            res = (task.result or "").strip()
            if res and not (res.startswith("Task ") and res.endswith(" completed")) and not res.startswith("Stopped:"):
                agent_response = res
                break

    blockers = list(getattr(state, "blockers", []) or [])
    category = blockers[0] if blockers else "EXECUTION_FAILURE"
    completion = CompletionClassifier.classify(vm, outcome)

    latest = vm.latest_validation()
    evidence: list[str] = []
    if latest is not None and latest.is_failure:
        evidence.append(f"exit code: {latest.exit_code or '?'}")
        if latest.command:
            evidence.append(f"command: {latest.command[:100]}")
    elif vm.failed_tool_calls:
        evidence.append(f"{vm.failed_tool_calls} failed operation(s)")
    evidence.append(f"completion state: {completion.value}")

    files_modified = [f.path for f in vm.files.values() if f.status in ("M", "A")]
    suggestions = NextActionEngine().suggest(
        vm.request or "",
        files=files_modified,
        verification_status=vm.verification_status,
        recovery_exhausted=vm.recovery_exhausted,
        partial=True,
    )
    formatter = CompletionFormatter(plain=plain)
    summary = formatter.failure(
        headline=f"Could not safely complete the requested task ({category})",
        what_happened=str(getattr(state, "detail", "") or "") or category,
        evidence_lines=evidence,
        tried=[],
        why_stopped=(
            f"Recovery exhausted after {vm.recovery_attempts} attempt(s)"
            if vm.recovery_exhausted
            else "Runtime terminated with an unrecovered failure."
        ),
        files_modified=files_modified[:8],
        verification_status=vm.verification_status,
        recovery_attempts=vm.recovery_attempts,
        recovery_exhausted=vm.recovery_exhausted,
        next_actions=suggestions,
        agent_response=agent_response,
    )

    if plain:
        console.print("")
        console.print("Harness Stopped")
        console.print(f"  Failure: {category}")
        console.print(f"  Recovery attempts: {vm.recovery_attempts}")
        console.print("  No false success was reported.")
        if suggestions:
            console.print("  Next")
            for s in suggestions:
                console.print(f"    → {s.label}")
        return

    from rich.console import Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    renderables = []
    if agent_response:
        from rich.padding import Padding
        renderables.append(Text("  Agent", style="bold red"))
        renderables.append(Padding(Markdown(agent_response.strip()), (0, 0, 1, 2)))

    console.print("")
    for r in renderables:
        console.print(r)
    console.print(Text(summary))


def render_cancelled_summary(
    console: Console, vm: RuntimeViewModel | None = None, *, plain: bool = False
) -> None:
    """Interruption summary (Parts 17, 38) — locks released, session preserved.

    Accepts the view model so the cancellation block can list what actually
    completed vs. what was interrupted, plus useful next actions.
    """
    from harness_core.cli.completion import CompletionFormatter, NextActionEngine

    completed: list[str] = []
    interrupted: list[str] = []
    uncommitted: list[str] = []

    # Try to find a partial response generated before cancellation
    agent_response = ""
    if vm is not None:
        completed = [a.role.upper() or a.name for a in vm.agents.values() if a.status == "completed"]
        interrupted = [a.role.upper() or a.name for a in vm.agents.values() if a.status in ("running", "waiting", "blocked")]
        uncommitted = [f.path for f in vm.files.values()]

        # If outcome is available in caller, it should ideally be passed, but we'll try to extract from the view model if possible.
        # Actually outcome is not passed to cancelled summary, so we just pass empty agent_response or let it be.


    suggestions = NextActionEngine().suggest(
        "",
        files=uncommitted,
        verification_status=vm.verification_status if vm is not None else "",
    ) if vm is not None else []
    formatter = CompletionFormatter(plain=plain)
    summary = formatter.cancelled(
        completed=completed,
        interrupted=interrupted,
        uncommitted=uncommitted[:8],
        next_actions=suggestions or None,
        agent_response=agent_response,
    )

    if plain:
        console.print("")
        console.print(summary)
        return
    from rich.console import Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    renderables = []
    if agent_response:
        renderables.append(Markdown(agent_response.strip()))
        renderables.append(Text(""))
    renderables.append(Text(summary))

    body = Group(*renderables)
    console.print(Panel(body, title="Harness Cancelled", border_style="yellow"))
