"""Professional live engineering dashboard for the unified runtime (Phase 10).

Architecture (event-driven, never fabricated):

    EventBus  →  RuntimeViewModel  →  ConversationRenderer (preferred)
                                 ↘  LiveTerminalUI (legacy unified path)

``RuntimeViewModel`` is a pure, Rich-free state adapter.  Every field it
exposes is derived from real EventBus events (or from the authoritative
``RuntimeOutcome`` merged in via ``finalize()``).  It never fakes agent
activity, progress, file changes, or success.

Live rendering is now unified via ``conversation.ConversationRenderer``.
``LiveTerminalUI`` is kept for backward compat and for the unified runtime's
throttled dashboard; its helpers delegate to ``ui`` so there is one canonical
elapsed/tool-display implementation (spec §17).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from harness_core.observability.events import Event, EventBus


# ── Small display helpers — delegate to ui so there is one canonical impl ──

def format_elapsed(seconds: float) -> str:
    """Canonical elapsed — delegates to ui.fmt_elapsed (spec §17)."""
    try:
        from harness_core.cli.ui import fmt_elapsed as _fmt
        return _fmt(seconds)
    except Exception:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        if mins < 60:
            return f"{mins}m{secs:02d}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h{mins:02d}m{secs:02d}s"


def tool_display_name(tool_name: str, args: dict[str, Any]) -> str:
    """Concise display — delegates to ui._tool_display (lowercase, test-compat)."""
    try:
        from harness_core.cli.ui import _tool_display as _canon
        return _canon(tool_name, args)
    except Exception:
        pass
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
    """Public status of one agent/task as seen by the dashboard."""

    name: str = ""
    role: str = ""
    task_id: str = ""
    status: str = "queued"  # queued|running|waiting|blocked|completed|failed|cancelled
    operation: str = ""
    detail: str = ""
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
    """One observed validation command result (Phase 10.5, Part 11)."""

    command: str = ""
    exit_code: int | None = None
    passed: int = 0
    total: int = 0
    success: bool = False
    scope: str = "latest"
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
    status: str = "M"
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "status": self.status}


# ── Event-driven view model ────────────────────────────────────────────────


class RuntimeViewModel:
    """State adapter between the EventBus and the terminal UI."""

    _STATUS_FROM_TASK = {
        "task.ready": "queued",
        "task.queued": "queued",
        "task.started": "running",
        "task.completed": "completed",
        "task.failed": "failed",
        "task.cancelled": "cancelled",
    }
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
        self.status: str = "running"
        self.terminal: bool = False
        self.started_at: float = time.time()
        self.completed_at: float = 0.0
        self.agents: dict[str, AgentView] = {}
        self._task_agent: dict[str, str] = {}
        self._agent_task: dict[str, str] = {}
        self._agent_role: dict[str, str] = {}
        self.activity: deque[tuple[str, str, str]] = deque(maxlen=10)
        self.files: dict[str, FileChange] = {}
        self.tests = TestState()
        self.validation_runs: list[ValidationRun] = []
        self.recovery_attempts: int = 0
        self.recovery_max: int = 0
        self.recovery_exhausted: bool = False
        self.recovery_active: bool = False
        self.verification_status: str = "not_started"
        self.verification_summary: str = ""
        self.warnings: list[str] = []
        self.git: dict[str, Any] = {
            "repo_detected": False,
            "branch": "",
            "remote": "",
            "dirty": False,
            "commit": "",
            "push": "",
            "identity_ok": False,
            "operations": [],
        }
        self.tool_calls: int = 0
        self.model_calls: int = 0
        self.failed_tool_calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self._plan_tasks: int = 0
        self._bus: EventBus | None = None

    def attach(self, bus: EventBus) -> None:
        if self._bus is bus:
            return
        self._bus = bus
        bus.on("*", self._on_event)

    def detach(self, bus: EventBus | None = None) -> None:
        target = bus or self._bus
        if target is not None:
            target.off("*", self._on_event)
        self._bus = None

    async def _on_event(self, event: Event) -> None:
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
        if etype in self._STATUS_FROM_TASK:
            task_id = str(data.get("task_id", ""))
            if not task_id:
                return
            new_status = self._STATUS_FROM_TASK[etype]
            agent = self.agents.setdefault(task_id, AgentView(task_id=task_id, name=data.get("agent_id", task_id)))
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
                agent.status = "partial"
                agent.detail = data.get("error", "")[:120] or agent.detail
            if new_status == "completed" and agent.operation:
                agent.operation = "done"
                agent.detail = ""
            if etype == "task.failed":
                self.failed_tool_calls += 0
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
            agent = self.agents.setdefault(task_id, AgentView(task_id=task_id, name=agent_id))
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
        if etype == "tool.call":
            self.tool_calls += 1
            task_id = self._current_task_id(data)
            tool = data.get("tool", "")
            args = data.get("args", {}) or {}
            if task_id:
                agent = self.agents.setdefault(task_id, AgentView(task_id=task_id, name=data.get("agent_id", task_id)))
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
                    self.agents[task_id].detail = str(data.get("error") or data.get("stderr") or status)[:80]
                    self._push_activity(now, self.agents[task_id].name, f"{tool} failed")
            if _looks_like_test_command(tool, data):
                self.tests.failed += 1
                self.tests.last_line = f"{tool} failed (exit {exit_code})"
                self._record_validation_run(ValidationRun(command=str(data.get("command") or ""), exit_code=exit_code, success=False, scope="latest"))
            self._observe_git_tool(tool, data)
            return
        if etype == "test.started":
            self.tests.running += 1
            return
        if etype == "test.completed":
            passed = data.get("passed")
            total = data.get("total")
            success = data.get("success")
            command = str(data.get("command") or "")
            self.tests.running = max(0, self.tests.running - 1)
            ok = bool(success) if success is not None else (bool(passed) and passed == total if passed is not None and total else False)
            run = ValidationRun(command=command, success=ok, scope="latest")
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
        if etype in ("verification.started", "verification_started"):
            self.verification_status = "in_progress"
            return
        if etype in ("verification.completed", "verification_completed"):
            passed = data.get("passed", False)
            self.verification_status = "passed" if passed else "failed"
            self.verification_summary = str(data.get("summary", ""))
            return
        if etype == "routing.decision":
            self.model_calls += 1
            return
        if etype == "model.usage":
            self.input_tokens += int(data.get("input_tokens", 0) or 0)
            self.output_tokens += int(data.get("output_tokens", 0) or 0)
            self.total_tokens += int(data.get("total_tokens", 0) or 0)
            return
        if etype == "model.error":
            self.warnings.append(str(data.get("error", ""))[:120])
            return
        if etype == "model.cooldown":
            model = str(data.get("model", "?"))
            cooldown = data.get("cooldown_seconds", 0)
            category = str(data.get("category", "error"))
            self.activity.append((
                time.strftime("%H:%M:%S"), "router",
                f"{model} {category} · cooldown {cooldown}s"[:90],
            ))
            return
        if etype == "test_integrity.warning":
            self.warnings.append(str(data.get("warning", "test integrity"))[:120])
            return

    def _current_task_id(self, data: dict[str, Any]) -> str:
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

    def _record_validation_run(self, run: ValidationRun) -> None:
        self.validation_runs.append(run)
        if len(self.validation_runs) > 32:
            self.validation_runs = self.validation_runs[-32:]

    def baseline_validation(self) -> ValidationRun | None:
        for run in reversed(self.validation_runs):
            if not run.is_failure:
                return run
        return None

    def latest_validation(self) -> ValidationRun | None:
        return self.validation_runs[-1] if self.validation_runs else None

    def _observe_git_tool(self, tool: str, data: dict[str, Any]) -> None:
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

    def finalize(self, outcome: Any) -> None:
        state = getattr(outcome, "state", None)
        if state is not None:
            self.status = state.status.value
            self.stage = state.stage.value
            self.terminal = state.stage.value in ("succeeded", "failed", "blocked", "cancelled")
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

    def snapshot(self) -> dict[str, Any]:
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
                "baseline": self.baseline_validation().to_dict() if self.baseline_validation() else None,
                "latest": self.latest_validation().to_dict() if self.latest_validation() else None,
                "runs": [r.to_dict() for r in self.validation_runs],
            },
            "git": dict(self.git),
            "recovery": {"attempts": self.recovery_attempts, "max": self.recovery_max, "active": self.recovery_active, "exhausted": self.recovery_exhausted},
            "verification": {"status": self.verification_status, "summary": self.verification_summary},
            "metrics": {"tool_calls": self.tool_calls, "model_calls": self.model_calls, "failed_tool_calls": self.failed_tool_calls, "plan_tasks": self._plan_tasks},
            "warnings": list(self.warnings),
        }


def _file_from_tool(tool: str, args: dict[str, Any]) -> str:
    if tool in ("read_file", "write_file", "edit_file", "delete_file"):
        path = args.get("path", args.get("file_path", ""))
        return str(path) if path else ""
    return ""


_TEST_MARKERS = ("pytest", "npm test", "yarn test", "pnpm test", "bun test", "cargo test", "go test", "jest", "vitest", "mocha", "phpunit", "dotnet test", "gradle test", "mvn test", "python -m pytest")


def _looks_like_test_command(tool: str, data: dict[str, Any]) -> bool:
    if tool != "run_command":
        return False
    status = data.get("status", "")
    if status == "success" or data.get("exit_code") in (None, 0):
        return False
    command = str(data.get("command") or data.get("args", {}).get("command", ""))
    lowered = command.lower()
    return any(marker in lowered for marker in _TEST_MARKERS)


# ── Live terminal UI — now conversation-aligned (spec §15) ────────────────


class LiveTerminalUI:
    """Throttled Rich region for unified runtime.

    Delegates visual style to conversation's compact language while keeping
    the view-model as data. For plain mode it remains a deduped single line.
    """

    STATUS_ICONS = {
        "queued": ("○", "dim"),
        "waiting": ("◌", "yellow"),
        "blocked": ("⊘", "red"),
        "running": ("●", "cyan"),
        "partial": ("⚠", "yellow"),
        "completed": ("✓", "green"),
        "failed": ("✗", "red"),
        "cancelled": ("—", "dim"),
    }

    def __init__(self, console: Console, *, plain: bool = False, refresh_per_second: float = 6.0, max_agents: int = 10) -> None:
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
        self._active = True
        if self.plain:
            self._render_plain(vm)
            return
        from rich.live import Live
        self._live = Live(get_renderable=lambda: self._renderable(vm), console=self.console, refresh_per_second=self.refresh_per_second, transient=False)
        self._live.start()

    def stop(self, vm: RuntimeViewModel) -> None:
        self._active = False
        if self._live is not None:
            try:
                self._live.update(self._renderable(vm))
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def update(self, vm: RuntimeViewModel) -> None:
        if not self._active:
            return
        now = time.time()
        if self.plain:
            self._render_plain(vm)
            return
        if now - self._last_refresh < self._min_interval:
            return
        self._last_refresh = now
        # Live thread handles the rendering automatically via get_renderable

    def _current_operation(self, vm: RuntimeViewModel) -> tuple[str, str]:
        running = [a for a in vm.agents.values() if a.status in ("running", "waiting", "blocked")]
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
        
        if vm.request:
            lines.append(Text.assemble(("> ", "#3b82f6"), (vm.request, "bold white")))
            lines.append(Text(""))
            
        stage_map = {
            "understanding": "Considering…",
            "exploring": "Exploring…",
            "planning": "Planning…",
            "implementing": "Building…",
            "testing": "Testing…",
            "diagnosing": "Diagnosing…",
            "fixing": "Fixing…",
            "verifying": "Verifying…",
            "waiting": "Waiting for model…",
        }
        
        is_waiting = any(a.status == "waiting" for a in vm.agents.values())
        if is_waiting:
            display_stage = stage_map["waiting"]
        else:
            stage = vm.stage.lower() if vm.stage else "understanding"
            display_stage = stage_map.get(stage, "Considering…")
        
        self._frame = getattr(self, "_frame", 0) + 1
        spinner_chars = ["◐", "◓", "◑", "◒"]
        spinner = spinner_chars[self._frame % len(spinner_chars)]
        
        if vm.terminal:
            if vm.status == "failed":
                lines.append(Text.assemble(("✗ ", "red"), ("Failed", "bold white"), (f" · {format_elapsed(time.time() - vm.started_at)}", "dim")))
            elif vm.status == "cancelled":
                lines.append(Text.assemble(("■ ", "yellow"), ("Cancelled", "bold white"), (f" · {format_elapsed(time.time() - vm.started_at)}", "dim")))
            else:
                lines.append(Text.assemble(("✓ ", "green"), ("Completed", "bold white"), (f" · {format_elapsed(time.time() - vm.started_at)}", "dim")))
        else:
            lines.append(Text.assemble((f"{spinner} ", "#3b82f6"), (display_stage, "bold white"), (f"                         {format_elapsed(time.time() - vm.started_at)}", "dim")))
            
        if vm.total_tokens > 0:
            in_k = vm.input_tokens / 1000.0
            out_k = vm.output_tokens / 1000.0
            tot_k = vm.total_tokens / 1000.0
            if vm.terminal:
                lines.append(Text.assemble((f"  {in_k:.1f}k input · {out_k:.1f}k output · {tot_k:.1f}k total", "dim")))
            else:
                lines.append(Text.assemble((f"  {in_k:.1f}k in · {out_k:.1f}k out · {tot_k:.1f}k total", "dim")))
            
        lines.append(Text(""))
        
        activity = self._dedup_activity(list(vm.activity))[-5:]
        if activity:
            for i, (ts, name, text) in enumerate(activity):
                is_last = (i == len(activity) - 1)
                prefix = "└─ " if is_last else "├─ "
                icon = spinner if is_last and not vm.terminal else "✓"
                if vm.terminal and vm.status == "failed" and is_last:
                    icon = "✗"
                color = "cyan" if icon == spinner else ("red" if icon == "✗" else "green")
                
                op_str = name
                if text:
                    op_str = f"{name} {text}"
                    
                lines.append(Text.assemble(("    " + prefix, "dim"), (f"{icon} ", color), (op_str[:70], "dim")))
        
        if vm.terminal and vm.status == "failed" and vm.warnings:
            lines.append(Text(""))
            lines.append(Text.assemble(("    ", ""), (vm.warnings[-1], "red")))
        
        return Group(*lines)

    def _render_plain(self, vm: RuntimeViewModel) -> None:
        line = f"[{vm.stage}] {vm.status} {len(vm.agents)} agents {vm.tool_calls} tools"
        if vm.tests.has_evidence and vm.tests.total:
            line += f" tests:{vm.tests.passed}/{vm.tests.total}"
        if line != self._last_plain_line:
            self._last_plain_line = line
            self.console.print(f"  {line} ({format_elapsed(time.time() - vm.started_at)})", highlight=False, markup=False)


# ── Evidence-based summary panels ──────────────────────────────────────────


def render_startup_panel(console: Console, workspace: str, checks: dict[str, bool], *, plain: bool = False) -> None:
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


def render_success_summary(console: Console, vm: RuntimeViewModel, outcome: Any, *, plain: bool = False) -> None:
    from harness_core.cli.completion import _fmt_elapsed
    graph = getattr(outcome, "graph", None)
    agent_response = ""
    if graph is not None:
        for task in reversed(list(graph.tasks.values())):
            res = (task.result or "").strip()
            if res and not (res.startswith("Task ") and res.endswith(" completed")) and not res.startswith("Stopped:"):
                agent_response = res
                break
    duration = (vm.completed_at - vm.started_at) if vm.completed_at else 0.0

    if plain:
        console.print(f"✓ Completed · {_fmt_elapsed(duration)}")
        if agent_response:
            console.print("")
            console.print(agent_response.strip())
        return

    console.print("")
    console.print(f"    [bold green]✓ Completed[/] · [dim]{_fmt_elapsed(duration)}[/]")
    console.print("")
    if agent_response:
        from rich.markdown import Markdown
        from rich.padding import Padding
        console.print(Padding(Markdown(agent_response.strip()), (0, 0, 1, 4)))


def render_failure_summary(console: Console, vm: RuntimeViewModel, outcome: Any, *, plain: bool = False) -> None:
    from harness_core.cli.completion import _fmt_elapsed
    state = getattr(outcome, "state", None)
    duration = (vm.completed_at - vm.started_at) if vm.completed_at else 0.0
    blockers = list(getattr(state, "blockers", []) or [])
    category = blockers[0] if blockers else "EXECUTION_FAILURE"
    
    if plain:
        console.print(f"✗ Task failed · {_fmt_elapsed(duration)}")
        console.print("")
        console.print(f"{category}")
        return

    console.print("")
    console.print(f"    [bold red]✗ Task failed[/] · [dim]{_fmt_elapsed(duration)}[/]")
    console.print("")
    console.print(f"    {category}")
    console.print("")


def render_cancelled_summary(console: Console, vm: RuntimeViewModel | None = None, *, plain: bool = False) -> None:
    if plain:
        console.print("")
        console.print("⚠ Task paused")
        console.print("")
        console.print("The task was cancelled. Your progress has been preserved.")
        return
        
    console.print("")
    console.print("    [bold yellow]⚠ Task paused[/]")
    console.print("")
    console.print("    [dim]The task was cancelled. Your progress has been preserved.[/]")
    console.print("")
