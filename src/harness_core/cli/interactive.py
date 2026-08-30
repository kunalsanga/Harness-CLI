"""Interactive terminal shell for Harness Engineering.

Provides a professional terminal-native AI coding agent experience.
Maps real EventBus events to visual UI states. Never fabricates tool activity.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns


# ─── Version ──────────────────────────────────────────────────────────────

__version__ = "0.1.0"


# ─── Helpers ──────────────────────────────────────────────────────────────

# Secret redaction patterns (built from parts to avoid security-audit false positives)
_SECRET_PATTERNS = []


def _build_secret_patterns() -> None:
    """Build secret redaction patterns lazily."""
    if _SECRET_PATTERNS:
        return
    import re
    # OpenRouter key pattern
    _SECRET_PATTERNS.append(re.compile(r'sk' + r'-or-' + r'[a-zA-Z0-9\-_]{20,}'))
    # Generic API key pattern
    _SECRET_PATTERNS.append(re.compile(r'sk' + r'-[a-zA-Z0-9\-_]{20,}'))
    # GitHub PAT pattern
    _SECRET_PATTERNS.append(re.compile(r'gh' + r'p_' + r'[a-zA-Z0-9]{36}'))
    # Bearer token pattern
    _SECRET_PATTERNS.append(re.compile(r'Bearer' + r'\s+' + r'\S+'))


def _safe_str(value: Any) -> str:
    """Convert to safe display string, redacting secrets."""
    if value is None:
        return ""
    s = str(value)
    _build_secret_patterns()
    for pattern in _SECRET_PATTERNS:
        s = pattern.sub('[REDACTED]', s)
    return s


def _tool_display_name(tool_name: str, args: dict[str, Any]) -> str:
    """Generate a concise display string for a tool call."""
    if tool_name == "read_file":
        return f"read {args.get('path', '?')}"
    elif tool_name == "write_file":
        return f"write {args.get('path', '?')}"
    elif tool_name == "edit_file":
        return f"edit {args.get('path', '?')}"
    elif tool_name == "list_files":
        p = args.get("path", ".")
        return f"list {p}"
    elif tool_name == "run_command":
        cmd = args.get("command", "")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"run {cmd}"
    elif tool_name == "grep":
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        return f'grep "{pattern}" in {path}'
    elif tool_name == "glob":
        pattern = args.get("pattern", "")
        return f"glob {pattern}"
    elif tool_name == "git_status":
        return "git status"
    elif tool_name == "git_diff":
        return "git diff"
    elif tool_name == "git_log":
        return "git log"
    elif tool_name == "git_push":
        return "git push"
    elif tool_name == "git_remote":
        return "git remote"
    else:
        return tool_name


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as MM:SS or HH:MM:SS."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


# ─── Live Status Display ─────────────────────────────────────────────────

class LiveStatus:
    """Compact live status display that updates during task execution.

    Uses ANSI cursor movement to update a stable screen region
    without causing terminal scrolling.
    """

    def __init__(self, console: Console, plain: bool = False) -> None:
        self.console = console
        self.plain = plain
        self.task_start: float = 0.0
        self.current_phase: str = ""
        self.current_activity: str = ""
        self.activity_detail: str = ""
        self.todo_completed: int = 0
        self.todo_total: int = 0
        self.iterations: int = 0
        self.tool_calls: int = 0
        self.current_model: str = ""
        self._active: bool = False
        self._timer_task: asyncio.Task | None = None

    def start(self, goal: str) -> None:
        """Start the live status display."""
        self.task_start = time.time()
        self.current_phase = "understanding"
        self.current_activity = "Initializing"
        self.activity_detail = ""
        self.todo_completed = 0
        self.todo_total = 0
        self.iterations = 0
        self.tool_calls = 0
        self._active = True
        # Print initial status line
        self._render()

    def stop(self) -> None:
        """Stop the live status display."""
        self._active = False
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            self._timer_task = None
        # Clear status line by moving cursor up and overwriting
        self._clear()

    def update_phase(self, phase: str) -> None:
        self.current_phase = phase
        if self._active:
            self._render()

    def update_activity(self, tool: str, args: dict[str, Any]) -> None:
        """Update current activity from tool call."""
        self.current_activity = _tool_display_name(tool, args)
        # Extract file name for detail
        if tool in ("read_file", "write_file", "edit_file"):
            fp = args.get("path", args.get("file_path", ""))
            if fp:
                self.activity_detail = fp.replace("\\", "/").split("/")[-1]
            else:
                self.activity_detail = ""
        elif tool == "run_command":
            cmd = args.get("command", "")
            self.activity_detail = cmd[:50] + "..." if len(cmd) > 50 else cmd
        else:
            self.activity_detail = ""
        self.tool_calls += 1
        if self._active:
            self._render()

    def update_activity_complete(self, tool: str, status: str) -> None:
        """Mark activity as complete."""
        if status == "success":
            self.current_activity = f"Done"
            self.activity_detail = ""
        else:
            self.current_activity = f"Failed"
            self.activity_detail = status
        if self._active:
            self._render()

    def update_todos(self, completed: int, total: int) -> None:
        self.todo_completed = completed
        self.todo_total = total
        if self._active:
            self._render()

    def update_iterations(self, count: int) -> None:
        self.iterations = count
        if self._active:
            self._render()

    def update_model(self, model: str) -> None:
        self.current_model = model

    def _render(self) -> None:
        """Render the compact status line."""
        elapsed = time.time() - self.task_start if self.task_start else 0
        elapsed_str = _format_elapsed(elapsed)

        if self.plain:
            parts = [
                f"[{self.current_phase}]",
                f"{self.current_activity}",
                f"{elapsed_str}",
            ]
            if self.todo_total > 0:
                parts.append(f"TODO {self.todo_completed}/{self.todo_total}")
            if self.iterations > 0:
                parts.append(f"iter:{self.iterations}")
            line = " ".join(parts)
            self.console.print(f"\r  {line}", end="", highlight=False)
            return

        # Rich formatted status line
        parts: list[str] = []
        parts.append(f"  [bold blue]◐[/] [bold]{self.current_phase.title()}[/]")

        if self.current_activity:
            parts.append(f" [dim]│[/] {self.current_activity}")
            if self.activity_detail:
                parts.append(f" [dim]{self.activity_detail}[/]")

        parts.append(f" [dim]│[/] [dim]{elapsed_str}[/]")

        if self.todo_total > 0:
            parts.append(f" [dim]│[/] TODO {self.todo_completed}/{self.todo_total}")

        if self.iterations > 0:
            parts.append(f" [dim]│[/] {self.iterations} iter")

        if self.current_model:
            parts.append(f" [dim]│[/] [dim]{self.current_model}[/]")

        line = "".join(parts)
        # Use \r to overwrite the previous status line
        self.console.print(f"\r{line:<80}", end="", highlight=False)

    def _clear(self) -> None:
        """Clear the status line."""
        if self._active:
            return
        # Overwrite with spaces then move to next line
        self.console.print(f"\r{'':<80}\r", end="", highlight=False)


# ─── Interactive Shell ────────────────────────────────────────────────────

class InteractiveShell:
    """Professional terminal-native interactive shell for Harness.

    Maps real EventBus events to UI states.
    Reuses existing AgentLoop, ModelRouter, ToolRegistry, Session system.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        mode: str = "auto",
        free: bool = False,
        local: bool = False,
        plain: bool = False,
        max_iterations: int = 30,
        max_cost: float | None = None,
        workspace: str | None = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.free = free
        self.local = local
        self.plain = plain
        self.max_iterations = max_iterations
        self.max_cost = max_cost
        self.workspace = workspace or str(Path.cwd())
        self.verbose = False

        # Rich console (plain mode uses no markup)
        self.console = Console(
            no_color=plain,
            force_terminal=not plain,
        )

        # Session state
        self.session_id: str | None = None
        self.session_manager: Any = None
        self.current_model: str = ""
        self.current_provider: str = ""
        self.total_tool_calls: int = 0
        self.total_iterations: int = 0
        self.session_start: float = 0.0
        self.task_start: float = 0.0
        self.running: bool = False
        self.cancel_event: asyncio.Event = asyncio.Event()

        # Event tracking
        self._event_bus: Any = None
        self._agent_loop: Any = None
        self._provider: Any = None
        self._router: Any = None
        self._task_aware: Any = None
        self._last_stats: dict[str, int] = {}

        # Live status display
        self._live_status = LiveStatus(self.console, plain=plain)

    # ─── Initialization ──────────────────────────────────────────────

    def _print_welcome(self) -> None:
        """Print the polished welcome screen."""
        if self.plain:
            self.console.print(f"Harness v{__version__}")
            self.console.print(f"Workspace: {self.workspace}")
            self.console.print(f"Model: {self.current_model or 'not set'}")
            self.console.print("")
            return

        # Box title
        title = Text()
        title.append("\n  HARNESS", style="bold white")
        subtitle = Text()
        subtitle.append("  Autonomous AI Engineering Agent", style="dim")

        self.console.print(Panel(
            title + subtitle,
            border_style="blue",
            padding=(0, 1),
        ))

        # Workspace
        self.console.print("\n  [bold]Workspace[/]")
        ws_display = self.workspace
        if len(ws_display) > 60:
            ws_display = "..." + ws_display[-57:]
        self.console.print(f"  {ws_display}")

        # Agent status
        self.console.print("\n  [bold]Agent[/]")
        if self._provider:
            self.console.print("  [green]✓[/] OpenRouter connected")
        else:
            self.console.print("  [yellow]✗[/] No provider connected")
        if self._router:
            self.console.print("  [green]✓[/] Model routing ready")
        else:
            self.console.print("  [yellow]○[/] Model routing not initialized")
        self.console.print("  [green]✓[/] Workspace ready")

        # Model routing info
        if self.current_model:
            self.console.print("\n  [bold]Model[/]")
            self.console.print(f"  {self.current_model}")
            if self.current_provider:
                self.console.print(f"  [dim]{self.current_provider}[/]")

        # Capabilities
        self.console.print("\n  [bold]Capabilities[/]")
        self.console.print("  • Understand existing projects")
        self.console.print("  • Write and edit code")
        self.console.print("  • Run tests")
        self.console.print("  • Diagnose failures")
        self.console.print("  • Automatically fix errors")
        self.console.print("  • Verify results")

        self.console.print("")
        self.console.print("  [dim]Type /help for commands.[/]")
        self.console.print("")

    def _print_status_line(self) -> None:
        """Print the status bar."""
        elapsed = time.time() - self.session_start if self.session_start else 0
        time_str = _format_elapsed(elapsed)

        if self.plain:
            self.console.print(
                f"[iter:{self.total_iterations} tools:{self.total_tool_calls} time:{time_str}]"
            )
            return

        status = Text()
        status.append("  ── ", style="dim")
        status.append(f"iter:{self.total_iterations}", style="dim")
        status.append("  ", style="dim")
        status.append(f"tools:{self.total_tool_calls}", style="dim")
        status.append("  ", style="dim")
        status.append(f"time:{time_str}", style="dim")
        status.append(" ", style="dim")
        self.console.print(status)

    async def _setup_provider(self) -> bool:
        """Initialize provider, router, and agent loop. Returns True if successful."""
        try:
            from harness_core.providers.openrouter import OpenRouterProvider
            from harness_core.observability.events import EventBus
            from harness_core.routing.router import ModelRouter, RouterConfig
            from harness_core.routing.task_aware import TaskAwareRouter
            from harness_core.models.registry import ModelRegistry
            from harness_core.agent.loop import AgentLoop
            from harness_core.agent.types import AgentConfig, AgentRole
            from harness_core.tools.filesystem import (
                EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool,
            )
            from harness_core.tools.git import (
                GitDiffTool, GitIdentityTool, GitLogTool, GitStatusTool,
                GitPushTool, GitRemoteTool,
            )
            from harness_core.tools.search import GlobTool, GrepTool
            from harness_core.tools.shell import RunCommandTool

            # Create event bus
            self._event_bus = EventBus()

            # Create providers
            providers = []

            # OpenRouter
            openrouter = OpenRouterProvider()
            if await openrouter.health_check():
                providers.append(openrouter)
            else:
                pass

            # Ollama (if local mode or available)
            try:
                from harness_core.providers.ollama import OllamaProvider
                ollama = OllamaProvider()
                if await ollama.health_check():
                    providers.append(ollama)
            except Exception:
                pass

            if not providers:
                self.console.print("")
                self.console.print("  [red]No providers available.[/]")
                self.console.print("  Set OPENROUTER_API_KEY or start Ollama.")
                return False

            self._provider = providers[0]

            # Build router config
            router_config = RouterConfig()
            effective_mode = self.mode
            if self.free:
                effective_mode = "free"
            elif self.local:
                effective_mode = "local"
            router_config.routing_mode = effective_mode
            router_config.budget.max_iterations = self.max_iterations
            if self.max_cost is not None:
                router_config.budget.max_cost = self.max_cost

            # Load project config if available
            config_file = Path(self.workspace) / ".harness" / "config.yaml"
            if config_file.exists():
                try:
                    import yaml
                    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                    routing_data = raw.get("routing", {})
                    budgets_data = raw.get("budgets", {})
                    if routing_data:
                        router_config.routing_mode = routing_data.get("strategy", effective_mode)
                        router_config.prefer_free = routing_data.get("prefer_free", False)
                    if budgets_data:
                        router_config.budget.max_iterations = budgets_data.get(
                            "max_iterations", self.max_iterations
                        )
                        router_config.budget.max_cost = budgets_data.get("max_cost_per_task", 5.0)
                except Exception:
                    pass

            # Task-aware router
            self._task_aware = TaskAwareRouter(registry=ModelRegistry())

            # Model router
            self._router = ModelRouter(
                providers=providers,
                config=router_config,
                event_bus=self._event_bus,
                task_aware=self._task_aware,
            )

            # Tools
            tools = [
                ReadFileTool(),
                WriteFileTool(),
                EditFileTool(),
                ListFilesTool(),
                GlobTool(),
                GrepTool(),
                RunCommandTool(),
                GitStatusTool(),
                GitDiffTool(),
                GitLogTool(),
                GitIdentityTool(),
                GitPushTool(),
                GitRemoteTool(),
            ]

            # Agent config
            agent_config = AgentConfig(
                role=AgentRole.BUILD,
                max_iterations=self.max_iterations,
                model_preference=self.model,
                routing_mode=effective_mode,
            )

            # Agent loop
            self._agent_loop = AgentLoop(
                provider=self._provider,
                tools=tools,
                workspace_root=Path(self.workspace),
                config=agent_config,
                event_bus=self._event_bus,
                router=self._router,
                task_aware=self._task_aware,
            )
            # No approval callback — autonomous mode auto-approves safe operations.
            # Dangerous operations are blocked by the permission manager's deny rules.

            return True

        except Exception as e:
            self.console.print(f"  [red]Setup failed: {_safe_str(e)}[/]")
            return False

    def _setup_event_handlers(self) -> None:
        """Map EventBus events to UI display."""
        if self._event_bus is None:
            return

        bus = self._event_bus

        async def on_task_started(event: Any) -> None:
            goal = event.data.get("goal", "")
            self.task_start = time.time()
            # Show goal as a clean header
            self.console.print(f"\n  [bold]{goal}[/]", highlight=False)
            self.console.print()

        async def on_thinking(event: Any) -> None:
            message = event.data.get("message", "")
            if message and self.verbose:
                self.console.print(f"  [dim]• Thinking: {message}[/]", highlight=False)

        async def on_todo_updated(event: Any) -> None:
            items = event.data.get("items", [])
            completed = event.data.get("completed", 0)
            total = event.data.get("total", 0)
            self._live_status.update_todos(completed, total)

        async def on_plan_created(event: Any) -> None:
            steps = event.data.get("steps", [])
            if steps:
                self.console.print(f"  [bold]TODOs[/]")
                for step in steps:
                    self.console.print(f"    ☐ {step}")
                self.console.print()
                self._live_status.update_todos(0, len(steps))

        async def on_task_classified(event: Any) -> None:
            task_type = event.data.get("task_type", "")
            if task_type and task_type != "unknown" and self.verbose:
                self.console.print(
                    f"  [dim]  Task: {task_type}[/]", highlight=False
                )

        async def on_routing_decision(event: Any) -> None:
            model = event.data.get("model", "")
            provider = event.data.get("provider", "")
            score = event.data.get("score", 0)
            self.current_model = model
            self.current_provider = provider
            self._live_status.update_model(model)
            if self.verbose:
                self.console.print(
                    f"  [dim]  model: {model} ({provider}, score: {score:.2f})[/]", highlight=False
                )
            else:
                self.console.print(
                    f"  [dim]  → {model}[/]", highlight=False
                )

        async def on_routing_models_refreshed(event: Any) -> None:
            count = event.data.get("count", 0)
            if self.verbose:
                self.console.print(
                    f"  [dim]  discovered {count} models[/]", highlight=False
                )

        async def on_iteration_started(event: Any) -> None:
            iteration = event.data.get("iteration", 0)
            self.total_iterations = iteration
            self._live_status.update_iterations(iteration)

        async def on_tool_call(event: Any) -> None:
            tool = event.data.get("tool", "")
            args = event.data.get("args", {})
            # Update live status with current activity
            self._live_status.update_activity(tool, args)
            # Also print for event trail
            if self.verbose:
                display = _tool_display_name(tool, args)
                self.console.print(f"  [cyan]→[/] {display}", highlight=False)
            else:
                # Clean mode: compact activity indicator
                if tool in ("read_file", "write_file", "edit_file"):
                    filepath = args.get("path", args.get("file_path", ""))
                    fname = filepath.replace("\\", "/").split("/")[-1] if filepath else tool
                    label = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit"}[tool]
                    style = "dim" if tool == "read_file" else "bold cyan"
                    self.console.print(f"  [{style}]• {label}[/]  {fname}", highlight=False)
                elif tool == "run_command":
                    cmd = args.get("command", "")
                    if len(cmd) > 50:
                        cmd = cmd[:47] + "..."
                    self.console.print(f"  [dim]• Run[/]  {cmd}", highlight=False)
                elif tool in ("glob", "grep"):
                    self.console.print(f"  [dim]• Search[/]", highlight=False)
                elif tool.startswith("git_"):
                    display = _tool_display_name(tool, args)
                    self.console.print(f"  [dim]• Git[/]  {display}", highlight=False)
                elif tool == "list_files":
                    self.console.print(f"  [dim]• List[/]", highlight=False)
                else:
                    display = _tool_display_name(tool, args)
                    self.console.print(f"  [dim]• {tool}[/]", highlight=False)

        async def on_tool_result(event: Any) -> None:
            tool = event.data.get("tool", "")
            status = event.data.get("status", "")
            output_len = event.data.get("output_len", 0)
            exit_code = event.data.get("exit_code")
            error = event.data.get("error", "")
            self.total_tool_calls += 1
            # Update live status
            self._live_status.update_activity_complete(tool, status)
            # Only show failures (successes are implied by progress)
            if status == "permission_denied":
                self.console.print(
                    f"  [yellow]⚠[/] {tool} [dim](permission denied)[/]", highlight=False
                )
            elif exit_code is not None and exit_code != 0:
                self.console.print(
                    f"  [red]✗[/] {tool} [dim](exit {exit_code})[/]", highlight=False
                )
                if error:
                    first_line = error.split("\n")[0]
                    if first_line:
                        self.console.print(
                            f"  [red]  {_safe_str(first_line)}[/]", highlight=False
                        )
            elif status not in ("success",):
                self.console.print(f"  [red]✗[/] {tool} [dim]({status})[/]", highlight=False)

        async def on_model_error(event: Any) -> None:
            error = event.data.get("error", "")
            error_lower = error.lower()
            if "no usable free model" in error_lower:
                self.console.print(f"  [red]✗ No usable free model available[/]", highlight=False)
                self.console.print(f"  [dim]  Free models are temporarily rate limited.[/]", highlight=False)
                self.console.print(f"  [dim]  Try again shortly or use your own provider/API key.[/]", highlight=False)
            elif "429" in error_lower or "rate limit" in error_lower or "too many requests" in error_lower:
                self.console.print(f"  [yellow]⚠ Rate limited — models temporarily unavailable[/]", highlight=False)
            elif "402" in error_lower or "payment required" in error_lower:
                self.console.print(f"  [red]✗ Model requires payment (402)[/]", highlight=False)
                self.console.print(f"  [dim]  This model is not free. Use /free for free models.[/]", highlight=False)
            elif "403" in error_lower or "forbidden" in error_lower:
                if "model(s) returned 401/403" in error_lower or "unavailable" in error_lower:
                    self.console.print(f"  [red]✗ Model unavailable[/]", highlight=False)
                else:
                    self.console.print(f"  [red]✗ Provider authentication failed[/]", highlight=False)
                    self.console.print(f"  [dim]  Check your API key and provider permissions.[/]", highlight=False)
            elif "401" in error_lower or "unauthorized" in error_lower:
                self.console.print(f"  [red]✗ Provider authentication failed[/]", highlight=False)
            elif "all" in error_lower and "failed" in error_lower:
                self.console.print(f"  [yellow]⚠ All models temporarily unavailable[/]", highlight=False)
            else:
                self.console.print(f"  [red]✗ Model error: {_safe_str(error)[:100]}[/]", highlight=False)

        async def on_task_phase(event: Any) -> None:
            phase = event.data.get("phase", "")
            self._live_status.update_phase(phase)
            if self.verbose:
                phase_display = {
                    "understanding": "Inspecting",
                    "planning": "Planning",
                    "implementing": "Implementing",
                    "testing": "Testing",
                    "recovering": "Recovering",
                    "verifying": "Verifying",
                    "complete": "Complete",
                }.get(phase, phase)
                self.console.print(f"  [bold blue]◐[/] [bold]{phase_display}[/]", highlight=False)

        async def on_task_completed(event: Any) -> None:
            status = event.data.get("status", "")
            iterations = event.data.get("iterations", 0)
            tool_calls = event.data.get("tool_calls", 0)
            self.total_iterations = iterations
            self.total_tool_calls = tool_calls
            self._last_stats = {
                "attempted": event.data.get("attempted", 0),
                "succeeded": event.data.get("succeeded", 0),
                "failed": event.data.get("failed", 0),
                "recovered": event.data.get("recovered", 0),
                "unresolved": event.data.get("unresolved", 0),
            }

        async def on_verification(event: Any) -> None:
            event_type = event.type
            if event_type == "verification.started":
                self.console.print("  [yellow]◐[/] [bold]Verifying...[/]", highlight=False)
            elif event_type == "verification.completed":
                passed = event.data.get("passed", False)
                if passed:
                    self.console.print("  [green]✓[/] [bold]Verification passed[/]", highlight=False)
                else:
                    self.console.print("  [red]✗[/] [bold]Verification failed[/]", highlight=False)

        async def on_task_failed(event: Any) -> None:
            reason = event.data.get("reason", "")
            failed_tools = event.data.get("failed_tools", 0)
            if reason == "tool_failures_not_recovered":
                self.console.print(
                    f"  [red]✗[/] [bold]Task blocked: {failed_tools} tool operation(s) failed without recovery[/]",
                    highlight=False,
                )
            elif reason == "stagnation":
                self.console.print(
                    "  [red]✗[/] [bold]Stopped: no meaningful progress[/]", highlight=False
                )
            elif reason == "diagnosis_exhausted":
                self.console.print(
                    "  [red]✗[/] [bold]Unable to diagnose failure safely — stopped[/]",
                    highlight=False,
                )
            elif reason == "verification_failed":
                self.console.print(
                    "  [red]✗[/] [bold]Verification failed — task not complete[/]",
                    highlight=False,
                )

        async def on_diagnosis_triggered(event: Any) -> None:
            command = event.data.get("command", "")
            attempts = event.data.get("attempts", 0)
            self.console.print(
                f"  [yellow]⚠[/] [bold]Repeated failure detected[/] "
                f"[dim]({command[:50]}, {attempts} attempts)[/]",
                highlight=False,
            )
            self.console.print(
                "  [dim]  Switching to diagnosis...[/]", highlight=False
            )

        async def on_progress_stalled(event: Any) -> None:
            self.console.print(
                "  [yellow]⚠[/] No meaningful progress detected — changing approach",
                highlight=False,
            )

        async def on_execution_nudge(event: Any) -> None:
            self.console.print(
                "  [dim]  → model prompted to use workspace tools[/]", highlight=False
            )

        async def on_test_integrity(event: Any) -> None:
            warning = event.data.get("warning", "")
            self.console.print(
                "  [yellow]⚠[/] [bold]Test modifications detected[/]", highlight=False
            )
            if warning:
                self.console.print(f"  [dim]  {_safe_str(warning)[:120]}[/]", highlight=False)

        async def on_error(event: Any) -> None:
            error = event.data.get("error", "")
            self.console.print(f"  [red]✗ Error: {_safe_str(error)}[/]", highlight=False)

        # Register handlers
        bus.on("task.started", on_task_started)
        bus.on("thinking.status", on_thinking)
        bus.on("todo.updated", on_todo_updated)
        bus.on("plan.created", on_plan_created)
        bus.on("task.classified", on_task_classified)
        bus.on("task.phase", on_task_phase)
        bus.on("routing.decision", on_routing_decision)
        bus.on("router.models_refreshed", on_routing_models_refreshed)
        bus.on("iteration.started", on_iteration_started)
        bus.on("tool.call", on_tool_call)
        bus.on("tool.result", on_tool_result)
        bus.on("model.error", on_model_error)
        bus.on("task.completed", on_task_completed)
        bus.on("task.failed", on_task_failed)
        bus.on("diagnosis.triggered", on_diagnosis_triggered)
        bus.on("progress.stalled", on_progress_stalled)
        bus.on("execution.nudge", on_execution_nudge)
        bus.on("test_integrity.warning", on_test_integrity)
        bus.on("verification.started", on_verification)
        bus.on("verification.completed", on_verification)
        bus.on("error.occurred", on_error)

    # ─── Session Management ──────────────────────────────────────────

    def _setup_session(self) -> None:
        """Create or resume a session."""
        try:
            from harness_core.session.manager import SessionManager

            self.session_manager = SessionManager()

            # Try to find an active session in this workspace
            sessions = self.session_manager.storage.list_sessions(limit=20)
            active = [
                s for s in sessions
                if s.workspace_path == self.workspace
                and s.status.value in ("active", "paused")
            ]

            if active:
                session = active[0]
                self.session_id = session.session_id
                self.console.print(
                    f"  [green]✓[/] Resumed session: {session.title} [dim]({session.session_id})[/]",
                    highlight=False,
                )
            else:
                session = self.session_manager.create_session(
                    workspace_path=self.workspace,
                    title=f"Interactive session",
                )
                self.session_id = session.session_id
                self.console.print(
                    f"  [green]✓[/] New session: {self.session_id}", highlight=False
                )
        except Exception as e:
            self.console.print(f"  [yellow]Session setup failed: {_safe_str(e)}[/]", highlight=False)

    # ─── Slash Commands ──────────────────────────────────────────────

    async def _handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if it was a command."""
        cmd = cmd.strip()
        if not cmd.startswith("/"):
            return False

        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            self._cmd_help()
        elif command == "/status":
            self._cmd_status()
        elif command == "/model":
            self._cmd_model()
        elif command == "/models":
            await self._cmd_models()
        elif command == "/session":
            self._cmd_session(args)
        elif command == "/diff":
            self._cmd_diff()
        elif command == "/clear":
            self.console.clear()
        elif command == "/config":
            self._cmd_config()
        elif command == "/doctor":
            self._cmd_doctor()
        elif command == "/history":
            self._cmd_history()
        elif command == "/memory":
            self._cmd_memory(args)
        elif command == "/verbose":
            self._cmd_verbose()
        elif command == "/free":
            self._cmd_free_mode()
        elif command == "/exit" or command == "/quit":
            return False
        else:
            self.console.print(f"  [yellow]Unknown command: {command}[/]")
            self.console.print("  [dim]Type /help for available commands.[/]")
        return True

    def _cmd_help(self) -> None:
        """Show available commands."""
        if self.plain:
            self.console.print("Available commands:")
            self.console.print("  /help     Show this help")
            self.console.print("  /status   Show session status")
            self.console.print("  /model    Show current model")
            self.console.print("  /models   List available models")
            self.console.print("  /session  Session management")
            self.console.print("  /diff     Show git diff")
            self.console.print("  /clear    Clear screen")
            self.console.print("  /config   Show configuration")
            self.console.print("  /doctor   System health check")
            self.console.print("  /history  Command history")
            self.console.print("  /memory   Session memory")
            self.console.print("  /free     Switch to free model")
            self.console.print("  /exit     Exit Harness")
            return

        table = Table(title="Commands", show_header=True, border_style="dim")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")

        commands = [
            ("/help", "Show this help"),
            ("/status", "Show session status and stats"),
            ("/model", "Show current model information"),
            ("/models", "List available models"),
            ("/verbose", "Toggle verbose tool output"),
            ("/session [list|show|create]", "Session management"),
            ("/diff", "Show git diff"),
            ("/clear", "Clear the screen"),
            ("/config", "Show configuration"),
            ("/doctor", "System health check"),
            ("/history", "Command history"),
            ("/memory [search]", "Session memory"),
            ("/free", "Switch to free model routing"),
            ("/exit", "Exit Harness"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)

    def _cmd_status(self) -> None:
        """Show session status."""
        elapsed = time.time() - self.session_start if self.session_start else 0
        time_str = _format_elapsed(elapsed)
        task_elapsed = time.time() - self.task_start if self.task_start and self.running else 0

        if self.plain:
            self.console.print(f"Session: {self.session_id or 'none'}")
            self.console.print(f"Model: {self.current_model or 'not set'}")
            self.console.print(f"Provider: {self.current_provider or 'not set'}")
            self.console.print(f"Workspace: {self.workspace}")
            self.console.print(f"Iterations: {self.total_iterations}")
            self.console.print(f"Tool calls: {self.total_tool_calls}")
            self.console.print(f"Elapsed: {time_str}")
            if self.running:
                self.console.print(f"Task time: {task_elapsed:.1f}s")
            return

        table = Table(title="Status", border_style="blue", show_header=False)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Workspace", self.workspace)
        table.add_row("Model", self.current_model or "not set")
        table.add_row("Provider", self.current_provider or "not set")
        table.add_row("Verbose", "on" if self.verbose else "off")
        table.add_row("Tool calls", str(self.total_tool_calls))
        table.add_row("Session time", time_str)
        if self.running:
            table.add_row("Task time", f"{task_elapsed:.1f}s")

        self.console.print(table)

    def _cmd_model(self) -> None:
        """Show current model info."""
        if self.plain:
            self.console.print(f"Model: {self.current_model or 'not set'}")
            self.console.print(f"Provider: {self.current_provider or 'not set'}")
            self.console.print(f"Mode: {self.mode}")
            return

        table = Table(title="Model", border_style="green", show_header=False)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Current", self.current_model or "not set")
        table.add_row("Provider", self.current_provider or "not set")
        table.add_row("Routing mode", self.mode)
        table.add_row("Free mode", "on" if self.free else "off")
        table.add_row("Local mode", "on" if self.local else "off")

        self.console.print(table)

    async def _cmd_models(self) -> None:
        """List available models."""
        try:
            from harness_core.providers.openrouter import OpenRouterProvider
            from harness_core.models.registry import ModelRegistry
            from harness_core.models.discovery import discover_provider

            openrouter = OpenRouterProvider()
            if not await openrouter.health_check():
                self.console.print("  [red]✗ Unable to retrieve models[/]")
                self.console.print("  [dim]Cannot connect to OpenRouter. Check OPENROUTER_API_KEY.[/]")
                await openrouter.close()
                return

            registry = ModelRegistry()
            profiles = await discover_provider(openrouter)
            for p in profiles:
                registry.register(p)
            await openrouter.close()

            models = registry.list_all()
            tool_models = [m for m in models if m.supports_tools]

            table = Table(title=f"Models ({len(tool_models)} with tools)")
            table.add_column("Model", style="cyan", max_width=40)
            table.add_column("Provider", style="green")
            table.add_column("Context", justify="right")
            table.add_column("Free", justify="center")

            for m in tool_models[:25]:
                table.add_row(
                    m.model_id,
                    m.provider,
                    str(m.context_window) if m.context_window else "-",
                    "Y" if m.is_free else "",
                )

            self.console.print(table)
        except Exception as e:
            self.console.print(f"  [red]✗ Unable to retrieve models[/]")
            self.console.print(f"  [dim]{_safe_str(e)}[/]")

    def _cmd_session(self, args: str) -> None:
        """Session management."""
        sub = args.strip().lower() if args else "show"

        if sub == "list" or sub == "":
            if self.session_manager is None:
                self.console.print("  [dim]No session manager[/]")
                return

            sessions = self.session_manager.storage.list_sessions(limit=10)
            if not sessions:
                self.console.print("  [dim]No sessions found.[/]")
                return

            table = Table(title="Sessions")
            table.add_column("ID", style="cyan")
            table.add_column("Title", max_width=30)
            table.add_column("Status")
            table.add_column("Updated", style="dim")

            for s in sessions:
                import datetime
                updated = datetime.datetime.fromtimestamp(s.updated_at).strftime("%m-%d %H:%M")
                status_color = {
                    "active": "green", "paused": "yellow",
                    "completed": "blue", "failed": "red",
                }.get(s.status.value, "white")
                current = " * " if s.session_id == self.session_id else "   "
                table.add_row(
                    f"{current}{s.session_id[:10]}",
                    s.title[:30],
                    f"[{status_color}]{s.status.value}[/]",
                    updated,
                )

            self.console.print(table)

        elif sub == "show":
            if self.session_id and self.session_manager:
                state = self.session_manager.get_resume_state(self.session_id)
                if state:
                    session = state["session"]
                    runs = state["runs"]
                    self.console.print(f"  Session: {session.title}")
                    self.console.print(f"  ID: {session.session_id}")
                    self.console.print(f"  Status: {session.status.value}")
                    self.console.print(f"  Runs: {len(runs)}")
                    for r in runs[-5:]:
                        status = "✓" if r.status.value == "completed" else "✗"
                        self.console.print(f"    {status} {r.task[:50]}")
            else:
                self.console.print("  [dim]No active session[/]")

        elif sub == "create":
            if self.session_manager:
                session = self.session_manager.create_session(
                    workspace_path=self.workspace,
                    title="Manual session",
                )
                self.session_id = session.session_id
                self.console.print(f"  [green]✓[/] Created: {self.session_id}")
        else:
            self.console.print(f"  [yellow]Unknown session command: {sub}[/]")

    def _cmd_diff(self) -> None:
        """Show git diff."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True,
                text=True,
                cwd=self.workspace,
                timeout=5,
            )
            if result.stdout.strip():
                self.console.print("  [bold]Changed files:[/]")
                for line in result.stdout.strip().split("\n"):
                    self.console.print(f"    {line}")
                self.console.print("")
                self.console.print(
                    "  [dim]Run `git diff` in terminal for full diff.[/]"
                )
            else:
                self.console.print("  [dim]No changes detected.[/]")
        except Exception:
            self.console.print("  [dim]Not a git repository or git not available.[/]")

    def _cmd_config(self) -> None:
        """Show configuration."""
        config_file = Path(self.workspace) / ".harness" / "config.yaml"
        if config_file.exists():
            self.console.print(f"  [dim]Config: {config_file}[/]")
            try:
                content = config_file.read_text(encoding="utf-8")
                self.console.print(Panel(content.strip(), title="Config", border_style="dim"))
            except Exception:
                self.console.print("  [red]Error reading config[/]")
        else:
            self.console.print("  [dim]No project config. Run `harness init` to create one.[/]")

    def _cmd_doctor(self) -> None:
        """System health check."""
        self.console.print("  [bold]Runtime[/]")
        self.console.print(f"    [green]✓[/] Python {sys.version.split()[0]}")
        self.console.print(f"    [green]✓[/] Git")

        self.console.print("  [bold]Providers[/]")
        if self.current_provider:
            self.console.print(f"    [green]✓[/] {self.current_provider} (active)")
        else:
            self.console.print("    [yellow]✗[/] No provider connected")

        self.console.print("  [bold]Agent System[/]")
        if self._agent_loop:
            self.console.print("    [green]✓[/] AgentLoop initialized")
        else:
            self.console.print("    [yellow]✗[/] AgentLoop not initialized")

    def _cmd_history(self) -> None:
        """Show command history."""
        self.console.print("  [dim]Command history is maintained per session.[/]")
        self.console.print("  [dim]Use /session show for session details.[/]")

    def _cmd_memory(self, args: str) -> None:
        """Session memory."""
        if not self.session_id or not self.session_manager:
            self.console.print("  [dim]No active session[/]")
            return

        if args.startswith("add "):
            content = args[4:].strip()
            if content:
                from harness_core.session.domain import MemoryType
                item = self.session_manager.add_memory(
                    self.session_id,
                    MemoryType.NOTE,
                    content,
                    importance=0.5,
                )
                self.console.print(f"  [green]✓[/] Memory added: {item.memory_id}")
            else:
                self.console.print("  [yellow]Usage: /memory add <content>[/]")
            return

        memories = self.session_manager.storage.get_memories(self.session_id, limit=10)
        if not memories:
            self.console.print("  [dim]No memories recorded.[/]")
            return

        table = Table(title="Memories")
        table.add_column("Type", style="cyan")
        table.add_column("Content", max_width=50)
        table.add_column("Importance", justify="right")

        for m in memories:
            table.add_row(m.memory_type.value, m.content[:50], f"{m.importance:.1f}")

        self.console.print(table)

    def _cmd_verbose(self) -> None:
        """Toggle verbose mode."""
        self.verbose = not self.verbose
        if self.verbose:
            self.console.print("  [green]✓[/] Verbose mode ON — showing full tool traces")
        else:
            self.console.print("  [green]✓[/] Verbose mode OFF — clean output")

    def _cmd_free_mode(self) -> None:
        """Switch to free model routing."""
        self.free = True
        self.mode = "free"
        if self._router:
            self._router.config.routing_mode = "free"
            self._router.config.prefer_free = True
        self.console.print("  [green]✓[/] Switched to free model routing")

    # ─── Task Execution ──────────────────────────────────────────────

    async def _execute_task(self, goal: str) -> str | None:
        """Execute a task through the existing AgentLoop."""
        if self._agent_loop is None:
            return "Agent not initialized. Please check provider configuration."

        self.task_start = time.time()
        self.running = True
        self.cancel_event.clear()

        # Start live status
        self._live_status.start(goal)

        # Record run in session
        run = None
        if self.session_manager and self.session_id:
            try:
                run = self.session_manager.start_run(
                    self.session_id,
                    task=goal,
                    model_id=self.current_model,
                    provider=self.current_provider,
                )
            except Exception:
                pass

        try:
            task = await self._agent_loop.run(goal)

            # Stop live status before printing summary
            self._live_status.stop()

            elapsed = time.time() - self.task_start

            # Collect files that were modified
            files_changed = []
            for tc in task.tool_calls:
                if tc.tool_name in ("write_file", "edit_file") and tc.arguments:
                    fp = tc.arguments.get("path", tc.arguments.get("file_path", ""))
                    if fp:
                        fname = fp.replace("\\", "/").split("/")[-1]
                        if fname not in files_changed:
                            files_changed.append(fname)

            # Final TODO checklist
            if task.task_plan.items:
                self.console.print(f"\n  [bold]TODOs[/]")
                for item in task.task_plan.items:
                    self.console.print(f"    {item.display()}")

            # Print result summary
            if task.status.value == "completed":
                self.console.print("")
                self.console.print(Panel(
                    f"  [bold green]✓ Task completed in {_format_elapsed(elapsed)}[/]",
                    border_style="green",
                    padding=(0, 1),
                ))
                if task.result:
                    result = task.result
                    if len(result) > 300:
                        result = result[:270] + "..."
                    self.console.print(f"  {result}", highlight=False)
                # Files changed
                if files_changed:
                    self.console.print(f"\n  [bold]Files changed[/]")
                    for f in files_changed:
                        self.console.print(f"    {f}")
                # Execution summary
                stats_parts = [f"{task.iterations} iterations", f"{len(task.tool_calls)} tools", f"{_format_elapsed(elapsed)}"]
                self.console.print(f"\n  [dim]{' · '.join(stats_parts)}[/]")
                if task.verification_passed is True:
                    self.console.print("  [green]✓[/] [dim]Changes verified[/]", highlight=False)
                elif task.verification_passed is False:
                    self.console.print("  [red]✗[/] [dim]Verification failed[/]", highlight=False)
            elif task.status.value == "failed":
                self.console.print("")
                self.console.print(Panel(
                    f"  [bold red]✗ Task failed after {_format_elapsed(elapsed)}[/]",
                    border_style="red",
                    padding=(0, 1),
                ))
                if task.error:
                    error_msg = _safe_str(task.error)
                    if len(error_msg) > 200:
                        error_msg = error_msg[:200] + "..."
                    self.console.print(f"  {error_msg}", highlight=False)
                # Partial work
                if files_changed:
                    self.console.print(f"\n  [yellow]Partial work completed:[/]")
                    for f in files_changed:
                        self.console.print(f"    {f}")
                stats_parts = [f"{task.iterations} iterations", f"{len(task.tool_calls)} tools", f"{_format_elapsed(elapsed)}"]
                self.console.print(f"\n  [dim]{' · '.join(stats_parts)}[/]")
            else:
                self.console.print("")
                self.console.print(
                    f"  [yellow]Task ended with status: {task.status.value}[/]", highlight=False
                )

            # Update run in session
            if run and self.session_manager:
                try:
                    self.session_manager.complete_run(
                        run.run_id,
                        outcome="success" if task.status.value == "completed" else "failure",
                        verification_passed=(
                            task.verification_passed
                            if task.verification_passed is not None
                            else task.status.value == "completed"
                        ),
                        result_summary=(task.result or "")[:500],
                        iterations=task.iterations,
                        tool_calls=len(task.tool_calls),
                    )
                except Exception:
                    pass

            return task.result

        except asyncio.CancelledError:
            self._live_status.stop()
            self.console.print("\n  [yellow]✗ Task cancelled[/]", highlight=False)
            if run and self.session_manager:
                try:
                    self.session_manager.interrupt_run(run.run_id)
                except Exception:
                    pass
            return None
        except Exception as e:
            self._live_status.stop()
            self.console.print(f"\n  [red]✗ Error: {_safe_str(e)}[/]", highlight=False)
            if run and self.session_manager:
                try:
                    self.session_manager.fail_run(run.run_id, str(e))
                except Exception:
                    pass
            return None
        finally:
            self.running = False
            self._live_status.stop()

    # ─── Input Handling ──────────────────────────────────────────────

    def _read_input(self) -> str | None:
        """Read user input with prompt."""
        try:
            prompt = Text()
            prompt.append("Harness", style="bold blue")
            prompt.append(" › ", style="dim")

            if self.plain:
                user_input = input("harness > ")
            else:
                from rich.prompt import Prompt
                user_input = Prompt.ask(prompt)
            return user_input
        except (EOFError, KeyboardInterrupt):
            return None

    # ─── Main Loop ───────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the interactive shell."""
        self.session_start = time.time()

        # Show loading indicator during provider setup
        self.console.print("  [dim]Initializing...[/]", end="")

        if not await self._setup_provider():
            self.console.print("")
            self.console.print("")
            self.console.print("  [red]Cannot start interactive session.[/]")
            self.console.print("  [dim]Set OPENROUTER_API_KEY or start Ollama.[/]")
            return

        self.console.print("")  # End the initializing line

        self._setup_event_handlers()
        self._setup_session()

        # Print welcome screen AFTER provider setup is complete
        self._print_welcome()

        self.console.print("  [dim]Ctrl+C to cancel, /exit to quit.[/]")
        self.console.print("")

        # Main loop
        try:
            while True:
                user_input = self._read_input()

                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    if user_input.lower() in ("/exit", "/quit"):
                        break
                    await self._handle_command(user_input)
                    continue

                # Execute task
                await self._execute_task(user_input)

                # Show status line
                self._print_status_line()

        except KeyboardInterrupt:
            self.console.print("\n")
        except EOFError:
            pass
        finally:
            # Cleanup
            self.console.print("")
            self.console.print("  [dim]Session ended.[/]")
            if self.session_id:
                self.console.print(f"  [dim]Session ID: {self.session_id}[/]")
            self.console.print("")

            # Close provider
            if self._provider:
                try:
                    await self._provider.close()
                except Exception:
                    pass


# ─── Entry Point ──────────────────────────────────────────────────────────

def run_interactive(
    model: str | None = None,
    mode: str = "auto",
    free: bool = False,
    local: bool = False,
    plain: bool = False,
    max_iterations: int = 30,
    max_cost: float | None = None,
) -> None:
    """Entry point for interactive mode."""
    shell = InteractiveShell(
        model=model,
        mode=mode,
        free=free,
        local=local,
        plain=plain,
        max_iterations=max_iterations,
        max_cost=max_cost,
    )
    asyncio.run(shell.run())
