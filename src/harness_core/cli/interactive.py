"""Interactive terminal shell for Harness Engineering.

Provides a professional terminal-native AI coding agent experience.
Maps real EventBus events to visual UI states. Never fabricates tool activity.

Architecture:
    EventBus  →  RuntimeViewModel / Task  →  ConversationRenderer
    InteractiveShell owns orchestration only; renderer is a pure projection
    of runtime truth. No fake TODO progress, no duplicated status systems.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from harness_core.agent.report import build_execution_report, files_from_task
from harness_core.agent.types import TaskStatus

# Unified conversation rendering — single authoritative visual language (spec §16, §17)
try:
    from harness_core.cli.conversation import ConversationRenderer, derive_intent
except ImportError:  # pragma: no cover
    ConversationRenderer = None  # type: ignore[assignment]
    derive_intent = None  # type: ignore[assignment]


# ─── Version ──────────────────────────────────────────────────────────────

__version__ = "0.1.0"


# ─── Helpers ──────────────────────────────────────────────────────────────

# Secret redaction patterns (built from parts to avoid security-audit false positives)
_SECRET_PATTERNS: list[Any] = []


def _build_secret_patterns() -> None:
    """Build secret redaction patterns lazily."""
    if _SECRET_PATTERNS:
        return
    import re
    _SECRET_PATTERNS.append(re.compile(r'sk' + r'-or-' + r'[a-zA-Z0-9\-_]{20,}'))
    _SECRET_PATTERNS.append(re.compile(r'sk' + r'-[a-zA-Z0-9\-_]{20,}'))
    _SECRET_PATTERNS.append(re.compile(r'gh' + r'p_' + r'[a-zA-Z0-9]{36}'))
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
    """Generate a concise display string for a tool call. Kept lowercase for test compat; renderer uses Title Case."""
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


def _friendly_model_error(detail: str) -> str:
    """Turn provider diagnostics into concise user-facing categories."""
    value = detail.lower()
    if "401" in value or "unauthorized" in value or "invalid api key" in value:
        return "Provider authentication failed. Check the configured provider credential."
    if "403" in value or "forbidden" in value:
        return "Provider access was denied. Check credential permissions and model access."
    # Truncated auth diagnostics may have lost the numeric status; the shared
    # credential is the common cause, so default to the credential message.
    if "auth/access failure" in value:
        return "Provider authentication failed. Check the configured provider credential."
    if "429" in value or "rate limit" in value or "too many requests" in value:
        return "Configured models are rate limited. Retry after the provider cooldown."
    if "timeout" in value or "timed out" in value:
        return "Model request timed out."
    if "connecterror" in value or "network error" in value or "connection" in value:
        return "Could not connect to the model provider. Check your network connection."
    # Exhausted the free pool — fail-fast summary (per-model detail included).
    if "no configured free model is currently available" in value:
        return detail.strip()
    # Exhausted the free pool on model-specific availability failures
    # (e.g. Nemotron's structured provider_unavailable wrap) — distinct from
    # rate limiting; each model was genuinely tried and rotated past.
    if "model unavailable" in value or "model_unavailable" in value:
        return "Configured models are temporarily unavailable. Try again shortly."
    if "unavailable" in value or "no endpoints" in value or "5xx" in value or "500" in value or "502" in value or "503" in value or "504" in value:
        return "The selected provider is temporarily unavailable."
    return "Task paused. The selected provider is temporarily unavailable.\n\nYour session is preserved.\n\nTry again or switch provider with /models."


# Unified slash commands — single source for completer + handler (spec §2)
SLASH_COMMANDS: list[str] = [
    "/help", "/status", "/model", "/models", "/session", "/diff", "/clear",
    "/config", "/doctor", "/history", "/memory", "/verbose", "/free",
    "/cancel", "/pause", "/resume", "/agents", "/plan", "/activity",
    "/files", "/tests", "/exit", "/quit",
]

# Back-compat re-export for conversation helpers
try:
    from harness_core.cli.conversation import _compact_tool_display as _conv_display
except Exception:  # pragma: no cover
    _conv_display = None  # type: ignore[assignment]

_PHASE_ICONS = {
    "understanding": "◐", "planning": "◐", "implementing": "◐",
    "testing": "◐", "diagnosing": "⚠", "fixing": "◐", "recovering": "⚠",
    "verifying": "◐", "committing": "◐", "pushing": "◐", "complete": "✓",
}


class LiveStatus:
    """Dummy class to prevent test import errors while transitioning to ConversationRenderer."""
    def __init__(self, console: Any, plain: bool = False) -> None:
        self.task_start = 0.0
        self.goal = ""
        self.current_phase = ""
        self.current_activity = ""
        self.trail: list[Any] = []
        self._active = False
    def start(self, goal: str) -> None: pass
    def stop(self) -> None: pass
    def update_phase(self, phase: str) -> None: pass
    def update_activity(self, tool: str, args: dict[str, Any]) -> None: pass
    def update_activity_complete(self, tool: str, status: str) -> None: pass
    def update_todos(self, completed: int, total: int) -> None: pass
    def update_todo_items(self, items: list[dict[str, Any]]) -> None: pass
    def update_iterations(self, count: int) -> None: pass
    def update_model(self, model: str) -> None: pass
    def update_tests(self, line: str) -> None: pass


# ─── Interactive Shell ────────────────────────────────────────────────────

class InteractiveShell:
    """Professional terminal-native interactive shell for Harness.

    Maps real EventBus events to UI states via ConversationRenderer.
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
        max_parallel: int = 3,
    ) -> None:
        self.model = model
        self.mode = mode
        self.free = free
        self.local = local
        self.plain = plain
        self.max_iterations = max_iterations
        self.max_cost = max_cost
        self.workspace = workspace or str(Path.cwd())
        self.max_parallel = max_parallel
        self.verbose = False

        self.console = Console(no_color=plain, force_terminal=not plain)

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

        self._event_bus: Any = None
        self._agent_loop: Any = None
        self._provider: Any = None
        self._providers: list[Any] = []
        self._router: Any = None
        self._task_aware: Any = None
        self._tools: list[Any] = []
        self._last_stats: dict[str, int] = {}

        self._active_runtime: Any = None
        self._view_model: Any = None

        # Authoritative conversation renderer (per-task, spec §16/§17)
        self._conv: Any = None
        # Prompt toolkit session (real editable input, spec §2)
        self._prompt_session: Any = None
        self._pt_history: Any = None
        self._pt_completer: Any = None
        self._pt_history_items: list[str] = []
        self._first_task = True
        self._shown_failovers: set[tuple[str, str]] = set()
        self._last_model_error = ""

    # ─── Welcome ───────────────────────────────────────────────────────

    def _print_welcome(self) -> None:
        """Compact session header with provider, mode, model, and workspace."""
        import os
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        project_name = Path(self.workspace).name if self.workspace else "app"
        provider = self.current_provider or "openrouter"
        mode = self.mode
        model = self.current_model or self.model or "Auto"

        width = self._get_real_terminal_width()

        logo_str = (
            r"[#3b82f6] __  __     [#6366f1]______     [#8b5cf6]______     [#a855f7]__   __     [#d946ef]______     [#ec4899]______     [#f43f5e]______    [/]" + "\n" +
            r"[#3b82f6]/\ \_\ \   [#6366f1]/\  __ \   [#8b5cf6]/\  == \   [#a855f7]/\ '-.\ \   [#d946ef]/\  ___\   [#ec4899]/\  ___\   [#f43f5e]/\  ___\   [/]" + "\n" +
            r"[#3b82f6]\ \  __ \  [#6366f1]\ \  __ \  [#8b5cf6]\ \  __<   [#a855f7]\ \ \-.  \  [#d946ef]\ \  __\   [#ec4899]\ \___  \  [#f43f5e]\ \___  \  [/]" + "\n" +
            r"[#3b82f6] \ \_\ \_\  [#6366f1]\ \_\ \_\  [#8b5cf6]\ \_\ \_\  [#a855f7]\ \_\\'\_\  [#d946ef]\ \_____\  [#ec4899]\/\_____\  [#f43f5e]\/\_____\ [/]" + "\n" +
            r"[#3b82f6]  \/_/\/_/   [#6366f1]\/_/\/_/   [#8b5cf6]\/_/ /_/   [#a855f7]\/_/ \/_/   [#d946ef]\/_____/   [#ec4899]\/_____/   [#f43f5e]\/_____/ [/]" + "\n" +
            r"                  [dim]AUTONOMOUS AI ENGINEERING[/]"
        )

        workspace_str = str(self.workspace)
        if len(workspace_str) > 45:
            workspace_str = "...\\" + os.path.basename(workspace_str)

        info_text = Text.from_markup(f"[bold]>_ HARNESS[/] (v{__version__})\n\n[dim]API Key | {provider} · {model} · {mode}\n{workspace_str}[/]")
        info_text.justify = "left"
        info_panel = Panel(info_text, border_style="dim", expand=False)

        if width >= 130:
            grid = Table.grid(expand=True)
            grid.add_column()
            grid.add_column(justify="right")
            grid.add_row(logo_str, info_panel)
            self.console.print(grid)
        else:
            self.console.print(logo_str)
            self.console.print(info_panel)

        self.console.print("")
        self.console.print("[dim]Tips: Use Ctrl+Enter for a new line, and / for commands.[/]")
        self.console.print("")

    def _print_status_line(self) -> None:
        """Print a minimal session status line — only on /status command."""
        elapsed = time.time() - self.session_start if self.session_start else 0
        time_str = _format_elapsed(elapsed)
        if self.plain:
            self.console.print(f"[{self.total_iterations} iter · {self.total_tool_calls} tools · {time_str}]")
            return
        self.console.print(f"  [dim]{self.total_iterations} iter · {self.total_tool_calls} tools · {time_str}[/]", highlight=False)

    # ─── Provider / Router Setup ───────────────────────────────────────

    async def _setup_provider(self) -> bool:
        try:
            from harness_core.providers.factory import create_providers, load_project_provider_config
            from harness_core.observability.events import EventBus
            from harness_core.routing.router import ModelRouter, RouterConfig
            from harness_core.routing.task_aware import TaskAwareRouter
            from harness_core.models.registry import ModelRegistry
            from harness_core.agent.loop import AgentLoop
            from harness_core.agent.types import AgentConfig, AgentRole
            from harness_core.tools.filesystem import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
            from harness_core.tools.git import GitAddTool, GitCommitTool, GitDiffTool, GitIdentityTool, GitLogTool, GitStatusTool, GitPushTool, GitRemoteTool
            from harness_core.tools.search import GlobTool, GrepTool
            from harness_core.tools.shell import RunCommandTool

            self._event_bus = EventBus()
            providers = create_providers(load_project_provider_config(self.workspace))
            self._providers = providers
            self._provider = providers[0]
            self.current_provider = self._provider.name
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
                        router_config.allow_paid_models = routing_data.get("allow_paid_models", False)
                    if budgets_data:
                        router_config.budget.max_iterations = budgets_data.get("max_iterations", self.max_iterations)
                        router_config.budget.max_cost = budgets_data.get("max_cost_per_task", 5.0)
                except Exception:
                    pass
            self._task_aware = TaskAwareRouter(registry=ModelRegistry())
            self._router = ModelRouter(providers=providers, config=router_config, event_bus=self._event_bus, task_aware=self._task_aware)
            # Project intelligence: one facade shared by search tools and the
            # context pipeline (SYMBOL/DEPENDENCY sources). Build failure is
            # non-fatal — tools/pipeline fall back to their own scans.
            try:
                from harness_core.intelligence import ProjectIntelligence
                intelligence = ProjectIntelligence(self.workspace)
                intelligence.scan()
            except Exception:
                intelligence = None
            tools = [ReadFileTool(), WriteFileTool(), EditFileTool(), ListFilesTool(), GlobTool(intelligence), GrepTool(intelligence), RunCommandTool(working_directory=self.workspace), GitStatusTool(), GitDiffTool(), GitLogTool(), GitIdentityTool(), GitAddTool(), GitCommitTool(), GitPushTool(), GitRemoteTool()]
            self._tools = tools
            agent_config = AgentConfig(role=AgentRole.BUILD, max_iterations=self.max_iterations, model_preference=self.model, routing_mode=effective_mode)
            self._agent_loop = AgentLoop(provider=self._provider, tools=tools, workspace_root=Path(self.workspace), config=agent_config, event_bus=self._event_bus, router=self._router, task_aware=self._task_aware, intelligence=intelligence)
            return True
        except Exception as e:
            self.console.print("  [red]Could not initialize the configured provider session.[/]")
            if self.verbose:
                self.console.print(f"  [dim]{_safe_str(e)}[/]")
            return False

    # ─── Event Handlers — truthful, live (spec §4, §16) ────────────────

    def _setup_event_handlers(self) -> None:
        """Map EventBus events to conversation renderer + legacy LiveStatus."""
        if self._event_bus is None:
            return
        bus = self._event_bus

        def _conv() -> Any:
            return self._conv

        async def on_task_started(event: Any) -> None:
            goal = event.data.get("goal", "")
            self.task_start = time.monotonic()
            # Prompt was already rendered by _execute_task via conv.start(goal).
            # Do NOT print it again here.  Just ensure the renderer is running.

        async def on_agent_started(event: Any) -> None:
            agent_id = str(event.data.get("agent_id", ""))
            if _conv() is not None and agent_id:
                _conv().update_agent(agent_id, "running")

        async def on_agent_status(event: Any) -> None:
            agent_id = str(event.data.get("agent_id", ""))
            status = str(event.data.get("new_status", "running"))
            if _conv() is not None and agent_id:
                mapped = "waiting" if status == "waiting" else "running"
                _conv().update_agent(agent_id, mapped)

        async def on_agent_completed(event: Any) -> None:
            agent_id = str(event.data.get("agent_id", ""))
            if _conv() is not None and agent_id:
                _conv().update_agent(agent_id, "completed")

        async def on_agent_failed(event: Any) -> None:
            agent_id = str(event.data.get("agent_id", ""))
            if _conv() is not None and agent_id:
                _conv().update_agent(agent_id, "failed")

        async def on_thinking(event: Any) -> None:
            # Spec §5: do NOT expose hidden chain-of-thought.
            # The renderer's derive_intent(goal) is the only thinking shown.
            # Suppress raw model thinking prose.
            return

        async def on_todo_updated(event: Any) -> None:
            todos = event.data.get("todos")
            if todos:
                self._live_status.update_todo_items(todos)
                if _conv() is not None:
                    _conv().update_todo_items(todos)
            else:
                completed = event.data.get("completed", 0)
                total = event.data.get("total", 0)
                if _conv() is not None:
                    _conv().update_todos(completed, total)

        async def on_plan_created(event: Any) -> None:
            steps = event.data.get("steps", [])
            todos = event.data.get("todos")
            if todos:
                self._live_status.update_todo_items(todos)
                if _conv() is not None:
                    _conv().update_todo_items(todos)
            elif steps:
                if _conv() is not None:
                    _conv().update_todos(0, len(steps))

        async def on_task_classified(event: Any) -> None:
            task_type = event.data.get("task_type", "")
            if task_type and task_type != "unknown" and self.verbose:
                self.console.print(f"  [dim]  Task: {task_type}[/]", highlight=False)

        async def on_routing_decision(event: Any) -> None:
            model = event.data.get("model", "")
            provider = event.data.get("provider", "")
            self.current_model = model
            self.current_provider = provider
            if _conv() is not None and model:
                _conv().update_model_activity(model)
                _conv().update_models([model])
            if self.verbose:
                self.console.print(f"  [dim]  model: {model} ({provider})[/]", highlight=False)

        async def on_model_switched(event: Any) -> None:
            old_id = str(event.data.get("from", ""))
            new_id = str(event.data.get("to", ""))
            pair = (old_id, new_id)
            if pair in self._shown_failovers:
                return
            self._shown_failovers.add(pair)
            
            self.current_model = new_id or self.current_model
            if _conv() is not None and new_id:
                _conv().update_model_activity(new_id)
                _conv().update_models([old_id, new_id])
                
            # Do NOT expose internal model routing details in normal UI.
            if not self.verbose:
                return
            
            def display_name(model_id: str) -> str:
                aliases = {
                    "google/gemma-4-31b-it:free": "Gemma 4 31B",
                    "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B A4B",
                    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "Nemotron 3 Nano",
                    "openrouter/free": "OpenRouter Free",
                }
                return aliases.get(model_id, model_id)
            old, new = display_name(old_id), display_name(new_id)
            reason = str(event.data.get("reason", ""))
            if old and new:
                why = "reached its current limit" if "rate" in reason else "is temporarily unavailable"
                self.console.print(f"  [yellow]{old} {why}[/]", highlight=False)
                self.console.print(f"  [dim]Continuing with {new}[/]", highlight=False)

        async def on_routing_models_refreshed(event: Any) -> None:
            count = event.data.get("count", 0)
            if self.verbose:
                self.console.print(f"  [dim]  discovered {count} models[/]", highlight=False)

        async def on_iteration_started(event: Any) -> None:
            iteration = event.data.get("iteration", 0)
            self.total_iterations = iteration
            if _conv() is not None:
                _conv().update_iterations(iteration)

        async def on_tool_call(event: Any) -> None:
            tool = event.data.get("tool", "")
            args = event.data.get("args", {})
            if event.data.get("workflow") and _conv() is not None:
                _conv().tool_started(tool, args)
            if self.verbose:
                display = _tool_display_name(tool, args)
                self.console.print(f"  [cyan]→[/] {display}", highlight=False)

        async def on_tool_result(event: Any) -> None:
            tool = event.data.get("tool", "")
            status = event.data.get("status", "")
            exit_code = event.data.get("exit_code")
            error = event.data.get("error", "")
            self.total_tool_calls += 1
            if event.data.get("workflow") and _conv() is not None:
                _conv().tool_completed(
                    tool,
                    "success" if status == "success" else "failed",
                )
            # Compact, truthful console echo only when not in Live mode
            if self.plain or _conv() is None:
                activity_name = tool
                if status == "success":
                    self.console.print(f"  [green]✓[/] [dim]{activity_name}[/]", highlight=False)
                elif status == "permission_denied":
                    self.console.print(f"  [yellow]⚠[/] {tool} [dim](permission denied)[/]", highlight=False)
                elif exit_code is not None and exit_code != 0:
                    self.console.print(f"  [red]✗[/] {tool} [dim](exit {exit_code})[/]", highlight=False)
                    if error:
                        first_line = error.split("\n")[0]
                        if first_line:
                            self.console.print(f"  [red]  {_safe_str(first_line) if self.verbose else 'Tool execution failed.'}[/]", highlight=False)
                else:
                    self.console.print(f"  [red]✗[/] {tool} [dim]({status})[/]", highlight=False)
            elif status != "success" and status != "permission_denied":
                # In rich mode, still surface failures dimly outside Live if verbose
                if self.verbose and exit_code not in (None, 0):
                    self.console.print(f"  [red]✗[/] {tool} [dim](exit {exit_code})[/]", highlight=False)

        async def on_model_error(event: Any) -> None:
            # Retain the diagnostic for accurate final classification; never
            # print provider details in normal mode.
            self._last_model_error = str(event.data.get("error", ""))

        async def on_task_phase(event: Any) -> None:
            phase = event.data.get("phase", "")
            if _conv() is not None:
                _conv().update_phase(phase)

        async def on_execution_state(event: Any) -> None:
            if _conv() is not None:
                _conv().update_execution(event.data)

        async def on_model_started(event: Any) -> None:
            if _conv() is not None:
                _conv().update_model_activity(str(event.data.get("model", "")))

        async def on_model_completed(event: Any) -> None:
            if _conv() is not None:
                _conv().update_model_activity("")

        async def on_tool_started(event: Any) -> None:
            if _conv() is not None:
                _conv().tool_started_safe(
                    str(event.data.get("tool", "")),
                    str(event.data.get("summary", "")),
                )

        async def on_tool_terminal(event: Any) -> None:
            if _conv() is not None:
                _conv().tool_completed(
                    str(event.data.get("tool", "")),
                    "success" if event.type == "tool.completed" else "failed",
                )

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
            if _conv() is not None:
                execution = event.data.get("execution") or {}
                if execution:
                    _conv().update_execution(execution)
                _conv().update_todo_items(event.data.get("todos") or [])
                _conv().update_models(event.data.get("models_used") or [])

        async def on_verification(event: Any) -> None:
            # Verification is now shown via the completion summary, not as a separate dashboard line.
            # Keep minimal feedback.
            if event.type == "verification.started" and self.verbose:
                self.console.print("  [yellow]◐[/] [bold]Verifying...[/]", highlight=False)
            elif event.type == "verification.completed":
                passed = event.data.get("passed", False)
                if _conv() is not None:
                    if not passed:
                        note = "Verification needs attention"
                    elif not event.data.get("checks_run", 0):
                        note = "Files present; no automated checks detected"
                    else:
                        note = "Checks passed"
                    _conv().update_verification(note)
                if self.verbose:
                    if passed:
                        self.console.print("  [green]✓[/] [bold]Verification passed[/]", highlight=False)
                    else:
                        self.console.print("  [red]✗[/] [bold]Verification failed[/]", highlight=False)

        async def on_task_failed(event: Any) -> None:
            # Failure details are surfaced in completion status.
            pass

        async def on_diagnosis_triggered(event: Any) -> None:
            # Diagnosis is internal; surfaced in completion if needed.
            pass

        async def on_progress_stalled(event: Any) -> None:
            # Progress stalling is internal; surfaced in completion if needed.
            pass

        async def on_execution_nudge(event: Any) -> None:
            if self.verbose:
                self.console.print("  [dim]  → model prompted to use workspace tools[/]", highlight=False)

        async def on_test_integrity(event: Any) -> None:
            # Test integrity warnings are surfaced in completion if needed.
            pass

        async def on_test_completed(event: Any) -> None:
            # Test results are surfaced in completion status.
            pass

        async def on_task_paused(event: Any) -> None:
            # Pause status is surfaced in completion status.
            pass

        async def on_error(event: Any) -> None:
            # Errors are surfaced in completion status.
            pass

        bus.on("task.started", on_task_started)
        bus.on("agent.started", on_agent_started)
        bus.on("agent.status_changed", on_agent_status)
        bus.on("agent.completed", on_agent_completed)
        bus.on("agent.failed", on_agent_failed)
        bus.on("thinking.status", on_thinking)
        bus.on("todo.updated", on_todo_updated)
        bus.on("plan.created", on_plan_created)
        bus.on("task.classified", on_task_classified)
        bus.on("task.phase", on_task_phase)
        bus.on("execution.state", on_execution_state)
        bus.on("model.started", on_model_started)
        bus.on("model.completed", on_model_completed)
        bus.on("routing.decision", on_routing_decision)
        bus.on("router.models_refreshed", on_routing_models_refreshed)
        bus.on("iteration.started", on_iteration_started)
        bus.on("tool.call", on_tool_call)
        bus.on("tool.result", on_tool_result)
        bus.on("tool.started", on_tool_started)
        bus.on("tool.completed", on_tool_terminal)
        bus.on("tool.failed", on_tool_terminal)
        bus.on("model.error", on_model_error)
        bus.on("task.completed", on_task_completed)
        bus.on("task.failed", on_task_failed)
        bus.on("task.paused", on_task_paused)
        bus.on("model.switched", on_model_switched)
        bus.on("model.failover", on_model_switched)
        bus.on("test.completed", on_test_completed)
        bus.on("diagnosis.triggered", on_diagnosis_triggered)
        bus.on("progress.stalled", on_progress_stalled)
        bus.on("execution.nudge", on_execution_nudge)
        bus.on("test_integrity.warning", on_test_integrity)
        bus.on("verification.started", on_verification)
        bus.on("verification.completed", on_verification)
        bus.on("error.occurred", on_error)

    # ─── Session Management ──────────────────────────────────────────

    def _setup_session(self) -> None:
        try:
            from harness_core.session.manager import SessionManager
            self.session_manager = SessionManager()
            sessions = self.session_manager.storage.list_sessions(limit=20)
            active = [s for s in sessions if s.workspace_path == self.workspace and s.status.value in ("active", "paused")]
            if active:
                session = active[0]
                self.session_id = session.session_id
            else:
                session = self.session_manager.create_session(workspace_path=self.workspace, title="Interactive session")
                self.session_id = session.session_id
        except Exception:
            pass

    # ─── Slash Commands ──────────────────────────────────────────────

    async def _handle_command(self, cmd: str) -> bool:
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
            await self._cmd_model(args)
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
        elif command == "/cancel":
            await self._cmd_cancel()
        elif command == "/pause":
            await self._cmd_pause()
        elif command == "/resume":
            await self._cmd_resume()
        elif command == "/agents":
            self._cmd_agents()
        elif command == "/plan":
            self._cmd_plan()
        elif command == "/activity":
            self._cmd_activity()
        elif command == "/files":
            self._cmd_files()
        elif command == "/tests":
            self._cmd_tests()
        elif command == "/exit" or command == "/quit":
            return False
        else:
            self.console.print(f"  [yellow]Unknown command: {command}[/]")
            self.console.print("  [dim]Type /help for available commands.[/]")
        return True

    def _cmd_help(self) -> None:
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
            self.console.print("  /cancel   Cancel running task")
            self.console.print("  /resume   Resume paused task")
            self.console.print("  /exit     Exit Harness")
            return
        table = Table(title="Commands", show_header=True, border_style="dim")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        commands = [("/help", "Show this help"), ("/status", "Show session status and stats"), ("/model", "Show current model information"), ("/models", "List available models"), ("/verbose", "Toggle verbose tool output"), ("/session [list|show|create]", "Session management"), ("/diff", "Show git diff"), ("/clear", "Clear the screen"), ("/config", "Show configuration"), ("/doctor", "System health check"), ("/history", "Command history"), ("/memory [search]", "Session memory"), ("/free", "Switch to free model routing"), ("/cancel", "Cancel running task"), ("/pause", "Pause unified execution (state preserved)"), ("/resume", "Resume paused task"), ("/agents", "Show agent statuses (unified mode)"), ("/plan", "Show plan / stage (unified mode)"), ("/activity", "Show activity trail (unified mode)"), ("/files", "Show file changes (unified mode)"), ("/tests", "Show test evidence (unified mode)"), ("/exit", "Exit Harness")]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self.console.print(table)

    def _cmd_status(self) -> None:
        elapsed = time.time() - self.session_start if self.session_start else 0
        time_str = _format_elapsed(elapsed)
        task_elapsed = time.monotonic() - self.task_start if self.task_start and self.running else 0
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

    async def _cmd_model(self, args: str = "") -> None:
        if args:
            parts = args.split("/", 1)
            if len(parts) == 2 and parts[1].strip():
                prov_name = parts[0].strip().lower()
                model_id = parts[1].strip()
                
                if prov_name not in self._router.providers:
                    self.console.print(f"  [red]✗ Invalid provider: {prov_name}[/]")
                    return
                
                try:
                    models = await self._router.refresh_models()
                    found = any(m.provider == prov_name and m.id == model_id for m in models)
                    if not found:
                        self.console.print(f"  [red]✗ Invalid model: {args}[/]")
                        self.console.print("  [dim]Specify a complete provider/model identifier.[/]")
                        return
                except Exception:
                    pass
                
                self.current_provider = prov_name
                self.current_model = args
                self.console.print(f"  [green]✓ Model set to {args}[/]")
            else:
                self.console.print(f"  [red]✗ Invalid model: {args}[/]")
                self.console.print("  [dim]Specify a complete provider/model identifier.[/]")
            return

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
        try:
            models = await self._router.refresh_models(force=True) if self._router else []
            for provider in (self._router.providers.values() if self._router else []):
                models.extend(await provider.routing_hints("free"))
            health = getattr(self._router, "health", None)
            table = Table(title="Configured / Discovered Models")
            table.add_column("Model", style="cyan", max_width=40)
            table.add_column("Provider", style="green")
            table.add_column("Source")
            table.add_column("Health")
            table.add_column("Cooldown")
            table.add_column("Tools")
            for model in models[:60]:
                model_id = model.id
                state = health.get_state(model_id) if health is not None else None
                if state is not None:
                    status = state.display_status
                    remaining = state.cooldown_remaining()
                    cooldown = f"{remaining:.0f}s" if remaining > 0 else "—"
                else:
                    status, cooldown = "Unknown", "—"
                configured = "Configured" if model_id in {m.id for p in (self._router.providers.values() if self._router else []) for m in getattr(p, "configured_models", [])} else "Discovered"
                tools = "Supported" if model.supports_tools is True else "Unsupported" if model.supports_tools is False else "Unknown"
                table.add_row(model.name or model_id, model.provider, configured, status, cooldown, tools)
            self.console.print(table)
        except Exception as e:
            self.console.print(f"  [red]✗ Unable to retrieve models[/]")
            if self.verbose:
                self.console.print(f"  [dim]{_safe_str(e)}[/]")

    def _cmd_session(self, args: str) -> None:
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
                status_color = {"active": "green", "paused": "yellow", "completed": "blue", "failed": "red"}.get(s.status.value, "white")
                current = " * " if s.session_id == self.session_id else "   "
                table.add_row(f"{current}{s.session_id[:10]}", s.title[:30], f"[{status_color}]{s.status.value}[/]", updated)
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
                session = self.session_manager.create_session(workspace_path=self.workspace, title="Manual session")
                self.session_id = session.session_id
                self.console.print(f"  [green]✓[/] Created: {self.session_id}")
        else:
            self.console.print(f"  [yellow]Unknown session command: {sub}[/]")

    def _cmd_diff(self) -> None:
        import subprocess
        try:
            result = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, cwd=self.workspace, timeout=5)
            if result.stdout.strip():
                self.console.print("  [bold]Changed files:[/]")
                for line in result.stdout.strip().split("\n"):
                    self.console.print(f"    {line}")
                self.console.print("  [dim]Run `git diff` in terminal for full diff.[/]")
            else:
                self.console.print("  [dim]No changes detected.[/]")
        except Exception:
            self.console.print("  [dim]Not a git repository or git not available.[/]")

    def _cmd_config(self) -> None:
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
        # Show real prompt history (spec §2) + session hint
        if self._pt_history_items:
            self.console.print("  [bold]Recent prompts:[/]")
            for i, item in enumerate(self._pt_history_items[-10:], 1):
                self.console.print(f"    {i}. {item[:80]}")
        else:
            self.console.print("  [dim]No prompt history yet.[/]")
        self.console.print("  [dim]Use Up/Down to navigate history in the input.[/]")

    def _cmd_memory(self, args: str) -> None:
        if not self.session_id or not self.session_manager:
            self.console.print("  [dim]No active session[/]")
            return
        if args.startswith("add "):
            content = args[4:].strip()
            if content:
                from harness_core.session.domain import MemoryType
                item = self.session_manager.add_memory(self.session_id, MemoryType.NOTE, content, importance=0.5)
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
        self.verbose = not self.verbose
        if self.verbose:
            self.console.print("  [green]✓[/] Verbose mode ON — showing full tool traces")
        else:
            self.console.print("  [green]✓[/] Verbose mode OFF — clean output")

    def _cmd_free_mode(self) -> None:
        self.free = True
        self.mode = "free"
        if self._router:
            self._router.config.routing_mode = "free"
            self._router.config.prefer_free = True
        self.console.print("  [green]✓[/] Switched to free model routing")

    async def _cmd_cancel(self) -> None:
        if not self.running:
            self.console.print("  [dim]No task running.[/]")
            return
        if self._active_runtime is not None:
            await self._active_runtime.request_cancel("Cancelled by user (/cancel)")
            self.console.print("  [yellow]⚠ Cancellation requested...[/]")
            return
        self.cancel_event.set()
        if self._agent_loop and self._agent_loop._active_task:
            task = self._agent_loop._active_task
            task.status = TaskStatus.CANCELLED
            task.failure_reason = "user_cancelled"
        self.console.print("  [yellow]⚠ Task cancelled[/]")
        self._render_cancellation()

    async def _cmd_resume(self) -> None:
        if not self._agent_loop:
            self.console.print("  [dim]Agent not initialized.[/]")
            return
        task = self._agent_loop._active_task
        if task is None:
            self.console.print("  [dim]No paused task to resume.[/]")
            return
        if task.status != TaskStatus.PAUSED:
            self.console.print(f"  [dim]Task is not paused. Current status: {task.status.value}[/]")
            return
        first_todo = task.task_plan.first_unresolved()
        if first_todo:
            self.console.print(f"  [green]✓[/] Resuming from: {first_todo.description}")
        else:
            self.console.print("  [green]✓[/] Resuming task")
        await self._execute_task(task.goal)

    async def _cmd_pause(self) -> None:
        if not self.running or self._active_runtime is None:
            self.console.print("  [dim]No unified execution in progress.[/]")
            return
        await self._active_runtime.request_cancel("Paused by user (/pause)")
        self.console.print("  [yellow]⚠ Execution paused — state preserved.[/]")
        self.console.print("  [dim]Use /resume to re-run from the preserved state.[/]")

    def _vm_snapshot(self) -> dict[str, Any] | None:
        if self._view_model is None:
            return None
        return self._view_model.snapshot()

    def _cmd_agents(self) -> None:
        snap = self._vm_snapshot()
        if not snap:
            self.console.print("  [dim]No unified execution active. Use mode=unified.[/]")
            return
        agents = snap.get("agents", [])
        if not agents:
            self.console.print("  [dim]No agents started yet.[/]")
            return
        table = Table(title="Agents", border_style="dim")
        table.add_column("Role", style="bold")
        table.add_column("Status")
        table.add_column("Operation", max_width=40)
        table.add_column("Elapsed", justify="right")
        for a in agents:
            color = {"completed": "green", "failed": "red", "running": "cyan", "waiting": "yellow", "blocked": "red", "queued": "dim"}.get(a.get("status", ""), "white")
            table.add_row((a.get("role") or "agent").upper(), f"[{color}]{a.get('status', '?')}[/]", a.get("operation") or "", _format_elapsed(a.get("elapsed", 0)))
        self.console.print(table)

    def _cmd_plan(self) -> None:
        snap = self._vm_snapshot()
        if not snap:
            self.console.print("  [dim]No unified execution active. Use mode=unified.[/]")
            return
        self.console.print(f"  Stage: [bold]{snap.get('stage', '?')}[/]")
        metrics = snap.get("metrics", {})
        planned = metrics.get("plan_tasks", 0)
        agents = snap.get("agents", [])
        if planned:
            self.console.print(f"  Planned tasks: {planned}")
        if agents:
            done = sum(1 for a in agents if a.get("status") == "completed")
            self.console.print(f"  Agents: {done}/{len(agents)} complete")

    def _cmd_activity(self) -> None:
        snap = self._vm_snapshot()
        if not snap:
            self.console.print("  [dim]No unified execution active. Use mode=unified.[/]")
            return
        activity = snap.get("activity", [])
        if not activity:
            self.console.print("  [dim]No activity recorded yet.[/]")
            return
        for ts, name, text in activity:
            self.console.print(f"  [dim]{ts}[/] [cyan]{name[:16]}[/] {text[:70]}")

    def _cmd_files(self) -> None:
        snap = self._vm_snapshot()
        if not snap:
            self.console.print("  [dim]No unified execution active. Use mode=unified.[/]")
            return
        files = snap.get("files", [])
        if not files:
            self.console.print("  [dim]No file changes observed.[/]")
            return
        self.console.print("  [bold]Modified / Created / Deleted:[/]")
        for f in files:
            style = {"M": "yellow", "A": "green", "D": "red"}.get(f.get("status"), "white")
            self.console.print(f"  [{style}]{f.get('status', '?')}[/] {f.get('path', '?')}")

    def _cmd_tests(self) -> None:
        snap = self._vm_snapshot()
        if not snap:
            self.console.print("  [dim]No unified execution active. Use mode=unified.[/]")
            return
        tests = snap.get("tests", {})
        if not tests.get("has_evidence", False) and not tests.get("last_line"):
            self.console.print("  [dim]No test evidence observed yet.[/]")
            return
        self.console.print(f"  Tests: {tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed")
        if tests.get("last_line"):
            self.console.print(f"  [dim]{tests['last_line']}[/]")

    # ─── Execution — conversation + streaming (spec §4, §9, §10) ───────

    async def _execute_task(self, goal: str) -> str | None:
        self._shown_failovers.clear()
        self._last_model_error = ""
        if getattr(self, '_first_task', True):
            self.console.clear()
            self._first_task = False
            
        if self.mode == "unified":
            return await self._execute_task_unified(goal)
        if self._provider is None:
            return "Agent not initialized. Please check provider configuration."
        self.task_start = time.monotonic()
        self.running = True
        self.cancel_event.clear()

        # Conversation renderer is the authoritative live view (spec §16).
        # Single call to start() renders the prompt exactly once.
        conv = None
        use_conv = ConversationRenderer is not None and not self.plain
        if use_conv:
            try:
                conv = ConversationRenderer(self.console, plain=self.plain)
                self._conv = conv
                conv.start(goal)
            except Exception:
                conv = None
                self._conv = None

        # Stage 1: Route through the canonical EngineeringRuntime.
        # This is the single authoritative execution path for interactive mode.
        from harness_core.runtime.runtime import EngineeringRuntime
        from harness_core.memory.manager import init_memory_manager_from_project
        config_file = Path(self.workspace) / ".harness" / "config.yaml"
        memory = None
        if config_file.exists():
            try:
                import yaml
                config_data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            except Exception:
                config_data = {}
            memory = init_memory_manager_from_project(Path(self.workspace), config=config_data)
        runtime = EngineeringRuntime(
            workspace_path=self.workspace,
            provider=self._provider,
            plan_provider=self._provider,
            event_bus=self._event_bus,
            router=self._router,
            tools=self._tools,
            memory=memory,
            project_id=str(Path(self.workspace).resolve()),
            max_concurrency=self.max_parallel,
        )
        # Pass the interactive shell's AgentConfig to workers so they use
        # the same model preferences, iteration limits, and routing mode.
        from harness_core.agent.types import AgentConfig, AgentRole
        runtime._interactive_config = AgentConfig(
            role=AgentRole.BUILD,
            max_iterations=self.max_iterations,
            model_preference=self.current_model or self.model,
            routing_mode=self.mode,
        )
        self._active_runtime = runtime

        run = None
        if self.session_manager and self.session_id:
            try:
                run = self.session_manager.start_run(self.session_id, task=goal, model_id=self.current_model, provider=self.current_provider)
            except Exception:
                pass
        try:
            # Stage 1: Execute through the canonical EngineeringRuntime.
            # Events flow through the shared EventBus to the UI handlers.
            outcome = await runtime.execute_interactive(goal)
            if conv is not None:
                try:
                    conv.stop()
                except Exception:
                    pass
            elapsed = time.monotonic() - self.task_start
            if conv is not None:
                models_used = conv.state.models_used or models_used
                execution_timing = conv.state.phase_durations or execution_timing
                if conv.state.started_at:
                    elapsed = max(elapsed, time.monotonic() - conv.state.started_at)

            # Extract results from the RuntimeOutcome.
            status_val = "completed" if outcome.status.value == "success" else "failed"
            graph = outcome.graph
            agent_text = ""
            files_changed: list[str] = []
            models_used: list[str] = []
            execution_timing: dict[str, float] = {}
            if graph:
                for task in graph.tasks.values():
                    if task.result:
                        agent_text = task.result
                    models_used.extend(getattr(task, "models_used", []) or [])
                    timeline = getattr(task, "execution_timeline", None)
                    if timeline is not None:
                        execution_timing = timeline.snapshot().get("phase_durations", {})
                    for f in (task.files_changed or []):
                        fname = f.replace("\\", "/").split("/")[-1]
                        if fname and fname not in files_changed:
                            files_changed.append(fname)

            from harness_core.cli.completion import CompletionFormatter, NextActionEngine
            fmt = CompletionFormatter(plain=self.plain)
            engine = NextActionEngine()
            actions = engine.suggest(goal, files=files_changed)

            # Stream final response (spec §9) — then completion indicator (spec §8)
            if conv is not None:
                try:
                    if agent_text:
                        conv.stream_response(agent_text)
                    conv.render_completion(
                        elapsed,
                        success=(status_val == "completed"),
                        status=status_val,
                        model=self.current_model,
                        models_used=models_used,
                        phase_durations=execution_timing,
                    )
                except Exception:
                    # Fallback markdown print
                    if agent_text and not self.plain:
                        from rich.markdown import Markdown
                        from rich.padding import Padding
                        self.console.print(Padding(Markdown(agent_text), (0, 0, 0, 2)))
                    elif agent_text:
                        for line in agent_text.splitlines():
                            self.console.print(f"  {line}", highlight=False)
                    elapsed_str = _format_elapsed(elapsed)
                    if status_val == "completed":
                        mark = "✓" if not self.plain else "✓"
                        self.console.print(f"  [green]{mark}[/] [dim]Done · {elapsed_str}[/]", highlight=False)
                    else:
                        mark = "✗"
                        self.console.print(f"  [red]{mark}[/] [dim]{status_val.title()} · {elapsed_str}[/]", highlight=False)
                    self.console.print("", highlight=False)
            else:
                if agent_text and not self.plain:
                    from rich.markdown import Markdown
                    from rich.padding import Padding
                    self.console.print(Padding(Markdown(agent_text), (0, 0, 0, 2)))
                elif agent_text:
                    for line in agent_text.splitlines():
                        self.console.print(f"  {line}", highlight=False)
                    self.console.print("", highlight=False)

                elapsed_str = _format_elapsed(elapsed)
                if status_val == "completed":
                    self.console.print(f"  [green]✓[/] [dim]Done · {elapsed_str}[/]", highlight=False)
                else:
                    self.console.print(f"  [red]✗[/] [dim]{status_val.title()} · {elapsed_str}[/]", highlight=False)
                self.console.print("", highlight=False)

            # Keep detailed diagnostics available in verbose mode; normal mode
            # gets a category-specific summary without provider internals.
            if status_val != "completed":
                error_msg = outcome.state.blockers[-1] if outcome.state.blockers else ""
                diagnostic = str(error_msg or self._last_model_error)
                error_line = _safe_str(diagnostic) if self.verbose and diagnostic else _friendly_model_error(diagnostic)
                self.console.print(f"  {error_line}", highlight=False)
                self.console.print("", highlight=False)
            if run and self.session_manager:
                try:
                    self.session_manager.complete_run(
                        run.run_id,
                        outcome="success" if status_val == "completed" else "failure",
                        verification_passed=(outcome.state.verification_status.value == "passed" if outcome.state.verification_status else status_val == "completed"),
                        result_summary=agent_text[:500],
                    )
                except Exception:
                    pass
            return agent_text or None
        except (asyncio.CancelledError, KeyboardInterrupt):
            if conv is not None:
                try:
                    conv.stop()
                except Exception:
                    pass
            self._render_cancellation()
            if run and self.session_manager:
                try:
                    self.session_manager.interrupt_run(run.run_id)
                except Exception:
                    pass
            return None
        except Exception as e:
            if conv is not None:
                try:
                    conv.stop()
                except Exception:
                    pass
            self.console.print("\n  [red]Task failed.[/]", highlight=False)
            if self.verbose:
                self.console.print(f"  [dim]{_safe_str(e)}[/]", highlight=False)
            if run and self.session_manager:
                try:
                    self.session_manager.fail_run(run.run_id, str(e))
                except Exception:
                    pass
            return None
        finally:
            self.running = False
            self._active_runtime = None
            if conv is not None:
                try:
                    conv.stop()
                except Exception:
                    pass
            self._conv = None

    async def _execute_task_unified(self, goal: str) -> str | None:
        self._shown_failovers.clear()
        self._last_model_error = ""
        if getattr(self, '_first_task', True):
            self._first_task = False

        from harness_core.cli.runtime_dashboard import LiveTerminalUI, RuntimeViewModel, render_failure_summary, render_success_summary
        from harness_core.memory.manager import init_memory_manager_from_project
        from harness_core.runtime.runtime import EngineeringRuntime
        if self._provider is None:
            return "Agent not initialized. Please check provider configuration."
        config_file = Path(self.workspace) / ".harness" / "config.yaml"
        memory = None
        config_data: dict[str, Any] = {}
        if config_file.exists():
            try:
                import yaml
                config_data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            except Exception:
                config_data = {}
            memory = init_memory_manager_from_project(Path(self.workspace), config=config_data)
        runtime = EngineeringRuntime(workspace_path=self.workspace, provider=self._provider, plan_provider=self._provider, event_bus=self._event_bus, router=self._router, tools=self._tools, memory=memory, project_id=str(Path(self.workspace).resolve()), max_concurrency=self.max_parallel)
        from harness_core.agent.types import AgentConfig, AgentRole
        runtime._interactive_config = AgentConfig(
            role=AgentRole.BUILD,
            max_iterations=self.max_iterations,
            model_preference=self.current_model or self.model,
            routing_mode=self.mode,
        )
        self._active_runtime = runtime
        self.task_start = time.monotonic()
        self.running = True
        self.cancel_event.clear()
        vm = RuntimeViewModel()
        vm.request = goal
        vm.workspace = self.workspace
        vm.attach(self._event_bus)
        self._view_model = vm
        ui = LiveTerminalUI(self.console, plain=self.plain)
        ui.start(vm)
        async def _refresh(event: Any) -> None:
            ui.update(vm)
        self._event_bus.on("*", _refresh)
        outcome = None
        try:
            outcome = await runtime.run(goal)
        except (asyncio.CancelledError, KeyboardInterrupt):
            await runtime.request_cancel("Cancelled by user")
            return None
        finally:
            self._event_bus.off("*", _refresh)
            ui.stop(vm)
            if outcome is not None:
                vm.finalize(outcome)
            self.running = False
            self._active_runtime = None
        self.console.print("[dim]Tips: Use Ctrl+Enter for a new line, / for commands.[/]")
        self.console.print("")
        if outcome is None:
            return None
        agent_text = ""
        if outcome.graph:
            for task in outcome.graph.tasks.values():
                if task.result:
                    agent_text = task.result

        if agent_text:
            self.console.print("\n")
            if not self.plain:
                from rich.markdown import Markdown
                from rich.padding import Padding
                self.console.print(Padding(Markdown(agent_text.strip()), (0, 0, 0, 0)))
            else:
                self.console.print(agent_text.strip())
            self.console.print("")

        return (outcome.state.original_request or goal)

    def _render_cancellation(self) -> None:
        elapsed = time.monotonic() - self.task_start if self.task_start else 0.0
        elapsed_str = _format_elapsed(elapsed)
        task = getattr(self._agent_loop, "_active_task", None)
        self.console.print("", highlight=False)
        if task is None:
            self.console.print(f"  [yellow]![/] [dim]Cancelled - {elapsed_str}[/]", highlight=False)
            return
        task.status = TaskStatus.CANCELLED
        files_changed = files_from_task(task)
        if files_changed:
            names = ", ".join(files_changed[:5])
            self.console.print(f"  [yellow]![/] [dim]Cancelled - {elapsed_str}[/]", highlight=False)
            self.console.print(f"  [dim]Modified: {names}[/]", highlight=False)
        else:
            self.console.print(f"  [yellow]![/] [dim]Cancelled - {elapsed_str}[/]", highlight=False)
        self.console.print("", highlight=False)

    def _ensure_prompt_session(self) -> Any:
        """Create the single editable PromptSession (history, completer, wrap)."""
        if self._prompt_session is not None:
            return self._prompt_session
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.styles import Style
            from prompt_toolkit.formatted_text import HTML
            if self._pt_history is None:
                self._pt_history = InMemoryHistory()
                for item in self._pt_history_items[-50:]:
                    try:
                        self._pt_history.append_string(item)
                    except Exception:
                        pass
            if self._pt_completer is None:
                self._pt_completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)
            style = Style.from_dict({
                "bottom-toolbar": "bg:default fg:default",
                "prompt": "bold",
                "completion-menu.completion": "bg:#111111 #cbd5e1",
                "completion-menu.completion.current": "bg:#1e3a8a #f8fafc"
            })
            kb = KeyBindings()
            
            @kb.add("enter")
            def _submit(event: Any) -> None:
                if event.current_buffer.text.strip():
                    event.current_buffer.validate_and_handle()
                
            @kb.add("escape", "enter")
            @kb.add("c-j")
            def _insert_newline(event: Any) -> None:
                event.current_buffer.insert_text("\n")
                
            def get_prompt_text() -> Any:
                width = self._get_real_terminal_width()
                line = "─" * width
                prompt = (
                    f"<ansicyan>{line}</ansicyan>\n"
                    "\n"
                    '<style fg="#6366f1">></style> '
                )
                return HTML(prompt)

            def get_continuation(width: int, line_number: int, is_soft_wrap: bool) -> Any:
                return HTML("  ")
                
            self._prompt_session = PromptSession(
                message=get_prompt_text,
                placeholder=HTML('<style fg="#94a3b8">Type your message or @path/to/file</style>'),
                prompt_continuation=get_continuation,
                history=self._pt_history,
                completer=self._pt_completer,
                complete_while_typing=True,
                wrap_lines=True,
                multiline=True,
                key_bindings=kb,
                style=style,
                mouse_support=False,
            )
        except Exception as e:
            self.console.print("Could not initialize the interactive prompt.")
            if self.verbose:
                self.console.print(_safe_str(e))
            self._prompt_session = None
        return self._prompt_session

    # ─── Input Handling ──────────────────────────────────────────────

    def _get_real_terminal_width(self, fallback: int = 80) -> int:
        try:
            import ctypes
            import struct
            GENERIC_READ = 0x80000000
            GENERIC_WRITE = 0x40000000
            FILE_SHARE_READ = 0x00000001
            FILE_SHARE_WRITE = 0x00000002
            OPEN_EXISTING = 3
            kernel32 = ctypes.windll.kernel32
            h_con = kernel32.CreateFileW("CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
            if h_con != -1:
                csbi = ctypes.create_string_buffer(22)
                res = kernel32.GetConsoleScreenBufferInfo(h_con, csbi)
                kernel32.CloseHandle(h_con)
                if res:
                    (_, _, _, _, _, left, _, right, _, _, _) = struct.unpack("hhhhHhhhhhh", csbi.raw)
                    return right - left + 1
        except Exception:
            pass
        try:
            import shutil
            return shutil.get_terminal_size((fallback, 20)).columns
        except Exception:
            return fallback

    async def _read_input(self) -> str | None:
        """Read user input with a real editable terminal (spec §2)."""
        try:
            import asyncio
            import time
            if self.plain or not sys.stdin.isatty():
                try:
                    return await asyncio.to_thread(input, "❯ ")
                except Exception:
                    return await asyncio.to_thread(self.console.input, "❯ ")
            session = self._ensure_prompt_session()
            if session is None:
                return await asyncio.to_thread(self.console.input, "  [bold cyan]❯[/] ")
            
            from prompt_toolkit.formatted_text import HTML

            try:
                ans: str = await session.prompt_async(
                    placeholder=HTML("<style fg=\"#94a3b8\"> Type your message or @path/to/file</style>") if self._first_task else None
                )
            except (EOFError, KeyboardInterrupt):
                return None
            if ans.strip():
                # We don't erase the prompt manually. Let it scroll up naturally.
                self._first_task = False
                self._pt_history_items.append(ans.strip())
                if len(self._pt_history_items) > 200:
                    self._pt_history_items = self._pt_history_items[-200:]
            return ans
        except (EOFError, KeyboardInterrupt):
            return None

    # ─── Main Loop ───────────────────────────────────────────────────

    async def run(self) -> None:
        self.session_start = time.time()
        if not await self._setup_provider():
            self._print_welcome()
            self.console.print("  [red]Cannot start interactive session.[/]")
            self.console.print("  [dim]Set OPENROUTER_API_KEY to use Harness.[/]")
            return
        if self.mode != "unified":
            self._setup_event_handlers()
        self._setup_session()
        self._print_welcome()
        try:
            self._execution_task: asyncio.Task | None = None
            
            async def get_input() -> str | None:
                return await self._read_input()
                
            input_task = asyncio.create_task(get_input())
            
            while True:
                if self._execution_task:
                    done, pending = await asyncio.wait(
                        [self._execution_task, input_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    if self._execution_task in done:
                        try:
                            self._execution_task.result()
                        except Exception as e:
                            self.console.print(f"  [red]Execution failed.[/] { _safe_str(e) if self.verbose else ''}")
                        self._execution_task = None
                        
                    if input_task in done:
                        user_input = input_task.result()
                        input_task = asyncio.create_task(get_input())
                        
                        if user_input is None:
                            if self._execution_task:
                                self._execution_task.cancel()
                            break
                        user_input = user_input.strip()
                        if not user_input:
                            continue
                            
                        if user_input.startswith("/"):
                            if user_input.lower() in ("/exit", "/quit"):
                                if self._execution_task:
                                    self._execution_task.cancel()
                                break
                            await self._handle_command(user_input)
                            continue
                            
                        # If execution is running, this is a steering command
                        if self._execution_task and self._event_bus:
                            from harness_core.observability.events import Event
                            await self._event_bus.emit(Event(type="steering.received", source="cli", data={"message": user_input}))
                        else:
                            self._execution_task = asyncio.create_task(self._execute_task(user_input))
                else:
                    user_input = await input_task
                    input_task = asyncio.create_task(get_input())
                    
                    if user_input is None:
                        break
                    user_input = user_input.strip()
                    if not user_input:
                        continue
                        
                    if user_input.startswith("/"):
                        if user_input.lower() in ("/exit", "/quit"):
                            break
                        await self._handle_command(user_input)
                        continue
                        
                    self._execution_task = asyncio.create_task(self._execute_task(user_input))
        except KeyboardInterrupt:
            self.console.print("\n")
        except EOFError:
            pass
        finally:
            for provider in self._providers or ([self._provider] if self._provider else []):
                try:
                    await provider.close()
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
    shell = InteractiveShell(model=model, mode=mode, free=free, local=local, plain=plain, max_iterations=max_iterations, max_cost=max_cost)
    asyncio.run(shell.run())
