"""Unified UI component library for the Harness CLI.

Provides a coherent visual language: semantic colors, reusable renderables,
and compact display helpers.  All rendering is separated from business logic.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.text import Text

# ── Color System ────────────────────────────────────────────────────────────

# Semantic palette — restrained, professional, terminal-compatible.
# These work on both dark and light terminals.

class Color:
    """Semantic color constants for Harness UI."""

    # Core brand / accent
    PRIMARY = "cyan"
    PRIMARY_BOLD = "bold cyan"

    # Status
    SUCCESS = "green"
    SUCCESS_BOLD = "bold green"
    WARNING = "yellow"
    WARNING_BOLD = "bold yellow"
    ERROR = "red"
    ERROR_BOLD = "bold red"

    # Content
    TEXT = "white"
    TEXT_BOLD = "bold white"

    # Secondary
    MUTED = "dim"
    MUTED_WHITE = "dim white"

    # Accent
    ACCENT = "blue"
    ACCENT_BOLD = "bold blue"

    # Dim variants
    DIM_CYAN = "dim cyan"
    DIM_GREEN = "dim green"
    DIM_YELLOW = "dim yellow"
    DIM_RED = "dim red"

    # No style
    NONE = ""


# ── Symbols ─────────────────────────────────────────────────────────────────

class Sym:
    """Unicode symbols for consistent visual language.

    Uses safe Unicode that renders well across terminals.
    Fallback: every symbol is ASCII-compatible in meaning.
    """

    # Status indicators
    DOT = "\u2022"          # •
    DIAMOND = "\u25C6"      # ◆
    CHECK = "\u2713"        # ✓
    CROSS = "\u2717"        # ✗
    WARN = "\u26A0"         # ⚠
    SPINNER = "\u25D0"      # ◐
    CIRCLE = "\u25CB"       # ○
    RING = "\u25CE"         # ◎
    DASH = "\u2014"         # —

    # Arrows
    ARROW = "\u2192"        # →
    ARROW_R = "\u25B6"      # ▶

    # Structure
    THIN_LINE = "\u2500"    # ─
    BULLET = "\u2022"       # •

    # Fallback versions (ASCII-safe)
    @classmethod
    def safe_check(cls) -> str:
        return "+"
    
    @classmethod
    def safe_cross(cls) -> str:
        return "x"
    
    @classmethod
    def safe_warn(cls) -> str:
        return "!"


# ── Terminal Width ──────────────────────────────────────────────────────────

def get_term_width(console: Console) -> int:
    """Get terminal width, with safe fallback."""
    try:
        w = console.width
        if w and w > 10:
            return w
    except Exception:
        pass
    return 80


def clamp(text: str, max_width: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return text[: max_width - 3] + "..."


# ── Compact Elapsed Time ────────────────────────────────────────────────────

def fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as compact human-readable string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m"


# ── Renderable Builders ────────────────────────────────────────────────────

def build_section_header(label: str, style: str = "dim") -> Text:
    """Build a dim section header like '  Changes'."""
    return Text(f"  {label}", style=style)


def build_status_line(
    left_parts: list[tuple[str, str]],
    right_parts: list[tuple[str, str]],
    width: int = 80,
) -> Group:
    """Build a two-column status line with left-aligned and right-aligned content.

    Used for the persistent session status bar.
    """
    left = Text()
    for i, (text, style) in enumerate(left_parts):
        if i > 0:
            left.append("  ", style=Color.MUTED)
        left.append(text, style=style)

    right = Text()
    for i, (text, style) in enumerate(right_parts):
        if i > 0:
            right.append("  ", style=Color.MUTED)
        right.append(text, style=style)

    from rich.table import Table
    table = Table(
        show_header=False, expand=True, box=None,
        padding=(0, 0), width=width,
    )
    table.add_column("left", ratio=3)
    table.add_column("right", ratio=1, justify="right")
    table.add_row(left, right)

    return Group(table)


def build_compact_header(
    status_icon: str,
    status_style: str,
    label: str,
    elapsed: str,
) -> Text:
    """Build the compact '● HARNESS  12.3s' header line."""
    header = Text()
    header.append(f"{status_icon} ", style=status_style)
    header.append("Harness", style=Color.TEXT_BOLD)
    header.append("  ")
    header.append(elapsed, style="bold green")
    return header


def build_phase_indicator(
    phase: str,
    elapsed: str = "",
    meta: str = "",
) -> Text:
    """Build a phase indicator like '◐ Understanding...'."""
    icon = _phase_icon(phase)
    text = Text()
    text.append(f"{icon} ", style="bold cyan")
    text.append(f"{phase.title()}...  ", style="bold cyan")
    if elapsed or meta:
        parts = []
        if elapsed:
            parts.append(elapsed)
        if meta:
            parts.append(meta)
        text.append(f"({ ' \u00b7 '.join(parts) })", style=Color.MUTED)
    return text


def build_tool_activity(
    tool: str,
    args: dict[str, Any],
    status: str = "",
) -> Text:
    """Build a concise single-line tool activity display."""
    display = _tool_display(tool, args)
    text = Text()

    if status == "success":
        text.append(f"  {Sym.CHECK} ", style=Color.SUCCESS)
        text.append(display, style="")
    elif status == "error":
        text.append(f"  {Sym.CROSS} ", style=Color.ERROR)
        text.append(display, style="")
    else:
        text.append(f"  {Sym.SPINNER} ", style=Color.PRIMARY)
        text.append(display, style="")

    return text


def build_verification_line(
    label: str,
    passed: bool,
    detail: str = "",
) -> Text:
    """Build a verification status line like '✓ Tests passed'."""
    text = Text()
    if passed:
        text.append(f"  {Sym.CHECK} ", style=Color.SUCCESS)
    else:
        text.append(f"  {Sym.CROSS} ", style=Color.ERROR)
    text.append(label, style=Color.TEXT_BOLD)
    if detail:
        text.append(f"  {detail}", style=Color.MUTED)
    return text


def build_separator(width: int = 0) -> str:
    """Build a thin horizontal separator line."""
    w = width if width > 0 else 50
    return f"  {Sym.THIN_LINE * w}"


# ── Tool Display Name ───────────────────────────────────────────────────────

def _tool_display(tool: str, args: dict[str, Any]) -> str:
    """Concise human-readable display for a tool call."""
    if tool in ("read_file", "write_file", "edit_file"):
        verb = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit"}[tool]
        path = args.get("path", args.get("file_path", "?"))
        # Show only the filename for readability
        short = str(path).replace("\\", "/").split("/")[-1] if path else "?"
        return f"{verb} {short}"
    if tool == "list_files":
        p = args.get("path", ".")
        return f"List {p}"
    if tool == "run_command":
        cmd = args.get("command", "")
        if len(cmd) > 50:
            cmd = cmd[:47] + "..."
        return f"Run {cmd}"
    if tool == "grep":
        pat = args.get("pattern", "")
        return f'Search "{pat}"'
    if tool == "glob":
        return f"Glob {args.get('pattern', '')}"
    if tool.startswith("git_"):
        return f"Git {tool[4:].replace('_', ' ')}"
    return tool


def _phase_icon(phase: str) -> str:
    """Map a phase name to its display icon."""
    icons = {
        "understanding": Sym.SPINNER,
        "planning": Sym.SPINNER,
        "implementing": Sym.SPINNER,
        "testing": Sym.SPINNER,
        "verifying": Sym.SPINNER,
        "diagnosing": Sym.WARN,
        "fixing": Sym.SPINNER,
        "recovering": Sym.WARN,
        "committing": Sym.SPINNER,
        "complete": Sym.CHECK,
    }
    return icons.get(phase.lower(), Sym.SPINNER)


# ── Welcome Screen ──────────────────────────────────────────────────────────

def render_welcome(
    console: Console,
    workspace: str,
    model: str = "",
    provider: str = "",
    has_router: bool = False,
    *,
    plain: bool = False,
) -> None:
    """Render the compact, polished welcome screen.

    Design: minimal branding, clear information, obvious next step.
    """
    ws_display = workspace if len(workspace) <= 60 else "..." + workspace[-57:]

    if plain:
        console.print("Harness \u2014 Autonomous AI Engineering Agent")
        console.print(f"  {ws_display}")
        console.print("")
        return

    # Brand mark + tagline
    logo = """
[bold cyan]██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗
██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝
███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗
██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║
██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝[/]"""
    console.print(logo)
    console.print("  [dim]Autonomous software engineering in your terminal.[/]")
    console.print("")

    # Workspace
    console.print(f"  [dim]{ws_display}[/]")
    console.print("")

    # Provider status
    provider_display = provider or "Not connected"
    model_display = model or "Not set"

    console.print(f"  [bold cyan]█[/] [dim]Provider:[/] {provider_display}", highlight=False)
    console.print(f"  [bold cyan]█[/] [dim]Model:[/]     {model_display}", highlight=False)
    if has_router:
        console.print(f"  [bold cyan]█[/] [dim]Routing:[/]   ready", highlight=False)
    else:
        console.print(f"  [bold cyan]█[/] [dim]Routing:[/]   not initialized", highlight=False)
    console.print("")

    console.print("")

    # Quick commands
    console.print(f"  [dim]/help[/]   commands        [dim]/models[/]  view models")
    console.print(f"  [dim]/doctor[/] diagnostics     [dim]/free[/]    free routing")
    console.print("")


# ── Status Line ─────────────────────────────────────────────────────────────

def render_status_line(
    console: Console,
    provider: str = "",
    model: str = "",
    tool_calls: int = 0,
    elapsed: float = 0.0,
    *,
    plain: bool = False,
) -> None:
    """Render the session status bar above the input prompt."""
    if plain:
        parts = []
        if provider and model:
            parts.append(f"{provider} / {model}")
        if tool_calls > 0:
            parts.append(f"{tool_calls} tools")
        parts.append(fmt_elapsed(elapsed))
        console.print(f"[ {' \u00b7 '.join(parts)} ]", highlight=False)
        return

    from rich.panel import Panel

    left_parts: list[tuple[str, str]] = []
    if provider and model:
        left_parts.append((f"{provider} \u00b7 {model}", Color.MUTED))
    elif provider:
        left_parts.append((provider, Color.MUTED))
    if tool_calls > 0:
        left_parts.append((f"{tool_calls} tools", Color.MUTED))
    left_parts.append((fmt_elapsed(elapsed), Color.MUTED))

    right_parts: list[tuple[str, str]] = [
        ("Ctrl+C cancel  /exit quit", Color.MUTED),
    ]

    status = build_status_line(left_parts, right_parts)
    console.print(
        Panel(
            status,
            border_style="dim",
            padding=(0, 1),
            expand=True,
            title="[dim]Harness[/]",
            title_align="left",
        )
    )


# ── Live Status (for task execution) ───────────────────────────────────────

@dataclass
class LiveDisplay:
    """Compact live display for single-task execution.

    Replaces the verbose LiveStatus with a cleaner, grouped rendering.
    """

    console: Console
    plain: bool = False

    # State
    task_start: float = 0.0
    goal: str = ""
    current_phase: str = ""
    current_activity: str = ""
    activity_detail: str = ""
    iterations: int = 0
    tool_calls: int = 0
    current_model: str = ""
    todo_completed: int = 0
    todo_failed: int = 0
    todo_total: int = 0
    todo_items: list[dict[str, Any]] = field(default_factory=list)

    # Trail (recent completed operations)
    trail: list[tuple[str, str]] = field(default_factory=list)  # (icon, text)
    MAX_TRAIL: int = 6

    # Internal
    _active: bool = False
    _live: Any = None
    _last_plain_line: str = ""

    def start(self, goal: str) -> None:
        """Start the live display."""
        self.task_start = time.time()
        self.goal = goal
        self.current_phase = "understanding"
        self.current_activity = ""
        self.activity_detail = ""
        self.iterations = 0
        self.tool_calls = 0
        self.todo_completed = 0
        self.todo_failed = 0
        self.todo_total = 0
        self.todo_items = []
        self.trail = []
        self._active = True
        self._last_plain_line = ""

        if not self.plain:
            from rich.live import Live
            self._live = Live(
                self._renderable(),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        else:
            self._render_plain()

    def stop(self) -> None:
        """Stop the live display."""
        self._active = False
        if self._live is not None:
            try:
                self._live.update(self._renderable())
                self._live.stop()
            except Exception:
                pass
            self._live = None
        else:
            self.console.print("", highlight=False)

    def _refresh(self) -> None:
        if not self._active:
            return
        if self._live is not None:
            self._live.update(self._renderable())
        else:
            self._render_plain()

    # ── State updates ─────────────────────────────────────────────────

    def update_phase(self, phase: str) -> None:
        self.current_phase = phase
        self._refresh()

    def update_activity(self, tool: str, args: dict[str, Any]) -> None:
        self.current_activity = _tool_display(tool, args)
        if tool in ("read_file", "write_file", "edit_file"):
            fp = args.get("path", args.get("file_path", ""))
            self.activity_detail = fp.replace("\\", "/").split("/")[-1] if fp else ""
        elif tool == "run_command":
            cmd = args.get("command", "")
            self.activity_detail = cmd[:50] + "..." if len(cmd) > 50 else cmd
        else:
            self.activity_detail = ""
        self.tool_calls += 1
        self._push_trail(Sym.SPINNER, self.current_activity)
        self._refresh()

    def update_activity_complete(self, tool: str, status: str) -> None:
        if self.trail:
            _, text = self.trail[-1]
            icon = Sym.CHECK if status == "success" else Sym.CROSS
            self.trail[-1] = (icon, text)
        self._refresh()

    def update_todos(self, completed: int, total: int) -> None:
        self.todo_completed = completed
        self.todo_total = total
        self._refresh()

    def update_todo_items(self, items: list[dict[str, Any]]) -> None:
        self.todo_items = items or []
        self.todo_completed = sum(1 for i in self.todo_items if i.get("status") == "completed")
        self.todo_failed = sum(1 for i in self.todo_items if i.get("status") == "failed")
        self.todo_total = len(self.todo_items)
        self._refresh()

    def update_iterations(self, count: int) -> None:
        self.iterations = count
        self._refresh()

    def update_model(self, model: str) -> None:
        self.current_model = model
        self._refresh()

    def update_tests(self, line: str) -> None:
        self._refresh()

    def _push_trail(self, icon: str, text: str) -> None:
        self.trail.append((icon, text))
        if len(self.trail) > self.MAX_TRAIL:
            self.trail = self.trail[-self.MAX_TRAIL:]

    # ── Rendering ─────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        elapsed = time.time() - self.task_start if self.task_start else 0
        return fmt_elapsed(elapsed)

    def _renderable(self) -> Any:
        """Build a compact Rich renderable from current state."""
        if self.current_phase == "complete":
            return Text.assemble((f"{Sym.CHECK} Complete", "bold green"))

        lines: list[Any] = []

        # Phase line with spinner
        meta = [self._elapsed_str()]
        if self.iterations > 0:
            meta.append(f"{self.iterations} iter")
        if self.tool_calls > 0:
            meta.append(f"{self.tool_calls} tools")

        status_text = Text()
        status_text.append(f"{self.current_phase.title()}... ", style="bold cyan")
        status_text.append(f"({ ' \u00b7 '.join(meta) })", style=Color.MUTED)

        from rich.spinner import Spinner
        spinner = Spinner("dots", text=status_text)
        lines.append(spinner)

        # Current activity (only if something is happening)
        if self.current_activity:
            act = Text()
            act.append("  \u2514 ", style=Color.MUTED)
            act.append(self.current_activity, style=Color.MUTED)
            if self.activity_detail:
                act.append(f"  {self.activity_detail}", style=Color.MUTED)
            lines.append(act)

        # Recent trail (last few completed operations)
        completed_trail = [t for t in self.trail if t[0] in (Sym.CHECK, Sym.CROSS)]
        if completed_trail:
            shown = completed_trail[-4:]
            for icon, text in shown:
                style = Color.SUCCESS if icon == Sym.CHECK else Color.ERROR
                trail_text = Text()
                trail_text.append(f"  {icon} ", style=style)
                trail_text.append(text, style=Color.MUTED)
                lines.append(trail_text)

        # TODO progress (if any)
        if self.todo_total > 0:
            todo_text = Text()
            todo_text.append("  ", style="")
            todo_text.append(f"{self.todo_completed}/{self.todo_total}", style=Color.MUTED)
            if self.todo_failed > 0:
                todo_text.append(f" ({self.todo_failed} failed)", style=Color.ERROR)
            todo_text.append(" tasks", style=Color.MUTED)
            lines.append(todo_text)

        return Group(*lines)

    def _render_plain(self) -> None:
        """Plain mode: one compact line, printed only on meaningful changes."""
        parts = [f"[{self.current_phase}]"]
        if self.todo_total > 0:
            parts.append(f"TODO {self.todo_completed}/{self.todo_total}")
        if self.iterations > 0:
            parts.append(f"iter:{self.iterations}")
        line = " ".join(parts)
        if line != self._last_plain_line:
            self._last_plain_line = line
            activity = f" {self.current_activity}" if self.current_activity else ""
            self.console.print(
                f"  {line}{activity} ({self._elapsed_str()})", highlight=False
            )


# ── Summary Renderers ───────────────────────────────────────────────────────

def render_task_header(console: Console, goal: str, *, plain: bool = False) -> None:
    """Render a clean task header when user submits a goal."""
    if plain:
        console.print(f"> {goal}")
        return
    console.print("")
    console.print(f"  [bold]{goal}[/]", highlight=False)
    console.print("")


def render_success(
    console: Console,
    headline: str = "Completed",
    files_modified: list[str] | None = None,
    files_created: list[str] | None = None,
    files_deleted: list[str] | None = None,
    tests_passed: bool | None = None,
    tests_detail: str = "",
    verification_status: str = "",
    duration: float = 0.0,
    tool_calls: int = 0,
    next_actions: list[str] | None = None,
    summary: str = "",
    *,
    plain: bool = False,
) -> None:
    """Render a polished success summary."""
    files_modified = files_modified or []
    files_created = files_created or []
    files_deleted = files_deleted or []

    if plain:
        console.print(f"\n  {Sym.CHECK} {headline}")
        if files_modified or files_created:
            console.print(f"  Changes: {len(files_modified)} modified, {len(files_created)} created")
        console.print("")
        return

    lines: list[Any] = []

    # Completion line
    line = Text()
    line.append(f"  {Sym.CHECK} ", style=Color.SUCCESS_BOLD)
    line.append(headline, style=Color.TEXT_BOLD)
    lines.append(line)
    lines.append(Text(""))

    # Verification
    if verification_status or tests_detail:
        sec = Text("  Verification", style=Color.MUTED)
        lines.append(sec)
        if verification_status in ("passed", "not_started"):
            lines.append(Text(f"    {Sym.CHECK} Verified", style=Color.SUCCESS))
        elif verification_status == "failed":
            lines.append(Text(f"    {Sym.CROSS} Verification failed", style=Color.ERROR))
        if tests_detail:
            lines.append(Text(f"    {tests_detail}", style=Color.MUTED))
        lines.append(Text(""))

    # Changes
    if files_modified or files_created or files_deleted:
        sec = Text("  Changes", style=Color.MUTED)
        lines.append(sec)
        if files_modified:
            lines.append(Text(f"    {len(files_modified)} file(s) modified", style=Color.TEXT))
        if files_created:
            lines.append(Text(f"    {len(files_created)} file(s) created", style=Color.SUCCESS))
        if files_deleted:
            lines.append(Text(f"    {len(files_deleted)} file(s) deleted", style=Color.ERROR))
        lines.append(Text(""))

    # Execution
    if duration > 0 or tool_calls > 0:
        sec = Text("  Execution", style=Color.MUTED)
        lines.append(sec)
        parts = [fmt_elapsed(duration)]
        if tool_calls:
            parts.append(f"{tool_calls} tool(s)")
        lines.append(Text(f"    { ' \u00b7 '.join(parts) }", style=Color.MUTED))
        lines.append(Text(""))

    # Summary (agent response)
    if summary:
        sec = Text("  Summary", style=Color.MUTED)
        lines.append(sec)
        for sl in summary[:400].splitlines():
            lines.append(Text(f"    {sl}", style=Color.TEXT))
        lines.append(Text(""))

    # Next actions
    if next_actions:
        sec = Text("  Next", style=Color.MUTED)
        lines.append(sec)
        for action in next_actions:
            lines.append(Text(f"    {Sym.ARROW} {action}", style=Color.PRIMARY))
        lines.append(Text(""))

    console.print(Group(*lines))


def render_failure(
    console: Console,
    headline: str = "Unable to continue",
    reason: str = "",
    evidence: list[str] | None = None,
    files_modified: list[str] | None = None,
    recovery_attempts: int = 0,
    next_actions: list[str] | None = None,
    summary: str = "",
    *,
    plain: bool = False,
) -> None:
    """Render a polished failure summary — calm, actionable, not catastrophic."""
    if plain:
        console.print(f"\n  {Sym.CROSS} {headline}")
        if reason:
            console.print(f"  Reason: {reason}")
        console.print("  Session preserved.")
        console.print("")
        return

    lines: list[Any] = []

    # Failure header
    line = Text()
    line.append(f"  {Sym.CROSS} ", style=Color.ERROR_BOLD)
    line.append(headline, style=Color.TEXT_BOLD)
    lines.append(line)
    lines.append(Text(""))

    # Reason
    if reason:
        lines.append(Text(f"  {reason[:300]}", style=Color.TEXT))
        lines.append(Text(""))

    # Evidence
    if evidence:
        sec = Text("  Details", style=Color.MUTED)
        lines.append(sec)
        for e in evidence[:6]:
            lines.append(Text(f"    {e}", style=Color.MUTED))
        lines.append(Text(""))

    # State
    if files_modified:
        lines.append(Text(f"  {len(files_modified)} file(s) modified", style=Color.MUTED))
    else:
        lines.append(Text("  No files were changed", style=Color.MUTED))
    lines.append(Text("  Session has been preserved", style=Color.MUTED))
    lines.append(Text(""))

    # Next
    if next_actions:
        sec = Text("  Next steps", style=Color.MUTED)
        lines.append(sec)
        for action in next_actions:
            lines.append(Text(f"    {Sym.ARROW} {action}", style=Color.PRIMARY))
        lines.append(Text(""))

    console.print(Group(*lines))


def render_cancelled(
    console: Console,
    completed: list[str] | None = None,
    interrupted: list[str] | None = None,
    next_actions: list[str] | None = None,
    *,
    plain: bool = False,
) -> None:
    """Render a clean cancellation summary."""
    if plain:
        console.print(f"\n  {Sym.WARN} Cancelled")
        console.print("  Session preserved.")
        console.print("")
        return

    lines: list[Any] = []

    line = Text()
    line.append(f"  {Sym.WARN} ", style=Color.WARNING_BOLD)
    line.append("Cancelled", style=Color.TEXT_BOLD)
    lines.append(line)
    lines.append(Text(""))

    lines.append(Text("  Agents stopped", style=Color.SUCCESS))
    lines.append(Text("  Locks released", style=Color.SUCCESS))
    lines.append(Text("  Session preserved", style=Color.SUCCESS))
    lines.append(Text(""))

    if completed:
        sec = Text("  Completed before cancellation", style=Color.MUTED)
        lines.append(sec)
        for name in completed[:6]:
            lines.append(Text(f"    {Sym.CHECK} {name}", style=Color.SUCCESS))
        lines.append(Text(""))

    if interrupted:
        sec = Text("  Interrupted", style=Color.MUTED)
        lines.append(sec)
        for name in interrupted[:6]:
            lines.append(Text(f"    {Sym.DASH} {name}", style=Color.WARNING))
        lines.append(Text(""))

    if next_actions:
        sec = Text("  Next", style=Color.MUTED)
        lines.append(sec)
        for action in next_actions:
            lines.append(Text(f"    {Sym.ARROW} {action}", style=Color.PRIMARY))
        lines.append(Text(""))

    console.print(Group(*lines))


def render_provider_error(
    console: Console,
    error: str,
    *,
    plain: bool = False,
) -> None:
    """Render a clean provider error — not catastrophic, just actionable."""
    error_lower = error.lower()

    # Determine user-friendly message
    if "no usable free model" in error_lower:
        title = "Free models temporarily unavailable"
        detail = "Free models are rate-limited. Try again shortly, or configure your own API key."
    elif "429" in error_lower or "rate limit" in error_lower or "too many requests" in error_lower:
        title = "Rate limited"
        detail = "Models are temporarily unavailable due to rate limits."
    elif "402" in error_lower or "payment required" in error_lower:
        title = "Model requires payment"
        detail = "This model is not free. Use /free for free model routing."
    elif "401" in error_lower or "403" in error_lower or "unauthorized" in error_lower or "forbidden" in error_lower:
        title = "Authentication failed"
        detail = "Check your API key and provider permissions."
    elif "422" in error_lower or "invalid request" in error_lower:
        title = "Invalid request"
        detail = "Check model parameters and tool schemas."
    elif "all" in error_lower and "failed" in error_lower:
        title = "All models temporarily unavailable"
        detail = "Try again shortly."
    else:
        title = "Provider error"
        detail = error[:120]

    if plain:
        console.print(f"  {Sym.WARN} {title}")
        console.print(f"  {detail}")
        return

    console.print(f"  [yellow]{Sym.WARN}[/] [bold]{title}[/]")
    console.print(f"  [dim]{detail}[/]")


# ── Model Status ────────────────────────────────────────────────────────────

def render_model_status(
    console: Console,
    model: str = "",
    provider: str = "",
    mode: str = "auto",
    *,
    plain: bool = False,
) -> None:
    """Render current model/provider status."""
    if plain:
        console.print(f"  Model: {model or 'not set'}")
        console.print(f"  Provider: {provider or 'not set'}")
        console.print(f"  Mode: {mode}")
        return

    if not model and not provider:
        console.print(f"  [yellow]{Sym.DOT}[/] No provider configured")
        console.print(f"  [dim]Set an API key or use /free[/]")
        return

    console.print(f"  [green]{Sym.DOT}[/] {model or 'unknown'} \u00b7 {provider or 'unknown'}")
    console.print(f"  [dim]Mode: {mode}[/]")


# ── Session Resume ──────────────────────────────────────────────────────────

def render_session_resumed(
    console: Console,
    title: str = "",
    *,
    plain: bool = False,
) -> None:
    """Render a polished session resume message."""
    if plain:
        console.print(f"  Resumed session: {title}")
        return

    console.print(f"  [green]{Sym.CHECK}[/] [bold]Resumed session[/]")
    if title:
        console.print(f"  [dim]Previous task state restored.[/]")


def render_session_new(
    console: Console,
    *,
    plain: bool = False,
) -> None:
    """Render a new session message."""
    if plain:
        console.print("  New session started.")
        return

    console.print(f"  [green]{Sym.CHECK}[/] [dim]Session started[/]")


# ── Slash Command Help ──────────────────────────────────────────────────────

def render_help(
    console: Console,
    *,
    plain: bool = False,
) -> None:
    """Render polished slash command help."""
    if plain:
        console.print("  Commands:")
        console.print("  /help, /status, /models, /doctor, /free, /diff, /cancel, /exit")
        console.print("")
        return

    from rich.table import Table

    def _section(title: str, cmds: list[tuple[str, str]]) -> Group:
        header = Text(f"  {title}", style="dim bold")
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column("Command", style="cyan", no_wrap=True, width=14)
        t.add_column("Description", style="dim")
        for c, d in cmds:
            t.add_row(c, d)
        return Group(header, t)

    console.print("")

    execution = _section("Execution", [
        ("/status", "Session status and stats"),
        ("/pause", "Pause execution (state preserved)"),
        ("/resume", "Resume paused task"),
        ("/cancel", "Cancel running task"),
    ])
    console.print(execution)
    console.print("")

    project = _section("Project", [
        ("/files", "Show file changes"),
        ("/diff", "Show git diff"),
        ("/tests", "Show test results"),
        ("/plan", "Show execution plan"),
        ("/agents", "Show agent statuses"),
    ])
    console.print(project)
    console.print("")

    system = _section("System", [
        ("/models", "List available models"),
        ("/model", "Current model info"),
        ("/free", "Switch to free routing"),
        ("/doctor", "System health check"),
        ("/config", "Show configuration"),
    ])
    console.print(system)
    console.print("")

    session = _section("Session", [
        ("/help", "Show this help"),
        ("/clear", "Clear screen"),
        ("/verbose", "Toggle verbose output"),
        ("/exit", "Exit Harness"),
    ])
    console.print(session)
    console.print("")
