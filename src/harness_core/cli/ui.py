"""Unified UI component library for the Harness CLI.

Provides the single coherent visual language. All rendering is separated from
business logic. This module is the *only* place for primitives (colors,
symbols, elapsed formatting, tool display). Live renderers live in
conversation.py (single-task) and runtime_dashboard.py (RuntimeViewModel).

Architecture (spec §16):
    EventBus  →  RuntimeViewModel / Task  →  ConversationRenderer
    ui.py provides only pure helpers — it never invents state.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console, Group
from rich.text import Text

# ── Color System ────────────────────────────────────────────────────────────

class Color:
    """Semantic color constants for Harness UI."""

    PRIMARY = "cyan"
    PRIMARY_BOLD = "bold cyan"
    SUCCESS = "green"
    SUCCESS_BOLD = "bold green"
    WARNING = "yellow"
    WARNING_BOLD = "bold yellow"
    ERROR = "red"
    ERROR_BOLD = "bold red"
    TEXT = "white"
    TEXT_BOLD = "bold white"
    MUTED = "dim"
    MUTED_WHITE = "dim white"
    ACCENT = "blue"
    ACCENT_BOLD = "bold blue"
    DIM_CYAN = "dim cyan"
    DIM_GREEN = "dim green"
    DIM_YELLOW = "dim yellow"
    DIM_RED = "dim red"
    NONE = ""


# ── Symbols ─────────────────────────────────────────────────────────────────

class Sym:
    """Unicode symbols for consistent visual language."""

    DOT = "•"          # •
    DIAMOND = "◆"      # ◆
    CHECK = "✓"        # ✓
    CROSS = "✗"        # ✗
    WARN = "⚠"         # ⚠
    SPINNER = "◐"      # ◐
    CIRCLE = "○"       # ○
    RING = "◎"         # ◎
    DASH = "—"         # —
    ARROW = "→"        # →
    ARROW_R = "▶"      # ▶
    THIN_LINE = "─"    # ─
    BULLET = "•"       # •

    @classmethod
    def safe_check(cls) -> str:
        return "+"

    @classmethod
    def safe_cross(cls) -> str:
        return "x"

    @classmethod
    def safe_warn(cls) -> str:
        return "!"


# ── Terminal helpers ────────────────────────────────────────────────────────

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


def fmt_elapsed(seconds: float) -> str:
    """Single canonical elapsed formatter (spec §17 — one implementation)."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m"


# ── Renderable Builders (pure, no state) ────────────────────────────────────

def build_section_header(label: str, style: str = "dim") -> Text:
    return Text(f"  {label}", style=style)


def build_status_line(
    left_parts: list[tuple[str, str]],
    right_parts: list[tuple[str, str]],
    width: int = 80,
) -> Group:
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
    table = Table(show_header=False, expand=True, box=None, padding=(0, 0), width=width)
    table.add_column("left", ratio=3)
    table.add_column("right", ratio=1, justify="right")
    table.add_row(left, right)
    return Group(table)


def build_compact_header(status_icon: str, status_style: str, label: str, elapsed: str) -> Text:
    header = Text()
    header.append(f"{status_icon} ", style=status_style)
    header.append("Harness", style=Color.TEXT_BOLD)
    header.append("  ")
    header.append(elapsed, style="bold green")
    return header


def build_phase_indicator(phase: str, elapsed: str = "", meta: str = "") -> Text:
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
        text.append(f"({ ' · '.join(parts) })", style=Color.MUTED)
    return text


def build_tool_activity(tool: str, args: dict[str, Any], status: str = "") -> Text:
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


def build_verification_line(label: str, passed: bool, detail: str = "") -> Text:
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
    w = width if width > 0 else 50
    return f"  {Sym.THIN_LINE * w}"


# ── Single canonical tool display (spec §17) ────────────────────────────────

def _tool_display(tool: str, args: dict[str, Any]) -> str:
    """Concise human-readable display — lowercase to keep test compat; conversation Title-Cases for Live."""
    if tool in ("read_file", "write_file", "edit_file"):
        verb = {"read_file": "read", "write_file": "write", "edit_file": "edit"}[tool]
        path = args.get("path", args.get("file_path", "?"))
        return f"{verb} {path}" if path else verb
    if tool == "list_files":
        p = args.get("path", ".")
        return f"list {p}"
    if tool == "run_command":
        cmd = args.get("command", "")
        if len(cmd) > 50:
            cmd = cmd[:47] + "..."
        return f"run {cmd}"
    if tool == "grep":
        pat = args.get("pattern", "")
        path = args.get("path", ".")
        return f'grep "{pat}" in {path}' if path else f'grep "{pat}"'
    if tool == "glob":
        return f"glob {args.get('pattern', '')}"
    if tool.startswith("git_"):
        return f"git {tool[4:].replace('_', ' ')}"
    return tool


def _phase_icon(phase: str) -> str:
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


# ── Startup / Status — compact, no dashboard clutter (spec §12) ────────────

def render_welcome(
    console: Console,
    workspace: str,
    model: str = "",
    provider: str = "",
    has_router: bool = False,
    *,
    plain: bool = False,
) -> None:
    """Compact welcome — minimal chrome. No giant ASCII, no dashboard."""
    import os
    project_name = os.path.basename(workspace)
    ws_display = workspace if len(workspace) <= 60 else "..." + workspace[-57:]
    if plain:
        console.print("Harness")
        console.print(f"{workspace}")
        console.print("")
        return
    console.print(f"  [bold]Harness[/] [dim]·[/] [bold]{project_name}[/]", highlight=False)
    console.print(f"  [dim]{ws_display}[/]", highlight=False)
    console.print(f"  [dim]{'─' * 48}[/]", highlight=False)


def render_status_line(
    console: Console,
    provider: str = "",
    model: str = "",
    tool_calls: int = 0,
    elapsed: float = 0.0,
    *,
    plain: bool = False,
) -> None:
    if plain:
        parts = []
        if tool_calls > 0:
            parts.append(f"{tool_calls} tools")
        parts.append(fmt_elapsed(elapsed))
        console.print(f"[ {' · '.join(parts)} ]", highlight=False)
        return
    parts = []
    if tool_calls > 0:
        parts.append(f"{tool_calls} tools")
    parts.append(fmt_elapsed(elapsed))
    console.print(f"  [dim]{' · '.join(parts)}[/]", highlight=False)


# ── Back-compat shim: LiveDisplay was the old ui live renderer. ────────────
# New code uses conversation.ConversationRenderer. This shim keeps the import
# path alive for any external code but makes it obvious it is deprecated.
try:
    from harness_core.cli.conversation import ConversationRenderer as LiveDisplay  # type: ignore
except Exception:  # pragma: no cover
    LiveDisplay = None  # type: ignore


# ── Summary renderers — delegate to completion.py (spec §17, keep thin) ─────

def render_task_header(console: Console, goal: str, *, plain: bool = False) -> None:
    """Conversation turn header — kept for non-interactive callers."""
    if plain:
        console.print(f"> {goal}")
        return
    console.print("")
    console.print(f"  [bold]❯ {goal}[/]", highlight=False)
    console.print("")


def render_success(console: Console, headline: str = "Completed", files_modified: list[str] | None = None, files_created: list[str] | None = None, files_deleted: list[str] | None = None, tests_passed: bool | None = None, tests_detail: str = "", verification_status: str = "", duration: float = 0.0, tool_calls: int = 0, next_actions: list[str] | None = None, summary: str = "", *, plain: bool = False) -> None:
    # Thin wrapper — real logic lives in completion.CompletionFormatter so there is one source.
    from harness_core.cli.completion import CompletionFormatter, NextAction
    fmt = CompletionFormatter(plain=plain)
    acts = [NextAction(label=a) for a in (next_actions or [])]
    text = fmt.success(headline=headline, files_modified=files_modified or [], files_created=files_created or [], files_deleted=files_deleted or [], tests_line=tests_detail, verification_status=verification_status, duration=duration, tool_calls=tool_calls, next_actions=acts, summary=summary)
    console.print(text)


def render_failure(console: Console, headline: str = "Unable to continue", reason: str = "", evidence: list[str] | None = None, files_modified: list[str] | None = None, recovery_attempts: int = 0, next_actions: list[str] | None = None, summary: str = "", *, plain: bool = False) -> None:
    from harness_core.cli.completion import CompletionFormatter, NextAction
    fmt = CompletionFormatter(plain=plain)
    acts = [NextAction(label=a) for a in (next_actions or [])]
    text = fmt.failure(headline=headline, what_happened=reason, evidence_lines=evidence, files_modified=files_modified, recovery_attempts=recovery_attempts, next_actions=acts, agent_response=summary)
    console.print(text)


def render_cancelled(console: Console, completed: list[str] | None = None, interrupted: list[str] | None = None, next_actions: list[str] | None = None, *, plain: bool = False) -> None:
    from harness_core.cli.completion import CompletionFormatter, NextAction
    fmt = CompletionFormatter(plain=plain)
    acts = [NextAction(label=a) for a in (next_actions or [])]
    text = fmt.cancelled(completed=completed, interrupted=interrupted, next_actions=acts)
    console.print(text)


def render_provider_error(console: Console, error: str, *, plain: bool = False) -> None:
    error_lower = error.lower()
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


def render_model_status(console: Console, model: str = "", provider: str = "", mode: str = "auto", *, plain: bool = False) -> None:
    if plain:
        console.print(f"  Model: {model or 'not set'}")
        console.print(f"  Provider: {provider or 'not set'}")
        console.print(f"  Mode: {mode}")
        return
    if not model and not provider:
        console.print(f"  [yellow]{Sym.DOT}[/] No provider configured")
        console.print(f"  [dim]Set an API key or use /free[/]")
        return
    console.print(f"  [green]{Sym.DOT}[/] {model or 'unknown'} · {provider or 'unknown'}")
    console.print(f"  [dim]Mode: {mode}[/]")


def render_session_resumed(console: Console, title: str = "", *, plain: bool = False) -> None:
    if plain:
        console.print(f"  Resumed session: {title}")
        return
    console.print(f"  [green]{Sym.CHECK}[/] [bold]Resumed session[/]")
    if title:
        console.print(f"  [dim]Previous task state restored.[/]")


def render_session_new(console: Console, *, plain: bool = False) -> None:
    if plain:
        console.print("  New session started.")
        return
    console.print(f"  [green]{Sym.CHECK}[/] [dim]Session started[/]")


def render_help(console: Console, *, plain: bool = False) -> None:
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
    execution = _section("Execution", [("/status", "Session status and stats"), ("/pause", "Pause execution (state preserved)"), ("/resume", "Resume paused task"), ("/cancel", "Cancel running task")])
    console.print(execution)
    console.print("")
    project = _section("Project", [("/files", "Show file changes"), ("/diff", "Show git diff"), ("/tests", "Show test results"), ("/plan", "Show execution plan"), ("/agents", "Show agent statuses")])
    console.print(project)
    console.print("")
    system = _section("System", [("/models", "List available models"), ("/model", "Current model info"), ("/free", "Switch to free routing"), ("/doctor", "System health check"), ("/config", "Show configuration")])
    console.print(system)
    console.print("")
    session = _section("Session", [("/help", "Show this help"), ("/clear", "Clear screen"), ("/verbose", "Toggle verbose output"), ("/exit", "Exit Harness")])
    console.print(session)
    console.print("")
