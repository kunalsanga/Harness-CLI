"""Unified conversation rendering for Harness CLI.

This module is the single authoritative visual language for the interactive
terminal.  It replaces the scattered LiveStatus / LiveDisplay /
LiveTerminalUI render paths with one coherent model:

    EventBus  ──>  RuntimeViewModel / ConversationState  ──>  ConversationRenderer

Every visible activity item must correspond to a real runtime event.
No fake progress, no invented thinking, no dashboard clutter.

Layout:

    ❯ explain this project

      I'll inspect the project structure and core runtime.

    ✓ Read README.md
    ✓ Read pyproject.toml
    ✎ Edit src/app.py
    ▶ pytest

    Harness Engineering is a model-agnostic autonomous
    software engineering agent ...

    ✓ Done · 18.4s
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.text import Text

from harness_core.cli.ui import Color, Sym, fmt_elapsed, shorten_path, PHASE_DISPLAY, PHASE_ICON

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def format_token_count(tokens: int, estimated: bool = False, concise: bool = False) -> str:
    """Format a token count with k suffix and optional estimation tag."""
    if tokens >= 1000:
        val = f"{tokens / 1000:.1f}k"
    else:
        val = str(tokens)
    if concise:
        return f"~{val} (est)" if estimated else val
    return f"~{val} tokens (est)" if estimated else f"{val} tokens"


# ── Intent-derived thinking ───────────────────────────────────────────────

def derive_intent(goal: str) -> str:
    """User-safe execution intent, derived from intent classification.

    Never exposes hidden chain-of-thought.  Short, concrete, and grounded
    in what the agent will actually do (inspect / implement / test / push).
    """
    try:
        from harness_core.agent.intent import classify_intent, Intent
        result = classify_intent(goal or "")
        intent = result.intent
        low = (goal or "").lower()
        if intent == Intent.GIT:
            return "I'll check the repository status and push the changes."
        if intent == Intent.TEST:
            if "fix" in low or "fail" in low:
                return "I'll inspect test failures and fix the implementation."
            return "I'll run the test suite and report the results."
        if intent == Intent.FIX:
            return "I'll diagnose the failure and apply a targeted fix."
        if intent == Intent.DOCUMENT:
            return "I'll update the documentation and verify the result."
        if intent == Intent.EXPLAIN or intent == Intent.INSPECT or intent == Intent.QUESTION or intent == Intent.REVIEW:
            return "I'll inspect the project structure and core runtime before explaining it."
        if intent == Intent.MODIFY:
            return "I'll inspect the codebase and implement the requested changes."
        # OTHER / fallthrough
        if any(k in low for k in ("push", "commit", "github")):
            return "I'll check the repository status and push the changes."
        if any(k in low for k in ("test", "pytest", "npm test")):
            return "I'll run the test suite and report the results."
        if any(k in low for k in ("explain", "describe", "what is", "how does", "overview")):
            return "I'll inspect the project structure and key files."
        return "I'll start working on your request."
    except Exception:
        return "I'll start working on your request."


def _compact_tool_display(tool: str, args: dict[str, Any]) -> str:
    """Single canonical compact tool display — one place fixes all three copies."""
    if tool in ("read_file", "write_file", "edit_file"):
        verb = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit"}[tool]
        path = args.get("path", args.get("file_path", "?"))
        short = shorten_path(str(path)) if path else "?"
        if len(short) > 28:
            short = ".../" + short.replace("\\", "/").rstrip("/").split("/")[-1]
        return f"{verb} {short}"
    if tool == "list_files":
        p = shorten_path(str(args.get("path", ".")))
        return f"List {p}"
    if tool == "run_command":
        cmd = args.get("command", "")
        if len(cmd) > 55:
            cmd = cmd[:52] + "..."
        return f"Run {cmd}"
    if tool == "grep":
        pat = args.get("pattern", "")
        return f'Search "{pat}"'
    if tool == "glob":
        return f"Glob {args.get('pattern', '')}"
    if tool.startswith("git_"):
        return f"Git {tool[4:].replace('_', ' ')}"
    return tool


# ── Conversation state ────────────────────────────────────────────────────

@dataclass
class AgentState:
    name: str
    status: str = "running"

@dataclass
class ConversationState:
    """Live conversation state — pure data, no Rich."""

    goal: str = ""
    intent: str = ""
    started_at: float = 0.0  # monotonic task start, set by execution.state
    execution_state: str = "idle"
    phase_durations: dict[str, float] = field(default_factory=dict)
    todos: list[dict[str, Any]] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    verification_note: str = ""
    model_activity: str = ""
    # Each entry: (icon, display, status)  status in running|success|failed
    items: list[tuple[str, str, str]] = field(default_factory=list)
    current_tool: str = ""
    iterations: int = 0
    tool_calls: int = 0
    phase: str = ""
    display_phase: str = ""  # user-facing phase from PHASE_DISPLAY
    streaming_text: str = ""
    max_items: int = 12
    agents: dict[str, AgentState] = field(default_factory=dict)
    workspace: str = ""
    # Activity aggregation: track consecutive reads for collapsing
    _consecutive_reads: int = 0
    stage: str = "Connecting…"
    current_turn_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    file_change_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    current_diff_preview: dict[str, Any] | None = None
    test_summary: dict[str, Any] | None = None

    def update_agent(self, name: str, status: str) -> None:
        if name not in self.agents:
            self.agents[name] = AgentState(name=name, status=status)
        else:
            self.agents[name].status = status

    def add_running(self, display: str, tool: str = "") -> None:
        # Activity aggregation: collapse consecutive reads into summary
        is_read = tool in ("read_file", "list_files", "glob", "grep")
        if is_read:
            self._consecutive_reads += 1
            if self._consecutive_reads >= 3:
                # Replace the aggregated entry or add one
                agg_display = f"Inspected {self._consecutive_reads} files"
                # Find and update existing aggregate entry
                for i in range(len(self.items) - 1, -1, -1):
                    if self.items[i][2] in ("success", "running") and self.items[i][1].startswith("Inspected "):
                        self.items[i] = (Sym.CHECK, agg_display, "success")
                        self.tool_calls += 1
                        return
                # First time aggregating: replace last 2 individual reads
                self.items.append((Sym.SPINNER, agg_display, "running"))
                self.tool_calls += 1
                return
        else:
            self._consecutive_reads = 0

        self.items.append((Sym.SPINNER, display, "running"))
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self.tool_calls += 1

    def complete_last(self, status: str) -> None:
        if not self.items:
            return
        icon, text, _ = self.items[-1]
        if status == "success":
            new_icon = Sym.CHECK
            new_status = "success"
        else:
            new_icon = Sym.CROSS
            new_status = "error" if status in ("error", "failed") else status
        self.items[-1] = (new_icon, text, new_status)

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "intent": self.intent,
            "elapsed": round(time.monotonic() - self.started_at, 1) if self.started_at else 0,
            "items": list(self.items),
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
        }


# ── Live conversation renderer ────────────────────────────────────────────

class ConversationRenderer:
    """Single canonical live renderer for conversation UX.

    Uses Rich's Live for in-place updates while a task runs, then stops and
    leaves the final frame.  Plain mode prints a compact line only on change.
    """

    def __init__(self, console: Console, plain: bool = False) -> None:
        self.console = console
        self.plain = plain
        self.state = ConversationState()
        self._active = False
        self._live: Any = None
        self._last_plain_line = ""
        self._thinking_shown = False

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self, goal: str) -> None:
        self.state = ConversationState(
            goal=goal,
            intent=derive_intent(goal),
            # Seed a monotonic fallback clock so elapsed time is never dead
            # before the first execution.state event arrives. The canonical
            # TaskExecutionTimeline value from the event overwrites this.
            started_at=time.monotonic(),
            stage="Connecting…",
        )
        self._thinking_shown = False
        self._active = True
        self._last_plain_line = ""
        # Render the user prompt as conversation turn (spec §3, §15)
        self._render_prompt(goal)

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
        self._active = False
        if self._live is not None:
            try:
                self._live.update(self._renderable())
                self._live.stop()
            except Exception:
                pass
            self._live = None
        else:
            # Plain mode: leave a blank line
            try:
                self.console.print("", highlight=False)
            except Exception:
                pass
        try:
            self.console.show_cursor(True)
        except Exception:
            pass

    def _refresh(self) -> None:
        if not self._active:
            return
        if self._live is not None:
            try:
                self._live.update(self._renderable())
            except Exception:
                pass
        else:
            self._render_plain()

    # ── prompt & thinking ─────────────────────────────────────────────

    def _render_prompt(self, goal: str) -> None:
        """Render the user prompt.
        
        The prompt is already visible from the input composer in rich mode.
        In plain mode, we print the prompt to keep history intact.
        """
        if self.plain:
            self.console.print(f"\n❯ {goal}", highlight=False)

    def _show_thinking(self) -> None:
        # No-op: intent is shown via _renderable(), not as a separate print.
        pass

    # ── event-driven updates ──────────────────────────────────────────

    def update_phase(self, phase: str) -> None:
        self.state.phase = phase
        self.state.display_phase = PHASE_DISPLAY.get(phase, "")
        if phase == "understanding":
            self.state.stage = "Investigating…"
        elif phase == "planning":
            self.state.stage = "Planning…"
        elif phase == "verifying":
            self.state.stage = "Verifying…"
        elif phase == "repairing":
            self.state.stage = "Fixing…"
        elif phase == "working":
            self.state.stage = "Building…"
        self._refresh()

    def update_execution(self, data: dict[str, Any]) -> None:
        self.state.execution_state = str(data.get("state", self.state.execution_state))
        self.state.started_at = float(data.get("started_at") or self.state.started_at or 0.0)
        phase_labels = {
            "understanding": "✦ Understanding…", 
            "planning": "✦ Planning…",
            "thinking": "✦ Considering…", 
            "working": "✦ Building…",
            "repairing": "✦ Fixing…", 
            "verifying": "✦ Verifying…",
            "completed": "Completed", "failed": "Failed",
            "cancelled": "Cancelled", "paused": "Paused",
        }
        self.state.display_phase = phase_labels.get(self.state.execution_state, self.state.display_phase)
        if self.state.execution_state == "thinking" and self.state.model_activity:
            self.state.display_phase = "✦ Considering…"
        self.state.phase_durations = {
            str(key): float(value) for key, value in (data.get("phase_durations") or {}).items()
        }
        self._refresh()

    def update_iterations(self, count: int) -> None:
        self.state.iterations = count
        self._refresh()

    def tool_started(self, tool: str, args: dict[str, Any]) -> None:
        display = _compact_tool_display(tool, args)
        if tool == "run_command":
            cmd = args.get("command", "")
            short = cmd[:45] + "..." if len(cmd) > 45 else cmd
            display = f"Run {short}" if short else display
            cmd_lower = str(cmd).lower()
            if any(w in cmd_lower for w in ("pytest", "test", "unittest", "cargo test", "npm test")):
                self.state.stage = "Running tests…"
            else:
                self.state.stage = "Running command…"
        elif tool in ("read_file", "list_files", "grep", "glob"):
            self.state.stage = "Reading files…"
        elif tool in ("write_file", "edit_file"):
            self.state.stage = "Editing code…"
        elif tool.startswith("git_"):
            self.state.stage = "Git operation…"
        elif tool.startswith("mcp_"):
            self.state.stage = f"Calling {tool}…"
        self.state.current_tool = tool
        self.state.add_running(display, tool=tool)
        self._refresh()

    def tool_started_safe(self, tool: str, summary: str) -> None:
        """Render the semantic tool summary without consuming raw arguments."""
        detail = summary.partition(":")[2].strip() if ":" in summary else ""
        if tool == "run_command":
            display = _compact_tool_display(tool, {"command": detail})
            cmd_lower = detail.lower()
            if any(w in cmd_lower for w in ("pytest", "test", "unittest", "cargo test", "npm test")):
                self.state.stage = "Running tests…"
            else:
                self.state.stage = "Running command…"
        elif tool in ("read_file", "write_file", "edit_file", "list_files"):
            display = _compact_tool_display(tool, {"path": detail})
            if tool in ("read_file", "list_files"):
                self.state.stage = "Reading files…"
            else:
                self.state.stage = "Editing code…"
        elif tool in ("glob", "grep"):
            display = _compact_tool_display(tool, {"pattern": detail})
            self.state.stage = "Reading files…"
        else:
            display = _compact_tool_display(tool, {})
            if tool.startswith("mcp_"):
                self.state.stage = f"Calling {tool}…"
            elif tool.startswith("git_"):
                self.state.stage = "Git operation…"
        self.state.current_tool = tool
        self.state.add_running(display, tool=tool)
        self._refresh()

    def tool_completed(self, tool: str, status: str) -> None:
        self.state.complete_last(status)
        self.state.current_tool = ""
        self._refresh()

    def update_tokens(
        self,
        output_tokens: int = 0,
        input_tokens: int | None = None,
        estimated: bool = False,
        delta: int | None = None,
    ) -> None:
        """Update turn token accounting (exact or estimated)."""
        if delta is not None:
            self.state.current_turn_tokens += delta
        elif output_tokens > 0:
            self.state.current_turn_tokens = output_tokens
        if input_tokens is not None:
            self.state.input_tokens = input_tokens
        self.state.output_tokens = self.state.current_turn_tokens
        if estimated:
            self.state.tokens_estimated = True
        self._refresh()

    def update_file_changed(
        self,
        path: str,
        added: int = 0,
        removed: int = 0,
        diff: str = "",
    ) -> None:
        """Record live file edit and refresh diff preview."""
        stats = self.state.file_change_stats.setdefault(path, {"added": 0, "removed": 0})
        stats["added"] += added
        stats["removed"] += removed
        self.state.current_diff_preview = {
            "path": path,
            "added": added,
            "removed": removed,
            "diff": diff,
        }
        self.state.stage = "Editing code…"
        self._refresh()

    def update_test_completed(
        self,
        passed: int = 0,
        total: int = 0,
        success: bool = True,
        command: str = "",
    ) -> None:
        """Record live test completion results."""
        failed = max(0, total - passed)
        self.state.test_summary = {
            "passed": passed,
            "total": total,
            "failed": failed,
            "success": success,
            "command": command,
        }
        self.state.stage = "Running tests…"
        self._refresh()

    def set_stage(self, stage: str) -> None:
        """Set the active stage label."""
        self.state.stage = stage
        self._refresh()

    def update_todos(self, completed: int, total: int) -> None:
        # No-op for now: TODO progress is NOT shown unless authoritative and non-fake.
        # Kept for API compat with existing callers.
        self._refresh()

    def update_todo_items(self, items: list[dict[str, Any]]) -> None:
        self.state.todos = list(items or [])
        self._refresh()

    def update_models(self, models: list[str]) -> None:
        self.state.models_used = list(dict.fromkeys(
            [*self.state.models_used, *(str(model) for model in models if model)]
        ))
        self._refresh()

    def update_verification(self, note: str) -> None:
        self.state.verification_note = (note or "")[:180]
        self._refresh()

    def update_model_activity(self, model: str = "") -> None:
        self.state.model_activity = f"Requesting {self._model_display_name(model)}" if model else ""
        if model:
            self.state.stage = "Considering…"
        if self.state.execution_state == "thinking":
            self.state.display_phase = "✦ Considering…"
        self._refresh()

    # ── rendering ─────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        if not self.state.started_at:
            return "0s"
        return fmt_elapsed(time.monotonic() - self.state.started_at)

    def _tool_icon(self, tool: str, status: str) -> tuple[str, str]:
        """Return (icon, style) for a tool based on its name and status.

        Uses semantic icons: ✓ read/list/glob, ✎ edit/write, ▶ run_command,
        ● other.
        """
        if status == "success":
            if tool in ("read_file", "list_files", "glob", "grep",
                        "git_status", "git_diff", "git_log", "git_identity"):
                return (Sym.CHECK, "green")
            if tool in ("write_file", "edit_file"):
                return ("✎", "yellow")
            if tool == "run_command":
                return ("▶", "cyan")
            return (Sym.CHECK, "green")
        if status in ("failed", "error"):
            return (Sym.CROSS, "red")
        # Running
        if tool == "run_command":
            return ("▶", "cyan")
        return (Sym.SPINNER, "cyan")

    def _tool_name_from_display(self, display: str) -> str:
        """Extract tool name from display string for icon lookup."""
        _map = {
            "read": "read_file", "write": "write_file",
            "edit": "edit_file", "list": "list_files",
            "run": "run_command", "grep": "grep",
            "glob": "glob", "git": "git_status",
        }
        for prefix, tool in _map.items():
            if display.lower().startswith(prefix):
                return tool
        return ""

    def _renderable(self) -> Any:
        """Build the live renderable: phase header + intent + compact tool activity.

        No 'Working...' placeholder.  No fake progress.  Just real activity.
        """
        lines: list[Any] = []

        # Agent display — progressive disclosure
        if len(self.state.agents) > 1:
            agent_count = len(self.state.agents)
            running = [a for a in self.state.agents.values() if a.status == "running"]
            completed = [a for a in self.state.agents.values() if a.status == "completed"]
            failed = [a for a in self.state.agents.values() if a.status == "failed"]
            if agent_count <= 3:
                # Show individual agents
                for name, agent in self.state.agents.items():
                    if agent.status == "completed":
                        lines.append(Text.assemble(("    ✓ ", "green"), (f"{name.capitalize()}", "dim")))
                    elif agent.status == "failed":
                        lines.append(Text.assemble(("    ✗ ", "red"), (f"{name.capitalize()}", "dim")))
                    elif agent.status == "waiting":
                        lines.append(Text.assemble(("    · ", "dim"), (f"{name.capitalize()}", "dim")))
                    else:
                        lines.append(Text.assemble(("    ◐ ", "cyan"), (f"{name.capitalize()}", "cyan")))
            else:
                # Compact: show count
                parts = []
                if running:
                    parts.append(f"{len(running)} active")
                if completed:
                    parts.append(f"{len(completed)} done")
                if failed:
                    parts.append(f"{len(failed)} failed")
                summary = " · ".join(parts) if parts else f"{agent_count} workers"
                lines.append(Text(f"  {Sym.SPINNER} {summary}", style="cyan"))
            lines.append(Text(""))

        # Phase section header — driven by real backend task.phase events
        if self.state.display_phase and self.state.display_phase != "Completed":
            phase_text = Text()
            phase_text.append(f"  {self.state.display_phase}", style="bold cyan")
            lines.append(phase_text)
            lines.append(Text(""))

        if self.state.todos:
            lines.append(Text("  Plan", style="bold"))
            limit = min(len(self.state.todos), 5)
            width = max(12, self.console.size.width - 10)
            for index, item in enumerate(self.state.todos[:limit]):
                status = str(item.get("status", "pending")).lower()
                is_unverified = bool(item.get("unverified") or (item.get("evidence") or {}).get("unverified"))
                cat = str(item.get("category", "")).lower()
                if is_unverified or (getattr(self.state, "failed", False) and cat in ("implement", "verify", "test") and status != "completed"):
                    icon = "!"
                    color = "yellow"
                else:
                    icon = {
                        "completed": "✓", "complete": "✓",
                        "in_progress": "●", "active": "●", "failed": "✗",
                    }.get(status, "○")
                    color = "green" if icon == "✓" else "red" if icon == "✗" else "cyan" if icon == "●" else "dim"
                title = str(item.get("title") or item.get("description") or "").replace("\n", " ")
                if is_unverified:
                    title += " (unverified)"
                if len(title) > width - 7:
                    title = title[: max(1, width - 10)] + "..."
                
                is_last = (index == limit - 1 and len(self.state.todos) <= limit)
                tree_prefix = "└─ " if is_last else "├─ "
                lines.append(Text(f"  {tree_prefix}{icon} {title}", style=color))
            if len(self.state.todos) > limit:
                lines.append(Text(f"  └─ +{len(self.state.todos) - limit} more", style="dim"))
            lines.append(Text(""))
        if self.state.verification_note:
            lines.append(Text(f"  {self.state.verification_note}", style="dim"))
        if self.state.model_activity:
            lines.append(Text(f"    {self.state.model_activity}", style="dim"))

        # Intent message — short, user-safe execution intent (spec §5)
        if self.state.intent and not self._thinking_shown:
            self._thinking_shown = True
            lines.append(Text(f"    {self.state.intent}", style="dim"))
            lines.append(Text(""))

        # Tool activity (spec §6) — compact list with tree structure
        total_items = len(self.state.items)
        for i, (icon, display, status) in enumerate(self.state.items):
            is_last = (i == total_items - 1)
            tree_prefix = "└─ " if is_last else "├─ "
            
            txt = Text()
            tool_name = self._tool_name_from_display(display)
            
            txt.append(f"  {tree_prefix}", style="dim")
            
            if status == "running":
                tool_icon, tool_style = self._tool_icon(tool_name, "running")
                txt.append(f"{tool_icon} ", style=tool_style)
                txt.append(display, style="dim")
                if tool_name == "run_command":
                    txt.append(" ...", style="dim")
            elif status == "success":
                tool_icon, tool_style = self._tool_icon(tool_name, "success")
                txt.append(f"{tool_icon} ", style=tool_style)
                txt.append(display, style="dim")
            elif status in ("failed", "error"):
                txt.append(f"{Sym.CROSS} ", style="red")
                txt.append(display, style="dim")
            else:
                txt.append(f"{Sym.DOT} ", style="dim")
                txt.append(display, style="dim")
            lines.append(txt)

        # Test results status if recent
        if self.state.test_summary:
            t = self.state.test_summary
            p = t.get("passed", 0)
            f = t.get("failed", 0)
            st_style = "green" if t.get("success") else "red"
            msg = f"  ✓ Tests completed  {p} passed" if f == 0 else f"  ✗ Tests failed  {p} passed, {f} failed"
            lines.append(Text(msg, style=st_style))

        # Diff preview box if a file was edited
        if self.state.current_diff_preview:
            diff_box = self._render_diff_preview(self.state.current_diff_preview)
            if diff_box:
                lines.append(diff_box)

        # Status line (spinner + stage + elapsed + token indicator)
        if self._active:
            lines.append(self._render_status_line())

        # No placeholder.  If nothing is happening, show nothing.
        return Group(*lines) if lines else Text("")

    def _render_status_line(self) -> Text:
        """Render the live animated status line (spinner + stage + elapsed + tokens)."""
        idx = int(time.monotonic() * 10) % len(_SPINNER_FRAMES)
        spinner = _SPINNER_FRAMES[idx]

        elapsed_sec = int(time.monotonic() - self.state.started_at) if self.state.started_at else 0
        mins, secs = divmod(max(0, elapsed_sec), 60)
        time_str = f"{mins:02d}:{secs:02d}"

        stage = self.state.stage or "Working…"
        tok_str = ""
        if self.state.current_turn_tokens > 0:
            tok_formatted = format_token_count(self.state.current_turn_tokens, self.state.tokens_estimated)
            tok_str = f" · ↓ {tok_formatted}"

        txt = Text()
        txt.append(f"  {spinner} ", style="bold cyan")
        txt.append(stage, style="bold")
        txt.append(f"   {time_str}{tok_str}", style="dim")
        return txt

    def _render_diff_preview(self, preview: dict[str, Any]) -> Any:
        """Render a compact 2-5 line diff preview box for live file modifications."""
        raw_path = str(preview.get("path", ""))
        short_path = shorten_path(raw_path)
        added = preview.get("added", 0)
        removed = preview.get("removed", 0)
        diff_text = str(preview.get("diff", "")).strip()

        # Terminal width adaptation (min 24, max 80)
        width = max(24, min(self.console.size.width - 4, 80))
        header = f" {short_path}  +{added} -{removed} "

        content_lines: list[tuple[str, str]] = []
        rem_lines: list[tuple[str, str]] = []
        add_lines: list[tuple[str, str]] = []
        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            if line.startswith("-") and len(rem_lines) < 2:
                rem_lines.append((line[: width - 6], "red"))
            elif line.startswith("+") and len(add_lines) < 2:
                add_lines.append((line[: width - 6], "green"))
            if len(rem_lines) >= 2 and len(add_lines) >= 2:
                break
        content_lines = rem_lines + add_lines
        if not content_lines:
            for line in diff_text.splitlines():
                if not (line.startswith("+++") or line.startswith("---") or line.startswith("@@")):
                    style = "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
                    content_lines.append((line[: width - 6], style))
                    if len(content_lines) >= 4:
                        break

        if not content_lines:
            return None

        box_lines = []
        top_border = f"  ┌─{header}" + "─" * max(2, width - len(header) - 4) + "┐"
        box_lines.append(Text(top_border, style="dim cyan"))
        for text, style in content_lines:
            pad = max(0, width - len(text) - 6)
            row = Text("  │ ", style="dim cyan")
            row.append(text, style=style)
            row.append(" " * pad + " │", style="dim cyan")
            box_lines.append(row)
        bottom_border = "  └" + "─" * (width - 4) + "┘"
        box_lines.append(Text(bottom_border, style="dim cyan"))
        return Group(*box_lines)

    def _render_plain(self) -> None:
        # Plain mode: one line per tool, printed immediately
        if self.state.items:
            icon, display, status = self.state.items[-1]
            elapsed_sec = int(time.monotonic() - self.state.started_at) if self.state.started_at else 0
            mins, secs = divmod(max(0, elapsed_sec), 60)
            time_str = f"{mins:02d}:{secs:02d}"
            tok_str = ""
            if self.state.current_turn_tokens > 0:
                tok_str = f" · ↓ {format_token_count(self.state.current_turn_tokens, self.state.tokens_estimated)}"
            line = f"  [{self.state.stage}] {icon} {display} ({time_str}{tok_str})"
            if line != self._last_plain_line:
                self._last_plain_line = line
                self.console.print(line, highlight=False)

    # ── streaming final response ──────────────────────────────────────

    def stream_response(self, text: str, delay: float = 0.0) -> None:
        """Stream the final response progressively (spec §9).

        In rich mode, we render Markdown incrementally via a Live region.
        In plain mode, we print line by line.
        If delay is 0, we print immediately (still appears as streaming to
        the user because prior Live activity was live).
        """
        if not text or not text.strip():
            return
        self.state.stage = "Preparing response…"
        clean = text.strip()
        if self.plain:
            for line in clean.splitlines():
                self.console.print(f"  {line}", highlight=False)
                if delay:
                    time.sleep(delay)
            self.console.print("", highlight=False)
            return

        # Rich mode: progressive reveal via Live + Markdown
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.padding import Padding

        paragraphs = clean.split("\n\n")
        shown: list[str] = []
        try:
            with Live(console=self.console, refresh_per_second=12, transient=False) as live:
                for para in paragraphs:
                    shown.append(para)
                    md = Markdown("\n\n".join(shown))
                    live.update(Padding(md, (0, 0, 0, 2)))
                    if delay:
                        time.sleep(min(delay, 0.08))
        except Exception:
            self.console.print(Padding(Markdown(clean), (0, 0, 0, 2)))
        self.console.print("", highlight=False)

    def render_completion(
        self,
        elapsed: float,
        success: bool = True,
        status: str = "completed",
        error_detail: str = "",
        model: str = "",
        models_used: list[str] | None = None,
        phase_durations: dict[str, float] | None = None,
        token_usage: dict[str, Any] | None = None,
        files_changed: list[str] | None = None,
        file_change_stats: dict[str, dict[str, int]] | None = None,
        test_results: dict[str, Any] | None = None,
    ) -> None:
        """Completion indicator with duration, tokens, file changes, and test results."""
        elapsed_str = fmt_elapsed(elapsed)
        if status == "cancelled":
            mark, label, style = "!", "Cancelled", "yellow"
        elif status == "paused":
            mark, label, style = "⏸", "Paused", "yellow"
        elif status == "blocked":
            mark, label, style = Sym.CROSS, "Blocked", "red"
        elif status == "partial":
            mark, label, style = Sym.WARN, "Partial", "yellow"
        elif success:
            mark, label, style = Sym.CHECK, "Completed", "green"
        else:
            mark, label, style = Sym.CROSS, "Failed", "red"
        count = self.state.tool_calls
        self.console.print(
            f"  {mark} {label}  {elapsed_str}  {count} tool calls",
            style=style if not self.plain else None,
            highlight=False,
        )
        model_list = list(dict.fromkeys(models_used or self.state.models_used or ([model] if model else [])))
        if model_list:
            names = [self._model_display_name(name) for name in model_list]
            label_key = "Model" if len(names) == 1 else "Models"
            self.console.print(f"    {label_key}  {'  →  '.join(names)}", style="dim", highlight=False)

        # Token usage accounting
        usage = token_usage or {
            "input_tokens": self.state.input_tokens,
            "output_tokens": self.state.output_tokens or self.state.current_turn_tokens,
            "estimated": self.state.tokens_estimated,
        }
        in_tok = usage.get("input_tokens") or 0
        out_tok = usage.get("output_tokens") or usage.get("current_turn_output_tokens") or 0
        is_est = bool(usage.get("estimated", False))
        if in_tok > 0 or out_tok > 0:
            in_fmt = format_token_count(in_tok, concise=True)
            out_fmt = format_token_count(out_tok, concise=True)
            tag = "estimated" if is_est else "exact"
            self.console.print(f"    Tokens  {in_fmt} in, {out_fmt} out ({tag})", style="dim", highlight=False)

        # Files changed & stats
        stats_map = dict(file_change_stats or self.state.file_change_stats)
        file_list = files_changed or list(stats_map.keys())
        if file_list:
            parts = []
            for f in file_list:
                short_f = shorten_path(str(f))
                st = stats_map.get(f) or stats_map.get(str(f))
                if st and (st.get("added") or st.get("removed")):
                    parts.append(f"{short_f} (+{st['added']} -{st['removed']})")
                else:
                    parts.append(short_f)
            self.console.print(f"    Files  {', '.join(parts)}", style="dim", highlight=False)

        # Test results
        tests = test_results or self.state.test_summary
        if tests:
            passed = tests.get("passed", 0)
            failed = tests.get("failed", 0)
            if failed > 0:
                self.console.print(f"    Tests  {passed} passed, {failed} failed", style="red", highlight=False)
            else:
                self.console.print(f"    Tests  {passed} passed", style="green", highlight=False)
        elif not success and file_list:
            self.console.print("    Tests  None run (task stopped before tests)", style="dim", highlight=False)
            self.console.print("    Verification  Not completed (task stopped before verification)", style="dim", highlight=False)

        durations = phase_durations or self.state.phase_durations
        phase_labels = ("planning", "thinking", "working", "repairing", "verifying")
        summary = "  ".join(
            f"{phase.title()} {fmt_elapsed(durations[phase])}"
            for phase in phase_labels if durations.get(phase, 0) >= 0.5
        )
        if summary:
            self.console.print(f"    {summary}", style="dim", highlight=False)
        # Optional one-line error detail
        if error_detail and not success:
            self.console.print(f"  [dim]{error_detail}[/]", highlight=False)
        self.console.print("", highlight=False)

    @staticmethod
    def _model_display_name(model_id: str) -> str:
        aliases = {
            "google/gemma-4-31b-it:free": "Gemma 4 31B",
            "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B A4B",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "Nemotron 3 Nano",
            "openrouter/free": "OpenRouter Free",
        }
        return aliases.get(model_id, model_id)
