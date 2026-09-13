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

from harness_core.cli.ui import Color, Sym, fmt_elapsed


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
        short = str(path).replace("\\", "/").split("/")[-1] if path else "?"
        # For list_files: show path; for read etc: show full relative if short?
        full = str(path).replace("\\", "/")
        # Prefer filename only for readability, but keep directory for ambiguous
        if "/" in full and len(full) < 40:
            return f"{verb} {full}"
        return f"{verb} {short}"
    if tool == "list_files":
        p = args.get("path", ".")
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
class ConversationState:
    """Live conversation state — pure data, no Rich."""

    goal: str = ""
    intent: str = ""
    started_at: float = 0.0
    # Each entry: (icon, display, status)  status in running|success|failed
    items: list[tuple[str, str, str]] = field(default_factory=list)
    current_tool: str = ""
    iterations: int = 0
    tool_calls: int = 0
    phase: str = ""
    streaming_text: str = ""
    max_items: int = 12

    def add_running(self, display: str) -> None:
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
            "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0,
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
            started_at=time.time(),
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

        In rich mode with prompt_toolkit, the prompt is already visible from
        the input composer.  We print a blank line for spacing only.
        In plain mode, we print the prompt since input() doesn't leave it.
        """
        if self.plain:
            self.console.print(f"\n❯ {goal}", highlight=False)
        # Rich mode: prompt_toolkit already displayed ❯ goal.
        # Just add a blank line for visual breathing room.

    def _show_thinking(self) -> None:
        # No-op: intent is shown via _renderable(), not as a separate print.
        pass

    # ── event-driven updates ──────────────────────────────────────────

    def update_phase(self, phase: str) -> None:
        self.state.phase = phase
        self._refresh()

    def update_iterations(self, count: int) -> None:
        self.state.iterations = count
        self._refresh()

    def tool_started(self, tool: str, args: dict[str, Any]) -> None:
        display = _compact_tool_display(tool, args)
        # For run_command failures we want "Running pytest..." style while running
        if tool == "run_command":
            cmd = args.get("command", "")
            short = cmd[:45] + "..." if len(cmd) > 45 else cmd
            display = f"Run {short}" if short else display
        self.state.current_tool = tool
        self.state.add_running(display)
        self._refresh()

    def tool_completed(self, tool: str, status: str) -> None:
        self.state.complete_last(status)
        self.state.current_tool = ""
        self._refresh()

    def update_todos(self, completed: int, total: int) -> None:
        # No-op for now: TODO progress is NOT shown unless authoritative and non-fake.
        # Kept for API compat with existing callers.
        self._refresh()

    def update_todo_items(self, items: list[dict[str, Any]]) -> None:
        self._refresh()

    # ── rendering ─────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        if not self.state.started_at:
            return "0s"
        return fmt_elapsed(time.time() - self.state.started_at)

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
        """Build the live renderable: intent + compact tool activity.

        No 'Working...' placeholder.  No fake progress.  Just real activity.
        """
        lines: list[Any] = []

        # Intent message — short, user-safe execution intent (spec §5)
        if self.state.intent and not self._thinking_shown:
            self._thinking_shown = True
            lines.append(Text(f"    {self.state.intent}", style="dim"))
            lines.append(Text(""))

        # Tool activity (spec §6) — compact list
        for icon, display, status in self.state.items:
            txt = Text()
            tool_name = self._tool_name_from_display(display)
            if status == "running":
                tool_icon, tool_style = self._tool_icon(tool_name, "running")
                txt.append(f"  {tool_icon} ", style=tool_style)
                txt.append(display, style="dim")
                if tool_name == "run_command":
                    txt.append(" ...", style="dim")
            elif status == "success":
                tool_icon, tool_style = self._tool_icon(tool_name, "success")
                txt.append(f"  {tool_icon} ", style=tool_style)
                txt.append(display, style="dim")
            elif status in ("failed", "error"):
                txt.append(f"  {Sym.CROSS} ", style="red")
                txt.append(display, style="dim")
            else:
                txt.append(f"  {Sym.DOT} ", style="dim")
                txt.append(display, style="dim")
            lines.append(txt)

        # No placeholder.  If nothing is happening, show nothing.
        return Group(*lines) if lines else Text("")

    def _render_plain(self) -> None:
        # Plain mode: one line per tool, printed immediately
        if self.state.items:
            icon, display, status = self.state.items[-1]
            line = f"  {icon} {display} ({self._elapsed_str()})"
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
        clean = text.strip()
        if self.plain:
            for line in clean.splitlines():
                self.console.print(f"  {line}", highlight=False)
                if delay:
                    time.sleep(delay)
            self.console.print("", highlight=False)
            return

        # Rich mode: progressive reveal via Live + Markdown
        # We chunk by paragraphs for natural streaming cadence without
        # blocking the event loop for long.
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.padding import Padding

        paragraphs = clean.split("\n\n")
        shown: list[str] = []
        # Use a transient Live that updates in place, then leaves final Markdown
        try:
            with Live(console=self.console, refresh_per_second=12, transient=False) as live:
                for para in paragraphs:
                    shown.append(para)
                    md = Markdown("\n\n".join(shown))
                    live.update(Padding(md, (0, 0, 0, 2)))
                    if delay:
                        time.sleep(min(delay, 0.08))
                # Final frame stays
        except Exception:
            # Fallback: just print
            from rich.markdown import Markdown
            from rich.padding import Padding
            self.console.print(Padding(Markdown(clean), (0, 0, 0, 2)))
        self.console.print("", highlight=False)

    def render_completion(self, elapsed: float, success: bool = True, status: str = "completed", error_detail: str = "") -> None:
        """Tiny completion indicator.  One line.  Optional error detail."""
        elapsed_str = fmt_elapsed(elapsed)
        if status == "cancelled":
            mark, label, style = "⏹", "Cancelled", "yellow"
        elif status == "paused":
            mark, label, style = "⏸", "Paused", "yellow"
        elif status == "blocked":
            mark, label, style = Sym.CROSS, "Blocked", "red"
        elif status == "partial":
            mark, label, style = Sym.WARN, "Partial", "yellow"
        elif success:
            mark, label, style = Sym.CHECK, "Done", "green"
        else:
            mark, label, style = Sym.CROSS, "Failed", "red"
        if self.plain:
            self.console.print(f"  {mark} {label} · {elapsed_str}", highlight=False)
        else:
            self.console.print(f"  [{style}]{mark}[/] [dim]{label} · {elapsed_str}[/]", highlight=False)
        # Optional one-line error detail
        if error_detail and not success:
            self.console.print(f"  [dim]{error_detail}[/]", highlight=False)
        self.console.print("", highlight=False)
