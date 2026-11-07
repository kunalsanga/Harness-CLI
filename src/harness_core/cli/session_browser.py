"""Interactive session history browser and conversation transcript viewer for Harness CLI.

Provides:
- Keyboard-first interactive session browser (/sessions)
- Date-grouped session organization (Today, Yesterday, This Week, Month/Year)
- Interactive keyboard navigation (Up/Down, PageUp/PageDown, Enter, Esc, Search)
- Dedicated read-only historical conversation transcript viewer
- Explicit, separate resume action with user confirmation flow
- Truthful session status derivation based on persisted run/lifecycle state
"""

from __future__ import annotations

import datetime
import enum
import io
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from harness_core.session.domain import Run, RunStatus, Session, SessionStatus
from harness_core.session.storage import SessionStorage


# ── Status Derivation ──────────────────────────────────────────────────────

def derive_session_status(
    session: Session,
    storage: SessionStorage,
    current_session_id: str = "",
) -> str:
    """Derive accurate session status from persisted lifecycle and run outcomes.

    Returns: 'Active', 'Completed', 'Failed', 'Paused', 'Interrupted', or 'Archived'.
    Old sessions are never automatically labeled Active.
    """
    if session.status == SessionStatus.ARCHIVED:
        return "Archived"
    if session.status == SessionStatus.PAUSED:
        return "Paused"
    if getattr(session.status, "value", str(session.status)).lower() in ("interrupted", "aborted"):
        return "Interrupted"

    is_current = bool(current_session_id and session.session_id == current_session_id)

    runs = storage.get_runs(session.session_id) if storage else []
    if not runs:
        if is_current:
            return "Active"
        if session.status == SessionStatus.COMPLETED:
            return "Completed"
        if session.status == SessionStatus.FAILED:
            return "Failed"
        return "Interrupted"

    # Sort runs chronologically
    sorted_runs = sorted(runs, key=lambda r: getattr(r, "started_at", 0) or 0)
    latest_run = sorted_runs[-1]

    # Current session running in memory
    if is_current:
        return "Active"

    st = latest_run.status
    st_val = st.value if hasattr(st, "value") else str(st).lower()
    outcome = getattr(latest_run, "outcome", "").lower()

    if outcome == "failure" or st_val == "failed":
        return "Failed"
    elif outcome == "success" or st_val == "completed":
        return "Completed"
    elif st_val == "paused":
        return "Paused"
    elif st_val in ("running", "pending", "started", "interrupted"):
        return "Interrupted"

    # Fallback to session record if explicitly set
    if session.status in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.PAUSED, SessionStatus.ABORTED):
        return session.status.value.capitalize()

    return "Interrupted"


# ── Date Grouping ─────────────────────────────────────────────────────────

def _format_date_group(timestamp: float) -> str:
    """Classify a timestamp into human-readable relative date groups."""
    now = datetime.datetime.now()
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
    except Exception:
        return "Older"

    diff_days = (now.date() - dt.date()).days
    if diff_days == 0:
        return f"Today · {dt.strftime('%B %d')}"
    elif diff_days == 1:
        return f"Yesterday · {dt.strftime('%B %d')}"
    elif diff_days < 7:
        return f"This Week · {dt.strftime('%A, %B %d')}"
    elif dt.year == now.year:
        return dt.strftime("%B %Y")
    else:
        return dt.strftime("%B %Y")


def group_sessions_by_date(sessions: list[Session]) -> dict[str, list[Session]]:
    """Group sessions by relative date headers in newest-first order."""
    groups: dict[str, list[Session]] = {}
    for s in sessions:
        group_name = _format_date_group(s.updated_at or s.created_at)
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(s)
    return groups


# ── Static / Fallback Rendering ───────────────────────────────────────────

def render_session_browser(
    console: Console,
    sessions: list[Session],
    storage: SessionStorage,
    *,
    current_session_id: str = "",
    query: str = "",
    workspace_filter: str = "",
    status_filter: str = "",
    plain: bool = False,
) -> None:
    """Render a rich, date-grouped static session list (used for plain mode or non-interactive terminals)."""
    filtered = []
    q_low = query.strip().lower()
    for s in sessions:
        if workspace_filter and workspace_filter.lower() not in s.workspace_path.lower():
            continue
        derived_status = derive_session_status(s, storage, current_session_id)
        if status_filter and status_filter.lower() != derived_status.lower():
            continue
        if q_low:
            in_meta = (
                q_low in s.title.lower()
                or q_low in s.session_id.lower()
                or q_low in s.workspace_path.lower()
            )
            runs = storage.get_runs(s.session_id)
            in_runs = any(q_low in r.task.lower() for r in runs)
            if not (in_meta or in_runs):
                continue
        filtered.append(s)

    if plain:
        console.print("=== HARNESS · Session History ===")
        if query:
            console.print(f"Filter: {query}")
        if not filtered:
            console.print("No sessions found.")
            return
        for s in filtered:
            cur = "*" if s.session_id == current_session_id else " "
            dt = datetime.datetime.fromtimestamp(s.created_at).strftime("%Y-%m-%d %H:%M")
            st = derive_session_status(s, storage, current_session_id)
            console.print(f"{cur} [{s.session_id[:10]}] {s.title} ({st}) - {dt}", markup=False)
        console.print("\nCommands: /sessions view <id>, /sessions resume <id>, /sessions new")
        return

    console.print("")
    header = Text()
    header.append("  ◆ ", style="cyan bold")
    header.append("HARNESS", style="bold white")
    header.append(" · Session History", style="dim")
    console.print(header)

    if query:
        console.print(f"  [dim]Searching for:[/] [cyan]{query}[/]")
    if workspace_filter:
        console.print(f"  [dim]Workspace filter:[/] [cyan]{workspace_filter}[/]")
    console.print("")

    if not filtered:
        console.print("  [dim]No sessions found matching your criteria.[/]")
        console.print("  [dim]Type [cyan]/new[/dim] to start a new engineering session.[/]\n")
        return

    grouped = group_sessions_by_date(filtered)

    for group_name, sess_list in grouped.items():
        console.print(f"  [bold cyan]{group_name.upper()}[/]", highlight=False)
        table = Table(
            box=box.SIMPLE,
            pad_edge=False,
            show_header=True,
            header_style="dim",
            collapse_padding=True,
        )
        table.add_column("", width=2)
        table.add_column("Session ID", style="cyan", width=12)
        table.add_column("Title / Task", width=34)
        table.add_column("Workspace", style="dim", width=18)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Runs", justify="right", width=6)
        table.add_column("Status", width=12)

        for s in sess_list:
            is_current = (s.session_id == current_session_id)
            cur_marker = "[bold cyan]●[/]" if is_current else " "

            st_val = derive_session_status(s, storage, current_session_id)
            status_style = {
                "active": "cyan bold",
                "completed": "green",
                "paused": "yellow",
                "failed": "red",
                "interrupted": "dim yellow",
                "archived": "dim",
            }.get(st_val.lower(), "white")
            status_text = f"[{status_style}]{st_val}[/]"

            runs = storage.get_runs(s.session_id)
            run_count_str = str(len(runs))
            time_str = datetime.datetime.fromtimestamp(s.updated_at or s.created_at).strftime("%I:%M %p")

            ws_name = Path(s.workspace_path).name or s.workspace_path
            if len(ws_name) > 16:
                ws_name = ws_name[:14] + "…"

            title_display = s.title
            if not title_display and runs:
                title_display = runs[0].task.replace("\n", " ").strip()
            if not title_display:
                title_display = "Untitled Session"
            if len(title_display) > 32:
                title_display = title_display[:30] + "…"

            table.add_row(
                cur_marker,
                s.session_id[:10],
                title_display,
                ws_name,
                time_str,
                run_count_str,
                status_text,
            )

        console.print(table)
        console.print("")

    footer = Text.from_markup(
        "  [dim]Actions:[/] [cyan]/sessions view <id>[/] [dim]View transcript[/]  ·  "
        "[cyan]/sessions resume <id>[/] [dim]Resume task[/]  ·  "
        "[cyan]/new[/] [dim]New session[/]"
    )
    console.print(footer)
    console.print("")


def render_historical_transcript(
    console: Console,
    session_id: str,
    storage: SessionStorage,
    *,
    plain: bool = False,
) -> None:
    """Render a clean, read-only historical conversation transcript."""
    session = storage.get_session(session_id)
    if session is None:
        all_s = storage.list_sessions(limit=100)
        matching = [s for s in all_s if s.session_id.startswith(session_id)]
        if len(matching) == 1:
            session = matching[0]
            session_id = session.session_id
        elif len(matching) > 1:
            console.print(f"  [yellow]Ambiguous session ID prefix '{session_id}'. Please specify more characters.[/]")
            return
        else:
            console.print(f"  [red]Session '{session_id}' not found in history.[/]")
            return

    runs = storage.get_runs(session_id)
    events = storage.get_events(session_id, limit=300)
    checkpoints = storage.get_checkpoints(session_id)
    st = derive_session_status(session, storage)

    runs.sort(key=lambda r: getattr(r, "started_at", 0) or 0)

    if plain:
        console.print(f"=== Session Transcript: {session.title} ({session_id[:10]}) ===")
        console.print(f"Workspace: {session.workspace_path} | Status: {st}")
        for idx, r in enumerate(runs, 1):
            console.print(f"\n--- Run {idx}: {r.task} ---")
            console.print(f"Model: {r.model_id or r.provider or 'auto'}")
            if r.result_summary:
                console.print(f"Assistant: {r.result_summary}")
            console.print(f"Result: {r.status.value} (verification: {'pass' if r.verification_passed else 'fail'})")
        console.print("\n=== End of Transcript ===")
        return

    console.print("")
    header_content = Text()
    header_content.append("◆ HARNESS · Conversation History (Read-Only)\n", style="bold cyan")
    header_content.append(f"Session: {session.title or 'Untitled'} ({session.session_id[:10]})\n", style="white")
    header_content.append(f"Workspace: {session.workspace_path}  ·  Status: {st}\n", style="dim")
    dt = datetime.datetime.fromtimestamp(session.created_at).strftime("%B %d, %Y at %I:%M %p")
    header_content.append(f"Created: {dt}  ·  Total Runs: {len(runs)}", style="dim")
    console.print(Panel(header_content, border_style="dim", box=box.ROUNDED))
    console.print("")

    if not runs:
        console.print("  [dim]No runs recorded in this session.[/]\n")
        return

    for idx, r in enumerate(runs, 1):
        dt_str = datetime.datetime.fromtimestamp(r.started_at).strftime("%I:%M %p") if r.started_at else ""
        elapsed_s = (r.duration_ms / 1000.0) if r.duration_ms > 0 else 0.0
        model_name = r.model_id or r.provider or "auto"
        tool_count = r.tool_calls

        console.print(f"  [dim]── Run {idx} · {dt_str} · {elapsed_s:.1f}s · {tool_count} tool calls · {model_name} ──[/]")
        console.print(f"  [bold cyan]❯[/] [bold white]{r.task}[/]", highlight=False)
        console.print("")

        run_events = [e for e in events if e.data.get("run_id") == r.run_id or e.data.get("task") == r.task]
        tool_results = [e for e in run_events if e.event_type in ("tool.result", "tool.completed", "tool.failed")]
        if tool_results:
            console.print("  [dim]Tool Activity[/]")
            for tr in tool_results[:8]:
                tname = tr.data.get("tool", "tool")
                tstatus = tr.data.get("status", "success")
                icon = "✓" if tstatus == "success" else "✗"
                tstyle = "green" if tstatus == "success" else "red"
                console.print(f"    [{tstyle}]{icon}[/] [dim]{tname}[/]")
            if len(tool_results) > 8:
                console.print(f"    [dim]└─ +{len(tool_results) - 8} more tool calls[/]")
            console.print("")

        if r.result_summary:
            try:
                md = Markdown(r.result_summary.strip())
                console.print(Panel(md, border_style="dim", box=box.SIMPLE))
            except Exception:
                console.print(f"  {r.result_summary}")
            console.print("")

        success_icon = "✓" if r.status == RunStatus.COMPLETED and r.outcome != "failure" else "✗"
        success_style = "green" if success_icon == "✓" else "red"
        status_label = "Completed" if success_icon == "✓" else "Failed"

        summary_parts = [
            f"[{success_style}]{success_icon} Run {idx} {status_label}[/]",
            f"[dim]{elapsed_s:.1f}s[/]",
            f"[dim]{r.tool_calls} tool calls[/]",
        ]
        if r.verification_passed:
            summary_parts.append("[green]Verification Passed[/]")
        elif success_icon == "✓":
            summary_parts.append("[dim]Verification: Not applicable[/]")
        else:
            summary_parts.append("[red]Verification Failed[/]")

        console.print("  " + "  ·  ".join(summary_parts), highlight=False)
        console.print("  " + "─" * min(60, console.size.width - 4), style="dim")
        console.print("")

    console.print("  [dim]End of transcript (read-only).[/]")
    console.print(f"  [dim]To continue this work, type:[/] [cyan]/sessions resume {session.session_id[:10]}[/]")
    console.print(f"  [dim]To browse all sessions, type:[/] [cyan]/sessions[/]\n")


# ── Interactive Keyboard-First Session Browser ────────────────────────────

class BrowserViewMode(enum.Enum):
    SESSIONS_LIST = "sessions_list"
    TRANSCRIPT_VIEWER = "transcript_viewer"
    CONFIRM_RESUME = "confirm_resume"


class InteractiveSessionBrowser:
    """Keyboard-first interactive session history browser and transcript viewer.

    Supports:
    - Arrow key navigation (Up/Down, PageUp/PageDown, Home/End)
    - Date-grouped session hierarchy
    - Enter to open historical conversation transcript
    - Esc/Back to return to session browser
    - Search filtering (/)
    - Explicit resume action (R) with confirmation flow
    - 100% read-only historical viewing: 0 model calls, 0 tool executions
    """

    def __init__(
        self,
        console: Console,
        storage: SessionStorage,
        *,
        current_session_id: str = "",
        workspace_filter: str = "",
        initial_query: str = "",
        plain: bool = False,
        input: Any = None,
        output: Any = None,
    ) -> None:
        self.console = console
        self.storage = storage
        self.current_session_id = current_session_id
        self.workspace_filter = workspace_filter
        self.search_query = initial_query.strip()
        self.search_active = False
        self.search_buffer = initial_query.strip()
        self.plain = plain

        self._custom_input = input
        self._custom_output = output

        self.mode = BrowserViewMode.SESSIONS_LIST
        self.selected_index = 0
        self.transcript_scroll_offset = 0
        self.transcript_search_active = False
        self.transcript_search_buffer = ""
        self.transcript_search_query = ""
        self._cached_transcript_lines: list[str] = []
        self.selected_session: Session | None = None
        self._result: dict[str, Any] | None = None

        self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        """Fetch and filter sessions from storage."""
        all_sessions = self.storage.list_sessions(limit=150)
        q_low = self.search_query.lower()

        filtered: list[Session] = []
        for s in all_sessions:
            if self.workspace_filter and self.workspace_filter.lower() not in s.workspace_path.lower():
                continue
            if q_low:
                in_meta = (
                    q_low in s.title.lower()
                    or q_low in s.session_id.lower()
                    or q_low in s.workspace_path.lower()
                )
                runs = self.storage.get_runs(s.session_id)
                in_runs = any(q_low in r.task.lower() for r in runs)
                if not (in_meta or in_runs):
                    continue
            filtered.append(s)

        # Sort newest first
        filtered.sort(key=lambda s: s.updated_at or s.created_at, reverse=True)
        self.sessions = filtered
        if self.sessions:
            self.selected_index = max(0, min(self.selected_index, len(self.sessions) - 1))
            self.selected_session = self.sessions[self.selected_index]
        else:
            self.selected_index = 0
            self.selected_session = None

    def _render_sessions_list(self) -> str:
        """Render the interactive sessions list view using Rich to ANSI string."""
        buf = io.StringIO()
        width = max(60, min(120, getattr(self.console.size, "width", 100)))
        con = Console(file=buf, force_terminal=True, color_system="truecolor", width=width)

        # Header
        header = Text()
        header.append("  ◆ ", style="cyan bold")
        header.append("HARNESS", style="bold white")
        header.append(" · Session History", style="bold")
        if self.workspace_filter:
            ws_short = Path(self.workspace_filter).name or self.workspace_filter
            header.append(f"  ·  Workspace: {ws_short}", style="dim")
        con.print(header)

        # Search Bar
        if self.search_active:
            con.print(f"  [cyan bold]Search:[/] [white underline]{self.search_buffer}[/][cyan]█[/] [dim](Enter to apply, Esc to cancel)[/]")
        elif self.search_query:
            con.print(f"  [dim]Filter active:[/] [cyan]{self.search_query}[/] [dim](press / to edit, Esc to clear)[/]")
        else:
            con.print("  [dim]Press \\[/] to search/filter sessions[/]")
        con.print("")

        if not self.sessions:
            con.print("  [dim]No sessions found matching criteria.[/]")
            con.print("  [dim]Press Esc to return to prompt.[/]\n")
            return buf.getvalue()

        # Date grouped sessions list
        # Map flattened indices
        grouped = group_sessions_by_date(self.sessions)
        flat_idx = 0

        # Available height budget
        max_rows = 14
        start_row = max(0, self.selected_index - (max_rows // 2))
        end_row = start_row + max_rows

        for group_name, sess_list in grouped.items():
            con.print(f"  [bold cyan]{group_name.upper()}[/]", highlight=False)
            table = Table(
                box=box.SIMPLE,
                pad_edge=False,
                show_header=True,
                header_style="dim",
                collapse_padding=True,
            )
            table.add_column("", width=2)
            table.add_column("Session ID", width=12)
            table.add_column("Title / Task", width=34)
            table.add_column("Workspace", style="dim", width=16)
            table.add_column("Time", style="dim", width=10)
            table.add_column("Runs", justify="right", width=6)
            table.add_column("Status", width=14)

            for s in sess_list:
                is_selected = (flat_idx == self.selected_index)
                is_current = (s.session_id == self.current_session_id)

                cur_icon = "▶" if is_selected else ("●" if is_current else " ")
                cur_style = "bold cyan" if is_selected else ("dim cyan" if is_current else "dim")

                st_val = derive_session_status(s, self.storage, self.current_session_id)
                status_style = {
                    "active": "cyan bold",
                    "completed": "green",
                    "paused": "yellow",
                    "failed": "red",
                    "interrupted": "dim yellow",
                    "archived": "dim",
                }.get(st_val.lower(), "white")

                runs = self.storage.get_runs(s.session_id)
                run_count_str = str(len(runs))
                time_str = datetime.datetime.fromtimestamp(s.updated_at or s.created_at).strftime("%I:%M %p")

                ws_name = Path(s.workspace_path).name or s.workspace_path
                if len(ws_name) > 14:
                    ws_name = ws_name[:12] + "…"

                title_display = s.title
                if not title_display and runs:
                    title_display = runs[0].task.replace("\n", " ").strip()
                if not title_display:
                    title_display = "Untitled Session"
                if len(title_display) > 32:
                    title_display = title_display[:30] + "…"

                row_style = "reverse bold" if is_selected else None

                table.add_row(
                    f"[{cur_style}]{cur_icon}[/]",
                    f"[cyan]{s.session_id[:10]}[/]",
                    title_display,
                    ws_name,
                    time_str,
                    run_count_str,
                    f"[{status_style}]{st_val}[/]",
                    style=row_style,
                )
                flat_idx += 1

            con.print(table)
            con.print("")

        # Selected Session Detail Box
        if self.selected_session:
            sel = self.selected_session
            st_sel = derive_session_status(sel, self.storage, self.current_session_id)
            runs_sel = self.storage.get_runs(sel.session_id)
            last_goal = runs_sel[-1].task if runs_sel else "None"
            if len(last_goal) > 60:
                last_goal = last_goal[:58] + "…"

            started_str = datetime.datetime.fromtimestamp(sel.created_at).strftime("%B %d, %Y at %I:%M %p")

            detail_text = Text()
            detail_text.append(f"  Selected: ", style="bold dim")
            detail_text.append(f"{sel.session_id[:10]}  ", style="bold cyan")
            detail_text.append(f"[{st_sel}]  ", style="bold green" if st_sel == "Completed" else "bold red" if st_sel == "Failed" else "bold cyan")
            detail_text.append(f"· Workspace: {sel.workspace_path}\n", style="dim")
            detail_text.append(f"  Started: {started_str}  ·  Runs: {len(runs_sel)}  ·  Last Goal: {last_goal}", style="dim")
            con.print(Panel(detail_text, border_style="dim", box=box.ROUNDED))

        # Action Footer
        con.print(
            "  [bold cyan][Enter][/] [white]Open conversation[/]   "
            "[bold cyan][R][/] [white]Resume session[/]   "
            "[bold cyan]\\[/][/] [white]Search[/]   "
            "[dim][↑/↓] Navigate   [Esc] Exit[/]"
        )

        return buf.getvalue()

    def _render_transcript_viewer(self) -> str:
        """Render the historical transcript viewer view using Rich to ANSI string."""
        buf = io.StringIO()
        width = max(60, min(120, getattr(self.console.size, "width", 100)))
        con = Console(file=buf, force_terminal=True, color_system="truecolor", width=width)

        if not self.selected_session:
            con.print("  [dim]No session selected.[/]")
            return buf.getvalue()

        session = self.selected_session
        runs = self.storage.get_runs(session.session_id)
        runs.sort(key=lambda r: getattr(r, "started_at", 0) or 0)
        events = self.storage.get_events(session.session_id, limit=300)
        st = derive_session_status(session, self.storage, self.current_session_id)

        # Header Panel
        header_content = Text()
        header_content.append("◆ HARNESS · Conversation History (Read-Only)\n", style="bold cyan")
        header_content.append(f"Session: {session.title or 'Untitled'} ({session.session_id[:10]})  ·  Status: {st}\n", style="white")
        header_content.append(f"Workspace: {session.workspace_path}  ·  Total Runs: {len(runs)}", style="dim")
        con.print(Panel(header_content, border_style="cyan", box=box.ROUNDED))
        con.print("")

        # Transcript Search Bar
        if self.transcript_search_active:
            con.print(f"  [cyan bold]Search transcript:[/] [white underline]{self.transcript_search_buffer}[/][cyan]█[/] [dim](Enter to apply, Esc to cancel)[/]")
            con.print("")
        elif self.transcript_search_query:
            con.print(f"  [dim]Filter active:[/] [cyan]{self.transcript_search_query}[/] [dim](press / to edit, n for next, Esc to clear)[/]")
            con.print("")

        if not runs:
            con.print("  [dim]No runs recorded in this session.[/]\n")
            con.print("  [dim]Press [Esc] or [Back] to return to session browser.[/]")
            return buf.getvalue()

        # Render runs
        full_content = io.StringIO()
        body_con = Console(file=full_content, force_terminal=True, color_system="truecolor", width=width)

        for idx, r in enumerate(runs, 1):
            dt_str = datetime.datetime.fromtimestamp(r.started_at).strftime("%I:%M %p") if r.started_at else ""
            elapsed_s = (r.duration_ms / 1000.0) if r.duration_ms > 0 else 0.0
            model_name = r.model_id or r.provider or "auto"
            tool_count = r.tool_calls

            body_con.print(f"  [dim]── Run {idx} · {dt_str} · {elapsed_s:.1f}s · {tool_count} tool calls · {model_name} ──[/]")
            body_con.print(f"  [bold cyan]❯[/] [bold white]{r.task}[/]", highlight=False)
            body_con.print("")

            run_events = [e for e in events if e.data.get("run_id") == r.run_id or e.data.get("task") == r.task]
            tool_results = [e for e in run_events if e.event_type in ("tool.result", "tool.completed", "tool.failed")]
            if tool_results:
                body_con.print("  [dim]Tool Activity[/]")
                for tr in tool_results[:10]:
                    tname = tr.data.get("tool", "tool")
                    tstatus = tr.data.get("status", "success")
                    icon = "✓" if tstatus == "success" else "✗"
                    tstyle = "green" if tstatus == "success" else "red"
                    body_con.print(f"    [{tstyle}]{icon}[/] [dim]{tname}[/]")
                if len(tool_results) > 10:
                    body_con.print(f"    [dim]└─ +{len(tool_results) - 10} more tool calls[/]")
                body_con.print("")

            if r.result_summary:
                try:
                    md = Markdown(r.result_summary.strip())
                    body_con.print(Panel(md, border_style="dim", box=box.SIMPLE))
                except Exception:
                    body_con.print(f"  {r.result_summary}")
                body_con.print("")

            success_icon = "✓" if r.status == RunStatus.COMPLETED and r.outcome != "failure" else "✗"
            success_style = "green" if success_icon == "✓" else "red"
            status_label = "Completed" if success_icon == "✓" else "Failed"

            summary_parts = [
                f"[{success_style}]{success_icon} Run {idx} {status_label}[/]",
                f"[dim]{elapsed_s:.1f}s[/]",
                f"[dim]{r.tool_calls} tool calls[/]",
            ]
            if r.verification_passed:
                summary_parts.append("[green]Verification Passed[/]")
            elif success_icon == "✓":
                summary_parts.append("[dim]Verification: Not applicable[/]")
            else:
                summary_parts.append("[red]Verification Failed[/]")

            body_con.print("  " + "  ·  ".join(summary_parts), highlight=False)
            body_con.print("  " + "─" * min(60, width - 4), style="dim")
            body_con.print("")

        lines = full_content.getvalue().splitlines()
        self._cached_transcript_lines = lines
        max_visible_lines = 18
        max_offset = max(0, len(lines) - max_visible_lines)
        self.transcript_scroll_offset = max(0, min(self.transcript_scroll_offset, max_offset))

        visible = lines[self.transcript_scroll_offset : self.transcript_scroll_offset + max_visible_lines]
        con.print("\n".join(visible))
        con.print("")

        # Transcript Navigation Footer
        scroll_pct = f"{self.transcript_scroll_offset + 1}-{min(len(lines), self.transcript_scroll_offset + max_visible_lines)} of {len(lines)} lines" if lines else "0 lines"
        con.print(
            f"  [dim]Lines: {scroll_pct}[/]   "
            "[bold cyan][↑/↓][/] [white]Scroll[/]   "
            "[bold cyan][PageUp/PageDown][/] [white]Page[/]   "
            "[bold cyan]\\[/][/] [white]Search[/]   "
            "[bold cyan][R][/] [white]Resume session[/]   "
            "[bold cyan][Esc / Back][/] [white]Return to Sessions[/]"
        )

        return buf.getvalue()

    def _jump_transcript_search(self, forward: bool = True) -> None:
        """Jump to next or previous search match in the transcript."""
        if not self.transcript_search_query or not self._cached_transcript_lines:
            return
        q_low = self.transcript_search_query.lower()
        indices = [i for i, line in enumerate(self._cached_transcript_lines) if q_low in line.lower()]
        if not indices:
            return
        if forward:
            next_indices = [i for i in indices if i > self.transcript_scroll_offset]
            target = next_indices[0] if next_indices else indices[0]
        else:
            prev_indices = [i for i in indices if i < self.transcript_scroll_offset]
            target = prev_indices[-1] if prev_indices else indices[-1]
        self.transcript_scroll_offset = target

    def _render_confirm_resume(self) -> str:
        """Render the explicit resume confirmation dialog using Rich to ANSI string."""
        buf = io.StringIO()
        width = max(60, min(100, getattr(self.console.size, "width", 80)))
        con = Console(file=buf, force_terminal=True, color_system="truecolor", width=width)

        if not self.selected_session:
            return ""

        sel = self.selected_session
        runs = self.storage.get_runs(sel.session_id)
        last_task = runs[-1].task if runs else "None"
        st = derive_session_status(sel, self.storage, self.current_session_id)

        content = Text()
        content.append("◆ Resume Engineering Session?\n\n", style="bold cyan")
        content.append(f"  Session:   {sel.title} ({sel.session_id[:10]})\n", style="bold white")
        content.append(f"  Workspace: {sel.workspace_path}\n", style="dim")
        content.append(f"  Status:    {st}\n", style="dim")
        content.append(f"  Last Goal: {last_task}\n\n", style="dim")
        content.append("  The agent will restore this session's context, memory, and checkpoint,\n", style="white")
        content.append("  reconcile the workspace, and continue working in this session.\n\n", style="white")
        content.append("  Confirm resume? [Y] Yes, resume session   [N / Esc] Cancel", style="bold yellow")

        con.print("")
        con.print(Panel(content, border_style="yellow", box=box.ROUNDED))
        con.print("")

        return buf.getvalue()

    def _render_current_view(self) -> Any:
        """Dispatcher returning prompt_toolkit ANSI formatted text for current view."""
        from prompt_toolkit.formatted_text import ANSI

        if self.mode == BrowserViewMode.SESSIONS_LIST:
            raw_ansi = self._render_sessions_list()
        elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
            raw_ansi = self._render_transcript_viewer()
        elif self.mode == BrowserViewMode.CONFIRM_RESUME:
            raw_ansi = self._render_confirm_resume()
        else:
            raw_ansi = ""

        return ANSI(raw_ansi)

    async def run_async(self) -> dict[str, Any] | None:
        """Execute the interactive event loop and return selected action or None."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        kb = KeyBindings()

        @kb.add("c-c")
        def _handle_sigint(event: Any) -> None:
            event.app.exit(result=None)

        @kb.add("up")
        def _handle_up(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST:
                if not self.search_active and self.sessions:
                    self.selected_index = max(0, self.selected_index - 1)
                    self.selected_session = self.sessions[self.selected_index]
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                self.transcript_scroll_offset = max(0, self.transcript_scroll_offset - 1)
                event.app.invalidate()

        @kb.add("k")
        def _handle_k(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "k"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and self.transcript_search_active:
                self.transcript_search_buffer += "k"
                self.transcript_search_query = self.transcript_search_buffer
                event.app.invalidate()
            else:
                _handle_up(event)

        @kb.add("down")
        def _handle_down(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST:
                if not self.search_active and self.sessions:
                    self.selected_index = min(len(self.sessions) - 1, self.selected_index + 1)
                    self.selected_session = self.sessions[self.selected_index]
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                self.transcript_scroll_offset += 1
                event.app.invalidate()

        @kb.add("j")
        def _handle_j(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "j"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and self.transcript_search_active:
                self.transcript_search_buffer += "j"
                self.transcript_search_query = self.transcript_search_buffer
                event.app.invalidate()
            else:
                _handle_down(event)

        @kb.add("pageup")
        def _handle_pageup(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST:
                if not self.search_active and self.sessions:
                    self.selected_index = max(0, self.selected_index - 5)
                    self.selected_session = self.sessions[self.selected_index]
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                self.transcript_scroll_offset = max(0, self.transcript_scroll_offset - 10)
                event.app.invalidate()

        @kb.add("pagedown")
        def _handle_pagedown(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST:
                if not self.search_active and self.sessions:
                    self.selected_index = min(len(self.sessions) - 1, self.selected_index + 5)
                    self.selected_session = self.sessions[self.selected_index]
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                self.transcript_scroll_offset += 10
                event.app.invalidate()

        @kb.add("home")
        def _handle_home(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.sessions:
                self.selected_index = 0
                self.selected_session = self.sessions[0]
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                self.transcript_scroll_offset = 0
                event.app.invalidate()

        @kb.add("end")
        def _handle_end(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.sessions:
                self.selected_index = len(self.sessions) - 1
                self.selected_session = self.sessions[-1]
                event.app.invalidate()

        @kb.add("enter")
        def _handle_enter(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST:
                if self.search_active:
                    self.search_active = False
                    event.app.invalidate()
                elif self.selected_session:
                    self.mode = BrowserViewMode.TRANSCRIPT_VIEWER
                    self.transcript_scroll_offset = 0
                    self.transcript_search_active = False
                    self.transcript_search_query = ""
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                if self.transcript_search_active:
                    self.transcript_search_active = False
                    self._jump_transcript_search(forward=True)
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.CONFIRM_RESUME:
                if self.selected_session:
                    event.app.exit(result={"action": "resume", "session_id": self.selected_session.session_id})

        @kb.add("escape")
        def _handle_escape(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST:
                if self.search_active:
                    self.search_active = False
                    self.search_buffer = ""
                    self.search_query = ""
                    self._refresh_sessions()
                    event.app.invalidate()
                elif self.search_query:
                    self.search_query = ""
                    self.search_buffer = ""
                    self._refresh_sessions()
                    event.app.invalidate()
                else:
                    event.app.exit(result=None)
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                if self.transcript_search_active:
                    self.transcript_search_active = False
                    self.transcript_search_buffer = ""
                    self.transcript_search_query = ""
                    event.app.invalidate()
                elif self.transcript_search_query:
                    self.transcript_search_query = ""
                    event.app.invalidate()
                else:
                    self.mode = BrowserViewMode.SESSIONS_LIST
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.CONFIRM_RESUME:
                self.mode = BrowserViewMode.SESSIONS_LIST
                event.app.invalidate()

        @kb.add("r")
        @kb.add("R")
        def _handle_resume_key(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "r"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and self.transcript_search_active:
                self.transcript_search_buffer += "r"
                self.transcript_search_query = self.transcript_search_buffer
                event.app.invalidate()
            elif self.mode in (BrowserViewMode.SESSIONS_LIST, BrowserViewMode.TRANSCRIPT_VIEWER):
                if self.selected_session:
                    self.mode = BrowserViewMode.CONFIRM_RESUME
                    event.app.invalidate()

        @kb.add("y")
        @kb.add("Y")
        def _handle_confirm_yes(event: Any) -> None:
            if self.mode == BrowserViewMode.CONFIRM_RESUME and self.selected_session:
                event.app.exit(result={"action": "resume", "session_id": self.selected_session.session_id})
            elif self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "y"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and self.transcript_search_active:
                self.transcript_search_buffer += "y"
                self.transcript_search_query = self.transcript_search_buffer
                event.app.invalidate()

        @kb.add("n")
        def _handle_n(event: Any) -> None:
            if self.mode == BrowserViewMode.CONFIRM_RESUME:
                self.mode = BrowserViewMode.SESSIONS_LIST
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                if self.transcript_search_active:
                    self.transcript_search_buffer += "n"
                    self.transcript_search_query = self.transcript_search_buffer
                    event.app.invalidate()
                elif self.transcript_search_query:
                    self._jump_transcript_search(forward=True)
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "n"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()

        @kb.add("N")
        def _handle_capital_n(event: Any) -> None:
            if self.mode == BrowserViewMode.CONFIRM_RESUME:
                self.mode = BrowserViewMode.SESSIONS_LIST
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                if self.transcript_search_active:
                    self.transcript_search_buffer += "N"
                    self.transcript_search_query = self.transcript_search_buffer
                    event.app.invalidate()
                elif self.transcript_search_query:
                    self._jump_transcript_search(forward=False)
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "N"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()

        @kb.add("backspace")
        def _handle_backspace(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                if self.search_buffer:
                    self.search_buffer = self.search_buffer[:-1]
                    self.search_query = self.search_buffer
                    self._refresh_sessions()
                    event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                if self.transcript_search_active:
                    if self.transcript_search_buffer:
                        self.transcript_search_buffer = self.transcript_search_buffer[:-1]
                        self.transcript_search_query = self.transcript_search_buffer
                        event.app.invalidate()
                else:
                    self.mode = BrowserViewMode.SESSIONS_LIST
                    event.app.invalidate()

        @kb.add("/")
        def _handle_search(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and not self.search_active:
                self.search_active = True
                self.search_buffer = ""
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and not self.transcript_search_active:
                self.transcript_search_active = True
                self.transcript_search_buffer = ""
                event.app.invalidate()

        @kb.add("q")
        @kb.add("Q")
        def _handle_quit(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                self.search_buffer += "q"
                self.search_query = self.search_buffer
                self._refresh_sessions()
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and self.transcript_search_active:
                self.transcript_search_buffer += "q"
                self.transcript_search_query = self.transcript_search_buffer
                event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER:
                self.mode = BrowserViewMode.SESSIONS_LIST
                event.app.invalidate()
            elif self.mode == BrowserViewMode.CONFIRM_RESUME:
                self.mode = BrowserViewMode.SESSIONS_LIST
                event.app.invalidate()
            elif self.mode == BrowserViewMode.SESSIONS_LIST:
                event.app.exit(result=None)

        # Typing in search mode
        @kb.add("<any>")
        def _handle_any_char(event: Any) -> None:
            if self.mode == BrowserViewMode.SESSIONS_LIST and self.search_active:
                for k in event.key_sequence:
                    if hasattr(k, "data") and k.data and len(k.data) == 1 and k.data.isprintable():
                        self.search_buffer += k.data
                        self.search_query = self.search_buffer
                        self._refresh_sessions()
                        event.app.invalidate()
            elif self.mode == BrowserViewMode.TRANSCRIPT_VIEWER and self.transcript_search_active:
                for k in event.key_sequence:
                    if hasattr(k, "data") and k.data and len(k.data) == 1 and k.data.isprintable():
                        self.transcript_search_buffer += k.data
                        self.transcript_search_query = self.transcript_search_buffer
                        event.app.invalidate()

        control = FormattedTextControl(self._render_current_view)
        layout = Layout(Window(content=control))

        app_kwargs: dict[str, Any] = {
            "layout": layout,
            "key_bindings": kb,
            "full_screen": True if self._custom_output is None else False,
        }
        if self._custom_input is not None:
            app_kwargs["input"] = self._custom_input
        if self._custom_output is not None:
            app_kwargs["output"] = self._custom_output

        app = Application(**app_kwargs)
        try:
            return await app.run_async()
        except Exception:
            return None
