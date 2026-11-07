"""Unit regression tests for read-only explanation tasks and Windows shell reliability.

Covers:
1. Windows shell command normalization and PowerShell statement chain translation (&&, ||).
2. Consecutive read tool aggregation preserving truthful success counts when a sub-call fails.
3. Plan reconciliation ensuring summary/synthesis steps are never falsely completed on failed tasks.
4. VerificationStatus.SKIPPED for read-only tasks ensuring verification is never reported as Failed.
5. ReadFileTool directory diagnostics and read-only inspection bounds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from harness_core.agent.todos import (
    reconcile_on_evidence,
    score_todo,
    todo_category,
)
from harness_core.agent.types import TaskPlan, TodoItem, TodoStatus
from harness_core.cli.conversation import ActivityItem, ConversationState, Sym
from harness_core.runtime.state import VerificationStatus
from harness_core.tools.filesystem import ReadFileTool
from harness_core.tools.shell import (
    normalize_windows_command,
    translate_powershell_chain,
)


# ── 1. Windows Shell & PowerShell Translation ─────────────────────────────

def test_translate_powershell_chain_simple() -> None:
    translated = translate_powershell_chain("cmd1 && cmd2")
    assert translated == "cmd1; if ($?) { cmd2 }"


def test_translate_powershell_chain_multiple() -> None:
    translated = translate_powershell_chain("cmd1 && cmd2 && cmd3")
    assert translated == "cmd1; if ($?) { cmd2 }; if ($?) { cmd3 }"


def test_translate_powershell_chain_preserves_quoted_operators() -> None:
    cmd = 'git commit -m "fix: bug && test" && pytest'
    translated = translate_powershell_chain(cmd)
    assert 'git commit -m "fix: bug && test"' in translated
    assert translated.endswith('; if ($?) { pytest }')


def test_translate_powershell_chain_single_quotes() -> None:
    cmd = "echo 'hello && world' && exit 0"
    translated = translate_powershell_chain(cmd)
    assert "echo 'hello && world'" in translated
    assert translated.endswith("; if ($?) { exit 0 }")


def test_normalize_windows_command_strips_cwd_with_spaces() -> None:
    cwd = r"C:\Users\kunal sanga\Desktop\New folder\app1"
    cmd = r'cd "C:\Users\kunal sanga\Desktop\New folder\app1" && pytest'
    norm = normalize_windows_command(cmd, cwd)
    assert norm == "pytest"


def test_normalize_windows_command_subpath_quotes_safely() -> None:
    cwd = r"C:\Users\kunal sanga\Desktop\New folder\app1"
    cmd = r"cd backend && pytest"
    norm = normalize_windows_command(cmd, cwd)
    # Should use Set-Location with literal path and translate && to PowerShell syntax
    assert "Set-Location -LiteralPath 'backend'" in norm
    assert "; if ($?) { pytest }" in norm


def test_normalize_windows_command_unquoted_spaced_target() -> None:
    cwd = r"C:\Users\kunal sanga"
    cmd = r"cd C:\Users\kunal sanga\Desktop\New folder\app1 && pytest"
    norm = normalize_windows_command(cmd, cwd)
    assert "Set-Location -LiteralPath " in norm
    assert "New folder" in norm
    assert "; if ($?) { pytest }" in norm


# ── 2. Tool Aggregation Partial Failure ───────────────────────────────────

def test_tool_aggregation_success_preserves_count() -> None:
    state = ConversationState()
    for i in range(1, 5):
        state.add_running(f"Read file_{i}.py", tool="read_file")
        state.complete_last(status="success")

    # Should have collapsed into Inspected 4 files
    inspected_items = [item for item in state.items if item[1].startswith("Inspected ")]
    assert len(inspected_items) == 1
    assert inspected_items[0][1] == "Inspected 4 files"
    assert inspected_items[0][2] == "success"


def test_tool_aggregation_failure_does_not_mark_all_inspections_failed() -> None:
    state = ConversationState()
    # 20 successful reads
    for i in range(1, 21):
        state.add_running(f"Read file_{i}.py", tool="read_file")
        state.complete_last(status="success")

    # 21st read fails
    state.add_running("Read frontend/src/components", tool="read_file")
    state.complete_last(
        status="error",
        error="Cannot read 'frontend/src/components': it is a directory, not a file.",
        tool="read_file",
    )

    # Truthful reporting:
    # 1. Successful 20 files should remain with Sym.CHECK and 'success'
    # 2. Failed call should be its own item with Sym.CROSS and 'error'
    success_items = [item for item in state.items if item[1] == "Inspected 20 files"]
    assert len(success_items) == 1
    assert success_items[0][0] == Sym.CHECK
    assert success_items[0][2] == "success"

    error_items = [item for item in state.items if item[2] == "error"]
    assert len(error_items) == 1
    assert "frontend/src/components" in error_items[0][1]
    assert error_items[0][0] == Sym.CROSS
    assert "directory" in error_items[0].error


# ── 3. Plan Reconciliation & Summary Step Integrity ───────────────────────

def test_summarize_category_not_matched_by_read_tools() -> None:
    item = TodoItem(id="3", description="Summarize findings")
    assert todo_category(item.description) == "summarize"

    # read_file tool should score 0 for summarize step
    score = score_todo(item, "read_file", {"path": "main.py"})
    assert score == 0


def test_reconcile_on_evidence_fails_incomplete_steps_on_task_failure() -> None:
    plan = TaskPlan(items=[
        TodoItem(id="1", description="Discover workspace structure", status=TodoStatus.COMPLETED),
        TodoItem(id="2", description="Read key project files", status=TodoStatus.COMPLETED),
        TodoItem(id="3", description="Summarize findings", status=TodoStatus.PENDING),
    ])

    changed = reconcile_on_evidence(
        plan,
        did_inspect=True,
        did_implement=False,
        tests_passed=None,
        verified=None,
        did_commit=False,
        did_push=False,
        task_failed=True,
    )

    # Step 3 must be failed, NOT left pending or completed
    step3 = plan.get("3")
    assert step3 is not None
    assert step3.status == TodoStatus.FAILED
    assert "Task failed" in (step3.error or "")


def test_reconcile_on_evidence_reverts_prematurely_completed_summarize_on_failure() -> None:
    plan = TaskPlan(items=[
        TodoItem(id="1", description="Discover workspace structure", status=TodoStatus.COMPLETED),
        TodoItem(id="2", description="Read key project files", status=TodoStatus.COMPLETED),
        TodoItem(id="3", description="Summarize findings", status=TodoStatus.COMPLETED),
    ])

    changed = reconcile_on_evidence(
        plan,
        did_inspect=True,
        did_implement=False,
        tests_passed=None,
        verified=None,
        did_commit=False,
        did_push=False,
        task_failed=True,
    )

    # Step 3 must be reverted to FAILED because the task terminated before final response
    step3 = plan.get("3")
    assert step3 is not None
    assert step3.status == TodoStatus.FAILED


def test_reconcile_on_evidence_completes_summarize_on_successful_task() -> None:
    plan = TaskPlan(items=[
        TodoItem(id="1", description="Discover workspace structure", status=TodoStatus.COMPLETED),
        TodoItem(id="2", description="Read key project files", status=TodoStatus.COMPLETED),
        TodoItem(id="3", description="Summarize findings", status=TodoStatus.PENDING),
    ])

    changed = reconcile_on_evidence(
        plan,
        did_inspect=True,
        did_implement=False,
        tests_passed=None,
        verified=None,
        did_commit=False,
        did_push=False,
        task_failed=False,
    )

    step3 = plan.get("3")
    assert step3 is not None
    assert step3.status == TodoStatus.COMPLETED


# ── 4. VerificationStatus.SKIPPED ─────────────────────────────────────────

def test_verification_status_skipped_member_exists() -> None:
    assert hasattr(VerificationStatus, "SKIPPED")
    assert VerificationStatus.SKIPPED.value == "skipped"


def test_conversation_summary_renders_skipped_as_not_applicable() -> None:
    from harness_core.cli.conversation import ConversationRenderer
    mock_console = MagicMock()
    mock_console.size.width = 100
    renderer = ConversationRenderer(console=mock_console, plain=False)

    renderer.render_completion(
        elapsed=10.0,
        success=False,
        status="failed",
        error_detail="Task paused. The selected provider is temporarily unavailable.",
        verification_status="skipped",
        verification_summary="read-only explain task — no verification needed",
    )
    # Check that console.print was called with dim style and 'Not applicable' / 'Skipped'
    calls = mock_console.print.call_args_list
    verif_calls = [c for c in calls if len(c.args) > 0 and "Verification" in str(c.args[0])]
    assert len(verif_calls) >= 1
    call_str = str(verif_calls[0].args[0])
    call_kwargs = verif_calls[0].kwargs
    assert "Not applicable" in call_str or "Skipped" in call_str
    assert call_kwargs.get("style") == "dim"
    assert "red" != call_kwargs.get("style")


# ── 5. ReadFileTool Directory Diagnostic ──────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_tool_directory_diagnostic(tmp_path: Path) -> None:
    tool = ReadFileTool()
    subdir = tmp_path / "somedir"
    subdir.mkdir()

    result = await tool.execute({"path": str(subdir)})
    assert result.error is not None
    assert "it is a directory, not a file" in result.error
    assert "list_files" in result.error
