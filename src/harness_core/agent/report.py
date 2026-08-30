"""Evidence-based final task reports. Never invented from model prose."""

from __future__ import annotations

from harness_core.agent.types import Task, TaskStatus, TodoStatus, ToolResultStatus


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs:02d}s"


def build_execution_report(task: Task, elapsed: float, files_changed: list[str]) -> str:
    """Build a concise engineering report from runtime state only."""
    lines: list[str] = []
    status = task.status

    if status == TaskStatus.COMPLETED:
        lines.append(f"✓ Task completed in {_fmt_elapsed(elapsed)}")
    elif status == TaskStatus.PAUSED:
        lines.append("⚠ Model unavailable")
        lines.append("Execution state preserved.")
        if task.paused_reason:
            lines.append(task.paused_reason)
    elif status == TaskStatus.CANCELLED:
        lines.append("⚠ Task cancelled")
        lines.append("Execution stopped safely.")
    elif status == TaskStatus.FAILED:
        lines.append("✗ Task failed")
    else:
        lines.append(f"Task ended ({status.value})")

    if files_changed:
        lines.append("")
        lines.append("Changes made:")
        for f in files_changed:
            lines.append(f"  • {f}")

    test_line = _test_line(task)
    verify_line = _verify_line(task)
    if test_line or verify_line:
        lines.append("")
        lines.append("Validation:")
        if test_line:
            lines.append(f"  {test_line}")
        if verify_line:
            lines.append(f"  {verify_line}")

    if task.task_plan.items:
        lines.append("")
        lines.append("TODOs:")
        for item in task.task_plan.items:
            lines.append(f"  {item.display()}")
        lines.append(
            f"TODO progress: {task.task_plan.completed_count}/{task.task_plan.total_count}"
        )

    if status == TaskStatus.FAILED:
        failed_items = [i for i in task.task_plan.items if i.status == TodoStatus.FAILED]
        failed_tools = [
            tc for tc in task.tool_calls
            if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied
        ]
        if failed_items or failed_tools:
            lines.append("")
            lines.append("Failed:")
            for item in failed_items:
                lines.append(f"  ✗ {item.description}")
            if not failed_items:
                for tc in failed_tools[-3:]:
                    cmd = tc.arguments.get("command", tc.tool_name)
                    lines.append(f"  ✗ {cmd}")
        root = _root_cause(task)
        if root:
            lines.append("")
            lines.append("Root cause:")
            lines.append(f"  {root}")
        lines.append("")
        lines.append("Next action:")
        lines.append("  Review the failure and run another implementation pass.")

    if status == TaskStatus.PAUSED:
        remaining = [
            i for i in task.task_plan.items
            if i.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS)
        ]
        completed = [i for i in task.task_plan.items if i.status == TodoStatus.COMPLETED]
        if completed:
            lines.append("")
            lines.append("Completed:")
            for item in completed:
                lines.append(f"  ✓ {item.description}")
        if remaining:
            lines.append("")
            lines.append("Remaining:")
            for item in remaining:
                lines.append(f"  ☐ {item.description}")
        lines.append("")
        lines.append("Task paused because no usable model is currently available. State preserved.")

    if status == TaskStatus.CANCELLED:
        lines.append("")
        lines.append(
            f"Completed: ✓ {task.task_plan.completed_count}/{task.task_plan.total_count} TODOs"
        )
        if files_changed:
            lines.append("Modified:")
            for f in files_changed:
                lines.append(f"  {f}")

    lines.append("")
    lines.append("Execution")
    lines.append(f"  {task.iterations} iterations")
    lines.append(f"  {len(task.tool_calls)} tool calls")
    if task.model_fallbacks:
        lines.append(f"  {task.model_fallbacks} model fallback")
    if task.models_used:
        lines.append(f"  models: {', '.join(task.models_used)}")

    if files_changed:
        lines.append("")
        lines.append("Files")
        for f in files_changed:
            lines.append(f"  {f}")

    if task.git_commit or task.git_push:
        lines.append("")
        lines.append("Git")
        if task.git_commit:
            lines.append(f"  Commit: {task.git_commit}")
        if task.git_push:
            lines.append(f"  Push: {task.git_push}")
        elif status == TaskStatus.COMPLETED and files_changed and not task.git_commit:
            lines.append("  Working tree modified")
            lines.append("  Commit: not created")

    if status == TaskStatus.COMPLETED:
        lines.append("")
        lines.append("Result")
        lines.append(f"  { _result_sentence(task, files_changed) }")

    return "\n".join(lines)


def _test_line(task: Task) -> str:
    if task.tests_run <= 0:
        return ""
    if task.tests_passed is None:
        return f"✓ tests run ({task.tests_run})"
    total = task.tests_run
    passed = task.tests_passed
    ok = passed == total
    mark = "✓" if ok else "✗"
    return f"{mark} {passed}/{total} passed"


def _verify_line(task: Task) -> str:
    if task.verification_passed is True:
        return "✓ Verification passed"
    if task.verification_passed is False:
        return "✗ Verification failed"
    return ""


def _root_cause(task: Task) -> str:
    if task.error:
        first = task.error.strip().splitlines()[0]
        return first[:300]
    for tc in reversed(task.tool_calls):
        if tc.result and tc.result.execution_failed:
            err = tc.result.error or tc.result.stderr or tc.result.output
            if err:
                return err.strip().splitlines()[0][:300]
    return ""


def _result_sentence(task: Task, files_changed: list[str]) -> str:
    if files_changed:
        return f"{len(files_changed)} file(s) changed; task completed with evidence."
    if task.git_push:
        return "Changes pushed."
    if task.git_commit:
        return "Commit created."
    return "Task completed with evidence."


def files_from_task(task: Task) -> list[str]:
    seen: list[str] = []
    for tc in task.tool_calls:
        if tc.tool_name not in ("write_file", "edit_file") or not tc.result:
            continue
        if tc.result.status != ToolResultStatus.SUCCESS:
            continue
        fp = str(tc.arguments.get("path") or tc.arguments.get("file_path") or "")
        if not fp:
            continue
        name = fp.replace("\\", "/").split("/")[-1]
        if name not in seen:
            seen.append(name)
    return seen
