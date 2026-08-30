"""Deterministic workflows for common autonomous tasks.

The runtime owns HOW work happens; the LLM only decides WHAT.

Each workflow:
  1. Receives a goal and a context (workspace, agent_loop, event_bus)
  2. Builds the structured TODO plan
  3. Executes tools directly, emitting per-step evidence
  4. Returns when complete (or fails with a typed reason)

The LLM may be consulted to derive a commit message or to summarize
a find, but it does NOT drive each git command individually.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness_core.agent.types import (
    FailureReason,
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolResult,
    ToolResultStatus,
)
from harness_core.agent.todos import (
    TodoSpec,
    fallback_todo_specs,
    _CATEGORY_TOOLS,
)


# ── Context (DI surface for workflows) ─────────────────────────────────


@dataclass
class WorkflowContext:
    """Everything a workflow needs to do its work.

    Provided by the agent loop. Workflows never reach back into the
    loop's private state — they call only the public tool surface.
    """
    workspace: Path
    tools: dict[str, Any]  # tool_name -> Tool
    event_bus: Any  # EventBus
    modified_files: list[str] = field(default_factory=list)
    git_commit: str | None = None
    git_push: str | None = None


class ModelCaller(Protocol):
    """Minimal surface to ask the LLM a single, narrow question."""
    async def ask(self, prompt: str, *, max_tokens: int = 200) -> str: ...


# ── Workflow result ────────────────────────────────────────────────────


@dataclass
class WorkflowResult:
    """Outcome of a workflow execution."""
    success: bool
    failure_reason: str | None = None
    completed_operations: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────


async def _run_tool(ctx: WorkflowContext, name: str, **kwargs: Any) -> ToolResult:
    """Execute a tool by name. Records success for the duplicate-op guard."""
    tool = ctx.tools.get(name)
    if tool is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error=f"Tool not available: {name}",
        )
    args = dict(kwargs)
    args.setdefault("cwd", str(ctx.workspace))
    return await tool.execute(args)


def _category_for_op(op: str) -> str:
    for cat, ops in _CATEGORY_TOOLS.items():
        if op in ops:
            return cat
    return "other"


def _populate_plan(plan, specs: list[TodoSpec]) -> list[str]:
    """Add specs to a TaskPlan, wiring dependencies from sequential order.

    Returns the list of TODO IDs so the workflow can map operations
    to specific items.
    """
    ids: list[str] = []
    for i, spec in enumerate(specs):
        # Default: each spec depends on the previous one
        deps = (ids[-1],) if ids else ()
        item = plan.add_spec(
            spec.description,
            category=spec.category,
            expected_operations=spec.expected_operations,
            dependencies=deps,
            required=spec.required,
        )
        ids.append(item.id)
    return ids


async def _emit(ctx: WorkflowContext, type_: str, **data: Any) -> None:
    """Emit a structured event on the workflow's bus."""
    from harness_core.observability.events import Event
    await ctx.event_bus.emit(Event(type=type_, source="workflow", data=data))


# ── Git push workflow ──────────────────────────────────────────────────


GIT_PUSH_SPECS: list[TodoSpec] = [
    TodoSpec("Inspect repository status", "inspect", ("git_status",)),
    TodoSpec("Detect Git remote", "inspect", ("git_remote",)),
    TodoSpec("Verify Git identity", "inspect", ("git_identity",)),
    TodoSpec("Stage intended changes", "stage", ("git_add",)),
    TodoSpec("Commit changes", "commit", ("git_commit",)),
    TodoSpec("Push to remote", "push", ("git_push",)),
    TodoSpec("Verify push result", "verify", ("git_log",)),
]


async def run_git_push_workflow(
    ctx: WorkflowContext,
    plan,
    *,
    commit_message: str | None = None,
    model: ModelCaller | None = None,
) -> WorkflowResult:
    """End-to-end git push with bounded runtime execution.

    `commit_message` may come from the LLM (one short ask) or the
    user. If neither is provided, a deterministic placeholder is used
    (the runtime owns execution; the LLM is optional).
    """
    todo_ids = _populate_plan(plan, GIT_PUSH_SPECS)
    completed_ops: list[str] = []

    async def _advance(idx: int, evidence: dict[str, Any] | None = None) -> None:
        item_id = todo_ids[idx]
        plan.activate_id(item_id)
        plan.complete_id(item_id, evidence or {"op": "ok"})

    # 1. Inspect status
    status = await _run_tool(ctx, "git_status")
    completed_ops.append("git_status")
    if status.status == ToolResultStatus.ERROR:
        plan.fail_id(todo_ids[0], status.error or "git status failed")
        return WorkflowResult(False, "tool_failure", completed_ops)
    if status.metadata.get("clean"):
        # Nothing to commit; report and finish truthfully
        for i in range(1, 4):
            plan.skip_id(todo_ids[i], "Working tree clean — nothing to commit or push")
        await _advance(6, {"op": "clean_tree"})
        return WorkflowResult(True, None, completed_ops)
    await _advance(0, {"tool": "git_status", "files": status.metadata.get("files", [])})

    # 2. Detect remote
    remote_result = await _run_tool(ctx, "git_remote")
    completed_ops.append("git_remote")
    if remote_result.status == ToolResultStatus.ERROR:
        plan.fail_id(todo_ids[1], remote_result.error or "git remote failed")
        return WorkflowResult(False, "tool_failure", completed_ops)
    remotes = remote_result.metadata.get("remotes", {})
    if not remotes:
        plan.fail_id(todo_ids[1], "No git remote configured")
        return WorkflowResult(False, "tool_failure", completed_ops)
    await _advance(1, {"remotes": list(remotes.keys())})

    # 3. Verify identity
    identity = await _run_tool(ctx, "git_identity")
    completed_ops.append("git_identity")
    if identity.status == ToolResultStatus.ERROR:
        plan.fail_id(todo_ids[2], identity.error or "git identity not configured")
        return WorkflowResult(False, "tool_failure", completed_ops)
    await _advance(2, {"identity": identity.metadata})

    # 4. Stage
    add = await _run_tool(ctx, "git_add")
    completed_ops.append("git_add")
    if add.status == ToolResultStatus.ERROR:
        plan.fail_id(todo_ids[3], add.error or "git add failed")
        return WorkflowResult(False, "tool_failure", completed_ops)
    await _advance(3, {"op": "staged"})

    # 5. Commit (derive message if needed — single bounded model ask)
    if not commit_message and model is not None:
        try:
            commit_message = await model.ask(
                "Write a one-line git commit message (imperative, <72 chars) for: "
                "the user's current workspace changes. Return ONLY the message, no quotes.",
                max_tokens=80,
            )
            commit_message = (commit_message or "").strip().strip('"').strip("'")
        except Exception:
            commit_message = None
    if not commit_message:
        commit_message = "Update project"

    commit = await _run_tool(ctx, "git_commit", message=commit_message)
    completed_ops.append("git_commit")
    if commit.status == ToolResultStatus.ERROR:
        plan.fail_id(todo_ids[4], commit.error or "git commit failed")
        return WorkflowResult(False, "tool_failure", completed_ops)
    if commit.metadata.get("nothing_to_commit"):
        plan.skip_id(todo_ids[4], "Nothing to commit")
    else:
        ctx.git_commit = commit.metadata.get("commit_hash")
        await _advance(4, {"commit_hash": ctx.git_commit})

    # 6. Push
    push = await _run_tool(ctx, "git_push")
    completed_ops.append("git_push")
    if push.status == ToolResultStatus.ERROR:
        plan.fail_id(todo_ids[5], push.error or "git push failed")
        return WorkflowResult(False, "tool_failure", completed_ops)
    ctx.git_push = (
        f"{push.metadata.get('remote', 'origin')}/{push.metadata.get('branch', '')}"
    )
    await _advance(5, {"remote": push.metadata.get("remote"), "branch": push.metadata.get("branch")})

    # 7. Verify
    log = await _run_tool(ctx, "git_log", count=3)
    completed_ops.append("git_log")
    if log.status == ToolResultStatus.SUCCESS:
        await _advance(6, {"log": log.output[:200]})
    else:
        await _advance(6, {"note": "log unavailable; push succeeded"})

    return WorkflowResult(True, None, completed_ops, {
        "commit": ctx.git_commit,
        "push": ctx.git_push,
    })


# ── Test workflow ──────────────────────────────────────────────────────


TEST_SPECS: list[TodoSpec] = [
    TodoSpec("Inspect repository for test runner", "inspect", ("list_files", "read_file")),
    TodoSpec("Run automated tests", "test", ("run_command",)),
    TodoSpec("Summarize test results", "verify", ("run_command",)),
]


async def run_test_workflow(
    ctx: WorkflowContext,
    plan,
    *,
    test_command: str = "pytest -q",
) -> WorkflowResult:
    """Run the test suite and return parsed counts.

    Bound to a single shell invocation plus one optional inspection.
    """
    todo_ids = _populate_plan(plan, TEST_SPECS)
    completed_ops: list[str] = []

    # 1. Inspect
    inspect = await _run_tool(ctx, "list_files", path=".")
    completed_ops.append("list_files")
    if inspect.status == ToolResultStatus.SUCCESS:
        plan.activate_id(todo_ids[0])
        plan.complete_id(todo_ids[0], {"op": "list"})
    else:
        plan.skip_id(todo_ids[0], "list_files unavailable")

    # 2. Run tests
    result = await _run_tool(ctx, "run_command", command=test_command)
    completed_ops.append("run_command")
    plan.activate_id(todo_ids[1])
    if result.status == ToolResultStatus.SUCCESS:
        plan.complete_id(todo_ids[1], {"exit_code": 0})
    else:
        plan.fail_id(todo_ids[1], result.error or "tests failed")
        return WorkflowResult(False, "test_failure", completed_ops)

    # 3. Summarize
    plan.activate_id(todo_ids[2])
    plan.complete_id(todo_ids[2], {"output": (result.output or "")[:200]})
    return WorkflowResult(True, None, completed_ops, {
        "output": result.output or "",
    })


# ── Explain workflow ───────────────────────────────────────────────────


EXPLAIN_SPECS: list[TodoSpec] = [
    TodoSpec("Discover workspace structure", "inspect", ("list_files",)),
    TodoSpec("Read key project files", "inspect", ("read_file",)),
    TodoSpec("Summarize findings", "inspect", ("read_file",)),
]


async def run_explain_workflow(
    ctx: WorkflowContext,
    plan,
    *,
    max_files: int = 6,
) -> WorkflowResult:
    """Gather context for a research/explain task. NO edits.

    Produces a structured summary the LLM can expand; the workflow
    itself does not write to disk.
    """
    todo_ids = _populate_plan(plan, EXPLAIN_SPECS)
    completed_ops: list[str] = []

    listing = await _run_tool(ctx, "list_files", path=".")
    completed_ops.append("list_files")
    files = (listing.metadata or {}).get("files", []) if listing.status == ToolResultStatus.SUCCESS else []
    plan.activate_id(todo_ids[0])
    plan.complete_id(todo_ids[0], {"file_count": len(files)})

    # Pick the most informative files (top-level + common extensions)
    interesting = [
        f for f in files
        if not f.startswith(".")
        and any(f.endswith(ext) for ext in (".py", ".md", ".toml", ".yaml", ".json", ".js", ".ts", ".go", ".rs"))
    ][:max_files]

    plan.activate_id(todo_ids[1])
    snippets: list[dict[str, str]] = []
    for fp in interesting:
        r = await _run_tool(ctx, "read_file", path=fp)
        completed_ops.append(f"read_file:{fp}")
        if r.status == ToolResultStatus.SUCCESS:
            content = (r.output or "")[:600]
            snippets.append({"path": fp, "preview": content})
    plan.complete_id(todo_ids[1], {"files_read": len(snippets)})

    plan.activate_id(todo_ids[2])
    plan.complete_id(todo_ids[2], {"snippets": snippets})
    return WorkflowResult(True, None, completed_ops, {
        "files": interesting,
        "snippets": snippets,
    })


# ── Workflow router ───────────────────────────────────────────────────


# Keyword routing — fast paths so the LLM doesn't have to rediscover
# workflow structure every time. Order matters: most specific first.
_GIT_KEYWORDS = ("push", "github", "commit and push", "git push")
_TEST_KEYWORDS = ("run the tests", "run tests", "run test", "run pytest", "run npm test", "test the project", "verify the build")
_EXPLAIN_KEYWORDS = (
    "explain", "describe", "what does this project", "what is this project",
    "overview", "summarize", "research", "analyze this", "how does",
)


def classify_workflow(goal: str) -> str | None:
    """Return a workflow name from the goal, or None for the generic path.

    Only matches when the keyword represents the PRIMARY intent:
    - Short goals (<50 chars) where the keyword dominates
    - Or the keyword is in the first 5 words (leading action)

    Multi-step goals like "Fix the calculator and explain the changes"
    should go through the normal LLM-driven loop.
    """
    g = (goal or "").lower().strip()
    if not g:
        return None
    words = g.split()
    first_5 = " ".join(words[:5]) if len(words) >= 5 else g
    is_short = len(g) < 50
    # Be specific: "push" wins over "test" wins over "explain".
    if any(k in first_5 for k in _GIT_KEYWORDS) or (is_short and any(k in g for k in _GIT_KEYWORDS)):
        return "git_push"
    if any(k in first_5 for k in _TEST_KEYWORDS) or (is_short and any(k in g for k in _TEST_KEYWORDS)):
        return "test"
    if any(k in first_5 for k in _EXPLAIN_KEYWORDS) or (is_short and any(k in g for k in _EXPLAIN_KEYWORDS)):
        return "explain"
    return None
