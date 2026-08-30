"""Runtime-owned TODO sanitization and evidence-based status updates.

The TODO system is owned by the runtime, not the model. Each TODO has:
  - description       — what the agent said it would do
  - status            — PENDING / IN_PROGRESS / COMPLETED / FAILED / SKIPPED
  - required          — if True, completion invariant requires resolution
  - category          — inspect | implement | test | verify | commit | push | stage | other
  - expected_operations — tool names that would evidence this TODO
  - dependencies      — IDs of TODOs that must complete before this one
  - evidence          — the structured result that resolved this TODO

Tools emit structured operation identifiers. The runtime maps tool -> TODO.
The model does NOT control TODO completion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from harness_core.agent.types import TaskPlan, TodoItem, TodoStatus, ToolCall, ToolResult


ACTION_VERBS = (
    "inspect", "read", "analyze", "identify", "implement", "update", "improve",
    "add", "fix", "run", "verify", "write", "edit", "create", "test", "commit",
    "push", "stage", "search", "review", "check", "diagnose", "refactor",
    "discover", "list", "install", "configure", "enhance", "modify", "replace",
    "validate", "build", "apply", "remove", "delete", "rename", "document",
    "summarize", "explain",
    "inspecting", "reading", "analyzing", "implementing", "updating",
    "improving", "running", "verifying", "writing", "editing", "creating",
    "testing", "committing", "pushing", "staging", "searching", "summarizing",
)

_PROSE_MARKERS = (
    "i don't have", "i do not have", "i don't see", "i cannot see",
    "i can't see", "no visibility", "not able to see", "unable to see",
    "please provide", "please share", "please tell", "please specify",
    "can you provide", "can you share", "can you tell", "could you provide",
    "what technology", "what tech stack", "what framework", "which framework",
    "what language", "which language", "what type of project",
    "more context", "more information", "need more info",
    "gives me hope", "looks good", "in my scope", "got the ",
    "ready for deployment", "ready for deploy", "the project's",
    "i think", "i believe", "seems like", "feels like",
)

_INSPECT = ("inspect", "read", "analyze", "discover", "list", "understand", "identify", "review", "search", "explain", "summarize")
_IMPLEMENT = ("implement", "edit", "update", "improve", "add", "write", "create", "refactor", "enhance", "modify", "apply", "replace")
_TEST = ("run test", "run tests", "pytest", "fix failing")
_TEST_WORDS = ("test",)
_VERIFY = ("verify", "validate")
_COMMIT = ("commit",)
_PUSH = ("push",)
_STAGE = ("stage",)

# Category → tool names that produce evidence for that category
_CATEGORY_TOOLS: dict[str, tuple[str, ...]] = {
    "inspect": ("read_file", "list_files", "glob", "grep", "git_status", "git_diff", "git_log"),
    "implement": ("write_file", "edit_file"),
    "test": ("run_command",),
    "verify": ("run_command",),
    "stage": ("git_add", "git_stage"),
    "commit": ("git_commit",),
    "push": ("git_push",),
}


def _has(low: str, words: tuple[str, ...]) -> bool:
    """Whole-word match allowing only plural/verb suffixes (s, es, ing, ed)."""
    for w in words:
        if re.search(rf"\b{re.escape(w)}(?:s|es|ing|ed)?\b", low):
            return True
    return False


def sanitize_todo_titles(steps: list[str]) -> list[str]:
    """Keep only actionable engineering tasks. Reject conversational prose."""
    valid: list[str] = []
    for step in steps:
        cleaned = _clean_step(step)
        if cleaned and is_actionable_todo(cleaned):
            valid.append(cleaned)
        if len(valid) >= 8:
            break
    return valid


def is_actionable_todo(title: str) -> bool:
    s = title.strip()
    if not s or len(s) > 140 or "?" in s:
        return False
    low = s.lower()
    if any(m in low for m in _PROSE_MARKERS):
        return False
    first = re.split(r"[\s/:-]+", low, maxsplit=1)[0]
    return first in ACTION_VERBS


@dataclass
class TodoSpec:
    """Structured metadata for a TODO step added from the plan."""
    description: str
    category: str
    expected_operations: tuple[str, ...]
    required: bool = True
    dependencies: tuple[str, ...] = ()


def _todo_spec(title: str) -> TodoSpec:
    """Build a structured TodoSpec from a raw title."""
    cat = todo_category(title)
    ops = _CATEGORY_TOOLS.get(cat, ())
    return TodoSpec(description=title, category=cat, expected_operations=ops)


def fallback_todo_plan(goal: str, project_info: dict[str, Any] | None = None) -> list[str]:
    """Deterministic plan from the task + workspace. Never conversational."""
    info = project_info or {}
    g = (goal or "").lower()
    steps: list[str] = ["Inspect repository structure"]

    if any(w in g for w in ("push", "github", "commit")):
        if "commit" in g or "push" in g:
            steps.append("Stage intended changes")
            steps.append("Commit changes")
        if "push" in g or "github" in g:
            steps.append("Push to remote")
        steps.append("Verify Git result")
        return steps[:8]

    # Research / explanation goals: inspect, analyze, summarize — no edits.
    if any(w in g for w in ("explain", "describe", "understand", "what is",
                            "what does", "overview", "summarize", "how does",
                            "research", "analyze")):
        steps.append("Read key project files")
        steps.append("Analyze current implementation")
        steps.append("Summarize findings")
        return steps[:8]

    steps.append("Analyze current implementation")
    if any(w in g for w in ("enhance", "improve", "implement", "add", "fix", "update", "ui", "refactor")):
        steps.append("Implement the requested changes")
    if info.get("has_tests") or any(w in g for w in ("test", "verify")):
        steps.append("Run automated tests")
        steps.append("Fix failing tests")
    steps.append("Verify changed behavior")
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in steps:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:8]


def fallback_todo_specs(goal: str, project_info: dict[str, Any] | None = None) -> list[TodoSpec]:
    """Structured version of fallback_todo_plan — use this when building plans."""
    titles = fallback_todo_plan(goal, project_info)
    return [_todo_spec(t) for t in titles]


def _clean_step(step: str) -> str:
    s = step.strip().strip("-•*").strip().strip("\"'")
    s = re.sub(r"^[\d]+[.)\s]+[-•*]?\s*", "", s).strip()
    return s


def todo_category(title: str) -> str:
    low = title.lower()
    if _has(low, _PUSH):
        return "push"
    if _has(low, _COMMIT):
        return "commit"
    if _has(low, _STAGE):
        return "stage"
    if _has(low, _VERIFY):
        return "verify"
    if any(k in low for k in _TEST) or _has(low, _TEST_WORDS):
        return "test"
    if "fix" in low and _has(low, _TEST_WORDS):
        return "test"
    if _has(low, _IMPLEMENT):
        return "implement"
    if _has(low, _INSPECT):
        return "inspect"
    return "other"


def _path_basename(args: dict[str, Any]) -> str:
    path = str(args.get("path") or args.get("file_path") or "")
    return path.replace("\\", "/").split("/")[-1].lower()


def score_todo(item: TodoItem, tool_name: str, arguments: dict[str, Any]) -> int:
    """Higher score = better match. 0 = no match."""
    if item.status in (TodoStatus.COMPLETED, TodoStatus.SKIPPED):
        return 0
    cat = todo_category(item.description)
    title = item.description.lower()
    fname = _path_basename(arguments)
    score = 0
    if fname and fname in title:
        score += 10

    if tool_name in ("read_file", "list_files", "glob", "grep") and cat == "inspect":
        score += 5
    elif tool_name in ("write_file", "edit_file") and cat == "implement":
        score += 5
    elif tool_name == "run_command":
        cmd = str(arguments.get("command", "")).lower()
        if cat == "test" and ("test" in cmd or "pytest" in cmd):
            score += 8
        elif cat == "inspect" and not any(x in cmd for x in ("test", "git")):
            score += 1
    elif tool_name == "git_add" and cat == "stage":
        score += 8
    elif tool_name == "git_commit" and cat == "commit":
        score += 8
    elif tool_name == "git_push" and cat == "push":
        score += 8
    elif tool_name.startswith("git_") and cat in ("inspect", "commit", "push", "stage"):
        score += 2
    return score


def select_todo(plan: TaskPlan, tool_name: str, arguments: dict[str, Any]) -> TodoItem | None:
    scored = [(score_todo(item, tool_name, arguments), item) for item in plan.items]
    scored = [(s, i) for s, i in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], 0 if t[1].status == TodoStatus.IN_PROGRESS else 1))
    return scored[0][1]


def apply_tool_started(plan: TaskPlan, call: ToolCall) -> TodoItem | None:
    item = select_todo(plan, call.tool_name, call.arguments)
    if item and item.status == TodoStatus.PENDING:
        plan.activate_id(item.id)
        return item
    return item if item and item.status == TodoStatus.IN_PROGRESS else None


def apply_tool_result(plan: TaskPlan, call: ToolCall, result: ToolResult) -> TodoItem | None:
    """Update TODO from real tool outcome. Returns the affected item, if any."""
    item = select_todo(plan, call.tool_name, call.arguments)
    if item is None:
        return None
    evidence = {
        "tool": call.tool_name,
        "command": call.arguments.get("command"),
        "path": call.arguments.get("path") or call.arguments.get("file_path"),
        "exit_code": result.exit_code,
        "status": result.status.value,
    }
    failed = result.execution_failed and not result.is_perm_denied
    cat = todo_category(item.description)
    if failed:
        # Inspect TODOs are not failed by a single missing file; keep pending.
        if cat == "inspect":
            return item
        plan.fail_id(item.id, result.error or result.output[:300] if result.output else "failed")
        item.evidence = evidence
        return item
    if item.status != TodoStatus.COMPLETED:
        plan.complete_id(item.id, evidence)
    return item


def reconcile_on_evidence(
    plan: TaskPlan,
    *,
    did_inspect: bool,
    did_implement: bool,
    tests_passed: bool | None,
    verified: bool | None,
    did_commit: bool,
    did_push: bool,
) -> list[TodoItem]:
    """Complete remaining TODOs that have matching execution evidence."""
    changed: list[TodoItem] = []
    for item in plan.items:
        if item.status not in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS):
            continue
        cat = todo_category(item.description)
        ok = (
            (cat == "inspect" and did_inspect)
            or (cat == "implement" and did_implement)
            or (cat == "test" and tests_passed is True)
            or (cat == "verify" and verified is True)
            or (cat == "commit" and did_commit)
            or (cat == "push" and did_push)
            or (cat == "stage" and (did_commit or did_push))
        )
        if ok:
            plan.complete_id(item.id, {"reconciled": True, "category": cat})
            changed.append(item)
        elif cat == "test" and tests_passed is False:
            plan.fail_id(item.id, "Tests failed")
            changed.append(item)
        elif cat == "verify" and verified is False:
            plan.fail_id(item.id, "Verification failed")
            changed.append(item)
    return changed
