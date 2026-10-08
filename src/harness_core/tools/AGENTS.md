# tools/ — Tool Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** All agent tools — filesystem operations, shell execution, git operations, code search, and path safety. All tools are gated by `PermissionManager`.

---

## READ FIRST

- `base.py` — `Tool` ABC (understand the interface before reading any implementation)
- `paths.py` — `cwd_in_workspace()`, `resolve_in_workspace()` (workspace boundary enforcement)

## READ IF NEEDED (by tool type)

- `filesystem.py` — file read, write, edit, list operations
- `shell.py` — sandboxed shell command execution
- `git.py` — git operations (diff, commit, status, log)
- `search.py` — code/text search
- `diagnosis.py` — command failure classification, test output parsing, shell normalization
- `parallel.py` — parallel tool execution helpers

## DO NOT READ FOR NORMAL TOOL TASKS

- `agent/loop.py` — loop orchestrates tool calls; tool implementations are separate
- `permissions/manager.py` — permissions gate tools; don't conflate with implementation

---

## Tool Execution Flow

```
AgentLoop
   → PermissionManager.check(tool_name, action)
   → Tool.execute(args)
   → ContextReuseManager.invalidate(path)  ← after any write
   → ToolResult
```

---

## Workspace Boundary

**Critical:** All file paths must be validated through `paths.py` before any operation:

```python
from harness_core.tools.paths import resolve_in_workspace
safe_path = resolve_in_workspace(user_path, workspace_root)
```

`resolve_in_workspace()` raises if the path escapes the workspace. Never skip this for user-provided paths.

---

## Dependencies

**Depends on:**
- `permissions/manager.py` — all tool calls must be gated
- `tools/paths.py` — all file operations must validate paths

**Used by:**
- `agent/loop.py` — `AgentLoop` executes tools
- `agents/worker.py` — tools are passed to `WorkerAgent`

---

## Architectural Invariants

- Never call `Tool.execute()` without first calling `PermissionManager`.
- Always validate file paths with `resolve_in_workspace()` before filesystem operations.
- After any write, call `ContextReuseManager.invalidate(path)` to bust the file snapshot.
- `shell.py` must not execute arbitrary commands without permission check.
- Tool results must not expose credentials or workspace secrets.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Bypassing `PermissionManager` | Always check permissions before execution |
| Accepting user-provided paths without validation | Use `resolve_in_workspace()` |
| Not invalidating context after a write | Call `ContextReuseManager.invalidate()` |
| Adding new tools without `Tool` subclass | Subclass `tools/base.py:Tool` |
| Running shell commands that may expose secrets | Check permission + sanitize output |

---

## Tests

- `tests/unit/test_tools.py`
- `tests/unit/test_git_accounting.py`
- `tests/unit/test_git_identity_and_completion.py`
- `tests/unit/test_tolerant_editing.py`
- `tests/unit/test_parallel.py`
- `tests/unit/test_permissions.py`
- `tests/unit/test_permission_fix.py`
- `tests/unit/test_tool_failure_propagation.py`

## Next: inspect

Permission issues → `permissions/manager.py`.
Context reuse after write → `context/reuse.py`.
