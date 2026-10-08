# agent/ — AgentLoop Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** The core per-task tool-use loop. `AgentLoop` drives the LLM ↔ tool cycle for a single `SubTask`. It manages `TodoItem` tracking, context assembly, steering signals, completion gating, and failure escalation.

---

## READ FIRST

- `loop.py` — `AgentLoop` (the entire loop logic; 2700+ lines — search for specific methods, do not read top to bottom)
- `types.py` — `Task`, `TodoItem`, `ToolCall`, `ToolResult`, `AgentConfig` and all core types

## READ IF NEEDED

- `todos.py` — TodoItem reconciliation helpers (`select_todo`, `reconcile_on_evidence`, `apply_tool_result`)
- `completion.py` — completion gate logic (`can_complete_task`, `completion_blockers`)
- `steering.py` — `SteeringBuffer` integration and `steering_event_data`
- `ask_user.py` — `AskUserManager` for mid-loop clarification
- `intent.py` — task intent classification
- `micro.py` — micro-agent registry (`default_micro_registry`)
- `recovery.py` — loop-level recovery helpers (distinct from `recovery/` orchestration)
- `report.py` — structured completion report generation
- `workflows.py` — higher-level workflow orchestration patterns

## DO NOT READ FOR NORMAL AGENT TASKS

- `cli/` — unrelated to loop internals
- `runtime/` — runtime calls `AgentLoop`; loop does not call runtime
- `planning/` — planning is upstream of `AgentLoop`

---

## Dependencies

**Depends on:**
- `context/` — `ContextEngine`, `ContextPipeline`, `ContextReuseManager`, `ContextCompactor`
- `routing/router.py` — `ModelRouter` for provider/model selection
- `providers/base.py` — `ModelProvider` for LLM calls
- `tools/` — all tool implementations
- `verification/engine.py` — `VerificationEngine`
- `permissions/manager.py` — `PermissionManager`
- `observability/events.py` — `EventBus`
- `agents/message_bus.py` — inter-agent messaging

**Used by:**
- `agents/worker.py` — `WorkerAgent` instantiates and runs `AgentLoop`

---

## Architectural Invariants

- `AgentLoop` does not call `EngineeringRuntime` or `Scheduler` directly — it only communicates upward via `EventBus` and return values.
- `VerificationEngine` must be invoked before `TodoItem` or task completion is claimed.
- Tool calls must go through `PermissionManager` — never invoke `Tool.execute()` directly without permission check.
- `ContextReuseManager.is_unchanged()` must be checked before re-injecting file contents.
- Do not add planning logic inside `AgentLoop` — planning is `planning/planner.py`'s job.
- `SteeringBuffer` signals must be processed before the next model call, not ignored.
- Do not expose tool call arguments or model responses on `EventBus` events.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Adding task decomposition inside loop | Use `planning/planner.py` instead |
| Claiming completion without `VerificationEngine` | Always invoke `VerificationEngine` |
| Reading `loop.py` top-to-bottom for a small fix | Search for the specific method/class needed |
| Bypassing `PermissionManager` for tool calls | Always check permissions first |
| Creating new file types inside `agent/` | New types belong in `types.py` |

---

## Tests

- `tests/unit/test_agent_loop_integration.py` — loop integration
- `tests/unit/test_canonical_runtime.py` — canonical runtime behavior
- `tests/unit/test_canonical_tool_result.py` — tool result handling
- `tests/unit/test_completion_invariant.py` — completion gating
- `tests/unit/test_steering_askuser_micro.py` — steering + ask-user
- `tests/unit/test_task_plan_ux.py` — todo plan UX
- `tests/unit/test_tool_failure_propagation.py` — failure propagation

## Next: inspect

If agent behavior is wrong → check `loop.py` method for the relevant phase.
If tool handling is wrong → check `tools/AGENTS.md`.
If routing is wrong → check `routing/AGENTS.md`.
