# agents/ — Multi-Agent Orchestration Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** Multi-agent orchestration domain. Contains `TaskGraph`, `SubTask`, `Scheduler`, `WorkerAgent`, `AgentRegistry`, `AgentMessageBus`, and workspace locking. This is the task-execution backbone — distinct from `agent/` (AgentLoop internals).

---

## READ FIRST

- `domain.py` — `TaskGraph`, `SubTask`, `AgentRole`, `AgentStatus`, `AgentContract`, `AgentResult`, `AgentMessage`
- `scheduler.py` — `Scheduler` (concurrent task dispatch, recovery integration)
- `worker.py` — `WorkerAgent` (per-task agent executor)

## READ IF NEEDED

- `registry.py` — `AgentRegistry` (role → capability mapping, agent instantiation)
- `message_bus.py` — `AgentMessageBus` (inter-agent typed messages)
- `locks.py` — `WorkspaceLockManager`, `WorkspaceResource` (resource contention)
- `orchestrator.py` — high-level orchestration patterns
- `executor.py` — task executor helpers
- `parallel.py` — parallel execution utilities
- `cancellation.py` — cancellation signal handling

## DO NOT READ FOR NORMAL TASKS

- `agent/loop.py` — loop internals are in `agent/`, not here
- `runtime/runtime.py` — runtime is above `Scheduler` in the hierarchy

---

## Key Relationships

```
EngineeringRuntime
   └─ Scheduler
        └─ WorkerAgent (one per SubTask)
             └─ AgentLoop (agent/loop.py)

TaskGraph
   └─ SubTask (nodes)
        └─ AgentContract (role + task spec)
             └─ WorkerAgent picks it up
```

- `TaskGraph.ready_tasks()` returns tasks with all dependencies satisfied.
- `Scheduler` polls `ready_tasks()` and spawns `WorkerAgent` up to `max_concurrency=3`.
- `WorkerAgent.run()` calls `AgentLoop` and returns `AgentResult`.
- `Scheduler` writes the result back into `TaskGraph`.

---

## Dependencies

**Depends on:**
- `agent/loop.py` — `AgentLoop` (WorkerAgent runs it)
- `recovery/` — `FailureClassifier`, `RecoveryOrchestrator` (Scheduler uses them)
- `observability/events.py` — `EventBus`

**Used by:**
- `runtime/runtime.py` — `EngineeringRuntime` creates `TaskGraph` and `Scheduler`

---

## Architectural Invariants

- All task execution goes through `Scheduler → WorkerAgent → AgentLoop`. Do not run `AgentLoop` directly from `EngineeringRuntime`.
- `TaskGraph` is the single source of truth for task state — never mutate `SubTask.status` outside `TaskGraph` methods.
- `WorkspaceLockManager` must gate any concurrent write access to shared resources.
- Recovery tasks are identified by `RECOVERY_MARKER` in their task ID — do not conflate with primary tasks.
- `AgentMessageBus` is for inter-agent communication; `EventBus` is for observability. Do not swap them.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Running `AgentLoop` directly from runtime | Always via `Scheduler → WorkerAgent` |
| Mutating `SubTask.status` directly | Use `TaskGraph` state-transition methods |
| Confusing `AgentMessageBus` with `EventBus` | MessageBus = inter-agent; EventBus = observability |
| Adding a new agent role not in `AgentRole` enum | Add it to `domain.py:AgentRole` first |
| Ignoring `WorkspaceLockManager` for concurrent writes | Always acquire lock for shared resources |

---

## Tests

- `tests/unit/test_scheduler.py`
- `tests/unit/test_scheduler_locks.py`
- `tests/unit/test_worker_agent.py`
- `tests/unit/test_advanced_agents.py`
- `tests/unit/test_message_bus.py`
- `tests/unit/test_locks.py`
- `tests/unit/test_handoff_integration.py`
- `tests/unit/test_parallel.py`

## Next: inspect

Scheduling issues → `scheduler.py`.
Task graph issues → `domain.py`.
Agent role/capability → `registry.py`.
Tool execution → `agent/AGENTS.md` + `tools/AGENTS.md`.
