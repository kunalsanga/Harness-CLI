# runtime/ — EngineeringRuntime Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** `EngineeringRuntime` is the thin orchestration seam of the project execution lifecycle. It coordinates `TaskGraph`, `Scheduler`, `RecoveryOrchestrator`, `MemoryManager`, `VerificationEngine`, and `EventBus`. It owns: project lifecycle stage, structured requirements, requirement→task→evidence traceability, artifact references, and interruption-safe cleanup.

---

## READ FIRST

- `runtime.py` — `EngineeringRuntime` (start here for any execution flow issue)
- `state.py` — `ProjectState`, `RuntimeStage`, `RuntimeStatus`, `ArtifactRef`

## READ IF NEEDED

- `requirements.py` — `Requirements`, `Evidence`, `EvidenceStatus`, `TraceabilityIndex`
- `governance.py` — budget and governance enforcement
- `context.py` — `assemble_role_context` helper
- `steering.py` — `SteeringBuffer` (re-exported here for convenience)

## DO NOT READ FOR NORMAL TASKS

- `agent/loop.py` — loop is below this layer
- `agents/scheduler.py` — runtime calls Scheduler; do not need to read Scheduler to fix runtime
- `cli/` — CLI calls runtime; runtime does not call CLI

---

## Key Design Principle

> *Model proposes, runtime decides.*

`EngineeringRuntime` does **not** re-implement authorities it coordinates. It only adds what nothing else owns: the project lifecycle, requirements, traceability, and cleanup.

---

## Dependencies

**Depends on:**
- `agents/domain.py` — `TaskGraph`, `SubTask`, `AgentRole`
- `agents/scheduler.py` — `Scheduler`
- `agents/registry.py` — `AgentRegistry`
- `agents/locks.py` — `WorkspaceLockManager`
- `planning/planner.py` — `Planner`
- `recovery/orchestrator.py` — `RecoveryOrchestrator`
- `observability/events.py` — `EventBus`
- `memory/manager.py` — `MemoryManager` (optional, non-fatal)

**Used by:**
- `cli/interactive.py` — `InteractiveShell` calls `EngineeringRuntime.execute()`

---

## Architectural Invariants

- **Do not create a second `EngineeringRuntime`.**
- `RuntimeStage` enum defines the canonical lifecycle — do not skip stages.
- `TraceabilityIndex` must link requirements → tasks → evidence — do not remove this mapping.
- `RECOVERY_MARKER` and `RETEST_MARKER` in task IDs identify recovery tasks — preserve this convention.
- `MemoryManager` is optional — failures must be non-fatal and logged, not raised.
- Do not hold mutable `ProjectState` references across async boundaries without locks.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Calling `AgentLoop` directly from runtime | Always via `Scheduler → WorkerAgent` |
| Skipping `RuntimeStage` transitions | Follow the `ProjectState.stage` lifecycle |
| Treating `MemoryManager` failures as fatal | Wrap in try/except; log and continue |
| Adding governance logic outside `governance.py` | Extend `governance.py` |

---

## Tests

- `tests/unit/test_runtime_state.py`
- `tests/unit/test_runtime_governance.py`
- `tests/unit/test_canonical_runtime.py`
- `tests/unit/test_requirements.py`
- `tests/unit/test_traceability.py`

## Next: inspect

Execution flow → `runtime.py`.
State shape → `state.py`.
Budget/governance → `governance.py`.
Recovery → `recovery/AGENTS.md`.
Scheduling → `agents/AGENTS.md`.
