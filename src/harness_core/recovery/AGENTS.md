# recovery/ — Recovery Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** Autonomous failure classification and recovery orchestration. `FailureClassifier` categorizes failures into 12 typed categories. `RecoveryOrchestrator` decides whether to retry, re-plan, or escalate. Recovery tasks are re-injected into `TaskGraph` via `Scheduler`.

---

## READ FIRST

- `classifier.py` — `FailureClassifier`, `FailureCategory` (12 categories), `FailureClassification`
- `orchestrator.py` — `RecoveryOrchestrator` (retry/re-plan logic)

## READ IF NEEDED

- `planner.py` — recovery-specific re-planning
- `validator.py` — recovery plan validation

---

## Failure Categories

```
TEST_FAILURE | BUILD_FAILURE | TYPE_ERROR | LINT_FAILURE
RUNTIME_ERROR | TOOL_FAILURE | MODEL_FAILURE | TIMEOUT
PERMISSION_FAILURE | DEPENDENCY_FAILURE | ENVIRONMENT_FAILURE | UNKNOWN
```

---

## Recovery Flow

```
AgentLoop failure
   → Scheduler detects failed SubTask
   → RecoveryOrchestrator.handle(failed_task, classification)
        → FailureClassifier.classify(error, output)
        → recovery planner (if re-plan needed)
        → re-queue SubTask with RECOVERY_MARKER in ID
             OR escalate to EngineeringRuntime
```

**Task ID convention:** Recovery tasks have `_recovery_` or `_retest_` in their ID.

---

## Dependencies

**Depends on:**
- `agents/domain.py` — `AgentResult`, `SubTask`, `AgentStatus`
- `agents/scheduler.py` — Scheduler re-queues recovered tasks

**Used by:**
- `agents/scheduler.py` — `Scheduler` imports `FailureClassifier` and `RecoveryOrchestrator`
- `runtime/runtime.py` — references `RECOVERY_MARKER`

---

## Architectural Invariants

- Recovery only mutates task state through `TaskGraph` and `Scheduler` interfaces.
- Do not bypass `EngineeringRuntime` state during recovery.
- Recovery tasks must carry `RECOVERY_MARKER` — never inject bare task IDs.
- `FailureCategory.UNKNOWN` is valid — do not force misclassification.
- Recovery must be finite — no infinite retry loops.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Re-queuing tasks without `RECOVERY_MARKER` | Always add marker to recovered task IDs |
| Infinite retry loops | Enforce maximum recovery attempts |
| Bypassing `TaskGraph` for task state mutation | Use TaskGraph/Scheduler interfaces |
| Conflating recovery re-planning with `Planner` | Recovery planner is a separate lighter-weight path |

---

## Tests

- `tests/unit/test_recovery.py`
- `tests/unit/test_autonomous_recovery.py`

## Next: inspect

Runtime state after recovery → `runtime/AGENTS.md`.
Scheduling after recovery → `agents/AGENTS.md`.
