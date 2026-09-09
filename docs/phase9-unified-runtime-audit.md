# Phase 9 Audit — Unified Engineering Runtime

Status: **COMPLETE**. Full suite: `1346 passed, 0 failed` (previous baseline
1315 — this phase adds 30 new tests and one environment-conditional skip now
passes). Collected `1346`, exit code `0`, 4 pre-existing
`PytestUnknownMark` warnings, duration `110.6s`.

## What was built

A thin orchestration seam, `src/harness_core/runtime/`, that owns one project
execution and coordinates the *existing* authorities — it re-implements none
of them:

```
EngineeringRuntime
 ├── ProjectState          (authoritative project view; references TaskGraph)
 ├── Requirements          (structured, runtime-owned)
 ├── TraceabilityIndex     (REQ -> tasks -> evidence -> status)
 ├── ProjectContext        (role-scoped context assembler, per §8)
 ├── Planner               (existing, memory-augmented)
 ├── TaskGraph             (existing, execution authority for tasks)
 ├── Scheduler             (existing, drives WorkerAgents + recovery)
 ├── RecoveryOrchestrator  (existing, now project-scoped memory writes)
 ├── AgentMessageBus       (existing, typed handoffs)
 ├── WorkspaceLockManager  (existing)
 ├── MemoryManager         (existing Phase 8)
 └── EventBus              (existing — the only observability mechanism)
```

Lifecycle stages (`DISCOVER → UNDERSTAND → PLAN → DECOMPOSE → EXECUTE →
INTEGRATE → TEST → DEBUG → REVIEW → VERIFY → DELIVER`) are entered as the
graph actually exercises them; `DESIGN` is defined but only entered when a
future stage introduces it (not every stage must run — §2). Terminal statuses
are `SUCCESS | FAILED | BLOCKED | CANCELLED`; `ProjectState.to_dict()` renders
the CLI/terminal snapshot (§16).

## Modified / new files

New:
- `src/harness_core/runtime/__init__.py`, `requirements.py`, `state.py`,
  `context.py`, `runtime.py`
- `tests/unit/test_runtime_state.py`, `test_requirements.py`,
  `test_project_context.py`, `test_traceability.py` (25 tests)
- `tests/e2e/test_unified_runtime.py` (5 E2E), `tests/e2e/test_cli_unified_runtime.py` (1 E2E)

Modified (additive, backward-compatible):
- `src/harness_core/planning/domain.py` — `PlannedTask.traceable_to`
- `src/harness_core/agents/domain.py` — `runtime_context` on `SubTask`/`AgentContract`
- `src/harness_core/agents/worker.py` — reads `contract.runtime_context` into the prompt
- `src/harness_core/agents/scheduler.py` — accepts a pre-built `AgentMessageBus`;
  threads `project_id` into `RecoveryOrchestrator`
- `src/harness_core/recovery/orchestrator.py` — `project_id` param (fixes a
  memory **project-isolation** leak: recovery failures were recorded with
  `project_id=""`)
- `src/harness_core/cli/main.py` — `harness run --mode unified` executes through
  `EngineeringRuntime` (legacy modes untouched)

## Final audit answers

**A. Is there exactly one authoritative runtime state?** Yes. One
`ProjectState` per execution, owned by `EngineeringRuntime`. It holds
project-level truth (stage, requirements, traces, artifacts, recovery
counters, verification posture) and only *derives* task numbers from the
TaskGraph — no second task database.

**B. Is TaskGraph still the authoritative task state?** Yes. Only the Scheduler
mutates task statuses; `ProjectState.task_graph` is a reference; readiness/
completion/failure all come from the graph.

**C. Can agents execute concurrently where dependencies permit?** Yes — the
real Scheduler dispatches independent tasks up to `max_concurrency`. The E2E
success scenario launches three independent roots (architect/database/
frontend) in one dispatch round; the run completes a 10-task graph with
`max_concurrency=3`.

**D. Can agents communicate through typed handoffs?** Yes. After execution the
runtime publishes `MessageType.HANDOFF` messages (structured payload:
`files_changed`, summary, artifact refs) over the shared `AgentMessageBus`;
the E2E asserts `integration_1` receives handoffs from `backend_1`.

**E. Can agents retrieve relevant persistent memory?** Yes. `MemoryManager`
is threaded into Planner (§8F), Scheduler/Worker RAG (§8G) and Recovery
(§8E). The memory E2E proves a real run writes SUCCESS + FAILURE memories,
all project-scoped, and that a disabled manager is non-fatal.

**F. Can failures trigger bounded autonomous recovery?** Yes. Test-failure →
`RecoveryOrchestrator` → debugger task → retest (all inside the real
scheduler). Exhaustion E2E confirms the bound: 3 fix attempts, then
`recovery_exhausted` and runtime `FAILED` — no infinite loop.

**G. Can recovery mutate the TaskGraph safely?** Yes.
`graph.insert_recovery_sequence` keeps the failed original `FAILED`
(immutable history) and rewires dependents to the retest. The runtime only
counts a failure as recovered when a completed retest of it exists
(`<base>_retest_<n>`), i.e. **new execution + evidence** (§18).

**H. Are requirements traceable through verification?** Yes. The trace index
maps REQ → planned tasks (runtime-validated `traceable_to` + keyword
fallback that never steals verifier-owned acceptance criteria). Evidence is
attached per completed task and per verifier verdict (`REQ-00X: PASS|FAIL`);
FAIL overrides; a requirement with mapped tasks is VERIFIED only when *all*
of them pass. The traceability E2E walks the full chain.

**I. Can the runtime resume safely after interruption?** Interruption is
**safe**: the cancellation E2E proves an in-flight task is marked CANCELLED
(never falsely COMPLETED), every workspace lock is released, and state is
terminal CANCELLED with no `task.completed`/`runtime_completed` events.
Honest limitation: this is in-process interruption safety — resuming a
half-finished graph *across a process restart* is not implemented (no new
persistence architecture was added, per the constraint).

**J. Can any model claim bypass verification?** No. Model output is only a
proposal: plans pass `PlanValidator` + runtime conversion; task results are
classified by the scheduler (never trusted verbatim); verifier verdicts are
parsed into evidence by the runtime and FAIL overrides PASS. Terminal SUCCESS
requires verification. With no structured requirements, verification falls
back to *graph-completion* evidence (matching legacy Orchestrator semantics)
rather than any model claim.

**K. Are permissions and workspace locks preserved?** Yes. The new layer
introduces no bypass: plans/resources still pass `PermissionManager`
path checks, `AgentProfile` role capabilities drive dispatch, and
`WorkspaceLockManager` acquire/release wraps every dispatched task (the
interruption test asserts the lock table is empty afterwards).

**L. Does the real CLI execute through EngineeringRuntime?** Yes —
`harness run --mode unified` runs the full CLI → EngineeringRuntime →
Planner → Scheduler → WorkerAgent → MemoryManager path; the CLI E2E test
invokes the real Typer command, asserts exit 0, a `success`/`succeeded`
snapshot, and project-scoped persisted memory. Legacy `run` modes
(`auto`/`single`/`multi-agent`) are preserved unchanged for backward
compatibility.

**M. Is the E2E workflow exercising real runtime orchestration rather than
test-only simulation?** Yes. Tests stub only the model provider and the
`WorkerAgent.run`/`AgentLoop` boundary. Planning, graph conversion, dispatch,
locking, concurrency, failure classification, recovery graph mutation,
handoffs, artifact recording and requirement verification are all production
code — the tests never set a task state directly.

## Remaining architectural limitations (honest)

1. **Knowledge-graph RAG is not consulted by planning/execution.** The Phase 8
   graph is written, persisted and queryable, and the runtime now stores
   artifact references via memory, but graph traversal is not yet blended into
   the RAG path. Left as a documented Phase 8 limitation per instruction
   (no forced graph abstractions).
2. **`DESIGN` lifecycle stage is defined but unused** by the current
   scheduler-driven flow (architecture work is carried by ARCHITECT-role
   tasks). No stage force-running was added.
3. **Cross-restart resume is not implemented** (see I). Interruption is safe
   in-process; durable resume would need session persistence integration,
   which was intentionally not rebuilt.
4. **Ad-hoc CLI runs without structured requirements** verify by graph
   completion rather than requirement evidence — acceptable and matches
   legacy behavior, but weaker than the REQ-trace path used when the runtime
   is given requirements.
5. **Recovery counting** mirrors the RecoveryOrchestrator: the exhausted
   attempt also emits `recovery_started`, so `recovery_attempts` is
   `max_attempts + 1` at exhaustion while actual fix sequences are bounded to
   `max_attempts` (3). Behavior is bounded and asserted; semantics preserved
   from Phase 7.
6. **CLI default modes are still Orchestrator-based**; EngineeringRuntime is
   the canonical path under `--mode unified`. A future phase may migrate the
   legacy modes onto the runtime behind compatibility wrappers.

Phase 10 has not been started.
