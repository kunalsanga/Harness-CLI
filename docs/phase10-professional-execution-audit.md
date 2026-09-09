# Phase 10 Audit — Professional Execution, Completion Honesty & Live Engineering Dashboard

Date: 2026-09-04 · Branch: `main` · Status: **all tests green**

Run the verification yourself:

```bash
uv run pytest -q                    # full suite: 1443 passed, 2 skipped, 0 failed
uv run pytest tests/unit/test_completion_ux.py tests/unit/test_runtime_dashboard.py \
                 tests/unit/test_governance.py tests/unit/test_validation_semantics.py \
                 tests/unit/test_intent_classification.py tests/unit/test_completion_invariant.py \
                 tests/unit/test_canonical_tool_result.py tests/e2e/test_professional_cli.py \
                 tests/e2e/test_live_ui_e2e.py tests/e2e/test_concurrency_e2e.py
```

Baseline: Phase 9 shipped at **1346 passed, 0 failed**. This phase adds the
professional execution layer (+97 tests). The full suite now collects
**1445**: `1443 passed, 2 skipped` on a cold run; the skip count is
environment-dependent because `tests/e2e/test_provider_validation.py` skips
the live-OpenRouter test when rate-limited (429) or when `OPENROUTER_API_KEY`
is unset — on a rate-limit-free run it is `1444 passed, 1 skipped`.

## What was delivered

| Area | Location |
|---|---|
| Authoritative task state machine + **completion invariant** (`can_complete_task`, `completion_blockers`, `can_transition`) | `src/harness_core/agent/completion.py` |
| `PARTIAL` task status — required work unresolved but nothing failed, **never rendered as success** | `src/harness_core/agent/types.py`, `agent/report.py` |
| Required-TODO skip trust rule (model prose can never waive required work) | `src/harness_core/agent/types.py` (`TaskPlan.skip_id(authorized=…)`) |
| Deterministic **intent classification** (EXPLAIN/QUESTION/INSPECT/REVIEW/MODIFY/FIX/TEST/DOCUMENT/GIT) + read-only plan guard | `src/harness_core/agent/intent.py` |
| Composite-request guard: "explain and fix" never satisfied by a read-only pass | `src/harness_core/agent/workflows.py` |
| **Completion UX**: `CompletionClassifier` (7 explicit states), `NextActionEngine` (evidence-derived 2–4 actions), `CompletionFormatter` (success/failure/cancellation panels) | `src/harness_core/cli/completion.py` |
| **Live engineering dashboard**: `RuntimeViewModel` (pure Rich-free adapter over real EventBus events) + `LiveTerminalUI` (throttled Rich `Live` render) | `src/harness_core/cli/runtime_dashboard.py` |
| Identity-stamped tool/test events (task_id + agent_id on every emitted event) | `src/harness_core/agent/loop.py` |
| **Runtime governance**: `ConvergenceGovernor` (stagnation evidence → strategy-switch suggestion) + `BudgetGovernor` (per-stage token/tool budgets) | `src/harness_core/runtime/governance.py`, wired in `runtime/runtime.py` |
| **Context reuse** (`ContextReuseManager` — snapshot-aware file re-read avoidance) | `src/harness_core/context/reuse.py` |
| **Enforced prompt token ceiling** (three-tier: history budget → per-result cap → hard ceiling; oversized results shrink head+tail, never exceed the model window) | `src/harness_core/agent/loop.py` |
| Canonical `ToolResult` contract — every tool goes through the `ToolResults` factory namespace | `src/harness_core/tools/*`, `tests/unit/test_canonical_tool_result.py` |
| Per-agent `model_policy` routing (reasoning_high / coding / fast / cheap mapped into the 14-dimension scoring context) | `src/harness_core/routing/router.py` |
| Git zero-tool-call accounting (workflow git ops recorded as first-class ToolCalls with events) | `src/harness_core/agent/workflows.py` |
| Validation semantics: a previous green suite never masks a later failing command; git evidence from real git events | `src/harness_core/cli/runtime_dashboard.py` |
| Professional CLI: startup panel, evidence-based completion/failure summaries, honest Ctrl+C cancellation (exit 130, locks released, session preserved) | `src/harness_core/cli/main.py`, `cli/interactive.py` |
| `pyproject.toml`: declared `e2e`/`integration` pytest markers (removes the 4 pre-existing `PytestUnknownMark` warnings) | `pyproject.toml` |

## Audit answers

### A. Can the runtime declare COMPLETED only when work truly completed? — **Yes (hard invariant)**

`can_complete_task(task)` is the single gate every code path must pass:
not cancelled/paused, **no pending/failed/in-progress REQUIRED TODO**, no
unresolved tool failure, no unresolved blocking error, required evidence
present, and verification passed whenever files were modified and
verification ran. `completion_blockers()` renders the human-readable
reasons. Proven by `tests/unit/test_completion_invariant.py`
("4/5 TODOs cannot complete a task", "5/5 can", "model failure must not
erase tool success", "no fake completion from model prose") and
`tests/unit/test_tool_failure_propagation.py`
("TOOL FAILURE ≠ TASK SUCCESS": a non-zero shell exit never marks the task
COMPLETED).

### B. Is PARTIAL honest? — **Yes**

`TaskStatus.PARTIAL` exists for "required work unresolved and nothing
failed", and `build_execution_report` renders it as `⚠ Partially
completed` with the completed/remaining TODO breakdown and a verification
line — never as `✓ Task completed`. `RuntimeViewModel` maps the same state
to a `partial` chip (`⚠`, yellow) in the dashboard.

### C. Can model prose waive required work? — **No**

`TaskPlan.skip_id(..., authorized=...)`: a REQUIRED TODO can only be
skipped when runtime code with evidence passes `authorized=True`
(dependency failure, clean-tree git skip). A model text response can never
skip required steps — the old behavior let 3/5 TODOs render as COMPLETE.
Covered by `test_completion_invariant.py` and the workflow
"nothing to commit" clean-tree path in `test_git_accounting.py`.

### D. Does the runtime understand *what* the user asked before planning? — **Yes (deterministic)**

`classify_intent()` maps goals to EXPLAIN/QUESTION/INSPECT/REVIEW/
MODIFY/FIX/TEST/DOCUMENT/GIT with a fixed precedence
(GIT > TEST/FIX > DOCUMENT > MODIFY > read-only) and no model call.
**Read-only intents never acquire REQUIRED modification tasks** — the
plan step-filter drops them (`is_read_only_verb`). Composite requests
("explain why tests fail and fix them") contain a modification verb, so
they are classified as work and routed through the generic LLM loop —
never the read-only explain workflow that would complete without doing the
fix (`test_intent_classification.py`, 20 tests).

### E. Is the dashboard a fabrication risk? — **No (pure adapter)**

`RuntimeViewModel` derives every field from real EventBus events (or the
authoritative `RuntimeOutcome` via `finalize()`); it performs no model
calls, no network I/O, no filesystem scans, and never synthesizes agent
activity, progress, file changes, or success. Tool/test events are
identity-stamped by the AgentLoop (`task_id` + `agent_id`), so the
dashboard attributes activity truthfully instead of guessing. The live-UI
E2E drives the *real* runtime → scheduler → worker chain and asserts the
terminal snapshot (agents, recovery, verification, files) matches the
authoritative runtime state.

### F. Does a completed test run ever mask a later failure? — **No**

Validation semantics (Parts 11–13, 30): a failing test command after an
earlier green run is recorded as its *own* latest-validation entry — the
stale green summary is overridden (`test_validation_semantics.py`). Git
state (`git_commit`, `git_push`) derives from real git tool events, never
from model prose.

### G. Is git activity counted truthfully? — **Yes**

Workflow-driven git operations (status/remote/identity/add/commit/push)
now record every execution as a real `ToolCall` via the
`record_tool_call` hook — the invalid state "Commit 06a6ae3 / Push
origin/main with 0 tool calls" is impossible
(`test_git_accounting.py`, incl. zero-tool-call regression).

### H. Can execution stagnate silently? — **No**

`ConvergenceGovernor` tracks repeated commands / identical failure
signatures / identical patches / no-progress iterations from real tool
evidence and reports `stalled` with a suggested strategy switch
(TESTER → DEBUGGER) and optional model escalation — it only reports
evidence; the runtime is the only authority that acts. `BudgetGovernor`
caps per-stage token/tool consumption so one phase (exploration,
debugging) cannot burn the whole run budget (`test_governance.py`).

### I. Can one oversized tool result blow the context window? — **No**

Three independent mechanisms in `_build_messages`: (1) history collapses
into a TASK STATE summary past a size trigger, (2) any single tool result
is capped, (3) `AgentConfig.context_token_budget` is a *hard ceiling* —
history is dropped until the assembled request fits, and if even the
prefix+summary+one tool pair overflows, the largest result is shrunk
(head + tail, marker inserted) rather than rejected by the provider.
`test_context_budget.py` asserts the ceiling holds for the exact failure
mode that previously died at 201,871 tokens against a 196,608 cap.

### J. Do role policies actually steer model choice? — **Yes**

`routing_mode_override` policies ("reasoning_high", "coding", "fast",
"cheap") are mapped onto the existing 14-dimension scoring vocabulary, so
an architect gets a reasoning-biased chain and a tester gets a fast/cheap
chain — policy tags take precedence over content heuristics
(`test_router_policy.py`, `test_agent_profiles.py`).

### K. Is parallel execution real, not simulated? — **Yes**

The concurrency E2E drives the REAL `EngineeringRuntime` → `Scheduler`
with three independent root tasks and one dependent task under
`max_concurrency=3`, and asserts *interval overlap* between the roots and
wall-clock duration well below the serial sum — demonstrable real
concurrency, not sequential dispatch (`test_concurrency_e2e.py`).

### L. Does the professional CLI stay honest end-to-end? — **Yes**

`harness run --mode unified` (headless and interactive): success renders
only what the runtime verified (`Status: VERIFIED`, `Tests: ✓ 2 passed`),
failure renders `Harness Stopped` + `No false success was reported.`,
`--json` remains the authoritative `ProjectState` snapshot, and Ctrl+C
renders the cancellation panel with exit code 130 while releasing locks
and preserving the session (`test_professional_cli.py`, 4 E2E).

## Defects found and fixed in this pass

1. **Model prose could waive required TODOs** — `skip_id` let a text
   response mark REQUIRED steps SKIPPED, so 3/5 TODOs could render as
   COMPLETE. Fix: `authorized=True` required for required steps; workflow
   clean-tree skips pass runtime evidence.
2. **PARTIAL was reported as success** — `build_execution_report` had no
   branch for unresolved-required-work; added the `⚠ Partially completed`
   path with completed/remaining/verification lines.
3. **Verify TODO could not be satisfied by a real test run** —
   `reconcile_on_evidence` only accepted the verification engine's verdict;
   a successful `run` with `tests_run > 0` now satisfies a "verify" step
   (secondary signal), and a later test failure fails it correctly.
4. **Oversized tool results could exceed the model window** — the old
   trigger-based compaction never enforced a ceiling; added the hard
   `context_token_budget` ceiling + single-result cap + head/tail
   shrinking (see I).
5. **Git workflows were invisible to accounting** — git operations ran
   without `ToolCall` records; the `record_tool_call` hook now records each
   with events and iteration/tool accounting.
6. **Undefined pytest markers** — `e2e`/`integration` markers undeclared
   (4 `PytestUnknownMark` warnings); declared in `pyproject.toml`.

## Known gaps / next steps

1. **Knowledge-graph RAG** (Phase 8 audit answer I, carried forward): the
   graph is written and queryable but Planner/Worker RAG still consults
   memory entries; wiring `related()`/`neighbors()` into prompt injection
   is the documented Phase 8 limitation.
2. **Cross-restart resume** (Phase 9 audit answer I, carried forward):
   interruption is safe in-process; durable resume of a half-finished
   graph across a process restart still needs session-persistence
   integration.
3. **Governance is advisory** — `ConvergenceGovernor` reports stagnation
   and recommends strategy switches, but the runtime's consumption of the
   report is currently limited to emitting `governor.stagnation`; wiring
   the recommendation into an actual TESTER→DEBUGGER dispatch decision is
   a follow-up.
4. **Legacy run modes remain Orchestrator-based** — `auto`/`single`/
   `multi-agent` are preserved unchanged; migrating them onto
   `EngineeringRuntime` behind compatibility wrappers is a future phase.
5. **Vector-database backend** — `memory_store_from_url` is interface-ready
   but only the local JSON store ships; Chroma/PGVector remain future work.