# planning/ — Planning Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** Decomposes user tasks into structured `Plan` objects with named steps. The `Planner` produces `PlanningResult` which `EngineeringRuntime` converts into a `TaskGraph`.

---

## READ FIRST

- `planner.py` — `Planner` (task decomposition logic)
- `domain.py` — `Plan`, `PlanningResult`, `PlanStep` (planning data types)

## READ IF NEEDED

- `validator.py` — `PlanValidator` (validates plan structure before execution)

## DO NOT READ FOR NORMAL PLANNING TASKS

- `agents/domain.py` — `TaskGraph` is downstream of planning; separate concern
- `runtime/runtime.py` — runtime converts plan to task graph; separate concern

---

## Planning Flow

```
User task string
   → Planner.plan(task, context)
   → LLM call (via provider)
   → Plan (named steps + requirements)
   → PlanningResult
   → EngineeringRuntime converts to TaskGraph
```

---

## Dependencies

**Depends on:**
- `providers/base.py` — LLM call for decomposition
- `routing/router.py` — model selection for planning call

**Used by:**
- `runtime/runtime.py` — `EngineeringRuntime` calls `Planner.plan()`

---

## Architectural Invariants

- Planning is a single LLM call — do not make it multi-step iteratively without understanding cost.
- `Planner` does not mutate `TaskGraph` — it only produces `Plan`.
- `PlanValidator` must be called before `EngineeringRuntime` accepts a plan.
- Do not add task-decomposition logic inside `AgentLoop` — all decomposition is `Planner`'s job.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Adding decomposition logic in `AgentLoop` | Use `Planner` instead |
| Skipping `PlanValidator` | Always validate plan structure before converting to `TaskGraph` |
| Directly creating `TaskGraph` without going through `Planner` | Planner produces the plan; runtime converts it |

---

## Tests

- `tests/unit/test_planning.py`
- `tests/unit/test_validation_semantics.py`
- `tests/unit/test_intent_classification.py`

## Next: inspect

Task graph issues → `agents/AGENTS.md`.
Planning LLM call → `providers/AGENTS.md`.
