# routing/ — Model Routing Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** `ModelRouter` selects the best available provider/model for each request using 14-dimension scoring, health tracking, fallback chains, budget enforcement, and task classification. This is Harness-side routing — distinct from 9router's internal routing.

---

## READ FIRST

- `router.py` — `ModelRouter`, `RouterConfig`, `RoutingDecision` (start here)
- `fallback.py` — `FallbackEngine`, `FallbackConfig`, `FallbackResult`

## READ IF NEEDED

- `scoring.py` — `rank_models()`, `ScoringContext`, `ScoringWeights` (14-dimension model scoring)
- `health.py` — `ModelHealthTracker` (per-model health state)
- `task_aware.py` — `TaskAwareRouter` (task-type routing policies)
- `budgets.py` — `BudgetManager`, `BudgetConfig` (cost enforcement)

## DO NOT READ FOR NORMAL ROUTING TASKS

- `providers/ninerouter.py` — 9router is a provider, not a routing concern
- `agent/loop.py` — loop calls router; don't read loop to fix routing
- `cli/` — unrelated

---

## Routing Modes

`RouterConfig.routing_mode`:
- `auto` — 14-dimension scoring, best available model
- `free` — prefer zero-cost models (`prefer_free=True`)
- `best` — highest quality score
- `fast` — lowest latency
- `local` — prefer local/Ollama models
- `cheap` — lowest cost

---

## 9router and ModelRouter — Critical Distinction

| | ModelRouter | NineRouterProvider |
|---|---|---|
| **What it does** | Selects which provider/model to call | Calls the 9router gateway |
| **Location** | `routing/router.py` | `providers/ninerouter.py` |
| **Scope** | Harness-side selection across all providers | Delegates to 9router gateway |

When 9router is selected by `ModelRouter`, the request goes to `NineRouterProvider` with `model="qd/auto"`. The 9router gateway then selects the underlying model. **Do not add underlying model logic in either `router.py` or `ninerouter.py`.**

---

## Dependencies

**Depends on:**
- `providers/base.py` — `ModelProvider`, `ModelInfo`
- `observability/events.py` — `EventBus` (routing decisions are observable)

**Used by:**
- `agent/loop.py` — `AgentLoop` calls `ModelRouter.route()`
- `agents/worker.py` — `WorkerAgent` passes router to loop

---

## Architectural Invariants

- Routing scoring uses 14 dimensions — do not collapse to a simpler heuristic without understanding the impact.
- Fallback chains must respect health state — do not retry unhealthy providers indefinitely.
- Budget enforcement must run before model selection — do not route after budget is exceeded.
- `RoutingDecision` should be emitted on `EventBus` for observability.
- Do not expand 9router's `qd/auto` into underlying model IDs in routing scoring.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Treating 9router as a scored model (with underlying IDs) | 9router is one provider; `qd/auto` is opaque to Harness |
| Adding routing logic inside `NineRouterProvider` | Routing is `router.py`'s concern |
| Ignoring health state when building fallback chains | Use `ModelHealthTracker` |
| Bypassing `BudgetManager` for "fast" paths | Budget enforcement is mandatory |

---

## Tests

- `tests/unit/test_routing.py`
- `tests/unit/test_14dim_routing.py`
- `tests/unit/test_empirical_routing.py`
- `tests/unit/test_router_policy.py`
- `tests/unit/test_free_routing.py`
- `tests/unit/test_context_budget.py` (budget integration)

## Next: inspect

Provider issue → `providers/AGENTS.md`.
Model selection wrong → `scoring.py` + `task_aware.py`.
Fallback wrong → `fallback.py`.
