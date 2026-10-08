# tests/ — Test Suite Orientation

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** All automated tests for Harness. Structured as `unit/`, `integration/`, `e2e/`, `benchmarks/`, and `fixtures/`.

---

## Test Commands

```bash
# Full suite (primary)
uv run pytest tests/ -x -q

# By subsystem
uv run pytest tests/unit/test_routing.py -x -q
uv run pytest tests/unit/test_ninerouter.py -x -q
uv run pytest tests/unit/test_recovery.py -x -q
uv run pytest tests/unit/test_planning.py -x -q
uv run pytest tests/unit/test_verification.py -x -q
uv run pytest tests/unit/test_worker_agent.py -x -q
uv run pytest tests/unit/test_tools.py -x -q

# Integration
uv run pytest tests/integration/ -x -q

# E2E
uv run pytest tests/e2e/ -x -q
```

Config: `pyproject.toml` → `asyncio_mode = "auto"`, `testpaths = ["tests"]`.

---

## Test → Subsystem Map

| Test file | Subsystem |
|---|---|
| `test_routing.py`, `test_14dim_routing.py`, `test_empirical_routing.py` | `routing/` |
| `test_ninerouter.py` | `providers/ninerouter.py` |
| `test_groq_discovery.py` | `providers/groq.py` |
| `test_nvidia_provider.py` | `providers/nvidia.py` |
| `test_free_routing.py`, `test_router_policy.py` | `routing/` + `providers/` |
| `test_agent_loop_integration.py`, `test_canonical_runtime.py` | `agent/loop.py` |
| `test_canonical_tool_result.py`, `test_tool_failure_propagation.py` | `agent/loop.py` + `tools/` |
| `test_completion_invariant.py` | `agent/completion.py` |
| `test_steering_askuser_micro.py` | `agent/steering.py` + `agent/ask_user.py` |
| `test_scheduler.py`, `test_scheduler_locks.py` | `agents/scheduler.py` |
| `test_worker_agent.py` | `agents/worker.py` |
| `test_planning.py`, `test_intent_classification.py` | `planning/` |
| `test_recovery.py`, `test_autonomous_recovery.py` | `recovery/` |
| `test_verification.py` | `verification/` |
| `test_tools.py`, `test_tolerant_editing.py` | `tools/` |
| `test_permissions.py`, `test_permission_fix.py` | `permissions/` |
| `test_git_accounting.py`, `test_git_identity_and_completion.py` | `tools/git.py` |
| `test_context_reuse.py`, `test_context_budget.py`, `test_context_compaction.py` | `context/` |
| `test_memory_manager.py`, `test_memory_store.py`, `test_memory_retrieval.py` | `memory/` |
| `test_interactive_shell.py`, `test_production_ux.py` | `cli/` |
| `test_runtime_state.py`, `test_runtime_governance.py` | `runtime/` |
| `test_session_m4.py` | `session/` |
| `test_metrics.py`, `test_observability.py`, `test_semantic_events.py` | `observability/` |
| `test_provider_auth_failure.py`, `test_provider_error_propagation.py` | `providers/` + `config/` |
| `test_security_audit.py` | `security/` + `config/` |

---

## Architectural Invariants for Tests

- Tests must not make real network calls to LLM providers — use mocks.
- Tests must not read or write outside `tests/fixtures/` or a temporary directory.
- Do not delete tests to fix failures — fix the implementation.
- `asyncio_mode = "auto"` means all `async def test_*` are auto-collected; no `@pytest.mark.asyncio` needed.
- Fixtures are in `tests/fixtures/` — do not duplicate fixture data in individual test files.
- Integration tests may use `conftest.py` at the test root.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Deleting tests to fix a test failure | Fix the implementation |
| Making real LLM calls in unit tests | Mock providers |
| Writing outside `tests/fixtures/` | Use `tmp_path` pytest fixture |
| Adding `@pytest.mark.asyncio` | Not needed — `asyncio_mode = "auto"` |
