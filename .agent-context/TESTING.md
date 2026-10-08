# Testing and Verification

## Project commands

```powershell
uv run pytest tests/ -x -q
uv run pytest tests/unit/test_<area>.py -x -q
uv run harness --help
uv run harness
uv run python -c "import harness_core; print('OK')"
```

Use the subsystem's `AGENTS.md` to choose focused tests. Do not run the full
suite automatically for every documentation or UI change.

## Live provider checks

- Unit tests do not establish live OpenRouter availability or credentials.
- `/models` in the interactive shell lists the configured free pool and does
  not require live model discovery.
- A real task requires network access and a valid `OPENROUTER_API_KEY`.
- For live failures, record HTTP status/category and a redacted response detail;
  never print or store the credential.
- Verify model failover using a real `model.failover` event or a deterministic
  test. Do not claim failover from a static list alone.

## Last observed verification (2026-09-29, P0 orchestration)

- `tests/unit/test_model_health_failover.py`: 15 tests, cases A–M (cooldowns
  for 429/wrapped-502/timeout, Retry-After, fail-fast timing <1s, 401/403
  stop, health survives new chains and mode switches, tool-not-repeated).
- Focused routing/fallback/health suites: **254 passed**. Full unit: 1,546
  passed, 2 failed (pre-existing completion-UX copy mismatches).
- `harness doctor`: [OK] OpenRouter. `harness --help`: OK.
- Live: all-unavailable → instant structured fail-fast with per-model
  cooldown lines (KI-009 account-level rate limiting active). Do not spam
  live retries; cooldown expiry restores normal selection automatically.

## Prior verification (2026-09-29, free-pool failover fix)

- Regression tests for the real Nemotron failure (HTTP-200-wrapped
  code-502/provider_unavailable) live in `tests/unit/test_openrouter_failover.py`:
  classification, provider message preservation, and a full 4-model chain
  ending in `openrouter/free` success.
- Focused failover suites (`test_openrouter_failover.py`,
  `test_provider_error_propagation.py`, `test_free_routing.py`,
  `test_provider_auth_failure.py`, `test_ux_upgrade.py`,
  `test_production_readiness.py`, `test_routing.py`,
  `test_runtime_governance.py`): **all pass**.
- Full unit run: **1,531 passed, 2 failed** — `test_failure_summary_is_honest`
  and `test_success_completion` (pre-existing UI-copy mismatches in files with
  another agent's uncommitted changes; fail at HEAD too).
- Live `harness run`: failover through the full pool observed (Gemma 31B
  rate-limited → Nemotron provider_unavailable → `openrouter/free` executed a
  tool and continued); simple task `Status: completed` on the primary.
- Live probing etiquette: use ONE minimal request (`max_tokens` small) when a
  probe is necessary; never spam rate-limited free models.

## Prior verification (2026-09-29, OpenRouter failure investigation)

- Direct OpenRouter probes (project `.env` key, secrets redacted): 401
  "User not found." for Gemma 26B, Gemma 31B, and `openrouter/free`; `GET
  /auth/key` also 401. Key loading and request construction verified correct.
- `uv run harness --help`: passed. `uv run harness doctor`: `[FAIL]
  OpenRouter` — consistent with the 401 (Harness path independently confirms
  the external credential rejection).
- `uv run pytest tests/unit/test_provider_auth_failure.py
  tests/unit/test_provider_error_propagation.py tests/unit/test_free_routing.py
  -q`: **54 passed** (includes the new truncation regression test).
- Live task completion, tool use, and live 429 failover remain blocked by the
  invalid OpenRouter key (KI-001); failover determinism is covered by unit
  tests.
