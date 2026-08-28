# M10 — Real-World Beta Validation & Coding Benchmark

## Executive Summary

M10 created a reproducible coding-task benchmark framework with 10 tasks across Python and TypeScript. The framework validates fixture correctness (8 intentional failures in Python fixtures confirmed). Real agent execution against providers was not completed due to OpenRouter rate limiting. The benchmark infrastructure is ready for future runs.

## Test Results

```
741 passed, 2 skipped in 42.43s
```

## Benchmark Framework

### Tasks Created (10)

| Task ID | Category | Difficulty | Language | Description |
|---------|----------|------------|----------|-------------|
| PY-BUG-001 | bug_fix | easy | Python | Fix add() for negative numbers |
| PY-BUG-002 | bug_fix | medium | Python | Fix capitalize_words and truncate |
| PY-FEAT-001 | feature | easy | Python | Add multiply and factorial |
| PY-REFACTOR-001 | refactor | medium | Python | Extract shared validation |
| PY-TEST-001 | testing | medium | Python | Add DataStore edge case tests |
| PY-DEBUG-001 | debugging | hard | Python | Fix rate limiter race condition |
| PY-SEC-001 | security | medium | Python | Fix SQL injection vulnerability |
| NODE-BUG-001 | bug_fix | easy | TypeScript | Fix fibonacci base cases |
| NODE-FEAT-001 | feature | easy | TypeScript | Add binarySearch |
| NODE-BUG-002 | bug_fix | medium | TypeScript | Fix URL parser ports/auth |

### Fixture Repositories

**Python App** — 7 source files, 4 test files, 8 intentional failures
- `src/calculator.py` — Calculator with negative-number bug
- `src/string_utils.py` — String utils with empty-string and truncation bugs
- `src/datastore.py` — DataStore with minimal tests (test-gen target)
- `src/user_service.py` — User service with duplicated validation
- `src/order_service.py` — Order service with duplicated validation
- `src/rate_limiter.py` — Rate limiter with race condition
- `tests/test_validation.py` — Security validation tests

**Node App** — 2 source files, 1 test file, 2 intentional failures
- `src/math.ts` — Math with swapped fibonacci base cases
- `src/url-utils.ts` — URL parser with port/auth bugs
- `tests/math.test.ts` — Math tests (2 should fail)

### Fixture Verification

Python fixture pre-fix: **8 failures, 24 passing** ✅ (confirmed intentional bugs exist)

## Benchmark Results

| Metric | Value |
|--------|-------|
| Tasks defined | 10 |
| Tasks executed | 0 (awaiting provider) |
| Fixture correctness | ✅ VERIFIED |
| Framework ready | ✅ YES |
| Real agent run | ⚠️ RATE LIMITED |

## Provider Status

| Provider | Status |
|----------|--------|
| OpenRouter | ⚠️ RATE LIMITED — framework ready, execution deferred |
| Ollama | ⚠️ UNVERIFIED — not running |
| LiteLLM | ⚠️ UNVERIFIED — not configured |

## Success Metrics Defined

- **FULL SUCCESS**: All tests pass, correct files modified, no regressions
- **PARTIAL SUCCESS**: Implementation mostly correct but tests fail
- **FAILURE**: Incorrect implementation

## Failure Taxonomy

| Category | Description |
|----------|-------------|
| MODEL | Model produced incorrect output |
| PROVIDER | Provider connection/auth failure |
| CONTEXT | Insufficient context for task |
| TOOL | Tool execution failure |
| PERMISSION | Security policy blocked operation |
| AGENT | Agent loop failure |
| ORCHESTRATION | Multi-agent coordination failure |
| VERIFICATION | Post-task test failure |
| PACKAGING | Installation/import failure |
| USER_ERROR | Invalid user input |

## Files Created

| File | Purpose |
|------|---------|
| `benchmarks/tasks.py` | 10 benchmark task definitions |
| `benchmarks/runner.py` | Benchmark runner with result tracking |
| `benchmarks/results/` | JSON result storage |
| `benchmark-fixtures/python-app/` | Python fixture (7 source, 4 test files) |
| `benchmark-fixtures/node-app/` | Node.js fixture (2 source, 1 test file) |
| `docs/m10-beta-report.md` | This report |

## Known Limitations

1. **Real agent execution not completed** — OpenRouter rate limiting prevented benchmark runs
2. **No Ollama** — Not running in test environment
3. **No LiteLLM** — Not configured
4. **No multi-agent benchmark** — Requires real provider access
5. **No cost data** — Requires provider usage reporting
6. **macOS not tested** — No environment available

## Product Decision

**B: Harness needs another reliability pass**

Rationale:
- Core product works (741 tests pass)
- Benchmark framework is ready
- Real coding task execution not yet validated with live provider
- Multi-platform CI incomplete
- Need real E2E runs to validate agent loop correctness

## Git Status

```
9 modified files
55+ untracked files
```

**No commit or push performed.**
