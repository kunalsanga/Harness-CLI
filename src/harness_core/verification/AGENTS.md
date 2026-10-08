# verification/ — Verification Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** Runs structured verification checks (tests, linting, type checking) and reports `VerificationReport`. Ensures agent work is actually correct before completion is claimed.

---

## READ FIRST

- `engine.py` — `VerificationEngine`, `VerificationCheck`, `VerificationResult`, `VerificationReport`
- `integrity.py` — `check_test_integrity` (validates tests actually exercised claimed code)

---

## Verification Flow

```
AgentLoop (near completion)
   → VerificationEngine.run(checks)
        → for each VerificationCheck:
             subprocess(command, timeout)
             → VerificationResult(passed, output, error)
   → VerificationReport(all_passed, results)
   → on failure: raise / return to AgentLoop for recovery
```

---

## Dependencies

**Depends on:**
- `agent/types.py` — `ToolResult`, `ToolResultStatus`

**Used by:**
- `agent/loop.py` — `AgentLoop` invokes `VerificationEngine`

---

## Architectural Invariants

- **Never bypass `VerificationEngine`.** Do not set `VerificationStatus.PASSED` manually.
- `VerificationReport.all_passed` must be `True` before task completion can be claimed.
- `check_test_integrity()` must be called when tests are the primary verification signal.
- Verification checks run as subprocesses — do not import or call project code directly.
- Timeouts are enforced — do not set infinite timeouts.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Faking `VerificationStatus.PASSED` | Always run `VerificationEngine` |
| Skipping integrity check for test verification | Call `check_test_integrity()` |
| Calling project code directly instead of subprocess | Verification runs as isolated subprocess |
| Not surfacing failed checks to `RecoveryOrchestrator` | Failed report must flow to recovery |

---

## Tests

- `tests/unit/test_verification.py`

## Next: inspect

Test failures → `recovery/AGENTS.md`.
Agent claiming false completion → `agent/AGENTS.md` + `agent/completion.py`.
