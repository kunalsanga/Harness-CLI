# Known Issues and Evidence Gaps

Last reviewed: 2026-09-30

## KI-010 — Latest OpenRouter doctor check failed (cause not established)

**Status:** Observed once on 2026-09-30 using `uv run --offline --no-sync
harness doctor`; Runtime and Agent System checks passed, Providers reported
OpenRouter `[FAIL]`. The command output did not expose a diagnostic category.
Earlier live success and rate-limit observations remain historical evidence;
do not infer that this latest result is an authentication failure. No repeated
live request was made during P1.

## KI-009 — OpenRouter free tier rate-limits the whole account (external, active)

**Status:** Verified live 2026-09-29. All four free-pool models return 429
under one shared account budget; a single task's planning call can consume
it and re-arm 60s cooldowns on every model. Harness behavior is now correct
(instant per-model fail-fast summary, no paid fallback, cooldown expiry
restores selection) but tasks cannot run until the account budget resets or
the user switches to a paid-capable key. Do not spam retries in testing.

## KI-001 — OpenRouter key rejected: 401 "User not found." — **RESOLVED (external side)**

**Status:** No longer reproduces as of the latest live session (2026-09-29).
`harness doctor` reports `[OK] OpenRouter`, and live chat requests succeed
(Gemma 31B primary, `openrouter/free` failover, tool execution). The earlier
401 "User not found." verdict was accurate at the time; the credential/account
condition has since been resolved externally. Keep this entry for history.

## KI-007 — Nemotron free endpoint exhausts its worker request limit (external, recurring)

**Status:** Verified live 2026-09-29 via one controlled probe.
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` returns OpenRouter
**HTTP 200** wrapping `error.code=502`,
`message="Upstream error from Nvidia: ResourceExhausted: Worker local total
request limit reached (16/16)"`, `metadata.error_type=provider_unavailable`.
This is a provider-side capacity limit on the free endpoint; Harness cannot
fix it. Harness now classifies it as MODEL_UNAVAILABLE and fails over —
verified live (openrouter/free took over and executed tools). Expect this
to recur while Nvidia's free capacity is saturated.

## KI-008 — Interactive completion gate ends simple tasks as "partial"

**Status:** Observed live in `harness run` (2026-09-29): a task that answered
the user's question still ended `Status: partial` with "required TODO(s)
still pending: Run automated tests, Fix failing tests, Verify changed
behavior" because the completion gate demanded test/verify TODOs for a
question-answering task. Unrelated to model failover; not fixed in this
session. Completion-gate logic lives in `agent/completion.py` / planner TODO
generation.

## KI-001-ARCHIVED — OpenRouter key rejected: 401 "User not found." (VERIFIED, external)

**Status:** Confirmed 2026-09-29 via direct probes using the project `.env`
key and Harness's own provider path. `OPENROUTER_API_KEY` loads correctly
(73 chars, `sk-or-v1-` + 64 hex, no whitespace/quotes, no registry override,
resolver source `env`). Direct requests to
`https://openrouter.ai/api/v1/chat/completions` returned **HTTP 401
`User not found.`** for `google/gemma-4-26b-a4b-it:free`,
`google/gemma-4-31b-it:free`, and `openrouter/free`; `GET /auth/key` returned
the same 401. Controls: no-auth request → 401 "Failed to authenticate request
with Clerk" (different error proves the header is read); fabricated
well-formed key → identical "User not found." Harness request construction is
correct; the key does not match an OpenRouter account. **Resolution requires
an OpenRouter account action (create/replace the key on the OpenRouter keys
page) — no code change can fix it.** `/verbose` shows the full diagnostic.

## KI-002 — Live successful task flow blocked by credential rejection

**Status:** Blocked by KI-001, not by connectivity. Real model response, tool
activity, completion, and live 429 failover remain unverified until a valid
key is present. Failover determinism is covered by unit tests (429→next,
5xx→next, 401/403→stop rotation: `test_provider_auth_failure.py`,
`test_provider_error_propagation.py`).

## KI-003 — Repository documentation has broader/older provider claims

**Status:** Documentation and CLI wiring were updated in the provider-neutral
implementation, but remain unverified. Adapter presence does not prove every
provider/model combination works end to end. Check the exact adapter and
capability declarations before claiming a provider feature is supported.

## KI-004 — Large unreviewed worktree

**Status:** Present at last check. Numerous modified and untracked application,
test, memory, and context files were visible. Ownership and intended commit
boundaries are unknown. Preserve these changes and inspect before modifying
overlapping areas.

## KI-005 — Full test suite failures in current worktree — **RESOLVED (2026-09-30)**

**Status:** Fixed. Full unit run is now **1,597 passed, 0 failed**. The 12
remaining failures were stale test fixtures predating two intentional
contracts: (1) paid-model opt-in routing gate (`allow_paid_models`, default
False — fixtures now use free models or explicit opt-in while preserving each
test's protected invariant: tool filtering, 429/timeout failover, cascading
failure, policy ordering); (2) read-only planning shortcut in AgentLoop —
read-only goals intentionally skip planning, so plan-backed TODO reconciliation
tests use an engineering goal. Also fixed production bug: `ConversationRenderer
.start()` never seeded `started_at` (dead elapsed clock before the first
execution.state event; now seeds `time.monotonic()`, event overwrites).
Wording/default drift reconciled: footer uses canonical "Completed",
`ModelProfile.supports_tools` is tri-state None, `providers list --json`
lists the full provider-neutral registry.

## KI-006 — Packaging build unavailable offline

**Status:** `uv build --offline --out-dir .tmp-build` could not resolve
`hatchling` because it was not cached. No wheel/sdist verification was
completed.
