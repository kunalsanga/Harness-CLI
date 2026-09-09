# Phase 8 Audit — Persistent Memory, Knowledge Graph & Cross-Task Learning

Date: 2026-09-04 · Branch: `main` · Status: **all tests green**

Run the verification yourself:

```bash
uv run pytest -q                 # full suite: 1306 passed, 1 skipped, 0 failed
uv run pytest tests/unit/test_memory_store.py tests/unit/test_memory_retrieval.py \
                 tests/unit/test_memory_manager.py tests/unit/test_knowledge_graph.py \
                 tests/unit/test_memory_rag_integration.py tests/e2e/test_memory_pipeline.py
```

## What was delivered

| Area | Location |
|---|---|
| Domain model (`MemoryType`, `MemoryEntry`, graph nodes/edges) | `src/harness_core/memory/domain.py` |
| Pluggable store (`MemoryStore` ABC + `LocalMemoryStore`, URL factory) | `src/harness_core/memory/store.py` |
| TF-IDF embedding indexer | `src/harness_core/memory/indexer.py` |
| Retrieval (`retrieve_similar/_failures/_successes/_architecture_decisions/_project_context`, role-aware) | `src/harness_core/memory/retriever.py` |
| Knowledge graph (nodes, edges, BFS, persistence) | `src/harness_core/memory/graph.py` |
| Secret sanitizer | `src/harness_core/memory/sanitizer.py` |
| Retention policy (delete / archive / summarise) | `src/harness_core/memory/retention.py` |
| `MemoryManager` — the **only** write surface, dedup, graph sync, RAG helpers | `src/harness_core/memory/manager.py` |
| Runtime wiring | `Planner` (8F), `WorkerAgent` (8G), `Scheduler` (8E/8G), `RecoveryOrchestrator` (8E) |

Memory is **opt-in**: nothing is recorded unless a configured `MemoryManager` (from
`init_memory_manager(workspace, enabled=True)`) is supplied to the orchestrator/planner/scheduler.
All record APIs short-circuit when `enabled=False`.

---

## Audit answers

### A. Can agents remember previous projects? — **Yes**

Every execution writes structured entries tagged with `project_id`, `agent_role`,
`task_id`, `type`, `importance_score`, and an embedding. Entries persist atomically to
`.harness/memory/entries.json`; the knowledge graph persists to `graph.json`.
`retrieve_project_context(project_id)` returns the per-project overview, and project
isolation is tested (`test_two_projects_do_not_leak_context`).

### B. Can Planner reuse prior solutions? — **Yes**

`Planner.plan()` asks the manager for prior similar projects, past failures, prior
successes, and architecture decisions and appends them as a `# Prior Project Context`
block in the system prompt (Phase 8F). Verified at the class boundary:
`test_planner_injects_prior_memory_context`, and a broken-memory fallback test
proves memory can never take planning down (`test_planner_memory_failure_does_not_break_planning`).

### C. Can WorkerAgents retrieve historical context? — **Yes**

`WorkerAgent._build_prompt_async()` (run at start-up, before execution) retrieves
role-permitted, project-scoped memories for the task objective and injects them as a
`## Historical Context` block under headings such as `Past Failures to Avoid`
(Phase 8G). Verified at the class boundary in `test_worker_prompt_includes_historical_context`.

### D. Are memories role-aware? — **Yes**

Each entry records the producing `agent_role`; the retriever whitelists the memory
*types* each role may read (`_READABLE_TYPES_BY_ROLE`). Specialists
(backend/frontend/database/coder/tester/…) cannot read ARCHITECTURE/DECISION entries;
planner/architect/reviewer/security_reviewer/orchestrator have broader visibility.
Verified by `test_retrieval_respects_permissions`,
`test_worker_context_respects_role_permissions`, and the role-mapping unit tests.

### E. Are secrets persisted? — **No**

Every write passes `MemorySanitizer` **before** reaching the store; no code path can
bypass it because `MemoryManager` is the only write surface. The sanitizer strips
OpenAI/Anthropic/Stripe keys, AWS keys, GitHub PATs, JWTs, bearer tokens, PEM keys,
connection strings, and `*_KEY=…` assignments, and additionally strips IPs, internal
hostnames, and absolute paths from security-reviewer entries. Proven by
`test_secret_persistence_blocked`, `test_secret_persisted_as_redacted`,
`test_secret_in_entry_is_redacted_on_retrieval`, and
`test_shell_smuggled_secret_is_sanitized`.

### F. Is memory poisoning possible? — **No (from the model/agent side)**

Models and agents have **read-only** access (the retriever). Only runtime code invokes
the typed `record_success/record_failure/record_architecture/record_decision/record_episode`
writes. Content is inert data: a memory cannot forge its own `type`, `role`, `task_id`,
or project (identity fields are runtime-owned) — `test_content_cannot_forge_memory_identity`;
prompt-injection text is stored and rendered strictly inside its own failure entry —
`test_prompt_injection_text_stored_as_inert_content`; duplicate writes are deduplicated
(hash + project) so spam cannot flood the store — `test_duplicate_dedup_within_project`.
*Residual risk (by design):* the runtime itself is the trust boundary — code that calls
`record_success` with false content can still record false memories. There is no
model-facing path that bypasses the manager.

### G. Is retrieval permission-aware? — **Yes**

Every read honours the role whitelist, project scope, tag filters, and archive flag,
and results are sanitised again before being formatted into a prompt
(`retrieve_for_role` / `get_context_for_worker`). If nothing scores above the
similarity threshold, the retriever falls back to the top *readable* entries rather
than silently returning empty context — permissions still gate what is visible.
Verified by the retrieval + e2e permission tests.

### H. Does memory survive process restart? — **Yes**

Entries (including their embeddings, tags, importance, and hashes) and the graph are
written to disk atomically (write-temp-then-rename). A brand-new manager built over the
same workspace reads the previous run's entries and can produce RAG context
(`test_memory_survives_process_restart`). Query embeddings also work against an empty
fresh indexer because IDF is smoothed (see defects, below).

### I. Is the knowledge graph actually queried? — **Yes (as an API), not yet by the RAG path**

The graph is written on every task record (TASK node + CREATED/CAUSED_BY edges),
persisted, and fully queryable — `neighbors()`, `related()` (BFS to depth), `find_path()`,
type/edge filters — with a dedicated test suite (`test_knowledge_graph.py`) and
exposure via `memory.stats()["graph_stats"]`. **Gap:** Planner/Worker prompt
augmentation currently queries *memory entries*, not graph traversal, so graph edges
are recorded and queryable but not yet consumed to expand task context. Follow-up:
have `get_context_for_planner`/`get_context_for_worker` consult `graph.related()`
for sibling decisions/failures before prompt injection.

### J. Is this true long-term memory or prompt stuffing? — **True persistent memory**

Not a chat-log replay: entries are typed, embedded, tagged, importance-scored,
project- and role-scoped objects persisted to disk; every run retrieves over the stored
corpus (with fallbacks), deduplicates on write, applies a retention policy
(delete → summarise → archive), and enforces permissions and redaction. Two honest
qualifiers: (1) embeddings are lexical TF-IDF-hash vectors, not semantic model
embeddings — good enough for "did I build JWT auth before", not for paraphrase
matching; (2) the shipped backend is the local JSON store — Chroma/Qdrant/Weaviate/
PGVector are interface-ready (`MemoryStore` + `memory_store_from_url`) but not yet
implemented.

---

## Adversarial proof matrix (all passing)

| Claim | Proof test |
|---|---|
| Memory poisoning is blocked | `test_content_cannot_forge_memory_identity`, `test_prompt_injection_text_stored_as_inert_content`, `test_memory_poisoning_not_possible` |
| Secret persistence is blocked | `test_secret_persistence_blocked`, `test_secret_persisted_as_redacted`, `test_secret_in_entry_is_redacted_on_retrieval`, `test_shell_smuggled_secret_is_sanitized`, `test_role_redaction_strict_for_security_reviewer` |
| Duplicates are deduplicated | `test_deduplication_same_content_same_project`, `test_duplicate_dedup_within_project`, `test_duplicate_content_different_projects_creates_separate_entries` |
| Retrieval respects permissions | `test_retrieval_respects_permissions`, `test_retrieve_for_role_restricts_types`, `test_worker_context_respects_role_permissions`, `test_retrieval_respects_role_permissions` |
| Robustness under attack | `test_no_path_traversal_in_memory_ids`, `test_poisoning_via_self_loop_does_not_loop_forever`, corrupt-file loads (`test_corrupt_json_file_loads_gracefully`, `test_malformed_entry_in_json_skipped`) |

## Defects found and fixed in this pass

1. **Indexer produced all-zero embeddings** (`indexer.py`): IDF was `log(N/df)`, which is
   `0` for a single-document corpus and after a restart with no doc-frequency stats —
   making every query score 0 and stores unsearchable. Switched to smoothed IDF
   (`log((1+N)/(1+df)) + 1`).
2. **Role-scoped retrieval returned nothing** when the query shared no tokens with
   readable entries; added a readable-types fallback ranked by importance/recency
   (`retriever.py`).
3. **Retention policy `AttributeError`** (`retention.py`): `apply()` called the
   non-existent `should_summarize` (defined as `should_summarise`); reordered branches
   (delete → summarise → archive) so summarise is reachable, and archive/summarise now
   preserve the full entry (previously they replaced entries with sparse stubs that lost
   content, tags, hash, and metadata).
4. **Broken test fixtures**: `tmp_path(self)` fixtures in `test_memory_manager.py` and
   `tests/e2e/test_memory_pipeline.py` shadowed pytest's built-in and failed with
   `fixture 'self' not found`; removed (tests use the built-in).
5. **Stale test API**: `test_handoff_integration.py` called the removed synchronous
   `WorkerAgent._build_prompt()`; updated to `await _build_prompt_async()`.
6. **`graph.py`**: corrected `deque[str]` annotation to `deque[tuple[str, int]]`.

## Integration hardening (post-audit follow-up)

Two audit gaps were closed/verified in a follow-up pass; full suite is now
**1315 passed, 0 failed** (`uv run pytest tests/`).

### 1. Memory is now wired into the real CLI path — YES

The normal `harness run --mode multi-agent` path now bootstraps persistent project
memory automatically:

```
CLI (run) → init_memory_manager_from_project(.harness/config.yaml)
          → Orchestrator(memory=…) → Planner(memory=…) / Scheduler(memory=…)
          → WorkerAgent(memory=…) → MemoryManager → .harness/memory/
```

- **Default on** for initialised projects — `harness init` now writes
  `memory: { enabled: true }` into `.harness/config.yaml`. Opt out by setting
  `enabled: false`; uninitialised directories (no `.harness/config.yaml`) get **no**
  memory side effects at all (legacy behaviour preserved).
- Reuses the existing `MemoryManager` / `init_memory_manager` — no second memory
  system. All write/read paths (sanitizer, dedup, role+project scoping, poisoning
  protections) are unchanged because the CLI simply constructs the same manager the
  tests construct.
- Bootstrap is best-effort: `init_memory_manager_from_project()` never raises (it
  degrades to a disabled manager / `None`), and every planner/worker/scheduler memory
  call is wrapped so a memory failure cannot fail an engineering task.

Evidence (integration tests driving the *real* runtime construction path, not a bare
`MemoryManager`):

| Test | Proves |
|---|---|
| `tests/e2e/test_cli_memory_integration.py::test_multi_agent_run_initializes_persistent_memory` | Real `harness run --mode multi-agent` (Typer CliRunner, stubs only at the provider/loop boundary) writes SUCCESS memories to `.harness/memory/entries.json`, project-scoped to the resolved workspace path |
| `...::test_multi_agent_run_opt_out_via_config` | `memory.enabled: false` keeps the run green with **no** memory store created |
| `...::test_run_in_uninitialized_dir_does_not_touch_memory` | Legacy path: no `.harness` side effects without project config |
| `tests/unit/test_memory_scheduler_wiring.py` | `init_memory_manager` → real `Scheduler` → real `WorkerAgent` records SUCCESS memory; seeded FAILURE memory is injected into the worker prompt (`## Historical Context`); seeded SUCCESS is injected into the Planner prompt (`# Prior Project Context`) |
| `...::test_worker_survives_memory_write_and_retrieval_failures` | Sabotaged store/retrieval: task still COMPLETES, nothing is persisted |
| `...::test_disabled_manager_keeps_runtime_working` | `memory=None` behaves like before Phase 8 |

**Project isolation under the CLI** is structural: each workspace resolves its own
`.harness/memory/` directory and a project_id of `Path.cwd().resolve()`; every entry is
stored and retrieved under that id, so Project A's entries can never surface in Project
B. Security properties (sanitizer redaction, role-permission-aware retrieval, identity
forgery/prompt-injection protections) ride along unchanged because the CLI path ends at
the same `MemoryManager`.

### 2. Knowledge-graph boundary — verified, left as an explicit limitation

Audit answer I is accurate and still stands: the graph is written on every task record
(TASK node + CREATED/CAUSED_BY edges in `manager._record`), persisted to
`.harness/memory/graph.json`, and fully queryable (`neighbors()`/`related()`/`find_path()`
via `tests/unit/test_knowledge_graph.py`), but Planner/Worker RAG consults *memory
entries* (vector store), not graph traversal. Wiring graph edges into RAG would need a
graph-aware expansion step plus blending heuristics against similarity results — real
new abstractions, not a trivial drop-in — so it remains the documented Phase 8
limitation rather than a forced integration.

## Known gaps / next steps

1. Query the knowledge graph (`related()` / `neighbors()`) during planner/worker RAG so
   graph edges feed prompts, closing audit question I's gap.
2. Implement the first real vector-database backend (Chroma or PGVector) behind
   `memory_store_from_url` when the corpus outgrows the JSON store.
3. Run retention pruning on a schedule/at startup (today it is invoked on demand via
   `manager.prune()`).
