# context/ — Context & Memory Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** Context assembly, token budget enforcement, context compaction, file-read deduplication (`ContextReuseManager`), and integration with the long-term `MemoryManager`. These are the systems that govern what enters each model call.

---

## READ FIRST

- `reuse.py` — `ContextReuseManager` (avoid re-reading unchanged files — highest leverage)
- `engine.py` — `ContextEngine` (context assembly coordinator)
- `pipeline.py` — `ContextPipeline` (pre-processing pipeline for each model call)

## READ IF NEEDED

- `compaction.py` — `ContextCompactor` (prune long context when token budget is exceeded)
- `pack.py` — `estimate_tokens()`, context packing utilities
- `relevance.py` — relevance scoring for context item selection
- `summarizer.py` — context summarization
- `budgets.py` — context budget enforcement
- `models.py` — `ContextRequest` and related types

## DO NOT READ FOR NORMAL CONTEXT TASKS

- `memory/` — `MemoryManager` is a separate persistence layer; read its `__init__.py` if needed
- `agent/loop.py` — loop uses context systems; context systems don't know about loop

---

## Three Distinct Systems

| System | File | Purpose |
|---|---|---|
| `ContextReuseManager` | `reuse.py` | Skip re-reading unchanged files within a run |
| `ContextPipeline` | `pipeline.py` | Assemble + compress context for each model call |
| `MemoryManager` | `memory/manager.py` | Persistent cross-session learned context (RAG) |

**Do not conflate these.** `ContextReuseManager` is intra-run; `MemoryManager` is cross-session.

---

## ContextReuseManager Usage Pattern

```python
crm = ContextReuseManager()

# After reading a file:
crm.record_read(path, size, mtime_ns, content)

# Before re-reading:
if crm.is_unchanged(path, size, mtime_ns, content):
    # skip — content hasn't changed
    pass

# After any write to a file:
crm.invalidate(path)

# On new task start:
crm.invalidate_all()
```

---

## Dependencies

**Depends on:**
- `memory/manager.py` — `MemoryManager` (optional RAG retrieval)
- `observability/events.py` — `EventBus`

**Used by:**
- `agent/loop.py` — `AgentLoop` uses `ContextEngine`, `ContextPipeline`, `ContextReuseManager`

---

## Architectural Invariants

- `ContextReuseManager` is intra-run only — clear state on task start.
- Token budget must be enforced before context is sent to model.
- `ContextCompactor` is the only sanctioned mechanism for context pruning.
- Do not implement ad-hoc truncation in `AgentLoop` — use `ContextPipeline`.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Re-reading files every model call | Check `ContextReuseManager.is_unchanged()` |
| Not invalidating after a write | Call `crm.invalidate(path)` after every write |
| Conflating `ContextReuseManager` with `MemoryManager` | Reuse = intra-run; Memory = persistent |
| Ad-hoc truncation in `AgentLoop` | Use `ContextCompactor` through `ContextPipeline` |

---

## Tests

- `tests/unit/test_context_reuse.py`
- `tests/unit/test_context_budget.py`
- `tests/unit/test_context_compaction.py`
- `tests/unit/test_context_intelligence.py`
- `tests/unit/test_context_pack.py`
- `tests/unit/test_project_context.py`

## Next: inspect

Memory / RAG → `memory/manager.py` + `memory/retriever.py`.
Token budget overflow → `context/budgets.py` + `context/compaction.py`.
