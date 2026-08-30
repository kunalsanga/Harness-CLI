# Harness CLI — Performance Analysis

## Architecture Overview

```
User Input
    ↓
Task Classification (LLM-based)
    ↓
Model Selection / Routing
    ↓
Agent Loop
    ├── Context Assembly
    ├── LLM Request → Response
    ├── Tool Execution (parallel where possible)
    └── Completion Check
    ↓
Verification
    ↓
Response
```

## Measured Bottlenecks

| Component | Current Implementation | Bottleneck | Expected Benefit | Candidate Rust/C++ | Priority |
|---|---|---|---|---|---|
| **LLM requests** | Async HTTP to OpenRouter | Network latency + model inference (dominant) | None (external) | No | N/A |
| **Model discovery** | HTTP fetch of 396+ models on startup | Initial request ~2-5s | Cache aggressively | No | High |
| **Workspace scanning** | `Path.rglob("*")` with per-file checks | ~0.1-0.5s for small projects | Already cached | `ignore` crate | Low |
| **Context assembly** | In-memory string operations | Negligible | N/A | No | N/A |
| **File reading** | `open().read()` per file | ~1-10ms per file | Batch reads | `tokio::fs` | Low |
| **Shell execution** | `asyncio.create_subprocess_exec` | Process spawn overhead | Use process pools | `std::process` | Medium |
| **Event bus** | In-memory async dispatch | Negligible | N/A | No | N/A |
| **Task classification** | LLM call to classify task type | ~0.5-2s | Use local classifier (already done) | No | Done |
| **Permission checks** | In-memory rule matching | Negligible | N/A | No | N/A |

## Key Insight: LLM Latency Dominates

The primary latency source is **LLM inference time** (1-30s per request depending on model). No amount of Python/Rust optimization will make an LLM respond faster.

### Optimization Strategy

1. **Reduce round trips** — Batch file reads, combine observations into single context
2. **Cache aggressively** — Workspace structure, model metadata, project discovery
3. **Use fast models** for simple tasks (explanation, classification)
4. **Parallel tool execution** where the architecture allows it

## Implemented Optimizations

### 1. Workspace Discovery Caching (ContextEngine)
- **Before**: Re-scans entire workspace on every `discover_project()` call
- **After**: Caches result for 30 seconds, skips `node_modules`, `__pycache__`, `.git`, etc.
- **Impact**: Eliminates redundant filesystem traversal for multi-iteration tasks

### 2. Autonomous Workspace Execution (PermissionManager)
- **Before**: Every `run_command` requires interactive approval
- **After**: Safe development commands auto-approved inside workspace
- **Impact**: Eliminates user interaction latency for normal development flows

### 3. Smart File Prioritization (ContextEngine)
- **Before**: Returns first 50 files alphabetically
- **After**: Prioritizes README, source files, config; skips deep nesting
- **Impact**: More relevant context in fewer tokens

### 4. Task Phase Events (AgentLoop)
- **Before**: Only `task.started` and `task.completed` events
- **After**: Phase transitions (`understanding` → `implementing` → `complete`)
- **Impact**: Better UX visibility, enables ETA estimation

## Future Rust/C++ Opportunities

If profiling shows these are actual bottlenecks (not LLM-bound):

### High Value
- **Filesystem watching** — `notify` crate for workspace change detection
- **Git operations** — `git2` crate for fast status/diff/log

### Medium Value
- **Process execution** — Faster subprocess management
- **File indexing** — `ignore` crate for fast workspace traversal

### Low Value (Python is fine)
- Context assembly (string operations)
- Permission checking (in-memory rules)
- Event dispatching (in-memory async)
- Tool schema generation

## Acceptance Targets

| Task | Target | Notes |
|---|---|---|
| Project explanation | <15s | Dominated by LLM latency |
| Simple file edit | <20s | 1-2 LLM round trips |
| Edit + test | <30s | 2-3 LLM round trips |
| Failure + recovery | <60s | Depends on failure complexity |

These are **targets**, not guarantees. LLM inference is the dominant cost.

## Profiling Commands

```bash
# Python profiling
python -m cProfile -o profile.stats -m harness_core.cli.main
snakeviz profile.stats

# Memory profiling
python -m memory_profiler harness_core/cli/main.py

# Timing specific operations
python -c "
import time
from harness_core.context.engine import ContextEngine
from pathlib import Path
e = ContextEngine(Path('.'))
t0 = time.time()
import asyncio
asyncio.run(e.discover_project())
print(f'Discovery: {time.time()-t0:.3f}s')
"
```
