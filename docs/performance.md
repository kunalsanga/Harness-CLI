# Harness Engineering CLI — Performance

## Overview

Milestone 3 introduced comprehensive performance infrastructure: caching, indexing, context intelligence, parallel execution, and metrics collection. This document covers architecture, measurements, and benchmark methodology.

## Performance Goals

```
Task success (minimize)
  ↓
Model calls (minimize)
  ↓
Tool calls (minimize)
  ↓
Tokens consumed (minimize)
  ↓
Latency (minimize)
```

## Caching Architecture

### FileContentCache
- **Type**: LRU with metadata tracking
- **Size limit**: Configurable (default 1000 entries, 100MB)
- **Invalidation**: Automatic via mtime/hash detection
- **Secret detection**: Skips .env, credentials, SSH keys, tokens
- **Thread-safe**: Uses threading.Lock

### SearchCache
- **Type**: Key-based with TTL eviction
- **Size limit**: Configurable (default 500 entries)
- **Key format**: `prefix:serialized_params` (deterministic)
- **Invalidation**: Manual + stale entry detection

### Cache Hit Rates

```
Operation              Cold (ms)    Warm (ms)    Speedup
─────────────────────────────────────────────────────────
20 file reads          ~15          ~0.5         ~30x
1000 search lookups    N/A          ~0.01ms      instant
```

## Repository Intelligence

### RepositoryAnalyzer
- **Ecosystem detection**: Python, Node.js, TypeScript, Rust, Go
- **Indicators**: Config files, source directories, test directories
- **Performance**: Analyzes harness-engineering-cli in <1s

### RelevanceRanker
- **Signals**: filename, path, extension, search, importance, test proximity
- **Configurable weights**: Adjust per-task type
- **Performance**: 1000 scoring iterations in <2s

### SymbolIndex
- **Languages**: Python, JavaScript/TypeScript, Rust, Go
- **Entities**: functions, classes, methods, imports
- **Parser**: Regex-based (not full AST — trades completeness for speed)
- **Build time**: 50 files in ~115ms
- **Lookup time**: <0.001ms per query

### DependencyGraph
- **Cycle detection**: DFS-based, O(V+E)
- **Transitive reachability**: With max depth
- **Thread-safe**: Uses threading.Lock

## Context Assembly

### ContextPackBuilder
- **Token budget**: Configurable (default 20K tokens)
- **Priority ordering**: Task > Instructions > Files > Symbols > Search > Summary
- **Deduplication**: Content-hash based, prevents duplicate files
- **Truncation**: Files exceeding max_tokens truncated with metadata

### Assembly Performance

```
Operation                     Time
────────────────────────────────────
Build 100 files (200 builds)  2.6ms total, 0.013ms/build
Deduplication                 O(n) per build
Token estimation              ~4 chars/token (fast heuristic)
```

## Parallel Execution

### ParallelToolExecutor
- **Concurrency**: Configurable (default 10)
- **Safety**: Read operations concurrent, mutating serialized
- **Deduplication**: ToolCallDeduplicator filters identical calls
- **Ordering**: Deterministic result ordering preserved

## Metrics Instrumentation

### MetricsCollector
- **Thread-safe**: All operations use threading.Lock
- **Types**: Counters, gauges, timers, summary statistics
- **Percentiles**: p50, p95, p99
- **Dashboard**: Real-time performance summary

### Overhead

```
Operation                    Time
────────────────────────────────────
10,000 metric operations     5.0ms (0.0005ms/op)
1000 dashboard generations   3.4ms (0.003ms/op)
```

## Benchmark Suite

Run benchmarks:
```bash
cd harness-engineering-cli
uv run pytest tests/benchmarks/ -v -s
```

### Benchmark Categories

| Benchmark | What it measures |
|-----------|-----------------|
| Cold vs warm file reads | Cache hit rate and speedup |
| Search cache hit rate | Lookup performance |
| Repo analysis | RepositoryAnalyzer speed |
| Relevance scoring | RelevanceRanker throughput |
| Symbol index build | Indexing speed and completeness |
| Symbol lookup | Query performance |
| Context pack assembly | ContextPackBuilder throughput |
| Deduplication | Duplicate content elimination |
| Sequential reads | Baseline file I/O |
| Metrics overhead | Instrumentation cost |

## Known Limitations

1. **Regex-based symbol parsing** — Not a full AST parser. Misses some edge cases (decorated methods, nested classes). Sufficient for navigation; not for semantic analysis.

2. **Token estimation** — ~4 chars/token heuristic. Not exact. Real tokenization varies by model tokenizer.

3. **Cache invalidation** — Based on mtime. Doesn't detect content changes without mtime updates (e.g., `touch` without modification).

4. **Parallel execution** — Currently uses asyncio for concurrent reads. Mutating operations serialized. True multi-process parallelism not yet implemented.

5. **No Rust/C++ acceleration yet** — All Python. Future: native indexing, sandbox, filesystem runtime based on profiling data.

## Future Optimizations

Based on profiling, these components could benefit from native implementations:

| Component | Python Bottleneck | Potential Rust/C++ Win |
|-----------|------------------|----------------------|
| Symbol indexing | Regex at scale | Tree-sitter parser |
| Dependency graph | Graph traversal | Native graph library |
| File content cache | GIL-limited | Lock-free concurrent cache |
| Search (grep/glob) | Python regex | ripgrep-like native search |
| Process sandbox | Subprocess overhead | Native process isolation |
