"""
Parallel tool execution and tool call deduplication.

Allows independent read-only operations to run concurrently while
serializing mutating operations. Prevents duplicate tool calls when
results are still valid.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema


# Operations that are safe to parallelize (read-only)
PARALLEL_SAFE_OPS = frozenset({
    'read_file', 'list_files', 'glob', 'grep', 'search_code',
    'git_status', 'git_diff', 'git_log',
})

# Operations that must be serialized (mutating or side-effecting)
SERIAL_OPS = frozenset({
    'write_file', 'edit_file', 'run_command', 'run_tests',
    'git_commit', 'git_branch',
})


@dataclass
class ToolCallEntry:
    """A recorded tool call for deduplication tracking."""
    tool_name: str
    args: dict[str, Any]
    result: ToolResult | None = None
    timestamp: float = 0.0
    duration_ms: float = 0.0
    call_hash: str = ''

    def __post_init__(self) -> None:
        if self.timestamp == 0:
            self.timestamp = time.monotonic()
        if not self.call_hash:
            self.call_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a deterministic hash of the tool call."""
        key = f'{self.tool_name}:{sorted(self.args.items())}'
        return hashlib.sha256(key.encode()).hexdigest()[:12]


class ToolCallDeduplicator:
    """Tracks tool calls and detects duplicates.

    If the same tool is called with the same arguments and the result
    is still valid, the cached result is returned instead of re-executing.
    """

    def __init__(self, max_cache_size: int = 500) -> None:
        self._cache: dict[str, ToolCallEntry] = {}
        self._max_cache_size = max_cache_size
        self._hits = 0
        self._misses = 0

    def get_cached(
        self, tool_name: str, args: dict[str, Any]
    ) -> Optional[ToolResult]:
        """Return a cached result if this exact call was made before."""
        entry = self._make_entry(tool_name, args)
        cached = self._cache.get(entry.call_hash)
        if cached and cached.result is not None:
            self._hits += 1
            return cached.result
        self._misses += 1
        return None

    def record(
        self, tool_name: str, args: dict[str, Any], result: ToolResult,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a tool call result for future deduplication."""
        entry = self._make_entry(tool_name, args)
        entry.result = result
        entry.duration_ms = duration_ms

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_cache_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp,
            )
            del self._cache[oldest_key]

        self._cache[entry.call_hash] = entry

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'cached_calls': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total else 0.0,
        }

    def _make_entry(self, tool_name: str, args: dict[str, Any]) -> ToolCallEntry:
        return ToolCallEntry(tool_name=tool_name, args=args)


class ParallelToolExecutor:
    """Executes tool calls with concurrency for read-only operations.

    Independent read-only operations (read_file, glob, grep, etc.) run
    concurrently. Mutating operations (write, edit, shell) run serially.

    Permission checks happen before execution. Workspace sandbox rules
    remain enforced.
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        max_concurrency: int = 10,
        deduplicator: ToolCallDeduplicator | None = None,
    ) -> None:
        self._tools = tools
        self._max_concurrency = max_concurrency
        self._deduplicator = deduplicator or ToolCallDeduplicator()

    @property
    def deduplicator(self) -> ToolCallDeduplicator:
        return self._deduplicator

    async def execute_single(
        self,
        tool_name: str,
        args: dict[str, Any],
        permission_check: Callable[[str, dict], bool] | None = None,
    ) -> ToolResult:
        """Execute a single tool call.

        Checks deduplication cache first. Runs permission check if provided.
        """
        # Check deduplication
        cached = self._deduplicator.get_cached(tool_name, args)
        if cached is not None:
            return cached

        # Permission check
        if permission_check and not permission_check(tool_name, args):
            return ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output='',
                error='Permission denied',
            )

        # Execute
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output='',
                error=f'Unknown tool: {tool_name}',
            )

        start = time.monotonic()
        try:
            result = await tool.execute(args)
        except Exception as e:
            result = ToolResult(
                status=ToolResultStatus.ERROR,
                output='',
                error=str(e),
            )
        duration_ms = (time.monotonic() - start) * 1000

        # Record for deduplication
        self._deduplicator.record(tool_name, args, result, duration_ms)
        return result

    async def execute_batch(
        self,
        calls: list[dict[str, Any]],
        permission_check: Callable[[str, dict], bool] | None = None,
    ) -> list[ToolResult]:
        """Execute a batch of tool calls with intelligent parallelism.

        Each call dict must have 'tool' and 'args' keys.

        Read-only operations are parallelized. Mutating operations are
        serialized in order.

        Returns results in the same order as input calls.
        """
        if not calls:
            return []

        # Partition into parallel-safe and serial
        parallel_batch: list[tuple[int, str, dict]] = []
        serial_batch: list[tuple[int, str, dict]] = []

        for i, call in enumerate(calls):
            tool_name = call.get('tool', '')
            args = call.get('args', {})
            if tool_name in PARALLEL_SAFE_OPS:
                parallel_batch.append((i, tool_name, args))
            else:
                serial_batch.append((i, tool_name, args))

        # Initialize results list
        results: list[ToolResult | None] = [None] * len(calls)

        # Execute parallel-safe operations concurrently
        if parallel_batch:
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def _run_parallel(
                idx: int, name: str, a: dict[str, Any]
            ) -> tuple[int, ToolResult]:
                async with semaphore:
                    result = await self.execute_single(name, a, permission_check)
                    return idx, result

            tasks = [
                _run_parallel(idx, name, args)
                for idx, name, args in parallel_batch
            ]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for item in completed:
                if isinstance(item, Exception):
                    continue
                idx, result = item
                results[idx] = result

        # Execute serial operations in order
        for idx, name, args in serial_batch:
            results[idx] = await self.execute_single(name, args, permission_check)

        # Cast away None (all slots should be filled)
        return [r for r in results if r is not None]

    @property
    def stats(self) -> dict:
        return {
            'dedup': self._deduplicator.stats,
            'tools_available': len(self._tools),
        }
