"""Tests for parallel tool execution and deduplication (Phase E).

Covers: ParallelToolExecutor (concurrent reads, serial writes),
ToolCallDeduplicator (cache hits, eviction).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema
from harness_core.tools.parallel import (
    PARALLEL_SAFE_OPS,
    SERIAL_OPS,
    ParallelToolExecutor,
    ToolCallDeduplicator,
)


# ── Mock Tools ──────────────────────────────────────────────────────────────

class MockReadTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name='read_file', description='Read a file',
            parameters={'path': {'type': 'string'}},
        )

    async def execute(self, args: dict) -> ToolResult:
        path = args.get('path', 'unknown')
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=f'content of {path}',
        )


class MockWriteTool(Tool):
    def __init__(self) -> None:
        self.writes: list[str] = []

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name='write_file', description='Write a file',
            parameters={'path': {'type': 'string'}, 'content': {'type': 'string'}},
        )

    async def execute(self, args: dict) -> ToolResult:
        path = args.get('path', 'unknown')
        self.writes.append(path)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=f'wrote {path}',
        )


class MockGrepTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name='grep', description='Search files',
            parameters={'pattern': {'type': 'string'}},
        )

    async def execute(self, args: dict) -> ToolResult:
        pattern = args.get('pattern', '')
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=f'matches for {pattern}',
        )


class MockSlowReadTool(Tool):
    """Read tool that simulates latency."""
    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name='read_file', description='Slow read',
            parameters={'path': {'type': 'string'}},
        )

    async def execute(self, args: dict) -> ToolResult:
        await asyncio.sleep(self._delay)
        path = args.get('path', 'unknown')
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=f'content of {path}',
        )


class FailingTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name='fail_tool', description='Always fails', parameters={},
        )

    async def execute(self, args: dict) -> ToolResult:
        raise RuntimeError('intentional failure')


def _ok(output: str = '') -> ToolResult:
    return ToolResult(status=ToolResultStatus.SUCCESS, output=output)


def _ok_tool(name: str, output: str) -> ToolResult:
    return ToolResult(status=ToolResultStatus.SUCCESS, output=output)


# ── ToolCallDeduplicator Tests ──────────────────────────────────────────────

class TestToolCallDeduplicator:
    def test_cache_miss(self):
        dedup = ToolCallDeduplicator()
        result = dedup.get_cached('read_file', {'path': 'a.py'})
        assert result is None

    def test_record_and_hit(self):
        dedup = ToolCallDeduplicator()
        tr = _ok('hello')
        dedup.record('read_file', {'path': 'a.py'}, tr)
        cached = dedup.get_cached('read_file', {'path': 'a.py'})
        assert cached is not None
        assert cached.output == 'hello'

    def test_different_args_different_entry(self):
        dedup = ToolCallDeduplicator()
        dedup.record('read_file', {'path': 'a.py'}, _ok('a'))
        result = dedup.get_cached('read_file', {'path': 'b.py'})
        assert result is None

    def test_eviction(self):
        dedup = ToolCallDeduplicator(max_cache_size=3)
        for i in range(5):
            dedup.record('read_file', {'path': f'f{i}.py'}, _ok(f'{i}'))
        assert len(dedup._cache) == 3

    def test_stats(self):
        dedup = ToolCallDeduplicator()
        dedup.record('read_file', {'path': 'a.py'}, _ok('a'))
        dedup.get_cached('read_file', {'path': 'a.py'})  # hit
        dedup.get_cached('read_file', {'path': 'b.py'})  # miss
        stats = dedup.stats
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5

    def test_clear(self):
        dedup = ToolCallDeduplicator()
        dedup.record('read_file', {'path': 'a.py'}, _ok('a'))
        dedup.clear()
        assert dedup.get_cached('read_file', {'path': 'a.py'}) is None


# ── ParallelToolExecutor Tests ──────────────────────────────────────────────

class TestParallelToolExecutor:
    @pytest.fixture
    def tools(self):
        return {
            'read_file': MockReadTool(),
            'write_file': MockWriteTool(),
            'grep': MockGrepTool(),
        }

    @pytest.fixture
    def executor(self, tools):
        return ParallelToolExecutor(tools=tools)

    @pytest.mark.asyncio
    async def test_execute_single(self, executor):
        result = await executor.execute_single('read_file', {'path': 'test.py'})
        assert result.status == ToolResultStatus.SUCCESS
        assert 'test.py' in result.output

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor):
        result = await executor.execute_single('nope', {})
        assert result.status == ToolResultStatus.ERROR
        assert 'Unknown tool' in result.error

    @pytest.mark.asyncio
    async def test_permission_deny(self, executor):
        def deny_all(tool_name, args):
            return False

        result = await executor.execute_single(
            'read_file', {'path': 'a.py'}, permission_check=deny_all
        )
        assert result.status == ToolResultStatus.PERMISSION_DENIED
        assert 'Permission denied' in result.error

    @pytest.mark.asyncio
    async def test_dedup_hit(self, executor):
        executor.deduplicator.record('read_file', {'path': 'a.py'}, _ok('cached'))
        result = await executor.execute_single('read_file', {'path': 'a.py'})
        assert result.output == 'cached'

    @pytest.mark.asyncio
    async def test_execute_batch_parallel_reads(self, executor):
        calls = [
            {'tool': 'read_file', 'args': {'path': f'file{i}.py'}}
            for i in range(5)
        ]
        results = await executor.execute_batch(calls)
        assert len(results) == 5
        assert all(r.status == ToolResultStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_execute_batch_mixed(self, executor):
        calls = [
            {'tool': 'read_file', 'args': {'path': 'a.py'}},
            {'tool': 'write_file', 'args': {'path': 'b.py', 'content': 'x'}},
            {'tool': 'read_file', 'args': {'path': 'c.py'}},
            {'tool': 'grep', 'args': {'pattern': 'foo'}},
        ]
        results = await executor.execute_batch(calls)
        assert len(results) == 4
        assert all(r.status == ToolResultStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, executor):
        calls = [
            {'tool': 'read_file', 'args': {'path': f'f{i}.py'}}
            for i in range(10)
        ]
        results = await executor.execute_batch(calls)
        assert len(results) == 10
        for i, r in enumerate(results):
            assert f'f{i}.py' in r.output

    @pytest.mark.asyncio
    async def test_empty_batch(self, executor):
        results = await executor.execute_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_concurrent_reads_faster_than_sequential(self):
        """Parallel reads should be faster than sequential."""
        tool = MockSlowReadTool(delay=0.05)
        tools = {'read_file': tool}
        executor = ParallelToolExecutor(tools=tools, max_concurrency=10)

        calls = [
            {'tool': 'read_file', 'args': {'path': f'f{i}.py'}}
            for i in range(5)
        ]

        start = time.monotonic()
        results = await executor.execute_batch(calls)
        parallel_time = time.monotonic() - start

        assert len(results) == 5
        # 5 reads x 0.05s = 0.25s sequential; parallel should be < 0.15s
        assert parallel_time < 0.2


class TestParallelToolExecutorFailingTool:
    @pytest.mark.asyncio
    async def test_exception_handling(self):
        tools = {'fail_tool': FailingTool()}
        executor = ParallelToolExecutor(tools=tools)
        result = await executor.execute_single('fail_tool', {})
        assert result.status == ToolResultStatus.ERROR
        assert 'intentional failure' in result.error


class TestToolClassification:
    def test_parallel_safe_ops(self):
        assert 'read_file' in PARALLEL_SAFE_OPS
        assert 'grep' in PARALLEL_SAFE_OPS
        assert 'glob' in PARALLEL_SAFE_OPS

    def test_serial_ops(self):
        assert 'write_file' in SERIAL_OPS
        assert 'edit_file' in SERIAL_OPS
        assert 'run_command' in SERIAL_OPS

    def test_no_overlap(self):
        assert PARALLEL_SAFE_OPS.isdisjoint(SERIAL_OPS)
