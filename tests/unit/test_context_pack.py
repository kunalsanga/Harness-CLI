"""Tests for context pack and compaction (Phase D).

Covers: ContextPackBuilder (dedup, budget, priority), ContextCompactor.
"""

from __future__ import annotations

import pytest

from harness_core.context.compaction import AgentMessage, ContextCompactor
from harness_core.context.pack import (
    ContextPack,
    ContextPackBuilder,
    ContextPiece,
    estimate_tokens,
)


# ── Token estimation ────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_empty(self):
        assert estimate_tokens('') == 1

    def test_short(self):
        assert estimate_tokens('hello') == 1  # 5 chars / 4 = 1

    def test_longer(self):
        assert estimate_tokens('a' * 100) == 25

    def test_newlines(self):
        assert estimate_tokens('hello\nworld\n') == 3  # 12 / 4 = 3


# ── ContextPiece ────────────────────────────────────────────────────────────

class TestContextPiece:
    def test_auto_token_estimate(self):
        piece = ContextPiece(kind='file', content='a' * 40, path='test.py')
        assert piece.tokens_estimate == 10

    def test_content_hash(self):
        p1 = ContextPiece(kind='file', content='hello')
        p2 = ContextPiece(kind='file', content='hello')
        p3 = ContextPiece(kind='file', content='world')
        assert p1.content_hash == p2.content_hash
        assert p1.content_hash != p3.content_hash


# ── ContextPackBuilder ──────────────────────────────────────────────────────

class TestContextPackBuilderBasic:
    def test_add_task(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_task('Fix the bug')
        pack = builder.build()
        assert len(pack.pieces) == 1
        assert pack.pieces[0].kind == 'task'

    def test_add_instruction(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_instruction('Use pytest for tests')
        pack = builder.build()
        assert len(pack.pieces) == 1
        assert pack.pieces[0].kind == 'instruction'

    def test_add_file(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_file('src/main.py', 'def hello(): pass')
        pack = builder.build()
        assert len(pack.pieces) == 1
        assert pack.pieces[0].path == 'src/main.py'


class TestContextPackBuilderDedup:
    def test_dedup_identical_content(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_file('a.py', 'same content')
        builder.add_file('b.py', 'same content')
        pack = builder.build()
        # Only one should be included (deduplicated)
        assert len(pack.pieces) == 1

    def test_dedup_different_content(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_file('a.py', 'content A')
        builder.add_file('b.py', 'content B')
        pack = builder.build()
        assert len(pack.pieces) == 2

    def test_dedup_across_kinds(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_instruction('same text')
        builder.add_summary('same text')
        pack = builder.build()
        # Deduplicated despite different kinds
        assert len(pack.pieces) == 1


class TestContextPackBuilderBudget:
    def test_respects_budget(self):
        builder = ContextPackBuilder(
            token_budget=100,
            system_prompt_tokens=0,
            output_reserve=0,
        )
        builder.add_task('Fix this')
        # Add files that exceed budget
        for i in range(20):
            builder.add_file(f'f{i}.py', 'x' * 400, priority=50 - i)
        pack = builder.build()
        assert pack.total_tokens <= 100

    def test_priority_ordering(self):
        builder = ContextPackBuilder(token_budget=200, system_prompt_tokens=0, output_reserve=0)
        builder.add_task('Fix bug')  # priority 100
        builder.add_file('low.py', 'low priority', priority=10)
        builder.add_file('high.py', 'high priority', priority=90)
        pack = builder.build()
        # Task (100) and high (90) should be included; low (10) might be dropped
        kinds = [p.kind for p in pack.pieces]
        assert 'task' in kinds
        assert 'file' in kinds

    def test_truncation(self):
        builder = ContextPackBuilder(token_budget=100_000)
        big_content = 'x' * 20_000  # ~5000 tokens
        builder.add_file('big.py', big_content, max_tokens=10)
        pack = builder.build()
        assert pack.pieces[0].metadata.get('truncated') is True


class TestContextPackBuilderSearch:
    def test_search_results(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_search_results(['file1.py:10', 'file2.py:20'], query='auth')
        pack = builder.build()
        assert len(pack.pieces) == 1
        assert 'auth' in pack.pieces[0].content


class TestContextPackOutput:
    def test_to_text(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_task('Fix tests')
        builder.add_file('main.py', 'print("hello")')
        pack = builder.build()
        text = pack.to_text()
        assert 'TASK:' in text
        assert 'FILE [main.py]' in text

    def test_to_prompt_parts(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_task('Fix tests')
        builder.add_instruction('Use pytest')
        pack = builder.build()
        parts = pack.to_prompt_parts()
        assert len(parts) == 2
        assert parts[0]['role'] == 'user'


class TestContextPackBuilderReset:
    def test_reset(self):
        builder = ContextPackBuilder(token_budget=10_000)
        builder.add_task('task 1')
        builder.add_file('a.py', 'content')
        builder.reset()
        builder.add_task('task 2')
        pack = builder.build()
        assert len(pack.pieces) == 1
        assert 'task 2' in pack.pieces[0].content


# ── ContextCompactor ────────────────────────────────────────────────────────

class TestContextCompactor:
    def _make_messages(self, count: int) -> list[AgentMessage]:
        msgs = []
        for i in range(count):
            msgs.append(AgentMessage(
                role='user' if i % 2 == 0 else 'assistant',
                content=f'Message {i}: ' + 'x' * 200,
                kind='normal',
            ))
        return msgs

    def test_no_compaction_when_small(self):
        compactor = ContextCompactor(max_tokens=100_000)
        msgs = self._make_messages(5)
        result = compactor.build_compacted_messages(msgs)
        assert len(result) == 5

    def test_compaction_reduces_messages(self):
        compactor = ContextCompactor(max_tokens=500, preserve_recent=3)
        msgs = self._make_messages(30)
        result = compactor.build_compacted_messages(msgs)
        assert len(result) < len(msgs)

    def test_preserves_recent(self):
        compactor = ContextCompactor(max_tokens=500, preserve_recent=5)
        msgs = self._make_messages(20)
        result = compactor.build_compacted_messages(msgs)
        # Recent messages should be preserved
        recent_contents = [m.content for m in msgs[-5:]]
        result_contents = [m.content for m in result]
        for rc in recent_contents:
            assert rc in result_contents

    def test_preserves_errors(self):
        compactor = ContextCompactor(max_tokens=500, preserve_recent=3)
        msgs = self._make_messages(20)
        msgs[10] = AgentMessage(role='assistant', content='ERROR: test failed', kind='error')
        result = compactor.build_compacted_messages(msgs)
        result_contents = [m.content for m in result]
        assert any('ERROR' in c for c in result_contents)

    def test_preserves_modified_files(self):
        compactor = ContextCompactor(max_tokens=500, preserve_recent=3)
        msgs = self._make_messages(20)
        msgs[5] = AgentMessage(
            role='assistant', content='wrote file', kind='tool_call',
            metadata={'tool_name': 'write_file', 'path': 'main.py'},
        )
        result = compactor.build_compacted_messages(msgs)
        result_contents = [m.content for m in result]
        assert any('main.py' in c for c in result_contents)

    def test_compaction_summary(self):
        compactor = ContextCompactor(max_tokens=500, preserve_recent=3)
        msgs = self._make_messages(20)
        summary = compactor.compact(msgs)
        assert summary.compacted_count > 0
        assert summary.tokens_saved > 0
        assert 'earlier messages' in summary.content


class TestContextCompactorToolCalls:
    def _make_messages(self, count: int) -> list[AgentMessage]:
        msgs = []
        for i in range(count):
            msgs.append(AgentMessage(
                role='user' if i % 2 == 0 else 'assistant',
                content=f'Message {i}: ' + 'x' * 200,
                kind='normal',
            ))
        return msgs

    def test_counts_tool_calls(self):
        compactor = ContextCompactor(max_tokens=500, preserve_recent=3)
        msgs = self._make_messages(15)
        msgs[2] = AgentMessage(
            role='assistant', content='calling tool', kind='tool_call',
            metadata={'tool_name': 'read_file'},
        )
        msgs[4] = AgentMessage(
            role='assistant', content='calling tool', kind='tool_call',
            metadata={'tool_name': 'run_command'},
        )
        summary = compactor.compact(msgs)
        assert 'Tool calls made: 2' in summary.content
