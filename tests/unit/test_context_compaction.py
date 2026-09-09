"""Tests for ContextCompactor — the summarizer behind the TASK STATE block.

As of Phase 2B this is live code: AgentLoop._compact_task_state builds
AgentMessages from the tool calls it drops and asks ContextCompactor to
roll them up. These tests moved here from test_context_pack.py when the
ContextPackBuilder tests were removed for covering unreachable code.
"""

from __future__ import annotations

from harness_core.context.compaction import AgentMessage, ContextCompactor


def _make_messages(count: int) -> list[AgentMessage]:
    msgs = []
    for i in range(count):
        msgs.append(AgentMessage(
            role='user' if i % 2 == 0 else 'assistant',
            content=f'Message {i}: ' + 'x' * 200,
            kind='normal',
        ))
    return msgs


class TestContextCompactor:
    def _make_messages(self, count: int) -> list[AgentMessage]:
        return _make_messages(count)

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
        return _make_messages(count)

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


class TestPreserveRecentZero:
    """`preserve_recent=0` means summarize everything, preserve nothing.

    This is the mode AgentLoop uses: every call it decided to drop belongs in
    the summary, none are replayed verbatim. It needs its own coverage because
    `messages[-0:]` is the whole list, not the empty list, so the naive slice
    silently preserved everything and produced no summary at all.
    """

    def test_summarizes_every_message(self):
        compactor = ContextCompactor(preserve_recent=0)
        msgs = _make_messages(6)
        summary = compactor.compact(msgs)
        assert summary.compacted_count == 6
        assert 'Summary of 6 earlier messages' in summary.content

    def test_preserves_no_normal_messages(self):
        compactor = ContextCompactor(preserve_recent=0)
        msgs = _make_messages(6)
        summary = compactor.compact(msgs)
        preserved = [m.content for m in summary.preserved_messages]
        assert not any(m.content in preserved for m in msgs)

    def test_still_reports_tools_used(self):
        compactor = ContextCompactor(preserve_recent=0)
        msgs = [
            AgentMessage(role='assistant', content='read', kind='tool_call',
                         metadata={'tool_name': 'read_file'}),
            AgentMessage(role='assistant', content='ran', kind='tool_call',
                         metadata={'tool_name': 'run_command'}),
        ]
        summary = compactor.compact(msgs)
        assert 'Tool calls made: 2' in summary.content
        assert 'read_file' in summary.content
        assert 'run_command' in summary.content

    def test_shorter_than_preserve_recent_keeps_messages(self):
        """Fewer messages than the window: nothing is old enough to summarize."""
        compactor = ContextCompactor(preserve_recent=10)
        msgs = _make_messages(3)
        summary = compactor.compact(msgs)
        assert summary.compacted_count == 0
        preserved = [m.content for m in summary.preserved_messages]
        for m in msgs:
            assert m.content in preserved
