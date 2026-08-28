"""
Context compaction for long-running agent sessions.

When context becomes too large, compaction preserves critical state
(current task, active plan, modified files, failures, unresolved errors)
while summarizing older observations and tool results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentMessage:
    """A single message in the agent conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: float = field(default_factory=time.monotonic)
    kind: str = 'normal'  # 'normal', 'tool_call', 'tool_result', 'plan', 'error'
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.tokens == 0:
            self.tokens = max(1, len(self.content) // 4)


@dataclass
class CompactSummary:
    """A summary of compacted older messages."""
    content: str
    preserved_messages: list[AgentMessage]
    original_count: int
    compacted_count: int
    tokens_saved: int


class ContextCompactor:
    """Compacts agent conversation context when it grows too large.

    Strategy:
        1. Always preserve: system prompt, current task, active plan
        2. Always preserve: modified files, unresolved errors, failed tests
        3. Summarize: older tool results and observations
        4. Keep recent N messages verbatim

    The compaction is deterministic — given the same messages and
    parameters, it produces the same result.
    """

    def __init__(
        self,
        max_tokens: int = 20_000,
        preserve_recent: int = 10,
        summary_max_tokens: int = 2_000,
    ) -> None:
        self._max_tokens = max_tokens
        self._preserve_recent = preserve_recent
        self._summary_max_tokens = summary_max_tokens

    def should_compact(self, messages: list[AgentMessage]) -> bool:
        """Check if compaction is needed."""
        total = sum(m.tokens for m in messages)
        return total > self._max_tokens

    def compact(self, messages: list[AgentMessage]) -> CompactSummary:
        """Compact messages to fit within the token budget.

        Returns a CompactSummary containing:
        - A summary of the compacted older messages
        - The preserved recent messages
        """
        if not messages:
            return CompactSummary(
                content='', preserved_messages=[],
                original_count=0, compacted_count=0, tokens_saved=0,
            )

        original_count = len(messages)
        original_tokens = sum(m.tokens for m in messages)

        # Categorize messages
        system_msgs = [m for m in messages if m.role == 'system']
        task_msgs = [m for m in messages if m.kind in ('plan',)]
        error_msgs = [m for m in messages if m.kind == 'error']
        modified_files = self._extract_modified_files(messages)
        recent = messages[-self._preserve_recent:] if len(messages) > self._preserve_recent else []
        older = messages[:-self._preserve_recent] if len(messages) > self._preserve_recent else []

        # Build summary of older messages
        summary_parts = []

        if older:
            summary_parts.append(f'## Summary of {len(older)} earlier messages')

            # Count tool calls
            tool_calls = [m for m in older if m.kind == 'tool_call']
            tool_results = [m for m in older if m.kind == 'tool_result']
            if tool_calls:
                tools_used = set()
                for tc in tool_calls:
                    if tc.metadata.get('tool_name'):
                        tools_used.add(tc.metadata['tool_name'])
                summary_parts.append(
                    f'Tool calls made: {len(tool_calls)} '
                    f'(tools: {", ".join(sorted(tools_used)) if tools_used else "unknown"})'
                )

            # Count errors
            errors_in_older = [m for m in older if m.kind == 'error']
            if errors_in_older:
                error_msgs_text = [m.content[:100] for m in errors_in_older[-3:]]
                summary_parts.append(f'Errors encountered: {len(errors_in_older)}')
                for em in error_msgs_text:
                    summary_parts.append(f'  - {em}')

            # Key decisions from plans
            plans_in_older = [m for m in older if m.kind == 'plan']
            if plans_in_older:
                summary_parts.append(f'Plans made: {len(plans_in_older)}')

        # Build the preserved context
        preserved: list[AgentMessage] = []
        preserved.extend(system_msgs)

        # Add task/plan messages if not already in recent
        for msg in task_msgs:
            if msg not in recent:
                preserved.append(msg)

        # Add error messages if not already in recent
        for msg in error_msgs[-5:]:  # keep last 5 errors
            if msg not in recent:
                preserved.append(msg)

        # Add modified files summary
        if modified_files:
            preserved.append(AgentMessage(
                role='system',
                content=f'Files modified so far: {", ".join(modified_files)}',
                kind='plan',
            ))

        # Add recent messages
        preserved.extend(recent)

        summary_text = '\n'.join(summary_parts) if summary_parts else 'No significant earlier activity.'
        compacted_count = len(older)
        new_tokens = sum(m.tokens for m in preserved)
        tokens_saved = original_tokens - new_tokens

        return CompactSummary(
            content=summary_text,
            preserved_messages=preserved,
            original_count=original_count,
            compacted_count=compacted_count,
            tokens_saved=tokens_saved,
        )

    def _extract_modified_files(self, messages: list[AgentMessage]) -> list[str]:
        """Extract file paths that were modified (via write/edit tool calls)."""
        modified = set()
        for msg in messages:
            if msg.kind == 'tool_call' and msg.metadata.get('tool_name') in ('write_file', 'edit_file'):
                path = msg.metadata.get('path', '')
                if path:
                    modified.add(path)
        return sorted(modified)

    def build_compacted_messages(
        self, messages: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Return compacted messages if needed, otherwise original messages."""
        if not self.should_compact(messages):
            return messages

        summary = self.compact(messages)

        # Rebuild: summary + preserved
        result = [
            AgentMessage(
                role='system',
                content=summary.content,
                kind='plan',
            ),
        ]
        result.extend(summary.preserved_messages)
        return result
