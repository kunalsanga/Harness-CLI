"""
Context pack builder — assembles the smallest useful context for a task.

Builds a ContextPack containing task description, repository summary,
relevant files, symbols, search results, and project instructions — all
within a hard token budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def estimate_tokens(text: str) -> int:
    """Fast token count estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


@dataclass
class ContextPiece:
    """A single piece of context to include in a pack."""
    kind: str  # 'task', 'file', 'symbol', 'search', 'instruction', 'summary'
    content: str
    path: str = ''
    priority: float = 0.0  # higher = more important
    tokens_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens_estimate == 0:
            self.tokens_estimate = estimate_tokens(self.content)

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


@dataclass
class ContextPack:
    """An assembled context ready to send to a model.

    Contains pieces sorted by priority within budget, plus metadata
    about what was included/excluded and token usage.
    """
    pieces: list[ContextPiece]
    total_tokens: int = 0
    budget: int = 0
    dropped_count: int = 0
    deduplication_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_parts(self) -> list[dict[str, str]]:
        """Convert to prompt message parts."""
        parts = []
        for piece in self.pieces:
            if piece.kind == 'task':
                parts.append({'role': 'user', 'content': piece.content})
            elif piece.kind == 'file':
                parts.append({
                    'role': 'user',
                    'content': f'File: {piece.path}\n```\n{piece.content}\n```',
                })
            elif piece.kind == 'instruction':
                parts.append({
                    'role': 'system',
                    'content': piece.content,
                })
            else:
                parts.append({'role': 'user', 'content': piece.content})
        return parts

    def to_text(self) -> str:
        """Render as a single text block for inspection."""
        lines = []
        for piece in self.pieces:
            if piece.kind == 'task':
                lines.append(f'TASK: {piece.content}')
            elif piece.kind == 'file':
                lines.append(f'FILE [{piece.path}] ({piece.tokens_estimate} tokens):')
                lines.append(piece.content[:500])
                if len(piece.content) > 500:
                    lines.append('  ... [truncated]')
            elif piece.kind == 'symbol':
                lines.append(f'SYMBOL [{piece.path}]: {piece.content[:200]}')
            elif piece.kind == 'instruction':
                lines.append(f'INSTRUCTION: {piece.content[:200]}')
            else:
                lines.append(f'{piece.kind.upper()}: {piece.content[:200]}')
        return '\n'.join(lines)


class ContextPackBuilder:
    """Builds ContextPacks within a token budget.

    Assembles context pieces, deduplicates, and selects the highest-value
    information that fits the budget.

    Priority order:
        1. Task description (always included)
        2. Project instructions (high priority)
        3. Task-relevant files (ranked by relevance)
        4. Relevant symbols
        5. Search results
        6. Repository summary
    """

    def __init__(
        self,
        token_budget: int = 20_000,
        system_prompt_tokens: int = 0,
        output_reserve: int = 4_000,
    ) -> None:
        self._token_budget = token_budget
        self._system_prompt_tokens = system_prompt_tokens
        self._output_reserve = output_reserve
        self._pieces: list[ContextPiece] = []
        self._seen_hashes: set[str] = set()

    @property
    def available_tokens(self) -> int:
        """Tokens available for context pieces."""
        return self._token_budget - self._system_prompt_tokens - self._output_reserve

    def add_task(self, task: str) -> None:
        """Add the task description (highest priority, always included)."""
        self._pieces.append(ContextPiece(
            kind='task',
            content=task,
            priority=100.0,
        ))

    def add_instruction(self, text: str, priority: float = 90.0) -> None:
        """Add project instructions."""
        self._add_deduplicated(ContextPiece(
            kind='instruction',
            content=text,
            priority=priority,
        ))

    def add_file(
        self,
        path: str,
        content: str,
        priority: float = 50.0,
        max_tokens: int = 4000,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Add a file's content. Returns False if it didn't fit.

        If content exceeds max_tokens, it is truncated.
        """
        if metadata is None:
            metadata = {}

        # Truncate if needed
        estimated = estimate_tokens(content)
        if estimated > max_tokens:
            # Rough truncation: keep first max_tokens worth
            chars_to_keep = max_tokens * 4
            content = content[:chars_to_keep] + f'\n... [truncated at {max_tokens} tokens]'
            metadata['truncated'] = True
            metadata['original_tokens'] = estimated

        piece = ContextPiece(
            kind='file',
            content=content,
            path=path,
            priority=priority,
            metadata=metadata,
        )
        return self._add_deduplicated(piece)

    def add_symbols(
        self,
        symbols: list[dict[str, Any]],
        priority: float = 40.0,
    ) -> None:
        """Add symbol summaries."""
        for sym in symbols:
            content = f"{sym.get('kind', '?')} {sym.get('name', '?')}"
            if sym.get('file_path'):
                content += f" in {sym['file_path']}"
            if sym.get('line_number'):
                content += f":{sym['line_number']}"
            self._add_deduplicated(ContextPiece(
                kind='symbol',
                content=content,
                path=sym.get('file_path', ''),
                priority=priority,
                metadata=sym,
            ))

    def add_search_results(
        self,
        results: list[str],
        query: str = '',
        priority: float = 30.0,
    ) -> None:
        """Add search results (e.g. grep matches, glob results)."""
        content = f'Search results'
        if query:
            content += f' for "{query}"'
        content += ':\n' + '\n'.join(results[:50])  # cap at 50 results
        if len(results) > 50:
            content += f'\n... and {len(results) - 50} more'

        self._add_deduplicated(ContextPiece(
            kind='search',
            content=content,
            priority=priority,
        ))

    def add_summary(self, text: str, priority: float = 20.0) -> None:
        """Add a repository summary."""
        self._add_deduplicated(ContextPiece(
            kind='summary',
            content=text,
            priority=priority,
        ))

    def build(self) -> ContextPack:
        """Build the final ContextPack, respecting the token budget.

        Pieces are sorted by priority (descending) and included until
        the budget is exhausted. Deduplication has already been applied.
        """
        # Sort by priority (highest first)
        sorted_pieces = sorted(self._pieces, key=lambda p: p.priority, reverse=True)

        included: list[ContextPiece] = []
        total_tokens = 0
        dropped = 0
        budget = self.available_tokens

        for piece in sorted_pieces:
            if total_tokens + piece.tokens_estimate <= budget:
                included.append(piece)
                total_tokens += piece.tokens_estimate
            else:
                dropped += 1

        return ContextPack(
            pieces=included,
            total_tokens=total_tokens,
            budget=budget,
            dropped_count=dropped,
            deduplication_count=len(self._seen_hashes) - len(self._pieces),
        )

    def reset(self) -> None:
        """Reset the builder for reuse."""
        self._pieces.clear()
        self._seen_hashes.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_deduplicated(self, piece: ContextPiece) -> bool:
        """Add a piece only if its content hasn't been seen before."""
        h = piece.content_hash
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)
        self._pieces.append(piece)
        return True
