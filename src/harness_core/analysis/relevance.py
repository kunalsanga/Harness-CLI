"""File relevance ranking for intelligent context selection.

Scores files based on task description, path similarity, file type,
search matches, and project structure to prioritize which files
to include in the model context.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RelevanceScore:
    """Relevance score for a single file."""

    path: str
    total_score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def rank(self) -> float:
        return self.total_score


@dataclass
class RelevanceConfig:
    """Configurable weights for relevance scoring."""

    filename_match: float = 0.25
    path_match: float = 0.20
    extension_match: float = 0.10
    search_match: float = 0.20
    importance: float = 0.10
    recency: float = 0.05
    test_proximity: float = 0.10


class RelevanceRanker:
    """Ranks files by relevance to a given task.

    Signals:
    - filename similarity to task keywords
    - path similarity to task keywords
    - extension/language match
    - search match presence
    - file importance (README, config, etc.)
    - test proximity (related test files)
    """

    def __init__(self, config: RelevanceConfig | None = None) -> None:
        self.config = config or RelevanceConfig()

    def _extract_keywords(self, task: str) -> list[str]:
        """Extract meaningful keywords from a task description."""
        # Simple keyword extraction: lowercase, split on whitespace/punctuation,
        # filter short/common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "have", "has", "had", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "it", "its", "not", "no", "if", "then", "so", "very", "just",
            "about", "up", "out", "all", "also", "than", "some", "any",
            "make", "fix", "add", "remove", "update", "change", "modify",
            "implement", "create", "write", "run", "check", "inspect",
        }
        words = []
        for w in task.lower().split():
            cleaned = "".join(c for c in w if c.isalnum())
            if len(cleaned) > 2 and cleaned not in stop_words:
                words.append(cleaned)
        return words

    def _score_filename(self, filepath: str, keywords: list[str]) -> float:
        """Score based on filename matching task keywords."""
        if not keywords:
            return 0.0
        name = Path(filepath).stem.lower()
        matches = sum(1 for kw in keywords if kw in name)
        return min(1.0, matches / max(1, len(keywords) * 0.5))

    def _score_path(self, filepath: str, keywords: list[str]) -> float:
        """Score based on directory path matching task keywords."""
        if not keywords:
            return 0.0
        parts = Path(filepath).parts
        path_str = "/".join(parts).lower()
        matches = sum(1 for kw in keywords if kw in path_str)
        return min(1.0, matches / max(1, len(keywords) * 0.4))

    def _score_extension(self, filepath: str, task: str) -> float:
        """Score based on file extension relevance to task type."""
        ext = Path(filepath).suffix.lower()
        task_lower = task.lower()

        # Coding tasks prefer source files
        coding_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java"}
        test_exts = {".test.py", ".test.js", ".test.ts", ".spec.py", ".spec.js"}
        config_exts = {".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}

        is_coding_task = any(kw in task_lower for kw in [
            "code", "fix", "bug", "implement", "refactor", "function", "class",
            "method", "module", "error", "crash", "test",
        ])
        is_config_task = any(kw in task_lower for kw in [
            "config", "setup", "install", "dependency", "package",
        ])

        if ext in test_exts or "test" in filepath.lower():
            return 0.8 if "test" in task_lower else 0.3
        if ext in coding_exts:
            return 0.7 if is_coding_task else 0.4
        if ext in config_exts:
            return 0.6 if is_config_task else 0.2
        if ext in {".md", ".rst", ".txt"}:
            return 0.3 if "doc" in task_lower or "readme" in task_lower else 0.1
        return 0.2

    def _score_search_match(
        self, filepath: str, search_matches: dict[str, list[str]] | None
    ) -> float:
        """Score based on search match presence."""
        if not search_matches:
            return 0.0
        if filepath in search_matches:
            return 1.0
        # Partial match: file path contains a match key
        for matched_path in search_matches:
            if filepath in matched_path or matched_path in filepath:
                return 0.5
        return 0.0

    def _score_importance(self, filepath: str) -> float:
        """Score based on file importance."""
        name = Path(filepath).name.lower()
        important = {
            "readme.md": 1.0,
            "readme": 1.0,
            "pyproject.toml": 0.9,
            "package.json": 0.9,
            "cargo.toml": 0.9,
            "go.mod": 0.9,
            "makefile": 0.8,
            "dockerfile": 0.7,
            "__init__.py": 0.6,
            "conftest.py": 0.7,
        }
        return important.get(name, 0.1)

    def _score_test_proximity(self, filepath: str, all_files: list[str]) -> float:
        """Score based on whether a corresponding test file exists."""
        name = Path(filepath).stem
        parent = Path(filepath).parent

        # Check if this IS a test file
        if "test" in name.lower() or "spec" in name.lower():
            return 0.8

        # Check if a corresponding test exists
        test_patterns = [
            f"test_{name}.py",
            f"{name}_test.py",
            f"{name}.test.js",
            f"{name}.test.ts",
            f"{name}.spec.js",
            f"{name}.spec.ts",
        ]
        for pattern in test_patterns:
            for f in all_files:
                if Path(f).name == pattern:
                    return 0.6
        return 0.1

    def score_file(
        self,
        filepath: str,
        task: str,
        all_files: list[str] | None = None,
        search_matches: dict[str, list[str]] | None = None,
    ) -> RelevanceScore:
        """Score a single file's relevance to a task."""
        keywords = self._extract_keywords(task)
        all_files = all_files or []

        signals = {
            "filename_match": self._score_filename(filepath, keywords),
            "path_match": self._score_path(filepath, keywords),
            "extension_match": self._score_extension(filepath, task),
            "search_match": self._score_search_match(filepath, search_matches),
            "importance": self._score_importance(filepath),
            "test_proximity": self._score_test_proximity(filepath, all_files),
        }

        # Weighted total
        total = (
            self.config.filename_match * signals["filename_match"]
            + self.config.path_match * signals["path_match"]
            + self.config.extension_match * signals["extension_match"]
            + self.config.search_match * signals["search_match"]
            + self.config.importance * signals["importance"]
            + self.config.test_proximity * signals["test_proximity"]
        )

        return RelevanceScore(
            path=filepath,
            total_score=total,
            signals=signals,
        )

    def rank_files(
        self,
        files: list[str],
        task: str,
        search_matches: dict[str, list[str]] | None = None,
        max_results: int = 50,
    ) -> list[RelevanceScore]:
        """Rank files by relevance to a task.

        Returns top N files sorted by relevance score descending.
        """
        scored = [
            self.score_file(f, task, files, search_matches)
            for f in files
        ]
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored[:max_results]
