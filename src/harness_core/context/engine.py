"""Context engine for assembling model context."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextBudget:
    """Budget for context assembly."""

    model_context_limit: int = 128000
    system_prompt_tokens: int = 0
    tool_schema_tokens: int = 0
    reserved_output_tokens: int = 4096

    @property
    def available_tokens(self) -> int:
        return (
            self.model_context_limit
            - self.system_prompt_tokens
            - self.tool_schema_tokens
            - self.reserved_output_tokens
        )


@dataclass
class ContextPiece:
    """A piece of context."""

    source: str
    content: str
    priority: int = 0
    tokens_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextEngine:
    """Manages context assembly and compression for model interactions."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or Path.cwd()
        self._file_cache: dict[str, str] = {}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ≈ 4 characters)."""
        return len(text) // 4

    async def discover_project(self) -> dict[str, Any]:
        """Discover project structure and metadata."""
        info: dict[str, Any] = {
            "root": str(self.workspace_root),
            "files": [],
            "languages": set(),
            "has_tests": False,
            "has_git": False,
        }

        # Check for git
        if (self.workspace_root / ".git").exists():
            info["has_git"] = True

        # Discover files
        try:
            for p in self.workspace_root.rglob("*"):
                if p.is_file() and not any(
                    part.startswith(".") for part in p.relative_to(self.workspace_root).parts
                ):
                    rel = str(p.relative_to(self.workspace_root))
                    info["files"].append(rel)

                    # Detect language
                    suffix = p.suffix.lower()
                    lang_map = {
                        ".py": "python",
                        ".js": "javascript",
                        ".ts": "typescript",
                        ".rs": "rust",
                        ".go": "go",
                        ".java": "java",
                        ".cpp": "cpp",
                        ".c": "c",
                    }
                    if suffix in lang_map:
                        info["languages"].add(lang_map[suffix])

                    # Detect tests
                    if "test" in p.name.lower() or "spec" in p.name.lower():
                        info["has_tests"] = True
        except Exception:
            pass

        info["languages"] = list(info["languages"])
        return info

    async def assemble_context(
        self,
        task: str,
        project_info: dict[str, Any] | None = None,
        budget: ContextBudget | None = None,
    ) -> list[ContextPiece]:
        """Assemble context for a task."""
        budget = budget or ContextBudget()
        pieces: list[ContextPiece] = []

        # Add project structure summary
        if project_info:
            files = project_info.get("files", [])[:50]
            structure = "\n".join(files)
            pieces.append(
                ContextPiece(
                    source="project_structure",
                    content=f"Project files:\n{structure}",
                    priority=10,
                )
            )

        # Add task context
        pieces.append(
            ContextPiece(
                source="task",
                content=f"Task: {task}",
                priority=100,
            )
        )

        # Sort by priority
        pieces.sort(key=lambda p: p.priority, reverse=True)

        # Fit within budget
        total_tokens = 0
        selected: list[ContextPiece] = []
        for piece in pieces:
            piece.tokens_estimate = self.estimate_tokens(piece.content)
            if total_tokens + piece.tokens_estimate <= budget.available_tokens:
                selected.append(piece)
                total_tokens += piece.tokens_estimate
            else:
                break

        return selected

    async def compress_context(self, pieces: list[ContextPiece]) -> list[ContextPiece]:
        """Compress context to fit within budget."""
        # Simple compression: truncate long pieces
        max_tokens_per_piece = 2000
        compressed: list[ContextPiece] = []
        for piece in pieces:
            if piece.tokens_estimate > max_tokens_per_piece:
                truncated = piece.content[: max_tokens_per_piece * 4]
                compressed.append(
                    ContextPiece(
                        source=piece.source,
                        content=truncated,
                        priority=piece.priority,
                        tokens_estimate=max_tokens_per_piece,
                    )
                )
            else:
                compressed.append(piece)
        return compressed
