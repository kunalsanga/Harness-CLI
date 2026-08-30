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
    """Manages context assembly and compression for model interactions.

    Caches workspace structure and file content for performance.
    """

    # Directories to always skip during discovery
    _SKIP_DIRS: set[str] = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "dist", "build", ".next", "target", "vendor", "coverage",
        ".tox", ".mypy_cache", ".pytest_cache", "egg-info",
    }

    # File extensions to prioritize for project understanding (higher = more important)
    _PRIORITY_EXTENSIONS: dict[str, int] = {
        ".md": 50,     # Documentation — often most useful for understanding
        ".html": 40,   # Web entry points
        ".css": 30,    # Styles
        ".js": 40,     # JavaScript source
        ".ts": 40,     # TypeScript source
        ".py": 40,     # Python source
        ".rs": 40,     # Rust source
        ".go": 40,     # Go source
        ".java": 40,   # Java source
        ".json": 20,   # Config
        ".yaml": 20,   # Config
        ".yml": 20,    # Config
        ".toml": 20,   # Config
        "": 0,         # Unknown
    }

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or Path.cwd()
        self._file_cache: dict[str, str] = {}
        self._project_cache: dict[str, Any] | None = None
        self._project_cache_time: float = 0.0

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ≈ 4 characters)."""
        return len(text) // 4

    def _should_skip_dir(self, name: str) -> bool:
        """Check if a directory should be skipped during discovery."""
        return name in self._SKIP_DIRS or name.startswith(".")

    async def discover_project(self) -> dict[str, Any]:
        """Discover project structure and metadata with caching."""
        import time as _time

        # Use cache if fresh (within 30 seconds)
        if self._project_cache is not None:
            age = _time.time() - self._project_cache_time
            if age < 30:
                return self._project_cache

        info: dict[str, Any] = {
            "root": str(self.workspace_root),
            "files": [],
            "languages": set(),
            "has_tests": False,
            "has_git": False,
            "package_manager": None,
            "config_files": [],
            "entry_points": [],
            "readme": None,
        }

        # Check for git
        if (self.workspace_root / ".git").exists():
            info["has_git"] = True

        # Detect package manager
        ws = self.workspace_root
        if (ws / "package.json").exists():
            info["package_manager"] = "npm"
            if (ws / "pnpm-lock.yaml").exists():
                info["package_manager"] = "pnpm"
            elif (ws / "yarn.lock").exists():
                info["package_manager"] = "yarn"
            elif (ws / "bun.lockb").exists():
                info["package_manager"] = "bun"
        elif (ws / "Cargo.toml").exists():
            info["package_manager"] = "cargo"
        elif (ws / "go.mod").exists():
            info["package_manager"] = "go"
        elif (ws / "pyproject.toml").exists() or (ws / "setup.py").exists():
            info["package_manager"] = "python"

        # Check for README
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = ws / readme_name
            if readme_path.exists():
                info["readme"] = readme_name
                break

        # Discover files (bounded, prioritized)
        try:
            source_files: list[tuple[int, str, str]] = []  # (priority, rel_path, ext)
            config_files: list[str] = []

            for p in ws.rglob("*"):
                if not p.is_file():
                    continue
                # Skip ignored directories
                parts = p.relative_to(ws).parts
                if any(self._should_skip_dir(part) for part in parts):
                    continue
                # Skip large binary files
                if p.stat().st_size > 1_000_000:
                    continue

                rel = str(p.relative_to(ws))
                suffix = p.suffix.lower()

                # Detect language
                lang_map = {
                    ".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".rs": "rust", ".go": "go", ".java": "java",
                    ".cpp": "cpp", ".c": "c",
                }
                if suffix in lang_map:
                    info["languages"].add(lang_map[suffix])

                # Detect tests
                if "test" in p.name.lower() or "spec" in p.name.lower():
                    info["has_tests"] = True

                # Detect entry points
                if p.name in ("index.html", "main.py", "main.ts", "main.js",
                              "app.py", "App.tsx", "App.jsx", "lib.rs", "main.rs"):
                    info["entry_points"].append(rel)

                # Config files
                if suffix in (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"):
                    config_files.append(rel)

                # Source files (skip deep nesting, limit total)
                depth = len(parts)
                priority = self._PRIORITY_EXTENSIONS.get(suffix, 0)
                if depth <= 3 and len(source_files) < 200:
                    source_files.append((priority, rel, suffix))

            # Sort: README first, then by priority, then by name
            source_files.sort(key=lambda x: (-x[0], x[1]))
            info["files"] = [f[1] for f in source_files]
            info["config_files"] = config_files[:20]

        except Exception:
            pass

        info["languages"] = list(info["languages"])

        # Cache the result
        self._project_cache = info
        self._project_cache_time = _time.time()

        return info

    async def assemble_context(
        self,
        task: str,
        project_info: dict[str, Any] | None = None,
        budget: ContextBudget | None = None,
    ) -> list[ContextPiece]:
        """Assemble context for a task.

        Provides rich project context in a single batch rather than requiring
        multiple model calls to discover project structure.
        """
        budget = budget or ContextBudget()
        pieces: list[ContextPiece] = []

        # Add project metadata summary (concise)
        if project_info:
            meta_parts: list[str] = []
            if project_info.get("languages"):
                meta_parts.append(f"Languages: {', '.join(project_info['languages'])}")
            if project_info.get("package_manager"):
                meta_parts.append(f"Package manager: {project_info['package_manager']}")
            if project_info.get("has_tests"):
                meta_parts.append("Has tests: yes")
            if project_info.get("readme"):
                meta_parts.append(f"README: {project_info['readme']}")
            if project_info.get("entry_points"):
                meta_parts.append(f"Entry points: {', '.join(project_info['entry_points'][:5])}")

            if meta_parts:
                pieces.append(
                    ContextPiece(
                        source="project_metadata",
                        content="Project info:\n" + "\n".join(meta_parts),
                        priority=20,
                    )
                )

            # Add file tree (prioritized, limited)
            files = project_info.get("files", [])[:30]
            if files:
                structure = "\n".join(files)
                pieces.append(
                    ContextPiece(
                        source="project_structure",
                        content=f"Project files:\n{structure}",
                        priority=10,
                    )
                )

        # Add task context (highest priority)
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
