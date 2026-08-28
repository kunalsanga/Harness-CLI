"""Repository structure analyzer.

Detects languages, package managers, build systems, test frameworks,
source/test directories, entry points, and configuration files.
Supports incremental ecosystem detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Ecosystem Definitions ────────────────────────────────────────────────────


@dataclass
class Ecosystem:
    """A detected technology ecosystem."""

    name: str
    version: str = ""
    config_files: list[str] = field(default_factory=list)
    source_dirs: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    lint_commands: list[str] = field(default_factory=list)
    confidence: float = 0.0


# Ecosystem detector registry — each returns an Ecosystem or None
ECOSYSTEM_DETECTORS: list[dict[str, Any]] = []


def _register_ecosystem(name: str, detector: Any) -> None:
    ECOSYSTEM_DETECTORS.append({"name": name, "detect": detector})


def _detect_python(root: Path) -> Ecosystem | None:
    indicators = []
    config_files = []
    source_dirs = []
    test_dirs = []

    if (root / "pyproject.toml").exists():
        indicators.append("pyproject.toml")
        config_files.append("pyproject.toml")
    if (root / "setup.py").exists():
        indicators.append("setup.py")
        config_files.append("setup.py")
    if (root / "setup.cfg").exists():
        indicators.append("setup.cfg")
        config_files.append("setup.cfg")
    if (root / "requirements.txt").exists():
        indicators.append("requirements.txt")
        config_files.append("requirements.txt")
    if (root / "Pipfile").exists():
        indicators.append("Pipfile")
        config_files.append("Pipfile")
    if (root / "poetry.lock").exists():
        indicators.append("poetry.lock")

    # Source dirs
    for d in ["src", "lib", "."]:
        if (root / d).exists():
            py_files = list((root / d).rglob("*.py"))
            if py_files:
                source_dirs.append(d)

    # Test dirs
    for d in ["tests", "test", "spec"]:
        if (root / d).exists():
            test_dirs.append(d)

    if not indicators:
        return None

    return Ecosystem(
        name="python",
        config_files=config_files,
        source_dirs=source_dirs,
        test_dirs=test_dirs,
        build_commands=["python -m build"],
        test_commands=["pytest", "python -m pytest"],
        lint_commands=["ruff check", "mypy"],
        confidence=min(1.0, len(indicators) * 0.3),
    )


def _detect_nodejs(root: Path) -> Ecosystem | None:
    indicators = []
    config_files = []

    if (root / "package.json").exists():
        indicators.append("package.json")
        config_files.append("package.json")
    if (root / "package-lock.json").exists():
        indicators.append("package-lock.json")
    if (root / "yarn.lock").exists():
        indicators.append("yarn.lock")
    if (root / "pnpm-lock.yaml").exists():
        indicators.append("pnpm-lock.yaml")
    if (root / "node_modules").exists():
        indicators.append("node_modules")

    if not indicators:
        return None

    test_dirs = []
    for d in ["tests", "test", "__tests__", "spec"]:
        if (root / d).exists():
            test_dirs.append(d)

    return Ecosystem(
        name="nodejs",
        config_files=config_files,
        source_dirs=["src", "lib", "."],
        test_dirs=test_dirs,
        build_commands=["npm run build", "yarn build"],
        test_commands=["npm test", "yarn test"],
        lint_commands=["npm run lint", "eslint"],
        confidence=min(1.0, len(indicators) * 0.3),
    )


def _detect_typescript(root: Path) -> Ecosystem | None:
    indicators = []
    config_files = []

    if (root / "tsconfig.json").exists():
        indicators.append("tsconfig.json")
        config_files.append("tsconfig.json")
    if (root / "package.json").exists():
        # Check for TS deps
        try:
            import json
            pkg = json.loads((root / "package.json").read_text())
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            ts_deps = [d for d in deps if "typescript" in d or "ts-" in d]
            if ts_deps:
                indicators.append("typescript-deps")
        except Exception:
            pass

    if not indicators:
        return None

    return Ecosystem(
        name="typescript",
        config_files=config_files,
        source_dirs=["src", "lib"],
        test_dirs=["tests", "__tests__", "test", "spec"],
        build_commands=["tsc", "npm run build"],
        test_commands=["jest", "vitest", "mocha"],
        lint_commands=["eslint", "tsc --noEmit"],
        confidence=min(1.0, len(indicators) * 0.4),
    )


def _detect_rust(root: Path) -> Ecosystem | None:
    indicators = []
    config_files = []

    if (root / "Cargo.toml").exists():
        indicators.append("Cargo.toml")
        config_files.append("Cargo.toml")
    if (root / "Cargo.lock").exists():
        indicators.append("Cargo.lock")

    if not indicators:
        return None

    return Ecosystem(
        name="rust",
        config_files=config_files,
        source_dirs=["src"],
        test_dirs=["tests"],
        build_commands=["cargo build"],
        test_commands=["cargo test"],
        lint_commands=["cargo clippy"],
        confidence=min(1.0, len(indicators) * 0.5),
    )


def _detect_go(root: Path) -> Ecosystem | None:
    indicators = []
    config_files = []

    if (root / "go.mod").exists():
        indicators.append("go.mod")
        config_files.append("go.mod")
    if (root / "go.sum").exists():
        indicators.append("go.sum")

    if not indicators:
        return None

    return Ecosystem(
        name="go",
        config_files=config_files,
        source_dirs=["."],
        test_dirs=["."],
        build_commands=["go build"],
        test_commands=["go test ./..."],
        lint_commands=["golangci-lint"],
        confidence=min(1.0, len(indicators) * 0.5),
    )


_register_ecosystem("python", _detect_python)
_register_ecosystem("nodejs", _detect_nodejs)
_register_ecosystem("typescript", _detect_typescript)
_register_ecosystem("rust", _detect_rust)
_register_ecosystem("go", _detect_go)


# ── Repository Analysis ──────────────────────────────────────────────────────


@dataclass
class RepositoryInfo:
    """Complete analysis of a repository."""

    root: Path
    ecosystems: list[Ecosystem] = field(default_factory=list)
    git_available: bool = False
    git_branch: str = ""
    total_files: int = 0
    source_files: int = 0
    test_files: int = 0
    config_files: int = 0
    documentation_files: int = 0
    all_files: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    important_files: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    @property
    def primary_ecosystem(self) -> Ecosystem | None:
        """The ecosystem with highest confidence."""
        if not self.ecosystems:
            return None
        return max(self.ecosystems, key=lambda e: e.confidence)

    @property
    def has_tests(self) -> bool:
        return self.test_files > 0

    @property
    def has_git(self) -> bool:
        return self.git_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "ecosystems": [
                {"name": e.name, "confidence": e.confidence, "config_files": e.config_files}
                for e in self.ecosystems
            ],
            "git_available": self.git_available,
            "git_branch": self.git_branch,
            "total_files": self.total_files,
            "source_files": self.source_files,
            "test_files": self.test_files,
            "config_files": self.config_files,
            "documentation_files": self.documentation_files,
            "languages": self.languages,
            "important_files": self.important_files[:20],
        }


class RepositoryAnalyzer:
    """Analyzes repository structure and technology stack.

    Detects ecosystems, files, directories, and project metadata.
    Results are cached after first analysis.
    """

    # Directories to always skip
    SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache",
        "dist", "build", ".next", ".nuxt", "target",
        ".eggs", "*.egg-info",
    }

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path.cwd()
        self._cached_info: RepositoryInfo | None = None

    async def analyze(self, force: bool = False) -> RepositoryInfo:
        """Analyze the repository. Results are cached."""
        if self._cached_info is not None and not force:
            return self._cached_info

        info = RepositoryInfo(root=self._root)

        # Detect ecosystems
        for detector in ECOSYSTEM_DETECTORS:
            eco = detector["detect"](self._root)
            if eco is not None:
                info.ecosystems.append(eco)
                info.languages.append(eco.name)

        # Detect git
        if (self._root / ".git").exists():
            info.git_available = True
            info.git_branch = self._get_git_branch()

        # Discover files
        self._discover_files(info)

        self._cached_info = info
        return info

    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _discover_files(self, info: RepositoryInfo) -> None:
        """Discover all files and categorize them."""
        source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".c", ".cpp", ".h"}
        test_keywords = {"test", "spec", "_test"}
        config_names = {
            "pyproject.toml", "setup.py", "setup.cfg", "package.json",
            "tsconfig.json", "Cargo.toml", "go.mod", ".eslintrc",
            "Makefile", "Dockerfile", "docker-compose.yml", ".github",
            "tox.ini", "mypy.ini", ".ruff.toml",
        }
        doc_names = {"README", "CHANGELOG", "CONTRIBUTING", "LICENSE", "docs"}

        for root_dir, dirs, files in os.walk(self._root):
            root_path = Path(root_dir)
            rel_root = root_path.relative_to(self._root)

            # Skip hidden and known unimportant dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in self.SKIP_DIRS
                and not d.endswith(".egg-info")
            ]

            # Track directories
            if str(rel_root) != ".":
                info.directories.append(str(rel_root))

            for fname in files:
                if fname.startswith("."):
                    continue

                rel_path = str(rel_root / fname) if str(rel_root) != "." else fname
                info.all_files.append(rel_path)
                info.total_files += 1

                # Categorize
                ext = Path(fname).suffix.lower()
                name_lower = fname.lower()

                if ext in source_exts:
                    info.source_files += 1

                if any(kw in name_lower for kw in test_keywords):
                    info.test_files += 1

                if fname in config_names or name_lower in {c.lower() for c in config_names}:
                    info.config_files += 1

                if any(doc in name_lower for doc in doc_names):
                    info.documentation_files += 1

        # Identify important files
        important = []
        for f in info.all_files:
            name = Path(f).name.lower()
            if name in {"readme.md", "readme", "pyproject.toml", "package.json",
                        "cargo.toml", "go.mod", "makefile", "dockerfile"}:
                important.append(f)
        info.important_files = sorted(set(important))

    def get_source_files(self) -> list[str]:
        """Get all source files."""
        if self._cached_info is None:
            return []
        source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java"}
        return [
            f for f in self._cached_info.all_files
            if Path(f).suffix.lower() in source_exts
        ]

    def get_test_files(self) -> list[str]:
        """Get all test files."""
        if self._cached_info is None:
            return []
        return [
            f for f in self._cached_info.all_files
            if "test" in Path(f).stem.lower() or "spec" in Path(f).stem.lower()
        ]
