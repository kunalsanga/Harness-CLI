"""Load environment variables from a .env file.

Searches for .env in this order:
  1. Explicit path (if provided)
  2. Current working directory (project workspace)
  3. Harness package directory (installation root)
  4. User home directory (~/.harness.env)

Configuration precedence (first set wins):
  explicit environment variable (already in os.environ)
  > .env in cwd
  > .env in Harness package dir
  > .env in home dir

Does NOT override variables that are already set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_harness_root() -> Path | None:
    """Find the Harness package root directory.

    Walks up from this file's location to find the directory containing
    pyproject.toml (the Harness project root).
    """
    # Start from this file's directory: src/harness_core/config/
    candidate = Path(__file__).resolve().parent
    for _ in range(5):  # Max 5 levels up
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "src").is_dir() and (candidate / "src" / "harness_core").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return None


def _load_single_env(path: Path) -> int:
    """Load one .env file. Returns count of variables loaded."""
    if not path.exists():
        return 0
    try:
        content = path.read_text(encoding="utf-8-sig")  # handles BOM
    except Exception:
        return 0

    loaded = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        # Only set if not already in environment (explicit env wins)
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def load_dotenv(path: str | Path | None = None) -> int:
    """Load .env file into os.environ.

    Configuration precedence:
      1. Explicit path (if provided)
      2. .env in current working directory
      3. .env in Harness package root
      4. ~/.harness.env

    Returns total number of variables loaded across all files.
    """
    total_loaded = 0

    if path is not None:
        return _load_single_env(Path(path))

    # Search locations in priority order
    search_paths: list[Path] = []

    # 1. CWD
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        search_paths.append(cwd_env)

    # 2. Harness package root
    harness_root = _find_harness_root()
    if harness_root is not None:
        root_env = harness_root / ".env"
        if root_env.exists() and root_env not in search_paths:
            search_paths.append(root_env)

    # 3. User home
    home_env = Path.home() / ".harness.env"
    if home_env.exists() and home_env not in search_paths:
        search_paths.append(home_env)

    for env_path in search_paths:
        total_loaded += _load_single_env(env_path)

    return total_loaded
