"""Workspace path confinement."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_BLOCKED_NAMES = {".env", "id_rsa", "id_ed25519"}


def _as_root(workspace_root: Any) -> Path | None:
    """Best-effort conversion of a workspace root to a resolved Path."""
    if workspace_root is None:
        return None
    try:
        return Path(workspace_root).resolve()
    except (TypeError, OSError, RuntimeError):
        return None


def resolve_in_workspace(
    workspace_root: Path | None,
    user_path: str | None,
    *,
    default_to_root: bool = True,
) -> tuple[Path | None, str | None]:
    """Resolve a user-supplied path and enforce the workspace boundary.

    Returns (resolved_path, error). When workspace_root is None (or not a
    real path), paths are resolved as given (legacy unconstrained mode).
    """
    root = _as_root(workspace_root)
    if root is None:
        if not user_path:
            return (Path.cwd() if default_to_root else None, None)
        return Path(user_path), None

    raw = (user_path or "").strip()
    if not raw:
        return (root if default_to_root else root, None)

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as e:
        return None, f"Invalid path: {e}"

    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"Path is outside the workspace: {raw}"

    return resolved, None


def cwd_in_workspace(workspace_root: Path | None, cwd: str | None) -> tuple[str, str | None]:
    """Return a confined working directory string."""
    root = _as_root(workspace_root)
    if root is None:
        return cwd or str(Path.cwd()), None
    path, err = resolve_in_workspace(workspace_root, cwd or "", default_to_root=True)
    if err or path is None:
        return str(root), err
    if not path.is_dir():
        return str(root), None
    return str(path), None
