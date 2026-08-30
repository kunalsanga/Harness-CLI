"""Test integrity checking.

Detects when an agent modifies test files while trying to fix failing
implementations, and flags suspicious changes (removed assertions,
skipped tests) so completion can be reported truthfully.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


_TEST_NAME_MARKERS = (
    "test_", "_test.py", ".test.", ".spec.", "test.js", "tests.js",
    "spec.js", "test.ts", "conftest",
)


def is_test_like_path(path: str) -> bool:
    """Heuristically determine whether a path points at a test file."""
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    if "/tests/" in normalized or "/test/" in normalized:
        return True
    return any(marker in name for marker in _TEST_NAME_MARKERS)


@dataclass
class IntegrityReport:
    """Result of a test integrity review."""

    test_files_modified: list[str] = field(default_factory=list)
    removed_assertions: int = 0
    added_assertions: int = 0
    removed_test_functions: int = 0
    warning: str = ""

    @property
    def suspicious(self) -> bool:
        return bool(self.warning)


def _diff_stats_for_file(workspace: Path, rel_path: str) -> tuple[int, int, int] | None:
    """Return (removed_assertions, added_assertions, removed_test_fns) via git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", "--", rel_path],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
    except Exception:
        return None

    removed_assertions = 0
    added_assertions = 0
    removed_test_fns = 0
    for line in result.stdout.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            body = line[1:].strip()
            if body.startswith("assert") or "assert." in body or ".expect(" in body:
                removed_assertions += 1
            if body.startswith("def test_") or body.startswith("test(") or "it(" in body:
                removed_test_fns += 1
        elif line.startswith("+"):
            body = line[1:].strip()
            if body.startswith("assert") or "assert." in body or ".expect(" in body:
                added_assertions += 1
    return removed_assertions, added_assertions, removed_test_fns


def check_test_integrity(
    workspace_root: Path,
    modified_files: list[str],
) -> IntegrityReport:
    """Review modifications to test files for suspicious weakening.

    Suspicious = assertions or test functions removed without
    compensating additions while the agent was fixing failures.
    """
    report = IntegrityReport()
    test_files = [f for f in modified_files if is_test_like_path(f)]
    if not test_files:
        return report
    report.test_files_modified = test_files

    try:
        is_git_repo = (Path(workspace_root) / ".git").exists()
    except (TypeError, OSError):
        is_git_repo = False

    total_removed = 0
    total_added = 0
    total_removed_fns = 0
    diff_available = False

    if is_git_repo:
        for tf in test_files:
            stats = _diff_stats_for_file(Path(workspace_root), tf)
            if stats is None:
                continue
            diff_available = True
            total_removed += stats[0]
            total_added += stats[1]
            total_removed_fns += stats[2]

    report.removed_assertions = total_removed
    report.added_assertions = total_added
    report.removed_test_functions = total_removed_fns

    if total_removed_fns > 0:
        report.warning = (
            f"{total_removed_fns} test function(s) removed from "
            f"{', '.join(test_files)}"
        )
    elif diff_available and total_removed > total_added:
        report.warning = (
            f"{total_removed} assertion(s) removed but only {total_added} added in "
            f"{', '.join(test_files)} — tests may have been weakened to force a pass"
        )
    elif not diff_available:
        report.warning = (
            f"Test file(s) modified: {', '.join(test_files)} — "
            "review that expectations were not weakened"
        )
    return report
