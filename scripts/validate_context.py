#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context file validator for Harness.

Checks that all paths referenced in AGENTS.md / CLAUDE.md / AI_CONTEXT.md
actually exist, that context files are not excessively large, and that no
obvious secrets have been accidentally embedded.

Usage:
    uv run python scripts/validate_context.py

Zero dependencies beyond the standard library.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 failures.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
CONTEXT_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "docs/AI_CONTEXT.md",
    "src/harness_core/AGENTS.md",
    "src/harness_core/agent/AGENTS.md",
    "src/harness_core/agents/AGENTS.md",
    "src/harness_core/cli/AGENTS.md",
    "src/harness_core/context/AGENTS.md",
    "src/harness_core/planning/AGENTS.md",
    "src/harness_core/providers/AGENTS.md",
    "src/harness_core/recovery/AGENTS.md",
    "src/harness_core/routing/AGENTS.md",
    "src/harness_core/runtime/AGENTS.md",
    "src/harness_core/tools/AGENTS.md",
    "src/harness_core/verification/AGENTS.md",
    "tests/AGENTS.md",
]

# Warn if a context file exceeds this word count.
MAX_WORDS = 2500

# Patterns that look like embedded secrets.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI/OpenRouter-style keys
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),          # Groq keys
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),      # NVIDIA keys
    re.compile(r"['\"](?:key|token|secret)['\"]:\s*['\"][A-Za-z0-9_\-]{16,}['\"]", re.I),
    re.compile(r"(?:API_KEY|SECRET|TOKEN)\s*=\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?"),
]

# Markdown link patterns: [text](path) and backtick-code path references like `src/...`
LINK_PATTERN = re.compile(r"\[.*?\]\(([^)]+)\)")
CODE_PATH_PATTERN = re.compile(r"`(src/harness_core/[^`]+\.py|tests/[^`]+\.py)`")


def _is_external(path_str: str) -> bool:
    return path_str.startswith("http") or path_str.startswith("#")


def check_context_file(rel_path: str) -> list[str]:
    """Check one context file. Returns list of error/warning strings."""
    issues: list[str] = []
    full_path = REPO_ROOT / rel_path

    if not full_path.exists():
        return [f"MISSING context file: {rel_path}"]

    content = full_path.read_text(encoding="utf-8", errors="replace")

    # 1. Size check
    word_count = len(content.split())
    if word_count > MAX_WORDS:
        issues.append(
            f"LARGE ({word_count} words > {MAX_WORDS}): {rel_path} — consider compressing"
        )

    # 2. Secret pattern check
    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            issues.append(f"POSSIBLE SECRET in {rel_path} — review before committing")
            break  # one warning per file is enough

    # 3. Referenced path existence
    for match in LINK_PATTERN.finditer(content):
        ref = match.group(1)
        if _is_external(ref):
            continue
        # Strip leading slash or ./ for resolution
        ref_clean = ref.lstrip("/").lstrip("./")
        # Strip fragment anchors
        ref_clean = ref_clean.split("#")[0]
        if not ref_clean:
            continue
        ref_path = REPO_ROOT / ref_clean
        if not ref_path.exists():
            issues.append(f"BROKEN LINK in {rel_path}: '{ref}' → '{ref_clean}' does not exist")

    # 4. Backtick code path references
    for match in CODE_PATH_PATTERN.finditer(content):
        ref = match.group(1)
        # Skip template placeholders like `tests/unit/test_<name>.py`
        if "<" in ref or ">" in ref:
            continue
        ref_path = REPO_ROOT / ref
        if not ref_path.exists():
            issues.append(f"MISSING PATH ref in {rel_path}: `{ref}` does not exist")

    return issues


def main() -> int:
    all_issues: list[str] = []

    print("Harness Context Validator")
    print("=" * 50)

    for rel_path in CONTEXT_FILES:
        issues = check_context_file(rel_path)
        if issues:
            for issue in issues:
                print(f"  FAIL {issue}")
            all_issues.extend(issues)
        else:
            print(f"  OK   {rel_path}")

    print("=" * 50)
    errors = [i for i in all_issues if i.startswith("MISSING") or i.startswith("BROKEN")]
    warnings = [i for i in all_issues if i not in errors]

    if warnings:
        print(f"\n{len(warnings)} warning(s) — review recommended.")
    if errors:
        print(f"\n{len(errors)} error(s) found — fix before committing context files.")
        return 1

    print("\nAll context files OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
