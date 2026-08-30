"""Deterministic classification of command and Git failures."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FailureDiagnosis:
    category: str
    reason: str
    retry_shell_format: bool = False


_MISSING_EXE = re.compile(
    r"(is not recognized as|command not found|no such file or directory|"
    r"not found|'.*' is not recognized|cannot find the (?:path|file))",
    re.I,
)
_PERMISSION = re.compile(r"(permission denied|access is denied|eacces|eperm)", re.I)
_NETWORK = re.compile(
    r"(could not resolve host|network is unreachable|connection refused|"
    r"timed out waiting|ssl|proxy|401 unauthorized|403 forbidden)",
    re.I,
)
_GIT_IDENTITY = re.compile(r"(please tell me who you are|author identity unknown|user\.name)", re.I)
_GIT_AUTH = re.compile(
    r"(authentication failed|could not read username|permission.*github|"
    r"could not read from remote|403|401|invalid credentials)",
    re.I,
)
_GIT_REMOTE = re.compile(r"(no such remote|does not appear to be a git repository|no upstream)", re.I)
_TEST_FAIL = re.compile(
    r"(failed|failure|assert|error:|tests? failed|failures=|[1-9]\d* failed)",
    re.I,
)
_SYNTAX = re.compile(
    r"(syntax error|unexpected token|is not recognized|'&&' is not|"
    r"the system cannot find the path|was unexpected at this time)",
    re.I,
)
_TIMEOUT = re.compile(r"timed? out", re.I)

_SHELL_WRAP = re.compile(
    r"^\s*(cmd(?:\.exe)?\s+/c\s+|powershell(?:\.exe)?\s+(-command\s+|/c\s+)?)",
    re.I,
)
_CD_PREFIX = re.compile(
    r"^\s*cd\s+(/d\s+)?(\"[^\"]+\"|'[^']+'|\S+)\s*(&&|\||;)\s*",
    re.I,
)


def normalize_shell_command(command: str) -> str:
    """Strip cd wrappers and shell-format guessing so retries compare equal."""
    cmd = (command or "").strip()
    for _ in range(4):
        next_cmd = _SHELL_WRAP.sub("", cmd).strip().strip('"')
        next_cmd = _CD_PREFIX.sub("", next_cmd).strip()
        if next_cmd == cmd:
            break
        cmd = next_cmd
    return cmd


def classify_command_failure(
    command: str,
    *,
    exit_code: int | None,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> FailureDiagnosis:
    blob = f"{stdout}\n{stderr}\n{command}"
    if timed_out:
        return FailureDiagnosis("timeout", f"Command timed out: {command[:80]}")
    if _MISSING_EXE.search(blob):
        return FailureDiagnosis("missing_executable", _first_line(stderr) or "Executable not found")
    if _PERMISSION.search(blob):
        return FailureDiagnosis("permission_failure", _first_line(stderr) or "Permission denied")
    if _NETWORK.search(blob):
        return FailureDiagnosis("network_failure", _first_line(stderr) or "Network failure")
    if command.strip().lower().startswith("git") or "git " in command.lower():
        git = classify_git_failure("command", stdout, stderr, exit_code)
        return FailureDiagnosis("git_failure", git.reason)
    if _SYNTAX.search(blob) or _looks_like_shell_guess(command):
        return FailureDiagnosis(
            "command_syntax",
            _first_line(stderr) or "Command syntax error",
            retry_shell_format=True,
        )
    if _is_test_command(command) and (exit_code not in (0, None)):
        return FailureDiagnosis(
            "test_failure",
            _first_line(stderr) or _first_line(stdout) or f"Tests failed (exit {exit_code})",
        )
    if exit_code not in (0, None):
        if _TEST_FAIL.search(blob) and _is_test_command(command):
            return FailureDiagnosis("test_failure", _first_line(stderr) or "Tests failed")
        return FailureDiagnosis(
            "environment_failure",
            _first_line(stderr) or f"Command failed with exit code {exit_code}",
        )
    return FailureDiagnosis("environment_failure", _first_line(stderr) or "Command failed")


def classify_git_failure(
    operation: str,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = None,
) -> FailureDiagnosis:
    blob = f"{stdout}\n{stderr}"
    if _GIT_IDENTITY.search(blob):
        return FailureDiagnosis("git_failure", "Git identity is not configured")
    if _GIT_AUTH.search(blob):
        return FailureDiagnosis("git_failure", "GitHub authentication failed")
    if _GIT_REMOTE.search(blob) or "no git remote" in blob.lower():
        return FailureDiagnosis("git_failure", "No Git remote configured")
    if "nothing to commit" in blob.lower() or "working tree clean" in blob.lower():
        return FailureDiagnosis("git_failure", "Working tree already clean")
    reason = _first_line(stderr) or _first_line(stdout) or f"Git {operation} failed"
    if exit_code not in (0, None):
        return FailureDiagnosis("git_failure", reason)
    return FailureDiagnosis("git_failure", reason)


def parse_test_counts(output: str) -> tuple[int, int] | None:
    """Return (passed, total) if a common test runner summary is present."""
    text = output or ""
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+passed", text, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\s+passed(?:.*\s(\d+)\s+failed)?", text, re.I)
    if m:
        passed = int(m.group(1))
        failed = int(m.group(2) or 0)
        return passed, passed + failed
    m = re.search(r"(\d+)\s+failed.*?(\d+)\s+passed", text, re.I)
    if m:
        failed, passed = int(m.group(1)), int(m.group(2))
        return passed, passed + failed
    return None


def _is_test_command(command: str) -> bool:
    cmd = command.lower()
    markers = (
        "pytest", "npm test", "yarn test", "pnpm test", "cargo test",
        "go test", "jest", "vitest", "mocha", "phpunit", "dotnet test",
        "node test", "python -m test",
    )
    return any(m in cmd for m in markers) or bool(re.search(r"\btest\S*\.(js|py|ts)\b", cmd))


def _looks_like_shell_guess(command: str) -> bool:
    low = command.lower()
    return any(
        token in low
        for token in ("cmd /c", "cmd.exe", "powershell -", "cd /d ", "&& git")
    )


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return ""
