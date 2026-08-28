"""Verification engine for validating agent work."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness_core.agent.types import ToolResult, ToolResultStatus


@dataclass
class VerificationCheck:
    """A verification check to run."""

    name: str
    command: str
    success_exit_code: int = 0
    timeout_seconds: float = 60.0


@dataclass
class VerificationResult:
    """Result of a verification check."""

    check_name: str
    passed: bool
    output: str
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class VerificationReport:
    """Complete verification report."""

    all_passed: bool = True
    results: list[VerificationResult] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0
    checks_failed: int = 0

    def add_result(self, result: VerificationResult) -> None:
        self.results.append(result)
        self.checks_run += 1
        if result.passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
            self.all_passed = False


class VerificationEngine:
    """Runs verification checks to validate agent work."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or Path.cwd()
        self._checks: list[VerificationCheck] = []

    def register_check(self, check: VerificationCheck) -> None:
        """Register a verification check."""
        self._checks.append(check)

    async def detect_ecosystem(self) -> list[VerificationCheck]:
        """Detect project ecosystem and return appropriate checks."""
        checks: list[VerificationCheck] = []

        # Python
        if (self.workspace_root / "pyproject.toml").exists() or (
            self.workspace_root / "setup.py"
        ).exists():
            if (self.workspace_root / "pytest.ini").exists() or (
                self.workspace_root / "pyproject.toml"
            ).read_text(encoding="utf-8", errors="replace").find("pytest") != -1:
                checks.append(
                    VerificationCheck(name="pytest", command="python -m pytest -x -q")
                )

        # Node.js
        if (self.workspace_root / "package.json").exists():
            pkg = (self.workspace_root / "package.json").read_text(
                encoding="utf-8", errors="replace"
            )
            if '"test"' in pkg:
                checks.append(
                    VerificationCheck(name="npm_test", command="npm test", timeout_seconds=120)
                )
            if '"lint"' in pkg:
                checks.append(
                    VerificationCheck(name="npm_lint", command="npm run lint")
                )

        # Rust
        if (self.workspace_root / "Cargo.toml").exists():
            checks.append(
                VerificationCheck(name="cargo_test", command="cargo test", timeout_seconds=120)
            )

        # Go
        if list(self.workspace_root.glob("*.go")):
            checks.append(
                VerificationCheck(name="go_test", command="go test ./...", timeout_seconds=120)
            )

        # TypeScript
        if list(self.workspace_root.glob("tsconfig.json")):
            checks.append(
                VerificationCheck(name="tsc_check", command="npx tsc --noEmit")
            )

        return checks

    async def run_checks(
        self, checks: list[VerificationCheck] | None = None
    ) -> VerificationReport:
        """Run verification checks."""
        checks = checks or self._checks
        report = VerificationReport()

        for check in checks:
            result = await self._run_check(check)
            report.add_result(result)

        return report

    async def _run_check(self, check: VerificationCheck) -> VerificationResult:
        """Run a single verification check."""
        import time

        start = time.time()
        try:
            process = await asyncio.create_subprocess_shell(
                check.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=check.timeout_seconds
            )

            duration_ms = (time.time() - start) * 1000
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            passed = process.returncode == check.success_exit_code

            return VerificationResult(
                check_name=check.name,
                passed=passed,
                output=output[:5000] if output else "",
                error=error[:2000] if error and not passed else None,
                duration_ms=duration_ms,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start) * 1000
            return VerificationResult(
                check_name=check.name,
                passed=False,
                output="",
                error=f"Timed out after {check.timeout_seconds}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return VerificationResult(
                check_name=check.name,
                passed=False,
                output="",
                error=str(e),
                duration_ms=duration_ms,
            )
