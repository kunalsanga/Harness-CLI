"""Unit tests for the verification engine."""

import tempfile
from pathlib import Path

import pytest

from harness_core.verification.engine import VerificationCheck, VerificationEngine


class TestVerificationEngine:
    """Tests for VerificationEngine."""

    @pytest.mark.asyncio
    async def test_run_check_success(self, tmp_path: Path):
        engine = VerificationEngine(tmp_path)
        check = VerificationCheck(name="echo", command="echo hello")
        report = await engine.run_checks([check])

        assert report.all_passed
        assert report.checks_passed == 1

    @pytest.mark.asyncio
    async def test_run_check_failure(self, tmp_path: Path):
        engine = VerificationEngine(tmp_path)
        check = VerificationCheck(name="false", command="false")
        report = await engine.run_checks([check])

        assert not report.all_passed
        assert report.checks_failed == 1

    @pytest.mark.asyncio
    async def test_detect_python_ecosystem(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\ntestpaths = ['tests']")

        engine = VerificationEngine(tmp_path)
        checks = await engine.detect_ecosystem()

        check_names = [c.name for c in checks]
        assert "pytest" in check_names

    @pytest.mark.asyncio
    async def test_detect_node_ecosystem(self, tmp_path: Path):
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')

        engine = VerificationEngine(tmp_path)
        checks = await engine.detect_ecosystem()

        check_names = [c.name for c in checks]
        assert "npm_test" in check_names

    @pytest.mark.asyncio
    async def test_empty_checks(self, tmp_path: Path):
        engine = VerificationEngine(tmp_path)
        report = await engine.run_checks([])

        assert report.all_passed
        assert report.checks_run == 0
