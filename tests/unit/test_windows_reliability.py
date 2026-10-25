"""Unit tests for Windows reliability, diff accuracy, shell normalization, and quota handling."""

import os
import platform
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from harness_core.tools.filesystem import EditFileTool
from harness_core.tools.shell import RunCommandTool, normalize_windows_command
from harness_core.tools.diagnosis import classify_command_failure, normalize_shell_command
from harness_core.routing.fallback import (
    FallbackEngine,
    FallbackConfig,
    ErrorClassification,
    ERROR_CATEGORY_RATE_LIMIT,
)
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.cli.interactive import _friendly_model_error


class TestDiffAccuracy:
    """Verify that unified diff generation never merges lines or corrupts diff previews."""

    def test_diff_builder_no_trailing_newline_does_not_merge(self):
        """Files without trailing newlines must not merge -old+new into a single line."""
        old_content = "initializeCalculator();"
        new_content = "initializeCalculator();\naddKeyboardTooltip();"
        diff = EditFileTool.build_diff(old_content, new_content, "script.js")

        # Must NOT contain the concatenated string
        assert "-initializeCalculator();+initializeCalculator();" not in diff

        # Each diff entry must be on its own line
        diff_lines = diff.splitlines()
        for line in diff_lines:
            assert not (line.startswith("-") and "+" in line[1:])

        # Addition and removal counts must be truthful
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        assert added == 1
        assert removed == 0

    def test_diff_builder_with_replacements(self):
        old_content = "const x = 1;\nconst y = 2;\n"
        new_content = "const x = 10;\nconst y = 2;\n"
        diff = EditFileTool.build_diff(old_content, new_content, "index.js")
        diff_lines = diff.splitlines()
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        assert added == 1
        assert removed == 1


class TestWindowsShellNormalization:
    """Verify shell command normalization for Windows environments."""

    def test_posix_drive_conversion(self):
        cwd = r"C:\Users\kunal sanga\Desktop\New folder\app"
        cmd = r"cd /c/Users/kunal sanga/Desktop/New folder/app && python --version"
        normalized = normalize_windows_command(cmd, cwd)
        assert normalized == "python --version"

    def test_bash_escaped_spaces_unescaped(self):
        cwd = r"C:\Users\kunal sanga\Desktop\New folder\app"
        cmd = r"cd /c/Users/kunal\ sanga/Desktop/New\ folder/app && python --version"
        normalized = normalize_windows_command(cmd, cwd)
        assert normalized == "python --version"

    def test_redundant_cd_semicolon_stripped(self):
        cwd = r"C:\Users\kunal sanga\Desktop\New folder\app"
        cmd = r"cd C:\Users\kunal sanga\Desktop\New folder\app; type script.js"
        normalized = normalize_windows_command(cmd, cwd)
        assert normalized == "type script.js"

    def test_python3_rewritten_when_python3_missing(self):
        cwd = r"C:\Users\kunal sanga\Desktop\New folder\app"
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda exe: None if "python3" in exe else r"C:\Python\python.exe"
            normalized = normalize_windows_command("python3 --version", cwd)
            assert normalized == "python --version"

    def test_normalize_shell_command_in_diagnosis(self):
        cmd = r"cd /c/Users/kunal\ sanga/Desktop/New\ folder/app && echo test"
        normalized = normalize_shell_command(cmd)
        assert normalized == "echo test"


class TestCommandFailureClassification:
    """Verify classification and guidance for command failures."""

    def test_file_inspection_via_shell_detected(self):
        diagnosis = classify_command_failure(
            "cat script.js | tail -40",
            exit_code=1,
            stderr="'cat' is not recognized as an internal or external command",
        )
        assert diagnosis.category == "file_inspection_via_shell"
        assert "read_file" in diagnosis.reason

    def test_ls_file_inspection_detected(self):
        diagnosis = classify_command_failure(
            "ls -la",
            exit_code=1,
            stderr="'ls' is not recognized as an internal or external command",
        )
        assert diagnosis.category == "file_inspection_via_shell"
        assert "list_files" in diagnosis.reason

    def test_python3_missing_guidance_on_windows(self):
        diagnosis = classify_command_failure(
            "python3 --version",
            exit_code=1,
            stderr="'python3' is not recognized as an internal or external command",
        )
        assert diagnosis.category == "missing_executable"
        assert "use 'python'" in diagnosis.reason.lower()


class TestRunCommandToolExecution:
    """Verify RunCommandTool execution, exit codes, and inspection tips."""

    @pytest.mark.asyncio
    async def test_run_command_file_inspection_tip(self, tmp_path: Path):
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": "cat nonexistent_file.txt"})
        assert result.execution_failed
        assert "read_file" in (result.error or "")

    @pytest.mark.asyncio
    async def test_run_command_powershell_execution(self, tmp_path: Path):
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": "echo test_harness_output"})
        assert result.exit_code == 0
        assert "test_harness_output" in result.output


class TestProviderQuotaAndFallback:
    """Verify provider quota exhaustion detection and short-circuiting."""

    @pytest.mark.asyncio
    async def test_account_wide_quota_short_circuits_fallback(self):
        engine = FallbackEngine(fallback_config=FallbackConfig(max_fallback_models=3))

        provider = MagicMock(spec=ModelProvider)
        provider.name = "openrouter"
        provider.api_key = "sk-test"
        provider.generate = AsyncMock(
            side_effect=ProviderRequestError(
                category=ProviderErrorCategory.RATE_LIMIT,
                message="Free model limit reached for this account (HTTP 429)",
                status_code=429,
                provider="openrouter",
            )
        )

        chain = [
            ("model-a:free", provider),
            ("model-b:free", provider),
            ("model-c:free", provider),
        ]

        req = CompletionRequest(messages=[{"role": "user", "content": "hi"}], model="model-a:free")
        result = await engine.execute(req, chain)

        assert not result.succeeded
        assert result.error_category == ERROR_CATEGORY_RATE_LIMIT
        assert "Account-wide free tier or rate limit reached" in (result.final_error or "")
        # Must NOT have attempted model-b or model-c after account quota error
        assert len(result.attempts) == 1

    def test_friendly_model_error_quota(self):
        detail = "Account-wide free tier or rate limit reached for openrouter"
        msg = _friendly_model_error(detail)
        assert "free-tier request limit" in msg.lower() or "quota reached" in msg.lower()


class TestAgentLoopWindowsIntegration:
    """Verify system prompt and post-processing integration in AgentLoop."""

    def test_system_prompt_windows_rules(self, tmp_path: Path):
        from harness_core.agent.loop import AgentLoop

        provider = MagicMock(spec=ModelProvider)
        provider.name = "mock"
        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=tmp_path,
        )
        with patch("platform.system", return_value="Windows"):
            prompt = loop._system_prompt()
            assert "WINDOWS EXECUTION ENVIRONMENT" in prompt
            assert "PowerShell" in prompt
            assert "NEVER prepend commands with 'cd" in prompt
            assert "read_file" in prompt

    @pytest.mark.asyncio
    async def test_postprocess_result_file_inspection_hint(self, tmp_path: Path):
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import Task, ToolCall, ToolResults

        provider = MagicMock(spec=ModelProvider)
        provider.name = "mock"
        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=tmp_path,
        )
        task = Task(id="t1", goal="inspect")
        call = ToolCall(id="c1", tool_name="run_command", arguments={"command": "cat script.js"})
        result = ToolResults.error("Command failed: 'cat' is not recognized", exit_code=1)

        await loop._postprocess_result(task, call, result)
        assert "read_file" in (result.error or "")
        assert "Do NOT use shell commands to view or list files" in (result.error or "")

