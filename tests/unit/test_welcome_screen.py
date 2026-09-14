"""Tests for welcome screen compact header."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock
from io import StringIO

from harness_core.cli.interactive import InteractiveShell


class TestWelcomeScreenProviderState:
    """Test that welcome screen shows compact header."""

    def _make_shell(self, provider=None, router=None, model="", provider_name="") -> InteractiveShell:
        """Create a shell with mocked state."""
        shell = InteractiveShell.__new__(InteractiveShell)
        shell.model = None
        shell.mode = "auto"
        shell.free = False
        shell.local = False
        shell.plain = False
        shell.max_iterations = 30
        shell.max_cost = None
        shell.workspace = "/tmp/test"
        shell.console = MagicMock()
        shell.session_id = None
        shell.session_manager = None
        shell.current_model = model
        shell.current_provider = provider_name
        shell.total_tool_calls = 0
        shell.total_iterations = 0
        shell.session_start = 0.0
        shell.task_start = 0.0
        shell.running = False
        shell.verbose = False
        shell._event_bus = None
        shell._agent_loop = None
        shell._provider = provider
        shell._router = router
        shell._task_aware = None
        return shell

    def test_welcome_shows_project_name(self):
        """Welcome screen should show the project name derived from workspace."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        # Should show 'Harness' and project name
        harness_calls = [c for c in calls if "Harness" in str(c)]
        project_calls = [c for c in calls if "test" in str(c)]
        assert len(harness_calls) > 0, "Should show 'Harness'"
        assert len(project_calls) > 0, "Should show project name"

    def test_welcome_no_verbose_chrome(self):
        """Welcome screen should NOT show verbose chrome."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        # Should NOT show verbose info
        verbose_calls = [c for c in calls if any(k in str(c) for k in ("Provider:", "Model:", "Routing:", "Ready.", "connected", "─"))]
        assert len(verbose_calls) == 0, f"Should NOT show verbose chrome, got: {verbose_calls}"

    def test_welcome_never_shows_stale_provider_state(self):
        """Welcome screen must not show stale state like 'No provider' when provider is connected."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        # Should NOT show "No provider connected" when we have a provider
        stale_calls = [c for c in calls if "No provider connected" in str(c)]
        assert len(stale_calls) == 0, "Should NOT show 'No provider connected' when provider is available"
