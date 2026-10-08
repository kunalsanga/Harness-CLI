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
        from rich.console import Console
        shell = InteractiveShell.__new__(InteractiveShell)
        shell.model = None
        shell.mode = "auto"
        shell.free = False
        shell.local = False
        shell.plain = False
        shell.max_iterations = 30
        shell.max_cost = None
        shell.workspace = "/tmp/test"
        shell.out_io = StringIO()
        shell.console = Console(file=shell.out_io, force_terminal=True, width=150)
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
        
        output = shell.out_io.getvalue().upper()
        assert "HARNESS" in output, "Should show 'Harness'"
        assert "TEST" in output, "Should show project name"

    def test_welcome_no_verbose_chrome(self):
        """Welcome screen should NOT show verbose chrome."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()
        
        output = shell.out_io.getvalue().lower()

        # Should NOT show verbose info (but we DO use "─" for the logo so don't assert against it)
        for k in ("provider:", "model:", "routing:", "ready.", "connected"):
            assert k not in output, f"Should NOT show verbose chrome, got: {k}"

    def test_welcome_never_shows_stale_provider_state(self):
        """Welcome screen must not show stale state like 'No provider' when provider is connected."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        output = shell.out_io.getvalue().lower()
        # Should NOT show "No provider connected" when we have a provider
        assert "no provider connected" not in output, "Should NOT show 'No provider connected' when provider is available"
